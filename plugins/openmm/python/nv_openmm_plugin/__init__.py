# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader helpers for the OpenMM Nvidia/Cupo plugin."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from . import _deterministic_forces
from . import _rng_seed
from ._cufft_jit import preload_cufft_jit_toolchain


_NV = None  # the SWIG module, once loaded

# The plugin libraries that ship WITH this package. Two layouts are supported:
#   Wheel: lib/ subdirectory of the installed package (pip install)
#   Source/tarball: <root>/build/cpp relative to <root>/python/nv_openmm_plugin
# There is deliberately no way to point the package at some OTHER build: the Python
# bindings and the plugin binaries are one artifact, and letting them be chosen
# independently is what allowed two copies into a process at once.
_PKG_DIR = Path(__file__).resolve().parent
_WHEEL_LIB = _PKG_DIR / "lib"
_SOURCE_LIB = _PKG_DIR.parent.parent / "build" / "cpp"
_PLUGIN_DIR = _WHEEL_LIB if (_WHEEL_LIB / "libOpenMMNvidia.so").exists() else _SOURCE_LIB


def _cuda_platform_dir(plugin_dir: Path) -> Path:
    """Directory holding libOpenMMNvidiaCUDA.so (the CUDA platform impl): the
    `platforms/cuda` subdir of a source build, or plugin_dir itself for a
    co-located/wheel layout. Both are explicit spots -- no directory search."""
    for d in (plugin_dir / "platforms" / "cuda", plugin_dir):
        if (d / "libOpenMMNvidiaCUDA.so").exists():
            return d
    raise FileNotFoundError(
        f"libOpenMMNvidiaCUDA.so not found under '{plugin_dir}' "
        "(looked in its platforms/cuda subdir and the dir itself).")


def _preload_cufft_jit_toolchain():
    """Roots for the installed layout, in priority order: the env's own lib/ first,
    because something the environment installed deliberately beats a transitive pip
    dependency, then the copies pip drops under site-packages/nvidia. See
    _cufft_jit for why this is needed at all."""
    site_packages = _PKG_DIR.parent          # <site-packages>/nv_openmm_plugin -> ..
    env_lib = _PKG_DIR.parents[2]            # <env>/lib/pythonX.Y/site-packages -> ../../..
    preload_cufft_jit_toolchain([(env_lib, "{stem}.so.*"),
                                 (site_packages, "nvidia/*/lib/{stem}.so.*")])


def _assert_openmm_series(mm, metadata_path=None):
    """Refuse to load when the running OpenMM is outside the series the plugin
    binaries were built against.

    runtime_build.json carries openmm_min/openmm_max (the pair the build's
    CMake probe wrote, e.g. "8.5" and "8.6"), and both package managers pin
    the same bounds -- but libOpenMM.so itself has no versioned soname, so
    nothing stops the loader from binding our binaries to any OpenMM in the
    process. recipe/meta.yaml documents the failure mode: a >=8.4 pin resolved
    8.6.0, loaded without a murmur, and the divergence only showed up later.
    This is the load-time gate against it. A dev build such as
    8.5.2.dev-36a30cb parses as 8.5.2. No metadata file, nothing to check: a
    tree without the built payload cannot load the plugin anyway, and the
    soft-import contract stays intact.
    """
    import json
    import re
    path = metadata_path if metadata_path is not None else _PKG_DIR / "runtime_build.json"
    try:
        metadata = json.loads(path.read_text())
        low = tuple(int(part) for part in str(metadata['openmm_min']).split('.'))
        high = tuple(int(part) for part in str(metadata['openmm_max']).split('.'))
    except (OSError, ValueError, KeyError, TypeError):
        return
    match = re.match(r'(\d+)\.(\d+)(?:\.(\d+))?', mm.version.full_version)
    if not match:
        return
    running = (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))
    # Pad the bounds to the running version's shape: "8.5" is (8, 5, 0).
    low += (0,) * (3 - len(low))
    high += (0,) * (3 - len(high))
    if not (low <= running < high):
        raise RuntimeError(
            "nv_openmm_plugin was built against OpenMM %s..<%s, but this "
            "process is running OpenMM %s. The plugin's binaries bind to "
            "whatever libOpenMM.so is present (it carries no versioned "
            "soname), so this pairing is refused here rather than allowed to "
            "diverge silently." % (metadata['openmm_min'], metadata['openmm_max'],
                                   mm.version.full_version))


def _load():
    """Load this package's own plugin libraries into stock OpenMM, once.

    Private and argument-less by design. The Python bindings and the plugin
    binaries are ONE artifact shipped together, so there is nothing to choose:
    ``import nv_openmm_plugin`` is the single source of truth for which libraries
    run. That also makes the double-load failure structurally impossible rather
    than merely forbidden -- a second call is a no-op instead of a second image.
    (Loading twice used to leave both resident, because OpenMM keeps the kernel
    factory the first loadPluginsFromDirectory registered.)

    Called at import (see the bottom of this module) and again, defensively, from
    each public entry point -- so a load that could not happen at import raises here,
    at the point of use, rather than surfacing as a missing kernel much later.

    Returns the SWIG module holding NonbondedForce, MonteCarloBarostat and
    LangevinMiddleIntegrator (NvOpenMM::, flattened into the module by SWIG).
    """

    global _NV

    if _NV is not None:
        return _NV

    import openmm as mm

    # Gate the OpenMM pairing before anything of ours is resident: our binaries
    # bind to whatever libOpenMM.so is in the process, so the series is checked
    # against the one the build recorded (see _assert_openmm_series).
    _assert_openmm_series(mm)

    # First, so a bad value fails every load rather than leaving a run unseeded.
    seed = _rng_seed.from_env()
    deterministic_forces = _deterministic_forces.from_env()

    # Before anything of ours is resident, so the first cuFFT plan cupo builds finds
    # the toolchain already loaded rather than dlopening for it. See above.
    _preload_cufft_jit_toolchain()

    api_lib = _PLUGIN_DIR / "libOpenMMNvidia.so"
    if not api_lib.exists():
        raise RuntimeError(
            f"libOpenMMNvidia.so not found at '{_PLUGIN_DIR}'. The plugin ships "
            "with this package; build it (cmake --build <plugins/openmm/build>) "
            "so the libraries sit alongside the Python package.")
    # Not for symbols any more -- the SWIG module below links them. This is a load
    # ORDER guarantee: libOpenMMNvidiaCUDA.so needs libOpenMMNvidia.so resident by
    # the time loadPluginsFromDirectory dlopens it, and its RUNPATH does not cover
    # the directory libOpenMMNvidia.so actually sits in. Drop this and the plugin
    # fails to load *silently* -- loadPluginsFromDirectory swallows the error into
    # getPluginLoadFailures(), and the only symptom is a later, misleading
    # "Platform ... does not support all required kernels".
    ctypes.CDLL(str(api_lib), mode=ctypes.RTLD_GLOBAL)

    # In the wheel layout the SWIG module sits in the package directory itself; in
    # the dev/source layout it lives in the plugin's cpp/swig.
    swig_dir = _PKG_DIR if (_PKG_DIR / "_nvplugin.so").exists() \
        else _PKG_DIR.parent.parent / "cpp" / "swig"
    if swig_dir.is_dir() and str(swig_dir) not in sys.path:
        sys.path.insert(0, str(swig_dir))
    import nvplugin
    from . import _force_cast, _units
    _force_cast.install(mm, nvplugin)
    _units.install(nvplugin)

    # loadPluginsFromDirectory does not raise; a plugin it could not dlopen goes into
    # getPluginLoadFailures() with the loader's reason. Selected by path rather than
    # by what this call added: the list is global and keeps OpenMM's own failures,
    # and a retry would see the previous attempt's entries as pre-existing.
    plugin_dir = _cuda_platform_dir(_PLUGIN_DIR)
    mm.Platform.loadPluginsFromDirectory(str(plugin_dir))
    failed = [f for f in mm.Platform.getPluginLoadFailures() if str(_PKG_DIR) in f]
    if failed:
        raise RuntimeError(
            "the CUDA platform plugin in '%s' could not be loaded:\n  %s"
            % (plugin_dir, "\n  ".join(failed)))

    # Our integrator and barostat are SWIG classes in nvplugin, not in openmm.
    if seed is not None:
        _rng_seed.install(seed, (mm, nvplugin))
    _deterministic_forces.install(deterministic_forces, seed, _rng_seed.ENV)
    _NV = nvplugin
    return _NV


def _in(value, target):
    """A Quantity converted to `target`; anything else handed on untouched.

    This is stock's rule, spelled once instead of per method:

        if unit.is_quantity(args[0]):
            args[0] = args[0].value_in_unit(unit.bar)

    `target` is named by the caller rather than inferred. Inferring it was the bug:
    the previous helper used value_in_unit_system(md_unit_system), which is right
    for every quantity whose OpenMM C++ API happens to take MD units -- nm, kJ/mol,
    Kelvin, 1/ps -- and wrong for the one that does not. Pressure crosses in BAR,
    while the MD system renders 1 bar as 1e-25, so a barostat built with
    `1.0*unit.bar` ran at 1e-25 bar: no error, no NaN, and an acceptance term small
    enough that the trajectory still looked reasonable.

    Non-Quantities are passed on rather than float()-ed, which is also stock's rule.
    SWIG's double typemap is then the only gate, so what this plugin accepts and
    what OpenMM accepts are decided by the same code -- NvPlugin.i includes OpenMM's
    own typemaps.i, which strips units in C and is why the SWIG classes take a
    Quantity directly, setters included. A float() here would have been a second,
    looser rule: float("300") is 300.0, and a Quantity from some OTHER unit library
    answers is_quantity() with False and then unwraps to whatever magnitude it holds.

    A helper where stock repeats itself, only because stock's wrappers are
    generated; these two are written by hand and share it.
    """
    from openmm import unit as _u
    return value.value_in_unit(target) if _u.is_quantity(value) else value


def LangevinMiddleIntegrator(temperature, friction, stepSize, comFrequency=100, seed=0):
    """Create an Nvidia-accelerated LangevinMiddle (BAOAB) integrator.

    A drop-in for ``openmm.LangevinMiddleIntegrator`` -- pass it to
    ``openmm.Context``. Runs the same LangevinMiddle step as stock, in single or
    mixed precision, with its own constraint code and random numbers. With
    friction 0 its trajectories differ from stock's by float rounding, not bit
    for bit: by less than two stock runs whose starts differ by one float32
    rounding. With friction they sample the same ensemble
    (tests/test_integrator_vs_stock.py). It removes center-of-mass motion itself
    every ``comFrequency`` steps (0 disables) instead of relying on an every-step
    ``CMMotionRemover``. Drop any stock remover from the System first, or both will
    run -- it is an ordinary System edit, so do it wherever you are already editing
    the System::

        for i, f in enumerate(system.getForces()):
            if isinstance(f, mm.CMMotionRemover):
                system.removeForce(i)
                break

    Accepts plain numbers or unit.Quantity (Kelvin, 1/ps, ps).

    Whole-step graph capture defaults on for supported single-GPU systems.
    Set ``CUPO_PLUGIN_WHOLE_STEP_GRAPH=0`` before creating the Context to disable it.
    ``getWholeStepGraphCaptureCount()`` and ``getWholeStepGraphReplayCount()``
    report actual use. See the plugin README for the supported force set.

    ``seed`` is the integrator's random number seed. A nonzero value is used as
    given. 0, the default, leaves the choice to the environment variable
    CUPO_OPENMM_PLUGIN_RNG_SEED:

    - unset: OpenMM picks a new random seed each time a Context is created with
      this integrator, and does not report it. This is stock OpenMM's behaviour.
    - set to n: the integrator uses n.
    """
    from openmm import unit as _u
    nvi = _load()
    integ = nvi.LangevinMiddleIntegrator(
        _in(temperature, _u.kelvin),
        _in(friction, _u.picosecond**-1),
        _in(stepSize, _u.picosecond),
        int(comFrequency))
    if seed:
        integ.setRandomNumberSeed(int(seed))
    return integ


def MonteCarloBarostat(pressure, temperature, frequency: int = 25, seed: int = 0):
    """Create the plugin's isotropic Monte Carlo barostat; add it yourself.

        system.addForce(nv.MonteCarloBarostat(1.0*unit.bar, 300*unit.kelvin))

    A drop-in for ``openmm.MonteCarloBarostat``: same move, same acceptance
    criterion, same adaptive volume-scale tuning, and the same two Context
    parameter names, so ``context.setParameter(MonteCarloBarostat.Pressure(), p)``
    keeps working. It is independent of the integrator -- stock LangevinMiddle or
    this package's works equally.

    What it avoids is the two places the stock path spends GPU time on nothing:
    the kinetic energy that ``Context.getState(State::Energy)`` computes on every
    call and a barostat never reads, and a coordinate-scaling kernel that runs one
    thread per molecule (so a couple of protein chains among tens of thousands of
    waters serialize it).

    A function rather than the bare class, for the `seed` argument and to name the
    units the C++ side reads. The SWIG class does take a unit.Quantity on its own --
    NvPlugin.i includes OpenMM's typemaps.i, which strips units in C -- so this
    accepts plain numbers (bar, Kelvin) or OpenMM quantities, exactly as the stock
    barostat does.

    ``seed`` is the barostat's random number seed. A nonzero value is used as
    given. 0, the default, leaves the choice to the environment variable
    CUPO_OPENMM_PLUGIN_RNG_SEED:

    - unset: OpenMM picks a new random seed each time a Context is created with
      this barostat, and does not report it. This is stock OpenMM's behaviour.
    - set to n: the barostat uses n.

    A nonzero seed, passed here or taken from CUPO_OPENMM_PLUGIN_RNG_SEED, gives
    the same sequence of volume moves on every run.

    The nv.NonbondedForce picks its own cell-grid headroom by default: it sees this
    barostat at Context creation and sizes for NonbondedForce.DefaultNPTBoxSafety
    instead of the NVT 1.0. Pass an explicit factor only if the run's PEAK volume
    needs more -- the envelope is allocated once and never resized.

    Adding this alongside a stock MonteCarloBarostat is rejected at Context creation
    -- they would share the two Context parameters and each propose their own moves.
    """
    nv = _load()
    from openmm import unit as _u
    barostat = nv.MonteCarloBarostat(
        _in(pressure, _u.bar), _in(temperature, _u.kelvin), int(frequency))
    if seed:
        barostat.setRandomNumberSeed(int(seed))
    return barostat


# Public names map to SWIG classes and factories in the NvOpenMM namespace.
# Compatibility spelling: both names construct the same class and runtime type.
_CLASSES = {"NonbondedForce": "NonbondedForce", "CustomNonbondedForce": "NonbondedForce",
            "CutoffInteraction": "CutoffInteraction", "LatticeSumInteraction": "LatticeSumInteraction",
            "RadialKernel": "RadialKernel", "CoulombRadialKernel": "CoulombRadialKernel",
            "DispersionRadialKernel": "DispersionRadialKernel"}


def __getattr__(name):
    """Expose the plugin's OpenMM classes and factories as package attributes.

    They live in the SWIG module, which is loaded at import -- but softly, so it may
    not be there. Resolving them here rather than binding them at import means a load
    that failed gets retried, and reports its real error, the first time someone
    reaches for a class.

    There is deliberately no replace_nonbonded(system) helper. Converting is just
    constructing a force:

        for i, f in enumerate(system.getForces()):
            if isinstance(f, mm.NonbondedForce):
                system.addForce(nv.NonbondedForce(f, 1.15))
                system.removeForce(i)
                break

    Conversion preserves the source exception-PBC flag, whose default is False;
    no explicit setExceptionsUsePeriodicBoundaryConditions(True) call is needed.
    The flag controls only physical Coulomb/LJ exception replacements. Reciprocal
    exclusion corrections always use periodic minimum-image distances, unlike
    stock OpenMM, where the flag also controls those corrections. Set True for
    physical pair overrides that should follow the nearest periodic image.

    which is how any OpenMM force is used, and leaves the caller holding the three
    decisions a System-level helper had to guess: WHICH force (a System may hold more
    than one), whether to remove the original, and where the replacement goes.
    """
    if name in _CLASSES:
        return getattr(_load(), _CLASSES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_CLASSES))


__all__ = list(_CLASSES) + ["MonteCarloBarostat", "LangevinMiddleIntegrator"]


# Load at import, as openmm/__init__.py does for OpenMM's own plugins -- it calls
# Platform.loadPluginsFromDirectory unconditionally, so `import openmm` is what
# registers Drude's and Amoeba's kernels. Our binaries live inside this package
# rather than OpenMM's plugin directory, so `import nv_openmm_plugin` is the only
# hook that can do the same job.
#
# Soft, like OpenMM's: a missing or unbuilt library must not make the import itself
# fail, or a machine without the binaries staged cannot even inspect the package.
# _load() runs again from every entry point, so the error still arrives -- with its
# full message -- the moment someone actually tries to use the plugin.
try:
    _load()
except Exception:
    pass
