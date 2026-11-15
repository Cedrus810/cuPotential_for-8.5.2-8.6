# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for the OpenMM Nvidia/Cupo plugin.

Measures step and force+energy latencies on multiple systems via the
OpenMM Context API, comparing plugin vs stock OpenMM. This module is both the
pytest benchmark and the CLI system bench (run directly; see the CLI section):

  - LangevinMiddleIntegrator (300K, 1/ps friction, 4fs timestep)
  - Fixed WARMUP_STEPS warmup + PRODUCTION_STEPS timed bulk run (no calibration)
  - Stabilization loop ensures GPU clocks have ramped
  - Mixed precision (production standard; HMR + constraints for the 4fs step)
  - Stock runs with DeterministicForces=true (STOCK_DETERMINISTIC_FORCES) so its
    nonbonded is deterministic like the plugin's -- a fair peer, not a stock that
    buys speed by forgoing reproducibility

Systems:
  - water_100k (~100k atoms / 33k waters)
  - apoa1 (~92k atoms, lipoprotein + POPC lipid)
  - apoa1_ljpme (same ApoA1 fixture with reciprocal dispersion enabled)
  - bace (~70k atoms, protein; long-range CYS160-CYS364 disulfide exclusions)
  - tyk2 (~49k atoms, protein)
  - dhfr (~24k atoms, protein)
  - dhfr_tip4p (~29k particles, same protein in TIP4P-Ew; water virtual sites)
  - tyk2_rbfe_dodec / tyk2_abfe_dodec: vendored ligand systems in dodecahedral
    boxes at their documented staged midpoint windows (see
    data/tyk2_alchemical/manifest.json)

Each runs when its data is in the tree; see _SYSTEMS.

The systems are split into benchmark groups (golden_runs.GROUPS), each run by a CI
job of its own and selected here by CUPO_BENCH_GROUP; unset, a run covers every
group. Each group's goldens are a tree of runs, plugins/openmm/tests/<tree>/<GPU>/
<run>/{nvt,npt}.json: golden/ for main, golden-triclinic/ for the cell-shape set.

Run:
    ./run_test -m benchmark -v
Skip:
    ./run_test -m "not benchmark"
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_runs  # noqa: E402


def _load_gpu_clocks():
    """tests/gpu_clocks.py: locks the clocks for the benchmarks and records what the
    GPU did. Shared with the core benchmarks, so loaded from there by path;
    the conftest may have loaded it already."""
    if "gpu_clocks" not in sys.modules:
        import importlib.util
        path = Path(__file__).resolve().parents[3] / "tests" / "gpu_clocks.py"
        spec = importlib.util.spec_from_file_location("gpu_clocks", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["gpu_clocks"] = module
        spec.loader.exec_module(module)
    return sys.modules["gpu_clocks"]


gpu_clocks = _load_gpu_clocks()

import openmm as mm
import openmm.unit as unit
from openmm import app

# ── Shared bench defaults ─────────────────────────────────────────────────────
# Production-standard config: 4 fs + HMR + skin=0 + mixed precision. This single
# module is both the pytest benchmark (run under pytest) and the CLI system bench
# (run directly: `python test_bench_plugin.py --pdb ...`),
# so the two can't drift apart.
TEMPERATURE_K = 300.0        # Langevin thermostat temperature
FRICTION_PS = 1.0            # Langevin friction (1/ps)
TIMESTEP_FS = 4.0            # integrator timestep (needs HMR + constraints)
CUTOFF_NM = 0.9              # nonbonded cutoff
APPLY_HMR = True             # apply hydrogen mass repartitioning (for the 4 fs step)
HYDROGEN_MASS_AMU = 1.5      # HMR target hydrogen mass when APPLY_HMR is set
PRECISION = "mixed"          # OpenMM production standard; accuracy peer for cupo
BAROSTAT = False             # NVT by default; True adds a MonteCarloBarostat (NPT)
NPT_BOX_SAFETY = 1.15        # NPT: plugin pre-allocates the cell grid for a box up
                             # to (initial * this), so a barostat expansion never
                             # reallocs (passed to replace_nonbonded(npt_box_safety_factor=)).
                             # 1.15 = 15% linear headroom.
WARMUP_STEPS = 500           # untimed warmup steps (GPU clock ramp) before timing
PRODUCTION_STEPS = 3000      # fixed timed step count -- no wall-clock calibration,
                             # so the golden ties to a known, reproducible step
                             # budget rather than a count that drifts with GPU speed
STOCK_DETERMINISTIC_FORCES = True  # run stock OpenMM with DeterministicForces=true
                             # (fixed-point PME charge spreading, not float atomicAdd)
                             # so its nonbonded is deterministic like the plugin's --
                             # a fair peer, since the plugin forgoes the faster
                             # non-deterministic path by design

_HYDROGEN_MASS = HYDROGEN_MASS_AMU * unit.amu


# Default forcefield used to assign forces to a system's PDB topology.
# amber14-all bundles the ff14SB protein FF, lipid17 (covers apoA1's POPC),
# nucleic, and ions; tip3p is the water model.
FORCEFIELD = ("amber14-all.xml", "amber14/tip3p.xml")

# Per-system force field overrides — matches canonical OpenMM benchmark exactly.
# Systems not listed here use the default FORCEFIELD above.
_SYSTEM_FORCEFIELD = {
    "apoa1": ("amber14-all.xml", "amber14/tip3p.xml"),
    "dhfr": ("amber99sb.xml", "tip3p.xml"),
    # The same protein in 4-site TIP4P-Ew, equilibrated with this force field:
    # benches the virtual-site path end to end.
    "dhfr_tip4p": ("amber14-all.xml", "amber14/tip4pew.xml"),
    # Cellulose is a carbohydrate: its glucose residues need GLYCAM, which
    # amber14-all does not bundle. GLYCAM's templates borrow ff14SB atom types
    # (glycam-Cg sits next to protein-H1), so it layers ON TOP of amber14-all
    # rather than replacing it -- loading it alone raises KeyError: 'protein-H1'.
    "cellulose": ("amber14-all.xml", "amber14/GLYCAM_06j-1.xml", "amber14/tip3p.xml"),
}


def _build_system(topology, cutoff_nm=CUTOFF_NM, forcefield=FORCEFIELD,
                  apply_hmr=APPLY_HMR):
    """Assign forces to a topology via an OpenMM ForceField.

    PME nonbonded, HBond constraints + rigid water, and HMR (via hydrogenMass)
    for the 4 fs step -- one HMR mechanism for every system. Without HMR the fast
    H-heavy angle librations go unstable at 4 fs and the towerplate cell matrix
    overflows.

    template_pins covers the residues OpenMM's matcher cannot resolve from
    connectivity alone (cellulose's GLYCAM sugars); it is empty for every other
    system, so it is passed unconditionally.
    """
    from data.datasets import template_pins

    ff = app.ForceField(*forcefield)
    return ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=cutoff_nm * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=(_HYDROGEN_MASS if apply_hmr else None),
        residueTemplates=template_pins(topology),
    )


def _read_pdb(path):
    """Read a PDB, transparently handling gzip (.pdb.gz, the dataset format)."""
    path = str(path)
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt") as fh:
            return app.PDBFile(fh)
    return app.PDBFile(path)

# ── Paths ─────────────────────────────────────────────────────────────────────

_TESTS_DIR = Path(__file__).resolve().parent
_PLUGIN_DIR = _TESTS_DIR.parent
_REPO = _PLUGIN_DIR.parent.parent
# The committed goldens are the reference, and the only thing a regression can be a
# regression from. GOLDEN_OUTPUT_DIR moves where a run WRITES its numbers -- CI points
# it at a scratch directory that it collects as an artifact -- and must not move where
# a run READS them. When it moved both, the scratch file was empty at the start of
# every CI run, so every metric was recorded and then compared against itself: 0%
# change, on every system, on every GPU, and no benchmark could ever fail.
#
# One committed tree per benchmark group (golden_runs.GROUPS): golden/ for main,
# golden-triclinic/ for the cell-shape set. A system's goldens live in its group's
# tree, so a run touches only the trees of the groups it measured.
def _golden_ref_dir(group: str) -> Path:
    return _TESTS_DIR / golden_runs.group_tree(group)


# Where a run writes what it measured. CI points GOLDEN_OUTPUT_DIR at a scratch
# directory it collects as an artifact; each group writes its own subdirectory of
# it (golden_runs.group_artifact_dir), so the jobs' artifacts never collide.
#
# Off to one side by default, rather than into the committed golden. A golden is a
# directory of runs now, so writing into it would not overwrite anything -- it would
# ADD this run to the golden, and the next local run would be measured against a
# reference that includes the last one. The committed golden stays what CI put
# there, and a developer's own numbers accumulate somewhere they can be read and
# deleted. Both are the same shape, so pointing the two at each other is one
# environment variable when that is genuinely wanted.
#
# A tree with no committed goldens at all -- the plugin's published source tree
# carries none -- writes into the plugin's build/ instead, so that a benchmark run
# there leaves no golden files in the tree.
_HAS_GOLDENS = any((_TESTS_DIR / tree).is_dir() for _, tree, _ in golden_runs.GROUPS)


def _golden_out_dir(group: str) -> Path:
    if "GOLDEN_OUTPUT_DIR" in os.environ:
        return Path(os.environ["GOLDEN_OUTPUT_DIR"]) / golden_runs.group_artifact_dir(group)
    if not _HAS_GOLDENS:
        return _PLUGIN_DIR / "build" / "bench-results" / golden_runs.group_artifact_dir(group)
    return _golden_ref_dir(group)

def _run_id() -> str:
    """What this run calls itself among the runs of its golden.

    The CI job id, so a number in a committed golden leads back to the log that
    produced it. Outside CI there is no job id, so the machine names the directory
    instead: a developer's runs land beside the committed ones, under their own
    hostname, where `git status` shows them and they can be looked at, kept or
    thrown away. Re-running on the same machine replaces that machine's directory
    rather than piling up a new one each time.

    A function rather than a module constant because the machine name is resolved
    by _node_name(), which is defined further down.
    """
    return os.environ.get("CI_JOB_ID") or _node_name()


def _sanitize_gpu_dir(name: str) -> str:
    """GPU name -> filesystem-safe directory name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


_GOLDEN_FILENAMES = {"nvt": "nvt.json", "npt": "npt.json"}


def _ensemble() -> str:
    return "npt" if BAROSTAT else "nvt"


def _golden_gpu_dir(group: str = golden_runs.MAIN_GROUP) -> Path:
    """A group's committed golden for this GPU: a directory of runs, not a file."""
    return _golden_ref_dir(group) / _sanitize_gpu_dir(_gpu_name())


def _golden_out_path(group: str = golden_runs.MAIN_GROUP) -> Path:
    """Where this run writes what it measured of one group.

    Its own directory, named for the run, so that five jobs of one pipeline do not
    write the same path and leave whichever unpacked last. The committed tree has
    the same shape, so collecting a measurement is a copy.
    """
    return (_golden_out_dir(group) / _sanitize_gpu_dir(_gpu_name())
            / _sanitize_gpu_dir(_run_id()) / _GOLDEN_FILENAMES[_ensemble()])

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── Golden file logic (same structure as core bench_helpers) ──────────────────

# A metric that moved more than this is worth saying something about in the log. It
# governs the WARNING and nothing else -- what a run measured is always what a run
# records, whatever this is set to. Left at 0 it would warn about every metric of
# every system, which says nothing.
_WARN_THRESHOLD = 0.05
_FAIL_THRESHOLD = 0.10


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5, text=True,
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "unknown GPU"


def _load_golden(path: Path = None) -> dict:
    """One measurement file. The committed reference is read by _reference_golden,
    which folds a directory of these."""
    path = path or _golden_out_path()
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_golden(data: dict, path: Path = None) -> None:
    path = path or _golden_out_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


_reference: dict = {}
# The runs each fold came from, so a caller that wants the spread rather than the
# centre does not read the directory a second time.
_reference_runs: dict = {}


def _reference_golden(group: str = golden_runs.MAIN_GROUP) -> dict:
    """A group's committed golden for this GPU and ensemble, read once and frozen for
    the session.

    Frozen because a run rewrites its own golden as it goes, and six tests share one
    measurement -- each of them asks for the entry. Re-reading the file would hand
    the second test whatever the first test had just written, so five of the six
    metrics would compare a number against itself and only the first could fail.

    Keyed by GPU directory and ensemble rather than by one cached dict: BAROSTAT is
    a module global that the ensemble fixture flips, so one session folds the NVT
    runs and then the NPT ones.
    """
    key = (_golden_gpu_dir(group), _ensemble())
    if key not in _reference:
        runs = golden_runs.load_runs(key[0], _GOLDEN_FILENAMES[key[1]])
        _reference[key] = golden_runs.fold(runs)
        _reference_runs[key] = runs
    return _reference[key]


def _record_measurement(baseline: dict, system_name: str, precision: str) -> None:
    """Write what this run measured to the output golden. Always.

    The output is this run's own directory: GOLDEN_OUTPUT_DIR in CI, which is what
    CI collects as an artifact, what the golden plots are drawn from, and what CI
    proposes as the next golden. Outside CI it is a directory
    named after the machine, beside the committed runs.

    It does not disturb the golden this run is measured against, because the
    reference is folded and frozen before anything is written (see below). A second
    session on the same machine does fold the first one's numbers in -- they are
    runs of that golden now, which is the point of putting them there.

    Unconditional, and that is the point. It used to write only when some metric
    had moved more than an update threshold, which meant a run's own numbers were
    sometimes in the file and sometimes not, with nothing in the output saying
    which -- so a golden could hold this run's measurement for one system and a
    months-old one for the next, and reading the file told you nothing about where
    its contents came from. What a run measured is now what a run records; the
    comparison is unaffected either way, because it is made against the reference
    frozen at the start of the run (see _reference_golden).
    """
    # Freeze the committed reference BEFORE writing anything. The write lands in
    # this run's own directory, so it cannot disturb the reference -- but the
    # ordering is kept because it is what makes that true of any output path a
    # caller points here, including one aimed at the committed golden. Reading
    # after writing would otherwise hand the caller numbers this call just wrote,
    # and every metric would be compared against itself. _reference_golden caches
    # for the session, so this call is what makes the ordering safe; the value is
    # unused here on purpose.
    group = _group_of(system_name)
    _reference_golden(group)
    _reference_golden(golden_runs.MAIN_GROUP)   # the fallback _get_golden_entry may read
    path = _golden_out_path(group)
    out = _load_golden(path)
    cell = dict(baseline)
    # The measurement's telemetry, written by the fixture before any test records.
    previous = out.get(precision, {}).get(system_name, {})
    if "telemetry" in previous:
        cell["telemetry"] = previous["telemetry"]
    out.setdefault(precision, {})[system_name] = cell
    _save_golden(out, path)


def _get_golden_entry(measured: dict, system_name: str, precision: str) -> dict:
    """Record this run's numbers and return the values to compare them against."""
    baseline = _current_baseline(measured)
    _record_measurement(baseline, system_name, precision)

    group = _group_of(system_name)
    entry = _reference_golden(group).get(precision, {}).get(system_name)
    if not entry and group != golden_runs.MAIN_GROUP:
        # A group newer than its committed golden: its systems were measured by the
        # main job before the group had a job of its own, so main's tree still holds
        # their last numbers. Compare against those until the group's own golden is
        # committed, rather than against nothing.
        entry = _reference_golden(golden_runs.MAIN_GROUP).get(precision, {}).get(system_name)
    if not entry:
        # A GPU or a system with no committed golden yet: the run records what it
        # measured and has nothing to hold it against.
        return baseline
    # A golden written before a metric existed has nothing to say about that metric.
    return {key: entry.get(key, baseline[key]) for key in baseline}


def _compared_entry(measured: dict, system_name: str, precision: str) -> dict:
    """_get_golden_entry, for a benchmark: the values to compare against, after
    recording this run's numbers. Skips the test when there is no committed golden
    for this GPU, ensemble and system, saying so: a number compared against itself
    always passes, and a pass would claim a comparison that was not made. Skips it
    too, after recording, when the clocks are not locked (CUPO_ALLOW_UNLOCKED_CLOCKS):
    such numbers are not comparable with anything."""
    entry = _get_golden_entry(measured, system_name, precision)
    group = _group_of(system_name)
    if gpu_clocks.SESSION_UNLOCKED:
        pytest.skip("measured, not compared: %s (%s=1); this run's numbers are in %s"
                    % (gpu_clocks.UNLOCKED, gpu_clocks.ALLOW_UNLOCKED_VAR,
                       _golden_out_path(group)))
    committed = any(_reference_golden(g).get(precision, {}).get(system_name)
                    for g in {group, golden_runs.MAIN_GROUP})
    if not committed:
        pytest.skip("measured, not compared: no committed golden for %s, %s, %s on %s "
                    "(this run's numbers are in %s)"
                    % (system_name, precision, _ensemble(), _gpu_name(),
                       _golden_out_path(group)))
    return entry


# ── Which metrics can fail a run ──────────────────────────────────────────────
#
# One. Every other metric is measured, recorded, printed and warned about, and
# none of them can turn a run red.
#
# plugin_ns_per_day is the plugin's own step throughput -- 1/plugin_step_ms
# carried to ns/day -- and it is the only number here that is both about our code
# and steady enough to hold a threshold. Measured over repeated benchmark runs of
# a single commit with no performance change in it, so every difference between
# two runs is noise:
#
#   gating it alone       21 of 23 attempts pass
#   gating all six        8 of 22 attempts pass
#
# and the two that still fail were hardware faults: a GPU that stopped responding
# mid-run, and a machine that was uniformly slow on nearly every number. Nothing
# in between: the worst regression among the 21 passing attempts is 5.1%, and the
# two failures are at 30% and 90%. The failure count is identical at every threshold from 7%
# to 20%, so there is nothing to tune.
#
# What the other five cost us, and why they are only reported:
#
#   *_force_and_energy_ms       land more than 10% off golden on 22-24% of
#                               measurements, against 8-9% for the step metrics.
#                               _time_force_and_energy is the thinnest estimator
#                               in the suite.
#   speedup_*                   ratios against stock OpenMM. Immune to a slow
#                               host, which is their virtue, but they inherit
#                               the noise of BOTH sides, and a stock-side fluke
#                               frozen into a golden fails forever after.
#   stock_*                     not our code at all.
#
# The trade this makes: an absolute metric cannot tell "our code got slower" from
# "this host is slow today", so a degraded runner now fails as a regression --
# classify the failure before retrying it. And it cannot see a regression that
# leaves the step path alone, or one where stock gets faster and we do not. Those
# are visible in the warnings, which is what the warnings are for.
_GATED = frozenset({"plugin_ns_per_day"})


def _check_regression(metric_name: str, measured_val: float,
                      golden_val: float, system_name: str, precision: str,
                      higher_is_worse: bool = True):
    # Compare at golden precision (4 sig figs) to avoid false warnings from
    # rounding noise when _sig(measured) differs from measured by an epsilon.
    measured_rounded = _sig(measured_val)
    if higher_is_worse:
        regression_frac = (measured_rounded - golden_val) / golden_val
    else:
        regression_frac = (golden_val - measured_rounded) / golden_val

    change_frac = abs(measured_rounded - golden_val) / golden_val

    gated = metric_name in _GATED

    if change_frac > _WARN_THRESHOLD:
        pct = change_frac * 100
        direction = "regression" if regression_frac > 0 else "improvement"
        # An ungated metric says so in its own warning, so that a large number in
        # a passing log does not read as the gate having missed it.
        note = "" if gated else " [reported only, not gated]"
        warnings.warn(
            f"\n{metric_name}: {pct:.1f}% {direction} vs golden "
            f"(measured={measured_val:.4f}, golden={golden_val:.4f}){note}.\n",
            stacklevel=3,
        )

    if gated and regression_frac > _FAIL_THRESHOLD:
        if higher_is_worse:
            pytest.fail(
                f"{metric_name}: {regression_frac*100:.1f}% regression "
                f"(measured={measured_val:.4f} ms, "
                f"golden={golden_val:.4f} ms, "
                f"fail threshold={_FAIL_THRESHOLD*100:.0f}%)")
        else:
            pytest.fail(
                f"{metric_name}: {regression_frac*100:.1f}% regression "
                f"(measured={measured_val:.1f}x, "
                f"golden={golden_val:.1f}x, "
                f"fail threshold={_FAIL_THRESHOLD*100:.0f}%)")


# ── Timing helpers ────────────────────────────────────────────────────────────


# ── GPU tuning introspection ─────────────────────────────────────────────────
# The clocks themselves are locked, and watched, by tests/gpu_clocks.py; what is
# recorded here is the rest of what a golden has to be read against.


class _BuildInfo(ctypes.Structure):
    """Mirror of cupo_dspace_build_info (include/cupo/dspace.h).
    Field order and types must match that struct exactly."""
    _fields_ = [("tuning_threads", ctypes.c_int),
                ("tuning_blocks_per_sm", ctypes.c_int),
                ("sm_count", ctypes.c_int),
                ("launch_blocks_per_grid", ctypes.c_int),
                ("shared_mem_bytes", ctypes.c_int),
                ("cuda_runtime_version", ctypes.c_int)]


def _build_info() -> "_BuildInfo | None":
    """One call into libcupo.so for every launch/build diagnostic. None when
    the runtime lib is not loadable (e.g. a --stock-only run)."""
    try:
        try:
            lib = ctypes.CDLL("libcupo.so")
        except OSError:
            # The library's SONAME is libcupo.so.<ABI>, so the loader does not
            # know the one already in the process as libcupo.so, and an installed
            # package has no link of that name: open the file that is mapped.
            lib = ctypes.CDLL(_loaded_lib_path("libcupo.so"))
        fn = lib.cupo_dspace_get_build_info
    except (OSError, AttributeError):
        return None
    fn.argtypes = [ctypes.POINTER(_BuildInfo)]
    fn.restype = ctypes.c_int
    info = _BuildInfo()
    if fn(ctypes.byref(info)) != 0:
        lib.cupo_last_error.restype = ctypes.c_char_p
        raise RuntimeError(lib.cupo_last_error().decode())
    return info


def _towerplate_launch_config() -> dict:
    """Launch config of the compiled cuda_towerplate_direct_force kernel, read from
    libcupo.so so it reflects the actual plugin build. All-None when the
    runtime lib is not loadable (e.g. a --stock-only run)."""
    info = _build_info()
    if info is None:
        return {"launch_bounds_max_threads_per_block": None, "launch_bounds_min_blocks_per_mp": None,
                "execution_grid_dim": None, "execution_block_dim": None, "shared_mem_bytes": None}
    # Actual launch: <<<execution_grid_dim, execution_block_dim>>> with the compiled
    # __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP); shared_mem_bytes = static bytes.
    return {"launch_bounds_max_threads_per_block": info.tuning_threads,
            "launch_bounds_min_blocks_per_mp": info.tuning_blocks_per_sm,
            "execution_grid_dim": info.launch_blocks_per_grid,
            "execution_block_dim": info.tuning_threads,
            "shared_mem_bytes": info.shared_mem_bytes}


def _sm_count() -> int | None:
    """SM count of the benchmark GPU. None when the runtime lib is not loadable."""
    info = _build_info()
    return None if info is None else info.sm_count


def _cuda_runtime_version() -> str:
    """CUDA runtime version the plugin kernel lib (libcupo.so) was built
    against, e.g. '13.1'. Raises: the bench requires a GPU build, so an unloadable
    lib or failed CUDA query is a hard error, not a null record."""
    info = _build_info()
    if info is None:
        raise RuntimeError("libcupo.so not loadable for CUDA version detection")
    v = info.cuda_runtime_version
    if v <= 0:
        raise RuntimeError(f"cudaRuntimeGetVersion failed (returned {v})")
    return f"{v // 1000}.{(v % 1000) // 10}"


def _driver_version() -> str:
    """NVIDIA driver version string (e.g. '590.48.01') via NVML. Raises: the bench
    requires a GPU, so an unavailable NVML/driver is a hard error, not a null record."""
    import ctypes
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError as exc:
        raise RuntimeError(f"libnvidia-ml.so.1 not loadable for driver version detection: {exc}")
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit failed for driver version detection")
    try:
        buf = ctypes.create_string_buffer(96)
        if lib.nvmlSystemGetDriverVersion(buf, 96) != 0:
            raise RuntimeError("nvmlSystemGetDriverVersion failed")
        ver = buf.value.decode()
        if not ver:
            raise RuntimeError("nvmlSystemGetDriverVersion returned an empty string")
        return ver
    finally:
        lib.nvmlShutdown()


def _parse_mhz(value: str | None) -> float | None:
    """Parse an lscpu/procfs MHz value, returning None when unavailable."""
    if not value:
        return None
    try:
        return float(value.strip().replace(",", "."))
    except ValueError:
        return None


def _proc_cpuinfo_model() -> str | None:
    """Fallback CPU model from /proc/cpuinfo."""
    model = None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key, sep, val = line.partition(":")
                if not sep:
                    continue
                key = key.strip()
                val = val.strip()
                if key == "model name" and model is None:
                    model = val
    except OSError:
        pass
    return model


def _cpu_config_block() -> dict:
    """CPU model and frequency metadata for benchmark provenance.

    The OpenMM benchmark is GPU-heavy, but context setup, launches, and host-side
    orchestration still depend on CPU hardware and clock policy. Environment values
    let wrapper scripts pin/report exactly what they observed; direct pytest/CLI runs
    fall back to lscpu and /proc/cpuinfo.
    """
    env_model = os.environ.get("CUPO_CPU_NAME") or None
    env_max = _parse_mhz(os.environ.get("CUPO_CPU_MAX_MHZ"))

    fields = {}
    try:
        out = subprocess.check_output(["lscpu"], timeout=5, text=True)
        for line in out.splitlines():
            key, sep, val = line.partition(":")
            if sep:
                fields[key.strip()] = val.strip()
    except Exception:
        pass

    proc_model = _proc_cpuinfo_model()
    return {
        "cpu_model": env_model or fields.get("Model name") or proc_model or "unknown",
        "cpu_max_mhz": env_max or _parse_mhz(fields.get("CPU max MHz")),
        "cpu_min_mhz": _parse_mhz(fields.get("CPU min MHz")),
    }


def _openmm_version() -> str:
    """OpenMM version string of the build producing the stock reference timings
    (e.g. '8.4.0'). Recorded so a golden ties to a specific OpenMM release."""
    import openmm.version
    return openmm.version.short_version


def _openmm_revision() -> str:
    """OpenMM git revision hash of the installed build (full 40-char sha, or ''
    for a release tarball without git metadata). Disambiguates dev builds that
    share a short_version but differ in commit."""
    import openmm.version
    return getattr(openmm.version, "git_revision", "") or ""


def _git_describe() -> str:
    """Short git hash of the cupo checkout (with '-dirty' if there are uncommitted
    TRACKED changes), for plot provenance. '-dirty' means the running binaries may
    not match the hash. Ignores untracked files (build artifacts, plots). Returns
    'unknown' outside a git checkout."""
    try:
        cwd = str(_TESTS_DIR)
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0
        return f"{h}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def _node_name() -> str:
    """The machine this ran on.

    A golden is many runs, which can come from different machines with the same
    GPU, so "these two low numbers are the same machine" has to be answerable from
    the golden rather than from a log that may be gone.

    CUPO_NODE_NAME can be set from outside the container, because this code cannot
    find the answer itself there: inside docker, gethostname() returns a container
    id. The remaining fallbacks -- a job scheduler's node name, then the host
    name -- are for a run outside a container.
    """
    return (os.environ.get("CUPO_NODE_NAME")
            or os.environ.get("SLURMD_NODENAME")
            or os.environ.get("SLURM_JOB_NODELIST")
            or socket.gethostname())


def _gpu_identity() -> dict:
    """Which physical card this ran on: {"gpu_uuid": ..., "gpu_pci_bus_id": ...}.

    `node` names the machine, and on a one-GPU machine that is the card. It is not
    on a machine with several, where a run may get a different GPU each time.
    Without this, two runs on one machine read as the same hardware in the golden
    when they may be different cards -- and a card that is slow or throttling is
    invisible.

    nvidia-smi inside the container sees only the allocated GPU, so the first row
    is the one that measured. Empty rather than fatal when it cannot be read: a
    provenance field missing is worth a gap in the record, not a failed benchmark.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id",
             "--format=csv,noheader"], timeout=5, text=True)
    except Exception:
        return {}
    first = out.strip().splitlines()[0] if out.strip() else ""
    parts = [f.strip() for f in first.split(",")]
    if len(parts) != 2 or not parts[0]:
        return {}
    return {"gpu_uuid": parts[0], "gpu_pci_bus_id": parts[1]}


def _gpu_config_block(locked) -> dict:
    """The per-GPU `config` conditions block recorded in the golden. Precision-independent
    (same kernel binary, GPU, locked clocks, step budget, and OpenMM build for mixed and
    single), so it lives at the GPU level rather than under a precision. Also records the
    fixed warmup/production step counts, the cupo neighbor-list skin, and the OpenMM
    version+revision that produced the stock reference timings -- the run conditions are
    identical across every entry here. `locked` is the gpu_clocks.LockedClocks the GPU
    was locked at; what the clocks actually did is recorded per system, as `telemetry` beside
    each system's metrics (see _record_telemetry)."""
    return {
        "node": _node_name(),
        **_gpu_identity(),
        "sm_count": _sm_count(),
        "cuda_version": _cuda_runtime_version(),
        "driver_version": _driver_version(),
        **_cpu_config_block(),
        "graphics_clock_mhz": locked.graphics_mhz,
        "memory_clock_mhz": locked.memory_mhz,
        # What was not as asked when the clocks were locked, such as a card that tops
        # out below the requested clock; compare_golden_plots marks these runs.
        "warnings": list(locked.warnings),
        "warmup_steps": WARMUP_STEPS,
        "production_steps": PRODUCTION_STEPS,
        "barostat": BAROSTAT,
        "npt_box_safety": NPT_BOX_SAFETY if BAROSTAT else None,
        "openmm_version": _openmm_version(),
        "openmm_revision": _openmm_revision(),
        "stock_deterministic_forces": STOCK_DETERMINISTIC_FORCES,
        "whole_step_graph_requested": os.environ.get("CUPO_PLUGIN_WHOLE_STEP_GRAPH", "1") == "1",
        "whole_step_batch_requested": int(os.environ.get("CUPO_PLUGIN_WHOLE_STEP_BATCH", "1")),
        "clear_parallel": os.environ.get("CUPO_PLUGIN_CLEAR_PARALLEL", "batched"),
        "integrator_graph": os.environ.get("CUPO_PLUGIN_INTEGRATOR_GRAPH", "0"),
        "bonded_schedule": os.environ.get("CUPO_PLUGIN_BONDED_SCHEDULE", "auto"),
        "cuda_graph": os.environ.get("CUPO_PLUGIN_CUDA_GRAPH", "1"),
        "kernel_launch_configs": {"cuda_towerplate_direct_force": _towerplate_launch_config()},
    }


def _record_config_block(block: dict, system_name: str) -> None:
    """Write the `config` conditions block (precision-independent) alongside the
    per-precision system entries this run measured, in the file of the system's
    group. The block says which group the file is, so a run read out of context
    still counts as that group's (golden_runs.run_count)."""
    group = _group_of(system_name)
    path = _golden_out_path(group)
    golden = _load_golden(path)
    golden["config"] = dict(block, bench_group=group)
    _save_golden(golden, path)


def _record_telemetry(telemetry: dict, system_name: str, precision: str) -> None:
    """Write what the GPU did while one system was measured -- min / max / median /
    mean of clock, power and temperature, from BenchClocks.get_telemetry -- beside that
    system's metrics. golden_runs.fold reads numbers only, so it passes this by."""
    path = _golden_out_path(_group_of(system_name))
    golden = _load_golden(path)
    golden.setdefault(precision, {}).setdefault(system_name, {})["telemetry"] = telemetry
    _save_golden(golden, path)


def _stabilize(measure_fn, label: str, threshold: float = 0.05,
               max_rounds: int = 5) -> float:
    """Repeat measurement until consecutive results converge.

    Keeps calling measure_fn() until relative change between consecutive
    results is within threshold.  Raises if max_rounds exceeded.
    """
    prev = measure_fn()
    print(f"    stabilize [{label}]: round 1 = {prev:.4f} ms", flush=True)
    for i in range(max_rounds):
        cur = measure_fn()
        diff = abs(cur - prev) / prev
        print(f"    stabilize [{label}]: round {i+2} = {cur:.4f} ms "
              f"(delta={diff*100:.1f}%)", flush=True)
        if diff <= threshold:
            return cur
        prev = cur
    raise RuntimeError(
        f"stabilize [{label}]: failed to converge within {max_rounds} rounds "
        f"(last delta={diff*100:.1f}%, threshold={threshold*100:.1f}%)"
    )


def _warmup(context) -> int:
    """Warm the GPU clocks with WARMUP_STEPS untimed steps, then return the fixed
    PRODUCTION_STEPS timed-run count. No wall-clock calibration: the golden ties to
    a known step budget rather than a count that drifts with GPU speed."""
    integrator = context.getIntegrator()
    integrator.step(WARMUP_STEPS)
    context.getState(getEnergy=True)
    print(f"      warmup: {WARMUP_STEPS} steps; production fixed at "
          f"{PRODUCTION_STEPS} steps", flush=True)
    return PRODUCTION_STEPS


def _time_step(context, steps: int) -> float:
    """Time a single bulk integrator.step(steps) run. Returns ms/step."""
    integrator = context.getIntegrator()
    t0 = time.perf_counter()
    integrator.step(steps)
    context.getState(getEnergy=True)
    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_step = elapsed / steps
    print(f"      _time_step: {steps} steps, {ms_per_step:.4f} ms/step",
          flush=True)
    return ms_per_step


def _time_force_and_energy(context, positions) -> float:
    """Time force+energy computation with fresh positions.

    setPositions invalidates cached state, forcing full recomputation.
    getState synchronizes GPU and returns data to host.
    Returns wall-clock ms per call (median of repeat measurements).
    """
    for _ in range(5):
        context.setPositions(positions)
        context.getState(getForces=True, getEnergy=True)

    samples = []
    for _ in range(25):
        context.setPositions(positions)
        t0 = time.perf_counter()
        context.getState(getForces=True, getEnergy=True)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def _time_energy(context, positions) -> float:
    """Time energy-only evaluation: the same shape as _time_force_and_energy,
    with getForces dropped. Returns wall-clock ms per call (median of 25).

    WHAT OF CUPO THIS ACTUALLY RUNS

    A whole nonbonded evaluation, minus the force writes. getState(getEnergy=True)
    reaches CudaCalcNonbondedForce::execute -> runStep -> dispatchStep with
    computeEnergy=true, computeForces=false, and what dispatchStep launches then
    (CudaKernels.cu) is:

        cupo_cell_list_execute    unconditional -- the cell list is rebuilt every
                                step either way, so this is the same work here as
                                on the force path. cupo has no Verlet skin, so it
                                is a real per-step cost, not setup.
        cupo_dspace_execute     the towerplate direct-space kernels, the bulk of
        cupo_dspace_finish      the evaluation, accumulating energy only
        cupo_kspace_execute     SPME: charge spreading, the FFTs, the reciprocal
                                sum and force interpolation, with the force
                                output pointer passed as nullptr
        the energy reduction    plus finishPendingEnergy() summing dspace, kspace,
                                self and dispersion terms

    So: cell list, direct space, reciprocal space, energy reduction -- the whole
    pipeline, on real positions, with the same box and the same occupancy. The
    flags do not skip stages; they switch off force accumulation inside kernels
    that run regardless. It even gets its own captured CUDA graph, since the
    graph slot is keyed on (energy, forces) -- graph_slots_[ce * 2 + cf].

    That is the closest thing in this file to a clean measurement of the code we
    write, per evaluation, and it is why the metric exists.

    WHAT IT LEAVES OUT, AND WHY THAT MATTERS

    _time_force_and_energy mostly does not measure us. Its timed call is
    getState(getForces=True, ...), and asking for forces makes OpenMM carry all
    N of them back to the host afterwards -- CommonKernels.cpp:286, which
    downloads the whole force buffer, rescales it out of fixed point, and
    scatters it into a Vec3 array in permuted atom order (forces[order[i]] =
    ...), after which SWIG builds N Python objects out of the result. None of
    that is cupo, all of it is per-atom work on the CPU, and measured against
    the committed goldens it is 62-77% of the reported number -- about 9 ns per
    atom, flat from dhfr's 23k atoms to cellulose's 409k.

    That dominant term is what makes the metric useless as a gate and hard to
    read as a result:

      - It swamps the difference it is supposed to show. The plugin is ~1.7x
        faster than stock per step, and force+energy reports 1.08-1.16x -- the
        same large constant added to both sides of a ratio drags it toward 1.
      - It moves with the HOST, not the GPU. Per-run deviations of plugin and
        stock force+energy correlate at +0.926 (against +0.157 for step), and 4
        of 42 repeat runs sat 11-32% off their siblings with step time flat.
        The GPU is exclusive to us; the CPU is shared with whatever else the
        runner is doing.

    Dropping getForces removes exactly that term. getState(getEnergy=True) never
    reaches Context.cpp:100-103, so no force download, no permuted scatter, no
    SWIG marshalling -- what is left is the GPU evaluating the system, which is
    the part we write.

    WHAT IT IS NOT

    Not the force path, and the difference is not free. Because
    LangevinMiddleIntegrator::kineticEnergyRequiresForce() returns false,
    Context.cpp:95-97 passes includeForces=false all the way down, so the force
    accumulation and the fixed-point scatter into OpenMM's buffer do not happen.
    Those are real work that real MD does, and the ratio here is smaller than the
    step ratio partly for that reason. plugin_step_ms remains the production-path
    number; this one is the cleanly measurable one.

    Nor is it cupo alone: calcForcesAndEnergy loops every force in the System
    (ContextImpl.cpp:310-311), so the bonded terms and, under NPT, the barostat
    are in this number too, on both sides. Ours is what NonbondedForceImpl
    contributes -- the largest part, but not the whole.

    Ungated for now, deliberately: no repeat history exists for it yet, and a
    metric gates here only once repeated runs on one commit have shown it can
    carry a threshold. It is the candidate to gate next if it proves as steady
    as the reasoning above predicts -- see _GATED.
    """
    for _ in range(5):
        context.setPositions(positions)
        context.getState(getEnergy=True)

    samples = []
    for _ in range(25):
        # setPositions for the same reason as above: it invalidates the cached
        # state, so each timed call is a full evaluation rather than a lookup.
        context.setPositions(positions)
        t0 = time.perf_counter()
        context.getState(getEnergy=True)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


# ── Baseline definition ───────────────────────────────────────────────────────

_PLUGIN_KEYS = (
    "plugin_step_ms",
    "plugin_force_and_energy_ms",
    "plugin_ns_per_day",
    "stock_step_ms",
    "stock_force_and_energy_ms",
    "stock_ns_per_day",
    # Energy-only, the host round trip left out. See _time_energy for why this
    # is the metric worth watching, and why it is not gated yet.
    "plugin_energy_ms",
    "stock_energy_ms",
)


def _sig(x, n=4):
    """Round x to n significant digits."""
    if x == 0:
        return 0.0
    from math import log10, floor
    return round(x, -int(floor(log10(abs(x)))) + n - 1)


def _current_baseline(measured: dict) -> dict:
    baseline = {key: _sig(measured[key]) for key in _PLUGIN_KEYS}
    baseline["speedup_step"] = _sig(measured["speedup_step"])
    baseline["speedup_force_and_energy"] = _sig(
        measured["speedup_force_and_energy"])
    baseline["speedup_energy"] = _sig(measured["speedup_energy"])
    for key in ("fixture", "validation"):
        if key in measured:
            baseline[key] = measured[key]
    return baseline


# ── Dataset setup ─────────────────────────────────────────────────────────────

# The benchmark systems by group (golden_runs.GROUPS), each group in column order --
# the single source of truth. Each group is a CI job of its own, so the groups
# measure in parallel, and keeps its goldens in its own tree.
#
# Each system is benched when its data is in the tree: the extra datasets and TYK2
# cases come from cupo-repo.ini at the repository root when present, and drop out
# where they are not (_SYSTEMS below; data/datasets.py, data/tyk2_alchemical.py).
from data.datasets import available_datasets  # noqa: E402
from data.tyk2_alchemical import SYSTEM_NAMES as _ALCHEMICAL_NAMES  # noqa: E402

_SYSTEM_GROUPS = {
    "main": ["cellulose", "water_100k", "apoa1", "apoa1_ljpme", "tyk2", "dhfr",
             "dhfr_tip4p"] + list(_ALCHEMICAL_NAMES),
    # bace in three cell shapes: the cubic cell (data/bace, no suffix) and its
    # re-solvated rhombic dodecahedron and truncated octahedron of the same width:
    # the orthorhombic versus triclinic
    # comparison, per GPU, on a globular solute (apoa1 is a membrane protein, whose
    # slab geometry is not what a tilted cell is for). compare_golden_plots derives
    # the shape ratios from these three cells, so the cube stays with its tilted
    # twins: a ratio wants all three measured in one job, on one node. bace's
    # goldens predate the group and live in main's tree; golden_runs falls back to
    # it until the triclinic group has goldens of its own.
    "triclinic": ["bace", "bace_dodeca", "bace_octa"],
}
assert [g for g, _, _ in golden_runs.GROUPS] == list(_SYSTEM_GROUPS), \
    "golden_runs.GROUPS and _SYSTEM_GROUPS name the same groups, in the same order"
_GROUP_OF = {s: g for g, systems in _SYSTEM_GROUPS.items() for s in systems}
assert len(_GROUP_OF) == sum(map(len, _SYSTEM_GROUPS.values())), "a system in two groups"


def _group_of(system_name: str) -> str:
    """The group a system's numbers are filed under. A system in no group -- one a
    test makes up -- is main's, as every system was before there were groups."""
    return _GROUP_OF.get(system_name, golden_runs.MAIN_GROUP)

# The group this run measures: CUPO_BENCH_GROUP, which each CI benchmark stage sets
# (ci/scripts/stages.py). Unset, every group, which is what a run by hand wants.
_BENCH_GROUP = os.environ.get("CUPO_BENCH_GROUP") or None
if _BENCH_GROUP is not None and _BENCH_GROUP not in _SYSTEM_GROUPS:
    raise ValueError("CUPO_BENCH_GROUP=%r: expected one of %s"
                     % (_BENCH_GROUP, ", ".join(_SYSTEM_GROUPS)))
# Method variants reuse the original coordinates, topology and force field.
_LJPME_SYSTEMS = {"apoa1_ljpme": "apoa1"}
_ON_DISK = set(available_datasets()) | set(_ALCHEMICAL_NAMES)
_ON_DISK.update(name for name, source in _LJPME_SYSTEMS.items() if source in _ON_DISK)
_SYSTEMS = [s for g, systems in _SYSTEM_GROUPS.items()
            if _BENCH_GROUP in (None, g) for s in systems if s in _ON_DISK]
_ALCHEMICAL_SYSTEMS = frozenset(_ALCHEMICAL_NAMES)
_PRECISIONS = ("mixed", "single")  # golden is nested golden[precision][system]
_ENSEMBLES = ("nvt", "npt")
# Filled in as each system is benched: system name -> atom count. Appended to the
# (optional) plot's x-axis labels at runtime (see _plot_label), so the labels track
# the data files instead of being hardcoded.
_SYSTEM_NATOMS: dict = {}


def _swap_in_nv_force(nv, system, box_safety_factor):
    """Replace the System's single NonbondedForce with an NvOpenMM::NonbondedForce.

    Spelled out rather than imported from conftest, because this module also runs
    as a standalone CLI where conftest is not importable. The package exposes the
    force class and no System-level helper -- converting is just constructing a
    force, and which force to convert, and whether to drop the original, are the
    caller's to decide.
    """
    import openmm as mm
    index = [i for i in range(system.getNumForces())
             if isinstance(system.getForce(i), mm.NonbondedForce)][0]
    system.addForce(nv.NonbondedForce(system.getForce(index), box_safety_factor))
    system.removeForce(index)
    return system


def _load_system(name: str):
    """Load (system, positions) for a dataset.

    Reads the system's solvated.pdb.gz and assigns forces from the per-system
    forcefield (or the default).
    """
    if name in _ALCHEMICAL_SYSTEMS:
        from data.tyk2_alchemical import load_tyk2_alchemical
        dataset = load_tyk2_alchemical(name)
        return dataset.create_system(), dataset.positions*unit.nanometer
    from data.datasets import load_dataset

    dataset_name = _LJPME_SYSTEMS.get(name, name)
    dataset = load_dataset(dataset_name)
    forcefield = _SYSTEM_FORCEFIELD.get(dataset_name, FORCEFIELD)
    system = _build_system(dataset.topology, forcefield=forcefield)
    if name in _LJPME_SYSTEMS:
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setNonbondedMethod(mm.NonbondedForce.LJPME)
                force.setUseDispersionCorrection(False)
    return system, dataset.positions


# ── Benchmark tests ───────────────────────────────────────────────────────────

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module", params=_PRECISIONS)
def precision(request):
    """CUDA platform precision the golden is keyed on (mixed | single)."""
    return request.param


@pytest.fixture(scope="module", params=_ENSEMBLES)
def ensemble(request):
    """Benchmark ensemble: NVT (no barostat) or NPT (MonteCarloBarostat)."""
    global BAROSTAT
    BAROSTAT = (request.param == "npt")
    return request.param


# When True, the plugin path uses the Nvidia LangevinMiddle integrator (its own
# SETTLE/SHAKE + fused periodic CoM removal + whole-step CUDA graph) instead of
# stock LangevinMiddle. On by default, so both the pytest golden suite and the CLI
# (via --nv-integrator, also default on) measure the plugin under the nv
# integrator; golden/nvt.json therefore holds nv-integrator numbers. The CLI
# overrides this per-run from --nv-integrator/--no-nv-integrator.
_USE_NV_INTEGRATOR = True


def _alchemical_snapshot(context):
    state = context.getState(getEnergy=True, getForces=True)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole/unit.nanometer)).copy()
    if not np.isfinite(energy) or not np.isfinite(forces).all():
        raise AssertionError("Alchemical benchmark produced nonfinite energy or forces")
    return energy, forces


def _validate_alchemical_snapshot(actual, expected, dataset):
    energy, forces = actual
    reference_energy, reference_forces = expected
    relative = float(np.linalg.norm(forces-reference_forces)/max(np.linalg.norm(reference_forces), 1e-30))
    energy_error = abs(energy-reference_energy)
    # Performance uses cuPo's default Prolate split against stock PME. The
    # fixture correctness suite separately compares the same PME decomposition.
    assert relative < 1e-3, f"{dataset.name}: force relative RMS error {relative}"
    assert energy_error/max(abs(reference_energy), 1.) < 1e-4, (
        f"{dataset.name}: energy error {energy_error} kJ/mol")
    result = {"energy_abs_kj_mol": energy_error, "force_relative_rms": relative}
    atoms = np.asarray(dataset.metadata["alchemical_atoms"], dtype=int)
    ligand_error = float(np.linalg.norm((forces-reference_forces)[atoms]) /
                         max(np.linalg.norm(reference_forces[atoms]), 1e-30))
    assert ligand_error < 2e-3, f"{dataset.name}: ligand force relative RMS error {ligand_error}"
    result["ligand_force_relative_rms"] = ligand_error
    return result


def _bench_system(system_name: str, precision: str) -> dict:
    """Time plugin and stock OpenMM on one (system, precision) and return the
    median timings dict. Shared by the pytest fixture and the CLI golden-generation
    mode so both measure identically. The plugin package must already be loaded
    (replace_nonbonded() loads the plugin lazily; nothing selects a directory)."""
    import nv_openmm_plugin

    alchemical = None
    if system_name in _ALCHEMICAL_SYSTEMS:
        from data.tyk2_alchemical import load_tyk2_alchemical
        alchemical = load_tyk2_alchemical(system_name)
        system = alchemical.create_system()
        positions = alchemical.positions*unit.nanometer
        # Match the native integrator's COM interval on the stock side too.
        for force in system.getForces():
            if isinstance(force, mm.CMMotionRemover):
                force.setFrequency(100)
    else:
        system, positions = _load_system(system_name)
    timestep_fs = float(alchemical.metadata["timestep_fs"]) if alchemical else TIMESTEP_FS
    _SYSTEM_NATOMS[system_name] = system.getNumParticles()  # for runtime plot labels
    system_xml = mm.XmlSerializer.serialize(system)
    platform = mm.Platform.getPlatformByName("CUDA")

    # --- Stock OpenMM (mixed precision) ---
    # Mixed is OpenMM's production-standard precision (fp32 forces, fp64
    # position/velocity accumulation). It is the accuracy-matched peer for the
    # plugin, whose fixed-point force accumulation is mixed-like; pure single
    # is faster but not production-viable for energy conservation.
    system_stock = mm.XmlSerializer.deserialize(system_xml)
    if BAROSTAT:
        # NPT: same 1-bar barostat on both sides for a fair comparison.
        system_stock.addForce(mm.MonteCarloBarostat(
            1.0 * unit.bar, TEMPERATURE_K * unit.kelvin, 25))
    integrator_stock = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, FRICTION_PS / unit.picosecond,
        timestep_fs * unit.femtoseconds)
    integrator_stock.setConstraintTolerance(1e-5)
    # DeterministicForces=true makes stock's PME charge spreading fixed-point
    # (deterministic) instead of float atomicAdd (non-deterministic but faster),
    # matching the plugin's deterministic nonbonded (int64 towerplate energy +
    # int32 SPME charge grid). Without it the plugin is unfairly compared against
    # a stock that buys speed by forgoing reproducibility -- something the plugin
    # deliberately does not do. (It only affects PME on CUDA; it is a no-op on the
    # plugin context below, whose NonbondedForce is removed so OpenMM does no
    # charge spreading there.)
    stock_props = {"Precision": precision}
    if STOCK_DETERMINISTIC_FORCES:
        stock_props["DeterministicForces"] = "true"
    ctx_stock = mm.Context(
        system_stock, integrator_stock, platform, stock_props)
    ctx_stock.setPositions(positions)
    if alchemical and alchemical.velocities is not None:
        ctx_stock.setVelocities(alchemical.velocities*unit.nanometer/unit.picosecond)
    reference = _alchemical_snapshot(ctx_stock) if alchemical else None
    stock_steps = _warmup(ctx_stock)

    stock_step_ms = _stabilize(lambda: _time_step(ctx_stock, stock_steps),
                               f"{system_name}/stock_step")
    stock_final_energy = _alchemical_snapshot(ctx_stock)[0] if alchemical else None
    stock_fe_ms = _stabilize(
        lambda: _time_force_and_energy(ctx_stock, positions),
        f"{system_name}/stock_fe")
    stock_energy_ms = _stabilize(
        lambda: _time_energy(ctx_stock, positions),
        f"{system_name}/stock_energy")

    del ctx_stock, integrator_stock

    # --- Plugin ---
    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    # cupo has no Verlet skin -- it rebuilds the cell list every step (cells are
    # exactly the cutoff), and the plate/stake caps are calibrated for that. Under
    # NPT, pass the cell-grid pre-allocation headroom explicitly so a barostat
    # expansion never reallocs (1.0 == NVT allocation).
    if alchemical:
        from data.tyk2_alchemical import convert_to_plugin
        convert_to_plugin(system_plugin, NPT_BOX_SAFETY if BAROSTAT else 1.0)
    else:
        _swap_in_nv_force(nv_openmm_plugin, system_plugin,
                          NPT_BOX_SAFETY if BAROSTAT else 1.0)
    if BAROSTAT:
        _add_npt_barostat(system_plugin, plugin=True)
    if _USE_NV_INTEGRATOR:
        # Nvidia integrator: periodic fused CoM removal (strip the stock every-step
        # CMMotionRemover) + our own SETTLE/SHAKE, so the whole step is captured as
        # one CUDA graph -- ON by default; CUPO_PLUGIN_CUDA_GRAPH=0 falls back
        # to the eager per-step-launch path (CudaLangevinMiddleKernel.cu).
        for _i, _f in enumerate(system_plugin.getForces()):
            if isinstance(_f, mm.CMMotionRemover):
                system_plugin.removeForce(_i)
                break
        integrator_plugin = nv_openmm_plugin.LangevinMiddleIntegrator(
            TEMPERATURE_K * unit.kelvin, FRICTION_PS / unit.picosecond,
            timestep_fs * unit.femtoseconds)
    else:
        integrator_plugin = mm.LangevinMiddleIntegrator(
            TEMPERATURE_K * unit.kelvin, FRICTION_PS / unit.picosecond,
            timestep_fs * unit.femtoseconds)
    integrator_plugin.setConstraintTolerance(1e-5)
    # Mixed here too: integrator-matched with stock (cupo does the forces in
    # fp32 fixed-point regardless; this pins the integrator accumulation to fp64
    # so the comparison isn't skewed by a cheaper fp32 integrator on one side).
    ctx_plugin = mm.Context(
        system_plugin, integrator_plugin, platform, {"Precision": precision})
    ctx_plugin.setPositions(positions)
    if alchemical and alchemical.velocities is not None:
        ctx_plugin.setVelocities(alchemical.velocities*unit.nanometer/unit.picosecond)
    validation = (_validate_alchemical_snapshot(
        _alchemical_snapshot(ctx_plugin), reference, alchemical) if alchemical else None)
    plugin_steps = _warmup(ctx_plugin)

    plugin_step_ms = _stabilize(lambda: _time_step(ctx_plugin, plugin_steps),
                                f"{system_name}/plugin_step")
    plugin_final_energy = _alchemical_snapshot(ctx_plugin)[0] if alchemical else None
    plugin_fe_ms = _stabilize(
        lambda: _time_force_and_energy(ctx_plugin, positions),
        f"{system_name}/plugin_fe")
    plugin_energy_ms = _stabilize(
        lambda: _time_energy(ctx_plugin, positions),
        f"{system_name}/plugin_energy")

    if _USE_NV_INTEGRATOR:
        print("WHOLE_STEP_GRAPH_COUNTS " + json.dumps({
            "system": system_name, "precision": precision,
            "captures": integrator_plugin.getWholeStepGraphCaptureCount(),
            "replays": integrator_plugin.getWholeStepGraphReplayCount(),
        }), flush=True)
    del ctx_plugin, integrator_plugin

    ms_per_day = 86_400_000.0
    ns_per_step = timestep_fs / 1e6  # timestep_fs / (fs per ns)
    plugin_steps_per_ms = 1.0 / plugin_step_ms
    stock_steps_per_ms = 1.0 / stock_step_ms
    results = {
        "plugin_step_ms": plugin_step_ms,
        "plugin_force_and_energy_ms": plugin_fe_ms,
        "plugin_ns_per_day": plugin_steps_per_ms * ms_per_day * ns_per_step,
        "stock_step_ms": stock_step_ms,
        "stock_force_and_energy_ms": stock_fe_ms,
        "stock_ns_per_day": stock_steps_per_ms * ms_per_day * ns_per_step,
        "plugin_energy_ms": plugin_energy_ms,
        "stock_energy_ms": stock_energy_ms,
    }
    results["speedup_step"] = stock_step_ms / plugin_step_ms
    results["speedup_force_and_energy"] = stock_fe_ms / plugin_fe_ms
    results["speedup_energy"] = stock_energy_ms / plugin_energy_ms
    results["system_name"] = system_name
    results["precision"] = precision
    if alchemical:
        results["fixture"] = alchemical.benchmark_metadata()
        results["fixture"].update(plugin_method="Prolate", stock_method="PME",
            plugin_integrator="cuPo" if _USE_NV_INTEGRATOR else "OpenMM",
            stock_integrator="OpenMM", com_interval_steps=100,
            initial_velocities="vendored" if alchemical.velocities is not None else "zero")
        validation.update(stock_final_energy_kj_mol=stock_final_energy,
                          plugin_final_energy_kj_mol=plugin_final_energy)
        results["validation"] = validation
    return results


@pytest.fixture(scope="module", params=_SYSTEMS)
def plugin_timings(request, plugin_dir, gpu_hw, precision, ensemble):
    """Time plugin and stock OpenMM on a system, return median timings."""
    gpu_hw.wait_for_locked_clocks()
    results = _bench_system(request.param, precision)
    telemetry = gpu_hw.get_telemetry()

    # Record the GPU-specific run conditions (launch config, sm_count, locked
    # clocks, step budget, OpenMM build) at the GPU level; precision-independent,
    # shared by both precisions. What the clocks actually did goes with the system.
    _record_config_block(_gpu_config_block(gpu_hw.locked), request.param)
    _record_telemetry(telemetry, request.param, precision)
    return results


# ── Test functions ────────────────────────────────────────────────────────────

def test_plugin_throughput(plugin_timings):
    """Plugin step throughput in ns/day -- the one metric that gates the run.

    Its own test rather than a second check inside test_plugin_step_time, so that
    the gate has a node name of its own: what fails a pipeline should be findable
    by the name it failed under, and `-k throughput` should run exactly the thing
    that can turn CI red.

    ns/day rather than the ms/step it is derived from, because it is the number
    the project is judged on and the one every report quotes. The two are the same
    measurement inverted, so gating either is arithmetically identical -- this one
    names it the way people read it. Higher is better, hence higher_is_worse=False.
    """
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "plugin_ns_per_day"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    print(f"\n  [{system_name}] {key}: {val:.1f} ns/day")
    print(f"  golden: {entry[key]:.1f} ns/day  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision,
                      higher_is_worse=False)


def test_plugin_step_time(plugin_timings):
    """Plugin step time (force eval), reported against golden. Not gated -- the
    gate is test_plugin_throughput, which measures the same thing as ns/day."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "plugin_step_ms"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    print(f"\n  [{system_name}] {key}: {val:.4f} ms")
    print(f"  golden: {entry[key]:.4f} ms  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision)


def test_plugin_force_and_energy_time(plugin_timings):
    """Plugin force+energy time, reported against golden. Not gated: it lands more
    than 10% off golden on 22-24% of measurements across repeat runs of one
    commit."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "plugin_force_and_energy_ms"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    print(f"\n  [{system_name}] {key}: {val:.4f} ms")
    print(f"  golden: {entry[key]:.4f} ms  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision)


def test_stock_step_time(plugin_timings):
    """Stock OpenMM step time (reference for speedup comparison)."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "stock_step_ms"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    print(f"\n  [{system_name}] {key}: {val:.4f} ms")
    print(f"  golden: {entry[key]:.4f} ms  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision)


def test_stock_force_and_energy_time(plugin_timings):
    """Stock OpenMM force+energy time (reference for speedup comparison)."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "stock_force_and_energy_ms"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    print(f"\n  [{system_name}] {key}: {val:.4f} ms")
    print(f"  golden: {entry[key]:.4f} ms  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision)


def test_plugin_speedup_step(plugin_timings):
    """Plugin step speedup vs stock, reported against golden. Not gated: a ratio
    inherits the noise of both sides, and a stock-side fluke frozen into a golden
    would fail every run after it."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    entry = _compared_entry(plugin_timings, system_name, precision)
    speedup = plugin_timings["speedup_step"]
    stock = plugin_timings["stock_step_ms"]
    plugin = plugin_timings["plugin_step_ms"]
    print(f"\n  [{system_name}] stock: {stock:.4f} ms ({plugin_timings['stock_ns_per_day']:.1f} ns/day), "
          f"plugin: {plugin:.4f} ms ({plugin_timings['plugin_ns_per_day']:.1f} ns/day), "
          f"speedup: {speedup:.1f}x (golden: {entry['speedup_step']:.1f}x)")
    _check_regression("speedup_step", speedup, entry["speedup_step"],
                      system_name, precision, higher_is_worse=False)


def test_plugin_speedup_force_and_energy(plugin_timings):
    """Plugin force+energy speedup vs stock, reported against golden. Not gated --
    the noisiest metric in the suite, and the one behind most of the spurious
    failures in that pipeline (see "Which metrics can fail a run")."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    entry = _compared_entry(plugin_timings, system_name, precision)
    speedup = plugin_timings["speedup_force_and_energy"]
    stock = plugin_timings["stock_force_and_energy_ms"]
    plugin = plugin_timings["plugin_force_and_energy_ms"]
    print(f"\n  [{system_name}] stock: {stock:.4f} ms, "
          f"plugin: {plugin:.4f} ms, "
          f"speedup: {speedup:.1f}x "
          f"(golden: {entry['speedup_force_and_energy']:.1f}x)")
    _check_regression("speedup_force_and_energy", speedup,
                      entry["speedup_force_and_energy"],
                      system_name, precision, higher_is_worse=False)


# ── CLI entry point ───────────────────────────────────────────────────────────
# Run directly (not under pytest) in one of two modes:
#
#   Golden mode (default, no --pdb): benchmark the full system x precision suite,
#   DIFF the measured perf against this GPU's golden nvt.json block, and print the
#   resulting JSON block to copy-paste into nvt.json. This CLI never writes the
#   golden itself (unlike the drift-guarded pytest path):
#       python test_bench_plugin.py
#
#   Single system (--pdb): benchmark one serialized system (e.g. for A/B'ing two
#   libcupo.so builds):
#       python test_bench_plugin.py --pdb <file.pdb[.gz]> [options]
#
# Defaults come from the shared constants above, so the CLI and the pytest
# benchmark agree on the physics.

def _add_npt_barostat(system, temperature_k=TEMPERATURE_K, plugin=False) -> str:
    """Add a 1-bar Monte Carlo barostat (25-step interval).

    Same move and same acceptance criterion either way; `plugin` selects WHOSE
    implementation runs it. The plugin arm gets NvMonteCarloBarostat, because the
    barostat is part of what the plugin provides -- it takes its trial energies
    from ContextImpl directly (skipping the kinetic energy getState always
    computes and a barostat never reads) and scales coordinates atom-parallel
    rather than one thread per molecule. Stock necessarily keeps OpenMM's, so the
    comparison stays "what you get with the plugin" vs "what you get without it".

    The plugin's cell-grid headroom is set separately and explicitly, via
    replace_nonbonded(npt_box_safety_factor=). Returns a summary string.
    """
    if plugin:
        # Self-sufficient about the import: golden mode puts the package on the
        # path early, but the single-system CLI does it later (inside the engine
        # branch), and this runs before that.
        py = str(_PLUGIN_DIR / "python")
        if py not in sys.path:
            sys.path.insert(0, py)
        import nv_openmm_plugin
        system.addForce(nv_openmm_plugin.MonteCarloBarostat(
            1.0 * unit.bar, temperature_k * unit.kelvin, frequency=25))
        return f"nv-monte-carlo:1bar,temperature_k={temperature_k:g},freq=25"
    system.addForce(mm.MonteCarloBarostat(
        1.0 * unit.bar, temperature_k * unit.kelvin, 25))
    return f"monte-carlo:1bar,temperature_k={temperature_k:g},freq=25"


def _configure_barostat(system, args) -> str:
    """Add a Monte Carlo barostat (NPT) if requested, else NVT. Systems are built
    fresh from the forcefield, so there is never a serialized barostat to keep or
    remove; a barostat only makes sense as MonteCarlo here.

    Whose barostat follows the engine, exactly as in golden mode: a plugin run
    gets the plugin's, a --stock run gets OpenMM's. The summary line records
    which, so a profile or log can never leave it ambiguous.
    """
    if not args.barostat:
        return "none"
    summary = _add_npt_barostat(system, args.temperature_k, plugin=not args.stock)
    return f"{summary},box_safety={args.npt_box_safety:g}"


def _configure_cutoff(system, cutoff_nm: Optional[float]) -> str:
    nonbonded = [f for f in system.getForces()
                 if isinstance(f, mm.NonbondedForce)]
    if not nonbonded:
        if cutoff_nm is None:
            return "none"
        raise ValueError("--cutoff-nm requires a System with NonbondedForce")
    if cutoff_nm is not None:
        for f in nonbonded:
            f.setCutoffDistance(cutoff_nm * unit.nanometer)
    cutoffs = sorted({round(f.getCutoffDistance().value_in_unit(unit.nanometer), 9)
                      for f in nonbonded})
    return ",".join(f"{c:g}" for c in cutoffs)


def _parse_cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark an OpenMM system through the Nvidia/Cupo plugin")
    p.add_argument("--pdb", type=Path, default=None,
                   help="PDB (or .pdb.gz) with the solvated system topology + positions. "
                        "Omit to run default golden mode: benchmark the full system x "
                        "precision suite, DIFF the measured perf against this GPU's golden "
                        "nvt.json block, and print the JSON block to copy-paste in "
                        "(the CLI never writes the golden itself).")
    p.add_argument("--stock", action="store_true",
                   help="Benchmark stock OpenMM (skip the cupo plugin) for comparison")
    p.add_argument("--nv-integrator", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the Nvidia LangevinMiddle integrator (periodic fused CoM "
                        "removal + our own SETTLE/SHAKE) for the plugin path instead of "
                        "stock LangevinMiddle (default: on; pass --no-nv-integrator to "
                        "use stock). The whole-step CUDA graph is on by default; disable "
                        "it with the env var CUPO_PLUGIN_WHOLE_STEP_GRAPH=0.")
    p.add_argument("--forcefield", nargs="+", default=list(FORCEFIELD),
                   help="OpenMM ForceField XML files used to assign forces")
    p.add_argument("--steps", type=int, default=PRODUCTION_STEPS,
                   help="Timed production steps")
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS,
                   help="Untimed warmup steps before timing")
    p.add_argument("--timestep-fs", type=float, default=TIMESTEP_FS,
                   help="Integrator timestep in femtoseconds")
    p.add_argument("--temperature-k", type=float, default=TEMPERATURE_K,
                   help="Integrator temperature in kelvin")
    p.add_argument("--precision", choices=("single", "mixed", "double"),
                   default=PRECISION, help="CUDA platform precision")
    p.add_argument("--hmr", action=argparse.BooleanOptionalAction, default=APPLY_HMR,
                   help="Apply hydrogen mass repartitioning (needed for a 4 fs step)")
    p.add_argument("--cutoff-nm", type=float, default=CUTOFF_NM,
                   help="Override NonbondedForce cutoff distance in nanometers")
    p.add_argument("--barostat", action=argparse.BooleanOptionalAction,
                   default=BAROSTAT,
                   help="Add a MonteCarloBarostat (NPT, 1 bar); default is NVT")
    p.add_argument("--npt-box-safety", type=float, default=NPT_BOX_SAFETY,
                   help="NPT: plugin cell-grid pre-allocation headroom factor "
                        "(replace_nonbonded npt_box_safety_factor); only used with --barostat")
    p.add_argument("--seed", type=int, default=12345,
                   help="Deterministic RNG seed (0 = random each run)")
    p.add_argument("--check-clocks-only", action="store_true",
                   help="Lock the clocks at the tests/hw_config.py values, print what the GPU does "
                        "for a few seconds, and exit without loading the plugin or "
                        "running benchmarks.")
    p.add_argument("--make-plot", "--make_plot", dest="make_plot", metavar="PATH",
                   default=None,
                   help="Golden mode only: after measuring, render a grouped bar "
                        "chart of stock vs plugin throughput (ns/day) per system, "
                        "one panel per precision, to PATH (e.g. perf.png). Requires "
                        "matplotlib.")
    return p.parse_args()


# Golden-mode diff metrics: (key, higher_is_better). Plugin timings are the headline;
# stock is the reference (a stock shift reflects the OpenMM build, not the cupo kernel);
# the speedups are what the pytest regression gate asserts.
_DIFF_METRICS = (
    ("plugin_step_ms", False),
    ("plugin_force_and_energy_ms", False),
    ("plugin_ns_per_day", True),
    ("stock_step_ms", False),
    ("stock_force_and_energy_ms", False),
    ("stock_ns_per_day", True),
    ("speedup_step", True),
    ("speedup_force_and_energy", True),
)

_DIFF_SAME_TOL = 0.5  # |Δ%| below this reads as noise ("same")


def _fmt_num(x) -> str:
    return "-" if x is None else f"{x:.4g}"


def _delta_verdict(golden, measured, higher_is_better) -> tuple:
    """(Δ% string, verdict) for measured vs golden. verdict in
    new/same/better/worse, keyed to the metric's direction."""
    if golden is None:
        return ("", "new")
    if golden == 0:
        return ("", "same" if measured == 0 else "new")
    pct = (measured - golden) / golden * 100.0
    if abs(pct) < _DIFF_SAME_TOL:
        return (f"{pct:+.1f}%", "same")
    better = (measured > golden) if higher_is_better else (measured < golden)
    return (f"{pct:+.1f}%", "better" if better else "worse")


def _print_config_drift(old_config: dict, new_config: dict) -> None:
    """Flag run-condition changes vs the golden's config block -- perf is only
    comparable when clocks, OpenMM build, launch config, and step budget match."""
    drift = []
    for f in ("openmm_version", "openmm_revision", "graphics_clock_mhz",
              "memory_clock_mhz", "cuda_version", "driver_version",
              "cpu_model", "cpu_max_mhz", "cpu_min_mhz",
              "warmup_steps", "production_steps", "sm_count"):
        o, n = old_config.get(f), new_config.get(f)
        if o != n:
            drift.append(f"{f}: {o} -> {n}")
    lo = (old_config.get("kernel_launch_configs") or {}).get("cuda_towerplate_direct_force", {})
    ln = (new_config.get("kernel_launch_configs") or {}).get("cuda_towerplate_direct_force", {})
    for f in ("launch_bounds_max_threads_per_block", "launch_bounds_min_blocks_per_mp",
              "execution_grid_dim", "execution_block_dim", "shared_mem_bytes"):
        o, n = lo.get(f), ln.get(f)
        if o != n:
            drift.append(f"launch.{f}: {o} -> {n}")
    if drift:
        print("\n  conditions CHANGED vs golden (perf not directly comparable):")
        for d in drift:
            print(f"      {d}")
    else:
        print("\n  conditions match golden (clocks, OpenMM build, launch config, steps)")


def _print_golden_diff(gpu: str, old_config: dict, new_config: dict,
                       old_cells: dict, measured_cells: dict) -> None:
    print("\n" + "=" * 84)
    print(f"perf diff vs golden [{gpu}]  (measured vs current nvt.json)")
    print("=" * 84)
    _print_config_drift(old_config, new_config)
    hdr = (f"{'precision':<10}{'system':<12}{'metric':<28}"
           f"{'golden':>11}{'measured':>11}{'Δ%':>9}  verdict")
    print("\n" + hdr)
    print("-" * (len(hdr) + 2))
    for precision in _PRECISIONS:
        for system_name in _SYSTEMS:
            measured = measured_cells.get((precision, system_name))
            if measured is None:
                continue
            gold = old_cells.get((precision, system_name)) or {}
            first = True
            for key, higher_better in _DIFF_METRICS:
                g, m = gold.get(key), measured.get(key)
                dpct, verdict = _delta_verdict(g, m, higher_better)
                # stock timings are the reference (a faster stock isn't a plugin win),
                # so report their drift but don't label it better/worse.
                if key.startswith("stock_") and verdict in ("better", "worse"):
                    verdict = "ref"
                pcol = precision if first else ""
                scol = system_name if first else ""
                print(f"{pcol:<10}{scol:<12}{key:<28}"
                      f"{_fmt_num(g):>11}{_fmt_num(m):>11}{dpct:>9}  {verdict}")
                first = False
            print()


# Okabe-Ito colorblind-safe pair; identity is also carried by bar position + labels.
_PLOT_STOCK, _PLOT_PLUGIN = "#E69F00", "#0072B2"


def _plot_label(system_name: str) -> str:
    """Pretty x-axis label = system name + runtime atom count (e.g. 'dhfr\\n(24k)').
    Falls back to just the name if the system wasn't measured this run."""
    n = _SYSTEM_NATOMS.get(system_name)
    return f"{system_name}\n({round(n / 1000)}k)" if n else system_name


def _plot_cpu_label(config: dict) -> str:
    """Compact CPU provenance label for the benchmark plot."""
    model = config.get("cpu_model") or "CPU unknown"
    maximum = config.get("cpu_max_mhz")
    if maximum:
        freq = f"{maximum:.0f} MHz max"
    else:
        freq = "frequency n/a"
    return f"CPU {model}  ·  {freq}"


def _make_plot(measured_cells: dict, gpu: str, out_path: str,
               config: dict = None) -> None:
    """Render a grouped bar chart of stock vs plugin throughput (ns/day) per system,
    one panel per precision, from the just-measured results. Bars are labeled with
    ns/day and the plugin/stock speedup; systems are ordered by plugin throughput.

    `config` is the GPU config block (see _gpu_config_block); it drives a two-line
    provenance caption under the title (clocks, cutoff, ensemble, stock
    DeterministicForces, run time, git hash) so a saved plot is self-describing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("--make-plot requires matplotlib (pip install matplotlib)")

    ink, muted, grid = "#202124", "#5F6368", "#DADCE0"
    precisions = [p for p in _PRECISIONS
                  if any((p, s) in measured_cells for s in _SYSTEMS)]
    if not precisions:
        print("  (--make-plot: no measured data to plot)", flush=True)
        return
    fig, axes = plt.subplots(1, len(precisions), figsize=(6.3 * len(precisions), 6.0),
                             squeeze=False)
    axes = axes[0]
    fig.suptitle(f"cupo plugin vs stock OpenMM -- throughput per system  ({gpu})",
                 fontsize=13.5, fontweight="bold", color=ink, y=0.965)

    # Run-condition provenance: two recessive lines under the title so a saved
    # plot is self-describing (clocks, cutoff, ensemble, det-forces, when, git).
    cfg = config or {}
    gr, mem = cfg.get("graphics_clock_mhz"), cfg.get("memory_clock_mhz")
    clocks = f"{gr} / {mem} MHz (gr/mem)" if gr and mem else "clocks n/a"
    det = cfg.get("stock_deterministic_forces", STOCK_DETERMINISTIC_FORCES)
    npt = cfg.get("barostat", BAROSTAT)
    ensemble = "NPT (1 bar)" if npt else "NVT"
    # Frozen alchemical XML carries its own cutoff, independent of the ordinary
    # benchmark setting. A mixed plot must describe the settings actually run.
    cutoffs = sorted({cell.get("fixture", {}).get("cutoff_nm", CUTOFF_NM)
                      for (precision, system), cell in measured_cells.items()
                      if precision in precisions and system in _SYSTEMS})
    cutoff_label = ("cutoff " if len(cutoffs) == 1 else "cutoffs ")
    cutoff_label += "/".join(f"{value:g}" for value in cutoffs) + " nm"
    meta1 = (f"{ensemble}  ·  {cutoff_label}  ·  clocks {clocks}"
             f"  ·  stock DeterministicForces: {'on' if det else 'off'}")
    meta2 = _plot_cpu_label(cfg)
    meta3 = (f"{time.strftime('%Y-%m-%d %H:%M %Z')}  ·  git {_git_describe()}"
             f"  ·  OpenMM {_openmm_version()}")
    fig.text(0.5, 0.912, meta1, ha="center", fontsize=8, color=muted)
    fig.text(0.5, 0.888, meta2, ha="center", fontsize=8, color=muted)
    fig.text(0.5, 0.864, meta3, ha="center", fontsize=8, color=muted)

    for ax, prec in zip(axes, precisions):
        systems = sorted(
            [s for s in _SYSTEMS if (prec, s) in measured_cells],
            key=lambda s: measured_cells[(prec, s)]["plugin_ns_per_day"])
        x = np.arange(len(systems))
        w = 0.38
        stock = [measured_cells[(prec, s)]["stock_ns_per_day"] for s in systems]
        plug = [measured_cells[(prec, s)]["plugin_ns_per_day"] for s in systems]
        b1 = ax.bar(x - w / 2, stock, w, label="stock OpenMM", color=_PLOT_STOCK, zorder=3)
        b2 = ax.bar(x + w / 2, plug, w, label="cupo plugin", color=_PLOT_PLUGIN, zorder=3)
        ymax = max(plug) * 1.18
        ax.set_ylim(0, ymax)
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + ymax * 0.012,
                        f"{r.get_height():.0f}", ha="center", va="bottom",
                        fontsize=8.5, color=muted)
        for i in range(len(systems)):
            sp = plug[i] / stock[i] if stock[i] else 0.0
            ax.text(i, ymax * 0.995, f"{sp:.2f}×", ha="center", va="top",
                    fontsize=9.5, fontweight="bold", color=_PLOT_PLUGIN)
        ax.set_title(f"{prec} precision", fontsize=11.5, color=ink, pad=16)
        ax.set_xticks(x)
        ax.set_xticklabels([_plot_label(s) for s in systems], fontsize=9, color=ink)
        ax.set_ylabel("throughput  (ns/day)", fontsize=10, color=muted)
        ax.tick_params(axis="y", labelsize=8.5, colors=muted, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for name in ("top", "right", "left"):
            ax.spines[name].set_visible(False)
        ax.spines["bottom"].set_color(grid)

    axes[0].legend(loc="upper left", frameon=False, fontsize=9.5,
                   handlelength=1.1, handleheight=1.1)
    integ = "nv integrator + graph" if _USE_NV_INTEGRATOR else "stock integrator both sides"
    fig.text(0.5, 0.005,
             f"labels = ns/day  |  bold × = plugin/stock speedup  |  {integ}",
             ha="center", fontsize=8, color=muted)
    fig.tight_layout(rect=[0, 0.02, 1, 0.82])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\ncupo_bench_plot written to {out_path}", flush=True)


def _loaded_lib_path(basename: str) -> str:
    """Resolved on-disk path of a shared lib currently mapped into this process
    (read from /proc/self/maps), or '<not loaded>' if it is not mapped. Reflects the
    file the dynamic loader actually resolved (via RUNPATH/LD_LIBRARY_PATH)."""
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                # basenames are distinct: 'libOpenMMNvidia.so' is not a substring of
                # 'libOpenMMNvidiaCUDA.so' (the char after the stem is 'C', not '.').
                # A versioned library is mapped under its file's name, which goes
                # on from the basename: libcupo.so.<ABI> or libcupo.so.<VERSION>.
                name = line.rstrip().rsplit("/", 1)[-1]
                if "/" in line and (name == basename or name.startswith(basename + ".")):
                    return line.rstrip().split()[-1]
    except OSError:
        pass
    return "<not loaded>"


def _print_plugin_libs() -> None:
    """Print the resolved location of the three plugin-stack shared libs actually
    loaded into this process: the OpenMM plugin API lib, its CUDA platform impl, and
    the cupo kernel runtime (a dependency resolved via LD_LIBRARY_PATH)."""
    print("cupo_plugin_libs (as loaded):", flush=True)
    for name, role in (("libOpenMMNvidia.so", "plugin API"),
                       ("libOpenMMNvidiaCUDA.so", "CUDA platform impl"),
                       ("libcupo.so", "cupo kernel runtime")):
        print(f"  {name:<24}{role:<22}{_loaded_lib_path(name)}", flush=True)


_plugin_libs_printed = False


def _print_plugin_libs_once() -> None:
    """_print_plugin_libs, but only the first time (the golden loop accelerates
    once per system x precision; the loaded set never changes after the first)."""
    global _plugin_libs_printed
    if not _plugin_libs_printed:
        _plugin_libs_printed = True
        _print_plugin_libs()


@contextlib.contextmanager
def _cli_locked_clocks():
    """For the CLI: the GPU locked at the tests/hw_config.py clocks while the block
    runs, with telemetry going. Yields the gpu_clocks.BenchClocks; the clocks are
    reset on the way out however the block ends. The pytest path does the same with
    the gpu_hw fixture."""
    clocks = gpu_clocks.BenchClocks()
    locking = False
    try:
        try:
            locking = True
            locked = clocks.lock()
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        print(f"cupo_locked_clocks graphics_mhz={locked.graphics_mhz} "
              f"memory_mhz={locked.memory_mhz}", flush=True)
        yield clocks
    finally:
        if locking:
            gpu_clocks.unlock()
        clocks.close()


def _run_golden(args) -> None:
    """Default mode (no --pdb): benchmark the full system x precision suite, DIFF the
    measured perf against this GPU's golden nvt.json block, and print the resulting
    JSON block for the user to copy-paste into nvt.json. This CLI never writes the
    golden itself -- updating it is a deliberate, manual copy-paste (the pytest path
    auto-updates a golden field on >5% drift; that is separate)."""
    # Push CLI ensemble flags onto the module globals that _bench_system,
    # _gpu_config_block, and _make_plot read (the pytest path has no args and
    # keeps the NVT defaults). Without this, --barostat is silently ignored in
    # golden/plot mode: no barostat is applied and the plot is labelled NVT.
    global BAROSTAT, NPT_BOX_SAFETY
    BAROSTAT = args.barostat
    NPT_BOX_SAFETY = args.npt_box_safety
    with _cli_locked_clocks() as clocks:
        _run_golden_locked(args, clocks)


def _run_golden_locked(args, clocks) -> None:
    """_run_golden's work, with the GPU locked."""
    cur_gr, cur_mem = clocks.locked.graphics_mhz, clocks.locked.memory_mhz

    # Make the plugin package importable when run directly. It loads the
    # libraries that ship with it -- there is nothing to point at.
    py = str(_PLUGIN_DIR / "python")
    if py not in sys.path:
        sys.path.insert(0, py)
    import nv_openmm_plugin  # noqa: F401  (replace_nonbonded() loads the plugin lazily)

    gpu = _gpu_name()

    # One file per group (_golden_out_path). The golden file IS this GPU's block --
    # the GPU is in the PATH (golden/<gpu>/{nvt,npt}.json), not a key inside it.
    # Indexing by gpu here returned None on every GPU, so old_config/old_cells were
    # always empty and every field reported "None -> <measured>": the conditions
    # check and the perf comparison both silently did nothing.
    goldens = {g: _load_golden(_golden_out_path(g)) for g in _SYSTEM_GROUPS}
    old_config = goldens[_BENCH_GROUP or golden_runs.MAIN_GROUP].get("config") or {}
    old_cells = {(p, s): goldens[_group_of(s)].get(p, {}).get(s)
                 for p in _PRECISIONS for s in _SYSTEMS}
    new_config = _gpu_config_block(clocks.locked)

    print(f"cupo_golden_bench gpu='{gpu}' graphics_mhz={cur_gr} memory_mhz={cur_mem} "
          f"systems={','.join(_SYSTEMS)} precisions={','.join(_PRECISIONS)} "
          f"warmup_steps={WARMUP_STEPS} production_steps={PRODUCTION_STEPS} "
          f"openmm={_openmm_version()}+{_openmm_revision()[:7]}", flush=True)

    measured_cells = {}
    telemetry_cells = {}
    for precision in _PRECISIONS:
        for system_name in _SYSTEMS:
            print(f"\n=== {gpu} | {precision} | {system_name} ===", flush=True)
            clocks.wait_for_locked_clocks()
            results = _bench_system(system_name, precision)
            telemetry_cells[(precision, system_name)] = clocks.get_telemetry()
            # First measurement has now converted a system, so the plugin
            # stack is resident: report what actually got loaded, once.
            _print_plugin_libs_once()
            baseline = _current_baseline(results)
            # The atom count, so a reader of the golden can put throughput per atom
            # beside ns/day (compare_golden_plots' shape ratios need it).
            baseline["natoms"] = _SYSTEM_NATOMS.get(system_name)
            measured_cells[(precision, system_name)] = baseline
            print(f"  plugin_step_ms={results['plugin_step_ms']:.4f} "
                  f"plugin_ns_per_day={results['plugin_ns_per_day']:.1f} "
                  f"speedup_step={results['speedup_step']:.3f}", flush=True)

    _print_golden_diff(gpu, old_config, new_config, old_cells, measured_cells)

    if args.make_plot:
        _make_plot(measured_cells, gpu, args.make_plot, config=new_config)

    # Assemble this GPU's block (config + per-precision baselines) and print it for the
    # user to copy-paste into nvt.json -- replacing this GPU's entry. Same shape and
    # 2-space/sorted formatting as _save_golden, so it drops in cleanly.
    block = {"config": new_config}
    for precision in _PRECISIONS:
        block[precision] = {s: dict(measured_cells[(precision, s)],
                                    telemetry=telemetry_cells[(precision, s)])
                            for s in _SYSTEMS}
    print("\n" + "=" * 84)
    print(f"golden block for [{gpu}] -- one run, for "
          f"{_golden_gpu_dir(_BENCH_GROUP or golden_runs.MAIN_GROUP) / '<run-id>' / _GOLDEN_FILENAMES[_ensemble()]}"
          + ("" if _BENCH_GROUP else " (each group's systems in its own tree)"))
    print("(a golden is a directory of these; the CLI does not write one itself, "
          "and one run cannot be committed as a golden: CI commits goldens from its runs)")
    print("=" * 84)
    print(json.dumps(block, indent=2, sort_keys=True), flush=True)


def _cli_main() -> None:
    args = _parse_cli_args()
    if args.check_clocks_only:
        with _cli_locked_clocks() as clocks:
            clocks.wait_for_locked_clocks()
            time.sleep(3)
            telemetry = clocks.get_telemetry()
        print(f"cupo_clock_check_ok "
              f"graphics_mhz={clocks.locked.graphics_mhz} memory_mhz={clocks.locked.memory_mhz} "
              f"telemetry={json.dumps(telemetry, sort_keys=True)}", flush=True)
        return
    # --nv-integrator defaults on but applies only to the plugin engine; a --stock
    # run benches stock OpenMM, so the flag is simply a no-op there (guarded by the
    # `engine == "plugin"` check below).
    global _USE_NV_INTEGRATOR
    _USE_NV_INTEGRATOR = args.nv_integrator
    if args.pdb is None:
        _run_golden(args)
        return
    with _cli_locked_clocks() as clocks:
        _run_pdb(args, clocks.locked)


def _run_pdb(args, locked) -> None:
    """--pdb mode, with the GPU locked at `locked` (gpu_clocks.LockedClocks)."""
    cur_gr, cur_mem = locked.graphics_mhz, locked.memory_mhz
    launch = _towerplate_launch_config()
    launch_kv = " ".join(f"{k}={v}" for k, v in launch.items())
    print(f"cupo_system_locked_clocks graphics_mhz={cur_gr} memory_mhz={cur_mem} "
          f"precision={args.precision} sm_count={_sm_count()} "
          f"cuda_towerplate_direct_force[{launch_kv}]", flush=True)
    pdb = _read_pdb(args.pdb)
    system = _build_system(pdb.topology, cutoff_nm=args.cutoff_nm,
                           forcefield=args.forcefield, apply_hmr=args.hmr)
    box = pdb.topology.getPeriodicBoxVectors()
    system.setDefaultPeriodicBoxVectors(*box)
    barostat_summary = _configure_barostat(system, args)
    cutoff_summary = _configure_cutoff(system, args.cutoff_nm)

    if args.stock:
        engine = "stock"  # run the system as-is through stock OpenMM CUDA
    else:
        # Make the plugin package importable when run directly. It loads the
        # libraries that ship with it -- there is nothing to point at.
        py = str(_PLUGIN_DIR / "python")
        if py not in sys.path:
            sys.path.insert(0, py)
        import nv_openmm_plugin as nv
        _swap_in_nv_force(nv, system,
                          args.npt_box_safety if args.barostat else 1.0)
        _print_plugin_libs()
        engine = "plugin"

    temperature = args.temperature_k * unit.kelvin
    if args.nv_integrator and engine == "plugin":
        for _i, _f in enumerate(system.getForces()):
            if isinstance(_f, mm.CMMotionRemover):
                system.removeForce(_i)
                break
        integrator = nv.LangevinMiddleIntegrator(
            temperature, FRICTION_PS / unit.picosecond,
            args.timestep_fs * unit.femtoseconds, seed=args.seed)
        integrator_kind = "nv"
    else:
        integrator = mm.LangevinMiddleIntegrator(
            temperature, FRICTION_PS / unit.picosecond,
            args.timestep_fs * unit.femtoseconds)
        integrator.setRandomNumberSeed(args.seed)
        integrator_kind = "stock"
    integrator.setConstraintTolerance(1e-5)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": args.precision})
    context.setPeriodicBoxVectors(*box)
    # asNumpy: a Quantity wrapping a list of Vec3 costs ~1 ms per 1000 atoms in
    # SWIG conversion, and churns the cache the timed call is about to use.
    context.setPositions(pdb.getPositions(asNumpy=True))
    context.setVelocitiesToTemperature(temperature, args.seed)

    print(f"cupo_system_pdb={args.pdb} engine={engine} integrator={integrator_kind} "
          f"barostat={barostat_summary} "
          f"cutoff_nm={cutoff_summary} "
          f"precision={args.precision} hmr={int(args.hmr)} "
          f"timestep_fs={args.timestep_fs:g}", flush=True)

    initial = context.getState(getEnergy=True, getForces=True)
    ie = initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    iforces = initial.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    print(f"cupo_system_initial_energy_kjmol={ie:.6g} "
          f"nan_forces={np.isnan(iforces).sum()} "
          f"max_force={np.nanmax(np.linalg.norm(iforces, axis=1)):.6g}", flush=True)

    if args.warmup > 0:
        integrator.step(args.warmup)
    start = time.perf_counter()
    if args.steps > 0:
        integrator.step(args.steps)
    elapsed = time.perf_counter() - start

    final = context.getState(getEnergy=True, getForces=True)
    fe = final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    fforces = final.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    nan_forces = int(np.isnan(fforces).sum())
    max_force = float(np.nanmax(np.linalg.norm(fforces, axis=1)))
    us_per_step = elapsed / args.steps * 1e6 if args.steps > 0 else 0.0
    ns_per_day = (args.timestep_fs * 1e-6 * args.steps / elapsed * 86400
                  if args.steps > 0 and elapsed > 0.0 else 0.0)
    print(f"cupo_system_perf_steps={args.steps} elapsed_s={elapsed:.6f} "
          f"us_per_step={us_per_step:.1f} ns_per_day={ns_per_day:.2f}", flush=True)
    print(f"cupo_system_final_energy_kjmol={fe:.6g} "
          f"nan_forces={nan_forces} max_force={max_force:.6g}", flush=True)
    if nan_forces:
        raise SystemExit("NaN forces detected")


if __name__ == "__main__":
    _cli_main()

# ── Energy-only: the same evaluation without the host round trip ──────────────
#
# Reported, not gated. What these measure and why it is the interesting number
# is in _time_energy; what keeps them out of _GATED is that no repeat history
# exists for them yet, and nothing gates here until repeated runs on one commit
# have shown it can hold a threshold.

def test_plugin_energy_time(plugin_timings):
    """Plugin energy-only evaluation, reported against golden."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "plugin_energy_ms"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    fe = plugin_timings["plugin_force_and_energy_ms"]
    # The share of force+energy that was the host carrying forces back, printed
    # next to the number so the two are read together rather than looked up.
    print(f"\n  [{system_name}] {key}: {val:.4f} ms "
          f"({(1 - val / fe) * 100:.0f}% of force+energy was host-side)")
    print(f"  golden: {entry[key]:.4f} ms  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision)


def test_stock_energy_time(plugin_timings):
    """Stock OpenMM energy-only evaluation, the peer for speedup_energy."""
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    key = "stock_energy_ms"
    entry = _compared_entry(plugin_timings, system_name, precision)
    val = plugin_timings[key]
    print(f"\n  [{system_name}] {key}: {val:.4f} ms")
    print(f"  golden: {entry[key]:.4f} ms  GPU: {_gpu_name()}")
    _check_regression(key, val, entry[key], system_name, precision)


def test_plugin_speedup_energy(plugin_timings):
    """Plugin energy-only speedup vs stock, reported against golden.

    Worth watching next to speedup_force_and_energy: both are the same ratio of
    the same two contexts, and the only difference is whether OpenMM's per-atom
    force round trip is inside the measurement. Force+energy reports 1.08-1.16x
    where the step ratio is ~1.7x, because that shared host cost is added to
    both sides and drags the ratio toward 1. This one should not be diluted
    that way.
    """
    system_name = plugin_timings["system_name"]
    precision = plugin_timings["precision"]
    entry = _compared_entry(plugin_timings, system_name, precision)
    speedup = plugin_timings["speedup_energy"]
    stock = plugin_timings["stock_energy_ms"]
    plugin = plugin_timings["plugin_energy_ms"]
    print(f"\n  [{system_name}] stock: {stock:.4f} ms, "
          f"plugin: {plugin:.4f} ms, "
          f"speedup: {speedup:.2f}x "
          f"(golden: {entry['speedup_energy']:.2f}x, "
          f"force+energy says {plugin_timings['speedup_force_and_energy']:.2f}x, "
          f"step says {plugin_timings['speedup_step']:.2f}x)")
    _check_regression("speedup_energy", speedup, entry["speedup_energy"],
                      system_name, precision, higher_is_worse=False)
