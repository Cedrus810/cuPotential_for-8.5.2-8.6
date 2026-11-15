# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUPO_OPENMM_STOCK_FORCE_DETERMINISTIC_FORCES=1: every Context computes the
same forces on every run.

OpenMM's CUDA PME spreads charges onto its grid with floating-point atomic adds.
The order in which those adds arrive is not fixed, and floating-point addition
is not associative, so the forces differ in the last bits from run to run. The
platform property DeterministicForces=true makes the spreading use fixed-point
integers instead, where order does not matter. Its default is false. The
plugin's own NonbondedForce spreads in fixed point regardless of the property.

Set to 1, every Context gets DeterministicForces=true on every platform that has
the property (CUDA and CPU). A Context created with DeterministicForces=false is
overridden too: the variable is for making a run reproducible without editing
the script. Unset, empty or 0, nothing is changed. Any other value is an error.

Each change is reported on stderr as an INFO line that names the variable, once
per process: once for a Context that did not set the property, and once for a
Context whose DeterministicForces=false was overridden.

If only one of this variable and CUPO_OPENMM_PLUGIN_RNG_SEED is set, a Context
may not be reproducible. Then a warning goes to stderr. It names the variable to
add, and is printed once per run.

- Only CUPO_OPENMM_PLUGIN_RNG_SEED set: warns about a Context with stock PME
  and DeterministicForces=false. Stock PME means OpenMM's own NonbondedForce with
  PME or LJPME, or a ConstantPotentialForce, at a precision other than double.
  Those are the only forces the property changes. A Context whose only PME is
  the plugin's NonbondedForce is deterministic, and is not warned about.
- Only this variable set: warns about a Context with an integrator or force
  whose seed is 0.

A script that sets those itself, seeds or DeterministicForces=true, gets no
warning.
"""

from __future__ import annotations

import functools
import os
import sys

ENV = "CUPO_OPENMM_STOCK_FORCE_DETERMINISTIC_FORCES"
_PROPERTY = "DeterministicForces"

# Marks a method this module has replaced, so installing twice wraps once.
_WRAPPED = "_cupo_deterministic_forces_wrapped"


def from_env() -> bool:
    """Whether CUPO_OPENMM_STOCK_FORCE_DETERMINISTIC_FORCES asks for it."""
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{ENV} must be 1, 0 or unset, got {raw!r}")


def install(force: bool, seed: int | None, seed_env: str) -> None:
    """Patch openmm.Context for whichever of the two variables are set.

    `force` is this module's variable; `seed` is CUPO_OPENMM_PLUGIN_RNG_SEED's
    value (None when unset), named by `seed_env` in the warning. With neither
    set, nothing is patched.
    """
    import openmm as mm

    # Only one of the two is set: a run that looks reproducible may not be.
    check = (seed is not None) != force
    if not force and not check:
        return

    init = mm.Context.__dict__["__init__"]
    if getattr(init, _WRAPPED, False):
        return

    if force:
        # Covers Contexts created without properties, or without a platform,
        # where OpenMM picks one. Which platform a Context gets is only known
        # once it exists, so the default is set on all of them.
        for i in range(mm.Platform.getNumPlatforms()):
            platform = mm.Platform.getPlatform(i)
            if _PROPERTY in platform.getPropertyNames():
                platform.setPropertyDefaultValue(_PROPERTY, "true")

    if seed is not None:
        have, missing = seed_env, f"{ENV}=1"
        reason = f"has stock PME and {_PROPERTY}=false"
        not_reproducible = lambda context: (_has_stock_pme(context)
                                            and not _forces_deterministic(context))
    else:
        have, missing = ENV, f"{seed_env}=<n>"
        reason = "has an integrator or force with random number seed 0"
        not_reproducible = _has_unseeded
    reported = set()

    def report_once(key, line):
        if key not in reported:
            reported.add(key)
            print(line, file=sys.stderr, flush=True)

    # SWIG takes (system, integrator[, platform[, properties]]), or (other).
    @functools.wraps(init)
    def context_init(self, *args):
        changed = None
        if force and len(args) >= 3 and _PROPERTY in args[2].getPropertyNames():
            given = dict(args[3]) if len(args) == 4 else {}
            if _PROPERTY not in given:
                changed = "default"         # the platform default, set above
            elif given[_PROPERTY].lower() != "true":
                changed = "explicit"
                args = (*args[:3], {**given, _PROPERTY: "true"})
        elif force and len(args) == 2:
            changed = "default"
        init(self, *args)
        platform_has_it = _PROPERTY in self.getPlatform().getPropertyNames()
        if changed is not None and platform_has_it:
            if changed == "explicit":
                report_once(changed, f"INFO: a Context was created with "
                            f"{_PROPERTY}=false; it was changed to true from env "
                            f"variable {ENV}.")
            else:
                report_once(changed, f"INFO: {_PROPERTY} was set to true from env "
                            f"variable {ENV}.")
        if check and not_reproducible(self):
            report_once("warning",
                        f"WARNING: {have} is set, but a Context {reason}, so this "
                        f"run can differ from the next one. Set {missing} too.")

    setattr(context_init, _WRAPPED, True)
    mm.Context.__init__ = context_init


def _has_stock_pme(context) -> bool:
    """Whether DeterministicForces changes this Context's forces.

    In OpenMM 8.4 the property is read only by stock NonbondedForce under PME
    or LJPME and by ConstantPotentialForce, and on CUDA Precision=double makes
    their charge spreading fixed point already. The plugin's NonbondedForce is
    a different class, so `type(...) is` leaves it out; it spreads in fixed
    point on its own.
    """
    import openmm as mm

    platform = context.getPlatform()
    if ("Precision" in platform.getPropertyNames()
            and platform.getPropertyValue(context, "Precision") == "double"):
        return False
    pme = (mm.NonbondedForce.PME, mm.NonbondedForce.LJPME)
    system = context.getSystem()
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if type(f) is mm.NonbondedForce and f.getNonbondedMethod() in pme:
            return True
        if type(f) is mm.ConstantPotentialForce:
            return True
    return False


def _forces_deterministic(context) -> bool:
    """False only for a platform that has the property and has it off. The
    Reference platform has no such property: it is single-threaded double
    precision, and the same on every run."""
    platform = context.getPlatform()
    if _PROPERTY not in platform.getPropertyNames():
        return True
    return platform.getPropertyValue(context, _PROPERTY).lower() == "true"


def _has_unseeded(context) -> bool:
    """Whether the integrator or a force draws from a seed of 0, OpenMM's value
    for "a new random seed per Context"."""
    system = context.getSystem()
    objects = [context.getIntegrator()]
    objects += [system.getForce(i) for i in range(system.getNumForces())]
    return any(hasattr(o, "getRandomNumberSeed") and o.getRandomNumberSeed() == 0
               for o in objects)
