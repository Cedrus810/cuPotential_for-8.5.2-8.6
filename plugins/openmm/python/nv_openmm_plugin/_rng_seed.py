# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUPO_OPENMM_PLUGIN_RNG_SEED: one environment variable that seeds every random
number a run draws.

Unset, nothing here runs and OpenMM behaves as it always does: a seed of 0 on an
integrator or barostat means "pick a fresh one per Context", and
Context.setVelocitiesToTemperature without a seed picks one too. Neither is ever
reported, so a run that fails cannot be run again with the same noise.

Set to an integer, every OpenMM class with a setRandomNumberSeed method -- stock
integrators, thermostats and barostats, and this plugin's own -- gets that seed
when it is constructed, and setVelocitiesToTemperature gets it when the caller
passes none. A script with no seeds in it becomes reproducible without being
edited.

An explicit seed in the script still wins. Ours is applied in the constructor, so
a later setRandomNumberSeed replaces it, and a seed passed to
setVelocitiesToTemperature is used as given.

Every seed this sets is reported on stderr as an INFO line that names the
variable, once per process for each class it was set on.

Every replica of a replica-exchange run started in one environment gets the same
noise while this is set. Do not set it for runs whose statistics depend on the
trajectories being independent.

What it does not cover: forces that are non-deterministic without any random
number. Stock OpenMM's PME on CUDA spreads charges with floating-point atomic
adds, so its forces differ in the last bits from run to run unless the Context
has DeterministicForces=true. The plugin's NonbondedForce spreads charges in
fixed point, so its forces are the same on every run without that property.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys

ENV = "CUPO_OPENMM_PLUGIN_RNG_SEED"

# Marks a method this module has replaced, so installing twice wraps once.
_WRAPPED = "_cupo_rng_seed_wrapped"


def from_env() -> int | None:
    """The seed CUPO_OPENMM_PLUGIN_RNG_SEED asks for, or None when it is unset or
    empty."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return None
    try:
        seed = int(raw)
    except ValueError:
        raise ValueError(f"{ENV} must be an integer, got {raw!r}") from None
    # 0 is OpenMM's "choose a fresh seed per Context". Accepting it would set the
    # variable and leave the run exactly as random as without it.
    if seed == 0:
        raise ValueError(
            f"{ENV}=0 would not seed anything: 0 is OpenMM's value for a random "
            "seed. Pick a nonzero integer, or unset the variable.")
    return seed


def install(seed: int, modules) -> None:
    """Seed every class in `modules` that has setRandomNumberSeed, and the
    velocities Context.setVelocitiesToTemperature draws.

    Classes are found by that method rather than listed by name, so an
    integrator added in a later OpenMM is covered too.

    Each use of the seed is reported on stderr, once per process for each
    class, as an INFO line naming the variable. An object's seed is reported
    when a Context is created with it and the seed is still ours, so an object
    the script seeds itself afterwards is not reported.
    """
    import openmm as mm

    # The C++ address of each object whose seed is still ours -> (its name for
    # the report, its class name). By address because the System hands back a
    # new Python wrapper for each force, and for a plugin force that wrapper is
    # a plain Force.
    seeded = {}
    reported = set()

    def report(what):
        if what not in reported:
            reported.add(what)
            print(f"INFO: {what} was set to {seed} from env variable {ENV}.",
                  file=sys.stderr, flush=True)

    for module in modules:
        for cls in list(vars(module).values()):
            if inspect.isclass(cls) and hasattr(cls, "setRandomNumberSeed"):
                _seed_after_init(cls, seed, seeded)

    context_init = mm.Context.__dict__["__init__"]
    if getattr(context_init, _WRAPPED, False):
        return

    # SWIG takes (system, integrator[, platform[, properties]]), or (other).
    @functools.wraps(context_init)
    def reporting_context_init(self, *args):
        context_init(self, *args)
        if len(args) < 2:
            return
        system, integrator = args[0], args[1]
        objects = [integrator]
        objects += [system.getForce(i) for i in range(system.getNumForces())]
        for o in objects:
            entry = seeded.get(int(o.this))
            # The type check guards against a freed object's address being
            # reused by an unrelated one. A plugin force comes back as Force.
            if entry is not None and type(o).__name__ in (entry[1], "Force"):
                report(f"the random number seed of {entry[0]}")

    setattr(reporting_context_init, _WRAPPED, True)
    mm.Context.__init__ = reporting_context_init

    set_velocities = mm.Context.__dict__.get("setVelocitiesToTemperature")
    if set_velocities is None or getattr(set_velocities, _WRAPPED, False):
        return

    # SWIG takes (temperature, randomSeed) positionally. Note the convention is
    # the opposite of setRandomNumberSeed's: 0 here is a real seed, not "random".
    @functools.wraps(set_velocities)
    def seeded_set_velocities(self, *args, **kwargs):
        if len(args) < 2 and "randomSeed" not in kwargs:
            args = (*args, seed)
            report("the random number seed of Context.setVelocitiesToTemperature")
        return set_velocities(self, *args, **kwargs)

    setattr(seeded_set_velocities, _WRAPPED, True)
    mm.Context.setVelocitiesToTemperature = seeded_set_velocities


def _seed_after_init(cls, seed: int, seeded: dict) -> None:
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, _WRAPPED, False):
        return
    package = "openmm" if cls.__module__.startswith("openmm") else "nv_openmm_plugin"
    name = f"{package}.{cls.__name__}"

    @functools.wraps(init)
    def seeded_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        cls.setRandomNumberSeed(self, seed)
        seeded[int(self.this)] = (name, cls.__name__)

    setattr(seeded_init, _WRAPPED, True)
    cls.__init__ = seeded_init

    # Only where the class defines the setter; a subclass inherits the wrapper.
    set_seed = cls.__dict__.get("setRandomNumberSeed")
    if set_seed is None or getattr(set_seed, _WRAPPED, False):
        return

    # A seed the script sets itself replaces ours, and is not reported.
    @functools.wraps(set_seed)
    def own_seed(self, *args, **kwargs):
        seeded.pop(int(self.this), None)
        return set_seed(self, *args, **kwargs)

    setattr(own_seed, _WRAPPED, True)
    cls.setRandomNumberSeed = own_seed
