# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom numerical accumulation policy, independent of expression parameters."""
import contextlib

import numpy as np
import openmm as mm
from openmm import unit
import pytest
import nv_openmm_plugin as nv


@contextlib.contextmanager
def pair_context(force):
    system = mm.System()
    for _ in range(2):
        system.addParticle(39.9)
        force.addParticle([])
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": "mixed"})
    ctx.setPositions([[.25, .25, .25], [.75, .25, .25]])
    try:
        yield ctx
    finally:
        del ctx, integrator


def snapshot(ctx, energy=True):
    state = ctx.getState(getEnergy=energy, getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    return forces, state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) if energy else None


@pytest.mark.parametrize("excluded", [False, True])
def test_default_accumulation_preserves_expression_units(plugin_dir, excluded):
    force = nv.NonbondedForce("1.04*r")
    if excluded:
        force.addExclusion(0, 1)
    with pair_context(force) as ctx:
        # Physical force and energy at r=0.5 nm. The tolerance allows half
        # the internal force increment (~6.625e-5 kJ/mol/nm) plus FP32 error.
        magnitude = 0. if excluded else 1.04
        expected = [[magnitude, 0., 0.], [-magnitude, 0., 0.]]
        for energy in (False, True):
            actual, potential = snapshot(ctx, energy)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=3.4e-5)
            if energy:
                assert potential == pytest.approx(0. if excluded else .52, abs=1e-7)
