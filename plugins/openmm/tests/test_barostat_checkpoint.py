# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPT checkpoints follow OpenMM semantics for stock and plugin components."""

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

from test_integrator_checkpoint import _context, _environment, _snapshot


_WARMUP = 83  # Checkpoint between volume moves, after adaptation has started.


def _system(seed, barostat_kind="plugin"):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(.1, 3.2, 0),
                                       mm.Vec3(.05, .15, 3.2))
    definition = mm.NonbondedForce()
    definition.setNonbondedMethod(mm.NonbondedForce.PME)
    definition.setCutoffDistance(.9)
    xyz = np.array([(x, y, z) for x in range(4) for y in range(4) for z in range(4)])*.7+.2
    for i in range(len(xyz)):
        system.addParticle(20.)
        definition.addParticle(.05 if i % 2 else -.05, .25, .05)
    system.addForce(nv.NonbondedForce(definition, 1.5))
    barostat_class = nv.MonteCarloBarostat if barostat_kind == "plugin" else mm.MonteCarloBarostat
    barostat = barostat_class(80., 300., 5)
    barostat.setRandomNumberSeed(seed)
    system.addForce(barostat)
    return system, xyz, np.random.default_rng(49).normal(scale=.01, size=xyz.shape)


def _npt_context(system, xyz, velocities, seed, precision="mixed", integrator_kind="plugin"):
    if integrator_kind == "plugin":
        return _context(system, xyz, velocities, seed, precision)
    integrator = mm.LangevinMiddleIntegrator(300., 1., .001)
    integrator.setConstraintTolerance(1e-6)
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision, "DeterministicForces": "true"})
    context.setPositions(xyz)
    context.setVelocities(velocities)
    return context, integrator


def _observation(context):
    snapshot = _snapshot(context)
    box = np.asarray(context.getState().getPeriodicBoxVectors(asNumpy=True)
                     .value_in_unit(unit.nanometer)).copy()
    return snapshot+(box,)


def _advance_npt(context, integrator):
    snapshots = []
    for steps in (2, 11, 71):
        integrator.step(steps)
        snapshots.append(_observation(context))
    return snapshots


def _assert_restored(reference, actual):
    # Positions, velocities, time, step count and box are checkpointed exactly.
    # Forces and energies are recomputed by the target Context.
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        assert np.isfinite(observed).all()
        if index in (0, 1, 5, 6, 7):
            np.testing.assert_array_equal(observed, expected)
        else:
            np.testing.assert_allclose(observed, expected, rtol=5e-6, atol=2e-6)


def _assert_continues(context, integrator):
    initial = _observation(context)
    snapshots = _advance_npt(context, integrator)
    step_size = integrator.getStepSize().value_in_unit(unit.picosecond)
    for steps, snapshot in zip((2, 13, 84), snapshots):
        for value in snapshot:
            assert np.isfinite(value).all()
        assert snapshot[6] == initial[6]+steps
        assert snapshot[5] == pytest.approx(initial[5]+steps*step_size, abs=1e-12)
        assert np.linalg.det(snapshot[7]) > 0
    # The restored run must actually execute accepted volume moves.
    assert not np.array_equal(initial[7], snapshots[-1][7])


@pytest.mark.parametrize("integrator_kind", ["plugin", "stock"])
@pytest.mark.parametrize("barostat_kind", ["plugin", "stock"])
@pytest.mark.parametrize("path", ["eager", "component", "whole-batch"])
@pytest.mark.parametrize("seed", [719, 0])
@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_npt_checkpoint_restores_state_and_resumes(
        monkeypatch, integrator_kind, barostat_kind, path, seed, precision):
    _environment(monkeypatch, path)
    system, xyz, velocities = _system(seed, barostat_kind)
    barostat = system.getForce(1)
    context, integrator = _npt_context(system, xyz, velocities, seed, precision, integrator_kind)
    try:
        integrator.step(_WARMUP)
        context.setParameter(barostat.Pressure(), 77.)
        context.setParameter(barostat.Temperature(), 305.)
        expected = _observation(context)
        checkpoint = context.createCheckpoint()
        _assert_continues(context, integrator)
        context.setParameter(barostat.Pressure(), 45.)
        context.setParameter(barostat.Temperature(), 290.)

        context.loadCheckpoint(checkpoint)
        _assert_restored(expected, _observation(context))
        assert context.getParameter(barostat.Pressure()) == 77.
        assert context.getParameter(barostat.Temperature()) == 305.
        _assert_continues(context, integrator)
        for used in (False, True):
            other_system, other_xyz, other_velocities = _system(913, barostat_kind)
            other, other_integrator = _npt_context(
                other_system, other_xyz, other_velocities, 817, precision, integrator_kind)
            try:
                if used:
                    other_integrator.step(107)
                other.loadCheckpoint(checkpoint)
                _assert_restored(expected, _observation(other))
                assert other.getParameter(barostat.Pressure()) == 77.
                assert other.getParameter(barostat.Temperature()) == 305.
                # Barostat phase/adaptation is not checkpointed in OpenMM.
                # Validate resumed dynamics without demanding trajectory replay.
                _assert_continues(other, other_integrator)
            finally:
                del other, other_integrator
    finally:
        del context, integrator
