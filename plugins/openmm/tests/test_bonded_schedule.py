# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic scheduling follows cutoff expressions and preserves MD results."""
import numpy as np
import openmm as mm
import nv_openmm_plugin as nv
import pytest

from test_nonbonded_execution import XYZ, _box, _custom_source, _standard_source


def make_system(case):
    source = _standard_source()
    if case == "offsets":
        source.addGlobalParameter("lambda", .35)
        source.addParticleParameterOffset("lambda", 4, .01, 0., 0.)
    force = nv.NonbondedForce(_custom_source() if case == "custom" else source, 1.25)
    if case == "expression":
        force.setCutoffInteraction(_custom_source())
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*_box(4.))
    for _ in XYZ:
        system.addParticle(39.9)
    system.addForce(force)
    bonded = mm.HarmonicBondForce()
    bonded.addBond(0, 1, float(np.linalg.norm(XYZ[0]-XYZ[1])), 3.)
    system.addForce(bonded)
    return system, force


def snapshot(ctx):
    state = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
    return [np.array(state.getPositions(asNumpy=True)._value),
            np.array(state.getVelocities(asNumpy=True)._value),
            np.array(state.getForces(asNumpy=True)._value),
            state.getPotentialEnergy()._value, state.getKineticEnergy()._value]


@pytest.fixture(autouse=True)
def graph_settings(monkeypatch):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_WHOLE_STEP_GRAPH", "1")
    monkeypatch.delenv("CUPO_PLUGIN_BONDED_SCHEDULE", raising=False)


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize("case", ["standard", "offsets", "expression", "custom"])
def test_default_and_overrides_preserve_trajectory(monkeypatch, capfd, precision, case):
    states = []
    for requested in [None, "auto", "overlap", "after-sort"]:
        if requested is None:
            monkeypatch.delenv("CUPO_PLUGIN_BONDED_SCHEDULE", raising=False)
        else:
            monkeypatch.setenv("CUPO_PLUGIN_BONDED_SCHEDULE", requested)
        system, force = make_system(case)
        expression = case in ("expression", "custom")
        assert force.hasCutoffExpression() == expression
        resolved = requested if requested in ("overlap", "after-sort") else \
            ("overlap" if expression else "after-sort")
        integ = nv.LangevinMiddleIntegrator(0., 1., .001, 0, seed=589)
        capfd.readouterr()
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision})
        ctx.setPositions(XYZ)
        ctx.setVelocities(np.zeros_like(XYZ))
        integ.step(75)
        trace = capfd.readouterr().err
        assert f"bonded schedule requested={requested or 'auto'} resolved={resolved} cutoff_expression={int(expression)}" in trace
        if expression:
            assert integ.getWholeStepGraphReplayCount() == 0
            assert "graph capture #" in trace
        else:
            assert integ.getWholeStepGraphReplayCount() > 0
        # Changing live parameters must still reach the device on either schedule.
        if case != "standard":
            ctx.setParameter("lambda", .65)
        integ.step(20)
        states.append(snapshot(ctx))
        del ctx, integ
    for actual in states[1:]:
        for expected, observed in zip(states[0], actual):
            np.testing.assert_array_equal(expected, observed)


def test_auto_schedule_refreshes_after_expression_reinitialize(monkeypatch, capfd):
    system, force = make_system("standard")
    integ = nv.LangevinMiddleIntegrator(0., 1., .001, 0, seed=589)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    ctx.setPositions(XYZ)
    ctx.setVelocities(np.zeros_like(XYZ))
    integ.step(75)
    assert "resolved=after-sort cutoff_expression=0" in capfd.readouterr().err
    assert integ.getWholeStepGraphReplayCount() > 0
    force.setCutoffInteraction(_custom_source())
    ctx.reinitialize(preserveState=True)
    integ.step(75)
    assert "resolved=overlap cutoff_expression=1" in capfd.readouterr().err
    assert integ.getWholeStepGraphReplayCount() == 0
    assert all(np.isfinite(value).all() for value in snapshot(ctx))
    del ctx, integ
