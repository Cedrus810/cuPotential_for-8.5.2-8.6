# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-step replay must preserve trajectories and execute actual captured steps."""
import os
from unittest.mock import patch

import numpy as np
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv
import pytest

from test_fused_integrator import system_for


@pytest.fixture(autouse=True)
def whole_environment(monkeypatch):
    monkeypatch.delenv("CUPO_PLUGIN_BONDED_SCHEDULE", raising=False)
    for key, value in {"CUPO_COMPACT_AUTOTUNE": "0", "CUPO_PLUGIN_CUDA_GRAPH": "1",
                       "CUPO_PLUGIN_INTEGRATOR_GRAPH": "0",
                       "CUPO_PLUGIN_WHOLE_STEP_GRAPH": "0"}.items():
        monkeypatch.setenv(key, value)


def nonbonded_system(method, box_safety=1., virtual=False, triclinic=False):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4.8, 0, 0), mm.Vec3(0, 4.8, 0), mm.Vec3(0, 0, 4.8))
    if triclinic:
        system.setDefaultPeriodicBoxVectors(mm.Vec3(4.8, 0, 0), mm.Vec3(.15, 4.8, 0), mm.Vec3(.1, .2, 4.8))
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.PME)
    force.setCutoffDistance(.9)
    xyz = np.array([(x, y, z) for x in range(8) for y in range(8) for z in range(8)]) * .55 + .15
    bonds = mm.HarmonicBondForce()
    bonds.setForceGroup(1)
    for i in range(len(xyz)):
        system.addParticle(20.)
        force.addParticle(.05 if i % 2 else -.05, .25, .05)
    for i in range(0, len(xyz), 8):
        force.addException(i, i+1, 0., .25, 0.)
        bonds.addBond(i, i+1, .55, 10.)
    if virtual:
        site = system.addParticle(0.)
        system.setVirtualSite(site, mm.TwoParticleAverageSite(0, 1, .5, .5))
        force.addParticle(.02, .2, .01)
        for parent in (0, 1):
            force.addException(parent, site, 0., .2, 0.)
        xyz = np.vstack((xyz, .5*(xyz[0]+xyz[1])))
    system.addForce(nv.NonbondedForce(force, box_safety, method))
    system.addForce(bonds)
    return system, xyz, np.random.default_rng(48).normal(scale=.01, size=xyz.shape)


def trajectory(case, precision, whole, fused=True, graph=True, mutate=False, unsupported=False,
               mode="auto", schedule=None):
    if case in ('prolate', 'ewald', 'triclinic'):
        method = nv.NonbondedForce.PME if case == 'ewald' else nv.NonbondedForce.Prolate
        system, xyz, vel = nonbonded_system(method, triclinic=case == 'triclinic')
    else:
        system, xyz, vel = system_for(case)
    if unsupported:
        system.addForce(mm.CMMotionRemover(7))
    integ = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
    integ.setRandomNumberSeed(58392)
    with patch.dict(os.environ, {'CUPO_PLUGIN_CUDA_GRAPH': str(int(graph)),
                                'CUPO_PLUGIN_FUSED_INTEGRATOR': str(int(fused)),
                                'CUPO_PLUGIN_INTEGRATOR_GRAPH': '1' if mode == 'graph' else '0'}):
        for name, value in {'CUPO_PLUGIN_WHOLE_STEP_GRAPH': None if whole is None else str(int(whole)),
                            'CUPO_PLUGIN_BONDED_SCHEDULE': schedule}.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision': precision})
    ctx.setPositions(xyz)
    ctx.setVelocities(vel)
    states = []
    for i, n in enumerate((60, 260, 13, 1, 10)):
        if mutate and i == 2:
            integ.setTemperature(275.)
            integ.setFriction(2.)
            integ.setStepSize(.0015)
            integ.setConstraintTolerance(5e-6)
            integ.setComFrequency(5)
            if case in ('prolate', 'ewald', 'triclinic'):
                bonds = system.getForce(1)
                a, b, length, k = bonds.getBondParameters(0)
                bonds.setBondParameters(0, a, b, length, 2*k)
                bonds.updateParametersInContext(ctx)
                ctx.setPeriodicBoxVectors(mm.Vec3(4.9, 0, 0), mm.Vec3(0, 4.9, 0), mm.Vec3(0, 0, 4.9))
            else:
                ctx.setParameter('k', 4.)
            # Positions changed between calls must be read by the next graph.
            pos = ctx.getState(getPositions=True).getPositions(asNumpy=True)
            ctx.setPositions(pos.value_in_unit(unit.nanometer) + np.array([.002, -.001, .003]))
        if mutate and i == 3 and case in ('prolate', 'ewald', 'triclinic'):
            integ.setIntegrationForceGroups(1)  # exclude the bonded group
        integ.step(n)
        state = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
        states.append((np.array(state.getPositions(asNumpy=True)._value),
                       np.array(state.getVelocities(asNumpy=True)._value),
                       np.array(state.getForces(asNumpy=True)._value),
                       state.getPotentialEnergy()._value, state.getKineticEnergy()._value,
                       state.getTime()._value, state.getStepCount()))
    counters = integ.getWholeStepGraphCaptureCount(), integ.getWholeStepGraphReplayCount()
    del ctx, integ
    return states, counters


@pytest.mark.parametrize('precision', ['single', 'mixed'])
@pytest.mark.parametrize('graph,unsupported', [(True, False), (False, False), (True, True)])
def test_default_whole_step_after_sort(precision, graph, unsupported):
    ref, _ = trajectory('prolate', precision, False, graph=graph, unsupported=unsupported,
                        mutate=True, schedule="overlap")
    explicit, explicit_counts = trajectory('prolate', precision, True, graph=graph,
                                          unsupported=unsupported, mutate=True, schedule="after-sort")
    got, counts = trajectory('prolate', precision, None, graph=graph,
                             unsupported=unsupported, mutate=True, schedule=None)
    assert counts == explicit_counts
    if graph and not unsupported:
        assert counts[1] > 250
    else:
        assert counts == (0, 0)
    for expected, enabled, actual in zip(ref, explicit, got):
        for a, b, c in zip(expected, enabled, actual):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, c)


@pytest.mark.parametrize('precision', ['single', 'mixed'])
@pytest.mark.parametrize('case', ['combined', 'water_virtual', 'prolate', 'ewald'])
def test_whole_step_matches_component_graphs(precision, case):
    ref, _ = trajectory(case, precision, False, mutate=True)
    got, counters = trajectory(case, precision, True, mutate=True)
    assert counters[0] >= 2
    assert counters[1] > 250
    for expected, actual in zip(ref, got):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('precision', ['single', 'mixed'])
def test_whole_step_separate_integrator(precision):
    ref, _ = trajectory('combined', precision, False, fused=False)
    got, counters = trajectory('combined', precision, True, fused=False)
    assert counters[1] > 250
    for expected, actual in zip(ref, got):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('graph,unsupported', [(False, False), (True, True)])
def test_whole_step_falls_back(graph, unsupported):
    ref, _ = trajectory('combined', 'mixed', False, graph=graph, unsupported=unsupported)
    got, counters = trajectory('combined', 'mixed', True, graph=graph, unsupported=unsupported)
    assert counters == (0, 0)
    for expected, actual in zip(ref, got):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('precision', ['single', 'mixed'])
def test_component_graph_refreshes_constraint_tolerance(precision):
    # The ordinary integrator graph must also pick up this by-value argument;
    # otherwise it disagrees with both eager execution and a fresh whole graph.
    eager, _ = trajectory('combined', precision, False, graph=False, mutate=True)
    captured, _ = trajectory('combined', precision, False, graph=True, mutate=True, mode='graph')
    for expected, actual in zip(eager, captured):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('barostat', ['stock', 'plugin'])
def test_whole_step_idle_barostat(barostat):
    system, xyz, vel = nonbonded_system(nv.NonbondedForce.Prolate, box_safety=1.2)
    cls = mm.MonteCarloBarostat if barostat == 'stock' else nv.MonteCarloBarostat
    system.addForce(cls(1., 300., 10000))
    integ = nv.LangevinMiddleIntegrator(300., 1., .001, 0)
    with patch.dict(os.environ, {'CUPO_PLUGIN_WHOLE_STEP_GRAPH': '1'}):
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'mixed'})
    ctx.setPositions(xyz)
    ctx.setVelocities(vel)
    integ.step(70)
    assert integ.getWholeStepGraphCaptureCount() == 1
    assert integ.getWholeStepGraphReplayCount() == 15
    assert ctx.getStepCount() == 70
    del ctx, integ


@pytest.mark.parametrize("mode", ["auto", "graph"])
@pytest.mark.parametrize("schedule", ["overlap", "after-sort"])
def test_whole_step_composes_with_stream_modes(mode, schedule):
    ref, counts = trajectory("prolate", "single", False, mutate=True,
                             mode=mode, schedule=schedule)
    got, whole_counts = trajectory("prolate", "single", True, mutate=True,
                                   mode=mode, schedule=schedule)
    assert counts == (0, 0)
    assert whole_counts[1] > 250
    for a, b in zip(ref, got):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize("case", ["triclinic"])
def test_whole_step_other_cells_and_electrostatics(precision, case):
    ref, _ = trajectory(case, precision, False, mutate=True, schedule="after-sort")
    got, counts = trajectory(case, precision, True, mutate=True, schedule="after-sort")
    assert counts[1] > 250
    for a, b in zip(ref, got):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_whole_step_late_force_group_and_reinitialize(monkeypatch):
    snapshots = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("CUPO_PLUGIN_WHOLE_STEP_GRAPH", enabled)
        system, xyz, vel = nonbonded_system(nv.NonbondedForce.Prolate)
        integ = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
        integ.setRandomNumberSeed(498)
        integ.setIntegrationForceGroups(2)  # Only bonded; nonbonded remains lazy.
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"), {"Precision": "single"})
        ctx.setPositions(xyz); ctx.setVelocities(vel)
        integ.step(70)
        integ.setIntegrationForceGroups(-1)
        integ.step(80)  # First inclusion of nonbonded after whole-step warmup.
        if enabled == "1":
            assert integ.getWholeStepGraphReplayCount() > 70
        ctx.reinitialize(preserveState=True)
        assert integ.getWholeStepGraphCaptureCount() == 0
        integ.step(80)
        if enabled == "1":
            assert integ.getWholeStepGraphReplayCount() > 0
        state = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
        snapshots.append([np.array(state.getPositions(asNumpy=True)._value),
                          np.array(state.getVelocities(asNumpy=True)._value),
                          np.array(state.getForces(asNumpy=True)._value),
                          state.getPotentialEnergy()._value, state.getKineticEnergy()._value])
        del ctx, integ
    for a, b in zip(*snapshots):
        np.testing.assert_array_equal(a, b)


def test_whole_step_invalid_setting(monkeypatch):
    monkeypatch.setenv("CUPO_PLUGIN_WHOLE_STEP_GRAPH", "typo")
    system, _, _ = system_for("free")
    integ = nv.LangevinMiddleIntegrator(300., 1., .001, 0)
    with pytest.raises(mm.OpenMMException, match="CUPO_PLUGIN_WHOLE_STEP_GRAPH"):
        mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"), {"Precision": "single"})


@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_custom_nonbonded_retains_host_updates(monkeypatch, precision):
    from test_nonbonded_execution import XYZ, _box, _custom_source

    snapshots = []
    for enabled in ("0", "1"):
        monkeypatch.setenv("CUPO_PLUGIN_WHOLE_STEP_GRAPH", enabled)
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(*_box(4.))
        for _ in XYZ:
            system.addParticle(39.9)
        force = nv.NonbondedForce(_custom_source(), 1.25)
        system.addForce(force)
        integ = nv.LangevinMiddleIntegrator(0., 1., .001, 0, seed=589)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision})
        ctx.setPositions(XYZ)
        ctx.setVelocities(np.zeros_like(XYZ))
        integ.step(70)
        ctx.setParameter("lambda", .75)
        ctx.setParameter("scale", 1.3)
        values = list(force.getParticleParameters(5))
        values[0] += .1
        force.setParticleParameters(5, values)
        force.updateParametersInContext(ctx)
        integ.step(90)
        assert integ.getWholeStepGraphCaptureCount() == 0
        assert integ.getWholeStepGraphReplayCount() == 0
        state = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
        snapshots.append((state.getPositions(asNumpy=True)._value,
                          state.getVelocities(asNumpy=True)._value,
                          state.getForces(asNumpy=True)._value,
                          state.getPotentialEnergy()._value))
        del ctx, integ
    for a, b in zip(*snapshots):
        np.testing.assert_array_equal(a, b)
