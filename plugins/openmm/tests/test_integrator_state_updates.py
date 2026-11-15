# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live integrator settings and cached force bindings survive state changes."""
import numpy as np
import openmm as mm
import nv_openmm_plugin as nv
import pytest
from test_fused_integrator import system_for


@pytest.mark.parametrize('precision', ['single', 'mixed'])
def test_component_graph_updates_constraint_tolerance(monkeypatch, precision):
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_GRAPH', '0')
    monkeypatch.setenv('CUPO_PLUGIN_FUSED_INTEGRATOR', '1')
    results = []
    for graph in ['0', '1']:
        monkeypatch.setenv('CUPO_PLUGIN_CUDA_GRAPH', graph)
        monkeypatch.setenv('CUPO_PLUGIN_INTEGRATOR_GRAPH', '1')
        system, xyz, vel = system_for('combined')
        integrator = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
        integrator.setRandomNumberSeed(771)
        integrator.setConstraintTolerance(1e-3)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName('CUDA'),
                             {'Precision': precision})
        context.setPositions(xyz)
        context.setVelocities(vel)
        integrator.step(90)
        integrator.setConstraintTolerance(1e-6)
        integrator.step(80)
        state = context.getState(getPositions=True, getVelocities=True)
        results.append((state.getPositions(asNumpy=True)._value.copy(),
                        state.getVelocities(asNumpy=True)._value.copy()))
        del context, integrator
    for expected, actual in zip(*results):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize('graph', ['0', '1'])
def test_live_com_frequency(monkeypatch, graph):
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_GRAPH', '0')
    monkeypatch.setenv('CUPO_PLUGIN_CUDA_GRAPH', graph)
    monkeypatch.setenv('CUPO_PLUGIN_INTEGRATOR_GRAPH', '1')
    system = mm.System()
    for _ in range(37):
        system.addParticle(12.)
    integrator = nv.LangevinMiddleIntegrator(0., 0., .001, 0)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('CUDA'),
                         {'Precision': 'mixed'})
    context.setPositions(np.zeros((37, 3)))
    context.setVelocities(np.tile([1., 2., 3.], (37, 1)))
    integrator.step(60)
    integrator.setComFrequency(1)
    integrator.step(1)
    velocity = context.getState(getVelocities=True).getVelocities(asNumpy=True)._value
    np.testing.assert_allclose(velocity.mean(axis=0), 0., atol=1e-6, rtol=0.)
    del context, integrator


def test_reinitialize_resolves_new_barostat(monkeypatch):
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_GRAPH', '0')
    monkeypatch.setenv('CUPO_COMPACT_AUTOTUNE', '0')
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(.9)
    for _ in range(27):
        system.addParticle(20.)
        nonbonded.addParticle(0., .2, 0.)
    system.addForce(nonbonded)
    integrator = nv.LangevinMiddleIntegrator(300., 1., .001, 0)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('CUDA'),
                         {'Precision': 'mixed'})
    xyz = np.array([(x, y, z) for x in range(3) for y in range(3) for z in range(3)], float)+.3
    context.setPositions(xyz)
    integrator.step(2)  # Cache the absence of a plugin barostat.
    barostat = nv.MonteCarloBarostat(1., 300., 1)
    barostat.setRandomNumberSeed(771)
    index = system.addForce(barostat)
    context.reinitialize(preserveState=True)
    integrator.step(4)
    assert np.isfinite(context.getState(getPositions=True).getPositions(asNumpy=True)._value).all()
    system.removeForce(index)
    context.reinitialize(preserveState=True)
    integrator.step(4)  # The cached pointer must not refer to the old ForceImpl.
    assert context.getStepCount() == 10
    del context, integrator
