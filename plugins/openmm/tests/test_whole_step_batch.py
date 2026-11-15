# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched replay must retain step, RNG, CoM, mutation and barostat semantics."""
import json
import re

import numpy as np
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv
import pytest

from test_whole_step_graph import trajectory, nonbonded_system


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.delenv('CUPO_PLUGIN_BONDED_SCHEDULE', raising=False)
    for key, value in {'CUPO_COMPACT_AUTOTUNE': '0', 'CUPO_PLUGIN_CUDA_GRAPH': '1',
                       'CUPO_PLUGIN_INTEGRATOR_GRAPH': '0',
                       'CUPO_PLUGIN_WHOLE_STEP_GRAPH': '1',
                       'CUPO_PLUGIN_WHOLE_STEP_BATCH_REPORT': '1'}.items():
        monkeypatch.setenv(key, value)


def reports(capfd):
    captured = capfd.readouterr()
    return [json.loads(s) for s in re.findall(r'WHOLE_STEP_BATCH (\{[^\n]+\})', captured.err)]


@pytest.mark.parametrize('precision', ['single', 'mixed'])
@pytest.mark.parametrize('case', ['prolate', 'water_virtual'])
@pytest.mark.parametrize('batch', [2, 8, 100, 1024])
def test_batch_matches_single_step(monkeypatch, capfd, precision, case, batch):
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH', '1')
    expected, counts = trajectory(case, precision, True, mutate=True)
    reports(capfd)
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH', str(batch))
    got, batch_counts = trajectory(case, precision, True, mutate=True)
    logs = reports(capfd)
    assert any(r['max_batch'] > 1 and r['batched_steps'] > 0 for r in logs)
    assert all(r['max_batch'] <= batch for r in logs)
    assert batch_counts[1] == counts[1]
    for ref, actual in zip(expected, got):
        for x, y in zip(ref, actual):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('com', [0, 1, 7, 100])
def test_large_batch_and_short_calls(monkeypatch, capfd, com):
    snapshots = []
    for batch in [1, 4096]:
        monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH', str(batch))
        system, xyz, vel = nonbonded_system(nv.NonbondedForce.Prolate)
        integ = nv.LangevinMiddleIntegrator(300., 1., .001, com)
        integ.setRandomNumberSeed(397)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'single'})
        ctx.setPositions(xyz); ctx.setVelocities(vel)
        states = []
        for n in [1, 54, 1, 0, 11, 8193, 3]:
            integ.step(n)
            state = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
            states.append([np.asarray(state.getPositions(asNumpy=True)._value),
                np.asarray(state.getVelocities(asNumpy=True)._value),
                np.asarray(state.getForces(asNumpy=True)._value), state.getPotentialEnergy()._value,
                state.getKineticEnergy()._value, state.getTime()._value, state.getStepCount()])
        logs = reports(capfd)
        if batch > 1:
            assert any(r['max_batch'] == batch for r in logs)
        snapshots.append(states)
        del ctx, integ
    for ref, actual in zip(*snapshots):
        for x, y in zip(ref, actual):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('kind', ['plugin', 'stock'])
@pytest.mark.parametrize('precision', ['single', 'mixed'])
def test_barostat_keeps_single_step_boundaries(monkeypatch, capfd, kind, precision):
    snapshots = []
    for batch in [1, 1024]:
        monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH', str(batch))
        system, xyz, vel = nonbonded_system(nv.NonbondedForce.Prolate, box_safety=1.2)
        barostat = (nv.MonteCarloBarostat if kind == 'plugin' else mm.MonteCarloBarostat)(1., 300., 7)
        barostat.setRandomNumberSeed(777)
        system.addForce(barostat)
        integ = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
        integ.setRandomNumberSeed(397)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision': precision})
        ctx.setPositions(xyz); ctx.setVelocities(vel)
        states = []
        for frequency, steps in [(7, 120), (5, 103), (0, 201)]:
            barostat.setFrequency(frequency)
            integ.step(steps)
            state = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
            states.append([np.asarray(state.getPositions(asNumpy=True)._value),
                np.asarray(state.getVelocities(asNumpy=True)._value),
                np.asarray(state.getForces(asNumpy=True)._value), state.getPotentialEnergy()._value,
                state.getKineticEnergy()._value, np.asarray(state.getPeriodicBoxVectors(asNumpy=True)._value),
                state.getTime()._value, state.getStepCount()])
        logs = reports(capfd)
        assert logs and all(not r['eligible'] and r['batched_steps'] == 0 for r in logs)
        assert all(r['clear_migration_overlaps'] == r['clear_grid_resizes'] == 0 for r in logs)
        assert not np.array_equal(states[0][5], np.eye(3)*4.8)
        snapshots.append(states)
        del ctx, integ
    for ref, actual in zip(*snapshots):
        for x, y in zip(ref, actual):
            np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('value', ['0', '-1', '16385', 'two', '2x', ''])
def test_invalid_batch_setting(monkeypatch, value):
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH', value)
    system, _, _ = nonbonded_system(nv.NonbondedForce.Prolate)
    integ = nv.LangevinMiddleIntegrator(300., 1., .001)
    with pytest.raises(mm.OpenMMException, match='CUPO_PLUGIN_WHOLE_STEP_BATCH'):
        mm.Context(system, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision': 'single'})
