# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPT whole-step replay must preserve barostat decisions and the kept state."""
import os
from unittest.mock import patch

import numpy as np
import openmm as mm
import nv_openmm_plugin as nv
import pytest

from test_whole_step_graph import nonbonded_system, whole_environment  # noqa: F401


def npt_trajectory(whole, precision, barostat, case, schedule="after-sort", reinitialize=False):
    system, xyz, vel = nonbonded_system(nv.NonbondedForce.Prolate, box_safety=1.35,
                                       virtual=case == 'virtual')
    if case == 'periodic':
        bonds = system.getForce(1)
        bonds.setUsesPeriodicBoundaryConditions(True)
        bonds.addBond(0, 7, .95, 10.)  # an interaction across the periodic boundary
    if case == 'exceptions':
        nonbonded = system.getForce(0)
        nonbonded.setExceptionsUsePeriodicBoundaryConditions(True)
        # The wrapped separation is 3.85 nm in a 4.8 nm box (MIC 0.95 nm).
        # Both physical terms must read the live box after accepted moves and
        # after rejected trials restore it, including whole-step graph replay.
        nonbonded.addException(0, 7, -.00125, .6, .05)
        nonbonded.setExceptionParameters(0, 0, 1, -.00075, .25, .03)
    cls = mm.MonteCarloBarostat if barostat == 'stock' else nv.MonteCarloBarostat
    baro = cls(1., 300., 7)
    baro.setRandomNumberSeed(777)
    system.addForce(baro)
    integ = nv.LangevinMiddleIntegrator(300., 1., .001, 11)
    integ.setRandomNumberSeed(58392)
    settings = {'CUPO_PLUGIN_WHOLE_STEP_GRAPH': str(int(whole)),
                'CUPO_PLUGIN_CUDA_GRAPH': '1', 'CUPO_PLUGIN_BONDED_SCHEDULE': schedule}
    with patch.dict(os.environ, settings):
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision': precision})
    ctx.setPositions(xyz)
    ctx.setVelocities(vel)
    ctx.computeVirtualSites()
    states, boxes = [], []

    def snapshot():
        s = ctx.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True)
        box = np.array(s.getPeriodicBoxVectors(asNumpy=True)._value)
        boxes.append(box)
        states.append((np.array(s.getPositions(asNumpy=True)._value),
                       np.array(s.getVelocities(asNumpy=True)._value),
                       np.array(s.getForces(asNumpy=True)._value), box,
                       s.getPotentialEnergy()._value, s.getKineticEnergy()._value,
                       s.getTime()._value, s.getStepCount()))

    # The long call crosses many actual moves without intervening state queries.
    integ.step(60)
    if reinitialize:
        with patch.dict(os.environ, settings):
            ctx.reinitialize(preserveState=True)
        assert integ.getWholeStepGraphCaptureCount() == 0
    if case == 'bandcross':
        # Put the box just below the 5-cell boundary AFTER graph warmup. Actual
        # accepted Monte Carlo expansions must cross it during the long call.
        ctx.setPeriodicBoxVectors(mm.Vec3(4.49, 0, 0), mm.Vec3(0, 4.49, 0), mm.Vec3(0, 0, 4.49))
    captures_before = integ.getWholeStepGraphCaptureCount()
    integ.step(250)
    captures_during_moves = integ.getWholeStepGraphCaptureCount()-captures_before
    snapshot()
    if case == 'bandcross':
        assert boxes[0][0, 0] > 4.5, boxes[0]
    # One attempt between snapshots exposes both accepted and rejected moves.
    for _ in range(24):
        integ.step(7)
        snapshot()
    # Frequency changes must use the barostat's private schedule, not step % freq.
    baro.setFrequency(0)
    integ.step(80)
    snapshot()
    baro.setFrequency(3)
    ctx.setParameter(mm.MonteCarloBarostat.Pressure(), 200.)
    ctx.setParameter(mm.MonteCarloBarostat.Temperature(), 280.)
    integ.step(35)
    snapshot()
    counters = integ.getWholeStepGraphCaptureCount(), integ.getWholeStepGraphReplayCount()
    decisions = [not np.array_equal(a, b) for a, b in zip(boxes[:24], boxes[1:25])]
    del ctx, integ
    return states, counters, captures_during_moves, decisions


@pytest.mark.parametrize('schedule', ['overlap', 'after-sort'])
@pytest.mark.parametrize('precision', ['single', 'mixed'])
@pytest.mark.parametrize('barostat', ['stock', 'plugin'])
@pytest.mark.parametrize('case', ['plain', 'exceptions', 'periodic', 'virtual', 'bandcross'])
def test_whole_step_npt_matches_component_graphs(precision, barostat, case, schedule):
    ref, ref_counts, _, ref_decisions = npt_trajectory(False, precision, barostat, case, schedule)
    got, counts, captures, decisions = npt_trajectory(True, precision, barostat, case, schedule)
    assert ref_counts == (0, 0)
    assert counts[1] > 250, counts
    assert any(decisions) and not all(decisions), decisions
    assert decisions == ref_decisions
    if case in ('plain', 'exceptions', 'bandcross'):
        assert captures == 2, 'box-independent graphs must survive barostat moves'
    else:
        assert captures > 2, 'box-dependent work must be recaptured after accepted moves'
    for expected, actual in zip(ref, got):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize("barostat", ["stock", "plugin"])
def test_whole_step_reinitialize_barostat(precision, barostat):
    ref, _, _, _ = npt_trajectory(False, precision, barostat, "plain", reinitialize=True)
    got, counts, _, _ = npt_trajectory(True, precision, barostat, "plain", reinitialize=True)
    assert counts[1] > 150
    for expected, actual in zip(ref, got):
        for a, b in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
