# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bare exception laws survive splitting choices, graph replay, and live updates."""
import contextlib

import numpy as np
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv
import pytest

BOX = np.array([[3.6, 0., 0.], [.35, 3.7, 0.], [-.2, .25, 3.8]])
XYZ = np.array([[.05, .4, .3], [3.4, .4, .3],
                [.3, 1.4, 1.2], [1.7, 1.5, 1.3],
                [1.2, 2.4, 2.3], [1.6, 2.6, 2.5]])
PAIRS = [(0, 1), (2, 3), (4, 5)]
GROUP = 31


def _source(ljpme, periodic=True):
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.LJPME if ljpme else force.PME)
    force.setCutoffDistance(.9)
    force.setExceptionsUsePeriodicBoundaryConditions(periodic)
    force.setUseDispersionCorrection(False)
    force.setForceGroup(GROUP)
    force.addGlobalParameter("lam", 0.)
    for _ in XYZ:
        force.addParticle(0., .25, 0.)
    for (i, j), dq, de in zip(PAIRS, [.002, -.003, 0.], [.05, .03, 0.]):
        row = force.addException(i, j, 0., .27, 0.)
        # Even initially zero terms have fixed topology. Reserve the stock
        # baseline's rows through offsets, including a zero-scale offset.
        force.addExceptionParameterOffset("lam", row, dq, .01, de)
    return force


def _cutoff(source):
    force = mm.CustomNonbondedForce("4*sqrt(epsilon1*epsilon2)*((sigma/r)^12-(sigma/r)^6);sigma=(sigma1+sigma2)/2")
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(.9)
    force.addPerParticleParameter("sigma")
    force.addPerParticleParameter("epsilon")
    for atom in range(source.getNumParticles()):
        _, sigma, epsilon = source.getParticleParameters(atom)
        force.addParticle([sigma.value_in_unit(unit.nanometer),
                           epsilon.value_in_unit(unit.kilojoule_per_mole)])
    for i, j in PAIRS:
        force.addExclusion(i, j)
    return force


@contextlib.contextmanager
def _context(force, precision, extra=()):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(12.)
    system.setDefaultPeriodicBoxVectors(*(mm.Vec3(*v) for v in BOX))
    system.addForce(force)
    for added in extra:
        system.addForce(added)
    integ = mm.LangevinMiddleIntegrator(0., 1., .001)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": precision, "DeterministicForces": "true"})
    ctx.setPositions(XYZ)
    try:
        yield ctx
    finally:
        del ctx, integ


def _state(ctx, mode=(True, True), groups=-1):
    state = ctx.getState(getEnergy=mode[0], getForces=mode[1], groups=groups)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) if mode[0] else None,
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer) if mode[1] else None)


def _analytic(source, lam, xyz, box, periodic=None):
    if periodic is None:
        periodic = source.getExceptionsUsePeriodicBoundaryConditions()
    offsets = {int(source.getExceptionParameterOffset(row)[1]):
               source.getExceptionParameterOffset(row)[2:] for row in range(source.getNumExceptionParameterOffsets())}
    energy = 0.
    force = np.zeros_like(xyz)
    for row in range(source.getNumExceptions()):
        i, j, q, sigma, epsilon = source.getExceptionParameters(row)
        dq, ds, de = offsets[row]
        q = q.value_in_unit(unit.elementary_charge**2)+lam*dq
        sigma = sigma.value_in_unit(unit.nanometer)+lam*ds
        epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole)+lam*de
        d = xyz[j]-xyz[i]
        if periodic:
            for axis in (2, 1, 0):
                d -= np.floor(d[axis]/box[axis, axis]+.5)*box[axis]
        r2 = float(d@d)
        r = np.sqrt(r2)
        s6 = (sigma*sigma/r2)**3
        energy += 138.93545764498*q/r+4*epsilon*(s6*s6-s6)
        f = (138.93545764498*q/(r2*r)+24*epsilon*s6*(2*s6-1)/r2)*d
        force[i] -= f
        force[j] += f
    return energy, force


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("ljpme,custom,periodic", [
    (False, False, False), (True, False, True), (True, True, False)])
def test_exception_only_zero_activation_offsets_and_graphs(
        monkeypatch, capfd, plugin_dir, precision, method, ljpme, custom, periodic):
    # Disabling grouped harmonic terms must not disable NonbondedForce's bare
    # exceptions, even when this is the only force in the Context.
    for name, value in {"CUPO_PLUGIN_BONDED": "0", "CUPO_COMPACT_AUTOTUNE": "0",
                        "CUPO_PLUGIN_CUDA_GRAPH": "1", "CUPO_PLUGIN_GRAPH_TRACE": "1"}.items():
        monkeypatch.setenv(name, value)
    source = _source(ljpme, periodic)
    candidate = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
    if custom:
        candidate.setCutoffInteraction(_cutoff(source))
    with _context(source, precision) as reference, _context(candidate, precision) as actual:
        modes = ((True, True), (False, True), (True, False))
        capfd.readouterr()
        for _ in range(56):
            for mode in modes:
                _state(actual, mode)
        trace = capfd.readouterr().err
        for marker in ("(energy=1 forces=1)", "(energy=0 forces=1)", "(energy=1 forces=0)"):
            assert any("graph capture #" in line and marker in line for line in trace.splitlines()), trace
        for phase, lam in enumerate((0., .4, .9, 0.)):
            xyz = XYZ + phase*np.array([.003, -.002, .001])
            box = BOX*(1.02 if phase >= 2 else 1.)
            for ctx in (reference, actual):
                ctx.setParameter("lam", lam)
                ctx.setPositions(xyz)
                ctx.setPeriodicBoxVectors(*(mm.Vec3(*v) for v in box))
            if phase == 2:
                for force, ctx in ((source, reference), (candidate, actual)):
                    i, j, _, sigma, _ = force.getExceptionParameters(2)
                    force.setExceptionParameters(2, i, j, .001, sigma, .07)
                    force.updateParametersInContext(ctx)
            expected = _analytic(source, lam, xyz, box)
            for mode in modes:
                for observed in (_state(reference, mode), _state(actual, mode)):
                    if mode[0]:
                        np.testing.assert_allclose(observed[0], expected[0], atol=3e-5, rtol=1e-5)
                    if mode[1]:
                        np.testing.assert_allclose(observed[1], expected[1], atol=2e-4, rtol=2e-5)
            absent = _state(actual, groups=1 << 3)
            assert absent[0] == 0.
            np.testing.assert_array_equal(absent[1], 0.)
            for observed, wanted in zip(_state(actual, groups=1 << GROUP), expected):
                np.testing.assert_allclose(observed, wanted, atol=2e-4, rtol=2e-5)
            if phase == 0:
                assert expected[0] == 0.
            else:
                assert abs(expected[0]) > 1e-4


@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("ljpme,custom", [
    (False, False), (False, True), (True, False), (True, True)],
    ids=["p1-native", "p1-expression", "p1-p6-native", "p1-p6-expression"])
def test_exception_pbc_changes_only_physical_replacements(
        monkeypatch, capfd, plugin_dir, method, ljpme, custom):
    """Periodic cancellation is identical; only bare q/r and LJ use this flag."""
    for name, value in {"CUPO_PLUGIN_BONDED": "0", "CUPO_COMPACT_AUTOTUNE": "0",
                        "CUPO_PLUGIN_CUDA_GRAPH": "1", "CUPO_PLUGIN_GRAPH_TRACE": "1"}.items():
        monkeypatch.setenv(name, value)
    sources, candidates = [], []
    for periodic in (False, True):
        source = _source(ljpme, periodic)
        # Nonzero ordinary weights make an incorrectly nonperiodic dspace or
        # mesh correction visible even when every physical exception is zero.
        for atom in range(len(XYZ)):
            source.setParticleParameters(atom, -.1+.04*atom, .25+.002*atom, .06+.01*atom)
        source.addParticleParameterOffset("lam", 0, .01, 0., 0.)
        source.addParticleParameterOffset("lam", 1, -.01, 0., 0.)
        candidate = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
        if custom:
            candidate.setCutoffInteraction(_cutoff(source))
        sources.append(source)
        candidates.append(candidate)
    with _context(candidates[0], "mixed") as raw, _context(candidates[1], "mixed") as periodic:
        capfd.readouterr()
        for _ in range(56):
            _state(raw)
            _state(periodic)
        trace = capfd.readouterr().err
        assert sum("graph capture #" in line and "(energy=1 forces=1)" in line
                   for line in trace.splitlines()) >= 2, trace
        for phase, lam in enumerate((0., .65, .9, 0.)):
            xyz = XYZ.copy()
            xyz[1] += phase*np.array([-.004, .002, -.001])
            box = BOX*(1.02 if phase >= 2 else 1.)
            for ctx in (raw, periodic):
                ctx.setParameter("lam", lam)
                ctx.setPositions(xyz)
                ctx.setPeriodicBoxVectors(*(mm.Vec3(*v) for v in box))
            raw_e, raw_f = _state(raw)
            periodic_e, periodic_f = _state(periodic)
            raw_pair = _analytic(sources[0], lam, xyz, box, periodic=False)
            periodic_pair = _analytic(sources[0], lam, xyz, box, periodic=True)
            delta_e = periodic_pair[0]-raw_pair[0]
            delta_f = periodic_pair[1]-raw_pair[1]
            np.testing.assert_allclose(periodic_e-raw_e, delta_e, atol=3e-5, rtol=2e-5)
            np.testing.assert_allclose(periodic_f-raw_f, delta_f, atol=2e-4, rtol=2e-5)
            if lam == 0.:
                # Both conventions must agree for pure exclusions, including
                # a cross-face pair within the direct-space cutoff.
                assert delta_e == 0.
                np.testing.assert_array_equal(delta_f, 0.)
                assert periodic_e == raw_e
                np.testing.assert_array_equal(periodic_f, raw_f)
            else:
                assert abs(delta_e) > 1e-2
                assert np.max(np.abs(delta_f)) > 1e-2


@pytest.mark.parametrize("custom", [False, True], ids=["native", "expression"])
def test_exception_pbc_change_requires_context_reinitialization(monkeypatch, plugin_dir, custom):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    source = _source(False, periodic=False)
    candidate = nv.NonbondedForce(source, 1.15, nv.NonbondedForce.PME)
    if custom:
        candidate.setCutoffInteraction(_cutoff(source))
    with _context(candidate, "mixed") as ctx:
        ctx.setParameter("lam", .65)
        previous = _state(ctx)
        candidate.setExceptionsUsePeriodicBoundaryConditions(True)
        for force in (source, candidate):
            i, j, _, sigma, _ = force.getExceptionParameters(1)
            force.setExceptionParameters(1, i, j, .0015, sigma, .02)
        with pytest.raises(Exception, match="(?i)(periodic|reinitial)"):
            candidate.updateParametersInContext(ctx)
        unchanged = _state(ctx)
        assert unchanged[0] == previous[0]
        np.testing.assert_array_equal(unchanged[1], previous[1])
        ctx.reinitialize(preserveState=True)
        expected = _analytic(source, .65, XYZ, BOX, periodic=True)
        observed = _state(ctx)
        np.testing.assert_allclose(observed[0], expected[0], atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose(observed[1], expected[1], atol=2e-4, rtol=2e-5)
        assert abs(observed[0]-previous[0]) > 1e-2


@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_harmonic_and_exception_updates_preserve_other_force_groups(monkeypatch, plugin_dir, precision):
    for name, value in {"CUPO_PLUGIN_BONDED": "1", "CUPO_COMPACT_AUTOTUNE": "0",
                        "CUPO_PLUGIN_CUDA_GRAPH": "1"}.items():
        monkeypatch.setenv(name, value)
    source = _source(False)
    candidate = nv.NonbondedForce(source, 1.15, nv.NonbondedForce.PME)
    bonds = [mm.HarmonicBondForce(), mm.HarmonicBondForce()]
    for bond in bonds:
        bond.setForceGroup(1)
        bond.addBond(4, 5, .45, 12.)
    with _context(source, precision, [bonds[0]]) as reference, \
         _context(candidate, precision, [bonds[1]]) as actual:
        for ctx in (reference, actual):
            ctx.setParameter("lam", .4)
        for _ in range(56):
            _state(actual)
        previous = {group: _state(actual, groups=1 << group) for group in (1, GROUP)}
        for change in ("bond", "exception", "bond", "exception"):
            if change == "bond":
                touched = 1
                for bond, ctx in zip(bonds, (reference, actual)):
                    i, j, length, k = bond.getBondParameters(0)
                    bond.setBondParameters(0, i, j, length, 1.5*k)
                    bond.updateParametersInContext(ctx)
            else:
                touched = GROUP
                for force, ctx in ((source, reference), (candidate, actual)):
                    i, j, q, sigma, epsilon = force.getExceptionParameters(0)
                    q = q.value_in_unit(unit.elementary_charge**2) if unit.is_quantity(q) else q
                    epsilon = epsilon.value_in_unit(unit.kilojoule_per_mole) if unit.is_quantity(epsilon) else epsilon
                    force.setExceptionParameters(0, i, j, q+.001, sigma, epsilon+.02)
                    force.updateParametersInContext(ctx)
            for group in (1, GROUP):
                got = _state(actual, groups=1 << group)
                expected = _state(reference, groups=1 << group)
                np.testing.assert_allclose(got[0], expected[0], atol=3e-5, rtol=1e-5)
                np.testing.assert_allclose(got[1], expected[1], atol=2e-4, rtol=2e-5)
                if group == touched:
                    assert got[0] != previous[group][0]
                else:
                    assert got[0] == previous[group][0]
                    np.testing.assert_array_equal(got[1], previous[group][1])
                previous[group] = got
            # A rejected proposal must leave both cached entries and the live
            # graph inputs unchanged before the next alternating valid update.
            saved = candidate.getExceptionParameters(0)
            candidate.setExceptionParameters(0, *saved[:4], -1.)
            with pytest.raises(Exception, match="(?i)(exception|nonnegative)"):
                candidate.updateParametersInContext(actual)
            candidate.setExceptionParameters(0, *saved)
            for group in (1, GROUP):
                unchanged = _state(actual, groups=1 << group)
                assert unchanged[0] == previous[group][0]
                np.testing.assert_array_equal(unchanged[1], previous[group][1])
