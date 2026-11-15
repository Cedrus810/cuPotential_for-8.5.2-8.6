# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Both pair modes use the same captured execution and OpenMM stream lifecycle."""

import contextlib
import itertools

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

from test_custom_nonbonded import switched_expression


# Some neighbors lie across the periodic boundary, so a stale device box changes
# the answer even if the cell list still happens to find the right particles.
XYZ = np.array(list(itertools.product((.2, .85, 3.65), repeat=3)))
OUTPUT_MODES = ((False, True), (True, False), (True, True))
FORCE_QUANTUM = float(np.float32(138.93545764498)) / 2**21


def _custom_source(record="precomputed"):
    expressions = {
        "zero": "lambda*scale/(1+r^2)",
        "three": "lambda*scale*(a1*a2*r^2-b1*b2*exp(-r)+c1*c2/(1+r^2))",
        "wide": "lambda*scale*(p01*p02+p641*p642)/(1+r^2)",
        "precomputed": (
            "q1*q2/(1+r^2);"
            "q1=(a1*(1-lambda)+b1*lambda)*sqrt(scale);"
            "q2=(a2*(1-lambda)+b2*lambda)*sqrt(scale)"
        ),
    }
    names = {"zero": (), "three": ("a", "b", "c"),
             "wide": tuple(f"p{i}" for i in range(65)),
             "precomputed": ("a", "b")}[record]
    force = mm.CustomNonbondedForce(switched_expression(expressions[record], cutoff=1.1))
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.1)
    force.addGlobalParameter("lambda", .35)
    force.addGlobalParameter("scale", .9)
    for name in names:
        force.addPerParticleParameter(name)
    for atom in range(len(XYZ)):
        force.addParticle([.2 + .007*atom + .003*field for field in range(len(names))])
    force.addExclusion(0, 1)
    force.addExclusion(2, 3)
    return force


def _standard_source():
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.PME)
    force.setCutoffDistance(1.1)
    force.setEwaldErrorTolerance(5e-4)
    # The active cross-face exception needs periodic geometry for stock Reference parity.
    force.setExceptionsUsePeriodicBoundaryConditions(True)
    force.setUseDispersionCorrection(False)
    for atom in range(len(XYZ)):
        force.addParticle((-1)**atom*.1, .25 + .001*atom, .2 + .002*atom)
    force.addException(0, 1, 0., .3, 0.)
    force.addException(2, 3, -.003, .27, .1)
    return force


def _box(length):
    return tuple(mm.Vec3(*(length if i == axis else 0. for i in range(3)))
                 for axis in range(3))


@contextlib.contextmanager
def _context(forces, platform, integrator=None):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(*_box(4.))
    for force in forces:
        system.addForce(force)
    # Verlet requests forces to evaluate kinetic energy, turning an ostensibly
    # energy-only getState() into E+F. Middle velocities need no force shift, so
    # these tests really visit all three backend output combinations.
    integrator = mm.LangevinMiddleIntegrator(0., 1., .001) if integrator is None else integrator
    properties = {"Precision": "mixed"} if platform == "CUDA" else {}
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    ctx.setPositions(XYZ)
    try:
        yield ctx
    finally:
        del ctx, integrator


def _snapshot(ctx, mode, groups=-1):
    energy, forces = mode
    state = ctx.getState(getEnergy=energy, getForces=forces, groups=groups)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) if energy else None,
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        if forces else None,
    )


def _compare(actual, expected, standard=False):
    for value in (*actual, *expected):
        if value is not None:
            assert np.isfinite(value).all()
    if expected[0] is not None:
        np.testing.assert_allclose(actual[0], expected[0], rtol=3e-5,
                                   atol=3e-4 if standard else 3e-5)
    if expected[1] is not None:
        # Each custom pair rounds to a 32-bit fixed-point force; account for at
        # most N-1 partners, in addition to FP32 arithmetic and PME truncation.
        atol = .5*FORCE_QUANTUM*(len(XYZ)-1) + (3e-4 if standard else 3e-5)
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=atol)


def _enable_graphs(monkeypatch, enabled=True):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", str(int(enabled)))
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")


def _capture_modes(ctx, capfd, force_count=1):
    capfd.readouterr()
    # Shared execution waits for 50 eager evaluations and two visits to an
    # output combination. Assert the trace: a test silently running eager would
    # not exercise the lifetime and stream-ordering contracts checked below.
    for _ in range(53):
        _snapshot(ctx, OUTPUT_MODES[0])
    for mode in OUTPUT_MODES[1:]:
        for _ in range(3):
            _snapshot(ctx, mode)
    trace = capfd.readouterr().err
    captures = [line for line in trace.splitlines()
                if "[nv-plugin cupo] graph capture #" in line]
    for energy, forces in OUTPUT_MODES:
        marker = f"(energy={int(energy)} forces={int(forces)})"
        assert sum(marker in line for line in captures) == force_count, trace


def _compare_modes(actual, reference, first=OUTPUT_MODES[0], groups=-1, standard=False):
    # Do not fetch positions/checkpoints or perform another evaluation before
    # this observation: either could synchronize or repair the state under test.
    for mode in (first,) + tuple(mode for mode in OUTPUT_MODES if mode != first):
        observed = _snapshot(actual, mode, groups)
        _compare(observed, _snapshot(reference, mode, groups), standard)


@pytest.mark.parametrize("record", ["zero", "three", "wide", "precomputed"])
def test_custom_captured_updates_and_cell_band_crossings(plugin_dir, monkeypatch, capfd, record):
    _enable_graphs(monkeypatch)
    stock = _custom_source(record)
    custom = nv.NonbondedForce(stock, 1.25)
    with _context([stock], "Reference") as reference, _context([custom], "CUDA") as actual:
        _capture_modes(actual, capfd)
        _compare_modes(actual, reference)

        # Every output mode must be correct when it is the first evaluation to
        # consume a changed global, including globals used by preprocessing.
        for mode, lam, scale in zip(OUTPUT_MODES, (.7, -.25, 1.1), (.6, 1.3, .8)):
            for ctx in (actual, reference):
                ctx.setParameter("lambda", lam)
                ctx.setParameter("scale", scale)
            _compare_modes(actual, reference, first=mode)

            if stock.getNumPerParticleParameters():
                for force, ctx in ((stock, reference), (custom, actual)):
                    values = list(force.getParticleParameters(5))
                    values[0] += .11
                    values[-1] -= .04
                    force.setParticleParameters(5, values)
                    force.updateParametersInContext(ctx)
                _compare_modes(actual, reference, first=mode)

        # floor(L/cutoff): 3 -> 3 -> 4 -> 3 -> 3. Scaling coordinates also moves
        # particles between cells and changes cross-boundary pair distances.
        for length, mode in zip((4.2, 4.5, 4.2, 4.),
                                (*OUTPUT_MODES, OUTPUT_MODES[0])):
            for ctx in (actual, reference):
                ctx.setPeriodicBoxVectors(*_box(length))
                ctx.setPositions(XYZ*(length/4.))
            _compare_modes(actual, reference, first=mode)
            # A few consecutive replays expose incomplete clearing after the
            # number of cells grows, without requiring a long MD trajectory.
            if length == 4.5:
                for _ in range(8):
                    _compare_modes(actual, reference, first=mode)
        trace = capfd.readouterr().err
        assert "graph capture #" not in trace, trace


def _inert_cutoff_force():
    # OpenMM only reorders atoms if a registered nonbonded interaction uses a
    # cutoff. The plugin owns its cell list, so supply a zero-valued stock force
    # to enable OpenMM's real reorder machinery without changing the potential.
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.1)
    force.setUseDispersionCorrection(False)
    for _ in XYZ:
        force.addParticle(0., .1, 0.)
    return force


def test_custom_first_evaluation_after_captured_reorder(plugin_dir, monkeypatch, capfd):
    _enable_graphs(monkeypatch)
    stock = _custom_source()
    # Equivalent excluded dimers and repeated singleton classes can reorder;
    # distinct classes and nonuniform records still expose misassociated values.
    for atom in range(len(XYZ)):
        group = atom % 2
        stock.setParticleParameters(atom, [.2 + .15*group, .65 - .1*group])
    for first in (4, 6, 8, 10):
        stock.addExclusion(first, first+1)
    custom = nv.NonbondedForce(stock, 1.25)
    with _context([stock, _inert_cutoff_force()], "Reference") as reference, \
            _context([custom, _inert_cutoff_force()], "CUDA") as actual:
        _capture_modes(actual, capfd)
        rng = np.random.default_rng(1729)
        for index, mode in enumerate(OUTPUT_MODES):
            for ctx in (actual, reference):
                ctx.setParameter("lambda", (.2, .6, .9)[index])
                ctx.setParameter("scale", (.7, 1.2, .5)[index])
            requested = XYZ[rng.permutation(len(XYZ))] + np.array([4., -4., 4.])
            # setPositions participates in OpenMM's 250-call reorder schedule.
            # Custom's trace is emitted during prepareStep only if the actual
            # atom-index vector changed. Retain that very evaluation's result;
            # a later evaluation could conceal a stale first replay.
            for _ in range(251):
                actual.setPositions(requested)
                observed = _snapshot(actual, mode)
                trace = capfd.readouterr().err
                assert "graph capture #" not in trace, trace
                if "[CustomNonbondedForce] atom reorder #" in trace:
                    break
            else:
                pytest.fail(f"no nontrivial Custom reorder after capture for mode {mode}")

            # Snapshot only AFTER observing the first reordered output. This
            # gives Reference the exact represented, periodically wrapped inputs
            # without adding a synchronization before the evaluation under test.
            reference.setPositions(actual.getState(getPositions=True).getPositions())
            expected = _snapshot(reference, mode)
            _compare(observed, expected)
            if mode[0]:
                assert abs(expected[0]) > 1e-3
            if mode[1]:
                assert np.max(np.abs(expected[1])) > 1e-3
        trace = capfd.readouterr().err
        assert "graph capture #" not in trace, trace


def _mixed_forces(plugin):
    standard = _standard_source()
    custom = _custom_source()
    if plugin:
        standard = nv.NonbondedForce(standard, 1.25, nv.NonbondedForce.PME)
        custom = nv.NonbondedForce(custom, 1.25)
    standard.setForceGroup(1)
    custom.setForceGroup(2)
    bonded = mm.HarmonicBondForce()
    bonded.setUsesPeriodicBoundaryConditions(True)
    bonded.setForceGroup(3)
    bonded.addBond(0, 1, .6, 5.)
    bonded.addBond(2, 3, .7, 3.)
    return standard, custom, bonded


def test_captured_standard_custom_and_stock_bonded_force_groups(plugin_dir, monkeypatch, capfd):
    _enable_graphs(monkeypatch)
    with _context(_mixed_forces(False), "Reference") as reference, \
            _context(_mixed_forces(True), "CUDA") as actual:
        _capture_modes(actual, capfd, force_count=2)
        for ctx in (actual, reference):
            ctx.setParameter("lambda", .8)
            ctx.setParameter("scale", .6)
        # Alternating selected and omitted components must neither leak pending
        # energy from a previous call nor skip the side-stream join for forces.
        masks = (2, 4, 8, 4 | 8, 2 | 4, 2 | 4 | 8, 16, 8, 4, 2 | 8)
        for index, groups in enumerate(masks):
            _compare_modes(actual, reference, first=OUTPUT_MODES[index % 3],
                           groups=groups, standard=True)
        trace = capfd.readouterr().err
        assert "graph capture #" not in trace, trace


def test_custom_with_nv_integrator_graph_matches_eager(plugin_dir, monkeypatch, capfd):
    results = []
    for graph in (False, True):
        _enable_graphs(monkeypatch, graph)
        force = nv.NonbondedForce(_custom_source(), 1.25)
        # Zero temperature removes stochastic divergence while retaining the nv
        # integrator's graph, force evaluation and velocity-update scheduling.
        integrator = nv.LangevinMiddleIntegrator(0., 1., .001, 0)
        integrator.setRandomNumberSeed(58392)
        with _context([force], "CUDA", integrator) as ctx:
            ctx.setVelocities(.005*np.sin(np.arange(XYZ.size)).reshape(XYZ.shape))
            capfd.readouterr()
            integrator.step(60)
            trace = capfd.readouterr().err
            if graph:
                assert "[nv-plugin cupo] graph capture #" in trace, trace
            else:
                assert "graph capture #" not in trace, trace
            ctx.setParameter("lambda", .8)
            ctx.setParameter("scale", .6)
            force.setParticleParameters(5, [.6, .15])
            force.updateParametersInContext(ctx)
            integrator.step(20)
            state = ctx.getState(getPositions=True, getVelocities=True)
            results.append((state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                            state.getVelocities(asNumpy=True).value_in_unit(
                                unit.nanometer/unit.picosecond)))
            trace = capfd.readouterr().err
            assert "graph capture #" not in trace, trace
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(results[1][1], results[0][1], rtol=0, atol=1e-7)


@pytest.mark.parametrize("component", ["cutoff", "lattice_sum", "expression"])
def test_component_changes_require_reinitialize_and_preserve_existing_context(plugin_dir, component):
    def make_force():
        return nv.NonbondedForce(_custom_source("three")) if component == "expression" else \
            nv.NonbondedForce(_standard_source(), 1.0, nv.NonbondedForce.PME)

    def change(force):
        if component == "lattice_sum":
            definition = force.getLatticeSumInteraction()
            definition.splitting = nv.LatticeSumInteraction.Prolate
            force.setLatticeSumInteraction(definition)
        else:
            definition = force.getCutoffInteraction()
            if component == "cutoff":
                definition.combiningRule = nv.CutoffInteraction.Geometric
                definition.useSwitching = True
                definition.switchingDistance = .7
                definition.tail = nv.CutoffInteraction.HomogeneousTail
            else:
                # Semicolon definitions belong after the main energy expression.
                energy, _, definitions = force.getEnergyFunction().partition(";")
                definition.expression = "1.7*(" + energy + ");" + definitions
            force.setCutoffInteraction(definition)

    force = make_force()
    with _context([force], "CUDA") as actual:
        before = _snapshot(actual, (True, True))
        change(force)
        with pytest.raises(Exception, match="reinitialize"):
            force.updateParametersInContext(actual)
        unchanged = _snapshot(actual, (True, True))
        for observed, expected in zip(unchanged, before):
            np.testing.assert_array_equal(observed, expected)
        actual.reinitialize(preserveState=True)
        expected_force = make_force()
        change(expected_force)
        with _context([expected_force], "CUDA") as fresh:
            _compare_modes(actual, fresh, standard=component != "expression")
        after = _snapshot(actual, (True, True))
        assert after[0] != before[0] or not np.array_equal(after[1], before[1])


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_cutoff_tail_policy_can_update_without_reinitialize(plugin_dir, method):
    force = nv.NonbondedForce(_standard_source(), 1.0, getattr(nv.NonbondedForce, method))
    with _context([force], "CUDA") as actual:
        before = _snapshot(actual, (True, True))
        cutoff = force.getCutoffInteraction()
        cutoff.tail = nv.CutoffInteraction.HomogeneousTail
        force.setCutoffInteraction(cutoff)
        force.updateParametersInContext(actual)
        with_tail = _snapshot(actual, (True, True))
        assert with_tail[0] != before[0]
        np.testing.assert_array_equal(with_tail[1], before[1])

        expected = nv.NonbondedForce(_standard_source(), 1.0, getattr(nv.NonbondedForce, method))
        expected.setUseDispersionCorrection(True)
        with _context([expected], "CUDA") as fresh:
            _compare_modes(actual, fresh, standard=True)

        cutoff.tail = nv.CutoffInteraction.NoTail
        force.setCutoffInteraction(cutoff)
        force.updateParametersInContext(actual)
        restored = _snapshot(actual, (True, True))
        for observed, initial in zip(restored, before):
            np.testing.assert_array_equal(observed, initial)


@pytest.mark.parametrize("config", ["352-384-2048-tc16", "416-384-2432-packed-tc16"])
@pytest.mark.parametrize("kernel", ["piece", "piece-general"])
def test_custom_zero_to_nonzero_updates_in_tensor_graphs(plugin_dir, monkeypatch, capfd,
                                                        config, kernel):
    """Zero-valued opaque records/globals never justify a native LJ shortcut."""
    _enable_graphs(monkeypatch)
    monkeypatch.setenv("CUPO_DSPACE_PIECE_CONFIG", config)
    monkeypatch.setenv("CUPO_DSPACE_PAIR_KERNEL", kernel)
    monkeypatch.setenv("CUPO_DSPACE_CLEAVE_TOWER", "1")
    monkeypatch.setenv("CUPO_DSPACE_SKIP_ZERO_PAIRS", "1")
    stock = mm.CustomNonbondedForce(
        "lambda*(r+q1*q2*r^2+epsilon1*epsilon2/(1+r^2))")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.1)
    stock.addGlobalParameter("lambda", 0.)
    for name in ("q", "sigma", "epsilon"):
        stock.addPerParticleParameter(name)
    for _ in XYZ:
        stock.addParticle([0., 0., 0.])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with _context([stock], "Reference") as reference, _context([custom], "CUDA") as actual:
        _capture_modes(actual, capfd)
        _compare_modes(actual, reference)
        for ctx in (reference, actual):
            ctx.setParameter("lambda", 1.)
        _compare_modes(actual, reference)
        assert np.any(_snapshot(actual, (False, True))[1] != 0.)

        # Change particle fields, globals and sorted membership, then consume
        # each update in a different output mode before any other evaluation.
        for phase, mode in enumerate(OUTPUT_MODES):
            for force, ctx in ((stock, reference), (custom, actual)):
                for atom in range(len(XYZ)):
                    values = ([(-1)**atom*.2, 0., .1+.002*atom]
                              if phase != 1 else [0., 0., 0.])
                    force.setParticleParameters(atom, values)
                force.updateParametersInContext(ctx)
                ctx.setParameter("lambda", (-.4, 0., 1.)[phase])
                # Move atom identities between occupied sites without exceeding
                # the sparse fixture's fixed per-cell occupancy envelope.
                ctx.setPositions(np.roll(XYZ, 5*(phase+1), axis=0))
            _compare_modes(actual, reference, first=mode)
        trace = capfd.readouterr().err
        assert "graph capture #" not in trace, trace
