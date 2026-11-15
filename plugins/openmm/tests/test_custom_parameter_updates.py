# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom records and exclusion topology have independent invalidation rules."""
import contextlib

import numpy as np
import openmm as mm
import pytest
import nv_openmm_plugin as nv

from test_custom_nonbonded import XYZ, compare, context, snapshot
from test_custom_precompute import alchemical_force, compare_modes
from test_custom_pme import (
    MODES, XYZ as FUSED_XYZ, _capture_modes, _compare, _context, _definitions,
    _inert_cutoff_force, _reference_forces, _repeated_classes, _single,
    _snapshot as fused_snapshot, _system,
)


PARTICLE_UPLOAD = "[CustomNonbondedForce] particle upload #"
EXCLUSION_UPLOAD = "[CustomNonbondedForce] exclusion upload #"
REORDER = "[CustomNonbondedForce] atom reorder #"
CHARGE_UPDATE = "[CustomNonbondedForce] managed charge update"
PREPROCESS = "[cuPo expression] particle preprocessing #"


def assert_uploads(capfd, particles, exclusions=0, charges=0):
    trace = capfd.readouterr().err
    assert trace.count(PARTICLE_UPLOAD) == particles, trace
    assert trace.count(EXCLUSION_UPLOAD) == exclusions, trace
    assert trace.count(CHARGE_UPDATE) == charges, trace
    assert "graph capture #" not in trace, trace
    return trace


def _record_expression(expression, layout):
    if layout == "compact":
        # Drop the unused region field; managed q changes its packed index.
        expression.setEnergyFunction("epsilon1*epsilon2/(r+sigma1+sigma2)")
    elif layout == "computed":
        # The managed raw charge must reach packed records alongside a transform
        # whose output also depends on a live global and public particle fields.
        expression.setEnergyFunction(
            "u1*u2/(r+sigma1+sigma2);"
            "u1=lambda_s*sqrt(epsilon1);u2=lambda_s*sqrt(epsilon2)")


@pytest.mark.parametrize("layout", ["raw", "compact", "computed"])
def test_fused_charge_and_record_updates_preserve_exclusion_order(
        plugin_dir, monkeypatch, capfd, layout):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    base, expression = _definitions(True, scenario="abfe")
    _record_expression(expression, layout)
    combined = _single(base, expression, "PME")
    reference_forces, platform = _reference_forces(base, expression, "PME")
    reference_base = reference_forces[0]
    with _context(_system([combined]), "CUDA") as actual, \
            _context(_system(reference_forces), platform) as reference:
        _capture_modes(actual, capfd)
        _compare(actual, reference)
        capfd.readouterr()

        # Revalidating identical public records must not upload them again.
        combined.updateParametersInContext(actual)
        _compare(actual, reference)
        assert_uploads(capfd, 0)

        # This is a different double lambda, but all managed FP32 charges are
        # identical. It still enters the native offset update path.
        for ctx in (actual, reference):
            ctx.setParameter("lambda_q", np.nextafter(.2, 1.))
        _compare(actual, reference)
        assert_uploads(capfd, 0)

        # Native LJ offsets and pair-only globals also leave managed q intact.
        for ctx in (actual, reference):
            ctx.setParameter("lambda_s", .72)
        _compare(actual, reference)
        assert_uploads(capfd, 0)

        for ctx in (actual, reference):
            ctx.setParameter("lambda_q", .63)
        _compare(actual, reference)
        trace = assert_uploads(capfd, 0, charges=1)
        assert trace.count(PREPROCESS) == int(layout != "raw"), trace
        _compare(actual, reference)
        assert_uploads(capfd, 0)

        # An independent exception law changes without changing particle order,
        # ordinary charges, or custom records.
        for ctx in (actual, reference):
            ctx.setParameter("lambda_14", .79)
        _compare(actual, reference)
        assert_uploads(capfd, 0)

        # A pending charge update and a public edit share one complete upload;
        # no subsequent scatter may overwrite its current logical charge values.
        for ctx in (actual, reference):
            ctx.setParameter("lambda_q", .41)
        record = list(expression.getParticleParameters(19))
        record[1] *= 1.7
        expression.setParticleParameters(19, record)
        combined.setCutoffInteraction(expression)
        combined.updateParametersInContext(actual)
        expression.updateParametersInContext(reference)
        _compare(actual, reference)
        assert_uploads(capfd, 1)

        # Signed-zero charge bits are significant to the prepared record even
        # though Coulomb energies do not distinguish these two states.
        for charge in (0., -0.):
            for force, ctx in ((combined, actual), (reference_base, reference)):
                _, sigma, epsilon = force.getParticleParameters(7)
                force.setParticleParameters(7, charge, sigma, epsilon)
                force.updateParametersInContext(ctx)
            _compare(actual, reference)
            assert_uploads(capfd, 0, charges=1)

        # A bad later record must not commit an earlier valid change.
        first = list(expression.getParticleParameters(0))
        first[0] *= 1.2
        expression.setParticleParameters(0, first)
        last = list(expression.getParticleParameters(expression.getNumParticles()-1))
        last[0] = float("nan")
        expression.setParticleParameters(expression.getNumParticles()-1, last)
        combined.setCutoffInteraction(expression)
        # The invalid Custom record must also prevent a valid native edit from
        # committing: the common owner validates before changing either model.
        charge, sigma, epsilon = combined.getParticleParameters(0)
        combined.setParticleParameters(0, charge*1.3, sigma, epsilon)
        with pytest.raises(Exception, match="finite FP32"):
            combined.updateParametersInContext(actual)
        _compare(actual, reference)
        assert_uploads(capfd, 0)


@pytest.mark.parametrize("layout", ["raw", "computed"])
def test_managed_charge_updates_with_reorder_and_checkpoint_are_exact(
        plugin_dir, monkeypatch, capfd, layout):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    # Independent contexts need the same reciprocal algorithm for bit equality.
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    monkeypatch.setenv("CUPO_DSPACE_AUTOTUNE", "0")
    base, expression = _definitions(True, scenario="abfe")
    _record_expression(expression, layout)
    base = _repeated_classes(base, expression)
    system = _system([_single(base, expression, "PME"), _inert_cutoff_force()])
    with _context(system, "CUDA") as actual:
        _capture_modes(actual, capfd)
        initial = actual.createCheckpoint()
        saved = []

        def save_exception_only_update(value):
            actual.setParameter("lambda_14", value)
            observed = fused_snapshot(actual, MODES[2])
            trace = assert_uploads(capfd, 0)
            assert REORDER not in trace, trace
            # Different 1-4 laws must affect this first captured result, while
            # the same coordinates/order permit reuse of the exception mapping.
            assert observed[0] != saved[-1][2][0]
            saved.append((actual.createCheckpoint(), MODES[2], observed))

        rng = np.random.default_rng(43)
        for index, mode in enumerate(MODES):
            requested = FUSED_XYZ[rng.permutation(len(FUSED_XYZ))]
            for attempt in range(251):
                # Change q on EVERY attempt so the first evaluation after the
                # observed permutation must handle both updates together.
                actual.setParameter("lambda_q", .1+.2*index+.03*(attempt % 2))
                actual.setPositions(requested)
                observed = fused_snapshot(actual, mode)
                trace = capfd.readouterr().err
                assert "graph capture #" not in trace, trace
                if REORDER in trace:
                    assert PARTICLE_UPLOAD in trace, trace
                    assert CHARGE_UPDATE not in trace, trace
                    break
            else:
                pytest.fail("no nontrivial fused reorder with a pending charge update")
            # Save only after retaining the first result; no synchronization or
            # coordinate reset is allowed to repair that tested evaluation.
            saved.append((actual.createCheckpoint(), mode, observed))

        # Reuse the mapping established by the last proven atom permutation.
        save_exception_only_update(.76)

        # The checkpoint restores its charge globals and atom permutation.
        # A further pending q edit must reach the first restored graph replay.
        actual.loadCheckpoint(initial)
        actual.setParameter("lambda_q", .87)
        observed = fused_snapshot(actual, MODES[2])
        trace = capfd.readouterr().err
        assert "graph capture #" not in trace, trace
        saved.append((actual.createCheckpoint(), MODES[2], observed))
        # Restoring a different order must invalidate that same mapping cache.
        save_exception_only_update(.43)

        # A fresh workspace uploads logical host records rather than reusing the
        # tested scatter or cell list. Checkpoint preserves exact positions/order.
        for checkpoint, mode, observed in saved:
            with _context(system, "CUDA") as oracle:
                oracle.loadCheckpoint(checkpoint)
                assert oracle.createCheckpoint() == checkpoint
                expected = fused_snapshot(oracle, mode)
                if mode[0]:
                    assert observed[0] == expected[0]
                if mode[1]:
                    np.testing.assert_array_equal(observed[1], expected[1])


def test_standalone_custom_reorder_notification_updates_both_payloads(plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    stock = alchemical_force()
    custom = nv.CustomNonbondedForce(stock)
    # The inert stock cutoff force enables actual OpenMM reordering. Standalone
    # Custom must receive notifications without a native lattice-sum component.
    with context(stock, "Reference", True, middle=True) as reference, \
            context(custom, "CUDA", True, middle=True) as actual:
        _capture_modes(actual, capfd)
        compare_modes(actual, reference)
        checkpoint = actual.createCheckpoint()
        checkpoint_positions = actual.getState(getPositions=True).getPositions()
        capfd.readouterr()
        requested = XYZ[np.random.default_rng(31).permutation(len(XYZ))]
        for _ in range(251):
            actual.setPositions(requested)
            observed = snapshot(actual, *MODES[1])
            trace = capfd.readouterr().err
            assert "graph capture #" not in trace, trace
            if REORDER in trace:
                assert PARTICLE_UPLOAD in trace and EXCLUSION_UPLOAD in trace, trace
                break
        else:
            pytest.fail("no nontrivial standalone Custom reorder after capture")
        # Compare the first captured output after reordering before any later
        # force evaluation could repair a stale permutation.
        reference.setPositions(actual.getState(getPositions=True).getPositions())
        compare(observed, snapshot(reference, *MODES[1]))
        compare_modes(actual, reference)
        assert_uploads(capfd, 0)

        # Restoring a checkpoint also restores its atom permutation. Validate
        # the first captured combined result before any other force evaluation
        # could refresh stale records, using the saved reference coordinates.
        actual.loadCheckpoint(checkpoint)
        reference.setPositions(checkpoint_positions)
        observed = snapshot(actual, *MODES[2])
        compare(observed, snapshot(reference, *MODES[2]))
        trace = assert_uploads(capfd, 1, 1)
        assert REORDER in trace, trace
        compare_modes(actual, reference)
        assert_uploads(capfd, 0)

        for force, ctx in ((stock, reference), (custom, actual)):
            force.setParticleParameters(5, [.4, .42, .3, -.2, .35, .25, 1., 1.])
            force.updateParametersInContext(ctx)
        compare_modes(actual, reference)
        # Explicit edits can reset molecule equivalence and legitimately reorder;
        # correctness above is required whether or not that mapping changes.
        capfd.readouterr()
        compare_modes(actual, reference)
        assert_uploads(capfd, 0)


@pytest.mark.parametrize("custom", [False, True], ids=["native", "custom"])
def test_sparse_exception_updates_preserve_unmodified_rows(
        plugin_dir, monkeypatch, capfd, custom):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    monkeypatch.setenv("CUPO_DSPACE_AUTOTUNE", "0")
    # Many independent exception rows, including a final incomplete block.
    # The assertions do not depend on the implementation's comparison size.
    rows = 513
    xyz = .25+.65*np.indices((9, 9, 9)).reshape(3, -1).T[:rows+1]
    base = mm.NonbondedForce()
    base.setNonbondedMethod(base.PME)
    base.setCutoffDistance(1.1)
    base.setUseDispersionCorrection(False)
    for name in ("first", "middle", "last", "span"):
        base.addGlobalParameter(name, 0.)
    for _ in xyz:
        base.addParticle(0., .25, 0.)
    for row in range(rows):
        base.addException(0, row+1, 0. if row == 9 else (-1)**row*.002,
                          .25, .02)
    for name, row in (("first", 0), ("middle", rows//2), ("last", rows-1),
                      ("span", 0), ("span", rows-1)):
        base.addExceptionParameterOffset(name, row, .01, .02, .005)
    force = nv.NonbondedForce(base, 1.15, nv.NonbondedForce.PME)
    if custom:
        expression = mm.CustomNonbondedForce("0")
        expression.setNonbondedMethod(expression.CutoffPeriodic)
        expression.setCutoffDistance(1.1)
        for _ in xyz:
            expression.addParticle([])
        for row in range(rows):
            expression.addExclusion(0, row+1)
        force.setCutoffInteraction(expression)
    system = mm.System()
    for _ in xyz:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(6.3, 0, 0),
                                       mm.Vec3(0, 6.3, 0), mm.Vec3(0, 0, 6.3))
    system.addForce(force)

    @contextlib.contextmanager
    def make_context():
        integrator = mm.LangevinMiddleIntegrator(0., 1., .001)
        ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
        ctx.setPositions(xyz)
        try:
            yield ctx
        finally:
            del ctx, integrator

    with make_context() as actual:
        _capture_modes(actual, capfd)

        def check_first_output(index):
            capfd.readouterr()
            # Retain the first captured evaluation before creating the oracle.
            modes = (MODES[index % len(MODES)], *(
                mode for mode in MODES if mode != MODES[index % len(MODES)]))
            observed = [fused_snapshot(actual, mode) for mode in modes]
            trace = capfd.readouterr().err
            assert "graph capture #" not in trace, trace
            checkpoint = actual.createCheckpoint()
            # A new workspace receives the complete current logical definition;
            # it cannot reuse any previous partial device-source upload.
            with make_context() as oracle:
                oracle.loadCheckpoint(checkpoint)
                assert oracle.createCheckpoint() == checkpoint
                for mode, saved in zip(modes, observed):
                    expected = fused_snapshot(oracle, mode)
                    if mode[0]:
                        assert saved[0] == expected[0]
                    if mode[1]:
                        np.testing.assert_array_equal(saved[1], expected[1])

        # Local middle updates, unchanged replay, edge updates, and both distant
        # edges changing together must preserve all intervening exception rows.
        for index, (name, value) in enumerate((
                ("middle", .4), ("middle", .4), ("first", .7),
                ("last", .3), ("span", .6), ("middle", 0.))):
            actual.setParameter(name, value)
            check_first_output(index)

        # Public signed-zero edits have identical physics but different source
        # bits. Row 9 has no offsets that could erase its sign by adding +0.
        for index, charge in enumerate((0., -0.), start=6):
            first, second, _, sigma, epsilon = force.getExceptionParameters(9)
            force.setExceptionParameters(9, first, second, charge, sigma, epsilon)
            force.updateParametersInContext(actual)
            check_first_output(index)


@pytest.mark.parametrize("custom", [False, True], ids=["native", "custom"])
def test_live_offsets_recover_after_rejected_updates_and_new_offset_laws(
        plugin_dir, monkeypatch, capfd, custom):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    monkeypatch.setenv("CUPO_DSPACE_AUTOTUNE", "0")
    base, expression = _definitions(True, scenario="abfe")
    force = (_single(base, expression, "PME") if custom else
             nv.NonbondedForce(base, 1.15, nv.NonbondedForce.PME))
    system = _system([force])
    with _context(system, "CUDA") as actual:
        _capture_modes(actual, capfd)

        update_index = 0

        def check_first_outputs():
            nonlocal update_index
            first = MODES[update_index % len(MODES)]
            modes = (first, *(mode for mode in MODES if mode != first))
            update_index += 1
            capfd.readouterr()
            observed = [fused_snapshot(actual, mode) for mode in modes]
            assert "graph capture #" not in capfd.readouterr().err
            checkpoint = actual.createCheckpoint()
            with _context(system, "CUDA") as oracle:
                oracle.loadCheckpoint(checkpoint)
                assert oracle.createCheckpoint() == checkpoint
                for mode, saved in zip(modes, observed):
                    expected = fused_snapshot(oracle, mode)
                    if mode[0]:
                        assert saved[0] == expected[0]
                    if mode[1]:
                        np.testing.assert_array_equal(saved[1], expected[1])

        # Populate both alternating states before rejecting a later proposal.
        for value in (.47, .83):
            actual.setParameter("lambda_q", value)
            check_first_outputs()
        for name, bad, restored in (("lambda_q", float("nan"), .61),
                                    ("lambda_14", float("inf"), .42),
                                    ("lambda_q", -float("inf"), .23),
                                    ("lambda_s", -5., .27),
                                    ("lambda_14", -5., .34)):
            actual.setParameter(name, bad)
            for _ in range(2):
                with pytest.raises(Exception, match="finite"):
                    fused_snapshot(actual, MODES[-1])
            # A new valid value forces another update; restoring the active
            # value alone would skip preparation and miss stale proposal state.
            actual.setParameter(name, restored)
            check_first_outputs()

        name, atom, charge, sigma, epsilon = force.getParticleParameterOffset(0)
        force.setParticleParameterOffset(0, name, atom, charge*1.7, sigma, epsilon)
        force.updateParametersInContext(actual)
        actual.setParameter(name, .74)
        check_first_outputs()
        name, row, charge, sigma, epsilon = force.getExceptionParameterOffset(0)
        force.setExceptionParameterOffset(0, name, row, charge*1.3, sigma, epsilon)
        force.updateParametersInContext(actual)
        actual.setParameter(name, .58)
        check_first_outputs()


def test_offset_defaults_validate_only_referenced_globals(plugin_dir, monkeypatch):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    monkeypatch.setenv("CUPO_DSPACE_AUTOTUNE", "0")
    # Both particle and exception laws must reject nonfinite initial values,
    # before any live Context update or captured evaluation has happened.
    for name, value in (("lambda_q", float("nan")), ("lambda_14", float("inf"))):
        base, _ = _definitions(True, scenario="abfe")
        index = next(i for i in range(base.getNumGlobalParameters())
                     if base.getGlobalParameterName(i) == name)
        base.setGlobalParameterDefaultValue(index, value)
        with pytest.raises(Exception, match="offset parameter value must be finite"):
            system = _system([nv.NonbondedForce(base, 1.15, nv.NonbondedForce.PME)])
            with _context(system, "CUDA"):
                pass

    # Declaring a global does not make it an offset dependency. Its value must
    # remain irrelevant at initialization and during another global's update.
    base, _ = _definitions(True, scenario="abfe")
    base.addGlobalParameter("unused", float("nan"))
    system = _system([nv.NonbondedForce(base, 1.15, nv.NonbondedForce.PME)])
    with _context(system, "CUDA") as actual:
        initial = fused_snapshot(actual, MODES[-1])
        actual.setParameter("lambda_q", .61)
        expected = fused_snapshot(actual, MODES[-1])
        assert np.isfinite(expected[0]) and np.all(np.isfinite(expected[1]))
        assert expected[0] != initial[0]
        for value in (float("inf"), -float("inf"), float("nan")):
            actual.setParameter("unused", value)
            observed = fused_snapshot(actual, MODES[-1])
            assert observed[0] == expected[0]
            np.testing.assert_array_equal(observed[1], expected[1])


@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("custom", [False, True], ids=["native", "custom"])
def test_live_exception_fields_reject_overflow_transactionally(
        plugin_dir, monkeypatch, capfd, method, custom):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    monkeypatch.setenv("CUPO_DSPACE_AUTOTUNE", "0")
    base, expression = _definitions(True, scenario="abfe")
    base.addGlobalParameter("particle_only", 0.)
    base.addGlobalParameter("exception_only", 0.)
    base.addExceptionParameterOffset("exception_only", 0, 0., 0., 1.)
    # Seven offset laws (including repeated indices) must take the sparse path.
    # Fixed rows retain independent nonzero values across each live update.
    for i, first in enumerate((20, 22, 24, 26, 28)):
        base.addException(first, first+1, .004*(i+1), .28+.003*i, .05+.01*i)
        expression.addExclusion(first+1, first)
    assert base.getNumExceptionParameterOffsets() < base.getNumExceptions()
    # The exception (0, 4) retains independent qprod/sigma/epsilon. This new
    # global changes both ordinary endpoint charges and no exception law.
    for atom in (0, 4):
        base.addParticleParameterOffset("particle_only", atom, 1., 0., 0.)
    force = (_single(base, expression, method) if custom else
             nv.NonbondedForce(base, 1.15, getattr(nv.NonbondedForce, method)))
    system = _system([force])
    with _context(system, "CUDA") as actual:
        _capture_modes(actual, capfd)

        def check_first_outputs(value):
            actual.setParameter("particle_only", value)
            capfd.readouterr()
            observed = [fused_snapshot(actual, mode) for mode in MODES]
            assert "graph capture #" not in capfd.readouterr().err
            checkpoint = actual.createCheckpoint()
            # Construct canonical rows through full validation at this target
            # lambda. Restoring coordinates/order then leaves the watched value
            # unchanged, so the oracle cannot derive qprod through the fast path.
            # Clone the stock source and retain all current independent edits;
            # this oracle does not depend on plugin-force serialization support.
            oracle_base = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(base))
            for k in range(force.getNumExceptions()):
                oracle_base.setExceptionParameters(k, *force.getExceptionParameters(k))
            for i in range(oracle_base.getNumGlobalParameters()):
                name = oracle_base.getGlobalParameterName(i)
                oracle_base.setGlobalParameterDefaultValue(i, actual.getParameter(name))
            oracle_force = (_single(oracle_base, expression, method) if custom else
                            nv.NonbondedForce(oracle_base, 1.15, getattr(nv.NonbondedForce, method)))
            oracle_system = _system([oracle_force])
            with _context(oracle_system, "CUDA") as oracle:
                oracle.loadCheckpoint(checkpoint)
                for mode, saved in zip(MODES, observed):
                    expected = fused_snapshot(oracle, mode)
                    if mode[0]:
                        assert saved[0] == expected[0]
                    if mode[1]:
                        np.testing.assert_array_equal(saved[1], expected[1])

        check_first_outputs(.03)
        for value in (2.e19, -2.e19):
            actual.setParameter("particle_only", value)
            # Each charge is finite in FP32, but their product is not. Reject
            # twice, then recover at a NEW valid value to exercise preparation.
            for _ in range(2):
                with pytest.raises(Exception, match="excluded weight products must be finite FP32"):
                    fused_snapshot(actual, MODES[-1])
            check_first_outputs(.04 if value > 0 else -.05)

        # Both globals affect only independent exception fields here. The
        # latter keeps epsilon itself finite in FP32 while 4*epsilon overflows.
        # Rows have multiple offset laws; sparse refresh must use their final sum.
        for name, bad, restored in (("lambda_14", 1.e40, .42),
                                    ("exception_only", 1.e38, .02)):
            actual.setParameter(name, bad)
            for _ in range(2):
                with pytest.raises(Exception, match="exception parameters must remain finite in FP32"):
                    fused_snapshot(actual, MODES[-1])
            actual.setParameter(name, restored)
            check_first_outputs(-.05)

        # An explicit independent-exception edit must still take full validation,
        # even when a particle-only update is pending.
        old = force.getExceptionParameters(0)
        actual.setParameter("particle_only", .06)
        force.setExceptionParameters(0, *old[:4], -1.)
        with pytest.raises(Exception, match="effective exception parameters must be finite"):
            force.updateParametersInContext(actual)
        force.setExceptionParameters(0, *old)
        check_first_outputs(.07)

        # A valid independent edit refreshes the canonical rows subsequently used
        # by particle-only proposals, including signed-zero source representations.
        force.setExceptionParameters(0, *old[:2], -0., old[3], old[4])
        force.updateParametersInContext(actual)
        check_first_outputs(-.08)
