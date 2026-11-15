# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mid-run atom reorder support for the NvidiaForce plugin.

OpenMM reorders periodically, or earlier when its neighbor list requests it, only
when a force registers a cutoff interaction. NvNonbondedForce registers none, so
these tests add an inert cutoff NonbondedForce to exercise the reorder listener.

The tests check that the cell list absorbs reordered positions, exclusions and
parameter updates remain valid, and CUDA graphs need no recapture. The uniform
water cases leave per-slot parameters unchanged when interchangeable molecules
move. A nonuniform live charge update additionally tests the association of
particle parameters and exclusion corrections with system atom indices.

Force and energy comparisons use a second plugin context that is not advanced by
the integrator. Exact equality at identical normalized positions is the measured
invariant for this fixture; absolute accuracy is tested in test_plugin_vs_stock.py.
The reference's initial setPositions can also reorder, so it is not assumed to
retain system atom order.

Mixed precision stores coordinates as fp32 posq plus fp32 posqCorrection. A
getPositions/setPositions round-trip re-splits them and can change the fp32 input
used by force kernels. Both contexts must therefore receive the same snapshot.
The setter can itself reorder and wrap molecules in fp32 after splitting them.
On Blackwell this changed 5,626 coordinate components at the second snapshot,
producing a 3.93e-6 force residual because the contexts evaluated different inputs.

_set_comparison_positions checks the exact expected mixed-precision round-trip.
If wrapping changed it, one repeat of the SAME input restores the coordinates
after the reorder counter resets. This verifies the comparison's coordinate
precondition while retaining exact force/energy and no-recapture assertions.

The complementary immediate-post-reorder tests do not normalize or reset the
tested context after a reorder. They evaluate immediately, then checkpoint its
unchanged coordinates and restore them into a fresh plugin context. OpenMM checkpoints
preserve raw posq, posqCorrection, atom order and periodic cell offsets, avoiding
the lossy positions round-trip. That fresh context builds its cuPo workspace from
scratch, independently of the tested context's live cell list and captured graph.

A separate heterogeneous particle-permutation fixture makes missing particle or
exception remaps observable, with an intentionally omitted remap for each field
as a sensitivity control. It exercises the native arithmetic and geometric LJ
policies without depending on OpenMM treating unlike molecules as equivalent.
"""
import os

import numpy as np
import pytest



import openmm as mm
import openmm.unit as unit
import openmm.app as app

from conftest import record_observed

KJ = unit.kilojoule_per_mole
NM = unit.nanometer
NMPS = KJ / NM
CUTOFF = 0.9
# 3.7 nm -> ncx 4 (above the 3-cell stencil floor) with 0.925 nm cells. Must NOT
# be an exact multiple of the cutoff: cells exactly AT the cutoff leave no rebuild
# room and CellList::compute_envelope rejects that outright.
BOX0 = 3.7
_SEED = 20260801
# OpenMM's periodic reorder interval; neighbor-list growth can force earlier ones.
_REORDER_PERIOD = 250
# No tolerances here on purpose: comparison inputs are either normalized on both
# sides or transferred unchanged by checkpoint, as described above.


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    # Exact comparisons between independent contexts require the same FFT
    # algorithm. Setup timings can otherwise pick different backends even for
    # identical meshes. Keep automatic compression, but disable timing selection
    # for this reorder invariant; selector/graph tests cover timing-enabled use.
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    saved = {k: os.environ.get(k) for k in ("CUPO_PLUGIN_CUDA_GRAPH",
                                            "CUPO_PLUGIN_GRAPH_TRACE")}
    yield
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _water_system():
    """A plain TIP3P water box.

    Salt water was tried first, on the theory that dissolved ions would make a
    molecule permutation actually reshuffle the per-atom data. It does not -- ions
    form their own molecule group and are permuted only among themselves, so the
    parameter array and exclusion-pair set stay invariant exactly as in pure water.
    Since the heterogeneity bought no coverage, this stays on the cheaper system.
    """
    import random
    ff = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    random.seed(_SEED)
    np.random.seed(_SEED)
    modeller.addSolvent(ff, model="tip3p",
                        boxSize=mm.Vec3(BOX0, BOX0, BOX0) * NM)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUTOFF * NM,
                             constraints=app.HBonds, rigidWater=True)
    return system, np.array(modeller.positions.value_in_unit(NM))


def _add_inert_cutoff_force(system):
    """Add a numerically inert NonbondedForce that still uses a cutoff.

    This is the whole trick for testing reorder: OpenMM will not permute atoms
    unless getUseCutoff() is true on the context's NonbondedUtilities, which needs
    a force that registers a cutoff interaction. Zeroing every particle's charge,
    sigma and epsilon makes the force contribute exactly nothing, so it changes no
    energy or force -- it exists purely to flip that gate.

    Only ever added to a PLUGIN system, whose source NonbondedForce
    has already been removed -- OpenMM refuses to build a Context
    holding two NonbondedForces with different exception lists.

    The unreordered reference gets one too, because it is built by the same
    _plugin_context(). That is deliberate: the reference must differ from the
    context under test in atom ORDER and nothing else, and a System with one fewer
    force is not the same System. It contributes exactly zero to energy and
    force. Reference setters can reorder too; comparison inputs are normalized
    and checked after those calls.
    """
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(CUTOFF * NM)
    for _ in range(system.getNumParticles()):
        nb.addParticle(0.0, 0.1, 0.0)   # q=0, sigma arbitrary, eps=0 -> no force
    system.addForce(nb)
    return nb


def _plugin_context(graph=True, precision="mixed", inert_cutoff=True, q_scale=1.0,
                    initialize_positions=True):
    """Return the water context, integrator, positions and retained plugin force.

    q_scale scales every charge before the Context is made.
    """
    import nv_openmm_plugin
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1" if graph else "0"
    system, pos = _water_system()
    source = _openmm_nonbonded(system)
    if q_scale != 1.0:
        _scale_charges(source, q_scale)
    force = nv_openmm_plugin.NonbondedForce(source, 1.0)
    system.removeForce(next(i for i in range(system.getNumForces())
                            if isinstance(system.getForce(i), mm.NonbondedForce)))
    system.addForce(force)
    if inert_cutoff:
        _add_inert_cutoff_force(system)
    for _i, _f in enumerate(system.getForces()):
        if isinstance(_f, mm.CMMotionRemover):
            system.removeForce(_i)
            break
    integ = nv_openmm_plugin.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integ.setConstraintTolerance(1e-5)
    integ.setRandomNumberSeed(4242)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": precision})
    if initialize_positions:
        ctx.setPositions(pos * NM)
    return ctx, integ, pos, force


def _openmm_nonbonded(system):
    """The OpenMM NonbondedForce in system, before replace_nonbonded() converts it."""
    return next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))


def _scale_charges(force, q_scale):
    for i in range(force.getNumParticles()):
        q, sig, eps = force.getParticleParameters(i)
        force.setParticleParameters(i, q * q_scale, sig, eps)


def _unreordered_reference(graph=True, precision="mixed", q_scale=1.0):
    """A second plugin context to compare a reordered one against.

    Identical System and precision, without the repeated integration/reordering
    history. setPositions can itself reorder, so comparison setters verify the
    coordinates rather than assuming they survive that call unchanged.

    CREATE THIS AFTER the graph-trace assertions have consumed capfd. It is a
    real plugin context and announces its own captures on the same stream; built
    inside the measured window it would be counted against the context actually
    being tested, and a "reorder recaptured the graph" failure would be this
    reference starting up.
    """
    ctx, _integ, _pos, _force = _plugin_context(graph=graph, precision=precision,
                                        q_scale=q_scale)
    return ctx


def _forces(ctx):
    return np.array(ctx.getState(getForces=True).getForces().value_in_unit(NMPS))


def _positions(ctx):
    return np.array(ctx.getState(getPositions=True).getPositions().value_in_unit(NM))


def _set_comparison_positions(ctx, positions):
    """Set mixed-precision coordinates, checking for a setter-triggered reorder.

    OpenMM stores float32(p) plus float32(p-float32(p)). Its setter then calls
    reorderAtoms(), which can wrap molecules and round their coordinates again.
    Compare against that split's exact round-trip, accounting for ordinary
    correction rounding without a tolerance. If wrapping changed it, one repeat
    of the same input restores the coordinates after the reorder counter resets.
    """
    high = positions.astype(np.float32).astype(np.float64)
    expected = high + (positions-high).astype(np.float32).astype(np.float64)
    for _ in range(2):
        ctx.setPositions(positions * NM)
        actual = _positions(ctx)
        if np.array_equal(actual, expected):
            return
    assert np.array_equal(actual, expected), (
        "setPositions did not preserve the expected mixed-precision coordinates "
        f"after retrying: max error={np.max(np.abs(actual-expected)):.3e} nm")


def _energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


def _erel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _frmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def _outputs(ctx, energy, forces):
    state = ctx.getState(getEnergy=energy, getForces=forces)
    e = state.getPotentialEnergy().value_in_unit(KJ) if energy else None
    f = np.array(state.getForces().value_in_unit(NMPS)) if forces else None
    return e, f


def _trigger_reorder(ctx, integrator, capfd, trigger):
    """Return immediately after the listener fires, before another evaluation."""
    requested = None
    if trigger == "setter":
        # Reverse whole waters to change spatial order, then translate outside the
        # primary box to exercise the setter's fp32 wrapping of their coordinates.
        requested = _positions(ctx).reshape(-1, 3, 3)[::-1].reshape(-1, 3).copy()
        requested += np.array([BOX0, -BOX0, BOX0])
    trace = []
    for _ in range(_REORDER_PERIOD + 1):
        if trigger == "integrator":
            integrator.step(1)  # reorderAtoms is the final operation of the step.
        else:
            ctx.setPositions(requested * NM)
        err = capfd.readouterr().err
        trace.append(err)
        if "atom reorder #" in err:
            return "".join(trace), requested
    pytest.fail(f"no {trigger}-triggered reorder within 251 calls:\n{''.join(trace)}")


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("trigger", ["integrator", "setter"])
def test_first_evaluation_after_reorder(request, plugin_dir, capfd, graph, trigger):
    """The first evaluation must use the actual reordered state without repair.

    Each output combination is first after its own observed reorder. A fresh eager
    context restored from the checkpoint is the oracle, so no persistent cuPo cell
    list or graph is shared. In particular, never call setPositions on the tested
    context between the observed reorder and its measured evaluation.
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, _force = _plugin_context(graph=graph)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)
    modes = ((False, True), (True, False), (True, True))
    for energy, forces in modes:
        for _ in range(3):
            _outputs(pctx, energy, forces)
    warm = capfd.readouterr().err
    if graph:
        for energy, forces in modes:
            assert f"(energy={int(energy)} forces={int(forces)})" in warm, (
                f"output mode did not capture a graph during warmup:\n{warm}")

    snapshots = []
    trace = []
    for energy, forces in modes:
        reorder_trace, requested = _trigger_reorder(pctx, pi, capfd, trigger)
        trace.append(reorder_trace)
        # No checkpoint, position readback or setter before this first evaluation:
        # an extra device synchronization could hide an ordering problem.
        e, f = _outputs(pctx, energy, forces)
        evaluation_trace = capfd.readouterr().err
        assert "atom reorder #" not in evaluation_trace, evaluation_trace
        trace.append(evaluation_trace)
        # Force/energy evaluation does not move coordinates. Snapshot afterwards
        # to give the oracle precisely those raw inputs without synchronizing
        # the tested context before its first evaluation.
        checkpoint = pctx.createCheckpoint()
        if requested is not None:
            high = requested.astype(np.float32).astype(np.float64)
            split = high + (requested-high).astype(np.float32).astype(np.float64)
            assert not np.array_equal(_positions(pctx), split), (
                "the setter did not exercise coordinate rounding during wrapping")
        assert e is None or np.isfinite(e), "non-finite energy immediately after reorder"
        assert f is None or np.all(np.isfinite(f)), "non-finite forces immediately after reorder"
        snapshots.append((checkpoint, energy, forces, e, f))
    err = "".join(trace)
    assert err.count("atom reorder #") >= len(modes), err
    assert "graph capture #" not in err, f"a reorder recaptured the graph:\n{err}"

    # Construct references only after the measured trace window. Do not evaluate
    # or set positions first: first execute must build an independent workspace
    # from the checkpoint's restored atom order, rather than remap an existing one.
    worst_e = worst_f = 0.0
    for checkpoint, energy, forces, e, f in snapshots:
        rctx, ri, _, _force = _plugin_context(graph=False, initialize_positions=False)
        rctx.loadCheckpoint(checkpoint)
        assert rctx.createCheckpoint() == checkpoint, "oracle changed raw coordinates/order"
        want_e, want_f = _outputs(rctx, energy, forces)
        if forces:
            worst_f = max(worst_f, _frmse(f, want_f))
            assert np.array_equal(f, want_f), (
                f"first forces after {trigger} reorder differ from fresh workspace: "
                f"relRMSE={_frmse(f, want_f):.2e}")
        if energy:
            worst_e = max(worst_e, _erel(e, want_e))
            assert e == want_e, (
                f"first energy after {trigger} reorder differs from fresh workspace: "
                f"{e!r} vs {want_e!r}")
        del rctx, ri
    tag = f"{trigger}, {'graph' if graph else 'eager'}"
    record_observed(request, f"first reordered energy vs fresh ({tag})",
                    0.0, worst_e)
    record_observed(request, f"first reordered force RMSE vs fresh ({tag})",
                    0.0, worst_f)


def test_inert_cutoff_force_triggers_a_reorder(plugin_dir, capfd):
    """The premise every other test here rests on: that this setup reorders at all.

    Asserted separately and first, because a reorder test that never provokes a
    reorder passes vacuously, and OpenMM exposes no signal for one -- hence the
    plugin's own trace counter.
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, _force = _plugin_context()
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    capfd.readouterr()
    pi.step(_REORDER_PERIOD + 60)
    err = capfd.readouterr().err
    assert err.count("atom reorder #") >= 1, (
        "no reorder fired in >250 steps -- the inert cutoff force is not flipping "
        f"getUseCutoff(), so the reorder path is untested:\n{err}")


def test_no_reorder_without_a_cutoff_force(plugin_dir, capfd):
    """Control for the test above: with NvNonbondedForce as the only nonbonded force, OpenMM
    registers no cutoff interaction and must never reorder. This is what makes the
    positive test meaningful -- it shows the trace tracks the reorder rather than
    firing unconditionally."""
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, _force = _plugin_context(inert_cutoff=False)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    capfd.readouterr()
    pi.step(_REORDER_PERIOD + 60)
    err = capfd.readouterr().err
    assert err.count("atom reorder #") == 0, (
        f"reordered without any cutoff-using force:\n{err}")


@pytest.mark.parametrize("graph", [True, False])
def test_forces_survive_a_reorder(request, plugin_dir, capfd, graph):
    """Forces and energy must be UNCHANGED, bit for bit, by repeated reorders.

    The substance here is the cell list: a reorder moves every atom to a different
    GPU slot, so from the cell list's side every atom teleports at once. Run with
    the CUDA graph both on and off; the graph case additionally asserts no
    recapture, since the per-atom buffers are held by pointer and only their
    contents can change.

    The comparison is against a non-stepped second plugin context at the same
    normalized positions, and is exact -- see the module docstring for its scope
    (and for the one thing it costs: this says a reorder does not move the answer,
    not that the answer is right).
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, _force = _plugin_context(graph=graph)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                       # warm up (and capture, when graph=True)
    # There is one graph per (energy, forces) combination, so warming up has to
    # cover every combination the measured window below uses -- stepping alone
    # only warms force-only, and the first getState(energy) would then capture
    # inside the window and look like a reorder-induced recapture. Three calls:
    # a combination captures on its third run (see GraphSlot::seen).
    for _ in range(3):
        _forces(pctx)
        _energy(pctx)
    capfd.readouterr()

    # Snapshot inside the traced window, compare outside it: the reference is
    # another plugin context and would otherwise be caught by the capture count.
    #
    # Normalize posq/posqCorrection and verify the setter did not wrap positions
    # during a reorder. Both contexts must evaluate the same requested snapshot.
    snapshots = []
    for _ in range(3):                # >=3 reorders
        pi.step(_REORDER_PERIOD + 20)
        p = _positions(pctx)
        _set_comparison_positions(pctx, p)
        f_p, e_p = _forces(pctx), _energy(pctx)
        assert np.all(np.isfinite(f_p)), "non-finite forces after a reorder"
        snapshots.append((p, e_p, f_p))
    err = capfd.readouterr().err

    n_reorder = err.count("atom reorder #")
    assert n_reorder >= 3, f"expected >=3 reorders, saw {n_reorder}:\n{err}"
    if graph:
        assert err.count("graph capture #") == 0, (
            f"a reorder recaptured the graph ({err.count('graph capture #')}x); "
            "the per-atom buffers are held by pointer, so rewriting their contents "
            f"should suffice:\n{err}")

    rctx = _unreordered_reference(graph=graph)
    tag = "graph" if graph else "eager"
    worst_e = worst_f = 0.0
    for k, (p, e_p, f_p) in enumerate(snapshots):
        _set_comparison_positions(rctx, p)
        f_r, e_r = _forces(rctx), _energy(rctx)
        # Residuals recorded even though the assertion is equality: if this ever
        # stops being bitwise, the size of the miss is what says whether it is a
        # reordering bug or the posq split creeping back in (~3.7e-6 / 5.9e-8).
        worst_e = max(worst_e, _erel(e_p, e_r))
        worst_f = max(worst_f, _frmse(f_p, f_r))
        assert np.array_equal(f_p, f_r), (
            f"reorder #{k} moved the forces ({tag}): a reordered context and a "
            f"non-stepped one at identical positions must agree bit for bit, "
            f"relRMSE={_frmse(f_p, f_r):.2e}. If this is ~3.7e-6, the two sides "
            f"are on different posq splits rather than reordering wrongly.")
        assert e_p == e_r, (
            f"reorder #{k} moved the energy ({tag}): {e_p!r} vs {e_r!r}, "
            f"rel={_erel(e_p, e_r):.2e}")

    record_observed(request, f"reorder energy vs unreordered plugin ({tag})",
                    0.0, worst_e)
    record_observed(request, f"reorder force RMSE vs unreordered plugin ({tag})",
                    0.0, worst_f)


def test_exclusions_survive_a_reorder(request, plugin_dir, capfd):
    """Exclusions must still be applied correctly after a reorder.

    A wrong exclusion list is silent -- it skips the wrong pairs and shows up only
    as a force error -- so it is worth asserting separately even though, per the
    module docstring, the pair SET provably survives a reorder untouched and this
    would also pass against an empty handler. What it genuinely guards is the
    combination: exclusions are applied to a cell decomposition that was rebuilt
    from scratch under the reorder.
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, _force = _plugin_context()
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)
    capfd.readouterr()

    # setPositions on both sides normalises the posq split; see the module docstring.
    p_before = _positions(pctx)
    _set_comparison_positions(pctx, p_before)
    f_before = _forces(pctx)

    pi.step(_REORDER_PERIOD + 20)
    err = capfd.readouterr().err
    assert err.count("atom reorder #") >= 1, f"no reorder fired:\n{err}"

    p_after = _positions(pctx)
    _set_comparison_positions(pctx, p_after)
    f_after = _forces(pctx)

    # Reference built after the trace assertion, for the reason in
    # _unreordered_reference. Both configurations go through it, so the
    # before/after comparison is like-for-like.
    rctx = _unreordered_reference()
    _set_comparison_positions(rctx, p_before)
    rmse_before = _frmse(f_before, _forces(rctx))
    _set_comparison_positions(rctx, p_after)
    rmse_after = _frmse(f_after, _forces(rctx))

    record_observed(request, "reorder exclusion-pair force RMSE vs unreordered plugin",
                    0.0, rmse_after)
    # Bitwise BEFORE the reorder as well as after. A mis-permuted exclusion list
    # would drop or add whole pair interactions -- orders of magnitude above this
    # -- but there is no reason to accept any movement at all, so none is allowed.
    assert rmse_before == 0.0, (
        f"exclusions already disagreed BEFORE any reorder: F_rmse={rmse_before:.2e}. "
        f"That is a plain evaluation difference, not a reordering one.")
    assert rmse_after == 0.0, (
        f"a reorder moved the exclusion-corrected forces: F_rmse={rmse_after:.2e} "
        f"(was {rmse_before:.2e} before it). If this is ~3.7e-6 the two sides are "
        f"on different posq splits; anything larger is a mis-permuted pair list.")


def test_an_update_survives_a_reorder(request, plugin_dir, capfd):
    """Parameters changed with updateParametersInContext must still be the ones in
    use after OpenMM reorders the atoms.

    A reorder re-derives the device's per-atom array from the system-order copy the
    kernel keeps. The update used to change only the device array, so the next
    reorder quietly put the old parameters back.
    """
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1"
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, force = _plugin_context()
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)
    _scale_charges(force, 0.5)
    force.updateParametersInContext(pctx)
    capfd.readouterr()

    pi.step(_REORDER_PERIOD + 20)
    err = capfd.readouterr().err
    assert err.count("atom reorder #") >= 1, f"no reorder fired:\n{err}"

    p_after = _positions(pctx)
    _set_comparison_positions(pctx, p_after)
    after = _energy(pctx)

    # The same halved charges, given at construction, without integration history.
    rctx = _unreordered_reference(q_scale=0.5)
    _set_comparison_positions(rctx, p_after)
    want = _energy(rctx)

    record_observed(request, "updated parameters after a reorder, energy vs unreordered",
                    0.0, _erel(after, want))
    assert _erel(after, want) < 1e-6, (
        f"after a reorder the updated context gives {after:.4f} kJ/mol, the same "
        f"parameters without a reorder {want:.4f}: the reorder put old parameters back")


def _vary_water_charges(force):
    """Keep each water neutral while making its charge scale system-index dependent."""
    for i in range(force.getNumParticles()):
        q, sig, eps = force.getParticleParameters(i)
        scale = 0.75 + 0.05*((i//3) % 5)
        force.setParticleParameters(i, q*scale, sig, eps)


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_nonuniform_update_survives_reorder(request, plugin_dir, capfd, graph):
    """Unequal water charges expose per-slot parameter and exclusion associations.

    Apply the update after the live workspace has already reordered, then exercise
    another real reorder. The fresh oracle receives the same update before its
    first evaluation, so it constructs the final parameters/exclusions directly.
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _, force = _plugin_context(graph=graph)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)
    for _ in range(3):
        _outputs(pctx, True, True)
    warm = capfd.readouterr().err
    if graph:
        assert "(energy=1 forces=1)" in warm, warm
    _trigger_reorder(pctx, pi, capfd, "integrator")
    _vary_water_charges(force)
    force.updateParametersInContext(pctx)
    update_trace = capfd.readouterr().err

    # Check both the update under the existing order and its survival through a
    # subsequent reorder, without a coordinate reset after either operation.
    snapshots = []
    trace = [update_trace]
    for reorder in (False, True):
        if reorder:
            reorder_trace, _ = _trigger_reorder(pctx, pi, capfd, "setter")
            trace.append(reorder_trace)
        e, f = _outputs(pctx, True, True)
        evaluation_trace = capfd.readouterr().err
        assert "atom reorder #" not in evaluation_trace, evaluation_trace
        trace.append(evaluation_trace)
        snapshots.append((pctx.createCheckpoint(), _positions(pctx), e, f))
    assert "graph capture #" not in "".join(trace), "parameter update/reorder recaptured graph"

    worst_e = worst_f = 0.0
    for index, (checkpoint, positions, e, f) in enumerate(snapshots):
        rctx, ri, _, reference_force = _plugin_context(
            graph=False, initialize_positions=False)
        _vary_water_charges(reference_force)
        reference_force.updateParametersInContext(rctx)
        rctx.loadCheckpoint(checkpoint)
        assert rctx.createCheckpoint() == checkpoint, (
            "checkpoint loading changed the raw state used by the comparison")
        assert np.array_equal(_positions(rctx), positions), (
            "checkpoint loading changed the represented coordinates")
        want_e, want_f = _outputs(rctx, True, True)
        worst_e = max(worst_e, _erel(e, want_e))
        worst_f = max(worst_f, _frmse(f, want_f))
        assert np.array_equal(f, want_f), (
            f"nonuniform charge update {'after reorder' if index else 'before reorder'}: "
            f"force relRMSE={_frmse(f, want_f):.2e}")
        assert e == want_e, (
            f"nonuniform charge update {'after reorder' if index else 'before reorder'}: "
            f"energy {e!r} vs {want_e!r}")
        del rctx, ri
    tag = "graph" if graph else "eager"
    record_observed(request, f"nonuniform update/reorder energy ({tag})", 0.0, worst_e)
    record_observed(request, f"nonuniform update/reorder force RMSE ({tag})", 0.0, worst_f)


def _exception_pair_context(graph, initialize_positions=True, periodic=True, replacements=False):
    """Neutral pairs whose only molecule edges are nonbonded exceptions."""
    import nv_openmm_plugin as nv
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1" if graph else "0"
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(BOX0, 0, 0), mm.Vec3(0, BOX0, 0), mm.Vec3(0, 0, BOX0))
    source = mm.NonbondedForce()
    source.setNonbondedMethod(mm.NonbondedForce.PME)
    source.setCutoffDistance(CUTOFF)
    source.setUseDispersionCorrection(False)
    source.setExceptionsUsePeriodicBoundaryConditions(periodic)
    centers = (np.indices((4, 4, 4)).reshape(3, -1).T + 0.5)*(BOX0/4)
    positions = []
    for k, center in enumerate(centers):
        offset = np.zeros(3)
        offset[k % 3] = 0.08
        positions.extend((center-offset, center+offset))
        for charge in (0.3, -0.3):
            system.addParticle(18)
            source.addParticle(charge, 0.2, 0.05)
        source.addException(2*k, 2*k+1, .003 if replacements else 0.,
                            .2, .015 if replacements else 0.)
    system.addForce(nv.NonbondedForce(source, 1.0))
    _add_inert_cutoff_force(system)
    integrator = mm.VerletIntegrator(0.001)
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": "mixed"})
    positions = np.asarray(positions)
    if initialize_positions:
        ctx.setPositions(positions*NM)
    return ctx, integrator, positions


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("periodic,replacements", [(True, False), (False, True)],
                         ids=["periodic-exclusions", "raw-replacements"])
def test_exception_only_molecules_survive_reorder(plugin_dir, capfd, graph, periodic, replacements):
    """Exceptions must keep pairs together even without bonds or constraints."""
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, positions = _exception_pair_context(
        graph, periodic=periodic, replacements=replacements)
    # Unlike the water cases, no integration/minimization has consumed the first
    # 50 eager warmup evaluations before the output-mode graph can be captured.
    for _ in range(53):
        _outputs(pctx, True, True)
    warm = capfd.readouterr().err
    if graph:
        assert "graph capture #" in warm, warm

    # Move the two charge classes independently. No coordinates overlap, but the
    # excluded partners now span different cells; sorting singleton atoms would
    # change the GPU pair set, whereas sorting exception-connected pairs cannot.
    rng = np.random.default_rng(_SEED)
    requested = positions.copy()
    npairs = len(positions)//2
    requested[::2] = positions[::2][rng.permutation(npairs)]
    requested[1::2] = positions[1::2][rng.permutation(npairs)]
    requested += np.array([BOX0, -BOX0, BOX0])
    for _ in range(_REORDER_PERIOD + 1):
        pctx.setPositions(requested*NM)
        reorder_trace = capfd.readouterr().err
        if "atom reorder #" in reorder_trace:
            break
    else:
        pytest.fail("exception-only fixture did not trigger a reorder")
    # Evaluate immediately: no setter or checkpoint after the observed reorder.
    e, f = _outputs(pctx, True, True)
    err = capfd.readouterr().err
    assert "atom reorder #" not in err, err
    assert "graph capture #" not in err, err
    assert np.all(np.isfinite(f)) and np.any(np.abs(f) > 1e-3)
    checkpoint = pctx.createCheckpoint()

    rctx, ri, _ = _exception_pair_context(
        False, initialize_positions=False, periodic=periodic, replacements=replacements)
    rctx.loadCheckpoint(checkpoint)
    assert rctx.createCheckpoint() == checkpoint, "oracle changed raw coordinates/order"
    want_e, want_f = _outputs(rctx, True, True)
    assert np.array_equal(f, want_f), (
        f"exception-only molecules reordered incorrectly: force relRMSE={_frmse(f, want_f):.2e}")
    assert e == want_e, f"exception-only molecules reordered energy: {e!r} vs {want_e!r}"


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("rule", ["LorentzBerthelot", "Geometric"])
def test_particle_permutation_preserves_parameter_and_exception_associations(
        plugin_dir, capfd, graph, rule):
    """Heterogeneous records expose a missing parameter or exception remap.

    Rebuild each workspace for its immutable exception topology. Coordinates are
    copied bit for bit and forces returned to physical particle order. Unlike the
    equivalent-water fixtures, changing indices here changes the per-slot data.
    """
    import nv_openmm_plugin as nv
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1" if graph else "0"
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    rng = np.random.default_rng(4501)
    xyz = np.array([[.1 + .75*x, .1 + .75*y, .1 + .75*z]
                    for x in range(4) for y in range(4) for z in range(4)], np.float32)
    xyz += rng.uniform(-.04, .04, xyz.shape).astype(np.float32)
    particles = np.column_stack((rng.uniform(-.5, .5, len(xyz)),
                                 rng.uniform(.15, .35, len(xyz)),
                                 rng.uniform(.1, .8, len(xyz))))
    excluded = np.array([[0, 1], [7, 23], [15, 3], [63, 47], [40, 44]], np.int32)

    def evaluate(order, *, remap_parameters=True, remap_exclusions=True):
        inverse = np.argsort(order)
        system = mm.System()
        system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0),
                                            mm.Vec3(0, 0, 3.2))
        force = nv.NonbondedForce(nv.NonbondedForce.PME, 1.0, 1.0)
        force.setUseDispersionCorrection(False)
        force.setLJCombiningRule(getattr(nv.NonbondedForce, rule))
        for q, sigma, epsilon in particles[order] if remap_parameters else particles:
            system.addParticle(18)
            force.addParticle(float(q), float(sigma), float(epsilon))
        for first, second in inverse[excluded] if remap_exclusions else excluded:
            force.addException(int(first), int(second), 0.0, .2, 0.0)
        system.addForce(force)
        integrator = mm.VerletIntegrator(.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                             {"Precision": "mixed"})
        try:
            context.setPositions(xyz[order]*NM)
            for _ in range(53 if graph else 1):
                energy, forces = _outputs(context, True, True)
            trace = capfd.readouterr().err
            if graph:
                assert "(energy=1 forces=1)" in trace, trace
            return energy, forces[inverse]
        finally:
            del context, integrator

    expected_energy, expected_forces = evaluate(np.arange(len(xyz)))
    shuffled = rng.permutation(len(xyz))
    for order in (np.arange(len(xyz))[::-1], shuffled):
        energy, forces = evaluate(order)
        # Arbitrary relabeling may alter FP32 pair/reduction ordering. These
        # gates retain native rounding tolerance rather than requiring bitwise
        # equality for unlike molecules. The real reorder tests remain exact.
        np.testing.assert_allclose(energy, expected_energy, rtol=4e-6, atol=4e-6)
        np.testing.assert_allclose(forces, expected_forces, rtol=4e-6, atol=2e-4)

    for field in ("remap_parameters", "remap_exclusions"):
        energy, forces = evaluate(shuffled, **{field: False})
        assert abs(energy-expected_energy) > 1e-3, field
        assert np.max(np.abs(forces-expected_forces)) > 1e-2, field
