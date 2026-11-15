# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPT (variable-box / MonteCarlo barostat) support for the NvidiaForce plugin.

The plugin runs under a CUDA graph. NPT support rests on two mechanisms:

  * box as a graph parameter -- the box lives in a device buffer that every
    kernel reads by pointer, so an in-band barostat move only refreshes d_box
    before replay (no recapture);
  * band-cross resume -- when the box crosses a cell-count boundary the cell grid
    (ncx/ncy/ncz, ncell) is recomputed in place (every buffer and every baked
    scalar pre-sized to the replace_nonbonded(npt_box_safety_factor=...) envelope). The
    captured kernels derive the grid from d_box on-device with persistent
    (ncell-independent) launch dims, so the graph is box-invariant and simply
    keeps replaying through the cross -- no eager step, no re-warm, no recapture.
    The cell list absorbs the regrid on its next ordinary build. A box move that
    compresses past the envelope's occupancy headroom fails hard.

cupo rebuilds the cell list every step (no Verlet skin), so cells are exactly the
cutoff and ncx = floor(box / cutoff). Band boundaries are at multiples of the
0.9 nm cutoff. NPT dynamics use only the energy-only (barostat) and force-only
(integrator) evaluation paths, so energies and forces are read with separate
getState calls (never the combined getState(energy+forces), a separate
pre-existing graph-readback issue).

WHAT THESE COMPARE AGAINST. A second PLUGIN context -- eager, and never given a
box change or a coordinate jump -- evaluated at the same (box, positions). Not
stock OpenMM, which used to serve here and no longer can: this branch decomposes
the Coulomb sum differently (prolate rather than Ewald), so cupo and stock now
disagree at production settings by ~2.5e-5 energy and ~5e-4 force, which is the
two splittings' truncation error and not anyone's mistake. Against a tolerance
sized for a shared splitting that reads as failure.

An unchanged copy of the same implementation says the thing these tests actually
mean: a box change, a band cross, a resume or a coordinate jump must not move the
answer. And because both sides are the same implementation, the comparison is
EXACT -- every assertion here is bitwise, with no tolerance to tune. Measured 0.0
on all seventeen recorded quantities, including a 400-step sustained replay at the
larger grid and a 120-block barostat run across a real down-cross.

That is not circular. test_npt_force_replay_box_param_matches_eager establishes,
independently and without a reference of any kind, that the graph replay path
agrees with an eager evaluation bit for bit; these then spend that fact.

The bitwise assertions are not vacuous, checked by perturbing the reference: a box
wrong by 0.02 nm is caught outright (F_rmse 5e-1), and moving ONE atom is caught
from 4 fp32 ULPs (~5e-7 nm, F_rmse 6.6e-8) upward. A 1-2 ULP move is not caught --
it falls under the force quantisation floor -- so the resolution here is a few
times 1e-7 nm on a single coordinate, not literally one bit. Every failure mode
these tests exist for (a stale grid, an under-cleared buffer, a mis-sized envelope,
a silently rebuilt graph) is orders of magnitude above that.
"""
import os

import numpy as np
import pytest



import openmm as mm
import openmm.unit as unit
import openmm.app as app

from conftest import record_observed, replace_nonbonded

KJ = unit.kilojoule_per_mole
NM = unit.nanometer
NMPS = KJ / NM
CUTOFF = 0.9          # nm; ncx = floor(box / cutoff)
BOX0 = 5.0            # nm -> ncx = 5 (comfortably above the 3-cell floor)
SAFETY = 1.35         # cell-grid pre-allocation envelope (covers ncx up to ~7)
_SEED = 20260723


@pytest.fixture(autouse=True)
def _restore_cuda_graph_env(monkeypatch):
    """The plugin reads these env vars at Context creation; snapshot/restore so a
    test's setting never leaks into another test or module. (The NPT box-safety
    factor is passed explicitly to replace_nonbonded(), not an env var.)"""
    # The bitwise oracle is a separate context. Keep its FFT algorithm identical
    # to the tested context while retaining automatic support/radial masking.
    # Timing-enabled box replay is covered by test_compact_policy.py.
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    keys = ("CUPO_PLUGIN_CUDA_GRAPH", "CUPO_PLUGIN_GRAPH_TRACE")
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _build_water(box0=BOX0):
    import random
    ff = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    random.seed(_SEED)
    np.random.seed(_SEED)
    modeller.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(box0, box0, box0) * NM)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUTOFF * NM, constraints=app.HBonds,
                             rigidWater=True)
    pos = np.array(modeller.positions.value_in_unit(NM))
    return system, pos


def _cube(L):
    return (mm.Vec3(L, 0, 0) * NM, mm.Vec3(0, L, 0) * NM, mm.Vec3(0, 0, L) * NM)


def _ortho(Lx, Ly, Lz):
    return (mm.Vec3(Lx, 0, 0) * NM, mm.Vec3(0, Ly, 0) * NM, mm.Vec3(0, 0, Lz) * NM)


def _plugin_context(graph, safety=SAFETY, barostat_bar=None, precision="mixed"):
    import nv_openmm_plugin
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1" if graph else "0"
    system, pos = _build_water()
    replace_nonbonded(system, npt_box_safety_factor=safety)
    if barostat_bar is not None:
        baro = mm.MonteCarloBarostat(barostat_bar * unit.bar, 300 * unit.kelvin, 20)
        baro.setRandomNumberSeed(777)
        system.addForce(baro)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                        0.002 * unit.picoseconds)
    integ.setRandomNumberSeed(4242)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": precision})
    ctx.setPositions(pos * NM)
    return ctx, integ, pos


def _energy(ctx):
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


def _forces(ctx):
    return np.array(ctx.getState(getForces=True).getForces().value_in_unit(NMPS))


def _positions(ctx):
    return np.array(ctx.getState(getPositions=True).getPositions().value_in_unit(NM))


def _box(ctx):
    return ctx.getState().getPeriodicBoxVectors().value_in_unit(NM)[0][0]


def _erel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0)


def _frmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))


def _plugin_reference(precision="mixed"):
    """A second plugin context to compare a box-changed one against.

    EAGER, and never given a box change or a setPositions jump, so it has taken
    none of the paths these tests exercise -- no capture, no band cross, no
    resume. Pointing it at the same (box, positions) therefore isolates exactly
    the path under test, which is what stock OpenMM used to stand in for and can
    no longer, its splitting no longer being ours.

    That the eager and graph paths agree at all is itself asserted, separately and
    first, by test_npt_force_replay_box_param_matches_eager -- so this is not
    circular: that test earns the reference, and these spend it.

    BUILD IT AFTER the stepping and the capfd assertions are done. It is a real
    plugin context; constructing one rewrites CUPO_PLUGIN_CUDA_GRAPH, and while
    graph=False emits no capture line of its own, keeping it out of the measured
    window removes the question entirely. Every test below snapshots what it needs
    during the run and compares afterwards for that reason.
    """
    ctx, _integ, _pos = _plugin_context(graph=False, precision=precision)
    return ctx


def _reference_at(rctx, box_vectors, positions):
    """Evaluate the reference at an exact (box, positions). NPT has two state
    variables, and pinning only the positions would compare different systems."""
    rctx.setPeriodicBoxVectors(*box_vectors)
    rctx.setPositions(positions * NM)
    return _energy(rctx), _forces(rctx)


def _assert_matches(where, e_p, f_p, e_r, f_r, forces=True, energy=True):
    """Bitwise agreement with the reference -- no tolerance, by measurement.

    Either half can be switched off, and both are used: the anisotropic-compression
    test asserts energy alone (an abrupt compression saturates the int32 force
    accumulator, a documented limit rather than a band-cross fault), while the two
    no-recapture tests assert forces alone (they drive the force-only graph, and
    reading an energy there would capture a second graph and confound the very
    capture count they exist to check).

    The message names the likely cause when it fails, because the two candidates
    look nothing alike: ~3.7e-6 force / ~5.9e-8 energy is the posq/posqCorrection
    re-split that getPositions/setPositions performs (see test_reorder.py, where
    it had to be normalised away), and anything larger is the band-cross or resume
    path actually computing something different.
    """
    if forces:
        assert np.array_equal(f_p, f_r), (
            f"{where}: forces moved vs an unchanged plugin context, "
            f"F_rmse={_frmse(f_p, f_r):.2e}. ~3.7e-6 means the two sides are on "
            f"different posq splits; larger is the path under test.")
    if energy:
        assert e_p == e_r, (
            f"{where}: energy moved vs an unchanged plugin context, {e_p!r} vs "
            f"{e_r!r}, rel={_erel(e_p, e_r):.2e}")


def test_npt_force_replay_box_param_matches_eager(plugin_dir):
    """Genuinely exercise the box-as-graph-parameter REPLAY path. Capture a
    force-only graph (stepping only ever evaluates cf=1,ce=0), then change the box
    IN-BAND (no cell-count change, no setPositions -> stays on the captured graph)
    and read forces -- this hits the replay fast path with the d_box refresh. The
    result must equal an eager evaluation at the same box+positions, bit for bit."""
    # Relax in a separate eager context (minimizing on the graph context would
    # capture the energy+force combo, not the force-only one we want to replay).
    ectx0, _, _ = _plugin_context(graph=False)
    mm.LocalEnergyMinimizer.minimize(ectx0, maxIterations=200)
    relaxed = _positions(ectx0)

    gctx, gi, _ = _plugin_context(graph=True)
    gctx.setPositions(relaxed * NM)
    gi.step(60)                              # force-only; captures at 50, replays after
    L2 = 5.2                                 # in-band nudge (floor(5.2/0.9)==floor(5.0/0.9)==5)
    assert int(L2 / CUTOFF) == int(BOX0 / CUTOFF)
    gctx.setPeriodicBoxVectors(*_cube(L2))   # box only -> UpdateBox 0, replay + d_box refresh
    f_replay = _forces(gctx)
    p = _positions(gctx)

    ectx, _, _ = _plugin_context(graph=False)
    ectx.setPeriodicBoxVectors(*_cube(L2))
    ectx.setPositions(p * NM)
    f_eager = _forces(ectx)

    assert np.array_equal(f_replay, f_eager), (
        f"replay != eager: max|df|={np.abs(f_replay - f_eager).max():.3e}")


@pytest.mark.parametrize("box_L", [5.00, 5.50])   # ncx 5 (baseline), 6 (up-cross at 5.4 nm)
def test_npt_bandcross_accuracy(request, plugin_dir, box_L):
    """Uniformly scale the box up across the ncx 5->6 boundary (5.4 nm) under an
    active graph; the recomputed grid must give energies + forces identical to an
    unchanged plugin context. Only expansion is tested by abrupt scaling: an abrupt
    compression of near-incompressible water produces forces beyond the int32
    accumulator range (a documented limit, not a band-cross issue) -- the
    compression / down-cross path is covered physically by the barostat test."""
    pctx, pi, pos0 = _plugin_context(graph=True)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                              # warm up + capture at BOX0 (ncx 5)

    scaled = pos0 * (box_L / BOX0)           # uniform scale -> physical config
    pctx.setPeriodicBoxVectors(*_cube(box_L))
    pctx.setPositions(scaled * NM)
    pi.step(1)                               # execute -> UpdateBox recompute/replay
    e_p, f_p, p = _energy(pctx), _forces(pctx), _positions(pctx)

    e_r, f_r = _reference_at(_plugin_reference(), _cube(box_L), p)

    ncx = int(box_L / CUTOFF)
    record_observed(request, f"NPT band-cross energy vs unchanged plugin, ncx={ncx}",
                    0.0, _erel(e_p, e_r))
    record_observed(request, f"NPT band-cross force RMSE vs unchanged plugin, ncx={ncx}",
                    0.0, _frmse(f_p, f_r))
    _assert_matches(f"ncx={ncx} L={box_L}", e_p, f_p, e_r, f_r)


def test_npt_expansion_sustained_replay(request, plugin_dir, capfd):
    """Expand across a band cross and then keep REPLAYING for a long run.

    The existing band-cross tests step once or twice after the cross, which only
    exercises the eager host step that follows it. This one stays on the replay
    path for hundreds of steps at the LARGER grid, which is the regime where a
    graph-baked, ncell-derived size goes wrong: every such size is frozen at
    capture time (at the SMALLER grid), and an expansion deliberately keeps the
    graph, so any buffer whose live extent grew is under-covered on every replay
    from then on -- the eager step patches it exactly once and never again.

    That class of bug does not show up in a single post-cross step, and a
    conservatively-gated one (a stale-SET mask flag costs an extra mask read that
    finds zeros) never shows up in the forces at all. What this test does catch is
    the malignant form: an under-CLEARED region read as live state, which drifts
    the forces away as the run accumulates writes into the tail.
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, pos0 = _plugin_context(graph=True)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                              # warm up + capture at BOX0 (ncx 5)
    assert capfd.readouterr().err.count("graph capture #") >= 1, \
        "trace not firing / graph never captured -- a later count of 0 is meaningless"

    L2 = 5.50                                # ncx 5 -> 6 (boundary at 5.4 nm)
    scaled = pos0 * (L2 / BOX0)              # uniform scale -> physical config
    pctx.setPeriodicBoxVectors(*_cube(L2))
    pctx.setPositions(scaled * NM)
    pi.step(400)                             # 1 eager step, then ~399 replays at ncx 6

    err = capfd.readouterr().err
    assert err.count("graph capture #") == 0, (
        f"expansion + sustained replay recaptured the graph:\n{err}")

    e_p, f_p, p = _energy(pctx), _forces(pctx), _positions(pctx)
    assert np.isfinite(e_p) and np.all(np.isfinite(f_p)), "run went non-finite"

    e_r, f_r = _reference_at(_plugin_reference(), _cube(L2), p)

    record_observed(request, "NPT sustained-replay E vs unchanged plugin, up-cross",
                    0.0, _erel(e_p, e_r))
    record_observed(request, "NPT sustained-replay F_rmse vs unchanged plugin, up-cross",
                    0.0, _frmse(f_p, f_r))
    _assert_matches("after 400 steps at ncx 6", e_p, f_p, e_r, f_r)


# Anisotropic: ncx/ncy/ncz cross INDEPENDENTLY in a single box change. Boxes chosen
# so exactly one axis goes up, one down, one stays (ncx = floor(box/0.9)):
#   (5.0,5.5,4.4) -> (5,6,4)  x same, y up, z down
#   (5.5,4.4,5.0) -> (6,4,5)  x up, y down, z same
#   (4.4,5.6,5.0) -> (4,6,5)  x down, y up, z same
_ANISO = [(5.0, 5.5, 4.4), (5.5, 4.4, 5.0), (4.4, 5.6, 5.0)]


@pytest.mark.parametrize("Lx,Ly,Lz", _ANISO)
def test_npt_anisotropic_bandcross_energy(request, plugin_dir, Lx, Ly, Lz):
    """Anisotropic band cross (per-axis grid recompute) under an active graph.
    Energy uses int64 accumulation, so it is robust to the int32 force-accumulator
    saturation that an abrupt anisotropic COMPRESSION triggers -- assert energy
    (read with no intervening step, i.e. no overflow-contaminated dynamics)
    is unchanged for every up/down/same axis combination."""
    pctx, pi, pos0 = _plugin_context(graph=True)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                              # warm + capture at cubic BOX0

    scaled = pos0.copy()
    scaled[:, 0] *= Lx / BOX0
    scaled[:, 1] *= Ly / BOX0
    scaled[:, 2] *= Lz / BOX0
    pctx.setPeriodicBoxVectors(*_ortho(Lx, Ly, Lz))
    pctx.setPositions(scaled * NM)           # clean recompute; NO step (avoids overflow)
    e_p = _energy(pctx)

    e_r, _f_r = _reference_at(_plugin_reference(), _ortho(Lx, Ly, Lz), scaled)

    nc = (int(Lx / CUTOFF), int(Ly / CUTOFF), int(Lz / CUTOFF))
    record_observed(request, f"NPT aniso band-cross E vs unchanged plugin nc={nc}",
                    0.0, _erel(e_p, e_r))
    # Energy only, as above: an abrupt anisotropic compression saturates the int32
    # force accumulator, a documented limit rather than a band-cross fault.
    _assert_matches(f"nc={nc}", e_p, None, e_r, None, forces=False)


@pytest.mark.parametrize("Lx,Ly,Lz", [(5.0, 5.5, 5.0), (5.5, 5.0, 5.6)])
def test_npt_anisotropic_dilation_forces(request, plugin_dir, Lx, Ly, Lz):
    """Anisotropic DILATION (all changing axes go up) under an active graph via
    setPeriodicBoxVectors only (barostat-style box change -> UpdateBox band-cross
    -> recapture). Dilation keeps forces physical (in the int32 range), so assert
    BOTH energy and forces are unchanged."""
    pctx, pi, _ = _plugin_context(graph=True)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                              # warm + capture at cubic BOX0
    pctx.setPeriodicBoxVectors(*_ortho(Lx, Ly, Lz))   # box only (no setPositions)
    pi.step(1)
    e_p, f_p, p = _energy(pctx), _forces(pctx), _positions(pctx)

    e_r, f_r = _reference_at(_plugin_reference(), _ortho(Lx, Ly, Lz), p)

    nc = (int(Lx / CUTOFF), int(Ly / CUTOFF), int(Lz / CUTOFF))
    record_observed(request, f"NPT aniso dilation E vs unchanged plugin nc={nc}",
                    0.0, _erel(e_p, e_r))
    record_observed(request, f"NPT aniso dilation F_rmse vs unchanged plugin nc={nc}",
                    0.0, _frmse(f_p, f_r))
    _assert_matches(f"nc={nc}", e_p, f_p, e_r, f_r)


def test_npt_barostat_band_cross_under_graph(request, plugin_dir):
    """Authentic run: dilate to just above the ncx 6->5 boundary (5.42 nm), then a
    barostat compresses back toward ~5.0 nm, genuinely crossing ncx 6->5 (at 5.4
    nm) under the graph -- a real down-cross with physical, in-range forces.
    Require no NaN, the crossing to happen, and energy/forces to stay identical to
    an unchanged plugin context throughout."""
    pctx, pi, pos = _plugin_context(graph=True, barostat_bar=50.0)
    pctx.setPeriodicBoxVectors(*_cube(5.42))
    pctx.setPositions(pos * (5.42 / BOX0) * NM)   # dilute, just above the boundary
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=300)

    # Snapshot during the run, compare afterwards -- the reference is another
    # plugin context and must not be built inside the window (see _plugin_reference).
    ncx_seen, snaps = set(), []
    for b in range(120):
        pi.step(5)
        e = _energy(pctx)
        assert np.isfinite(e), f"non-finite PE at block {b}"
        L = _box(pctx)
        ncx_seen.add(int(L / CUTOFF))
        if b % 20 == 0 or b == 119:
            snaps.append((b, L, _positions(pctx), e, _forces(pctx)))
    assert len(ncx_seen) > 1, f"no band cross exercised (ncx seen: {sorted(ncx_seen)})"

    rctx = _plugin_reference()
    worst_e = worst_f = 0.0
    for b, L, p, e, f in snaps:
        e_r, f_r = _reference_at(rctx, _cube(L), p)
        worst_e = max(worst_e, _erel(e, e_r))
        worst_f = max(worst_f, _frmse(f, f_r))
        _assert_matches(f"barostat block {b} at L={L:.4f}", e, f, e_r, f_r)

    record_observed(request, "NPT barostat run energy vs unchanged plugin", 0.0, worst_e)
    record_observed(request, "NPT barostat run force RMSE vs unchanged plugin", 0.0, worst_f)


def test_npt_repeated_bandcross_no_recapture(plugin_dir, capfd):
    """A band cross must NOT recapture the graph. Warm up + capture a force-only
    graph, then drive several EXPANSION band crosses (box-only setPeriodicBoxVectors
    + step, so density only drops -> physical, no overflow) and confirm via the
    CUPO_PLUGIN_GRAPH_TRACE stderr trace that ZERO (re)captures happen across the
    crosses -- the graph is replayed through every one, with no eager host step
    interrupting it either."""
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    # safety 2.0 -> max grid floor(5*2/0.9)=11 cells/axis, room for many up-crosses.
    pctx, pi, _ = _plugin_context(graph=True, safety=2.0)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                 # warm up + capture a force-only graph at ncx 5
    # Sanity: the trace mechanism works and the graph really was captured -- so a
    # later count of 0 means "no recapture", not "trace never fired".
    warmup = capfd.readouterr().err
    assert warmup.count("graph capture #") >= 1, (
        f"expected >=1 capture during warmup (trace not firing?):\n{warmup}")

    ncx_seen = set()
    for L in (5.5, 6.4, 7.3, 8.2, 9.1):   # ncx 6,7,8,9,10 -- box-only up-crosses
        pctx.setPeriodicBoxVectors(*_cube(L))
        pi.step(1)              # force-only; band cross must RESUME the graph
        ncx_seen.add(int(L / CUTOFF))
    err = capfd.readouterr().err

    n_captures = err.count("graph capture #")
    assert len(ncx_seen) >= 4, f"too few distinct grids crossed: {sorted(ncx_seen)}"
    assert n_captures == 0, (
        f"band crosses triggered {n_captures} recapture(s) (expected 0):\n{err}")


def test_npt_setpos_no_recapture(request, plugin_dir, capfd):
    """A setPositions must not disturb the captured graph at all.

    Nothing tells the plugin a coordinate jump happened -- there is no per-force
    callback for Context.setPositions, and nothing intercepts it -- so this is
    purely the cell list absorbing the jump on its next ordinary build. Warm and
    capture a force-only graph, then drive several jumps that move every atom more
    than a full cell, and assert ZERO recaptures while the forces stay identical to
    an unchanged plugin context. Both halves matter: 0 captures alone would also be
    consistent with silently replaying a stale decomposition, and it is the force
    agreement that rules that out."""
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _ = _plugin_context(graph=True)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                              # warm + capture force-only graph at ncx 5
    relaxed = _positions(pctx)
    warm = capfd.readouterr().err
    assert warm.count("graph capture #") >= 1, (
        f"expected >=1 capture during warmup (trace not firing?):\n{warm}")

    # Each shift moves every atom >1 cell (cutoff 0.9 nm); a stale matrix would give
    # wrong forces, so matching an unchanged context proves the light reseed ran.
    snaps = []
    for shift in (1.5, -1.7, 2.3, -0.8):
        shifted = relaxed + np.array([shift, 0.0, 0.0])
        pctx.setPositions(shifted * NM)      # -> setPos light-resume path
        snaps.append((shift, shifted, _forces(pctx)))  # force-only; RESUME not recapture
    err = capfd.readouterr().err
    n_captures = err.count("graph capture #")
    assert n_captures == 0, (
        f"setPositions triggered {n_captures} recapture(s) (expected 0):\n{err}")

    rctx = _plugin_reference()
    box_now = _cube(_box(pctx))
    worst_f = 0.0
    for shift, shifted, f_p in snaps:
        _e_r, f_r = _reference_at(rctx, box_now, shifted)
        worst_f = max(worst_f, _frmse(f_p, f_r))
        _assert_matches(f"setPositions shift {shift:+.1f} nm", None, f_p, None, f_r,
                        energy=False)
    record_observed(request, "NPT setPositions resume force RMSE vs unchanged plugin",
                    0.0, worst_f)


def test_npt_box_oscillation_no_recapture(request, plugin_dir, capfd):
    """Modest box moves in BOTH directions must not disturb the captured graph.

    The box counterpart to test_npt_setpos_no_recapture, and the same two-sided
    assertion: zero recaptures AND forces identical to an unchanged plugin
    context. Either half alone is weak -- 0 captures is also what replaying a stale decomposition
    looks like, and a force check alone would not notice the graph being silently
    rebuilt underneath.

    Moves are barostat-style: the box and the coordinates are scaled TOGETHER, so
    density is preserved and the forces stay physical. Compressing the box without
    rescaling is not a modest move -- positions drift outside the box as a run
    proceeds (OpenMM keeps molecules whole rather than wrapping), so shrinking the
    box folds those atoms back in on top of others, and the resulting clashes run
    straight into the documented int32 force-accumulator limit. That is a property
    of abrupt compression, not of the band cross.
    """
    os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    pctx, pi, _ = _plugin_context(graph=True)
    mm.LocalEnergyMinimizer.minimize(pctx, maxIterations=200)
    pi.step(60)                              # warm + capture a force-only graph at ncx 5
    warm = capfd.readouterr().err
    assert warm.count("graph capture #") >= 1, (
        f"expected >=1 capture during warmup (trace not firing?):\n{warm}")

    # (length, expected ncx). Boundaries are multiples of the 0.9 nm cutoff.
    schedule = [(5.20, 5),   # in-band nudge
                (5.50, 6),   # up-cross
                (5.30, 5),   # down-cross
                (5.45, 6),   # up-cross again
                (5.00, 5)]   # down-cross back to the start
    snaps, ncx_seen = [], set()
    for L, want_ncx in schedule:
        assert int(L / CUTOFF) == want_ncx, f"L={L} is not ncx {want_ncx}"
        scale = L / _box(pctx)
        pctx.setPeriodicBoxVectors(*_cube(L))
        pctx.setPositions(_positions(pctx) * scale * NM)
        pi.step(1)                           # force-only: must replay through the change
        ncx_seen.add(int(_box(pctx) / CUTOFF))
        snaps.append((L, _positions(pctx), _forces(pctx)))
    err = capfd.readouterr().err

    n_captures = err.count("graph capture #")
    assert ncx_seen == {5, 6}, f"expected both grids to be visited, saw {sorted(ncx_seen)}"
    assert n_captures == 0, (
        f"box oscillation triggered {n_captures} recapture(s) (expected 0):\n{err}")

    rctx = _plugin_reference()
    worst_f = 0.0
    for L, p, f_p in snaps:
        _e_r, f_r = _reference_at(rctx, _cube(L), p)
        worst_f = max(worst_f, _frmse(f_p, f_r))
        _assert_matches(f"box oscillation at L={L}", None, f_p, None, f_r, energy=False)
    record_observed(request, "NPT box-oscillation force RMSE vs unchanged plugin",
                    0.0, worst_f)


def test_npt_bandcross_beyond_envelope_fails_hard(plugin_dir):
    """A box move that compresses past the pre-allocated occupancy envelope must
    FAIL HARD rather than silently realloc buffers the captured graph baked.
    Overflow detection is device-side: the seed/migrate kernel __trap()s, which
    corrupts the CUDA context so the next CUDA call errors and the process exits
    non-zero. Run in a subprocess (the trap terminates the process): safety=1.0
    leaves no occupancy headroom, so an abrupt ncx 5->4 compression trips it."""
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    from conftest import PLUGIN_PATH
    script = textwrap.dedent(f"""
        import os, sys, random
        import numpy as np
        sys.path[:0] = {PLUGIN_PATH!r}
        import openmm as mm, openmm.unit as unit, openmm.app as app
        import nv_openmm_plugin
        NM = unit.nanometer
        CUTOFF, BOX0 = 0.9, 5.0
        ff = app.ForceField("tip3p.xml")
        modeller = app.Modeller(app.Topology(), [])
        random.seed(20260723); np.random.seed(20260723)
        modeller.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(BOX0, BOX0, BOX0) * NM)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                 nonbondedCutoff=CUTOFF * NM, constraints=app.HBonds,
                                 rigidWater=True)
        pos = np.array(modeller.positions.value_in_unit(NM))
        os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1"
        os.environ["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
        # safety=1.0: envelope == the box-5 grid, ZERO occupancy headroom.
        # This child has no conftest helpers in scope. Convert explicitly so an
        # import/NameError cannot masquerade as the expected device failure.
        indices = [i for i in range(system.getNumForces())
                   if isinstance(system.getForce(i), mm.NonbondedForce)]
        assert len(indices) == 1
        source = system.getForce(indices[0])
        system.addForce(nv_openmm_plugin.NonbondedForce(source, 1.0))
        system.removeForce(indices[0])
        integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                            0.002 * unit.picoseconds)
        integ.setRandomNumberSeed(20260723)
        ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                         {{"Precision": "mixed"}})
        ctx.setPositions(pos * NM)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
        integ.step(60)   # warm + lock the envelope at ncx 5
        ctx.getState(getEnergy=True)  # surface any setup failure before arming
        print("ARMED_OCCUPANCY_OVERFLOW", flush=True)
        L = 4.0          # ncx 5 -> 4: denser cells, occupancy past the factor=1.0 envelope
        ctx.setPeriodicBoxVectors(mm.Vec3(L, 0, 0) * NM, mm.Vec3(0, L, 0) * NM,
                                  mm.Vec3(0, 0, L) * NM)
        ctx.setPositions((pos * (L / BOX0)) * NM)
        # The compression-step reseed __trap()s on the overflowing cell, corrupting
        # the context; a subsequent CUDA call then surfaces the error. Step a few
        # times so a call hits the poisoned context.
        try:
            for _ in range(5):
                integ.step(1)
                ctx.getState(getEnergy=True)
        except Exception as error:
            # A poisoned CUDA context can abort during teardown. Preserve the
            # actual evaluation diagnostic, then exit without CUDA destructors.
            print("OVERFLOW_EXCEPTION: " + str(error), flush=True)
            os._exit(86)
        print("NO_ABORT", flush=True)
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=300)
    details = f"exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert "ARMED_OCCUPANCY_OVERFLOW" in proc.stdout, details
    assert "[nv-plugin cupo] graph capture #" in proc.stderr, details
    assert "NO_ABORT" not in proc.stdout, details
    assert proc.returncode == 86, details
    assert "OVERFLOW_EXCEPTION:" in proc.stdout, details
    assert any(message in proc.stdout.lower() for message in (
        "unspecified launch failure", "illegal instruction", "cuda_error_launch_failed",
    )), f"expected a CUDA device-trap diagnostic after compression:\n{details}"
