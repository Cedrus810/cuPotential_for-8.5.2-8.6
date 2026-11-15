# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin's own isotropic Monte Carlo barostat (NvMonteCarloBarostat).

It is a drop-in for openmm.MonteCarloBarostat -- same move, same acceptance
criterion, same adaptive volume-scale tuning -- so the tests here are about the
two places it deliberately does something different, plus the guard rails:

  * the coordinate scaling. Stock runs one thread per molecule and each thread
    walks its own molecule twice, so a system with a few large molecules among
    many small ones serializes on the large ones. Ours is warp-per-molecule for
    the solvent, block-per-molecule for the big ones, and flat over atoms for the
    displacement. That is a different summation order, so the check is not
    "bitwise equals stock" but the two invariants the move is actually defined
    by: intramolecular geometry is rigid, and each centroid scales by exactly s.

  * the energy source. Stock asks Context::getState(State::Energy), which always
    also computes the kinetic energy -- which a barostat never reads. Ours calls
    ContextImpl::calcForcesAndEnergy directly. Nothing observable should change,
    which is what the ensemble test pins down.

Everything runs on the same TIP3P water box test_npt.py uses.
"""
import os

import numpy as np
import pytest

import openmm as mm
import openmm.unit as unit

from conftest import replace_nonbonded
from test_npt import _build_water, _restore_cuda_graph_env, NM, KJ, CUTOFF  # noqa: F401

SAFETY = 1.35
PRESSURE_BAR = 1.0
TEMP_K = 300.0


def _molecule_table(ctx):
    """Molecule decomposition as OpenMM computes it, as a list of atom-index arrays.

    Taken from the live Context: it is the same decomposition the barostat kernel
    was handed, and an accelerated System cannot be instantiated on any other
    platform to ask a second time.
    """
    return [np.asarray(m, dtype=int) for m in ctx.getMolecules()]


def _frozen_integrator():
    """An integrator that runs the barostat and nothing else.

    A CustomIntegrator executes only the computations it is given, and
    addUpdateContextState() is the one that fires ForceImpl hooks -- so with no
    other computation added, stepping moves an atom if and only if the barostat
    moved it. That isolation is what makes the scaling tests below a measurement
    of the kernel rather than of a whole dynamics step.

    (The obvious alternative, a Langevin integrator with dt = 0, is degenerate:
    it blows the system up within a couple of steps, with the stock barostat just
    as much as with ours.)
    """
    integ = mm.CustomIntegrator(0.002 * unit.picoseconds)
    integ.addUpdateContextState()
    return integ


def _nv_context(*, frequency=20, seed=777, dt_ps=0.002, precision="mixed", graph=True,
                frozen=False):
    """A plugin context carrying OUR barostat."""
    import nv_openmm_plugin as nv
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1" if graph else "0"
    system, pos = _build_water()
    replace_nonbonded(system, npt_box_safety_factor=SAFETY)
    system.addForce(nv.MonteCarloBarostat(
        PRESSURE_BAR * unit.bar, TEMP_K * unit.kelvin, frequency=frequency, seed=seed))
    if frozen:
        integ = _frozen_integrator()
    else:
        integ = mm.LangevinMiddleIntegrator(TEMP_K * unit.kelvin, 1.0 / unit.picosecond,
                                            dt_ps * unit.picoseconds)
        integ.setRandomNumberSeed(4242)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": precision})
    ctx.setPositions(pos * NM)
    return ctx, integ, system


def _stock_barostat_context(*, frequency=20, seed=777, dt_ps=0.002, precision="mixed"):
    """The same system carrying OpenMM's barostat, for an ensemble comparison."""
    import nv_openmm_plugin as nv
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1"
    system, pos = _build_water()
    replace_nonbonded(system, npt_box_safety_factor=SAFETY)
    baro = mm.MonteCarloBarostat(PRESSURE_BAR * unit.bar, TEMP_K * unit.kelvin, frequency)
    baro.setRandomNumberSeed(seed)
    system.addForce(baro)
    integ = mm.LangevinMiddleIntegrator(TEMP_K * unit.kelvin, 1.0 / unit.picosecond,
                                        dt_ps * unit.picoseconds)
    integ.setRandomNumberSeed(4242)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": precision})
    ctx.setPositions(pos * NM)
    return ctx, integ, system


def _boxL(ctx):
    a, _, _ = ctx.getState().getPeriodicBoxVectors()
    return a[0].value_in_unit(NM)


def _positions(ctx):
    return np.asarray(ctx.getState(getPositions=True)
                      .getPositions(asNumpy=True).value_in_unit(NM), dtype=np.float64)


# ---------------------------------------------------------------------------
# The scaling kernel
# ---------------------------------------------------------------------------

def test_scaling_preserves_molecular_geometry_and_scales_centroids(plugin_dir):
    """The move's defining invariants, measured on a real accepted move.

    A volume move displaces each molecule's CENTROID by center*(s-1) and moves
    every atom of that molecule by the same vector. So:

      (a) every atom's offset from its own centroid is unchanged -- bonds, angles
          and rigid-water geometry survive the move untouched;
      (b) every centroid is multiplied by exactly s.

    Integration is switched off entirely (see _frozen_integrator) so the
    barostat's scaling is the only thing that can move an atom, which makes this
    a direct measurement of the kernel rather than of a whole step.
    """
    ctx, integ, system = _nv_context(frequency=1, seed=31337, frozen=True)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
    mols = _molecule_table(ctx)

    # Step until a move is accepted, which the box length reveals.
    before, L0 = _positions(ctx), _boxL(ctx)
    for _ in range(40):
        integ.step(1)
        L1 = _boxL(ctx)
        if L1 != L0:
            break
        before, L0 = _positions(ctx), L0
    else:
        pytest.fail("no volume move was accepted in 40 attempts")

    after = _positions(ctx)
    s = L1 / L0

    worst_geom, worst_centroid = 0.0, 0.0
    for atoms in mols:
        c_before = before[atoms].mean(axis=0)
        c_after = after[atoms].mean(axis=0)
        # (a) intramolecular geometry is rigid
        worst_geom = max(worst_geom, np.abs(
            (after[atoms] - c_after) - (before[atoms] - c_before)).max())
        # (b) the centroid scaled by exactly s
        worst_centroid = max(worst_centroid, np.abs(c_after - s * c_before).max())

    # float32 positions on a ~5 nm box: one ulp is ~5e-7 nm.
    assert worst_geom < 2e-6, f"intramolecular geometry moved by {worst_geom:.2e} nm"
    assert worst_centroid < 2e-6, f"centroid scaling off by {worst_centroid:.2e} nm"


def test_scaling_accuracy_against_float64_reference(plugin_dir):
    """Scaled positions match a float64 centroid computation.

    Our centroids accumulate in double through a tree reduction, so they should
    land within a float32 ulp of an exact computation -- this is the check that
    the two-tier warp/block split did not quietly get the arithmetic wrong for
    one of the two size classes.
    """
    ctx, integ, system = _nv_context(frequency=1, seed=4711, frozen=True)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
    mols = _molecule_table(ctx)

    before, L0 = _positions(ctx), _boxL(ctx)
    for _ in range(40):
        integ.step(1)
        L1 = _boxL(ctx)
        if L1 != L0:
            break
        before, L0 = _positions(ctx), L0
    else:
        pytest.fail("no volume move was accepted in 40 attempts")

    after = _positions(ctx)
    s = L1 / L0

    reference = before.copy()
    for atoms in mols:
        reference[atoms] += before[atoms].mean(axis=0) * (s - 1.0)

    err = np.abs(after - reference).max()
    assert err < 2e-6, f"max deviation from the float64 reference: {err:.3e} nm"


def test_molecule_size_classes_are_both_exercised(plugin_dir):
    """The water box is all-small; confirm the kernel is correct when the LARGE
    class is populated too, since that is a separate code path (block per
    molecule, shared-memory tree, rather than warp per molecule).

    A rigid-water system has only 3-atom molecules, so the large path would never
    run and a bug in it would go unseen. Bonding a subset of waters into one big
    molecule forces the split.
    """
    import nv_openmm_plugin as nv
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = "1"
    system, pos = _build_water()
    # Tie 200 oxygens together into a single 600-atom molecule, well past the
    # 32-atom warp threshold. Zero-strength bonds: they change the molecule
    # decomposition without perturbing the physics.
    oxygens = [i for i in range(0, 600, 3)]
    tether = mm.HarmonicBondForce()
    for a, b in zip(oxygens, oxygens[1:]):
        tether.addBond(a, b, 0.3 * NM, 0.0 * KJ / NM**2)
    system.addForce(tether)

    replace_nonbonded(system, npt_box_safety_factor=SAFETY)
    system.addForce(nv.MonteCarloBarostat(
        PRESSURE_BAR * unit.bar, TEMP_K * unit.kelvin, frequency=1, seed=99))
    integ = _frozen_integrator()
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": "mixed"})
    ctx.setPositions(pos * NM)
    mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)

    mols = _molecule_table(ctx)
    sizes = sorted((len(m) for m in mols), reverse=True)
    assert sizes[0] > 32, f"the large-molecule path was not exercised (max size {sizes[0]})"

    before, L0 = _positions(ctx), _boxL(ctx)
    for _ in range(40):
        integ.step(1)
        L1 = _boxL(ctx)
        if L1 != L0:
            break
        before, L0 = _positions(ctx), L0
    else:
        pytest.fail("no volume move was accepted in 40 attempts")

    after, s = _positions(ctx), L1 / L0
    reference = before.copy()
    for atoms in mols:
        reference[atoms] += before[atoms].mean(axis=0) * (s - 1.0)
    err = np.abs(after - reference).max()
    assert err < 2e-6, f"max deviation with a large molecule present: {err:.3e} nm"


# ---------------------------------------------------------------------------
# The ensemble
# ---------------------------------------------------------------------------

def test_equilibrium_density_matches_stock_barostat(plugin_dir):
    """Ours and OpenMM's barostat must equilibrate to the same box.

    Not a bitwise comparison: the two draw the same random stream but scale
    coordinates with different summation orders, so their trajectories separate
    almost immediately. What has to agree is the ensemble they sample, so this
    compares the mean box length over the back half of a run against the spread
    within each run.
    """
    samples = {}
    for name, factory in (("nv", _nv_context), ("stock", _stock_barostat_context)):
        ctx, integ, _ = factory(frequency=20, seed=2468, dt_ps=0.002)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=200)
        integ.step(2000)                      # equilibrate
        L = []
        for _ in range(40):
            integ.step(50)
            L.append(_boxL(ctx))
        samples[name] = np.asarray(L)
        del ctx, integ

    nv_mean, stock_mean = samples["nv"].mean(), samples["stock"].mean()
    # Standard error of the difference; the samples are correlated, so this is
    # generous by construction rather than a tight statistical claim.
    sem = np.hypot(samples["nv"].std(ddof=1), samples["stock"].std(ddof=1)) / np.sqrt(40)
    diff = abs(nv_mean - stock_mean)
    assert diff < max(5 * sem, 0.004 * stock_mean), (
        f"mean box length {nv_mean:.4f} nm (nv) vs {stock_mean:.4f} nm (stock); "
        f"difference {diff:.4f} nm, 1 sem {sem:.4f} nm")


def test_same_seed_reproduces_the_same_trajectory(plugin_dir):
    """Two runs with the same seed take the same sequence of volume moves.

    The plugin is deterministic by design, and the barostat must not be the thing
    that breaks it.
    """
    trace = []
    for _ in range(2):
        ctx, integ, _ = _nv_context(frequency=10, seed=13579, dt_ps=0.002)
        mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=100)
        run = []
        for _ in range(10):
            integ.step(20)
            run.append(_boxL(ctx))
        trace.append(run)
        del ctx, integ
    assert trace[0] == trace[1], (
        f"same seed gave different box trajectories:\n  {trace[0]}\n  {trace[1]}")


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------

def test_two_barostats_are_rejected(plugin_dir):
    """Ours shares OpenMM's parameter names so scripts can swap one for the
    other; the price is that having both is a silent configuration error, so it
    has to be a loud one instead.

    Caught at Context creation, in NvMonteCarloBarostatImpl -- not when adding.
    Adding is just System.addForce now, which the plugin never sees, and the
    C++ check is the stronger place anyway: it counts BOTH kinds, so two of ours
    is caught as well as one of each."""
    import nv_openmm_plugin as nv
    system, _ = _build_water()
    replace_nonbonded(system, npt_box_safety_factor=SAFETY)
    system.addForce(mm.MonteCarloBarostat(PRESSURE_BAR * unit.bar, TEMP_K * unit.kelvin, 25))
    system.addForce(nv.MonteCarloBarostat(PRESSURE_BAR * unit.bar, TEMP_K * unit.kelvin))
    with pytest.raises(Exception, match="the System contains 2 barostats"):
        mm.Context(system, _frozen_integrator(),
                   mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})


def test_frequency_zero_disables_the_barostat(plugin_dir):
    """frequency=0 is OpenMM's "off" switch; the box must not move at all."""
    ctx, integ, _ = _nv_context(frequency=0, seed=777, dt_ps=0.002)
    L0 = _boxL(ctx)
    integ.step(200)
    assert _boxL(ctx) == L0, "the box moved with frequency=0"
