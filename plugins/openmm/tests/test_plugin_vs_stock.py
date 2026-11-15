# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two kinds of correctness check for NvNonbondedForce.

  1. Apples-to-apples -- test_energy / test_force. cupo and stock OpenMM run at
     the SAME cutoff, Ewald alpha and PME grid, so anything that differs is an
     implementation difference. This is the "did we build it right" test.

  2. High accuracy -- test_accuracy_energy / test_accuracy_force. cupo at
     production settings against a CONVERGED reference: same alpha, twice the
     cutoff, twice the PME grid per dimension, fp64. This is the "is the science
     right" test, and the one that has to keep holding when the splitting itself
     is replaced.

Systems: two synthetic boxes (monatomic LJ+charges with no exclusions, and a
small TIP3P water box) plus every solvated system under data/. The further
synthetic boxes below each isolate one failure mode no real system exhibits.
"""

import copy
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest


import openmm as mm
import openmm.unit as unit

from conftest import record_observed, replace_nonbonded


def _create_lj_charged_box(n_atoms=256, seed=42):
    """Create a box of charged LJ particles (no bonds, no exclusions).

    Positions on a cubic grid to avoid overlaps.
    """
    rng = np.random.default_rng(seed)

    n_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    spacing = 0.45  # nm (> sigma to avoid LJ blowup)
    box_nm = n_side * spacing

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0, 0) * unit.nanometers,
        mm.Vec3(0, box_nm, 0) * unit.nanometers,
        mm.Vec3(0, 0, box_nm) * unit.nanometers,
    )

    sigma = 0.34  # nm
    epsilon = 1.0  # kJ/mol
    charges = rng.normal(size=n_atoms).astype(np.float64)
    charges -= charges.mean()  # neutralize

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometers)
    nb.setUseDispersionCorrection(True)

    for i in range(n_atoms):
        system.addParticle(39.948)
        nb.addParticle(charges[i], sigma, epsilon)

    system.addForce(nb)

    positions = []
    for i in range(n_atoms):
        ix = i % n_side
        iy = (i // n_side) % n_side
        iz = i // (n_side * n_side)
        pos = np.array([ix, iy, iz], dtype=np.float64) * spacing + 0.5 * spacing
        pos += rng.uniform(-0.05, 0.05, size=3)
        positions.append(pos)
    positions_mm = [mm.Vec3(*p) * unit.nanometers for p in positions]

    return system, positions_mm


_WATER_BOX_SEED = 42


def _create_water_box():
    """Create a TIP3P water box (has bonds and exclusions).

    Box must be > 3*cutoff in each dimension to ensure the towerplate
    cell grid has at least 3 cells per axis (avoids pair aliasing).

    Uses fixed seed (_WATER_BOX_SEED) for deterministic addSolvent so that
    cupo's own tests get identical positions.
    """
    import random
    import openmm.app as app

    ff = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    random.seed(_WATER_BOX_SEED)
    np.random.seed(_WATER_BOX_SEED)
    modeller.addSolvent(
        ff, model="tip3p", boxSize=mm.Vec3(3, 3, 3) * unit.nanometers)
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometers,
        constraints=None,
    )
    return system, modeller.positions


_CUTOFF_NM = 0.9

# One force field for every dataset, so a tolerance means the same thing across
# the sweep. It has to be amber14: apoa1 is a lipoprotein and amber99sbildn has
# no template for its POPC lipids, while amber14 parameterises all eleven.
_DATASET_FORCEFIELD = ("amber14-all.xml", "amber14/tip3pfb.xml")

# cellulose is the exception, and has to be: it is a carbohydrate and amber14-all
# has no glucose templates. GLYCAM supplies them, layered ON TOP of amber14-all
# whose atom types its templates borrow (loading it alone raises KeyError on
# 'protein-H1').
#
# It is deliberately NOT added to the shared tuple. GLYCAM is not additive: its
# ACX acetyl cap is graph-isomorphic to ff14SB's ACE, so every protein dataset
# with an acetyl cap -- bace, cdk2, p38, ptp1b, thrombin, tyk2, jnk1, mcl1 --
# stops parameterising with "Multiple non-identical matching templates found for
# residue 0 (ACE): ACX, ACE". Scoping it to the one system that needs it leaves
# the other eleven bit-for-bit as they were.
#
# The *_tip4p datasets use 4-site TIP4P-Ew, whose residues carry a massless
# charged EPW virtual site that no 3-site template matches. These datasets put
# virtual sites through the nonbonded comparison.
_SYSTEM_FORCEFIELD = {
    "cellulose": ("amber14-all.xml", "amber14/GLYCAM_06j-1.xml",
                  "amber14/tip3pfb.xml"),
    "bace_tip4p": ("amber14-all.xml", "amber14/tip4pew.xml"),
    "cdk2_tip4p": ("amber14-all.xml", "amber14/tip4pew.xml"),
    "dhfr_tip4p": ("amber14-all.xml", "amber14/tip4pew.xml"),
    "tyk2_tip4p": ("amber14-all.xml", "amber14/tip4pew.xml"),
}


def _dataset_names():
    """Every solvated system under data/, as load_dataset() names them."""
    _REPO = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_REPO))
    from data.datasets import available_datasets
    return available_datasets()


_DATASET_NAMES = _dataset_names()


# Each dataset's system as the force field builds it, kept for the rest of the
# module: three module-scoped fixtures each want every dataset, and the build is
# most of their time -- cellulose takes 23 s, all thirteen 47 s, and a copy of a
# built system 0.1-2.9 s. Kept rather than rebuilt per fixture, because pytest runs
# all of one fixture's datasets before the next fixture's first.
_built_systems = {}


def _create_dataset_system(name):
    """A copy of the solvated system from data/<name>/, with its positions. A copy
    because callers strip forces from theirs; the positions are shared, and nobody
    changes them. See _build_dataset_system."""
    if name not in _built_systems:
        _built_systems[name] = _build_dataset_system(name)
    system, positions = _built_systems[name]
    return copy.deepcopy(system), positions


def _build_dataset_system(name):
    """Build a solvated system from data/<name>/ with full nonbonded parameters.

    Real systems reach paths the synthetic boxes cannot: non-trivial 1-4 scaled
    pairs, a net charge (apoa1 -14, dhfr -11, which exercises the PME plasma
    correction), and in bace a CYS160-CYS364 disulfide whose exclusions span
    |i-j| up to 3189 -- far outside any local sequence window.

    Returns (system, positions) -- same interface as _create_water_box().
    """
    import openmm.app as app

    _REPO = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(_REPO))
    from data.datasets import load_dataset, template_pins

    dataset = load_dataset(name)
    ff = app.ForceField(*_SYSTEM_FORCEFIELD.get(name, _DATASET_FORCEFIELD))
    system = ff.createSystem(
        dataset.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=_CUTOFF_NM * unit.nanometers,
        constraints=None,
        rigidWater=False,
        residueTemplates=template_pins(dataset.topology))
    return system, dataset.positions


def _strip_bonded_forces(system):
    """Remove HarmonicBond, HarmonicAngle, PeriodicTorsion from system in-place."""
    for idx in reversed(range(system.getNumForces())):
        force = system.getForce(idx)
        if isinstance(force, (mm.HarmonicBondForce, mm.HarmonicAngleForce,
                              mm.PeriodicTorsionForce)):
            system.removeForce(idx)


def _get_energy_and_forces(system, positions, precision="single"):
    """Compute potential energy (kJ/mol) and forces (kJ/mol/nm)."""
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    props = {"Precision": precision}
    context = mm.Context(system, integrator, platform, props)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometers)
    del context, integrator
    return energy, forces


def _compare(system, positions):
    """Run stock (double, single) and plugin (single).

    Returns (e_fp64, e_fp32, e_plugin, f_fp64, f_fp32, f_plugin).
    """
    import nv_openmm_plugin

    system_xml = mm.XmlSerializer.serialize(system)

    # Stock OpenMM in double precision (gold reference)
    e_fp64, f_fp64 = _get_energy_and_forces(system, positions, precision="double")

    # Stock OpenMM in single precision (fp32 baseline)
    system_fp32 = mm.XmlSerializer.deserialize(system_xml)
    e_fp32, f_fp32 = _get_energy_and_forces(system_fp32, positions, precision="single")

    # Plugin in single precision (its kernels are fp32)
    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    _replace_with_pme(system_plugin)
    e_plugin, f_plugin = _get_energy_and_forces(
        system_plugin, positions, precision="single")

    return (e_fp64, e_fp32, e_plugin,
            np.array(f_fp64), np.array(f_fp32), np.array(f_plugin))


_SYSTEMS = {
    "monatomic": _create_lj_charged_box,
    "water": _create_water_box,
    **{name: partial(_create_dataset_system, name) for name in _DATASET_NAMES},
}

# Per-(system, precision) energy tolerances; default 1e-6 against stock fp64.
# The accurate erfcf() in the real-space energy path (vs the biased ~1.5e-7 A&S
# polynomial) puts cupo's energy at ~1e-7 vs OpenMM double, so 1e-6 guards
# against an erfc regression (which reverts to ~1.6e-6).
_ENERGY_TOL: dict[tuple[str, str], float] = {}
_ENERGY_TOL_DEFAULT = 1e-6
# The fp32 comparison (plugin fp32 vs stock fp32) mostly measures stock OpenMM's
# own fp32 error. Stock compiles its kernels with the environment's nvrtc, and with
# nvrtc 13.4 its fp32 energy is up to 1.7e-6 from its fp64 energy (0.5e-6 to
# 1.2e-6 with nvrtc 13.1), while the plugin's stays within 6e-7. Over five runs of
# every system on an RTX 6000 Ada the comparison saw at most 1.31e-6 (cdk2_dodeca);
# 1.5e-6 leaves a margin over that.
_ENERGY_TOL_FP32_DEFAULT = 1.5e-6
# cellulose is 408k atoms, 4.4x the next largest, and the fp32 REFERENCE is what
# degrades: stock's own fp32 energy is 3.4e-6 from its fp64, while cupo's is
# 9.3e-7 -- cupo is the more accurate of the pair by 3.6x, and the 2.4e-6 the
# comparison sees is mostly stock's error, not cupo's.
_ENERGY_TOL[("cellulose", "fp32")] = 1e-5
# Unconstrained DHFR with grouped MIT bonded arithmetic: cupo is 2.4e-7
# from fp64, while stock fp32 is 8.2e-7 on the opposite side. Their mutual
# error is 1.05e-6. Keep the fp64 accuracy requirement at 1e-6; permit the
# sum of two such error budgets when comparing these fp32 implementations.
_ENERGY_TOL[("dhfr", "fp32")] = 2e-6
# dhfr_tip4p likewise: the fp32 reference is the weaker side (stock fp32 is ~8e-7
# from fp64, cupo ~3.6e-7), so the two disagree by ~1.15e-6.
_ENERGY_TOL[("dhfr_tip4p", "fp32")] = 2e-6
# On RTX 5000 Ada, apoa1_dodeca puts cupo 4.16e-7 from fp64 and stock
# fp32 5.86e-7 on the opposite side, yielding 1.00215e-6 between them.
# Allow the sum of their error budgets; the fp64 accuracy gate stays at 1e-6.
_ENERGY_TOL[("apoa1_dodeca", "fp32")] = 2e-6
# cdk2_dodeca likewise puts cupo 1.87e-7 from fp64 and stock fp32
# 8.28e-7 on the opposite side on RTX 5090, yielding 1.01512e-6 between
# them. Permit both rounding budgets only for this fp32 comparison;
# the independent fp64 accuracy gate remains at 1e-6.
_ENERGY_TOL[("cdk2_dodeca", "fp32")] = 2e-6
# On L20, cdk2_cube puts cupo 5.43e-8 from fp64 and stock fp32
# 9.48e-7 on the opposite side, yielding 1.00194e-6 between them.
# Permit both rounding budgets only for this fp32 comparison;
# the independent fp64 accuracy gate remains at 1e-6.
_ENERGY_TOL[("cdk2_cube", "fp32")] = 2e-6

# The two force comparisons measure different things, which is why their
# tolerances differ by an order of magnitude:
#
#   fp32 -- plugin against stock at the SAME precision. On most systems the two
#     make very nearly the same fp32 error and agree to ~3e-6, which is what this
#     comparison exists to check. apoa1 and cellulose are the exceptions, at
#     4.5e-5 and 3.1e-5: there the two errors are uncorrelated, and cupo is the
#     more accurate of the pair -- apoa1 2.1e-5 from fp64 against stock fp32's
#     5.1e-5, cellulose 7.6e-5 against 8.2e-5. Diffuse rather than localised in
#     both: the worst 10 atoms carry 0.3% of the squared error (cellulose's worst
#     100 carry 2.4%), so it is spread fp32 noise, not a clash or a mishandled
#     pair. Both are systems where the fp32 REFERENCE is the weaker number.
#
#   fp64 -- plugin against stock in double. Dominated by fp32-vs-fp64, so it is a
#     property of the precision rather than of cupo: stock's own fp32 error and
#     cupo's agree to three figures on cdk2 (1.04e-4 each) and bace (8.75e-5
#     each). cdk2 sets the ceiling at 1.0e-4, hence 2e-4 here.
_FORCE_TOL_DEFAULT = 1e-5
_FORCE_TOL = {(name, "fp64"): 2e-4 for name in _DATASET_NAMES}
_FORCE_TOL[("apoa1", "fp32")] = 1e-4
_FORCE_TOL[("cellulose", "fp32")] = 1e-4
# These additional TIP4P-Ew protein systems show 2.5-3.9e-5 full-force
# disagreement between the two fp32 implementations on sm_120. Removing bonded
# terms reduces it to 3.9-7.3e-6; the difference is in bonded arithmetic, not
# the nonbonded kernels. Against fp64, cupo is 7.1-8.6e-5 and stock fp32 is
# 7.5-9.5e-5. Use the same fp32 comparison bound as apoa1/cellulose while
# retaining the independent, unchanged 2e-4 fp64 accuracy requirement.
for _name in ("bace_tip4p", "cdk2_tip4p", "tyk2_tip4p"):
    _FORCE_TOL[(_name, "fp32")] = 1e-4
# The re-solvated cubes (NPT-equilibrated at 1 bar) are
# the rectangular references of the cell-shape comparison. Their fp32 deviation from stock
# is 3-4e-5 (bace_cube 3.0e-5 on L40S, apoa1_cube 4.1e-5 on L40S and RTX PRO 6000), where
# the original cells of the same solutes sit under 1e-5: the same class as apoa1 and
# cellulose, so the same bound.
for _name in _DATASET_NAMES:
    if _name.endswith("_cube"):
        _FORCE_TOL[(_name, "fp32")] = 1e-4
# The triclinic protein cells (data/<system>_{dodeca,octa}): the plugin is 2-3e-5
# from stock fp64 on them, the same as on apoa1 and better than on the rectangular
# p38 (8.5e-5), so its accuracy is not what moved. What moved is the fp32
# COMPARISON: on a rectangular cell the plugin and stock fp32 deviate from fp64 in
# the same direction (p38: both 8.5e-5 from fp64, 4e-6 from each other); on a
# tilted one stock's fp32 PME takes its own 3-5e-5 from its fp64 through the
# fractional-coordinate path and the plugin's does not follow it, so the two fp32
# arms disagree by 2-5e-5. 1e-5 between two fp32 implementations was never a
# property either could promise; 1e-4, apoa1's, is.
for _name in _DATASET_NAMES:
    if _name.endswith(("_dodeca", "_octa")):
        _FORCE_TOL[(_name, "fp32")] = 1e-4


@pytest.fixture(scope="module", params=list(_SYSTEMS))
def compare_results(request, plugin_dir):
    """Run plugin vs stock comparison, once per system."""
    system, positions = _SYSTEMS[request.param]()
    return request.param, _compare(system, positions)


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_energy(request, compare_results, precision):
    """Plugin energy relative error vs stock OpenMM."""
    name, (e_fp64, e_fp32, e_plugin, _, _, _) = compare_results
    e_ref = e_fp64 if precision == "fp64" else e_fp32
    rel_err = abs(e_ref - e_plugin) / abs(e_ref)
    default = _ENERGY_TOL_FP32_DEFAULT if precision == "fp32" else _ENERGY_TOL_DEFAULT
    tol = _ENERGY_TOL.get((name, precision), default)
    record_observed(
        request, f"Plugin vs stock {precision} relative energy error, {name}",
        tol, rel_err)
    assert rel_err < tol, (
        f"Energy relative error {rel_err:.6e} exceeds {tol:.0e}: "
        f"{precision}={e_ref:.6f}, plugin={e_plugin:.6f}, stock fp64={e_fp64:.6f}")


@pytest.mark.parametrize("precision", ["fp64", "fp32"])
def test_force(request, compare_results, precision):
    """Plugin force relative RMSE vs stock OpenMM."""
    name, (_, _, _, f_fp64, f_fp32, f_plugin) = compare_results
    f_ref = f_fp64 if precision == "fp64" else f_fp32
    force_rmse = np.sqrt(np.mean((f_ref - f_plugin)**2))
    f_mag = np.sqrt(np.mean(f_ref**2))
    rel_rmse = force_rmse / f_mag
    tol = _FORCE_TOL.get((name, precision), _FORCE_TOL_DEFAULT)
    record_observed(
        request, f"Plugin vs stock {precision} relative force RMSE, {name}",
        tol, rel_rmse)
    assert rel_rmse < tol, (
        f"Relative force RMSE {rel_rmse:.2e} exceeds {tol:.0e}")


# ─── High accuracy: is the science right? ────────────────────────────────────
#
# The two tests above pin cupo and stock to the same cutoff, alpha and PME grid,
# so all they can answer is whether cupo implements OpenMM's PME faithfully -- a
# splitting that was wrong in the same way on both sides would sail through them.
# These answer the other question: how far is cupo's total from the CONVERGED
# lattice sum? That is the number which has to stay put when the splitting itself
# is replaced (prolate, u-series, ...), because those change the decomposition
# and not the physical answer it decomposes.
#
# The reference holds alpha FIXED and pushes both truncations out: twice the
# direct-space cutoff, twice the PME grid per dimension, fp64. At fixed alpha the
# Ewald sum converges to the exact Coulomb lattice sum, so what remains is
# precisely the truncation error of the production settings -- expected to land
# near OpenMM's own default Ewald error tolerance (5e-4), not at zero.
#
# ELECTROSTATICS ONLY: every epsilon is zeroed. Lennard-Jones is plain-truncated
# rather than Ewald-split, so doubling the cutoff would explicitly sum an r^-6
# shell that production only approximates with the mean-field dispersion
# correction -- which contributes no force at all. That convention gap is worth
# ~3e-4 in relative force RMSE, enough to swamp the electrostatic signal this
# test exists to measure. Revisit when LJPME is available on both sides.

_ACCURACY_FACTOR = 2          # reference cutoff, and each grid dimension, x this
_ACCURACY_ENERGY_TOL = 1e-4
_ACCURACY_FORCE_TOL = 1e-3


def _electrostatics_only(system):
    """Reduce a System in place to one NonbondedForce carrying charges but no LJ.

    Exceptions keep their charge products, so 1-2/1-3 pairs stay fully excluded
    and 1-4 pairs keep their scaled electrostatics: the exclusion and correction
    machinery runs exactly as in production, only without LJ. Bonded terms go as
    well -- they are identical on both sides, so all they would do is inflate the
    denominator of a relative error and hide a real discrepancy.

    The single surviving force is left at index 0, which is where every helper
    below reaches for it.
    """
    found = [i for i in range(system.getNumForces())
             if isinstance(system.getForce(i), mm.NonbondedForce)]
    assert len(found) == 1, f"expected exactly one NonbondedForce, found {len(found)}"
    for idx in reversed(range(system.getNumForces())):
        if idx not in found:
            system.removeForce(idx)

    nb = system.getForce(0)
    for i in range(nb.getNumParticles()):
        charge, sigma, _eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, charge, sigma, 0.0)
    for k in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, _eps = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, p1, p2, charge_prod, sigma, 0.0)
    nb.setUseDispersionCorrection(False)


def _replace_with_pme(system, **kwargs):
    """Convert to the plugin under PME specifically, rather than under the default.

    Every caller of this measures against STOCK OpenMM -- directly, against an
    fp64 stock reference, or against the cupo pybind path, which runs the Ewald
    splitting itself. That comparison only means anything under the same
    splitting. The plugin's default is Prolate, which decomposes the same sum
    differently and so disagrees with Ewald by the two truncation errors (~2.5e-5
    in energy, ~5e-4 in force RMSE) rather than by fp32 round-off.

    Left on the default these would not fail cleanly. They would go from measuring
    round-off to measuring a splitting difference -- up to a hundredfold slacker,
    still passing the looser tolerances -- so the choice is named at every one of
    them instead of inherited.
    """
    import nv_openmm_plugin
    return replace_nonbonded(
        system, method=nv_openmm_plugin.NonbondedForce.PME, **kwargs)


def _pinned_system(system_xml, alpha, grid, cutoff_nm):
    """Deserialize, then pin the NonbondedForce to an exact (cutoff, alpha, grid)."""
    system = mm.XmlSerializer.deserialize(system_xml)
    nb = system.getForce(0)
    nb.setCutoffDistance(cutoff_nm * unit.nanometers)
    nb.setPMEParameters(alpha, *grid)
    return system


def _unpinned_system(system_xml, cutoff_nm):
    """Deserialize and set only the cutoff, leaving alpha and the mesh to derive.

    How every PLUGIN arm is built. NvNonbondedForce derives both from the cutoff
    and the box and refuses a pinned pair outright, so a system on its way to the
    plugin must not carry one -- _pinned_system is for the stock arms, which are
    pinned precisely so they cannot drift while the plugin derives.

    It also matters most for Prolate, whose mesh is sized by a different question
    -- the sampling rate a splitting of fixed bandwidth needs, not the resolution
    an Ewald error estimate asks for -- so an Ewald grid pinned onto it would
    measure an operating point that never ships.
    """
    system = mm.XmlSerializer.deserialize(system_xml)
    system.getForce(0).setCutoffDistance(cutoff_nm * unit.nanometers)
    return system


def _stock_energy_forces(system, positions, precision, *, verify_pinned=True):
    """Evaluate a stock system, checking OpenMM honoured whatever was pinned.

    setPMEParameters states a preference, not a fact: OpenMM derives alpha and
    the grid from the Ewald error tolerance instead whenever it does not take
    them, and a reference quietly computed at a different alpha would still
    return an entirely plausible number. Checked here rather than assumed.

    verify_pinned=False reads the derived values back instead, which is how the
    production settings are discovered in the first place.

    Returns (energy, forces, alpha, grid).
    """
    nb = system.getForce(0)
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision, "DeterministicForces": "true"})
    context.setPositions(positions)

    alpha, *grid = nb.getPMEParametersInContext(context)
    alpha = float(alpha)
    grid = tuple(int(g) for g in grid)
    if verify_pinned:
        # getPMEParameters hands back alpha as a Quantity in 1/nm, while
        # getPMEParametersInContext hands back a bare float. Normalise first.
        want_alpha, *want_grid = nb.getPMEParameters()
        want_alpha = want_alpha.value_in_unit(unit.nanometer ** -1)
        assert abs(alpha - want_alpha) < 1e-9 * want_alpha, (
            f"OpenMM ignored the pinned alpha: asked {want_alpha}, got {alpha}")
        assert grid == tuple(want_grid), (
            f"OpenMM ignored the pinned grid: asked {tuple(want_grid)}, got {grid}")

    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometers))
    del context, integrator
    return energy, forces, alpha, grid


def _rel_energy_error(energy, reference):
    return abs(energy - reference) / abs(reference)


def _rel_force_rmse(forces, reference):
    """L2 force RMSE over atoms, relative to the reference's RMS force magnitude."""
    rmse = np.sqrt(np.mean(np.sum((forces - reference) ** 2, axis=1)))
    return rmse / np.sqrt(np.mean(np.sum(reference ** 2, axis=1)))


@pytest.fixture(scope="module", params=_DATASET_NAMES)
def accuracy_results(request, plugin_dir):
    """Plugin AND stock at production settings, vs a converged fp64 reference.

    Stock is measured against the same reference and held to the same tolerance,
    not merely reported: the residual is PME truncation the two implementations
    share, so stock's number is what says how much this cutoff and grid are worth
    in the first place. Without it a failure cannot be read -- a regression in
    cupo and a production setting that was never that accurate look identical.
    One line moving is a cupo regression; both moving means the reference or the
    settings changed.
    """
    import nv_openmm_plugin

    name = request.param
    system, positions = _create_dataset_system(name)
    _electrostatics_only(system)

    # Production settings are whatever OpenMM derives at the production cutoff;
    # discover them once, then pin every run to them so nothing re-derives.
    _, _, alpha, grid = _stock_energy_forces(
        system, positions, "single", verify_pinned=False)
    system_xml = mm.XmlSerializer.serialize(system)

    reference = _pinned_system(
        system_xml, alpha, tuple(_ACCURACY_FACTOR * g for g in grid),
        _ACCURACY_FACTOR * _CUTOFF_NM)
    e_ref, f_ref, _, _ = _stock_energy_forces(reference, positions, "double")

    production = _pinned_system(system_xml, alpha, grid, _CUTOFF_NM)
    e_stock, f_stock, _, _ = _stock_energy_forces(production, positions, "single")

    # UNPINNED, unlike the two stock arms above, and not because the settings
    # differ -- because they cannot differ. NvNonbondedForce refuses a pinned pair
    # rather than carrying it (cupo derives alpha and the mesh itself), and under
    # PME it derives them by OpenMM's own rule from the same box, so this arm lands
    # on the very (alpha, grid) the stock arm is pinned to. What used to be
    # asserted by pinning is now a property of the derivation, and it is measured
    # rather than assumed: test_energy/test_force put the two within ~1e-6, which
    # a one-step difference in mesh would widen to ~5e-5.
    system_plugin = _unpinned_system(system_xml, _CUTOFF_NM)
    _replace_with_pme(system_plugin)
    e_plugin, f_plugin = _get_energy_and_forces(
        system_plugin, positions, precision="single")

    # The third arm: the same plugin under the PROLATE splitting, at its OWN
    # derived mesh. It cannot be compared against stock the way the two above are
    # -- a different decomposition of the same sum disagrees with Ewald by the two
    # truncation errors, ~2.5e-5 energy and ~5e-4 force, which is nobody's mistake
    # -- but it can be compared against the converged reference, because that
    # reference is the exact Coulomb lattice sum and both splittings decompose it.
    # This is the one comparison in the suite that stays valid across a change of
    # splitting, and so the one that has to hold when the splitting changes.
    system_prolate = _unpinned_system(system_xml, _CUTOFF_NM)
    replace_nonbonded(system_prolate, method=nv_openmm_plugin.NonbondedForce.Prolate)
    e_prolate, f_prolate = _get_energy_and_forces(
        system_prolate, positions, precision="single")

    return {
        "name": name,
        "alpha": alpha,
        "grid": grid,
        "energy": {"stock": _rel_energy_error(e_stock, e_ref),
                   "plugin": _rel_energy_error(e_plugin, e_ref),
                   "prolate": _rel_energy_error(e_prolate, e_ref)},
        "force": {"stock": _rel_force_rmse(f_stock, f_ref),
                  "plugin": _rel_force_rmse(np.asarray(f_plugin), f_ref),
                  "prolate": _rel_force_rmse(np.asarray(f_prolate), f_ref)},
    }


# What each arm is read against when it fails. plugin and stock are the same
# splitting at the same settings, so each is the other's control; prolate is a
# different splitting, and the thing that says whether its number is a regression
# or just the splitting is whether the OTHER two moved with it.
_OTHER_IMPLEMENTATION = {"plugin": "stock", "stock": "plugin", "prolate": "plugin"}


@pytest.mark.parametrize("implementation", ["plugin", "stock", "prolate"])
def test_accuracy_energy(request, accuracy_results, implementation):
    """Electrostatic energy vs a converged reference at the same alpha."""
    r = accuracy_results
    rel_err = r["energy"][implementation]
    other = _OTHER_IMPLEMENTATION[implementation]
    record_observed(
        request,
        f"High-accuracy ref, {implementation} rel energy error, {r['name']}",
        _ACCURACY_ENERGY_TOL, rel_err)
    assert rel_err < _ACCURACY_ENERGY_TOL, (
        f"{r['name']}: {implementation} energy relative error {rel_err:.2e} exceeds "
        f"{_ACCURACY_ENERGY_TOL:.0e} against a reference at alpha={r['alpha']:.4f}/nm, "
        f"{_ACCURACY_FACTOR}x cutoff, grid "
        f"{'x'.join(str(_ACCURACY_FACTOR * g) for g in r['grid'])}, fp64. "
        f"{other} at the same production settings scores "
        f"{r['energy'][other]:.2e}; if both moved, the reference or the settings "
        f"changed rather than either implementation.")


@pytest.mark.parametrize("implementation", ["plugin", "stock", "prolate"])
def test_accuracy_force(request, accuracy_results, implementation):
    """Electrostatic forces vs a converged reference at the same alpha.

    The floor here is real physics, not noise: at the production cutoff and grid
    the PME truncation error is what OpenMM's default Ewald error tolerance buys,
    around 5e-4 relative. A splitting change that keeps the same total will land
    in the same place; one that does not will move this number well before it
    moves anything in the apples-to-apples tests.
    """
    r = accuracy_results
    rel_rmse = r["force"][implementation]
    other = _OTHER_IMPLEMENTATION[implementation]
    record_observed(
        request,
        f"High-accuracy ref, {implementation} rel force RMSE, {r['name']}",
        _ACCURACY_FORCE_TOL, rel_rmse)
    assert rel_rmse < _ACCURACY_FORCE_TOL, (
        f"{r['name']}: {implementation} relative force RMSE {rel_rmse:.2e} exceeds "
        f"{_ACCURACY_FORCE_TOL:.0e}. {other} at the same production settings scores "
        f"{r['force'][other]:.2e}; if both moved, the reference or the settings "
        f"changed rather than either implementation.")


# ─── van der Waals, the other half of the nonbonded total ────────────────────
#
# The electrostatic block above zeroes every epsilon, so nothing there measures
# Lennard-Jones at all. This block is its mirror: every CHARGE is zeroed instead.
#
# It needs no converged reference and no 2x anything, and that is the point.
# Truncated LJ is not split into a real-space and a reciprocal part, so there is
# no splitting parameter to converge and no decomposition to get wrong -- the
# cutoff does not approximate the model, the cutoff IS the model. Both sides
# therefore compute the SAME quantity at the SAME cutoff, and fp64 stock is an
# exact reference rather than a better-converged one.
#
# That also makes this the one nonbonded comparison that stays apples-to-apples
# while the Coulomb splitting is being replaced: with q = 0 there is no Ewald and
# no prolate, so a disagreement here cannot be blamed on the splitting and is a
# real LJ, exclusion, or 1-4 bug. It is the residual gap left by the block above,
# which defers LJ because doubling the cutoff would explicitly sum an r^-6 shell
# that production only approximates -- a convention gap worth ~3e-4. Zeroing the
# charges sidesteps that entirely: nothing is doubled, so nothing diverges.
#
# The dispersion correction is off. It is a mean-field analytic tail that adds
# energy and no force, cupo does not implement it, and leaving it on would put a
# constant offset in the energy comparison that has nothing to do with the pair
# math under test.
#
# Tolerances from measurement, not habit, and the measurement says the residual is
# fp32 rather than cupo: across the 11 systems the worst plugin numbers are 2.0e-6
# energy (cdk2) and 3.1e-5 force (apoa1), against stock's own 9.6e-7 and 4.6e-5 --
# and stock is the WORSE of the two on apoa1 force and on p38/ptp1b/jnk1 energy.
# Two implementations of the same unsplit function landing either side of each
# other at fp32 is the expected answer; anything real lands far above it, the
# dropped zero-charge exclusion this file also guards against having been ~2.4e-4.
_VDW_ENERGY_TOL = 1e-5        # 5.0x the worst observed plugin value
_VDW_FORCE_TOL = 1e-4         # 3.2x plugin's worst, 2.2x stock's


def _vdw_only(system):
    """Reduce a System in place to one NonbondedForce carrying LJ but no charge.

    The mirror of _electrostatics_only. Exceptions keep their epsilon and sigma,
    so 1-2/1-3 pairs stay fully excluded and 1-4 pairs keep their scaled LJ: the
    exclusion and correction machinery runs exactly as in production, only with
    the Coulomb half switched off. Bonded terms go for the same reason as there.

    Zeroing the charge on the PARTICLES is what makes this splitting-independent;
    zeroing it on the EXCEPTIONS too keeps the 1-4 correction from reintroducing a
    Coulomb term the particles no longer have.
    """
    found = [i for i in range(system.getNumForces())
             if isinstance(system.getForce(i), mm.NonbondedForce)]
    assert len(found) == 1, f"expected exactly one NonbondedForce, found {len(found)}"
    for idx in reversed(range(system.getNumForces())):
        if idx not in found:
            system.removeForce(idx)

    nb = system.getForce(0)
    for i in range(nb.getNumParticles()):
        _charge, sigma, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0, sigma, eps)
    for k in range(nb.getNumExceptions()):
        p1, p2, _charge_prod, sigma, eps = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, p1, p2, 0.0, sigma, eps)
    nb.setUseDispersionCorrection(False)


@pytest.fixture(scope="module", params=_DATASET_NAMES)
def vdw_accuracy_results(request, plugin_dir):
    """Plugin AND stock at production settings vs stock fp64, charges zeroed.

    Same shape as accuracy_results, with two differences that follow from LJ
    being unsplit: the reference sits at the SAME cutoff (there is nothing to
    converge), and stock fp32 is carried alongside for the same reason as there
    -- it says how much of any residual is simply fp32, so a cupo regression and
    a precision floor cannot be confused.
    """
    import nv_openmm_plugin

    name = request.param
    system, positions = _create_dataset_system(name)
    _vdw_only(system)

    # NOT pinned, unlike the electrostatic fixture, and deliberately: with every
    # charge zero OpenMM reports alpha = 0, and pinning it would both assert on a
    # degenerate 0 == 0 comparison and claim to fix a parameter that has nothing
    # left to govern. The reciprocal sum is identically zero here whatever alpha
    # and grid say, so all three runs agree on the only thing that matters -- the
    # cutoff, which is the LJ model itself and is carried in the serialized XML.
    system_xml = mm.XmlSerializer.serialize(system)

    reference = mm.XmlSerializer.deserialize(system_xml)
    e_ref, f_ref, _, _ = _stock_energy_forces(
        reference, positions, "double", verify_pinned=False)

    production = mm.XmlSerializer.deserialize(system_xml)
    e_stock, f_stock, _, _ = _stock_energy_forces(
        production, positions, "single", verify_pinned=False)

    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    replace_nonbonded(system_plugin)
    e_plugin, f_plugin = _get_energy_and_forces(
        system_plugin, positions, precision="single")

    return {
        "name": name,
        "energy": {"stock": _rel_energy_error(e_stock, e_ref),
                   "plugin": _rel_energy_error(e_plugin, e_ref)},
        "force": {"stock": _rel_force_rmse(f_stock, f_ref),
                  "plugin": _rel_force_rmse(np.asarray(f_plugin), f_ref)},
    }


@pytest.mark.parametrize("implementation", ["plugin", "stock"])
def test_vdw_energy(request, vdw_accuracy_results, implementation):
    """Lennard-Jones energy vs stock fp64 at the same cutoff, charges zeroed."""
    r = vdw_accuracy_results
    rel_err = r["energy"][implementation]
    other = _OTHER_IMPLEMENTATION[implementation]
    record_observed(
        request,
        f"vdW-only, {implementation} rel energy error, {r['name']}",
        _VDW_ENERGY_TOL, rel_err)
    assert rel_err < _VDW_ENERGY_TOL, (
        f"{r['name']}: {implementation} vdW energy relative error {rel_err:.2e} "
        f"exceeds {_VDW_ENERGY_TOL:.0e} against stock fp64 at the SAME cutoff. "
        f"There is no splitting here -- charges are zero -- so this is an LJ, "
        f"exclusion or 1-4 discrepancy. {other} scores {r['energy'][other]:.2e}.")


@pytest.mark.parametrize("implementation", ["plugin", "stock"])
def test_vdw_force(request, vdw_accuracy_results, implementation):
    """Lennard-Jones forces vs stock fp64 at the same cutoff, charges zeroed."""
    r = vdw_accuracy_results
    rel_rmse = r["force"][implementation]
    other = _OTHER_IMPLEMENTATION[implementation]
    record_observed(
        request,
        f"vdW-only, {implementation} rel force RMSE, {r['name']}",
        _VDW_FORCE_TOL, rel_rmse)
    assert rel_rmse < _VDW_FORCE_TOL, (
        f"{r['name']}: {implementation} vdW relative force RMSE {rel_rmse:.2e} "
        f"exceeds {_VDW_FORCE_TOL:.0e} against stock fp64 at the SAME cutoff. "
        f"There is no splitting here -- charges are zero -- so this is an LJ, "
        f"exclusion or 1-4 discrepancy. {other} scores {r['force'][other]:.2e}.")


# ─── Net-charge (PME plasma) correction ──────────────────────────────────────


def _create_net_charged_box(net_charge=15.0, n_atoms=256, seed=7):
    """Charged LJ box carrying a deliberate net charge (Sum q = net_charge).

    A non-neutral cell exercises the k = 0 background ("plasma") correction,
    E = G_0 (Sum q)^2 k_e / 2V, an energy-only (zero-force) term. Without it the
    plugin energy carries a constant offset ~ (Sum q)^2 / V that the neutral
    water/tyk2/bace systems cannot expose.

    net_charge is 15 rather than a physical 1-3, and the reason is sensitivity
    rather than realism. G_0 is the finite part of the SPLITTING's Green's
    function at k -> 0 and is NOT the same constant for every method -- Ewald's
    -pi/alpha^2 is 24% smaller in magnitude than the prolate P=1 one at a 0.9 nm
    cutoff -- so what this test has to separate is one wrong constant from
    another, not a missing term from a present one. The offset scales as
    (Sum q)^2 while the splitting's own truncation error does not, and at Sum q = 3
    the two are the SAME SIZE (measured: 1.9e-4 offset against a 1.4e-4 floor, a
    1x margin). At 15 the offset is 5.8e-3 against the same floor, a 38x margin.
    """
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(n_atoms ** (1.0 / 3.0)))
    spacing = 0.45  # nm
    box_nm = n_side * spacing

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0, 0) * unit.nanometers,
        mm.Vec3(0, box_nm, 0) * unit.nanometers,
        mm.Vec3(0, 0, box_nm) * unit.nanometers)

    charges = rng.normal(size=n_atoms)
    charges -= charges.mean()            # neutralize, then...
    charges += net_charge / n_atoms      # ...impose exactly Sum q = net_charge

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometers)
    nb.setUseDispersionCorrection(True)
    for i in range(n_atoms):
        system.addParticle(39.948)
        nb.addParticle(charges[i], 0.34, 1.0)
    system.addForce(nb)

    positions = []
    for i in range(n_atoms):
        ix = i % n_side
        iy = (i // n_side) % n_side
        iz = i // (n_side * n_side)
        p = np.array([ix, iy, iz]) * spacing + 0.5 * spacing + rng.uniform(-0.05, 0.05, 3)
        positions.append(mm.Vec3(*p) * unit.nanometers)
    return system, positions


# Measured on this box, per method: PME lands at ~1.0e-6 (it shares OpenMM's
# formula exactly) while Prolate lands at ~1.5e-4, which is that splitting's own
# truncation error against a stock-Ewald reference and not an error in the
# background term. Using the WRONG constant for either method costs 5.8e-3 here,
# so both tolerances sit well clear of the floor and well below the failure.
_NET_CHARGE_TOL = {"PME": 1e-5, "Prolate": 1e-3}


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_net_charge_correction(request, plugin_dir, precision, method):
    """Plugin energy for a NON-NEUTRAL cell must match OpenMM double.

    Regression guard for the k = 0 background (plasma) correction. Without it a
    net-charged system carries a constant energy offset (zero force), largest for
    small boxes -- the neutral systems in this module cannot catch it.

    PARAMETRIZED OVER THE METHOD, and that is the point. This test predates the
    second splitting and ran only the converting constructor's default, which is
    PME; the prolate background constant therefore had no coverage anywhere, and
    it shipped carrying Ewald's -pi/alpha^2. The failure was silent twice over: it
    moves no force, and every bundled benchmark except dhfr and apoa1 is neutral,
    so the term is identically zero in them.
    """
    import nv_openmm_plugin

    net_charge = 15.0
    system, positions = _create_net_charged_box(net_charge=net_charge)
    system_xml = mm.XmlSerializer.serialize(system)

    e_ref, _ = _get_energy_and_forces(system, positions, precision="double")

    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    replace_nonbonded(system_plugin,
                      method=getattr(nv_openmm_plugin.NonbondedForce, method))
    e_plugin, _ = _get_energy_and_forces(system_plugin, positions, precision=precision)

    rel_err = abs(e_plugin - e_ref) / abs(e_ref)
    tol = _NET_CHARGE_TOL[method]
    record_observed(
        request,
        f"Net-charged energy rel err, {method}, {precision}",
        tol, rel_err)
    assert rel_err < tol, (
        f"Net-charged (Sum q={net_charge:g}) {method} energy rel error "
        f"{rel_err:.2e} exceeds {tol:.0e} (double={e_ref:.4f}, "
        f"plugin={e_plugin:.4f}); is the k=0 background correction missing, or "
        f"carrying the other method's constant?")


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_net_charge_correction_tracks_the_box_volume(request, plugin_dir, method):
    """The background term is 1/V, so a box change must re-evaluate it.

    Under NVT the term is constant, which makes computing it once look free. It is
    not: under a barostat V moves every accepted step, and a frozen volume is
    wrong by the box's fractional volume change -- silently, energy-only, and only
    for charged systems. Nothing else in the suite covers this pairing; every NPT
    test here builds pure water, where Sum q == 0 makes the term identically zero
    at any volume.

    Stock OpenMM is the reference because it demonstrably tracks the volume: it
    reads the live box inside its energy call on every platform (common
    CommonCalcNonbondedForce.cpp:892-898, reference ReferenceLJCoulombIxn.cpp:231,
    cpu CpuKernels.cpp:717) and caches only the volume-independent charge sum.

    The box is changed WITHOUT moving the particles, so the reference and the
    plugin see the same configuration in two different cells and the comparison
    stays exact. A plugin that froze the volume would agree at the first box and
    disagree at the second by the background term's fractional change.
    """
    import nv_openmm_plugin

    system, positions = _create_net_charged_box(net_charge=15.0)
    system_xml = mm.XmlSerializer.serialize(system)
    edge = system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(unit.nanometers)

    def _energy_at(scale, plugin):
        s = mm.XmlSerializer.deserialize(system_xml)
        if plugin:
            replace_nonbonded(
                s, method=getattr(nv_openmm_plugin.NonbondedForce, method))
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        ctx = mm.Context(s, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "double" if not plugin else "mixed",
                          "DeterministicForces": "true"})
        ctx.setPositions(positions)
        if scale != 1.0:
            L = edge * scale
            ctx.setPeriodicBoxVectors(mm.Vec3(L, 0, 0) * unit.nanometers,
                                      mm.Vec3(0, L, 0) * unit.nanometers,
                                      mm.Vec3(0, 0, L) * unit.nanometers)
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole)
        del ctx, integrator
        return e

    # 1.15 linear is a 1.52x volume change -- comfortably more than a barostat
    # step, so a frozen volume shows up as a large discrepancy rather than a
    # marginal one, while staying inside the plugin's NVT cell-grid envelope.
    tol = _NET_CHARGE_TOL[method]
    worst = 0.0
    for scale in (1.0, 1.15):
        e_ref = _energy_at(scale, plugin=False)
        e_plugin = _energy_at(scale, plugin=True)
        rel = abs(e_plugin - e_ref) / abs(e_ref)
        worst = max(worst, rel)
        assert rel < tol, (
            f"{method} at box scale {scale}: energy rel error {rel:.2e} exceeds "
            f"{tol:.0e} (double={e_ref:.4f}, plugin={e_plugin:.4f}). Agreeing at "
            f"scale 1.0 and failing here means the 1/V background term was "
            f"computed once and cached rather than re-evaluated for the new box.")
    record_observed(
        request, f"Net-charged energy across a box change, {method}",
        tol, worst)


def _create_zero_charge_lj_dimer_box(n_dimers=128, seed=11):
    """Bonded dimers whose atoms carry ZERO charge but real LJ.

    The one property no PDB bench dataset has, and the only one that exposes a
    dropped exclusion: a full 1-2 exclusion carries chargeProd=0 AND
    epsilon=0 by construction, so if the pair is registered on the strength of
    its charge product alone, a zero-charge dimer is not registered at all --
    and the direct kernel, which takes LJ from the PARTICLE parameters rather
    than the exception's, evaluates its LJ and never has it cancelled.

    The dimers sit at 0.16 nm, well inside sigma, so an uncancelled 1-2 LJ is
    enormous rather than subtle. Charged filler particles keep PME meaningful
    and the cell neutral.
    """
    rng = np.random.default_rng(seed)
    spacing = 0.55                       # nm between lattice sites
    n_side = 6                           # 216 sites
    box_nm = n_side * spacing            # 3.3 nm -> 3 cells at a 0.9 nm cutoff

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0, 0) * unit.nanometers,
        mm.Vec3(0, box_nm, 0) * unit.nanometers,
        mm.Vec3(0, 0, box_nm) * unit.nanometers)

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometers)
    nb.setUseDispersionCorrection(True)

    # Everything on the lattice: a random placement clashes, and a clash drives
    # the force past cupo's int32 accumulator range, which would fail this test
    # for an unrelated reason. Nearest inter-site atom separation is
    # 0.55 - 0.16 = 0.39 nm, comfortably outside sigma.
    positions = []
    bonds = []
    filler_idx = []
    for s in range(n_side ** 3):
        ix = s % n_side
        iy = (s // n_side) % n_side
        iz = s // (n_side * n_side)
        centre = np.array([ix, iy, iz]) * spacing + 0.5 * spacing
        centre = centre + rng.uniform(-0.02, 0.02, 3)
        if s % 5 == 0:                   # charged filler: exercises the reciprocal sum
            filler_idx.append(len(positions))
            system.addParticle(39.948)
            nb.addParticle(0.0, 0.34, 1.0)          # charge patched below
            positions.append(mm.Vec3(*centre) * unit.nanometers)
        else:                            # zero-charge LJ dimer: the case under test
            a = len(positions)
            for off in (-0.08, +0.08):   # 0.16 nm apart, deep inside sigma
                system.addParticle(12.0)
                nb.addParticle(0.0, 0.32, 0.8)      # q = 0 EXACTLY, real LJ
                positions.append(
                    mm.Vec3(*(centre + np.array([off, 0.0, 0.0]))) * unit.nanometers)
            bonds.append((a, a + 1))

    q = rng.normal(size=len(filler_idx))
    q -= q.mean()                        # keep the cell neutral
    for k, idx in enumerate(filler_idx):
        _, sig, eps = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, q[k], sig, eps)

    # Full 1-2 exclusions: chargeProd = 0, epsilon = 0 -- exactly the shape that
    # a charge-product-only registration test throws away.
    for i, j in bonds:
        nb.addException(i, j, 0.0, 0.32, 0.0)

    system.addForce(nb)
    return system, positions


# WHY THE TEST BELOW IS SKIPPED ON THIS BRANCH, and why that is NOT the old bug
# coming back. The dimers carry zero charge, but the box does not: the filler
# particles above are charged on purpose, to keep PME meaningful. So this system's
# total still contains a Coulomb part, and prolate-vs-Ewald moves it.
#
# Measured, by zeroing the FILLER charges too and re-running the same comparison:
#
#     as shipped (charged fillers)   E_rel = 2.40e-04   F_rmse = 6.57e-04
#     every charge zeroed            E_rel = 7.36e-08   F_rmse = 4.55e-05
#
# The second line is this test's actual subject -- a full 1-2 exclusion between two
# zero-charge atoms having its LJ cancelled -- and it is intact to 7.4e-8. The
# first line is the charged fillers' electrostatics, measured against an Ewald
# reference. The guard still holds; only the reference around it moved. The
# vdW block above covers the same LJ/exclusion machinery on all 11 real systems
# and is NOT skipped, so this branch is not flying blind on it.


@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_zero_charge_exclusion_is_cancelled(request, plugin_dir, precision):
    """A full exclusion between two ZERO-CHARGE atoms must still cancel its LJ.

    Regression guard for a silent-wrong-forces bug: replace_nonbonded() registered an
    exception only when |qi*qj|, |chargeProd| or the EXCEPTION's epsilon was
    non-zero. A full 1-2/1-3 exclusion has the latter two zero by construction,
    so an excluded pair whose charge product was zero was dropped entirely --
    while the pair kernel still evaluated its LJ from the particle parameters.
    Found on cellulose (30240 zero-charge LJ atoms, 116640 such pairs), where
    forces came out 79x wrong; every PDB dataset here has exactly zero of them.
    """
    import nv_openmm_plugin

    system, positions = _create_zero_charge_lj_dimer_box()
    system_xml = mm.XmlSerializer.serialize(system)

    e_ref, f_ref = _get_energy_and_forces(system, positions, precision="double")

    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    _replace_with_pme(system_plugin)
    e_plugin, f_plugin = _get_energy_and_forces(system_plugin, positions,
                                                precision=precision)

    rel_err = abs(e_plugin - e_ref) / abs(e_ref)
    d = np.linalg.norm(f_plugin - f_ref, axis=1)
    rel_rmse = np.sqrt((d ** 2).mean()) / np.sqrt((np.linalg.norm(f_ref, axis=1) ** 2).mean())

    tol_e, tol_f = 1e-5, 1e-5     # observed 6.4e-7 and 9.2e-7
    record_observed(request, f"Zero-charge exclusion energy rel err, {precision}",
                    tol_e, rel_err)
    record_observed(request, f"Zero-charge exclusion force rel rmse, {precision}",
                    tol_f, rel_rmse)
    assert rel_err < tol_e, (
        f"Zero-charge-exclusion energy rel error {rel_err:.2e} exceeds {tol_e:.0e} "
        f"(double={e_ref:.4f}, plugin={e_plugin:.4f}); is a full exclusion whose "
        f"charge product is zero being dropped instead of having its LJ cancelled?")
    assert rel_rmse < tol_f, (
        f"Zero-charge-exclusion force rel rmse {rel_rmse:.2e} exceeds {tol_f:.0e}; "
        f"an uncancelled 1-2 LJ at 0.16 nm is the expected cause.")


def _create_distant_exception_box(box_nm=5.0, sep_nm=2.4, n_filler=60, seed=7):
    """A PME box whose only exception joins two particles FAR apart.

    Nothing in OpenMM requires an exception's particles to be bonded neighbours,
    and cupo's cancellation kernel has a distinct geometry path for the case where
    the two atoms' cells are more than one apart. Every bench system misses it --
    their exclusions are all 1-2/1-3/1-4 pairs, which are always within a cell of
    each other -- so without a contrived box that path is never executed at all.

    box/floor(box/cutoff) = 5.0/5 = 1.0 nm cells (the cell list requires cells
    strictly wider than the 0.9 nm cutoff), and 2.4 nm of separation puts the pair
    two cells apart under the minimum image: outside the cutoff, so the pair kernel
    never evaluates it and there is nothing to cancel -- but reciprocal space still
    deposited its erf term, so the correction must still be applied.
    """
    rng = np.random.default_rng(seed)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_nm, 0, 0) * unit.nanometers,
        mm.Vec3(0, box_nm, 0) * unit.nanometers,
        mm.Vec3(0, 0, box_nm) * unit.nanometers)

    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometers)

    positions = []
    # The pair under test: oppositely charged so its reciprocal-space term is large
    # enough that getting the correction wrong cannot hide inside the tolerance.
    for x, q in ((0.6, +0.8), (0.6 + sep_nm, -0.8)):
        system.addParticle(12.0)
        nb.addParticle(q, 0.32, 0.5)
        positions.append(mm.Vec3(x, 0.5 * box_nm, 0.5 * box_nm) * unit.nanometers)

    # Filler so the reciprocal sum has real work; net charge zero.
    q = rng.normal(size=n_filler)
    q -= q.mean()
    for k in range(n_filler):
        system.addParticle(39.948)
        nb.addParticle(q[k], 0.34, 0.6)
        positions.append(mm.Vec3(*rng.uniform(0.1, box_nm - 0.1, 3)) * unit.nanometers)

    # A PURE exclusion (chargeProd 0, epsilon 0) between the two distant particles.
    nb.addException(0, 1, 0.0, 0.32, 0.0)
    system.addForce(nb)
    return system, positions


@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_exclusion_between_distant_cells(request, plugin_dir, precision):
    """An exclusion whose particles sit in non-adjacent cells must still be corrected.

    Regression guard for a silent-wrong-energy bug in the fused exclusion kernel.
    The cancellation and the correction do NOT have the same scope: cancellation
    applies only where the pair kernel evaluated (adjacent cells, inside the
    cutoff), while the correction applies at every distance, because reciprocal
    space deposits q_i*q_j*erf(alpha*r)/r for an excluded pair however far apart it
    is. The non-adjacent case therefore takes its own geometry path.

    That path first shipped computing a minimum image over posq_sorted, which is
    wrong because those coordinates are CELL-RELATIVE (each in [0, cell_size)) --
    subtracting two of them gives a sub-cell number no box wrap can repair, so the
    correction was applied at r ~ 0.5 nm instead of 2.4 nm and the energy came out
    300% high. No bench system detects it: every exclusion they contain is a bonded
    pair whose atoms share a cell or a neighbour.

    2.4 nm is deliberately just UNDER half the 5.0 nm box: the minimum image and the
    unwrapped separation are the same number there, so cupo and OpenMM must agree
    exactly. Past L/2 they stop being the same number and cupo refuses the job --
    see test_exclusion_beyond_half_box_fails_hard.
    """
    import nv_openmm_plugin

    system, positions = _create_distant_exception_box(sep_nm=2.4)
    system_xml = mm.XmlSerializer.serialize(system)

    e_ref, f_ref = _get_energy_and_forces(system, positions, precision="double")

    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    _replace_with_pme(system_plugin)
    e_plugin, f_plugin = _get_energy_and_forces(system_plugin, positions,
                                                precision=precision)

    rel_err = abs(e_plugin - e_ref) / abs(e_ref)
    d = np.linalg.norm(f_plugin - f_ref, axis=1)
    rel_rmse = np.sqrt((d ** 2).mean()) / np.sqrt((np.linalg.norm(f_ref, axis=1) ** 2).mean())

    # The energy tolerance stays at 1e-4 while the force drops two decades: this
    # box is 62 particles, and the excluded pair's own correction is a large
    # fraction of a small total, so its relative energy error is 4.4e-5 -- already
    # within 2.3x of the line. The force RMSE has no such concentration.
    tol_e, tol_f = 1e-4, 1e-5     # observed 4.4e-5 and 2.8e-6
    record_observed(request, f"Distant-exclusion energy rel err, {precision}",
                    tol_e, rel_err)
    record_observed(request, f"Distant-exclusion force rel rmse, {precision}",
                    tol_f, rel_rmse)
    assert rel_err < tol_e, (
        f"Distant-exclusion energy rel error {rel_err:.2e} exceeds {tol_e:.0e} "
        f"(double={e_ref:.4f}, plugin={e_plugin:.4f}); is the correction for a pair "
        f"in non-adjacent cells being applied at the wrong separation?")
    assert rel_rmse < tol_f, (
        f"Distant-exclusion force rel rmse {rel_rmse:.2e} exceeds {tol_f:.0e}.")


def test_exclusion_beyond_half_box_agrees_with_stock(plugin_dir):
    """An excluded pair spanning more than half the box, with both engines on the
    same convention.

    This used to abort. OpenMM computed an exception from the RAW coordinate
    difference -- setExceptionsUsePeriodicBoundaryConditions defaults to False --
    while cupo, seeing only positions its cell list had already wrapped, could form
    nothing but the minimum image. Under L/2 those are the same number; over it they
    are different PAIRS, and on this box (5.0 nm, pair at 3.0 nm) the two answers
    differ by 34%. The kernel trapped rather than pick one.

    This parity test explicitly enables periodic exceptions on both forces. The
    independent raw-versus-periodic physical replacement contract is covered by
    test_exception_bonded.py.
    """
    system, positions = _create_distant_exception_box(sep_nm=3.0)   # box 5.0 -> over L/2
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            force.setExceptionsUsePeriodicBoundaryConditions(True)
    system_xml = mm.XmlSerializer.serialize(system)

    stock = mm.XmlSerializer.deserialize(system_xml)
    e_stock, f_stock = _get_energy_and_forces(stock, positions, precision="mixed")

    system_plugin = mm.XmlSerializer.deserialize(system_xml)
    _replace_with_pme(system_plugin)
    e_plugin, f_plugin = _get_energy_and_forces(system_plugin, positions,
                                                precision="mixed")

    rel = abs(e_plugin - e_stock) / abs(e_stock)
    assert rel < 1e-4, (
        f"plugin {e_plugin:.4f} vs stock {e_stock:.4f} kJ/mol, relative {rel:.2e}: "
        "with both engines on the minimum image an exclusion past L/2 must agree, "
        "so the correction is not being applied at the image OpenMM used")
    rmse = _rel_force_rmse(f_plugin, f_stock)
    assert rmse < 1e-3, f"force rel RMSE {rmse:.2e} on the same pair"


# ─── Determinism (run-to-run bitwise reproducibility) ────────────────────────


@pytest.mark.parametrize("precision", ["single", "mixed"])
@pytest.mark.parametrize("system_name", [n for n in ("water", "tyk2", "bace") if n in _SYSTEMS])
@pytest.mark.parametrize("convolution", ["off", "auto"], ids=["cufft", "compact"])
def test_determinism(request, plugin_dir, monkeypatch, system_name, precision,
                     convolution):
    """Plugin total potential energy AND forces must be bit-identical run-to-run.

    cupo's nonbonded is deterministic (forces via int64/int32 fixed-point
    accumulation; energy via int64 towerplate + int32 fixed-point SPME charge
    grid). The grouped bonded backend and its stock fallback both accumulate
    forces in the fixed-point long-long buffer and reduce energies in a fixed
    order. Total PE and the full force array are therefore
    bitwise-reproducible across independent Context evaluations of identical
    positions using the same numerical algorithm.  Timing-based convolution
    selection can choose different FFT algorithms in separate Contexts, so
    exercise each backend with that selection disabled.  Automatic selection
    is covered separately by test_compact_policy.py.
    (This does NOT depend on OpenMM's DeterministicForces property,
    which on CUDA only gates PME charge spreading -- a path OpenMM no longer runs
    here since cupo owns the whole nonbonded.)
    """
    import nv_openmm_plugin

    monkeypatch.setenv("CUPO_COMPACT_CONVOLUTION", convolution)
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    system, positions = _SYSTEMS[system_name]()
    replace_nonbonded(system)

    n_runs = 5
    energies = []
    forces = []
    for _ in range(n_runs):
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        platform = mm.Platform.getPlatformByName("CUDA")
        context = mm.Context(system, integrator, platform, {"Precision": precision})
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole))
        forces.append(np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer)))
        del context, integrator

    # Energy: bitwise-identical total PE across runs.
    e_spread = max(energies) - min(energies)
    e_distinct = sorted({e.hex() for e in energies})
    record_observed(
        request, f"Plugin total-PE run-to-run spread, {system_name}/{precision}",
        0.0, e_spread)

    # Forces: bitwise-identical force array across runs (compare raw bytes vs
    # run 0; max component deviation is the recorded/reported metric).
    f_distinct = len({f.tobytes() for f in forces})
    f_max_dev = max(float(np.abs(f - forces[0]).max()) for f in forces[1:])
    record_observed(
        request, f"Plugin force run-to-run max deviation, {system_name}/{precision}",
        0.0, f_max_dev)

    assert len(e_distinct) == 1, (
        f"Non-deterministic total PE: {len(e_distinct)} distinct values over "
        f"{n_runs} runs (spread {e_spread:.3e} kJ/mol), hex={e_distinct}")
    assert f_distinct == 1, (
        f"Non-deterministic forces: {f_distinct} distinct force arrays over "
        f"{n_runs} runs (max component deviation {f_max_dev:.3e} kJ/mol/nm)")


@pytest.mark.parametrize("precision", ("mixed", "single"))
def test_update_params_matches_construction(request, plugin_dir, precision):
    """Changing parameters in a live Context must equal building with them.

    Regression guard for three quantities that were derived once, at Context
    creation, and then never refreshed -- so a changed charge or sigma kept
    reporting the old system's energy, silently and by a lot (~48000 kJ/mol on
    DHFR for halved charges):

      * the per-pair q_i*q_j the exclusion correction carries, and the 4*epsilon
        the kernel's LJ form wants. Both used to be baked into NvNonbondedForce, which made
        them stale the instant a particle or exception parameter moved. NvNonbondedForce
        now holds only the raw exception parameters, as NonbondedForce does, and
        both derived values are formed at upload -- then re-INGESTED via
        cupo_dspace_set_exclusion_params, because that call keeps its own
        permuted, Coulomb-scaled copy that a bare device memcpy never reaches.
      * the Ewald self-energy and net charge, both sums over the charges.
      * the long-range dispersion coefficient, a function of the sigmas/epsilons.

    Scaling charges, particle epsilons and exception parameters separately keeps
    the failure modes distinguishable: charges alone exercise the pair product and
    the two charge sums, particle epsilons the dispersion coefficient, and
    exception parameters the raw-exception path that setExceptionParameters made
    reachable at all.
    """
    import nv_openmm_plugin
    nvp = nv_openmm_plugin

    def _system(q_scale, e_scale, exc_scale=1.0):
        system, positions = _create_water_box()
        nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        if (q_scale, e_scale) != (1.0, 1.0):
            for i in range(nb.getNumParticles()):
                q, sig, eps = nb.getParticleParameters(i)
                nb.setParticleParameters(i, q * q_scale, sig, eps * e_scale)
        if exc_scale != 1.0:
            for k in range(nb.getNumExceptions()):
                p1, p2, qq, sig, eps = nb.getExceptionParameters(k)
                nb.setExceptionParameters(k, p1, p2, qq * exc_scale, sig, eps * exc_scale)
        idx = [i for i, f in enumerate(system.getForces())
               if isinstance(f, mm.NonbondedForce)][0]
        force = nvp.NonbondedForce(nb, 1.0)
        system.addForce(force)
        system.removeForce(idx)
        return system, positions, force

    def _context(system, positions):
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        context = mm.Context(system, integrator,
                             mm.Platform.getPlatformByName("CUDA"),
                             {"Precision": precision})
        context.setPositions(positions)
        return context, integrator

    # Updated before the first energy evaluation as well as after it. Before it the
    # device does not exist yet, and the update used to be dropped: the first
    # evaluation then built the device from the old parameters (a reported bug, whose
    # workaround was an energy evaluation before the update).
    for evaluate_first in (True, False):
        for q_scale, e_scale, exc_scale in ((0.5, 1.0, 1.0), (1.0, 2.0, 1.0),
                                            (1.0, 1.0, 0.5), (0.7, 1.3, 1.4)):
            system, positions, _ = _system(q_scale, e_scale, exc_scale)
            ctx, integ = _context(system, positions)
            reference = ctx.getState(getEnergy=True).getPotentialEnergy(
            ).value_in_unit(unit.kilojoule_per_mole)
            del ctx, integ

            system, positions, force = _system(1.0, 1.0)
            ctx, integ = _context(system, positions)
            if evaluate_first:
                ctx.getState(getEnergy=True)      # the device exists before the update
            for i in range(force.getNumParticles()):
                q, sig, eps = force.getParticleParameters(i)
                force.setParticleParameters(i, q * q_scale, sig, eps * e_scale)
            for k in range(force.getNumExceptions()):
                p1, p2, qq, sig, eps = force.getExceptionParameters(k)
                force.setExceptionParameters(k, p1, p2, qq * exc_scale, sig,
                                             eps * exc_scale)
            force.updateParametersInContext(ctx)
            updated = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)
            del ctx, integ

            rel = abs(updated - reference) / abs(reference)
            record_observed(request, "updateParametersInContext vs construction",
                            1e-6, rel)
            assert rel < 1e-6, (
                f"q*{q_scale} eps*{e_scale} exc*{exc_scale}, "
                f"{'after' if evaluate_first else 'before'} the first evaluation: "
                f"updateParametersInContext gave {updated:.6f} kJ/mol, building with "
                f"the same parameters gave {reference:.6f} (rel {rel:.3e})")


def _minimal_pme_force(system):
    """A 3-particle PME NonbondedForce with one exception, added to `system`."""
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometers)
    for q in (0.1, -0.1, 0.0):
        system.addParticle(1.0)
        nb.addParticle(q, 0.3, 0.5)
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    system.addForce(nb)
    return nb


def _empty_periodic_system():
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(3, 0, 0) * unit.nanometers,
        mm.Vec3(0, 3, 0) * unit.nanometers,
        mm.Vec3(0, 0, 3) * unit.nanometers)
    return system


def test_unsupported_features_are_rejected(plugin_dir):
    """Everything NvNonbondedForce cannot model must fail loudly at conversion.

    Each of these produces plausible-looking numbers if accepted, which is the
    reason they are checked at all rather than left to the physics:

      * unsupported nonperiodic and non-mesh Ewald methods need different kernels.

    Global parameters and particle/exception offsets are supported; their live
    behavior is exercised in test_parameter_offsets.py.
    """
    import nv_openmm_plugin

    for method in (mm.NonbondedForce.NoCutoff, mm.NonbondedForce.CutoffNonPeriodic,
                   mm.NonbondedForce.Ewald):
        system = _empty_periodic_system()
        _minimal_pme_force(system).setNonbondedMethod(method)
        with pytest.raises(Exception, match="only PME or LJPME source forces are supported"):
            replace_nonbonded(system)

    # A global parameter on its own offsets nothing, so it must NOT be rejected.
    system = _empty_periodic_system()
    _minimal_pme_force(system).addGlobalParameter("lambda", 0.0)
    replace_nonbonded(system)


def test_the_default_splitting_is_prolates(plugin_dir):
    """Both constructors default to Prolate, and that is a shipping decision.

    Worth a test of its own because nothing else in the suite fails if it moves.
    A flip back to PME would leave every test here passing -- the stock-comparison
    arms name PME explicitly, and the arms that do not name a method are the ones
    chosen for being splitting-independent -- while every user silently got a
    ~5x larger reciprocal mesh and the speed that goes with it.

    Checked on the CONVERTING constructor too, not just the hand-assembled one.
    That is the one nearly every caller reaches, its method argument sits last
    behind boxSafetyFactor, and a source force is required to be PME or LJPME -- three
    separate reasons its default could drift back to PME without anyone noticing.
    """
    import nv_openmm_plugin as nv

    assert nv.NonbondedForce().getNonbondedMethod() == nv.NonbondedForce.Prolate

    system = _empty_periodic_system()
    source = _minimal_pme_force(system)
    assert (nv.NonbondedForce(source).getNonbondedMethod()
            == nv.NonbondedForce.Prolate)
    assert (nv.NonbondedForce(source, 1.0).getNonbondedMethod()
            == nv.NonbondedForce.Prolate)
    # ... and asking for the reference arm still gets it.
    assert (nv.NonbondedForce(source, 1.0, nv.NonbondedForce.PME)
            .getNonbondedMethod() == nv.NonbondedForce.PME)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_pinned_pme_parameters_are_refused_not_ignored(plugin_dir, method):
    """A pinned (alpha, mesh) has nowhere to go, so it must not go quietly.

    cupo derives both -- alpha from the cutoff at the one implemented tolerance,
    the mesh from the box at Context creation -- and NvNonbondedForce has no field
    for either. That leaves refuse or drop, and dropping is the failure this class
    is built around: the caller reads back the parameters it set, on an object that
    is not the one computing, and never learns that the run used others.

    Refused under BOTH methods, and the asymmetry is the point. Under PME a pin is
    at least the right KIND of quantity and could look honoured by accident; under
    Prolate there is no alpha at all and the mesh answers a different question. A
    knob one method honoured and the other ignored is exactly what the fixed error
    tolerance already removed.
    """
    import nv_openmm_plugin
    nb_method = getattr(nv_openmm_plugin.NonbondedForce, method)

    # A mesh cannot be checked here at all -- it is a function of the box, and the
    # converting constructor has no box -- so any mesh is refused, matching or not.
    system = _empty_periodic_system()
    _minimal_pme_force(system).setPMEParameters(0.0, 32, 32, 32)
    with pytest.raises(Exception, match="pins a PME mesh"):
        replace_nonbonded(system, method=nb_method)

    # An alpha that asks for a different splitting than the cutoff implies.
    system = _empty_periodic_system()
    _minimal_pme_force(system).setPMEParameters(1.5, 0, 0, 0)
    with pytest.raises(Exception, match="pins an Ewald alpha"):
        replace_nonbonded(system, method=nb_method)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_an_agreeing_alpha_still_converts(plugin_dir, method):
    """The ordinary pinned force must keep converting.

    Real systems carry an alpha OpenMM derived and something pinned back, and that
    value agrees with what cupo derives because it is the same rule at the same
    tolerance -- so refusing every pin would reject systems this plugin runs
    correctly. Only a DIFFERENT request is an error.

    The expected value is recomputed here from its published form rather than read
    back, so this also pins cupo's derivation to OpenMM's:
    alpha = sqrt(-ln(2*tol)) / rc at tol = 5e-4.
    """
    import nv_openmm_plugin
    nb_method = getattr(nv_openmm_plugin.NonbondedForce, method)
    cutoff = 0.9
    alpha = math.sqrt(-math.log(2.0 * 5e-4)) / cutoff

    system = _empty_periodic_system()
    _minimal_pme_force(system).setPMEParameters(alpha, 0, 0, 0)
    replace_nonbonded(system, method=nb_method)      # must not raise

    # ... and the tolerance on "agrees" is tight enough to be about representation
    # and not about physics: 0.1% off is a different splitting.
    system = _empty_periodic_system()
    _minimal_pme_force(system).setPMEParameters(alpha * 1.001, 0, 0, 0)
    with pytest.raises(Exception, match="pins an Ewald alpha"):
        replace_nonbonded(system, method=nb_method)
