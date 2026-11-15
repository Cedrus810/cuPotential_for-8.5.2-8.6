# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenMM Nvidia plugin sample — correctness + speedup on standard MD systems.

Mirrors the canonical OpenMM benchmark setup:
  AMBER14, PME 0.9 nm cutoff, Ewald tolerance 5e-4,
  HBonds constrained, rigid water, H mass repartitioning (1.5 amu),
  LangevinMiddle 4 fs, constraint tolerance 1e-5, mixed precision.

For each system (apoa1, bace, dhfr):
  1. Evaluates energy/forces: plugin vs stock OpenMM (mixed precision)
  2. Times 3000 steps of LangevinMiddle dynamics for both, prints speedup

bace comes with the sample, in data/bace/. apoa1 and dhfr are OpenMM's own
benchmark systems and do not: they are read from the installed OpenMM
(conda-forge's openmm installs them under share/openmm/examples/benchmarks/), or,
when it has none, downloaded once from OpenMM's GitHub repository and checked by
SHA-256 (nv_openmm_plugin._benchmark_systems).

Runs each system under both NVT and NPT ensembles.

Exit 0 + prints PASS on success.
"""

import gzip
import sys
import time
from pathlib import Path

import numpy as np

import openmm as mm
import openmm.app as app
import openmm.unit as unit

import nv_openmm_plugin as nv
# The plugin's own finder for OpenMM's benchmark files, the one its tests use.
from nv_openmm_plugin._benchmark_systems import SYSTEMS as OPENMM_SYSTEMS, openmm_benchmark_file


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

SYSTEMS = ["apoa1", "bace", "dhfr"]

CUTOFF_NM = 0.9

# ACCURACY IS MEASURED AGAINST A CONVERGED REFERENCE, not against stock OpenMM at
# the same settings. Pinning both sides to one cutoff, alpha and PME grid only ever
# answers "does this reproduce OpenMM's PME", which is the wrong question for an
# implementation free to decompose the Coulomb sum differently -- it would score a
# more accurate answer as an error. So the reference here holds alpha fixed and
# pushes both truncations out: twice the cutoff, twice the PME grid per dimension,
# fp64. At fixed alpha that converges to the exact Coulomb lattice sum, and what
# remains is the truncation error of the production settings, expected to land near
# OpenMM's own default Ewald tolerance (5e-4) rather than at zero.
#
# Stock is measured against that same reference and printed next to the plugin,
# because a number with nothing beside it cannot be read: it is what says how much
# these production settings are worth in the first place.
#
# The two halves are checked separately because they need different references.
# Electrostatics is split into a real-space and a reciprocal part, so it converges
# as above. Lennard-Jones is not split -- it is plain-truncated, so the cutoff does
# not approximate the model, the cutoff IS the model -- which means doubling it
# would explicitly sum an r^-6 shell that production only approximates, comparing
# two different physics. Its reference is therefore fp64 at the SAME cutoff, which
# is exact rather than better-converged.
ACCURACY_FACTOR = 2
ENERGY_TOL = 1e-4        # electrostatics, vs the converged reference
FORCE_TOL = 1e-3
VDW_ENERGY_TOL = 1e-5    # van der Waals, vs fp64 at the same cutoff
VDW_FORCE_TOL = 1e-4

# NPT (--npt). Nothing plugin-specific left here: the cell-grid headroom a
# barostat needs is chosen by the plugin itself -- see swap_in_nv_force() below.
PRESSURE_BAR = 1.0
TEMPERATURE_K = 300.0
BAROSTAT_FREQUENCY = 25


def load_pdb(name):
    """Load a system's solvated PDB: OpenMM's own file for apoa1 and dhfr, else
    data/<name>/solvated.pdb.gz."""
    if name in OPENMM_SYSTEMS:
        return app.PDBFile(str(openmm_benchmark_file(name)))
    pdb_path = DATA_DIR / name / "solvated.pdb.gz"
    with gzip.open(str(pdb_path), "rt") as f:
        pdb = app.PDBFile(f)
    return pdb


def create_system(topology):
    """Parameterize a topology — mirrors canonical OpenMM benchmark."""
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    return ff.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=CUTOFF_NM * unit.nanometers,
        ewaldErrorTolerance=0.0005,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.5 * unit.amu,
    )


def swap_in_nv_force(system):
    """Replace the System's NonbondedForce with the plugin's NvNonbondedForce.

    The plugin exposes the force class and no System-level helper, so converting
    is just constructing a force and deciding what to do with the original -- the
    same way you would use any OpenMM force. A System holding more than one
    NonbondedForce, or one you want to keep, is handled by writing this loop
    differently rather than by a flag.

    Nothing is said here about NPT. The cell grid has to be pre-sized for however
    far the box may expand, and the force picks that headroom itself at Context
    creation -- by which point the System is complete, so it can see whether there
    is a barostat. Order does not matter: this may run before or after the
    barostat is added. Pass an explicit factor only if the run's PEAK volume needs
    more than the default, since the grid is allocated once and never resized.
    """
    for i, force in enumerate(system.getForces()):
        if isinstance(force, mm.NonbondedForce):
            # Conversion preserves the source's physical exception-PBC setting,
            # including its False default. Reciprocal corrections stay periodic.
            system.addForce(nv.NonbondedForce(force))
            system.removeForce(i)
            return system
    raise ValueError("System has no NonbondedForce to replace")


def add_barostat(system, engine):
    """Add an isotropic MonteCarlo barostat -- the plugin's, or OpenMM's.

    Whose barostat follows whose force, for the same reason the integrator does:
    timing the plugin against stock is only a comparison if each side runs its
    own throughout. They are drop-ins for each other, sharing the two Context
    parameter names, so having both in one System is a configuration error the
    plugin rejects at Context creation.
    """
    if engine == "plugin":
        system.addForce(nv.MonteCarloBarostat(
            PRESSURE_BAR * unit.bar, TEMPERATURE_K * unit.kelvin,
            frequency=BAROSTAT_FREQUENCY))
    elif engine == "stock":
        system.addForce(mm.MonteCarloBarostat(
            PRESSURE_BAR * unit.bar, TEMPERATURE_K * unit.kelvin,
            BAROSTAT_FREQUENCY))
    else:
        raise ValueError(f"engine must be 'plugin' or 'stock', got {engine!r}")
    return system


def get_energy_and_forces(system, positions, precision="mixed"):
    """Compute potential energy (kJ/mol) and forces (kJ/mol/nm)."""
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    integrator.setConstraintTolerance(1e-5)
    platform = mm.Platform.getPlatformByName("CUDA")
    props = {"Precision": precision, "DeterministicForces": "true"}
    context = mm.Context(system, integrator, platform, props)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometers)
    del context, integrator
    return energy, np.array(forces)


def isolate_nonbonded(system):
    """Strip every force but the single NonbondedForce, which is left at index 0.

    Bonded terms are identical on both sides of every comparison below, so all
    they would do is inflate the denominator of a relative error and hide a real
    discrepancy in the part being measured.
    """
    found = [i for i in range(system.getNumForces())
             if isinstance(system.getForce(i), mm.NonbondedForce)]
    if len(found) != 1:
        raise ValueError(f"expected exactly one NonbondedForce, found {len(found)}")
    for idx in reversed(range(system.getNumForces())):
        if idx not in found:
            system.removeForce(idx)
    return system.getForce(0)


def electrostatics_only(system):
    """Zero every epsilon, in place. Exceptions keep their charge products, so
    1-2/1-3 pairs stay fully excluded and 1-4 pairs keep their scaled
    electrostatics -- the exclusion machinery runs exactly as in production."""
    nb = isolate_nonbonded(system)
    for i in range(nb.getNumParticles()):
        charge, sigma, _eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, charge, sigma, 0.0)
    for k in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, _eps = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, p1, p2, charge_prod, sigma, 0.0)
    nb.setUseDispersionCorrection(False)
    return system


def vdw_only(system):
    """Zero every charge, in place -- the mirror of electrostatics_only.

    The dispersion correction goes too: it is a mean-field analytic tail that adds
    energy and no force, which the plugin does not implement, so leaving it on
    would put a constant offset in the comparison that is not about the pair math.
    """
    nb = isolate_nonbonded(system)
    for i in range(nb.getNumParticles()):
        _charge, sigma, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0, sigma, eps)
    for k in range(nb.getNumExceptions()):
        p1, p2, _charge_prod, sigma, eps = nb.getExceptionParameters(k)
        nb.setExceptionParameters(k, p1, p2, 0.0, sigma, eps)
    nb.setUseDispersionCorrection(False)
    return system


def pme_settings(system, positions):
    """The alpha and grid OpenMM derives at the production cutoff.

    Discovered rather than assumed, then pinned onto every run, so the reference
    and the thing it measures cannot end up on different splitting parameters.
    """
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator,
                         mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "single"})
    context.setPositions(positions)
    alpha, nx, ny, nz = system.getForce(0).getPMEParametersInContext(context)
    del context, integrator
    return float(alpha), (int(nx), int(ny), int(nz))


def pinned(system_xml, alpha, grid, cutoff_nm):
    """Deserialize, then pin the NonbondedForce to an exact (cutoff, alpha, grid)."""
    system = mm.XmlSerializer.deserialize(system_xml)
    nb = system.getForce(0)
    nb.setCutoffDistance(cutoff_nm * unit.nanometers)
    nb.setPMEParameters(alpha, *grid)
    return system


def rel_energy(energy, reference):
    return abs(energy - reference) / abs(reference)


def rel_force_rmse(forces, reference):
    """L2 force RMSE over atoms, relative to the reference's RMS force magnitude."""
    rmse = np.sqrt(np.mean(np.sum((forces - reference) ** 2, axis=1)))
    return float(rmse / np.sqrt(np.mean(np.sum(reference ** 2, axis=1))))


def check_accuracy(name, system_xml, positions):
    """Plugin and stock, both against a reference neither of them can flatter."""
    # ── Electrostatics, vs a CONVERGED reference ────────────────────────────
    elec_xml = mm.XmlSerializer.serialize(
        electrostatics_only(mm.XmlSerializer.deserialize(system_xml)))
    alpha, grid = pme_settings(mm.XmlSerializer.deserialize(elec_xml), positions)

    print(f"  electrostatics: reference at alpha={alpha:.4f}/nm, "
          f"{ACCURACY_FACTOR}x cutoff, grid "
          f"{'x'.join(str(ACCURACY_FACTOR * g) for g in grid)}, fp64")
    e_ref, f_ref = get_energy_and_forces(
        pinned(elec_xml, alpha, tuple(ACCURACY_FACTOR * g for g in grid),
               ACCURACY_FACTOR * CUTOFF_NM), positions, "double")

    e_stock, f_stock = get_energy_and_forces(
        pinned(elec_xml, alpha, grid, CUTOFF_NM), positions)

    # The plugin arm is NOT pinned, and cannot be: NvNonbondedForce derives its own
    # splitting parameters and refuses a pinned pair rather than carrying one. It
    # also runs the DEFAULT splitting here, which is the prolate one rather than
    # Ewald -- so this arm is not the stock arm recomputed, it is the same sum
    # decomposed a different way at the same 5e-4 target. That is why both are
    # scored against the converged fp64 reference below and not against each
    # other: the reference is the exact lattice sum, which both decompose, while
    # plugin-minus-stock is just the two truncation errors added together.
    plugin_system = mm.XmlSerializer.deserialize(elec_xml)
    plugin_system.getForce(0).setCutoffDistance(CUTOFF_NM * unit.nanometers)
    swap_in_nv_force(plugin_system)
    e_plugin, f_plugin = get_energy_and_forces(plugin_system, positions)

    e_err, f_err = rel_energy(e_plugin, e_ref), rel_force_rmse(f_plugin, f_ref)
    print(f"    energy  plugin {e_err:.2e}   stock {rel_energy(e_stock, e_ref):.2e}"
          f"   (tol {ENERGY_TOL:.0e})")
    print(f"    force   plugin {f_err:.2e}   stock "
          f"{rel_force_rmse(f_stock, f_ref):.2e}   (tol {FORCE_TOL:.0e})")
    assert e_err < ENERGY_TOL, (
        f"{name}: electrostatic energy error {e_err:.2e} exceeds {ENERGY_TOL:.0e} "
        f"against the converged reference; stock scores "
        f"{rel_energy(e_stock, e_ref):.2e}. If BOTH moved, the reference or the "
        f"settings changed rather than the plugin.")
    assert f_err < FORCE_TOL, (
        f"{name}: electrostatic force RMSE {f_err:.2e} exceeds {FORCE_TOL:.0e} "
        f"against the converged reference; stock scores "
        f"{rel_force_rmse(f_stock, f_ref):.2e}.")

    # ── van der Waals, vs fp64 at the SAME cutoff ───────────────────────────
    # Nothing is doubled and nothing needs to converge, so fp64 here is exact.
    # With every charge zero there is no splitting involved at all, which makes
    # this the half that stays a like-for-like comparison with stock no matter
    # how the Coulomb sum is decomposed.
    vdw_xml = mm.XmlSerializer.serialize(
        vdw_only(mm.XmlSerializer.deserialize(system_xml)))
    print(f"  van der Waals:  reference at the same {CUTOFF_NM} nm cutoff, fp64")

    e_ref, f_ref = get_energy_and_forces(
        mm.XmlSerializer.deserialize(vdw_xml), positions, "double")
    e_stock, f_stock = get_energy_and_forces(
        mm.XmlSerializer.deserialize(vdw_xml), positions)

    plugin_system = mm.XmlSerializer.deserialize(vdw_xml)
    swap_in_nv_force(plugin_system)
    e_plugin, f_plugin = get_energy_and_forces(plugin_system, positions)

    e_err, f_err = rel_energy(e_plugin, e_ref), rel_force_rmse(f_plugin, f_ref)
    print(f"    energy  plugin {e_err:.2e}   stock {rel_energy(e_stock, e_ref):.2e}"
          f"   (tol {VDW_ENERGY_TOL:.0e})")
    print(f"    force   plugin {f_err:.2e}   stock "
          f"{rel_force_rmse(f_stock, f_ref):.2e}   (tol {VDW_FORCE_TOL:.0e})")
    assert e_err < VDW_ENERGY_TOL, (
        f"{name}: vdW energy error {e_err:.2e} exceeds {VDW_ENERGY_TOL:.0e}. "
        f"There is no splitting here -- charges are zero -- so this is an LJ, "
        f"exclusion or 1-4 discrepancy. stock scores "
        f"{rel_energy(e_stock, e_ref):.2e}.")
    assert f_err < VDW_FORCE_TOL, (
        f"{name}: vdW force RMSE {f_err:.2e} exceeds {VDW_FORCE_TOL:.0e}. "
        f"stock scores {rel_force_rmse(f_stock, f_ref):.2e}.")


def time_steps(system, positions, engine, n_warmup=500, n_steps=3000):
    """Time n_steps of LangevinMiddle dynamics, return (seconds/step, final energy).

    `engine` is "plugin" or "stock", and is required. The plugin arm always runs
    the Nvidia integrator -- a plugin force under OpenMM's integrator is a real
    configuration, but it is not the one this sample reports, and it would still
    print a plausible speedup. Naming the arm makes that unmissable where a
    defaulted keyword made it a thing to remember.
    """
    if engine == "plugin":
        # The Nvidia integrator performs the center-of-mass motion removal.
        for _i, _f in enumerate(system.getForces()):
            if isinstance(_f, mm.CMMotionRemover):
                system.removeForce(_i)
                break
        integrator = nv.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picoseconds, 0.004 * unit.picoseconds)
    elif engine == "stock":
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picoseconds, 0.004 * unit.picoseconds)
    else:
        raise ValueError(f"engine must be 'plugin' or 'stock', got {engine!r}")
    integrator.setConstraintTolerance(1e-5)
    platform = mm.Platform.getPlatformByName("CUDA")
    props = {"Precision": "mixed", "DeterministicForces": "true"}
    context = mm.Context(system, integrator, platform, props)
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin)
    # Warmup (GPU clock ramp + JIT)
    integrator.step(n_warmup)
    context.getState(getEnergy=True)  # sync
    t0 = time.perf_counter()
    integrator.step(n_steps)
    context.getState(getEnergy=True)  # sync
    elapsed = time.perf_counter() - t0
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole)
    del context, integrator
    return elapsed / n_steps, energy


def run_system(name, npt=False):
    """Run correctness + performance checks for one system. Returns speedup."""
    ensemble = "NPT" if npt else "NVT"
    print(f"\n{'=' * 60}")
    print(f"  {name.upper()}  ({ensemble})")
    print(f"{'=' * 60}")

    print(f"Loading {name}...")
    pdb = load_pdb(name)
    positions = pdb.getPositions()
    n_atoms = len(positions)
    print(f"  {n_atoms} atoms")

    print("Creating system (Amber14, PME 0.9 nm, HBonds, rigid water, HMR)...")
    system = create_system(pdb.topology)
    system_xml = mm.XmlSerializer.serialize(system)

    # Whole-system totals, reported and NOT asserted on. The two sides split the
    # Coulomb sum differently, so this difference is the two decompositions'
    # truncation error rather than either one's mistake, and there is no
    # meaningful threshold to put on it. The accuracy that IS asserted is below,
    # against references that do not depend on how the sum was split.
    e_stock_total, _ = get_energy_and_forces(system, positions)
    plugin_total_system = mm.XmlSerializer.deserialize(system_xml)
    swap_in_nv_force(plugin_total_system)
    e_plugin_total, _ = get_energy_and_forces(plugin_total_system, positions)
    print(f"Total energy: plugin {e_plugin_total:.4f}  stock {e_stock_total:.4f} "
          f"kJ/mol  (differ by {rel_energy(e_plugin_total, e_stock_total):.2e}, "
          f"not a tolerance)")

    # NVT only. Accuracy is a property of the force field and the settings, not of
    # the ensemble -- the NPT arm differs by a barostat, which does not run during
    # a single-point evaluation. Repeating it there would buy nothing and cost a
    # second converged fp64 pass, which at 2x cutoff on apoa1 is the most expensive
    # thing in this script.
    if not npt:
        print("Checking accuracy against high-accuracy references...")
        check_accuracy(name, system_xml, positions)

    # Performance: time stock vs plugin dynamics. Under NPT each side gets its own
    # barostat, and that is the whole difference -- the plugin sizes its own cell
    # grid for the expansion.
    print(f"Timing stock OpenMM (3000 steps, {ensemble})...")
    stock_system = mm.XmlSerializer.deserialize(system_xml)
    if npt:
        add_barostat(stock_system, "stock")
    stock_sps, _ = time_steps(stock_system, positions, "stock")
    stock_ns_day = (0.004 / stock_sps) * 86400 / 1000
    print(f"  {stock_sps*1e3:.3f} ms/step  ({stock_ns_day:.1f} ns/day)")

    print(f"Timing Nvidia plugin (3000 steps, {ensemble})...")
    plugin_system = mm.XmlSerializer.deserialize(system_xml)
    swap_in_nv_force(plugin_system)
    if npt:
        add_barostat(plugin_system, "plugin")
    plugin_sps, e_final = time_steps(plugin_system, positions, "plugin")
    plugin_ns_day = (0.004 / plugin_sps) * 86400 / 1000
    print(f"  {plugin_sps*1e3:.3f} ms/step  ({plugin_ns_day:.1f} ns/day)")
    if not np.isfinite(e_final):
        print(f"  WARNING: energy diverged ({e_final})")
        return float("nan")

    speedup = stock_sps / plugin_sps
    print(f"  Speedup: {speedup:.2f}x")

    return speedup


def main():
    results = {}
    errors = []
    for name in SYSTEMS:
        for ensemble in ("nvt", "npt"):
            key = f"{name}_{ensemble}"
            results[key] = run_system(name, npt=(ensemble == "npt"))
            if not np.isfinite(results[key]):
                errors.append(f"{name} ({ensemble}): energy diverged")
            elif results[key] < 1.0:
                errors.append(f"{name} ({ensemble}): plugin slower ({results[key]:.2f}x)")

    # Summary
    for ensemble in ("nvt", "npt"):
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY  ({ensemble.upper()})")
        print(f"{'=' * 60}")
        for name in SYSTEMS:
            print(f"  {name:8s}  {results[f'{name}_{ensemble}']:.2f}x")

    if errors:
        print()
        for e in errors:
            print(f"FAIL: {e}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
