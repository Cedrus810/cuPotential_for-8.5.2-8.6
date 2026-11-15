# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triclinic cells through the plugin.

cupo decomposes the cell OpenMM describes -- lower triangular, in reduced form --
so the shapes Modeller emits for boxShape='dodecahedron' and 'octahedron', and any
sheared cell, run. Each one is held against stock OpenMM at the same settings, and
the rectangular cell is measured beside them so a triclinic-specific gap would show:

  - electrostatics at production settings against a converged fp64 reference (the
    same alpha, twice the mesh and twice the cutoff), to the tolerances the
    orthorhombic plugin-vs-stock tests hold; stock's own error is measured too;
  - the full force field against stock in double at the same derived settings;
  - NVT, isotropic-NPT and flexible-NPT steps on a tilted cell;
  - what remains refused: the flexible barostat on a RECTANGULAR box (it changes
    the off-diagonals every move, and the decomposition's kind is fixed at Context
    creation, so only a Context created tilted can follow it), and a box that
    acquires off-diagonals after the Context was created.

Before triclinic cells were supported, the plugin refused them; the last bullet
is what remains of the tests of that refusal.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

CUTOFF_NM = 0.9
# As test_plugin_vs_stock: the converged reference doubles the mesh and the cutoff
# at the same alpha, and production is held to these.
_ACCURACY_FACTOR = 2
_ACCURACY_ENERGY_TOL = 1e-4
_ACCURACY_FORCE_TOL = 1e-3
# Off-diagonals a reduced-form cell can carry, in nm, for a ~4 nm box.
_TILT = 0.3


def _shapes(V):
    """The cells OpenMM's own Modeller emits for boxShape=, plus bare shears, plus
    the rectangular control. Every one is in reduced form."""
    # Large enough that the tilted cells' face spacings exceed twice the converged
    # reference's doubled cutoff (OpenMM's half-box rule), and three cutoffs.
    L = 5.2
    return {
        "cube": [V(L, 0, 0), V(0, L, 0), V(0, 0, L)],
        "shear_b": [V(L, 0, 0), V(_TILT, L, 0), V(0, 0, L)],
        "shear_c": [V(L, 0, 0), V(0, L, 0), V(_TILT, _TILT, L)],
        "rhombic_dodecahedron":
            [V(L, 0, 0), V(0, L, 0), V(L / 2, L / 2, L / np.sqrt(2))],
        "truncated_octahedron":
            [V(L, 0, 0), V(L / 3, 2 * np.sqrt(2) * L / 3, 0),
             V(-L / 3, np.sqrt(2) * L / 3, np.sqrt(6) * L / 3)],
    }


def _water_system(box_vectors, box_safety=1.0, barostat=None, method=None):
    """A TIP3P cell of the given vectors, solvated INTO that cell, carrying the
    plugin's NonbondedForce (PME, so stock is the same splitting)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    import nv_openmm_plugin as nv
    from conftest import replace_nonbonded

    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, model="tip3p",
                        boxVectors=[v * unit.nanometer for v in box_vectors])
    _remove_overlapping_waters(modeller, box_vectors)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUTOFF_NM * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    system.setDefaultPeriodicBoxVectors(*box_vectors)
    stock_xml = mm.XmlSerializer.serialize(system)
    replace_nonbonded(system, box_safety,
                      nv.NonbondedForce.PME if method is None else method)
    if barostat is not None:
        system.addForce(barostat)
    return system, modeller.positions, stock_xml


def _remove_overlapping_waters(modeller, box_vectors):
    """Modeller.addSolvent can place two waters on top of each other across a
    triclinic cell's periodic boundary (seen at O-O 0.18 nm, across b, on the
    octahedron and dodecahedron here). Such a pair is unphysical and sits past
    the plugin's int32 fixed-point force range (cupo's fixed-point format:
    about 1.4e5 kJ/mol/nm per component), so it would measure saturation, not
    geometry. Drop one water of every pair closer than 0.25 nm under the minimum
    image of the given cell."""
    import openmm.unit as unit
    h = np.array([[v.x, v.y, v.z] for v in box_vectors])
    pos = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    oxygens = [a for a in modeller.topology.atoms() if a.element is not None and a.element.symbol == "O"]
    o_idx = np.array([a.index for a in oxygens])
    p = pos[o_idx]
    s = np.linalg.solve(h.T, p.T).T
    doomed = set()
    for i in range(len(o_idx)):
        ds = s[i + 1:] - s[i]
        ds -= np.round(ds)
        d = ds @ h
        close = np.nonzero(np.einsum("ij,ij->i", d, d) < 0.25 ** 2)[0]
        for j in close:
            doomed.add(oxygens[i + 1 + j].residue)
    if doomed:
        modeller.delete(sorted(doomed, key=lambda r: r.index))


def _context(system, positions, precision="mixed"):
    import openmm as mm
    import openmm.unit as unit
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                             0.001 * unit.picoseconds)
    integrator.setRandomNumberSeed(1)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision})
    context.setPositions(positions)
    return context, integrator


def _energy_forces(system, positions, precision):
    import openmm as mm
    import openmm.unit as unit
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision})
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    f = np.asarray(state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometers))
    nb = [system.getForce(i) for i in range(system.getNumForces())
          if isinstance(system.getForce(i), mm.NonbondedForce)]
    derived = nb[0].getPMEParametersInContext(context) if nb else None
    del context, integrator
    return e, f, derived


def _electrostatics_only(system):
    """Zero every LJ well so the comparison isolates the periodic electrostatics."""
    import openmm as mm
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.NonbondedForce):
            for p in range(f.getNumParticles()):
                q, sig, _ = f.getParticleParameters(p)
                f.setParticleParameters(p, q, sig, 0.0)
            for e in range(f.getNumExceptions()):
                a, b, qq, sig, _ = f.getExceptionParameters(e)
                f.setExceptionParameters(e, a, b, qq, sig, 0.0)
            f.setUseDispersionCorrection(False)


def _rel_energy_error(e, ref):
    return abs(e - ref) / abs(ref)


def _rel_force_rmse(f, ref):
    rmse = np.sqrt(np.mean(np.sum((f - ref) ** 2, axis=1)))
    return rmse / np.sqrt(np.mean(np.sum(ref ** 2, axis=1)))


@pytest.mark.parametrize("shape", sorted(_shapes(lambda *v: v)))
def test_electrostatics_match_converged_reference(plugin_dir, shape, record_property):
    """Plugin and stock, both at production settings, against a converged fp64
    reference, on a tilted cell -- and on the rectangular control under the same
    thresholds, so a triclinic-specific loss of accuracy would stand out."""
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    vectors = _shapes(V)[shape]
    _, positions, stock_xml = _water_system(vectors)

    stock = mm.XmlSerializer.deserialize(stock_xml)
    _electrostatics_only(stock)
    _, _, (alpha, *grid) = _energy_forces(stock, positions, "single")
    alpha = float(alpha)
    grid = [int(g) for g in grid]

    reference = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(stock))
    nb = reference.getForce([i for i in range(reference.getNumForces())
                             if isinstance(reference.getForce(i), mm.NonbondedForce)][0])
    nb.setCutoffDistance(_ACCURACY_FACTOR * CUTOFF_NM * unit.nanometers)
    nb.setPMEParameters(alpha, *(_ACCURACY_FACTOR * g for g in grid))
    e_ref, f_ref, _ = _energy_forces(reference, positions, "double")

    nb = stock.getForce([i for i in range(stock.getNumForces())
                         if isinstance(stock.getForce(i), mm.NonbondedForce)][0])
    nb.setPMEParameters(alpha, *grid)
    e_stock, f_stock, _ = _energy_forces(stock, positions, "single")

    # The plugin arm: the same electrostatics-only stock force, converted. (Zeroed
    # BEFORE the conversion -- the converted force is not an mm.NonbondedForce.)
    import nv_openmm_plugin as nv
    from conftest import replace_nonbonded
    plugin = mm.XmlSerializer.deserialize(stock_xml)
    _electrostatics_only(plugin)
    replace_nonbonded(plugin, 1.0, nv.NonbondedForce.PME)
    e_plugin, f_plugin, _ = _energy_forces(plugin, positions, "single")

    errors = {"stock_energy": _rel_energy_error(e_stock, e_ref),
              "plugin_energy": _rel_energy_error(e_plugin, e_ref),
              "stock_force": _rel_force_rmse(f_stock, f_ref),
              "plugin_force": _rel_force_rmse(f_plugin, f_ref)}
    for k, v in errors.items():
        record_property(f"{shape}.{k}", v)
    assert errors["plugin_energy"] < _ACCURACY_ENERGY_TOL, errors
    assert errors["plugin_force"] < _ACCURACY_FORCE_TOL, errors
    assert errors["stock_energy"] < _ACCURACY_ENERGY_TOL, errors
    assert errors["stock_force"] < _ACCURACY_FORCE_TOL, errors


@pytest.mark.parametrize("shape", ["rhombic_dodecahedron", "truncated_octahedron", "cube"])
def test_full_force_field_matches_stock(plugin_dir, shape):
    """The whole force field (LJ, exclusions, the dispersion correction's volume)
    against stock in double at the settings both derive from the same box."""
    import openmm as mm
    V = mm.Vec3
    system, positions, stock_xml = _water_system(_shapes(V)[shape])
    stock = mm.XmlSerializer.deserialize(stock_xml)
    e_stock, f_stock, _ = _energy_forces(stock, positions, "double")
    e_plugin, f_plugin, _ = _energy_forces(system, positions, "mixed")
    # Two PME truncations of the same sum (the meshes may differ), so looser than
    # the pinned comparison above; a wrong lattice shift would miss by orders.
    assert _rel_energy_error(e_plugin, e_stock) < 3e-4, (e_plugin, e_stock)
    assert _rel_force_rmse(f_plugin, f_stock) < 3e-3


def test_nvt_steps_on_a_tilted_cell(plugin_dir):
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    system, positions, _ = _water_system(_shapes(V)["rhombic_dodecahedron"])
    context, integrator = _context(system, positions)
    try:
        e0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        integrator.step(100)
        e1 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        assert np.isfinite(e0) and np.isfinite(e1)
        # Modeller's water relaxes over the first picosecond; a wrong periodic
        # image would send the energy up, not down to the liquid's value.
        n_water = system.getNumParticles() // 3
        assert e1 <= e0 + 1e-3 * abs(e0)
        assert 20.0 < -e1 / n_water < 80.0, (e0, e1, n_water)
    finally:
        del context, integrator


def test_isotropic_barostat_on_a_tilted_cell(plugin_dir):
    """MonteCarloBarostat scales the vectors and keeps the cell's shape."""
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    barostat = mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 5)
    barostat.setRandomNumberSeed(1)
    system, positions, _ = _water_system(_shapes(V)["truncated_octahedron"],
                                         box_safety=1.2, barostat=barostat)
    context, integrator = _context(system, positions)
    try:
        integrator.step(100)
        state = context.getState(getEnergy=True)
        a, b, c = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        # Still tilted, still reduced, scaled as one.
        assert b[0] != 0.0 and c[0] != 0.0 and c[1] != 0.0
        assert b[0] / a[0] == pytest.approx(1.0 / 3.0, rel=1e-5)
    finally:
        del context, integrator


def test_flexible_barostat_is_refused_on_a_rectangular_box(plugin_dir):
    """A Context created rectangular runs the rectangular decomposition, which
    cannot follow the off-diagonals the flexible barostat introduces."""
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    barostat = mm.MonteCarloFlexibleBarostat(1.0 * unit.bar, 300 * unit.kelvin, 1)
    barostat.setRandomNumberSeed(1)
    system, positions, _ = _water_system(_shapes(V)["cube"], box_safety=1.2, barostat=barostat)
    with pytest.raises(mm.OpenMMException, match="MonteCarloFlexibleBarostat"):
        _context(system, positions)


@pytest.mark.parametrize("shape", ["rhombic_dodecahedron", "shear_c"])
def test_flexible_barostat_on_a_tilted_cell(plugin_dir, shape):
    """From a tilted box the triclinic decomposition follows the flexible
    barostat's moves. The box must change shape (not only scale), stay reduced,
    and the plugin's energy at the final box must agree with stock."""
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    barostat = mm.MonteCarloFlexibleBarostat(1.0 * unit.bar, 300 * unit.kelvin, 1)
    barostat.setRandomNumberSeed(1)
    system, positions, stock_xml = _water_system(_shapes(V)[shape], box_safety=1.3, barostat=barostat)
    context, integrator = _context(system, positions)
    try:
        a0, b0, c0 = (np.array([v.x, v.y, v.z]) for v in _shapes(V)[shape])
        integrator.step(200)
        state = context.getState(getEnergy=True, getPositions=True, enforcePeriodicBox=False)
        a, b, c = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        e_plugin = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert np.isfinite(e_plugin)
        # Reduced form (OpenMM's): positive diagonal, |b.x| <= a.x/2, |c.x| <= a.x/2, |c.y| <= b.y/2.
        assert a[0] > 0 and b[1] > 0 and c[2] > 0
        assert abs(b[0]) <= a[0] / 2 + 1e-6 and abs(c[0]) <= a[0] / 2 + 1e-6 and abs(c[1]) <= b[1] / 2 + 1e-6
        # The shape changed: the off-diagonal to diagonal ratios are not all the initial ones.
        ratios0 = np.array([b0[0] / a0[0], c0[0] / a0[0], c0[1] / b0[1]])
        ratios = np.array([b[0] / a[0], c[0] / a[0], c[1] / b[1]])
        assert not np.allclose(ratios, ratios0, atol=1e-6), (ratios0, ratios)
        # Stock at the same positions and the final box, double precision, same splitting.
        pos = state.getPositions(asNumpy=True)
        stock = mm.XmlSerializer.deserialize(stock_xml)
        stock.setDefaultPeriodicBoxVectors(V(*a), V(*b), V(*c))
        ref_integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        ref = mm.Context(stock, ref_integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "double"})
        try:
            ref.setPeriodicBoxVectors(V(*a), V(*b), V(*c))
            ref.setPositions(pos)
            e_stock = ref.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        finally:
            del ref, ref_integrator
        n_water = system.getNumParticles() // 3
        assert abs(e_plugin - e_stock) / n_water < 0.5, (e_plugin, e_stock, n_water)
    finally:
        del context, integrator


def test_box_acquiring_off_diagonals_is_refused(plugin_dir):
    """A rectangular decomposition is fixed at creation; a cell that tilts after
    it is refused rather than evaluated as if the off-diagonals were zero. OpenMM
    itself rejects the change first (its Context fixes the box shape too), and the
    plugin's own guard behind it catches any route that gets past that."""
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    system, positions, _ = _water_system(_shapes(V)["cube"], box_safety=1.2)
    context, integrator = _context(system, positions)
    try:
        context.getState(getEnergy=True)
        with pytest.raises(mm.OpenMMException, match="triclinic|off-diagonals"):
            context.setPeriodicBoxVectors(*[v * unit.nanometer for v in _shapes(V)["shear_b"]])
            context.getState(getEnergy=True)
    finally:
        del context, integrator


def test_tilted_cell_is_not_the_rectangular_one(plugin_dir):
    """The regression that motivated the original guard, from the other side: a
    tilted cell must give a different energy from the rectangular cell of the
    same diagonal, because it IS a different periodic system."""
    import openmm as mm
    import openmm.unit as unit
    V = mm.Vec3
    energies = {}
    for shape in ("cube", "shear_b"):
        system, positions, _ = _water_system(_shapes(V)[shape])
        context, integrator = _context(system, positions)
        try:
            energies[shape] = context.getState(getEnergy=True).getPotentialEnergy() \
                .value_in_unit(unit.kilojoule_per_mole)
        finally:
            del context, integrator
    assert energies["cube"] != energies["shear_b"]


# ── Vendored triclinic protein cells ───────────────────────────────────────────
#
# data/{apoa1,cdk2,bace,p38}_{dodeca,octa}: the proteins re-solvated in a rhombic
# dodecahedron and a truncated octahedron of the original cube's width. Every
# dataset under
# data/ already runs through test_plugin_vs_stock.py's energy, force and converged
# accuracy tests; these two tests put the production configuration on them -- the
# full force field with constraints and HMR as the bench builds it -- against stock,
# and step them, so a triclinic-specific failure on a real solute shows up here
# rather than in a benchmark.

def _protein_cells():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from data.datasets import available_datasets
    return [d for d in available_datasets() if d.endswith(("_dodeca", "_octa"))]


def _protein_cell_system(name):
    """(plugin system, positions, stock xml, box vectors) as the bench builds it."""
    import openmm as mm
    import nv_openmm_plugin as nv
    from conftest import replace_nonbonded
    from openmm import app, unit
    from data.datasets import load_dataset
    from test_bench_plugin import _load_system
    # These cells were prepared with the force fields below. Changing TIP3P-FB
    # waters to TIP3P changes their constraints and causes a large initial
    # relaxation even with stock OpenMM.
    base = name.rsplit("_", 1)[0]
    fields = {
        "bace": ("amber99sbildn.xml", "tip3p.xml"),
        "cdk2": ("amber14-all.xml", "amber14/tip3pfb.xml"),
        "tyk2": ("amber14-all.xml", "amber14/tip3pfb.xml"),
        "p38": ("amber14-all.xml", "amber14/tip3pfb.xml"),
    }
    if base in fields:
        dataset = load_dataset(name)
        system = app.ForceField(*fields[base]).createSystem(
            dataset.topology, nonbondedMethod=app.PME,
            nonbondedCutoff=0.9 * unit.nanometer,
            constraints=app.HBonds, rigidWater=True,
            hydrogenMass=3.024 * unit.dalton,
        )
        positions = dataset.positions
    else:
        system, positions = _load_system(name)
    stock_xml = mm.XmlSerializer.serialize(system)
    replace_nonbonded(system, 1.0, nv.NonbondedForce.PME)
    return system, positions, stock_xml, system.getDefaultPeriodicBoxVectors()


@pytest.mark.parametrize("name", _protein_cells())
def test_protein_cell_matches_stock(plugin_dir, name):
    import openmm as mm
    import openmm.unit as unit
    system, positions, stock_xml, box = _protein_cell_system(name)
    a, b, c = (v.value_in_unit(unit.nanometer) for v in box)
    # The data must be what it claims: a reduced-form cell with off-diagonals.
    assert b[2] == 0 and a[1] == a[2] == 0
    assert abs(c[0]) > 0.1 * a[0] or abs(c[1]) > 0.1 * b[1] or abs(b[0]) > 0.1 * a[0], box
    assert 2 * abs(b[0]) <= a[0] and 2 * abs(c[0]) <= a[0] and 2 * abs(c[1]) <= b[1], box
    stock = mm.XmlSerializer.deserialize(stock_xml)
    e_stock, f_stock, _ = _energy_forces(stock, positions, "double")
    e_plugin, f_plugin, _ = _energy_forces(system, positions, "mixed")
    # As test_full_force_field_matches_stock: two PME truncations of one sum.
    assert _rel_energy_error(e_plugin, e_stock) < 3e-4, (name, e_plugin, e_stock)
    assert _rel_force_rmse(f_plugin, f_stock) < 3e-3, name


@pytest.mark.parametrize("name", _protein_cells())
def test_protein_cell_nvt_steps(plugin_dir, name):
    """100 steps at the bench's 4 fs (HMR) on the equilibrated cell: the energy
    stays finite and near where it started, which a wrong lattice shift or a cell
    row the other side never reads would not allow."""
    import openmm as mm
    import openmm.unit as unit
    system, positions, _, _ = _protein_cell_system(name)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                             0.004 * unit.picoseconds)
    integrator.setRandomNumberSeed(1)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    try:
        context.setPositions(positions)
        context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
        e0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole)
        integrator.step(100)
        st = context.getState(getEnergy=True, getForces=True)
        e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f = st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        assert np.isfinite(e0) and np.isfinite(e1) and np.all(np.isfinite(f)), (name, e0, e1)
        assert abs(e1 - e0) < 2e-2 * abs(e0), (name, e0, e1)
    finally:
        del context, integrator
