# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Particle parameter offsets, against stock OpenMM.

An offset makes a particle's charge, sigma and epsilon a linear function of a
Context parameter: base + param*scale. The plugin carries the offsets over in the
converting constructor and recomputes the parameters when an evaluation sees a
parameter that moved.

Most tests compare with stock OpenMM on the same System, at several values of
the parameter. Those comparisons use PME on both sides and DeterministicForces,
and explicitly select the frozen-default tail when comparing live offsets with
stock NonbondedForce.
Geometric mixing and switched tails also use an independent pair-sum oracle;
Prolate live updates are checked against freshly constructed plugin contexts.

The solute is a few water molecules turned off by the offsets: charge and epsilon
go to zero at lambda = 1. Their exceptions -- the O-H and H-H exclusions -- keep
their base parameters, which is what an exclusion is supposed to do whatever the
charges are, and is also what exercises the q_i*q_j the exclusion correction
carries.
"""

from __future__ import annotations

import numpy as np
import pytest

import openmm as mm
import openmm.app as app
import openmm.unit as unit

from conftest import record_observed

NM = unit.nanometer
KJ = unit.kilojoule_per_mole
CUTOFF = 0.9
BOX = 3.2
N_SOLUTE_ATOMS = 9        # the first three water molecules
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Energy agreement with stock at fp32 round-off, for PME on both sides. Measured
# 5.1e-7 at every lambda on this system, so 2e-6 is a 4x margin.
ENERGY_TOL = 2e-6
# Relative force RMSE. Measured below 1e-6 here.
FORCE_TOL = 1e-5

_SEED = 20261013


def _water_system():
    import random
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    random.seed(_SEED)
    np.random.seed(_SEED)
    modeller.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(BOX, BOX, BOX) * NM)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUTOFF * NM, constraints=app.HBonds)
    return system, modeller.positions


def _nonbonded(system):
    return next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))


def _add_offsets(nb, parameter="lambda", default=0.0):
    """Turn the solute off as `parameter` goes 0 -> 1."""
    nb.addGlobalParameter(parameter, default)
    for i in range(N_SOLUTE_ATOMS):
        q, _sig, eps = nb.getParticleParameters(i)
        nb.addParticleParameterOffset(parameter, i, -q.value_in_unit(unit.elementary_charge),
                                      0.0, -eps.value_in_unit(KJ))


def _swap(system, method="PME", follow=None):
    """Replace the stock NonbondedForce by the plugin's, under PME. Returns it."""
    import nv_openmm_plugin as nv
    index, nb = next((i, f) for i, f in enumerate(system.getForces())
                     if isinstance(f, mm.NonbondedForce))
    force = nv.NonbondedForce(nb, 1.0, getattr(nv.NonbondedForce, method))
    if follow is not None:
        force.setDispersionCorrectionFollowsParameters(follow)
    system.addForce(force)
    system.removeForce(index)
    return force


def _context(system, positions, precision="mixed", integrator=None):
    if integrator is None:
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision, "DeterministicForces": "true"})
    context.setPositions(positions)
    return context, integrator


def _state(context):
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(KJ),
            state.getForces(asNumpy=True).value_in_unit(KJ / NM))


def _pair(precision="mixed", integrator=None, **offsets):
    """A stock and a plugin Context on the same System with the same offsets."""
    system, positions = _water_system()
    _add_offsets(_nonbonded(system), **offsets)
    stock = _context(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)),
                     positions, precision)
    _swap(system, follow=False)
    plugin = _context(system, positions, precision, integrator)
    return stock, plugin


class _Worst:
    """The largest errors a test saw, recorded once when it ends: the docs table
    takes one row per test and description."""

    def __init__(self, request):
        self.request, self.energy, self.force = request, 0.0, 0.0
        self.energy_tolerance = ENERGY_TOL

    def compare(self, label, stock_ctx, plugin_ctx):
        e_stock, f_stock = _state(stock_ctx)
        e_plugin, f_plugin = _state(plugin_ctx)
        e_rel = abs(e_plugin - e_stock) / abs(e_stock)
        f_rel = (np.sqrt(np.mean(np.sum((f_plugin - f_stock) ** 2, axis=1)))
                 / np.sqrt(np.mean(np.sum(f_stock ** 2, axis=1))))
        self.energy, self.force = max(self.energy, e_rel), max(self.force, f_rel)
        assert e_rel < self.energy_tolerance, (
            f"{label}: plugin {e_plugin:.6f} kJ/mol, stock {e_stock:.6f} (rel {e_rel:.2e})")
        assert f_rel < FORCE_TOL, f"{label}: relative force RMSE {f_rel:.2e}"
        return e_plugin

    def record(self):
        record_observed(self.request, "parameter offsets vs stock, energy",
                        self.energy_tolerance, self.energy)
        record_observed(self.request, "parameter offsets vs stock, force",
                        FORCE_TOL, self.force)


@pytest.fixture
def worst(request):
    w = _Worst(request)
    yield w
    w.record()


@pytest.mark.parametrize("precision", ("mixed", "single"))
def test_a_lambda_sweep_matches_stock(worst, plugin_dir, precision):
    """Energy and forces agree with stock at every lambda, in both directions of a
    sweep, and the energy actually moves -- an offset that is accepted and then
    ignored would agree with stock only at lambda = 0."""
    (stock, si), (plugin, pi) = _pair(precision)
    energies = []
    for lam in LAMBDAS + LAMBDAS[::-1]:
        stock.setParameter("lambda", lam)
        plugin.setParameter("lambda", lam)
        energies.append(worst.compare(f"{precision}, lambda={lam}", stock, plugin))
    assert abs(energies[0] - energies[len(LAMBDAS) - 1]) > 1.0, (
        f"the energy did not move between lambda 0 and 1: {energies}")


def test_a_parameter_set_before_the_first_evaluation(worst, plugin_dir):
    """setParameter before the first getState must reach the device the first
    evaluation builds -- the case updateParametersInContext once dropped."""
    (stock, si), (plugin, pi) = _pair()
    stock.setParameter("lambda", 0.6)
    plugin.setParameter("lambda", 0.6)
    worst.compare("set before the first evaluation", stock, plugin)


def test_a_nonzero_default_value(worst, plugin_dir):
    """A Context starts at the parameter's default value, which need not be 0."""
    (stock, si), (plugin, pi) = _pair(default=0.4)
    assert plugin.getParameter("lambda") == pytest.approx(0.4)
    worst.compare("default 0.4", stock, plugin)


def test_two_parameters_on_one_particle(worst, plugin_dir):
    """Offsets from two parameters on the same particle add, as OpenMM defines."""
    system, positions = _water_system()
    nb = _nonbonded(system)
    _add_offsets(nb, "lambda_q")
    # The second parameter moves the oxygens' LJ only. TIP3P hydrogens carry no LJ,
    # and giving them some puts the energy in the millions from H-H overlap, where
    # plugin and stock differ by 6e-6 with no offsets at all.
    nb.addGlobalParameter("lambda_lj", 0.0)
    for i in range(0, N_SOLUTE_ATOMS, 3):
        nb.addParticleParameterOffset("lambda_lj", i, 0.0, 0.01, 0.1)
    stock = _context(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)),
                     positions)
    _swap(system, follow=False)
    plugin = _context(system, positions)
    for lq, llj in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.3, 0.8)):
        for ctx in (stock[0], plugin[0]):
            ctx.setParameter("lambda_q", lq)
            ctx.setParameter("lambda_lj", llj)
        worst.compare(f"lambda_q={lq}, lambda_lj={llj}", stock[0], plugin[0])


def test_an_update_changes_the_offsets(worst, plugin_dir):
    """updateParametersInContext carries changed offset scales, as OpenMM's does."""
    system, positions = _water_system()
    nb = _nonbonded(system)
    _add_offsets(nb)
    stock_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    stock_nb = _nonbonded(stock_system)
    stock, si = _context(stock_system, positions)
    force = _swap(system, follow=False)
    plugin, pi = _context(system, positions)
    for ctx in (stock, plugin):
        ctx.setParameter("lambda", 0.5)
    worst.compare("before the update", stock, plugin)

    for f in (stock_nb, force):
        for k in range(f.getNumParticleParameterOffsets()):
            parameter, i, cs, ss, es = f.getParticleParameterOffset(k)
            f.setParticleParameterOffset(k, parameter, i, 0.5 * cs, ss, 0.5 * es)
    stock_nb.updateParametersInContext(stock)
    force.updateParametersInContext(plugin)
    worst.compare("after halving the scales", stock, plugin)


def test_an_update_may_not_move_an_offset(plugin_dir):
    """Which particle an offset applies to is fixed once a Context exists."""
    system, positions = _water_system()
    _add_offsets(_nonbonded(system))
    force = _swap(system)
    plugin, pi = _context(system, positions)
    parameter, i, cs, ss, es = force.getParticleParameterOffset(0)
    force.setParticleParameterOffset(0, parameter, i + 100, cs, ss, es)
    with pytest.raises(Exception, match="changed its particle or its parameter"):
        force.updateParametersInContext(plugin)


def test_offsets_survive_a_captured_graph(worst, plugin_dir, monkeypatch, capfd):
    """Replay every captured output mode after a global changes, without recapture."""
    import test_geometric_combining as geo

    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    # Verlet's kinetic-energy expression requests forces even for getEnergy=True.
    # A force-free kinetic expression makes the energy-only graph observable.
    integrator = mm.CustomIntegrator(0.001)
    integrator.setKineticEnergyExpression("0.5*m*v*v")
    (stock, si), (plugin, pi) = _pair(integrator=integrator)
    modes = ((False, True), (True, False), (True, True))
    capfd.readouterr()
    for energy, forces in modes:
        for _ in range(56 if not energy else 4):
            geo._snapshot(plugin, energy, forces)
    trace = capfd.readouterr().err
    for energy, forces in modes:
        assert f"energy={int(energy)} forces={int(forces)}" in trace, trace

    for lam in (0.3, 1.0, 0.0):
        stock.setParameter("lambda", lam)
        plugin.setParameter("lambda", lam)
        expected_energy, expected_forces = _state(stock)
        for energy, forces in modes:
            actual_energy, actual_forces = geo._snapshot(plugin, energy, forces)
            if energy:
                assert abs(actual_energy - expected_energy) < ENERGY_TOL * abs(expected_energy)
            if forces:
                assert (np.linalg.norm(actual_forces - expected_forces)
                        < FORCE_TOL * np.linalg.norm(expected_forces))
        worst.compare(f"after capture, lambda={lam}", stock, plugin)
    assert "graph capture #" not in capfd.readouterr().err


def test_the_converted_force_reports_the_offsets(plugin_dir):
    """The converting constructor copies the global parameters and offsets."""
    system, _ = _water_system()
    nb = _nonbonded(system)
    _add_offsets(nb, default=0.25)
    # Read before the swap: removing the stock force from the System deletes it.
    expected = [list(nb.getParticleParameterOffset(k)) for k in range(N_SOLUTE_ATOMS)]
    force = _swap(system)
    assert force.getNumGlobalParameters() == 1
    assert force.getGlobalParameterName(0) == "lambda"
    assert force.getGlobalParameterDefaultValue(0) == pytest.approx(0.25)
    assert force.getNumParticleParameterOffsets() == N_SOLUTE_ATOMS
    assert [force.getParticleParameterOffset(k) for k in range(N_SOLUTE_ATOMS)] == expected


def test_offsets_survive_a_reorder(worst, plugin_dir, monkeypatch, capfd):
    """A lambda changed after capture survives cuPo integration and atom reorder.

    A reorder re-derives the device's per-atom array from the system-order copy the
    kernel keeps, and that copy has to be the one with the offsets applied. OpenMM
    reorders only when some force uses a cutoff, so an inert one is added, as in
    test_reorder.py.
    """
    import nv_openmm_plugin as nv
    from test_reorder import _add_inert_cutoff_force
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    system, positions = _water_system()
    _add_offsets(_nonbonded(system))
    stock_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    _swap(system, follow=False)
    _add_inert_cutoff_force(system)
    for i, f in enumerate(system.getForces()):
        if isinstance(f, mm.CMMotionRemover):
            system.removeForce(i)
            break
    integ = nv.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond,
                                        0.002 * unit.picoseconds, seed=4242)
    integ.setConstraintTolerance(1e-5)
    plugin = mm.Context(system, integ, mm.Platform.getPlatformByName("CUDA"),
                        {"Precision": "mixed"})
    plugin.setPositions(positions)
    plugin.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    capfd.readouterr()
    mm.LocalEnergyMinimizer.minimize(plugin, maxIterations=100)
    plugin.setParameter("lambda", 0.3)
    integ.step(80)
    trace = capfd.readouterr().err
    assert "graph capture #" in trace and "energy=0 forces=1" in trace, trace
    plugin.setParameter("lambda", 0.7)
    integ.step(300)                    # past OpenMM's 250-step reorder period
    err = capfd.readouterr().err
    assert err.count("atom reorder #") >= 1, f"no reorder fired:\n{err}"
    assert "graph capture #" not in err, err
    # What the context holds before the comparison below re-evaluates it: a
    # mis-permuted array shows up here as well as there.
    e_after = _state(plugin)[0]

    stock, si = _context(stock_system, plugin.getState(getPositions=True).getPositions())
    stock.setParameter("lambda", 0.7)
    assert worst.compare("after a reorder", stock, plugin) == e_after


# ---- Exception offsets ---------------------------------------------------------
#
# A molecule with real 1-4 pairs, so an exception offset has something to move: the
# alanine dipeptide, built from OpenMM's own amber14 templates and solvated. Its
# 1-4 exceptions carry a scaled chargeProd and an epsilon, both of which the
# offsets below take to zero, and its particles are turned off as well, so the two
# kinds of offset run together the way an alchemical decoupling uses them.


def _dipeptide_system():
    import io
    import random
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    pdb = app.PDBFile(io.StringIO(_DIPEPTIDE_PDB))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    random.seed(_SEED)
    np.random.seed(_SEED)
    modeller.addSolvent(ff, model="tip3p", boxSize=mm.Vec3(BOX, BOX, BOX) * NM)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=CUTOFF * NM, constraints=app.HBonds)
    n_solute = pdb.topology.getNumAtoms()
    return system, _minimized(system, modeller.positions), n_solute


_minimized_positions = {}


def _minimized(system, positions):
    """Positions relaxed by stock OpenMM, once per process.

    The PDB below is written by hand and has non-excluded atoms 0.2 nm apart. At
    those positions the LJ energy is large enough that plugin and stock differ by
    5e-6 with no offsets at all, which says nothing about offsets. Relaxed, they
    agree to round-off as they do on water. Deterministic forces and a fixed
    iteration count, so every run of the suite compares at the same positions.
    """
    if "dipeptide" not in _minimized_positions:
        context = mm.Context(system, mm.VerletIntegrator(0.001),
                             mm.Platform.getPlatformByName("CUDA"),
                             {"Precision": "mixed", "DeterministicForces": "true"})
        context.setPositions(positions)
        mm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
        _minimized_positions["dipeptide"] = context.getState(getPositions=True).getPositions()
    return _minimized_positions["dipeptide"]


def _add_exception_offsets(nb, n_solute, parameter="lambda"):
    """Turn the solute's particles and its 1-4 pairs off as `parameter` -> 1.
    Returns how many exception offsets were added."""
    nb.addGlobalParameter(parameter, 0.0)
    for i in range(n_solute):
        q, _sig, eps = nb.getParticleParameters(i)
        nb.addParticleParameterOffset(parameter, i, -q.value_in_unit(unit.elementary_charge),
                                      0.0, -eps.value_in_unit(KJ))
    added = 0
    for k in range(nb.getNumExceptions()):
        i, j, qq, _sig, eps = nb.getExceptionParameters(k)
        if i < n_solute and j < n_solute and (qq.value_in_unit(unit.elementary_charge**2) != 0
                                              or eps.value_in_unit(KJ) != 0):
            nb.addExceptionParameterOffset(parameter, k, -qq.value_in_unit(unit.elementary_charge**2),
                                           0.0, -eps.value_in_unit(KJ))
            added += 1
    return added


def _dipeptide_pair(precision="mixed"):
    system, positions, n_solute = _dipeptide_system()
    added = _add_exception_offsets(_nonbonded(system), n_solute)
    stock = _context(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)),
                     positions, precision)
    force = _swap(system, follow=False)
    plugin = _context(system, positions, precision)
    return stock, plugin, force, added


@pytest.mark.parametrize("precision", ("mixed", "single"))
def test_exception_offsets_match_stock(worst, plugin_dir, precision):
    """A decoupling sweep with particle and 1-4 offsets agrees with stock, and the
    1-4 pairs actually move: the same sweep with the exception offsets dropped
    gives a different energy at lambda = 1."""
    (stock, si), (plugin, pi), force, added = _dipeptide_pair(precision)
    assert added > 10, f"the dipeptide should have tens of 1-4 pairs, got {added}"
    energies = []
    for lam in LAMBDAS + LAMBDAS[::-1]:
        stock.setParameter("lambda", lam)
        plugin.setParameter("lambda", lam)
        energies.append(worst.compare(f"{precision}, lambda={lam}", stock, plugin))

    system, positions, n_solute = _dipeptide_system()
    nb = _nonbonded(system)
    nb.addGlobalParameter("lambda", 1.0)
    for i in range(n_solute):
        q, _sig, eps = nb.getParticleParameters(i)
        nb.addParticleParameterOffset("lambda", i, -q.value_in_unit(unit.elementary_charge),
                                      0.0, -eps.value_in_unit(KJ))
    particles_only, _ = _context(system, positions, precision)
    e_particles_only = _state(particles_only)[0]
    assert abs(e_particles_only - energies[len(LAMBDAS) - 1]) > 1.0, (
        "the exception offsets made no difference at lambda = 1")


def test_exception_offsets_alone(worst, plugin_dir):
    """Offsets on the exceptions and none on the particles: the parameter change
    reaches the exception list without the per-atom array."""
    system, positions, n_solute = _dipeptide_system()
    nb = _nonbonded(system)
    nb.addGlobalParameter("lambda14", 0.0)
    for k in range(nb.getNumExceptions()):
        i, j, qq, _sig, eps = nb.getExceptionParameters(k)
        if i < n_solute and j < n_solute and eps.value_in_unit(KJ) != 0:
            nb.addExceptionParameterOffset("lambda14", k, 0.0, 0.0, -eps.value_in_unit(KJ))
    stock = _context(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system)),
                     positions)
    _swap(system)
    plugin = _context(system, positions)
    for lam in (0.0, 0.5, 1.0, 0.25):
        stock[0].setParameter("lambda14", lam)
        plugin[0].setParameter("lambda14", lam)
        worst.compare(f"lambda14={lam}", stock[0], plugin[0])


def test_exception_offset_indices_follow_the_conversion(plugin_dir):
    """Conversion preserves every exception and its original offset index."""
    import nv_openmm_plugin as nv
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9)
    for q, eps in ((0.0, 0.0), (0.0, 0.0), (0.5, 0.2), (-0.5, 0.2)):
        system.addParticle(1.0)
        nb.addParticle(q, 0.3, eps)
    nb.addException(0, 1, 0.0, 0.3, 0.0)          # retained even while inactive
    nb.addException(2, 3, -0.1, 0.3, 0.1)
    nb.addGlobalParameter("lambda", 0.0)
    nb.addExceptionParameterOffset("lambda", 1, 0.1, 0.0, -0.1)
    force = nv.NonbondedForce(nb, 1.0, nv.NonbondedForce.PME)
    assert force.getNumExceptions() == 2
    assert force.getNumExceptionParameterOffsets() == 1
    parameter, index, cs, ss, es = force.getExceptionParameterOffset(0)
    assert (parameter, index) == ("lambda", 1)
    assert force.getExceptionParameters(index)[:2] == [2, 3]


@pytest.mark.parametrize("endpoints", ((-1, 1), (0, 4)))
def test_invalid_exception_particle_index_is_rejected(plugin_dir, endpoints):
    """Validate exception endpoints before indexing effective particle records."""
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0),
                                        mm.Vec3(0, 0, BOX))
    source = mm.NonbondedForce()
    source.setNonbondedMethod(mm.NonbondedForce.PME)
    source.setCutoffDistance(CUTOFF)
    for charge in (0.1, -0.1, 0.1, -0.1):
        system.addParticle(18.0)
        source.addParticle(charge, 0.3, 0.2)
    # The invalid index belongs to the exception itself, not a parameter offset.
    source.addException(*endpoints, 0.0, 0.3, 0.0)
    system.addForce(source)
    _swap(system)
    integrator = mm.VerletIntegrator(0.001)
    with pytest.raises(Exception, match="illegal exception particle index"):
        mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                   {"Precision": "mixed"})


def test_an_update_may_not_move_an_exception_offset(plugin_dir):
    (stock, si), (plugin, pi), force, added = _dipeptide_pair()
    parameter, k, cs, ss, es = force.getExceptionParameterOffset(0)
    force.setExceptionParameterOffset(0, parameter, k + 1, cs, ss, es)
    with pytest.raises(Exception, match="changed its exception or its parameter"):
        force.updateParametersInContext(plugin)


def test_zero_exception_can_activate_and_update_its_offset(plugin_dir):
    """An initially empty exception keeps its identity and accepts live scales."""
    import test_geometric_combining as geo

    # Ordinary charges stay zero: a nonzero independent exception charge product
    # must still activate, without a separate PME baseline masking that change.
    positions, particles, exceptions = geo._fixture(charged=False)
    system, source = geo._source("PME", particles, exceptions, False)
    source.addGlobalParameter("lambda14", 0.0)
    source.addExceptionParameterOffset("lambda14", 2, 0.08, 0.02, 0.25)
    system.addForce(source)
    stock_system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    stock_force = _nonbonded(stock_system)
    stock, si = _context(stock_system, positions)
    force = _swap(system)
    plugin, pi = _context(system, positions)
    assert force.getExceptionParameterOffset(0)[:2] == ["lambda14", 2]

    def compare(lam, scale=1.0):
        stock_energy, stock_forces = _state(stock)
        plugin_energy, plugin_forces = _state(plugin)
        assert abs(plugin_energy-stock_energy) < ENERGY_TOL*abs(stock_energy)
        # This weak-LJ fixture is dominated by the fixed force quantization step at lambda
        # zero, so the water fixture's relative RMSE is not its precision bound.
        # Count contributing pairs using the independent FP64 oracle and retain
        # the same per-pair rounding + FP32 bound as the native mixing tests.
        effective = [list(values) for values in exceptions]
        effective[2][2:] = [0.08*lam*scale, 0.3+0.02*lam*scale, 0.25*lam*scale]
        _, _, rounding, arithmetic = geo._lj_reference(
            positions, particles, effective, False, geometric=False)
        i, j, charge_product, _, _ = effective[2]
        delta = positions[i]-positions[j]
        delta -= geo.BOX*np.rint(delta/geo.BOX)
        coulomb_force = 138.93545764498*charge_product*delta/np.linalg.norm(delta)**3
        arithmetic[i] += np.abs(coulomb_force)
        arithmetic[j] += np.abs(coulomb_force)
        geo._assert_force_close(plugin_forces, stock_forces, rounding, arithmetic)
        return plugin_energy

    energies = []
    for lam in (0.0, 0.6, 0.0, 1.0):
        for context in (stock, plugin):
            context.setParameter("lambda14", lam)
        energies.append(compare(lam))
    assert abs(energies[-1] - energies[0]) > 1.0

    for updated, context in ((stock_force, stock), (force, plugin)):
        parameter, index, charge, sigma, epsilon = updated.getExceptionParameterOffset(0)
        updated.setExceptionParameterOffset(0, parameter, index, charge/2, sigma/2, epsilon/2)
        updated.updateParametersInContext(context)
    changed = compare(1.0, scale=0.5)
    assert abs(changed - energies[-1]) > 1.0


@pytest.mark.parametrize("method", ("PME", "Prolate"))
def test_triclinic_live_particle_and_exception_offsets(worst, plugin_dir, method):
    """Live values reach direct, reciprocal and exception terms in a tilted box."""
    import test_geometric_combining as geo

    # This non-water fixture has a small total energy. Compact and dense FFTs
    # differ by FP32 rounding: the unchanged native path measures 2.87e-6 versus
    # 1.78e-6 relative energy error. Use the existing compact oracle budget here;
    # keep the water fixtures' tighter 2e-6 bound and the force bound unchanged.
    worst.energy_tolerance = 5e-6
    positions, particles, exceptions = geo._fixture(charged=True)
    system, source = geo._source("PME", particles, exceptions, False)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(geo.BOX, 0, 0),
                                        mm.Vec3(0.3, geo.BOX, 0),
                                        mm.Vec3(0.2, 0.15, geo.BOX))
    source.addGlobalParameter("lambda", 0.0)
    source.addParticleParameterOffset("lambda", 10, 0.1, 0.02, -0.1)
    source.addParticleParameterOffset("lambda", 11, -0.05, 0.01, 0.12)
    source.addExceptionParameterOffset("lambda", 2, 0.03, 0.01, 0.12)
    system.addForce(source)
    reference_xml = mm.XmlSerializer.serialize(system)
    _swap(system, method)
    plugin, pi = _context(system, positions)
    # Start away from the default before the first evaluation, then return to it.
    for lam in (0.6, 1.0, 0.0):
        plugin.setParameter("lambda", lam)
        reference_system = mm.XmlSerializer.deserialize(reference_xml)
        _nonbonded(reference_system).setGlobalParameterDefaultValue(0, lam)
        if method == "Prolate":
            _swap(reference_system, method)
        reference, ri = _context(reference_system, positions)
        worst.compare(f"triclinic {method}, lambda={lam}", reference, plugin)
        del reference, ri


@pytest.mark.parametrize("rule", ("LorentzBerthelot", "Geometric"))
@pytest.mark.parametrize("follow", (False, True))
def test_switched_offset_tail_respects_combining_rule(plugin_dir, rule, follow):
    """Switching/mixing use live LJ while the tail follows its selected policy."""
    import nv_openmm_plugin as nv
    import test_geometric_combining as geo
    import test_lj_switching as switching

    positions, particles, exceptions = switching._fixture()
    system, source = geo._source("Prolate", particles, exceptions, True)
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(switching.SWITCH)
    default = 0.25
    source.addGlobalParameter("lambda", default)
    scales = np.zeros_like(particles)
    scales[10, 1:] = (0.03, 0.2)
    scales[11, 1:] = (0.02, -0.05)
    for index in (10, 11):
        source.addParticleParameterOffset("lambda", index, *scales[index])
    system.addForce(source)
    force = _swap(system, "Prolate")
    force.setLJCombiningRule(getattr(nv.NonbondedForce, rule))
    force.setDispersionCorrectionFollowsParameters(follow)
    context, integrator = _context(system, positions)

    defaults = particles + default*scales
    default_tail = (switching._reference(positions, defaults, exceptions, True, rule)[0]
                    - switching._reference(positions, defaults, exceptions, False, rule)[0])
    for lam in (default, 0.8, 0.0):
        context.setParameter("lambda", lam)
        effective = particles + lam*scales
        reference = list(switching._reference(positions, effective, exceptions, follow, rule))
        if not follow:
            reference[0] += default_tail
        geo._assert_close(_state(context), reference)


def test_invalid_live_switched_tail_does_not_commit_parameters(plugin_dir):
    """A rejected live update remains rejected and restoring lambda restores state."""
    import test_geometric_combining as geo

    positions, particles, exceptions = geo._fixture()
    particles[:, 2] = 0.0
    for exception in exceptions:
        exception[4] = 0.0
    system, source = geo._source("Prolate", particles, exceptions, True)
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(0.0)
    source.addGlobalParameter("lambda", 0.0)
    for index in (10, 11):
        source.addParticleParameterOffset("lambda", index, 0.0, 0.0, 0.7)
    system.addForce(source)
    force = _swap(system, "Prolate")
    force.setDispersionCorrectionFollowsParameters(True)
    context, integrator = _context(system, positions)
    original = _state(context)
    assert original[0] == 0.0
    np.testing.assert_array_equal(original[1], np.zeros_like(positions))
    context.setParameter("lambda", 1.0)
    for _ in range(2):
        with pytest.raises(Exception, match="(?i)positive switching distance"):
            _state(context)
    context.setParameter("lambda", 0.0)
    context.setPositions(positions + (0.01, 0.0, 0.0))
    restored = _state(context)
    assert restored[0] == original[0]
    np.testing.assert_array_equal(restored[1], original[1])


# ACE-ALA-NME in OpenMM's amber14 naming, from a minimized structure.
_DIPEPTIDE_PDB = """\
ATOM      1 HH31 ACE A   1       2.000   1.000   0.000  1.00  0.00           H
ATOM      2  CH3 ACE A   1       2.000   2.090   0.000  1.00  0.00           C
ATOM      3 HH32 ACE A   1       1.486   2.454   0.890  1.00  0.00           H
ATOM      4 HH33 ACE A   1       1.486   2.454  -0.890  1.00  0.00           H
ATOM      5  C   ACE A   1       3.427   2.641  -0.000  1.00  0.00           C
ATOM      6  O   ACE A   1       4.391   1.877  -0.000  1.00  0.00           O
ATOM      7  N   ALA A   2       3.555   3.970  -0.000  1.00  0.00           N
ATOM      8  H   ALA A   2       2.733   4.556  -0.000  1.00  0.00           H
ATOM      9  CA  ALA A   2       4.853   4.614  -0.000  1.00  0.00           C
ATOM     10  HA  ALA A   2       5.408   4.316   0.890  1.00  0.00           H
ATOM     11  CB  ALA A   2       5.661   4.221  -1.232  1.00  0.00           C
ATOM     12  HB1 ALA A   2       5.123   4.521  -2.131  1.00  0.00           H
ATOM     13  HB2 ALA A   2       6.630   4.719  -1.206  1.00  0.00           H
ATOM     14  HB3 ALA A   2       5.809   3.141  -1.241  1.00  0.00           H
ATOM     15  C   ALA A   2       4.713   6.129   0.000  1.00  0.00           C
ATOM     16  O   ALA A   2       3.601   6.653   0.000  1.00  0.00           O
ATOM     17  N   NME A   3       5.846   6.835   0.000  1.00  0.00           N
ATOM     18  H   NME A   3       6.737   6.359  -0.000  1.00  0.00           H
ATOM     19  CH3 NME A   3       5.846   8.284   0.000  1.00  0.00           C
ATOM     20 HH31 NME A   3       4.819   8.648   0.000  1.00  0.00           H
ATOM     21 HH32 NME A   3       6.360   8.648   0.890  1.00  0.00           H
ATOM     22 HH33 NME A   3       6.360   8.648  -0.890  1.00  0.00           H
TER
END
"""


# ---- The dispersion correction and the global parameters ----
#
# The term is isolated as E(correction on) - E(correction off), at the same positions
# and parameters: everything else in the two energies is the same computation, so the
# difference is the tail term alone. The reference for "the tail at lambda x" is
# stock OpenMM on a System whose default value is x, since stock folds the offsets in
# at the defaults.

# Relative error allowed on the tail term. Against stock NonbondedForce the plugin
# computes the coefficient with the same arithmetic, so what is left is the fp32
# round-off of the energies the term is a difference of: measured at most 4.5e-8
# across these tests, so 1e-5 is a 220x margin.
TAIL_TOL = 1e-5
# Against a stock CustomNonbondedForce, which integrates the tail of its expression
# numerically where the plugin evaluates the 12-6 tail in closed form. Measured, the
# two differ by 2.4e-7 at every lambda, so 2e-6 is an 8x margin.
CUSTOM_TAIL_TOL = 2e-6


def _energy(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ)


def _tail_contexts(plugin, follow=None, default=0.0, energy_only=False):
    """Two Contexts on the same positions, with the dispersion correction on and off.
    Returns ((on, integrator), (off, integrator), the plugin force with it on)."""
    out, force_on = [], None
    for use in (True, False):
        system, positions = _water_system()
        nb = _nonbonded(system)
        _add_offsets(nb, default=default)
        nb.setUseDispersionCorrection(use)
        if plugin:
            force = _swap(system, follow=follow)
            if use:
                force_on = force
        integrator = None
        if energy_only:
            integrator = mm.CustomIntegrator(0.001)
            integrator.setKineticEnergyExpression("0.5*m*v*v")
        out.append(_context(system, positions, integrator=integrator))
    return out[0], out[1], force_on


def _tail(on, off):
    return _energy(on) - _energy(off)


def _stock_tail(default):
    (on, _), (off, _), _ = _tail_contexts(False, default=default)
    return _tail(on, off)


def _check_tail(label, plugin_term, stock_term, tol=TAIL_TOL):
    rel = abs(plugin_term - stock_term) / abs(stock_term)
    assert rel < tol, (f"{label}: plugin tail {plugin_term:.6f} kJ/mol, "
                            f"stock {stock_term:.6f} (rel {rel:.2e})")


def test_the_tail_can_preserve_stock_default_parameter_behavior(plugin_dir):
    """With the option off, the tail term is the one at the default value whatever
    the Context parameter is, as in stock OpenMM."""
    stock = _stock_tail(0.0)
    (on, _), (off, _), _ = _tail_contexts(True, follow=False)
    for lam in (0.0, 0.5, 1.0):
        on.setParameter("lambda", lam)
        off.setParameter("lambda", lam)
        _check_tail(f"option off, lambda={lam}", _tail(on, off), stock)
    # The offsets do move the tail, so a term that stayed put exercises the
    # explicit compatibility policy rather than a test that cannot fail.
    assert abs(_stock_tail(1.0) - stock) / abs(stock) > 10 * TAIL_TOL


def test_the_tail_follows_lambda_by_default(plugin_dir):
    """The default tail at lambda x is stock's tail when its default value is x."""
    (on, _), (off, _), force = _tail_contexts(True)
    assert force.getDispersionCorrectionFollowsParameters() is True
    for lam in (0.5, 1.0, 0.0):
        on.setParameter("lambda", lam)
        off.setParameter("lambda", lam)
        _check_tail(f"option on, lambda={lam}", _tail(on, off), _stock_tail(lam))


def test_the_tail_follows_lambda_after_a_captured_graph(plugin_dir, monkeypatch, capfd):
    """The correction changes when globals change on the already-captured mode."""
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    (on, integrator), (off, _), _ = _tail_contexts(True, energy_only=True)
    capfd.readouterr()
    for _ in range(56):
        _tail(on, off)
    trace = capfd.readouterr().err
    assert trace.count("graph capture #") == 2, trace
    assert trace.count("energy=1 forces=0") == 2, trace
    for lam in (0.3, 1.0):
        on.setParameter("lambda", lam)
        off.setParameter("lambda", lam)
        (stock_on, _), (stock_off, _), _ = _tail_contexts(False, default=lam)
        positions = on.getState(getPositions=True).getPositions()
        stock_on.setPositions(positions)
        stock_off.setPositions(positions)
        _check_tail(f"after capture, lambda={lam}", _tail(on, off),
                    _tail(stock_on, stock_off))
    assert "graph capture #" not in capfd.readouterr().err


def test_the_tail_option_round_trips(plugin_dir):
    """Both constructors follow live LJ by default; stock compatibility is explicit."""
    system, _ = _water_system()
    _add_offsets(_nonbonded(system))
    force = _swap(system)
    assert force.getDispersionCorrectionFollowsParameters() is True
    force.setDispersionCorrectionFollowsParameters(False)
    assert force.getDispersionCorrectionFollowsParameters() is False
    force.setDispersionCorrectionFollowsParameters(True)
    assert force.getDispersionCorrectionFollowsParameters() is True
    import nv_openmm_plugin as nv
    assert nv.NonbondedForce().getDispersionCorrectionFollowsParameters() is True


def _independent_tail_contexts(follow=None):
    """A small fixture with independent charge, particle LJ, and 1-4 globals."""
    import test_geometric_combining as geo

    positions, particles, exceptions = geo._fixture()
    scales = np.zeros_like(particles)
    scales[10, 1:] = (0.03, 0.2)
    scales[11, 1:] = (0.02, -0.05)
    contexts, forces = [], []
    for use in (True, False):
        system, source = geo._source("PME", particles, exceptions, use)
        for name, default in (("lambda_q", 0.0), ("lambda_lj", 0.25), ("lambda14", 0.0)):
            source.addGlobalParameter(name, default)
        for index in (10, 11):
            source.addParticleParameterOffset("lambda_lj", index, *scales[index])
        source.addParticleParameterOffset("lambda_q", 10, 0.15, 0.0, 0.0)
        source.addParticleParameterOffset("lambda_q", 11, -0.15, 0.0, 0.0)
        source.addExceptionParameterOffset("lambda14", 2, 0.08, 0.01, 0.1)
        system.addForce(source)
        forces.append(_swap(system, follow=follow))
        contexts.append(_context(system, positions))
    return contexts, forces, positions, particles, exceptions, scales


def _independent_tail_reference(positions, particles, exceptions):
    import test_geometric_combining as geo

    return (geo._lj_reference(positions, particles, exceptions, True, geometric=False)[0]
            - geo._lj_reference(positions, particles, exceptions, False, geometric=False)[0])


def test_live_tail_with_independent_charge_lj_and_exception_updates(plugin_dir):
    """Charge/1-4 changes preserve the cached LJ tail, including a force-only update."""
    contexts, forces, positions, particles, exceptions, scales = _independent_tail_contexts()
    (on, _), (off, _) = contexts
    for lq, llj, l14 in ((0.0, 0.25, 0.0), (0.6, 0.25, 0.0),
                         (0.6, 0.25, 0.7), (0.6, 0.8, 0.7),
                         (0.0, 0.8, 0.7), (0.0, 0.8, 0.0), (0.0, 0.25, 0.0)):
        for context in (on, off):
            context.setParameter("lambda_q", lq)
            context.setParameter("lambda_lj", llj)
            context.setParameter("lambda14", l14)
            # The first evaluation after a change need not request energy. The
            # later energy request must use the already-updated coefficient.
            context.getState(getForces=True)
        expected = _independent_tail_reference(positions, particles + llj*scales, exceptions)
        _check_tail(f"q={lq}, LJ={llj}, 1-4={l14}", _tail(on, off), expected)


@pytest.mark.parametrize("follow", (False, True))
def test_tail_refreshes_after_base_and_offset_scale_updates(plugin_dir, follow):
    """Both policies refresh explicit LJ edits, including nonzero-default offsets."""
    contexts, forces, positions, particles, exceptions, scales = _independent_tail_contexts(follow)
    (on, _), (off, _) = contexts
    for context in (on, off):
        context.setParameter("lambda_lj", 0.8)
    tail_lambda = 0.8 if follow else 0.25
    before = _tail(on, off)
    _check_tail("before explicit updates", before,
                _independent_tail_reference(positions, particles + tail_lambda*scales, exceptions))

    # Change both a base particle and an offset law while the Context lambda is
    # away from its default. Neither coefficient may remain cached across this.
    particles[12, 1:] *= (1.1, 0.7)
    scales[10:12, 1:] *= 0.5
    for force, (context, _) in zip(forces, contexts):
        force.setParticleParameters(12, *particles[12])
        for offset in (0, 1):
            name, index, charge, sigma, epsilon = force.getParticleParameterOffset(offset)
            force.setParticleParameterOffset(offset, name, index, charge, sigma/2, epsilon/2)
        force.updateParametersInContext(context)
    after = _tail(on, off)
    _check_tail("after explicit updates", after,
                _independent_tail_reference(positions, particles + tail_lambda*scales, exceptions))
    assert abs(after - before) / abs(before) > 10*TAIL_TOL


def test_tail_policy_changes_refresh_without_a_lambda_change(plugin_dir):
    """Changing the policy or disabling/re-enabling the tail invalidates its cache."""
    contexts, forces, positions, particles, exceptions, scales = _independent_tail_contexts()
    (on, _), (off, _) = contexts
    for context in (on, off):
        context.setParameter("lambda_lj", 0.8)
    live = _independent_tail_reference(positions, particles + 0.8*scales, exceptions)
    frozen = _independent_tail_reference(positions, particles + 0.25*scales, exceptions)
    _check_tail("initial live policy", _tail(on, off), live)
    for follow, expected in ((False, frozen), (True, live)):
        forces[0].setDispersionCorrectionFollowsParameters(follow)
        forces[0].updateParametersInContext(on)
        _check_tail(f"follow={follow}", _tail(on, off), expected)
    forces[0].setUseDispersionCorrection(False)
    forces[0].updateParametersInContext(on)
    assert _tail(on, off) == 0.0
    forces[0].setUseDispersionCorrection(True)
    forces[0].updateParametersInContext(on)
    _check_tail("re-enabled at current lambda", _tail(on, off), live)


def _custom_lj_system(long_range_correction, dsigma):
    """The water box as an alchemical tool writes it for stock OpenMM: the charges in
    a NonbondedForce with no LJ, and the LJ in a CustomNonbondedForce whose sigma and
    epsilon move linearly with lambda, over the same exclusions. With its long-range
    correction on, stock OpenMM recomputes the tail whenever lambda changes.

    The solute's change is _add_offsets': charge and epsilon go to 0 at lambda = 1.
    The charges keep doing that in the NonbondedForce, as charge offsets. On top,
    dsigma (nm) goes on the sigma of every solute atom that carries LJ, so that both
    the sigma and the epsilon terms of the tail move."""
    system, positions = _water_system()
    nb = _nonbonded(system)
    nb.setUseDispersionCorrection(False)
    nb.addGlobalParameter("lambda", 0.0)
    for i in range(N_SOLUTE_ATOMS):
        q, _sig, _eps = nb.getParticleParameters(i)
        nb.addParticleParameterOffset("lambda", i, -q.value_in_unit(unit.elementary_charge),
                                      0.0, 0.0)
    lj = mm.CustomNonbondedForce(
        "4*eps*((sig/r)^12-(sig/r)^6); sig=0.5*(s1+s2); eps=sqrt(e1*e2); "
        "s1=sig01+lambda*dsig1; s2=sig02+lambda*dsig2; "
        "e1=max(0,eps01+lambda*deps1); e2=max(0,eps02+lambda*deps2)")
    for name in ("sig0", "dsig", "eps0", "deps"):
        lj.addPerParticleParameter(name)
    # Declared by both forces with the same default, which OpenMM allows: one Context
    # parameter drives the charges and the LJ.
    lj.addGlobalParameter("lambda", 0.0)
    lj.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    lj.setCutoffDistance(CUTOFF * NM)
    lj.setUseLongRangeCorrection(long_range_correction)
    for i in range(nb.getNumParticles()):
        q, sig, eps = nb.getParticleParameters(i)
        sig, eps = sig.value_in_unit(NM), eps.value_in_unit(KJ)
        solute = i < N_SOLUTE_ATOMS
        lj.addParticle([sig, dsigma if solute and eps > 0 else 0.0,
                        eps, -eps if solute else 0.0])
        nb.setParticleParameters(i, q, sig, 0.0)
    for k in range(nb.getNumExceptions()):
        a, b, _qq, _sig, eps = nb.getExceptionParameters(k)
        # Water has no 1-4 pairs: every exception is an exclusion, which the custom
        # force's exclusion list carries in full.
        assert eps.value_in_unit(KJ) == 0.0
        lj.addExclusion(a, b)
    system.addForce(lj)
    return _context(system, positions)


def _plugin_lj_contexts(dsigma, follow=None):
    """The same change as parameter offsets on the plugin's NonbondedForce, as two
    Contexts: the dispersion correction on and off. `follow` is the tail option."""
    out = []
    for use in (True, False):
        system, positions = _water_system()
        nb = _nonbonded(system)
        _add_offsets(nb)
        for k in range(nb.getNumParticleParameterOffsets()):
            parameter, i, cs, _ss, es = nb.getParticleParameterOffset(k)
            nb.setParticleParameterOffset(k, parameter, i, cs, dsigma if es else 0.0, es)
        nb.setUseDispersionCorrection(use)
        _swap(system, follow=follow)
        out.append(_context(system, positions)[0])
    return out


def test_the_tail_matches_a_custom_nonbonded_force(plugin_dir):
    """By default, the plugin's NonbondedForce with offsets gives the tail a
    stock CustomNonbondedForce with its long-range correction gives, for the same 12-6
    LJ moved linearly by lambda -- the case where the two describe the same physics.

    CustomNonbondedForce integrates its expression's tail numerically; the plugin
    evaluates the 12-6 tail in closed form. Measured, they differ by 2.4e-7 of the
    term at every lambda; CUSTOM_TAIL_TOL is the gate. The total energies agree to
    fp32 round-off, as the other comparisons with stock do."""
    dsigma = 0.03
    stock_on = _custom_lj_system(True, dsigma)[0]
    stock_off = _custom_lj_system(False, dsigma)[0]
    plugin_on, plugin_off = _plugin_lj_contexts(dsigma)
    tails = []
    for lam in (0.0, 0.25, 0.5, 1.0, 0.0):
        for c in (stock_on, stock_off, plugin_on, plugin_off):
            c.setParameter("lambda", lam)
        e_stock, e_plugin = _energy(stock_on), _energy(plugin_on)
        rel = abs(e_plugin - e_stock) / abs(e_stock)
        assert rel < ENERGY_TOL, (f"lambda={lam}: plugin {e_plugin:.6f} kJ/mol, stock "
                                  f"{e_stock:.6f} (rel {rel:.2e})")
        stock_tail = e_stock - _energy(stock_off)
        _check_tail(f"vs CustomNonbondedForce, lambda={lam}",
                    e_plugin - _energy(plugin_off), stock_tail, CUSTOM_TAIL_TOL)
        tails.append(stock_tail)
    # The tail does move across the sweep, so agreeing at every lambda is a check
    # that it followed, not that both stayed at the start.
    assert abs(tails[3] - tails[0]) / abs(tails[0]) > 10 * CUSTOM_TAIL_TOL


def test_frozen_tail_does_not_match_a_custom_nonbonded_force(plugin_dir):
    """The negative case of the test above: with the option off, the plugin's tail
    stays at the default lambda's, so away from the default it misses the
    CustomNonbondedForce's by far more than CUSTOM_TAIL_TOL. Without this, the test
    above could pass with an option that did nothing.

    Measured, the two differ by 1.0e-3 of the term at lambda = 0.5 and 5.6e-3 at
    lambda = 1, against the 10*CUSTOM_TAIL_TOL = 2e-5 asked for."""
    dsigma = 0.03
    stock_on = _custom_lj_system(True, dsigma)[0]
    stock_off = _custom_lj_system(False, dsigma)[0]
    plugin_on, plugin_off = _plugin_lj_contexts(dsigma, follow=False)
    for lam in (0.5, 1.0):
        for c in (stock_on, stock_off, plugin_on, plugin_off):
            c.setParameter("lambda", lam)
        stock_tail = _energy(stock_on) - _energy(stock_off)
        plugin_tail = _energy(plugin_on) - _energy(plugin_off)
        rel = abs(plugin_tail - stock_tail) / abs(stock_tail)
        assert rel > 10 * CUSTOM_TAIL_TOL, (
            f"lambda={lam}: with the option off the plugin tail {plugin_tail:.6f} kJ/mol "
            f"still matches the CustomNonbondedForce's {stock_tail:.6f} (rel {rel:.2e}); "
            f"the comparison cannot tell the option apart")
