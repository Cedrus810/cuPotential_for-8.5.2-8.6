# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen ligand fixtures and pair selection without an interaction-group backend."""
from __future__ import annotations

import gzip
import hashlib
import json
import ctypes as C
import runpy

import numpy as np
import openmm as mm
from openmm import unit
import pytest

from data import tyk2_alchemical as fixtures


@pytest.mark.parametrize("case", fixtures.CASES)
def test_frozen_stock_system_loads_without_parameterization(case):
    dataset = fixtures.load_tyk2_alchemical(case)
    system = dataset.create_system()
    assert system.getNumParticles() == len(dataset.positions) > 1000
    assert system.getNumConstraints() > 0
    assert dataset.velocities is not None
    assert dataset.velocities.shape == dataset.positions.shape
    assert system.usesPeriodicBoundaryConditions()
    box = np.asarray([vector.value_in_unit(unit.nanometer)
                      for vector in system.getDefaultPeriodicBoxVectors()])
    np.testing.assert_allclose(box, dataset.box, rtol=0, atol=1e-12)
    assert sum(isinstance(force, mm.NonbondedForce) for force in system.getForces()) == 1
    assert any(isinstance(force, mm.CustomNonbondedForce) for force in system.getForces())
    assert any(isinstance(force, mm.HarmonicBondForce) for force in system.getForces())
    assert not any(isinstance(force, mm.MonteCarloBarostat) for force in system.getForces())
    com = [force for force in system.getForces() if isinstance(force, mm.CMMotionRemover)]
    assert len(com) == 1 and com[0].getFrequency() == 100
    assert dataset.metadata["cutoff_nm"] == pytest.approx(.9)
    force_metadata = dataset.metadata["forces"]
    assert len(force_metadata) == system.getNumForces()
    for index, force in enumerate(system.getForces()):
        if isinstance(force, (mm.NonbondedForce, mm.CustomNonbondedForce)):
            metadata = force_metadata[index]
            assert metadata["index"] == index
            cutoff = force.getCutoffDistance().value_in_unit(unit.nanometer)
            assert cutoff == pytest.approx(.9)
            assert metadata["cutoff_nm"] == cutoff
            switching = force.getUseSwitchingFunction()
            if case in ("rbfe", "abfe"):
                # Every pair force of the factory-built cases switches its LJ.
                assert switching
            assert metadata["switching"] == switching
            start = force.getSwitchingDistance().value_in_unit(unit.nanometer)
            assert metadata["switching_nm"] == start
            if switching:
                assert start == pytest.approx(.8)
            if isinstance(force, mm.NonbondedForce):
                assert not force.getExceptionsUsePeriodicBoundaryConditions()
                assert metadata["exceptions_periodic"] is False
    atoms = dataset.metadata["alchemical_atoms"]
    assert atoms and len(set(atoms)) == len(atoms)
    assert min(atoms) >= 0 and max(atoms) < system.getNumParticles()
    found = {}
    for force in system.getForces():
        if hasattr(force, "getNumGlobalParameters"):
            for index in range(force.getNumGlobalParameters()):
                name = force.getGlobalParameterName(index)
                value = force.getGlobalParameterDefaultValue(index)
                if name in dataset.globals:
                    assert value == dataset.globals[name]
                    found[name] = value
    assert found == dataset.globals
    assert dataset.metadata["lambda_value"] == .5
    assert dataset.metadata["native_exceptions_periodic"] is False
    if case != "abfe":
        # OpenFE explicitly enables this option; the lower-level factory's
        # default leaves unique/core 1-4 terms coupled and produces zero bonds.
        assert dataset.metadata["interpolate_old_and_new_14s"] is True
        exceptions = [force for force in system.getForces()
                      if isinstance(force, mm.CustomBondForce)
                      and force.getName() == "CustomBondForce_exceptions"]
        assert len(exceptions) == 1
        assert exceptions[0].getNumBonds() == dataset.metadata["interpolated_exception_bonds"] > 0
        assert not exceptions[0].usesPeriodicBoundaryConditions()


def test_corrupt_fixture_is_rejected(tmp_path, monkeypatch):
    """Checksums guard the serialized physical inputs before deserialization."""
    xml = gzip.compress(b"not even an OpenMM System")
    folder = tmp_path / "rbfe"
    folder.mkdir()
    (folder / "system.xml.gz").write_bytes(xml + b"changed")
    manifest = {"schema_version": 1, "systems": {"rbfe": {
        "files": {"system": "rbfe/system.xml.gz", "positions": "rbfe/positions.npz"}}},
        "file_sha256": {"rbfe/system.xml.gz": hashlib.sha256(xml).hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(fixtures, "_DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="system.xml.gz checksum mismatch"):
        fixtures.load_tyk2_alchemical("rbfe")


def _small_stock_system(groups=None, order=None, periodic=False):
    order = np.arange(6) if order is None else np.asarray(order)
    inverse = np.argsort(order)
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    native = mm.NonbondedForce()
    native.setNonbondedMethod(native.PME)
    native.setCutoffDistance(1.0)
    native.setUseDispersionCorrection(False)
    native.setUseSwitchingFunction(True)
    native.setSwitchingDistance(.7)
    native.setExceptionsUsePeriodicBoundaryConditions(periodic)
    native.addGlobalParameter("lambda", .5)
    for index in order:
        system.addParticle(12)
        native.addParticle(0, .25+.01*index, .2+.03*index)
    native.addException(int(inverse[0]), int(inverse[1]), 0, .25, .05)
    native.addParticleParameterOffset("lambda", int(inverse[2]), 0, .03, -.04)
    system.addForce(native)
    for index in range(2):
        custom = mm.CustomNonbondedForce("lambda*a/(1+r^2);a=k1*k2")
        custom.setNonbondedMethod(custom.CutoffPeriodic)
        custom.setCutoffDistance(1)
        custom.setUseSwitchingFunction(index == 1)
        custom.setSwitchingDistance(.7)
        custom.addGlobalParameter("lambda", .5)
        custom.addPerParticleParameter("k")
        for particle in order:
            custom.addParticle([.1*(particle+1)*(index+1)])
        custom.addExclusion(int(inverse[0]), int(inverse[1]))
        selected = groups[index] if groups is not None else []
        if groups is None and index == 0:
            # Both within-group orientations and overlap between groups.
            selected = [({0, 1, 2}, {1, 2, 3, 4}), ({2, 3}, {3, 4, 5})]
        for left, right in selected:
            custom.addInteractionGroup({int(inverse[i]) for i in left},
                                       {int(inverse[i]) for i in right})
        system.addForce(custom)
    bond = mm.CustomBondForce("lambda*(r-r0)^2")
    bond.addGlobalParameter("lambda", .5)
    bond.addPerBondParameter("r0")
    bond.addBond(int(inverse[0]), int(inverse[1]), [.4])
    system.addForce(bond)
    return system


def _evaluate_reference(system, lam, positions=None):
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    try:
        context.setPositions(positions if positions is not None else
                             np.array([[.2+.18*i, .2+.05*(i % 2), .2] for i in range(6)]))
        context.setParameter("lambda", lam)
        state = context.getState(getEnergy=True, getForces=True)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    finally:
        del context, integrator


def test_fusion_rejects_requested_energy_derivatives_instead_of_dropping_them():
    source = _small_stock_system()
    source.getForce(1).addEnergyParameterDerivative("lambda")
    before = mm.XmlSerializer.serialize(source)
    with pytest.raises(ValueError, match="energy parameter derivatives are not supported"):
        fixtures.fuse_pair_definition(source)
    assert mm.XmlSerializer.serialize(source) == before


def test_conversion_preserves_stock_native_tail_policy(plugin_dir):
    """Benchmark conversion explicitly preserves the stock Hamiltonian."""
    import nv_openmm_plugin as nv

    source = _small_stock_system()
    native = source.getForce(0)
    native.setUseDispersionCorrection(True)
    # This input has live sigma/epsilon offsets and a nonzero default lambda,
    # so the two policies differ away from that default.
    assert nv.NonbondedForce(native).getDispersionCorrectionFollowsParameters()
    combined = fixtures.convert_to_plugin(source, method="PME")
    assert combined.getUseDispersionCorrection()
    assert not combined.getDispersionCorrectionFollowsParameters()


@pytest.mark.parametrize("periodic", [False, True], ids=["raw", "periodic"])
def test_conversion_preserves_native_and_custom_exception_conventions(periodic):
    source = _small_stock_system(periodic=periodic)
    native = source.getForce(0)
    native.setExceptionParameters(0, 0, 1, .003, .25, .06)
    native.addExceptionParameterOffset("lambda", 0, -.001, .02, .03)
    saved = mm.XmlSerializer.serialize(source)
    sparse = mm.XmlSerializer.serialize(source.getForce(3))
    fixtures.fuse_pair_definition(source)
    assert mm.XmlSerializer.serialize(source) == saved
    plugin = mm.XmlSerializer.deserialize(saved)
    combined = fixtures.convert_to_plugin(plugin, method="PME")
    assert combined.getExceptionsUsePeriodicBoundaryConditions() == periodic
    assert combined.getNumExceptions() == native.getNumExceptions()
    assert combined.getNumExceptionParameterOffsets() == native.getNumExceptionParameterOffsets()

    def numbers(values):
        return tuple(value.value_in_unit_system(unit.md_unit_system)
                     if unit.is_quantity(value) else value for value in values)

    assert numbers(combined.getExceptionParameters(0)) == numbers(native.getExceptionParameters(0))
    assert numbers(combined.getExceptionParameterOffset(0)) == numbers(native.getExceptionParameterOffset(0))
    retained = [force for force in plugin.getForces() if isinstance(force, mm.CustomBondForce)]
    assert len(retained) == 1
    assert not retained[0].usesPeriodicBoundaryConditions()
    assert mm.XmlSerializer.serialize(retained[0]) == sparse
    assert mm.XmlSerializer.serialize(source) == saved


@pytest.mark.parametrize("periodic", [False, True], ids=["raw", "periodic"])
def test_fused_exception_conventions_survive_graph_and_live_updates(
        plugin_dir, monkeypatch, capfd, periodic):
    for name, value in {"CUPO_PLUGIN_BONDED": "0", "CUPO_PLUGIN_CUDA_GRAPH": "1",
                        "CUPO_PLUGIN_GRAPH_TRACE": "1", "CUPO_COMPACT_AUTOTUNE": "0"}.items():
        monkeypatch.setenv(name, value)
    source = _small_stock_system(periodic=periodic)
    native = source.getForce(0)
    native.setExceptionParameters(0, 0, 1, .003, .25, .06)
    native.addExceptionParameterOffset("lambda", 0, -.001, .02, .03)
    plugin = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(source))
    combined = fixtures.convert_to_plugin(plugin, method="PME")
    assert combined.getExceptionsUsePeriodicBoundaryConditions() == periodic
    # The exception crosses a box face, so raw and periodic replacements differ.
    # Ordinary charges are zero: stock's reciprocal exception geometry cannot
    # obscure this independent test of the physical replacement and converter.
    positions = np.array([[.1, .3, .4], [3.7, .3, .4], [.5, 1.3, .5],
                          [1., 1.4, .6], [1.7, 1.6, 1.1], [2.3, 1.3, 1.4]])
    contexts, integrators = [], []

    def observe(context):
        state = context.getState(getEnergy=True, getForces=True)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.asarray(state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole/unit.nanometer)))

    try:
        for system in (source, plugin):
            integrator = mm.VerletIntegrator(.001)
            integrators.append(integrator)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                                 {"Precision": "mixed", "DeterministicForces": "true"})
            contexts.append(context)
            context.setPositions(positions)
        capfd.readouterr()
        for _ in range(56):
            observe(contexts[1])
        trace = capfd.readouterr().err
        assert any("graph capture #" in line and "(energy=1 forces=1)" in line
                   for line in trace.splitlines()), trace
        for phase, lam in enumerate((0., .8, .3, 0.)):
            for context in contexts:
                context.setParameter("lambda", lam)
            if phase == 2:
                for force, context in zip((native, combined), contexts):
                    force.setExceptionParameters(0, 0, 1, .004, .27, .08)
                    force.updateParametersInContext(context)
            expected, actual = (observe(context) for context in contexts)
            np.testing.assert_allclose(actual[0], expected[0], rtol=3e-6, atol=3e-5)
            np.testing.assert_allclose(actual[1], expected[1], rtol=3e-5, atol=2e-4)
    finally:
        contexts.clear()
        integrators.clear()


def _reference_fused_system(source, lowered):
    """Bind the lowered pair expression to a stock Reference system."""
    combined = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(source))
    native = combined.getForce(lowered.native_index)
    for index in range(native.getNumParticles()):
        charge, sigma, epsilon = native.getParticleParameters(index)
        native.setParticleParameters(index, charge, sigma, 0.)
    for index in range(native.getNumParticleParameterOffsets()):
        parameter, atom, charge, sigma, epsilon = native.getParticleParameterOffset(index)
        native.setParticleParameterOffset(index, parameter, atom, charge, 0., 0.)
    native.setUseSwitchingFunction(False)
    native.setUseDispersionCorrection(False)
    for index in reversed(lowered.custom_indices):
        combined.removeForce(index)
    combined.addForce(lowered.expression)
    return combined


@pytest.mark.parametrize("lam", (0., .5, 1.))
def test_expression_masks_preserve_groups_offsets_and_switching(lam):
    source = _small_stock_system()
    saved = mm.XmlSerializer.serialize(source)
    lowered = fixtures.fuse_pair_definition(source)
    assert mm.XmlSerializer.serialize(source) == saved
    assert lowered.expression.getNumInteractionGroups() == 0
    assert not lowered.expression.getUseSwitchingFunction()
    assert lowered.expression.getNumExclusions() == 1
    combined = _reference_fused_system(source, lowered)
    expected = _evaluate_reference(source, lam)
    observed = _evaluate_reference(combined, lam)
    np.testing.assert_allclose(observed[0], expected[0], rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(observed[1], expected[1], rtol=1e-12, atol=1e-10)


def _match_reference_mesh(stock, box):
    import nv_openmm_plugin as nv

    # OpenMM and cuPo round automatic mesh dimensions differently. Keep this
    # expression/charge correctness test on the same PME discretization; the
    # end-to-end benchmark compares each implementation's production defaults.
    native = next(force for force in stock.getForces() if isinstance(force, mm.NonbondedForce))
    lib = C.CDLL(str(nv._cuda_platform_dir(nv._PLUGIN_DIR) / "libOpenMMNvidiaCUDA.so"))
    lib.cupo_abi_version.restype = C.c_int
    assert lib.cupo_abi_version() == 32
    lib.cupo_splitting_plan_create_ewald.argtypes = [
        C.c_double, C.c_double, C.POINTER(C.c_void_p)]
    lib.cupo_splitting_plan_create_ewald.restype = C.c_int
    lib.cupo_plan_grid.argtypes = [
        C.c_void_p, C.POINTER(C.c_double), C.c_int, C.POINTER(C.c_int)]
    lib.cupo_plan_grid.restype = C.c_int
    lib.cupo_splitting_plan_destroy.argtypes = [C.c_void_p]
    lib.cupo_splitting_plan_destroy.restype = C.c_int
    cutoff = native.getCutoffDistance().value_in_unit(unit.nanometer)
    plan = C.c_void_p()
    assert lib.cupo_splitting_plan_create_ewald(cutoff, 138.93545764498, C.byref(plan)) == 0
    dimensions = (C.c_int*3)()
    try:
        edges = (C.c_double*3)(*np.diag(box))
        assert lib.cupo_plan_grid(plan, edges, 0, dimensions) == 0
    finally:
        assert lib.cupo_splitting_plan_destroy(plan) == 0
    alpha = np.sqrt(-np.log(2*native.getEwaldErrorTolerance()))/cutoff
    native.setPMEParameters(alpha, *dimensions)
    return tuple(dimensions)


@pytest.mark.parametrize("case", fixtures.CASES)
def test_tyk2_plugin_matches_stock_at_midpoint(plugin_dir, case):
    import nv_openmm_plugin as nv

    dataset = fixtures.load_tyk2_alchemical(case)
    stock = dataset.create_system()
    plugin = dataset.create_system()
    combined = fixtures.convert_to_plugin(plugin, method="PME")
    _match_reference_mesh(stock, dataset.box)
    assert combined.hasCutoffExpression()
    assert not combined.getExceptionsUsePeriodicBoundaryConditions()
    assert combined.getCutoffExpression().getNumInteractionGroups() == 0
    assert sum(isinstance(force, nv.NonbondedForce) for force in plugin.getForces()) == 1
    assert not any(isinstance(force, (mm.NonbondedForce, mm.CustomNonbondedForce))
                   for force in plugin.getForces())
    observations = []
    for system in (stock, plugin):
        integrator = mm.VerletIntegrator(.001)
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                             {"Precision": "mixed", "DeterministicForces": "true"})
        try:
            context.setPositions(dataset.positions)
            state = context.getState(getEnergy=True, getForces=True)
            observations.append((state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))))
        finally:
            del context, integrator
    energy, force = observations[0]
    actual_energy, actual_force = observations[1]
    assert np.isfinite(actual_energy) and np.isfinite(actual_force).all()
    assert abs(actual_energy-energy)/max(abs(energy), 1.) < 3e-6
    relative_force_error = np.linalg.norm(actual_force-force)/np.linalg.norm(force)
    # Use the existing native PME suite's FP32 triclinic error budget: matching
    # the mesh does not make the two fractional-coordinate implementations
    # identical in finite precision (see test_plugin_vs_stock.py).
    assert relative_force_error < 1e-4, relative_force_error
    # Solvent dominates the complete system norm. Check the transforming ligand
    # separately so a mistake in a small alchemical region cannot be hidden.
    atoms = np.asarray(dataset.metadata["alchemical_atoms"], dtype=int)
    assert len(atoms) > 0
    assert np.linalg.norm((actual_force-force)[atoms])/max(np.linalg.norm(force[atoms]), 1e-30) < 1e-4


def test_grouped_tail_is_not_the_tail_of_the_masked_expression():
    """Stock group tails omit self pairs; ordinary masked class averages include them."""
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    native = mm.NonbondedForce()
    native.setNonbondedMethod(native.PME)
    native.setCutoffDistance(1.)
    native.setUseDispersionCorrection(False)
    for _ in range(6):
        system.addParticle(12)
        native.addParticle(0, .3, 0)
    system.addForce(native)
    custom = mm.CustomNonbondedForce("lambda*k1*k2/r^6")
    custom.addGlobalParameter("lambda", .5)
    custom.addPerParticleParameter("k")
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(1.)
    custom.setUseLongRangeCorrection(True)
    for _ in range(6):
        custom.addParticle([.2])
    custom.addInteractionGroup({0, 1, 2}, set(range(6)))
    system.addForce(custom)

    def tail(definition):
        on = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(definition))
        off = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(definition))
        off.getForce(1).setUseLongRangeCorrection(False)
        return _evaluate_reference(on, .5)[0]-_evaluate_reference(off, .5)[0]

    grouped = tail(system)
    lowered = fixtures.fuse_pair_definition(system)
    assert len(lowered.custom_tails) == 1
    assert lowered.custom_tails[0].getNumInteractionGroups() == 1
    naive = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))
    naive.removeForce(1)
    lowered.expression.setUseLongRangeCorrection(True)
    naive.addForce(lowered.expression)
    masked = tail(naive)
    # 12 physical group pairs versus 15 masked pairs after adding three self
    # contributions, normalized by the same total N(N+1)/2=21 in stock.
    expected = 2*np.pi*6**2/21 * 12 * (.5*.2**2/3) / 4**3
    np.testing.assert_allclose(grouped, expected, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(masked/grouped, 15/12, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize("follow", [False, True])
def test_fused_grouped_tails_and_live_globals_match_stock(plugin_dir, follow):
    """Live charges, source-specific tails, and inline switches share one traversal."""
    source = _small_stock_system()
    native = source.getForce(0)
    native.setUseDispersionCorrection(True)
    # Amplify the native LJ offset so using cuPo's live-tail default instead
    # of the converter's stock-compatible policy fails the energy comparison.
    native.setParticleParameterOffset(0, "lambda", 2, 0, .15, .4)
    for index in range(6):
        _, sigma, epsilon = native.getParticleParameters(index)
        native.setParticleParameters(index, .1*(-1)**index, sigma, epsilon)
    native.addParticleParameterOffset("lambda", 3, .04, 0, 0)
    native.setExceptionParameters(0, 0, 1, -.005, .25, .05)
    native.addExceptionParameterOffset("lambda", 0, .002, .01, .02)
    for force in source.getForces():
        if isinstance(force, mm.CustomNonbondedForce):
            force.setEnergyFunction("lambda*a/(1+r^6);a=k1*k2")
            force.setUseLongRangeCorrection(True)
            for index in range(6):
                force.setParticleParameters(index, [.5*(index+1)])
    plugin = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(source))
    combined = fixtures.convert_to_plugin(plugin, method="PME")
    assert combined.getNumExpressionTailCorrections() == 2
    assert combined.getUseDispersionCorrection()
    assert not combined.getDispersionCorrectionFollowsParameters()
    combined.setDispersionCorrectionFollowsParameters(follow)
    assert not combined.getUseSwitchingFunction()
    assert not combined.getCutoffExpression().getUseLongRangeCorrection()
    positions = np.asarray([[.3+.35*i, .3+.3*(i % 2), .3+.2*((i//2) % 2)] for i in range(6)])
    contexts, integrators = [], []
    try:
        for system in (source, plugin):
            integrator = mm.VerletIntegrator(.001)
            integrators.append(integrator)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                                 {"Precision": "mixed", "DeterministicForces": "true"})
            contexts.append(context)
            context.setPositions(positions)
        # Revisiting .5 also catches stale lambda-dependent tail coefficients.
        for lam in (0., .5, 1., .5):
            if follow:
                # Stock's native coefficient uses global defaults. Refresh only
                # that force's default to provide an independent live-tail
                # oracle; its custom tails already follow the Context lambda.
                native.setGlobalParameterDefaultValue(0, lam)
                native.updateParametersInContext(contexts[0])
            states = []
            for context in contexts:
                context.setParameter("lambda", lam)
                states.append(context.getState(getEnergy=True, getForces=True))
            energy, actual_energy = [state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                                     for state in states]
            force, actual_force = [np.asarray(state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole/unit.nanometer)) for state in states]
            np.testing.assert_allclose(actual_energy, energy, rtol=3e-6, atol=5e-5)
            np.testing.assert_allclose(actual_force, force, rtol=3e-5, atol=2e-4)
    finally:
        contexts.clear()
        integrators.clear()


@pytest.mark.parametrize("kind,encoded", [
    ("simple", False), ("overlap", False), ("classes", True), ("empty", True), ("all", False)])
@pytest.mark.parametrize("permuted", [False, True])
def test_lowered_membership_masks_preserve_pairs_and_tail_definitions(kind, encoded, permuted):
    """Region lowering preserves group multiplicity, topology and tail policies."""
    a, b = {0, 1}, {2, 3, 4, 5}
    if kind == "simple":
        groups = [[(a, b)], [(a, a)]]
    elif kind == "overlap":
        pair = ({0, 1, 2}, {1, 2, 3, 4})
        groups = [[pair, pair, ({2, 3}, {3, 4, 5})], []]
    elif kind == "classes":
        new, environment, core, old = {0}, {1, 2}, {3, 4}, {5}
        selected = [(old, core), (old, environment), (new, core), (new, environment),
                    (core, environment), (core, core), (new, new), (old, old)]
        # Duplicate all groups, then add an overlapping group; the encoded
        # finite table must retain counts >1 and union within each group.
        groups = [selected*2+[(old | core, new | core)], [(new, environment)]]
    elif kind == "empty":
        groups = [[(set(), set(range(6)))], []]
    else:
        groups = [[], []]
    order = np.array([5, 3, 1, 4, 0, 2]) if permuted else np.arange(6)
    source = _small_stock_system(groups, order)
    source.getForce(0).setUseDispersionCorrection(True)
    for index in (1, 2):
        # A convergent tail with a live lambda and independent source switches.
        force = source.getForce(index)
        force.setEnergyFunction("lambda*k1*k2/(1+r^6)")
        force.setUseLongRangeCorrection(index == 2)
    saved = mm.XmlSerializer.serialize(source)
    lowered = fixtures.fuse_pair_definition(source)
    names = {lowered.expression.getPerParticleParameterName(i)
             for i in range(lowered.expression.getNumPerParticleParameters())}
    assert ("interactionClass" in names) == encoded
    assert lowered.expression.getNumInteractionGroups() == 0
    assert not lowered.expression.getUseLongRangeCorrection()
    assert lowered.native_tail
    assert len(lowered.custom_tails) == 1
    assert mm.XmlSerializer.serialize(lowered.custom_tails[0]) == mm.XmlSerializer.serialize(source.getForce(2))
    assert mm.XmlSerializer.serialize(source) == saved
    positions = np.array([[.2+.18*i, .2+.05*(i % 2), .2] for i in range(6)])[order]
    combined = _reference_fused_system(source, lowered)
    # Compare the complete pair force independently of its retained source tail
    # definitions; separate grouped-tail tests cover the numerical corrections.
    direct = mm.XmlSerializer.deserialize(saved)
    direct.getForce(0).setUseDispersionCorrection(False)
    for index in (1, 2):
        direct.getForce(index).setUseLongRangeCorrection(False)
    for lam in (0., .5, 1.):
        expected = _evaluate_reference(direct, lam, positions)
        observed = _evaluate_reference(combined, lam, positions)
        np.testing.assert_allclose(observed[0], expected[0], rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(observed[1], expected[1], rtol=1e-12, atol=1e-10)
    assert mm.XmlSerializer.serialize(source) == saved


def test_membership_class_codes_are_exact_at_float32_boundary_and_fall_back():
    # Binary membership signatures generate 24/25 distinct classes without
    # prescribing the converter's class order or its emitted expression.
    for count in (24, 25):
        groups = [(set(i for i in range(count) if i & (1 << bit)), set()) for bit in range(5)]
        plan = fixtures._interaction_group_class_plan(count, groups)
        if count == 25:
            assert plan is None
            assert fixtures._choose_interaction_group_classes(count, groups, [(0, 5)]) is None
        else:
            codes = plan["codes"]
            first, second = np.triu_indices(count)
            exact = codes[first].astype(np.float64)+codes[second].astype(np.float64)
            actual = np.asarray(codes[first]+codes[second], dtype=np.float32)
            np.testing.assert_array_equal(actual.astype(np.float64), exact)
            assert len(set(actual.tolist())) == count*(count+1)//2
            assert max(actual) == 2**24
    assert fixtures._interaction_group_class_plan(0, [(set(), set())]) is None
    assert fixtures._choose_interaction_group_classes(0, [(set(), set())], [(0, 1)]) is None


def test_fused_fixture_shares_exact_particle_snapshots_and_reconverts_updates():
    source = _small_stock_system()
    native = source.getForce(0)
    for index in (1, 2):
        force = source.getForce(index)
        for atom in range(source.getNumParticles()):
            sigma = native.getParticleParameters(atom)[1].value_in_unit(unit.nanometer)
            force.setParticleParameters(atom, [sigma])
    saved = mm.XmlSerializer.serialize(source)
    lowered = fixtures.fuse_pair_definition(source)
    assert mm.XmlSerializer.serialize(source) == saved
    names = [lowered.expression.getPerParticleParameterName(i)
             for i in range(lowered.expression.getNumPerParticleParameters())]
    assert "pair1_k" not in names and "pair2_k" not in names
    expected = _evaluate_reference(source, .5)
    # Adding the expression transfers its ownership to the System. Keep that
    # System alive while the snapshot assertions below still read the force.
    combined = _reference_fused_system(source, lowered)
    actual = _evaluate_reference(combined, .5)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-10)

    # Sharing describes a complete fixed snapshot, independent of live globals.
    for force in source.getForces():
        if hasattr(force, "getNumGlobalParameters"):
            for index in range(force.getNumGlobalParameters()):
                force.setGlobalParameterDefaultValue(index, .25)
    changed = fixtures.fuse_pair_definition(source).expression
    assert names == [changed.getPerParticleParameterName(i)
                     for i in range(changed.getNumPerParticleParameters())]
    assert [changed.getParticleParameters(i) for i in range(source.getNumParticles())] == [
        lowered.expression.getParticleParameters(i) for i in range(source.getNumParticles())]

    # A distinct double value must remain independent even if FP32 would merge it.
    force = source.getForce(1)
    value = force.getParticleParameters(0)[0]
    next_value = np.nextafter(value, np.inf)
    assert np.float32(value).tobytes() == np.float32(next_value).tobytes()
    force.setParticleParameters(0, [next_value])
    reconverted = fixtures.fuse_pair_definition(source).expression
    assert reconverted.getNumPerParticleParameters() == len(names)+1


@pytest.mark.parametrize("case", ["zero", "sigma_only", "latent_epsilon",
                                  "cancelling_epsilon", "nonzero_base",
                                  "nonfinite_sigma", "nonfinite_sigma_offset"])
def test_fused_fixture_omits_only_permanently_zero_native_lj(case):
    source = _small_stock_system()
    native = source.getForce(0)
    for atom in range(source.getNumParticles()):
        _, sigma, _ = native.getParticleParameters(atom)
        native.setParticleParameters(atom, .1*(atom-2), sigma, -0. if atom % 2 else 0.)
    native.setExceptionParameters(0, 0, 1, .02, .25, .05)
    native.setParticleParameterOffset(0, "lambda", 2, 0., .03 if case == "sigma_only" else 0., 0.)
    if case == "latent_epsilon":
        for force in source.getForces():
            if hasattr(force, "getNumGlobalParameters"):
                for index in range(force.getNumGlobalParameters()):
                    force.setGlobalParameterDefaultValue(index, 0.)
        native.addParticleParameterOffset("lambda", 3, 0., 0., .1)
    elif case == "cancelling_epsilon":
        native.addParticleParameterOffset("lambda", 3, 0., 0., .1)
        native.addParticleParameterOffset("lambda", 3, 0., 0., -.1)
    elif case == "nonzero_base":
        charge, sigma, _ = native.getParticleParameters(3)
        native.setParticleParameters(3, charge, sigma, .1)
    elif case == "nonfinite_sigma":
        charge, _, epsilon = native.getParticleParameters(3)
        native.setParticleParameters(3, charge, np.inf, epsilon)
    elif case == "nonfinite_sigma_offset":
        native.addParticleParameterOffset("lambda", 3, 0., np.inf, 0.)
    saved = mm.XmlSerializer.serialize(source)
    lowered = fixtures.fuse_pair_definition(source)
    assert mm.XmlSerializer.serialize(source) == saved
    assert ("nativeSr6=" in lowered.expression.getEnergyFunction()) == (
        case not in ("zero", "sigma_only"))
    if case in ("zero", "sigma_only", "latent_epsilon", "nonzero_base"):
        combined = _reference_fused_system(source, lowered)
        for lam in (0., .5, 1.):
            expected, actual = _evaluate_reference(source, lam), _evaluate_reference(combined, lam)
            np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-10)
            np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-10)
    if case == "zero":
        # Retain source-specific tail policies even when the ordinary LJ is zero.
        native.setUseDispersionCorrection(True)
        source.getForce(1).setUseLongRangeCorrection(True)
        lowered = fixtures.fuse_pair_definition(source)
        assert lowered.native_tail and len(lowered.custom_tails) == 1
        assert mm.XmlSerializer.serialize(lowered.custom_tails[0]) == mm.XmlSerializer.serialize(source.getForce(1))
        charge, sigma, _ = native.getParticleParameters(0)
        native.setParticleParameters(0, charge, sigma, .2)
        assert "nativeSr6=" in fixtures.fuse_pair_definition(source).expression.getEnergyFunction()


def test_zero_native_lj_with_parameter_free_custom_expression():
    source = mm.System()
    source.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    native = mm.NonbondedForce()
    native.setNonbondedMethod(native.PME)
    native.setCutoffDistance(1.)
    native.setUseDispersionCorrection(False)
    custom = mm.CustomNonbondedForce("lambda/(1+r^6)")
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(1.)
    custom.addGlobalParameter("lambda", .5)
    for charge in (-.1, .1):
        source.addParticle(12)
        native.addParticle(charge, .3, 0.)
        custom.addParticle([])
    source.addForce(native)
    source.addForce(custom)
    lowered = fixtures.fuse_pair_definition(source)
    assert lowered.expression.getNumPerParticleParameters() == 0
    assert lowered.expression.getNumParticles() == 2
    positions = np.array([[.1, .1, .1], [.6, .1, .1]])
    expected = _evaluate_reference(source, .5, positions)
    actual = _evaluate_reference(_reference_fused_system(source, lowered), .5, positions)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-10)
