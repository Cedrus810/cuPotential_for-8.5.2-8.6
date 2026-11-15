# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bulk cutoff definitions preserve the complete-definition API's behavior."""

import contextlib

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv


def _definition():
    definition = mm.CustomNonbondedForce("lambda*a1*a2/r^6")
    definition.setNonbondedMethod(definition.CutoffPeriodic)
    definition.setCutoffDistance(1.)
    definition.setUseSwitchingFunction(True)
    definition.setSwitchingDistance(.8)
    definition.addGlobalParameter("lambda", .5)
    definition.addPerParticleParameter("a")
    for value in (.02, .03, .04, .05):
        definition.addParticle([value])
    definition.addExclusion(0, 1)
    definition.addExclusion(2, 3)
    return definition


def _force(mode):
    if mode == "custom":
        definition = _definition()
        # The bulk call must turn off the original automatic expression LRC.
        definition.setUseLongRangeCorrection(True)
        return nv.NonbondedForce(definition)
    source = mm.NonbondedForce()
    source.setNonbondedMethod(source.PME)
    source.setCutoffDistance(1.)
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(.8)
    source.setUseDispersionCorrection(True)
    source.addGlobalParameter("lambda", .5)
    for charge, epsilon in zip((.1, -.1, .05, -.05), (.15, .2, .25, .3)):
        source.addParticle(charge, .3, epsilon)
    source.addException(0, 1, -.003, .25, .01)
    source.addException(2, 3, 0., .3, 0.)
    source.addParticleParameterOffset("lambda", 0, .03, 0., 0.)
    source.addExceptionParameterOffset("lambda", 0, .001, .01, .02)
    force = nv.NonbondedForce(source, 1., nv.NonbondedForce.PME)
    force.setDispersionCorrectionFollowsParameters(False)
    return force


def _bulk_arguments():
    return dict(expression="lambda*a1*a2/r^6", parameterNames=["a"],
                particleParameters=[[.02], [.03], [.04], [.05]],
                globalParameters={"lambda": .5})


def _add_tail(force):
    tail = _definition()
    tail.setEnergyFunction("tail_scale*a1*a2/r^6")
    tail.addGlobalParameter("tail_scale", .7)
    tail.setUseLongRangeCorrection(True)
    tail.addInteractionGroup({0, 1}, {2, 3})
    force.addExpressionTailCorrection(tail)


@pytest.mark.parametrize("mode,preserve", [("standard", False), ("standard", True),
                                           ("custom", False)])
def test_bulk_definition_matches_complete_definition_and_round_trips(mode, preserve):
    bulk, complete = _force(mode), _force(mode)
    # Tail-only globals and independently owned group/switch definitions survive.
    for force in (bulk, complete):
        _add_tail(force)
    bulk.setCutoffInteraction(**_bulk_arguments(), preserveDispersionCorrection=preserve)
    complete.setCutoffInteraction(_definition(), preserveDispersionCorrection=preserve)

    xml = mm.XmlSerializer.serialize(bulk)
    assert xml == mm.XmlSerializer.serialize(complete)
    for restored in (mm.XmlSerializer.deserialize(xml), mm.XmlSerializer.clone(bulk)):
        assert isinstance(restored, nv.NonbondedForce)
        assert mm.XmlSerializer.serialize(restored) == xml
    installed = bulk.getCutoffExpression()
    assert installed.getCutoffDistance() == 1.*unit.nanometer
    assert installed.getUseSwitchingFunction()
    assert installed.getSwitchingDistance() == .8*unit.nanometer
    assert not installed.getUseLongRangeCorrection()
    assert {tuple(installed.getExclusionParticles(i))
            for i in range(installed.getNumExclusions())} == {(0, 1), (2, 3)}
    if mode == "standard":
        assert bulk.getUseDispersionCorrection() == preserve


@pytest.mark.parametrize("mode", ["standard", "custom"])
def test_bulk_definition_copies_inputs_and_supports_parameter_free_defaults(mode):
    force = _force(mode)
    arguments = _bulk_arguments()
    force.setCutoffInteraction(**arguments)
    saved = mm.XmlSerializer.serialize(force)
    arguments["parameterNames"][0] = "changed"
    arguments["particleParameters"][0][0] = .9
    arguments["globalParameters"]["lambda"] = .9
    returned = force.getCutoffExpression()
    returned.setParticleParameters(1, [.9])
    assert mm.XmlSerializer.serialize(force) == saved

    # Empty records still specify the particle count; omitted globals and the
    # preservation flag use their documented defaults.
    force.setCutoffInteraction(expression=".002/r^6", parameterNames=[],
                               particleParameters=[[] for _ in range(4)])
    installed = force.getCutoffExpression()
    assert installed.getNumPerParticleParameters() == 0
    assert installed.getNumParticles() == 4
    assert not installed.getUseLongRangeCorrection()
    assert force.getGlobalParameterName(0) == "lambda"
    if mode == "standard":
        assert not force.getUseDispersionCorrection()


@pytest.mark.parametrize("rows", ["tuples", "numpy", "quantities", "quantity_array"])
def test_bulk_particle_records_accept_python_sequences_and_openmm_units(rows):
    force = _force("standard")
    arguments = _bulk_arguments()
    values = [.02, .03, .04, .05]
    if rows == "tuples":
        arguments["particleParameters"] = list(zip(values))
    elif rows == "numpy":
        arguments["particleParameters"] = np.asarray(values).reshape(4, 1)
    elif rows == "quantities":
        arguments["particleParameters"] = [[10*value*unit.angstrom] for value in values]
    else:
        arguments["particleParameters"] = unit.Quantity(
            10*np.asarray(values).reshape(4, 1), unit.angstrom)
    arguments["globalParameters"]["length_scale"] = 5*unit.angstrom
    force.setCutoffInteraction(**arguments)
    installed = force.getCutoffExpression()
    np.testing.assert_allclose([installed.getParticleParameters(i) for i in range(4)],
                               np.asarray(values).reshape(4, 1))
    global_parameters = {installed.getGlobalParameterName(i): installed.getGlobalParameterDefaultValue(i)
                         for i in range(installed.getNumGlobalParameters())}
    assert global_parameters["length_scale"] == .5


@pytest.mark.parametrize("invalid", ["name_item", "name_string", "row_value", "row_string",
                                      "global_key", "global_value", "global_container"])
def test_bulk_input_conversion_rejection_is_atomic(invalid):
    force = _force("standard")
    force.setCutoffInteraction(**_bulk_arguments(), preserveDispersionCorrection=True)
    saved = mm.XmlSerializer.serialize(force)
    bad = _bulk_arguments()
    bad["particleParameters"][0][0] = .9
    bad["globalParameters"]["must_not_be_added"] = 2.
    if invalid == "name_item":
        bad["parameterNames"] = [42]
    elif invalid == "name_string":
        bad["parameterNames"] = "a"
    elif invalid == "row_value":
        bad["particleParameters"][2] = ["not a number"]
    elif invalid == "row_string":
        bad["particleParameters"][2] = "0.04"
    elif invalid == "global_key":
        bad["globalParameters"][42] = .1
    elif invalid == "global_value":
        bad["globalParameters"]["must_not_be_added"] = "not a number"
    else:
        bad["globalParameters"] = list(bad["globalParameters"].items())
    with pytest.raises(TypeError):
        force.setCutoffInteraction(**bad, preserveDispersionCorrection=True)
    assert mm.XmlSerializer.serialize(force) == saved


@pytest.mark.parametrize("mode", ["standard", "custom"])
@pytest.mark.parametrize("invalid", ["count", "width", "duplicate", "name_syntax", "nan", "inf",
                                      "overflow", "global_conflict", "global_nonfinite"])
def test_invalid_bulk_definition_is_atomic(mode, invalid):
    force = _force(mode)
    preserve = mode == "standard"
    force.setCutoffInteraction(**_bulk_arguments(), preserveDispersionCorrection=preserve)
    _add_tail(force)
    saved = mm.XmlSerializer.serialize(force)
    bad = _bulk_arguments()
    bad["globalParameters"]["must_not_be_added"] = 2.
    bad["particleParameters"][0][0] = .9
    if invalid == "count":
        bad["particleParameters"].pop()
    elif invalid == "width":
        bad["particleParameters"][2].append(.1)
    elif invalid == "duplicate":
        bad["parameterNames"] = ["a", "a"]
        bad["particleParameters"] = [[.1, .2] for _ in range(4)]
    elif invalid == "name_syntax":
        bad["parameterNames"] = ["a-b"]
    elif invalid in ("nan", "inf", "overflow"):
        bad["particleParameters"][2][0] = {
            "nan": float("nan"), "inf": float("inf"), "overflow": 1e40}[invalid]
    elif invalid == "global_conflict":
        bad["globalParameters"]["lambda"] = .9
    else:
        bad["globalParameters"]["must_not_be_added"] = float("nan")
    with pytest.raises(Exception):
        force.setCutoffInteraction(**bad, preserveDispersionCorrection=preserve)
    assert mm.XmlSerializer.serialize(force) == saved


def test_bulk_custom_mode_cannot_preserve_standard_dispersion_correction():
    force = _force("custom")
    saved = mm.XmlSerializer.serialize(force)
    with pytest.raises(Exception, match="Standard|native"):
        force.setCutoffInteraction(**_bulk_arguments(), preserveDispersionCorrection=True)
    assert mm.XmlSerializer.serialize(force) == saved


@pytest.mark.parametrize("invalid", ["mixed", "missing_names", "missing_records"])
def test_bulk_arguments_reject_ambiguous_or_incomplete_calls(invalid):
    force = _force("standard")
    saved = mm.XmlSerializer.serialize(force)
    arguments = _bulk_arguments()
    if invalid == "mixed":
        arguments["interaction"] = _definition()
    elif invalid == "missing_names":
        del arguments["parameterNames"]
    else:
        del arguments["particleParameters"]
    with pytest.raises(TypeError):
        force.setCutoffInteraction(**arguments)
    assert mm.XmlSerializer.serialize(force) == saved


@pytest.mark.parametrize("mode,preserve", [("standard", False), ("standard", True),
                                           ("custom", False)])
def test_bulk_matches_complete_definition_with_live_updates_cuda(plugin_dir, mode, preserve):
    forces = [_force(mode), _force(mode)]
    arguments = _bulk_arguments()
    complete_definition = _definition()
    forces[0].setCutoffInteraction(**arguments, preserveDispersionCorrection=preserve)
    forces[1].setCutoffInteraction(complete_definition, preserve)
    systems, integrators, contexts = [], [], []
    try:
        for force in forces:
            _add_tail(force)
            system = mm.System()
            for _ in range(4):
                system.addParticle(12.)
            system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
            system.addForce(force)
            integrator = mm.VerletIntegrator(.001)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                                 {"Precision": "mixed"})
            # Nonexcluded pairs cover the unswitched, switched, and cutoff regions.
            context.setPositions([mm.Vec3(x, .3, .4) for x in (.2, .5, 1.05, 1.35)])
            systems.append(system)
            integrators.append(integrator)
            contexts.append(context)

        def compare():
            states = [context.getState(getEnergy=True, getForces=True) for context in contexts]
            energies = [state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        for state in states]
            np.testing.assert_allclose(energies[0], energies[1], rtol=1e-6, atol=1e-5)
            force_unit = unit.kilojoule_per_mole/unit.nanometer
            np.testing.assert_allclose(states[0].getForces(asNumpy=True).value_in_unit(force_unit),
                                       states[1].getForces(asNumpy=True).value_in_unit(force_unit),
                                       rtol=1e-6, atol=1e-5)
            return energies[0]

        initial = compare()
        for context in contexts:
            context.setParameter("lambda", .73)
            context.setParameter("tail_scale", .9)
        live = compare()
        assert live != initial

        arguments["particleParameters"][2][0] = .09
        complete_definition.setParticleParameters(2, [.09])
        forces[0].setCutoffInteraction(**arguments, preserveDispersionCorrection=preserve)
        forces[1].setCutoffInteraction(complete_definition, preserve)
        for force, context in zip(forces, contexts):
            if mode == "standard":
                force.setParticleParameters(2, -.12, .32, .24)
                force.setExceptionParameters(0, 0, 1, -.02, .27, .04)
            force.updateParametersInContext(context)
            assert context.getParameter("lambda") == .73
            assert context.getParameter("tail_scale") == .9
        assert compare() != live
    finally:
        # Drop the loop's last Context reference as well as the owning lists.
        context = None
        contexts.clear()
        integrators.clear()


@pytest.mark.parametrize("mode", ["standard", "custom"])
@pytest.mark.parametrize("tail", [False, True])
def test_bulk_tail_correction_matches_complete_definition_and_round_trips(mode, tail):
    bulk, complete = _force(mode), _force(mode)
    arguments = _bulk_arguments()
    if tail:
        # Existing owner globals can supply the expression's defaults.
        del arguments["globalParameters"]
        arguments["tailCorrection"] = True
    bulk.setCutoffInteraction(**arguments)
    definition = _definition()
    definition.setUseLongRangeCorrection(tail)
    complete.setCutoffInteraction(definition)
    xml = mm.XmlSerializer.serialize(bulk)
    assert xml == mm.XmlSerializer.serialize(complete)
    for restored in (bulk, mm.XmlSerializer.deserialize(xml), mm.XmlSerializer.clone(bulk)):
        assert mm.XmlSerializer.serialize(restored) == xml
        installed = restored.getCutoffExpression()
        assert installed.getUseLongRangeCorrection() == tail
        assert installed.getUseSwitchingFunction()
        assert installed.getSwitchingDistance() == .8*unit.nanometer
        assert restored.getNumExpressionTailCorrections() == 0
        if mode == "standard":
            assert not restored.getUseDispersionCorrection()


@pytest.mark.parametrize("mode", ["standard", "custom"])
def test_bulk_tail_keeps_independent_source_corrections(mode):
    force = _force(mode)
    _add_tail(force)
    before = mm.XmlSerializer.serialize(force.getExpressionTailCorrection(0))
    force.setCutoffInteraction(**_bulk_arguments(), tailCorrection=True)
    assert force.getCutoffExpression().getUseLongRangeCorrection()
    assert force.getNumExpressionTailCorrections() == 1
    assert mm.XmlSerializer.serialize(force.getExpressionTailCorrection(0)) == before


@pytest.mark.parametrize("mode", ["standard", "custom"])
def test_bulk_tail_and_preserved_standard_tail_are_mutually_exclusive(mode):
    force = _force(mode)
    force.setCutoffInteraction(**_bulk_arguments())
    saved = mm.XmlSerializer.serialize(force)
    arguments = _bulk_arguments()
    arguments["particleParameters"][0][0] = .9
    arguments["globalParameters"]["must_not_be_added"] = 2.
    with pytest.raises(Exception):
        force.setCutoffInteraction(**arguments, preserveDispersionCorrection=True,
                                   tailCorrection=True)
    assert mm.XmlSerializer.serialize(force) == saved


@pytest.mark.parametrize("kind", ["definition", "descriptor"])
@pytest.mark.parametrize("tail", [False, True])
def test_tail_correction_keyword_rejects_legacy_overloads_atomically(kind, tail):
    force = _force("standard")
    saved = mm.XmlSerializer.serialize(force)
    interaction = _definition() if kind == "definition" else force.getCutoffInteraction()
    with pytest.raises(TypeError):
        force.setCutoffInteraction(interaction, tailCorrection=tail)
    assert mm.XmlSerializer.serialize(force) == saved


def test_bulk_tail_correction_rejects_ljpme_atomically():
    source = mm.NonbondedForce()
    source.setNonbondedMethod(source.LJPME)
    source.setCutoffDistance(1.)
    source.addGlobalParameter("lambda", .5)
    for _ in range(4):
        source.addParticle(0., .3, .2)
    force = nv.NonbondedForce(source, 1., nv.NonbondedForce.PME)
    saved = mm.XmlSerializer.serialize(force)
    with pytest.raises(Exception, match="LJPME|dispersion|P6"):
        force.setCutoffInteraction(**_bulk_arguments(), tailCorrection=True)
    assert mm.XmlSerializer.serialize(force) == saved


@contextlib.contextmanager
def _tail_context(force, platform, positions, length):
    system = mm.System()
    for _ in range(force.getNumParticles()):
        system.addParticle(12.)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0),
                                        mm.Vec3(0, 0, length))
    integrator = mm.VerletIntegrator(.001)
    properties = {"Precision": "mixed"} if platform == "CUDA" else {}
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    context.setPositions(positions)
    try:
        yield context
    finally:
        del context, integrator


def _energy_and_forces(context):
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))


@pytest.mark.parametrize("mode", ["standard", "custom"])
def test_bulk_tail_follows_retained_globals_volume_and_records_cuda(plugin_dir, mode):
    reference = _definition()
    reference.setUseLongRangeCorrection(True)
    if mode == "custom":
        force = nv.NonbondedForce(reference, 1.25)
    else:
        source = mm.NonbondedForce()
        source.setNonbondedMethod(source.PME)
        source.setCutoffDistance(1.)
        source.setUseSwitchingFunction(True)
        source.setSwitchingDistance(.8)
        source.setUseDispersionCorrection(True)
        source.addGlobalParameter("lambda", .5)
        for _ in range(4):
            source.addParticle(0., .3, .2)
        source.addException(0, 1, 0., .3, 0.)
        source.addException(2, 3, 0., .3, 0.)
        force = nv.NonbondedForce(source, 1.25, nv.NonbondedForce.PME)
    arguments = _bulk_arguments()
    del arguments["globalParameters"]
    force.setCutoffInteraction(**arguments, tailCorrection=True)
    assert force.getNumExpressionTailCorrections() == 0
    # All pairs lie beyond the cutoff, so the only energy is C(lambda)/V.
    positions = [mm.Vec3(.2, .2, .2), mm.Vec3(1.7, .2, .2),
                 mm.Vec3(.2, 1.7, .2), mm.Vec3(.2, .2, 1.7)]
    with _tail_context(reference, "Reference", positions, 6.) as expected, \
            _tail_context(force, "CUDA", positions, 6.) as actual:
        def compare():
            wanted_energy, wanted_forces = _energy_and_forces(expected)
            actual_energy, actual_forces = _energy_and_forces(actual)
            np.testing.assert_allclose(actual_energy, wanted_energy, rtol=2e-5, atol=1e-10)
            np.testing.assert_array_equal(wanted_forces, np.zeros((4, 3)))
            np.testing.assert_array_equal(actual_forces, np.zeros((4, 3)))
            return actual_energy

        initial = compare()
        assert initial > 0.
        for lam in (0., .37, 1.):
            for length in (6., 6.3):
                for context in (expected, actual):
                    context.setParameter("lambda", lam)
                    context.setPeriodicBoxVectors(mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0),
                                                  mm.Vec3(0, 0, length))
                np.testing.assert_allclose(compare(), initial*(lam/.5)*(6./length)**3,
                                           rtol=2e-5, atol=1e-10)
        before = compare()
        arguments["particleParameters"][2][0] = .09
        reference.setParticleParameters(2, [.09])
        force.setCutoffInteraction(**arguments, tailCorrection=True)
        for definition, context in ((reference, expected), (force, actual)):
            definition.updateParametersInContext(context)
            assert context.getParameter("lambda") == 1.
        assert compare() > before


@pytest.mark.parametrize("switching", [False, True])
def test_combined_softcore_endpoint_tail_matches_original_lj_cuda(plugin_dir, switching):
    source = mm.NonbondedForce()
    source.setNonbondedMethod(source.PME)
    source.setCutoffDistance(1.)
    source.setUseSwitchingFunction(switching)
    source.setSwitchingDistance(.8)
    source.setUseDispersionCorrection(True)
    rows = [[.24, .12, 0.], [.29, .3, 0.], [.35, .18, 1.], [.41, .45, 1.]]
    for sigma, epsilon, _ in rows:
        source.addParticle(0., sigma, epsilon)
    source.addException(0, 1, 0., .25, .01)
    # At lambda=1 the selected softcore branch equals ordinary LJ. The region
    # mask still divides environment pairs from all pairs touching the solute.
    expression = ("4*eps*((1-selected)*(u*u-u)+selected*lambda*(v*v-v));"
                  "u=(sig/r)^6;v=1/(.5*(1-lambda)+(r/sig)^6);"
                  "eps=sqrt(epsilon1*epsilon2);sig=.5*(sigma1+sigma2);"
                  "selected=region1+region2-region1*region2")
    stock_on, stock_off = mm.XmlSerializer.clone(source), mm.XmlSerializer.clone(source)
    stock_off.setUseDispersionCorrection(False)
    bulk_on = nv.NonbondedForce(source, 1., nv.NonbondedForce.PME)
    bulk_off = nv.NonbondedForce(source, 1., nv.NonbondedForce.PME)
    for force, tail in ((bulk_on, True), (bulk_off, False)):
        force.setCutoffInteraction(expression=expression,
                                   parameterNames=["sigma", "epsilon", "region"],
                                   particleParameters=rows, globalParameters={"lambda": 1.},
                                   tailCorrection=tail)
        assert not force.getUseDispersionCorrection()
        assert force.getNumExpressionTailCorrections() == 0
    positions = [mm.Vec3(x, .3, .4) for x in (.2, .5, 1.05, 1.35)]
    with contextlib.ExitStack() as stack:
        contexts = [stack.enter_context(_tail_context(force, platform, positions, 4.))
                    for force, platform in ((stock_on, "Reference"), (stock_off, "Reference"),
                                            (bulk_on, "CUDA"), (bulk_off, "CUDA"))]
        observations = [_energy_and_forces(context) for context in contexts]
        stock_tail = observations[0][0]-observations[1][0]
        expression_tail = observations[2][0]-observations[3][0]
        assert stock_tail < 0.
        # Subtraction isolates the tail from direct and reciprocal numerical
        # differences. Agreement also rules out adding the original LJ tail twice.
        np.testing.assert_allclose(expression_tail, stock_tail, rtol=2e-5, atol=1e-8)
        np.testing.assert_array_equal(observations[0][1], observations[1][1])
        np.testing.assert_array_equal(observations[2][1], observations[3][1])
