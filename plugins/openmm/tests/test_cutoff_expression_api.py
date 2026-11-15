# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cutoff API ownership, explicit-update atomicity, and tail definitions.

Tests without ``cuda`` in their names do not create a CUDA Context.
Plugin serialization is intentionally not part of this feature slice.
"""
import numpy as np
import openmm as mm
from openmm import unit
import pytest
import nv_openmm_plugin as nv


def _definitions(method="PME"):
    source = mm.NonbondedForce()
    source.setNonbondedMethod(source.PME)
    source.setCutoffDistance(1.0)
    source.setUseDispersionCorrection(False)
    source.addGlobalParameter("lambda", .5)
    for charge in (.1, -.1, 0., 0.):
        source.addParticle(charge, .3, 0.)
    source.addException(0, 1, -.003, .25, .01)
    # This pair is inactive under Standard, but the expression activates it.
    source.addException(2, 3, 0., .3, 0.)
    source.addParticleParameterOffset("lambda", 0, .03, 0., 0.)
    force = nv.NonbondedForce(source, 1., getattr(nv.NonbondedForce, method))
    custom = mm.CustomNonbondedForce("lambda*a1*a2/r^6")
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(1.0)
    custom.addGlobalParameter("lambda", .5)
    custom.addPerParticleParameter("a")
    for value in (.2, .3, .4, .5):
        custom.addParticle([value])
    # Exclusion ordering and pair orientation need not match Standard.
    custom.addExclusion(3, 2)
    custom.addExclusion(1, 0)
    return source, force, custom


def _xml(force):
    return mm.XmlSerializer.serialize(force)


def _numbers(values):
    return [value.value_in_unit_system(unit.md_unit_system) if unit.is_quantity(value)
            else value for value in values]


def _definition_state(force):
    """Snapshot public values without requiring plugin serialization."""
    return (
        tuple((force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i))
              for i in range(force.getNumGlobalParameters())),
        tuple(tuple(_numbers(force.getParticleParameters(i))) for i in range(force.getNumParticles())),
        tuple(tuple(_numbers(force.getExceptionParameters(i))) for i in range(force.getNumExceptions())),
        tuple(tuple(_numbers(force.getParticleParameterOffset(i)))
              for i in range(force.getNumParticleParameterOffsets())),
        force.getUseSwitchingFunction(), force.getSwitchingDistance(),
        force.getUseDispersionCorrection(),
        _xml(force.getCutoffExpression()) if force.hasCutoffExpression() else None,
        tuple(_xml(force.getExpressionTailCorrection(i))
              for i in range(force.getNumExpressionTailCorrections())),
    )


@pytest.mark.parametrize("preserve", (False, True))
def test_replacement_explicitly_preserves_native_tail_and_owns_source_switch(preserve):
    _, force, custom = _definitions()
    force.setUseDispersionCorrection(True)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(.8)
    force.setCutoffInteraction(custom, preserveDispersionCorrection=preserve)
    assert force.getUseDispersionCorrection() == preserve
    assert not force.getUseSwitchingFunction()
    assert not force.getCutoffExpression().getUseSwitchingFunction()
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(.6)
    assert force.getCutoffExpression().getUseSwitchingFunction()
    assert force.getCutoffExpression().getSwitchingDistance().value_in_unit(unit.nanometer) == .6
    assert force.getUseDispersionCorrection() == preserve
    # Record updates may preserve the original native reference tail again;
    # they must not reconstruct it from the now-unswitched replacement.
    custom.setParticleParameters(0, [.7])
    force.setCutoffInteraction(custom, preserve)
    assert force.getUseDispersionCorrection() == preserve
    assert not force.getUseSwitchingFunction()


@pytest.mark.parametrize("mode", ["standalone", "fused"])
def test_cutoff_switch_metadata_round_trips_through_all_configuration_paths(mode):
    _, force, custom = _definitions()
    custom.setUseSwitchingFunction(True)
    custom.setSwitchingDistance(.7)
    custom.setUseLongRangeCorrection(True)
    original = _xml(custom)
    if mode == "standalone":
        force = nv.NonbondedForce(custom)
    else:
        force.setCutoffInteraction(custom)
    assert _xml(custom) == original
    assert _xml(force.getCutoffExpression()) == original
    assert force.getUseSwitchingFunction()
    assert force.getSwitchingDistance() == .7

    cutoff = force.getCutoffInteraction()
    assert cutoff.useSwitching and cutoff.switchingDistance == .7
    assert cutoff.tail == nv.CutoffInteraction.HomogeneousTail
    assert cutoff.expression == custom.getEnergyFunction()
    cutoff.switchingDistance = .65
    force.setCutoffInteraction(cutoff)
    assert force.getSwitchingDistance() == .65
    assert force.getCutoffExpression().getSwitchingDistance().value_in_unit(unit.nanometer) == .65

    # Compatibility setters and descriptor setters must update the same owned
    # source definition. The GPU's lowered S*U is never exposed by these getters.
    force.setSwitchingDistance(.6)
    force.setUseSwitchingFunction(False)
    cutoff = force.getCutoffInteraction()
    assert not cutoff.useSwitching and cutoff.switchingDistance == .6
    returned = force.getCutoffExpression()
    assert not returned.getUseSwitchingFunction()
    assert returned.getEnergyFunction() == custom.getEnergyFunction()
    returned.setUseSwitchingFunction(True)
    returned.setSwitchingDistance(.8)
    assert not force.getUseSwitchingFunction()  # Getter remains an owned copy.
    force.setCutoffInteraction(returned)
    assert force.getUseSwitchingFunction() and force.getSwitchingDistance() == .8


@pytest.mark.parametrize("mode", ["standalone", "fused"])
@pytest.mark.parametrize("start", [-.1, 1., 1.1, float("nan"), float("inf"), 1.-1e-9],
                         ids=["negative", "at_cutoff", "above_cutoff", "nan", "inf", "fp32_collapse"])
def test_invalid_explicit_cutoff_switch_definition_is_atomic(mode, start):
    _, force, custom = _definitions()
    if mode == "standalone":
        force = nv.NonbondedForce(custom)
    else:
        force.setCutoffInteraction(custom)
    before = _xml(force.getCutoffExpression())
    globals_before = force.getNumGlobalParameters()
    bad = force.getCutoffExpression()
    bad.addGlobalParameter("must_not_be_added", 2.)
    bad.setParticleParameters(0, [.9])
    bad.setUseSwitchingFunction(True)
    bad.setSwitchingDistance(start)
    with pytest.raises(Exception, match="switching|FP32"):
        force.setCutoffInteraction(bad)
    assert _xml(force.getCutoffExpression()) == before
    assert force.getNumGlobalParameters() == globals_before

    # The compact descriptor follows the same active-interval contract.
    cutoff = force.getCutoffInteraction()
    cutoff.expression = "2*lambda*a1*a2/r^6"
    cutoff.useSwitching = True
    cutoff.switchingDistance = start
    with pytest.raises(Exception, match="switching|FP32"):
        force.setCutoffInteraction(cutoff)
    assert _xml(force.getCutoffExpression()) == before


@pytest.mark.parametrize("mode", ["standalone", "fused"])
def test_inactive_cutoff_switch_distance_remains_metadata(mode):
    _, force, custom = _definitions()
    custom.setSwitchingDistance(1.1)
    if mode == "standalone":
        force = nv.NonbondedForce(custom)
    else:
        force.setCutoffInteraction(custom)
    assert not force.getUseSwitchingFunction()
    assert force.getSwitchingDistance() == 1.1
    cutoff = force.getCutoffInteraction()
    cutoff.switchingDistance = -.3
    force.setCutoffInteraction(cutoff)
    assert not force.getUseSwitchingFunction()
    assert force.getSwitchingDistance() == -.3
    assert force.getCutoffExpression().getSwitchingDistance().value_in_unit(unit.nanometer) == -.3


def test_tail_definitions_own_group_switch_policy_and_share_globals():
    _, force, custom = _definitions()
    force.setCutoffInteraction(custom)
    tail = mm.XmlSerializer.deserialize(_xml(custom))
    tail.setUseLongRangeCorrection(True)
    tail.setUseSwitchingFunction(True)
    tail.setSwitchingDistance(.8)
    tail.addInteractionGroup({0, 1}, {1, 2, 3})
    tail.addGlobalParameter("tail_scale", .75)
    tail.setEnergyFunction("lambda*tail_scale*a1*a2/r^6")
    assert force.addExpressionTailCorrection(tail) == 0
    saved = _xml(force.getExpressionTailCorrection(0))
    tail.setParticleParameters(0, [.9])
    returned = force.getExpressionTailCorrection(0)
    returned.setEnergyFunction("0")
    assert _xml(force.getExpressionTailCorrection(0)) == saved
    assert force.getExpressionTailCorrection(0).getNumInteractionGroups() == 1
    assert force.getExpressionTailCorrection(0).getUseSwitchingFunction()
    force.setGlobalParameterDefaultValue(0, .3)
    assert force.getCutoffExpression().getGlobalParameterDefaultValue(0) == .3
    assert force.getExpressionTailCorrection(0).getGlobalParameterDefaultValue(0) == .3
    force.setGlobalParameterName(1, "tail_amplitude")
    assert force.getExpressionTailCorrection(0).getGlobalParameterName(1) == "tail_amplitude"
    before = _definition_state(force)
    bad = force.getExpressionTailCorrection(0)
    bad.addGlobalParameter("must_not_be_added", 2.)
    bad.setGlobalParameterDefaultValue(0, .4)
    with pytest.raises(Exception, match="conflicting default"):
        force.setExpressionTailCorrection(0, bad)
    assert _definition_state(force) == before


@pytest.mark.parametrize("invalid", ("disabled", "nonperiodic", "cutoff", "particles", "records", "group", "switch"))
def test_invalid_tail_definition_is_atomic(invalid):
    _, force, custom = _definitions()
    force.setCutoffInteraction(custom)
    custom.setUseLongRangeCorrection(True)
    force.addExpressionTailCorrection(custom)
    before = _definition_state(force)
    bad = force.getExpressionTailCorrection(0)
    bad.addGlobalParameter("must_not_be_added", 1.)
    if invalid == "disabled":
        bad.setUseLongRangeCorrection(False)
    elif invalid == "nonperiodic":
        bad.setNonbondedMethod(bad.CutoffNonPeriodic)
    elif invalid == "cutoff":
        bad.setCutoffDistance(.9)
    elif invalid == "particles":
        bad.addParticle([.1])
    elif invalid == "records":
        bad.setParticleParameters(0, [float("nan")])
    elif invalid == "group":
        bad.addInteractionGroup({0}, {99})
    else:
        bad.setUseSwitchingFunction(True)
        bad.setSwitchingDistance(1.1)
    with pytest.raises(Exception, match="tail"):
        force.setExpressionTailCorrection(0, bad)
    assert _definition_state(force) == before


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_replacement_owns_definition_and_preserves_standard_metadata(method):
    source, force, custom = _definitions(method)
    assert not force.hasCutoffExpression()
    with pytest.raises(Exception, match="no cutoff expression"):
        force.getCutoffExpression()
    assert force.getNumExceptions() == source.getNumExceptions() == 2
    force.setCutoffInteraction(custom)
    assert force.getPairMode() == force.Standard
    assert force.hasCutoffExpression()
    assert force.getParticleParameterOffset(0)[0] == "lambda"
    assert _numbers(force.getExceptionParameters(0)) == _numbers(source.getExceptionParameters(0))
    expected = _xml(force.getCutoffExpression())
    custom.setParticleParameters(0, [.9])
    returned = force.getCutoffExpression()
    returned.setEnergyFunction("0")
    assert _xml(force.getCutoffExpression()) == expected


def test_inactive_source_exclusions_survive_conversion_without_offsets():
    source = mm.NonbondedForce()
    source.setNonbondedMethod(source.PME)
    source.setUseDispersionCorrection(False)
    for _ in range(2):
        source.addParticle(0., .3, 0.)
    source.addException(0, 1, 0., .3, 0.)
    force = nv.NonbondedForce(source)
    assert force.getNumExceptions() == 1
    assert _numbers(force.getExceptionParameters(0)) == _numbers(source.getExceptionParameters(0))


def test_replacement_globals_share_names_defaults_and_updates():
    _, force, custom = _definitions()
    custom.addGlobalParameter("scale", 1.2)
    force.setCutoffInteraction(custom)
    assert force.getNumGlobalParameters() == 2
    force.setGlobalParameterDefaultValue(0, .7)
    assert force.getCutoffExpression().getGlobalParameterDefaultValue(0) == .7
    force.setGlobalParameterName(1, "amplitude")
    assert force.getCutoffExpression().getGlobalParameterName(1) == "amplitude"
    saved = _definition_state(force)
    bad = force.getCutoffExpression()
    bad.setGlobalParameterDefaultValue(0, .3)
    with pytest.raises(Exception, match="conflicting default"):
        force.setCutoffInteraction(bad)
    assert _definition_state(force) == saved


@pytest.mark.parametrize("invalid", ["mode", "cutoff", "particles", "exclusions", "groups", "switch"])
def test_invalid_replacement_does_not_change_existing_definition(invalid):
    _, force, custom = _definitions()
    force.setCutoffInteraction(custom)
    saved = _definition_state(force)
    if invalid == "mode":
        custom.setNonbondedMethod(custom.NoCutoff)
        message = "CutoffPeriodic"
    elif invalid == "cutoff":
        custom.setCutoffDistance(.9)
        message = "cutoff"
    elif invalid == "particles":
        custom.addParticle([.6])
        message = "particle count"
    elif invalid == "exclusions":
        custom.setExclusionParticles(0, 0, 2)
        message = "exactly match"
    elif invalid == "groups":
        custom.addInteractionGroup({0}, {1, 2, 3})
        message = "interaction groups"
    else:
        custom.setUseSwitchingFunction(True)
        custom.setSwitchingDistance(1.0)
        message = "switching"
    with pytest.raises(Exception, match=message):
        force.setCutoffInteraction(custom)
    assert _definition_state(force) == saved


def _custom_definition_state(force):
    return (_xml(force.getCutoffExpression()),
            tuple(_xml(force.getExpressionTailCorrection(i))
                  for i in range(force.getNumExpressionTailCorrections())))


def test_standalone_cutoff_definition_owns_records_and_preserves_tail_globals():
    _, _, definition = _definitions()
    force = nv.NonbondedForce(definition)
    assert force.hasCutoffExpression()
    assert not force.hasLatticeSumInteraction()
    tail = mm.XmlSerializer.deserialize(_xml(definition))
    tail.addGlobalParameter("tail_scale", .75)
    tail.setEnergyFunction("tail_scale*a1*a2/r^6")
    tail.setUseLongRangeCorrection(True)
    force.addExpressionTailCorrection(tail)
    saved_tail = _xml(force.getExpressionTailCorrection(0))

    # The incoming definition predates the tail-only global. A complete source
    # update must keep that global available to the independent correction.
    definition.addGlobalParameter("amplitude", 1.2)
    definition.setEnergyFunction("amplitude*lambda*a1*a2/r^6")
    definition.setParticleParameters(0, [.8])
    force.setCutoffInteraction(definition, preserveDispersionCorrection=False)
    assert force.getPairMode() == force.Custom
    assert not force.hasLatticeSumInteraction()
    assert dict((force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i))
                for i in range(force.getNumGlobalParameters())) == {
                    "lambda": .5, "tail_scale": .75, "amplitude": 1.2}
    assert list(force.getParticleParameters(0)) == [.8]
    assert _xml(force.getExpressionTailCorrection(0)) == saved_tail
    saved = _custom_definition_state(force)
    definition.setParticleParameters(0, [.1])
    returned = force.getCutoffExpression()
    returned.setEnergyFunction("0")
    assert _custom_definition_state(force) == saved
    force.setCutoffInteraction(force.getCutoffExpression())
    assert _custom_definition_state(force) == saved


@pytest.mark.parametrize("invalid", ["cutoff", "particles", "preserve_native_tail",
                                     "conflict", "tail_conflict", "duplicate", "nonfinite"])
def test_standalone_complete_cutoff_definition_rejection_is_atomic(invalid):
    _, _, definition = _definitions()
    force = nv.NonbondedForce(definition)
    tail = mm.XmlSerializer.deserialize(_xml(definition))
    tail.addGlobalParameter("tail_scale", .75)
    tail.setUseLongRangeCorrection(True)
    force.addExpressionTailCorrection(tail)
    saved = _custom_definition_state(force)
    bad = mm.XmlSerializer.deserialize(_xml(definition))
    bad.addGlobalParameter("must_not_be_added", 2.)
    bad.setParticleParameters(0, [.9])
    preserve = False
    if invalid == "cutoff":
        bad.setCutoffDistance(.9)
        message = "cutoff"
    elif invalid == "particles":
        bad.addParticle([.7])
        message = "particle count"
    elif invalid == "preserve_native_tail":
        preserve = True
        message = "Standard|native"
    elif invalid == "conflict":
        bad.setGlobalParameterDefaultValue(0, .9)
        message = "conflicting default"
    elif invalid == "tail_conflict":
        bad.addGlobalParameter("tail_scale", .8)
        message = "conflicting default"
    elif invalid == "duplicate":
        bad.addGlobalParameter("lambda", .5)
        message = "duplicate"
    else:
        bad.addGlobalParameter("invalid", float("nan"))
        message = "finite"
    with pytest.raises(Exception, match=message):
        force.setCutoffInteraction(bad, preserve)
    assert _custom_definition_state(force) == saved


def test_cutoff_descriptor_updates_expression_and_restores_native_lj():
    source, force, definition = _definitions()
    force.setLJCombiningRule(force.Geometric)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(.8)
    force.setUseDispersionCorrection(True)
    native = force.getCutoffInteraction()
    force.setCutoffInteraction(definition, True)
    cutoff = force.getCutoffInteraction()
    cutoff.expression = "2*lambda*a1*a2/r^6"
    cutoff.tail = nv.CutoffInteraction.HomogeneousTail
    force.setCutoffInteraction(cutoff)
    assert force.getCutoffExpression().getEnergyFunction() == cutoff.expression
    assert force.getCutoffExpression().getUseLongRangeCorrection()
    assert list(force.getCutoffExpression().getParticleParameters(0)) == [.2]
    assert force.getUseDispersionCorrection()
    force.setCutoffInteraction(interaction=native)
    assert not force.hasCutoffExpression()
    assert force.hasLatticeSumInteraction()
    assert force.getCutoffInteraction().evaluator == nv.CutoffInteraction.BuiltinLJ
    assert force.getLJCombiningRule() == force.Geometric
    assert force.getUseSwitchingFunction()
    assert force.getSwitchingDistance() == .8
    assert force.getUseDispersionCorrection()
    assert force.getNumExceptions() == source.getNumExceptions()
    with pytest.raises(Exception, match="no cutoff expression"):
        force.getCutoffExpression()


@pytest.mark.parametrize("change", ["add", "expression", "layout", "switch", "switch_distance",
                                     "disable_switch", "tail"])
def test_replacement_structure_changes_require_reinitialize_cuda(plugin_dir, change):
    _, force, custom = _definitions()
    if change in ("switch_distance", "disable_switch"):
        custom.setUseSwitchingFunction(True)
        custom.setSwitchingDistance(.8)
    if change != "add":
        force.setCutoffInteraction(custom)
    system = mm.System()
    for _ in range(force.getNumParticles()):
        system.addParticle(12.)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    try:
        context.setPositions([mm.Vec3(.2 + .6*i, .3, .4) for i in range(force.getNumParticles())])
        before = context.getState(getEnergy=True).getPotentialEnergy()
        if change == "expression":
            custom.setEnergyFunction("2*lambda*a1*a2/r^6")
        elif change == "layout":
            custom.setPerParticleParameterName(0, "b")
        elif change == "switch":
            custom.setUseSwitchingFunction(True)
            custom.setSwitchingDistance(.8)
        elif change == "switch_distance":
            custom.setSwitchingDistance(.65)
        elif change == "disable_switch":
            custom.setUseSwitchingFunction(False)
        elif change == "tail":
            custom.setUseLongRangeCorrection(True)
        force.setCutoffInteraction(custom)
        with pytest.raises(Exception, match="reinitialize"):
            force.updateParametersInContext(context)
        after = context.getState(getEnergy=True).getPotentialEnergy()
        assert after.value_in_unit(unit.kilojoule_per_mole) == before.value_in_unit(unit.kilojoule_per_mole)
    finally:
        del context, integrator


@pytest.mark.parametrize("mode", ["standalone", "fused"])
def test_inactive_cutoff_switch_distance_update_preserves_context_cuda(plugin_dir, mode):
    _, force, custom = _definitions()
    custom.setSwitchingDistance(.8)
    if mode == "standalone":
        force = nv.NonbondedForce(custom)
    else:
        force.setCutoffInteraction(custom)
    system = mm.System()
    for _ in range(force.getNumParticles()):
        system.addParticle(12.)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    try:
        context.setPositions([mm.Vec3(.2 + .6*i, .3, .4) for i in range(force.getNumParticles())])
        before = context.getState(getEnergy=True, getForces=True)
        force.setSwitchingDistance(.6)
        force.updateParametersInContext(context)
        assert force.getSwitchingDistance() == .6
        assert not force.getUseSwitchingFunction()
        after = context.getState(getEnergy=True, getForces=True)
        assert after.getPotentialEnergy() == before.getPotentialEnergy()
        force_unit = unit.kilojoule_per_mole/unit.nanometer
        np.testing.assert_array_equal(after.getForces(asNumpy=True).value_in_unit(force_unit),
                                      before.getForces(asNumpy=True).value_in_unit(force_unit))
    finally:
        del context, integrator


@pytest.mark.parametrize("invalid", ["wrong_length", "nan", "inf", "fp32_overflow"])
@pytest.mark.parametrize("tail", [False, True], ids=["no_tail", "live_tail"])
def test_invalid_replacement_records_preserve_all_context_parameters_cuda(plugin_dir, invalid, tail):
    _, force, custom = _definitions()
    custom.setUseLongRangeCorrection(tail)
    force.setCutoffInteraction(custom)
    system = mm.System()
    for _ in range(force.getNumParticles()):
        system.addParticle(12.)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    try:
        context.setPositions([mm.Vec3(.2 + .6*i, .3, .4) for i in range(force.getNumParticles())])
        context.setParameter("lambda", .73)

        def snapshot():
            state = context.getState(getEnergy=True, getForces=True)
            return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))

        before_energy, before_forces = snapshot()
        # Both are valid updates, but must remain unapplied when another part of
        # this same update fails validation. Include reciprocal, bare 1-4 data,
        # and the previously integrated tail coefficient when enabled.
        force.setParticleParameters(0, .4, .3, 0.)
        force.setExceptionParameters(0, 0, 1, -.021, .27, .032)
        bad_record = {"wrong_length": [.2, .3], "nan": [float("nan")],
                      "inf": [float("inf")], "fp32_overflow": [1e40]}[invalid]
        custom.setParticleParameters(2, bad_record)
        force.setCutoffInteraction(custom)
        message = "number of particle parameters" if invalid == "wrong_length" else "finite FP32"
        with pytest.raises(Exception, match=message):
            force.updateParametersInContext(context)
        after_energy, after_forces = snapshot()
        assert after_energy == before_energy
        np.testing.assert_array_equal(after_forces, before_forces)
    finally:
        del context, integrator


def test_invalid_additional_tail_preserves_direct_and_all_corrections_cuda(plugin_dir):
    _, force, custom = _definitions()
    force.setCutoffInteraction(custom)
    first = mm.XmlSerializer.deserialize(_xml(custom))
    first.setUseLongRangeCorrection(True)
    second = mm.XmlSerializer.deserialize(_xml(first))
    second.setEnergyFunction("lambda*sqrt(a1*a2)/r^6")
    second.addInteractionGroup({0, 1}, {1, 2, 3})
    force.addExpressionTailCorrection(first)
    force.addExpressionTailCorrection(second)
    system = mm.System()
    for _ in range(force.getNumParticles()):
        system.addParticle(12.)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    try:
        context.setPositions([mm.Vec3(.2+.6*i, .3, .4) for i in range(force.getNumParticles())])
        context.setParameter("lambda", .73)

        def snapshot():
            state = context.getState(getEnergy=True, getForces=True)
            return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))

        before = snapshot()
        force.setParticleParameters(0, .4, .3, 0.)
        force.setExceptionParameters(0, 0, 1, -.021, .27, .032)
        custom.setParticleParameters(0, [.8])
        force.setCutoffInteraction(custom)
        first.setParticleParameters(0, [.9])
        force.setExpressionTailCorrection(0, first)
        # Finite records pass the public setter, but the second candidate tail
        # has a non-real pair energy. Failure must leave the first tail, direct
        # records, live reciprocal charges and exception parameters unchanged.
        second.setParticleParameters(2, [-.4])
        force.setExpressionTailCorrection(1, second)
        with pytest.raises(Exception, match="did not converge|not finite"):
            force.updateParametersInContext(context)
        after = snapshot()
        assert after[0] == before[0]
        np.testing.assert_array_equal(after[1], before[1])
        second.setParticleParameters(2, [.4])
        force.setExpressionTailCorrection(1, second)
        force.updateParametersInContext(context)
        assert snapshot()[0] != before[0]
    finally:
        del context, integrator
