# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for unified nonbonded modes and Python ownership.

No Context is created. Run with --noconftest and the built plugin Python
directory on PYTHONPATH to avoid the GPU fixture in the broader plugin suite.
"""

import gc
import itertools

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

XYZ = np.array(list(itertools.product((.2, .85, 1.5), repeat=3)))
GROUP = 5
NAME = "pair definition <original> & copy"
EXPRESSION = "(a1*(1-lambda)+b1*lambda)*(a2*(1-lambda)+b2*lambda)/(1+r^2)"


def _source(mode):
    if mode == "custom":
        force = mm.CustomNonbondedForce(EXPRESSION)
        force.setNonbondedMethod(force.CutoffPeriodic)
        force.addPerParticleParameter("a")
        force.addPerParticleParameter("b")
        force.addGlobalParameter("lambda", .35)
        for i in range(len(XYZ)):
            force.addParticle([.2 + .01*i, .7 - .005*i])
        force.addExclusion(0, 1)
        force.addExclusion(2, 3)
    else:
        force = mm.NonbondedForce()
        force.setNonbondedMethod(force.PME)
        force.setEwaldErrorTolerance(5e-4)
        force.setUseDispersionCorrection(False)
        for i in range(len(XYZ)):
            force.addParticle(((-1)**i)*.1, .25 + .001*i, .2 + .002*i)
        force.addException(0, 1, 0., .3, 0.)
        force.addException(2, 3, -.003, .27, .1)
    force.setCutoffDistance(1.1)
    if mode != "custom":
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(.8)
    force.setForceGroup(GROUP)
    force.setName(NAME)
    return force


def _convert(mode, source=None):
    source = _source(mode) if source is None else source
    if mode == "custom":
        return nv.NonbondedForce(source)
    method = nv.NonbondedForce.PME if mode == "pme" else nv.NonbondedForce.Prolate
    return nv.NonbondedForce(source, 1.17, method)


def _numbers(values):
    return tuple(
        value.value_in_unit_system(unit.md_unit_system)
        if unit.is_quantity(value) else value
        for value in values
    )


def _definition(force):
    """Public physical definition, independent of XML formatting or caches."""
    result = {
        "mode": force.getPairMode(),
        "method": force.getNonbondedMethod(),
        "cutoff": force.getCutoffDistance(),
        "name": force.getName(),
        "group": force.getForceGroup(),
        "switch": (force.getUseSwitchingFunction(), force.getSwitchingDistance()),
        "particles": tuple(_numbers(force.getParticleParameters(i))
                           for i in range(force.getNumParticles())),
        "globals": tuple((force.getGlobalParameterName(i),
                           force.getGlobalParameterDefaultValue(i))
                          for i in range(force.getNumGlobalParameters())),
    }
    if force.getPairMode() == nv.NonbondedForce.Custom:
        result.update(
            expression=force.getEnergyFunction(),
            parameter_names=tuple(force.getPerParticleParameterName(i)
                                  for i in range(force.getNumPerParticleParameters())),
            exclusions=tuple(tuple(force.getExclusionParticles(i))
                             for i in range(force.getNumExclusions())),
            long_range=force.getUseLongRangeCorrection(),
        )
    else:
        result.update(
            combining_rule=force.getLJCombiningRule(),
            dispersion=force.getUseDispersionCorrection(),
            periodic_exceptions=force.getExceptionsUsePeriodicBoundaryConditions(),
            box_safety=force.getBoxSafetyFactor(),
            exceptions=tuple(_numbers(force.getExceptionParameters(i))
                             for i in range(force.getNumExceptions())),
        )
    return result


def _system(*forces):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0),
                                        mm.Vec3(0, 0, 4))
    for force in forces:
        system.addForce(force)
    return system


def test_mode_construction_and_custom_alias():
    assert nv.CustomNonbondedForce is nv.NonbondedForce
    assert nv.NonbondedForce().getPairMode() == nv.NonbondedForce.Standard
    # Global-parameter discovery must be safe for generic OpenMM state tools.
    assert nv.NonbondedForce().getNumGlobalParameters() == 0
    force = nv.NonbondedForce("lambda*a1*a2/r")
    force.setNonbondedMethod(nv.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(.9)
    force.addGlobalParameter("lambda", .4)
    force.addPerParticleParameter("a")
    force.addParticle([.3])
    assert force.getPairMode() == nv.NonbondedForce.Custom
    assert force.getNonbondedMethod() == nv.NonbondedForce.CutoffPeriodic
    assert force.getEnergyFunction() == "lambda*a1*a2/r"
    assert tuple(force.getParticleParameters(0)) == (.3,)


def test_custom_box_safety_is_part_of_the_definition():
    for force in (nv.NonbondedForce(_source("custom"), 1.25),
                  nv.NonbondedForce("r^2", 1.25)):
        force.setNonbondedMethod(nv.NonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(1.1)
        assert force.getPairMode() == nv.NonbondedForce.Custom
        assert force.getBoxSafetyFactor() == 1.25


@pytest.mark.parametrize("container", ["list", "tuple", "numpy"])
@pytest.mark.parametrize("with_units", [False, True])
def test_custom_particle_sequence_and_units_match_stock(container, with_units):
    stock = mm.CustomNonbondedForce("q1*q2/r")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    for name in ("q", "sigma", "epsilon"):
        stock.addPerParticleParameter(name)
    force = nv.NonbondedForce(stock)

    def record(scale):
        if with_units:
            values = [scale*.25*unit.elementary_charge,
                      scale*3.5*unit.angstrom,
                      scale*.2*unit.kilocalorie_per_mole]
        else:
            values = [scale*.25, scale*.35, scale*.8368]
        if container == "numpy":
            # Assignment avoids NumPy trying to interpret mixed physical units
            # as one homogeneous Quantity before the binding receives them.
            result = np.empty(3, dtype=object if with_units else np.float64)
            for i, value in enumerate(values):
                result[i] = value
            return result
        return tuple(values) if container == "tuple" else values

    assert stock.addParticle([.25, .35, .8368]) == 0
    assert force.addParticle(record(1)) == 0
    np.testing.assert_allclose(force.getParticleParameters(0), [.25, .35, .8368],
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(force.getParticleParameters(0), stock.getParticleParameters(0),
                               rtol=0, atol=1e-14)
    stock.setParticleParameters(0, [.5, .7, 1.6736])
    force.setParticleParameters(0, record(2))
    np.testing.assert_allclose(force.getParticleParameters(0), [.5, .7, 1.6736],
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(force.getParticleParameters(0), stock.getParticleParameters(0),
                               rtol=0, atol=1e-14)


@pytest.mark.parametrize("record", ["0.1 0.2", b"\x01\x02", [.1, object()],
                                     [[.1], [.2]], np.array([[.1, .2]])],
                         ids=["string", "bytes", "nonnumeric", "nested", "matrix"])
def test_malformed_custom_records_do_not_change_definition(record):
    force = _convert("custom")
    saved = _definition(force)
    for operation in (lambda: force.addParticle(record),
                      lambda: force.setParticleParameters(0, record)):
        with pytest.raises((TypeError, ValueError)):
            operation()
        assert _definition(force) == saved


@pytest.mark.parametrize("mode", ["pme", "custom"])
def test_wrong_mode_operations_fail_without_changing_definition(mode):
    force = _convert(mode)
    saved = _definition(force)
    if mode == "custom":
        assert force.getNumExceptions() == 0
        operations = (
            lambda: force.addParticle(.2, .3, .4),
            lambda: force.addException(0, 1, .1, .3, .2),
            lambda: force.setLJCombiningRule(nv.NonbondedForce.Geometric),
            force.getLJCombiningRule,
            lambda: force.setUseDispersionCorrection(True),
            lambda: force.addParticleParameterOffset("lambda", 0, .1, 0, 0),
            lambda: force.addExceptionParameterOffset("lambda", 0, .1, 0, 0),
        )
    else:
        assert force.getNumExclusions() == 0
        assert force.getNumGlobalParameters() == 0
        assert force.getNumPerParticleParameters() == 0
        operations = (
            force.getEnergyFunction,
            lambda: force.setEnergyFunction("r"),
            lambda: force.addParticle([.2, .3]),
            lambda: force.addPerParticleParameter("a"),
            lambda: force.setUseLongRangeCorrection(False),
            lambda: force.addExclusion(0, 1),
            lambda: force.createExclusionsFromBonds([(0, 1), (1, 2)], 2),
        )
    for operation in operations:
        with pytest.raises(Exception, match="[Mm]ode|[Ss]tandard|[Cc]ustom"):
            operation()
        assert _definition(force) == saved


@pytest.mark.parametrize("bond_cutoff", [1, 2, 3, 8])
@pytest.mark.parametrize("container", ["list", "tuple", "numpy"])
def test_bond_derived_exclusions_match_stock(bond_cutoff, container):
    stock = mm.CustomNonbondedForce("r^2")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    for _ in range(9):
        stock.addParticle([])
    force = nv.NonbondedForce(stock)
    # An eight-atom chain plus one isolated particle tests both path depth and
    # disconnected atoms. The last depth exceeds the chain's diameter.
    bonds = [(i, i + 1) for i in range(7)]
    stock.createExclusionsFromBonds(bonds, bond_cutoff)
    if container == "tuple":
        bonds = tuple(bonds)
    elif container == "numpy":
        bonds = np.asarray(bonds, dtype=np.int64)
    force.createExclusionsFromBonds(bonds, bond_cutoff)

    def pairs(current):
        result = [tuple(sorted(current.getExclusionParticles(i)))
                  for i in range(current.getNumExclusions())]
        assert len(result) == len(set(result))
        return set(result)

    expected = pairs(stock)
    assert pairs(force) == expected
    assert all(8 not in pair for pair in expected)
    assert (0, min(bond_cutoff, 7)) in expected
    if bond_cutoff < 7:
        assert (0, bond_cutoff + 1) not in expected


@pytest.mark.parametrize("bonds", [[(4, 5), (6,)], [(4, 5), (6, 7, 8)],
                                    [(4, 5), (6, 7.5)], [(4, 5), (6, 2**40)]],
                         ids=["short-pair", "long-pair", "noninteger", "overflow"])
def test_malformed_bond_sequences_do_not_add_partial_exclusions(bonds):
    force = _convert("custom")
    saved = _definition(force)
    # The first bond is valid: failure in a later record must precede mutation.
    with pytest.raises((TypeError, ValueError, OverflowError)):
        force.createExclusionsFromBonds(bonds, 2)
    assert _definition(force) == saved


@pytest.mark.parametrize("mode", ["pme", "custom"])
def test_source_conversion_owns_an_independent_definition(mode):
    source = _source(mode)
    force = _convert(mode, source)
    assert force.getName() == source.getName()
    assert force.getForceGroup() == source.getForceGroup()
    assert _numbers(force.getParticleParameters(0)) == _numbers(source.getParticleParameters(0))
    saved = _definition(force)
    if mode == "custom":
        source.setParticleParameters(0, [.8, .9])
        source.setGlobalParameterDefaultValue(0, .9)
        source.setEnergyFunction("a1*a2*r")
        source.setExclusionParticles(0, 0, 2)
    else:
        source.setParticleParameters(0, .5, .4, .7)
        source.setExceptionParameters(0, 0, 1, .1, .4, .2)
    source.setName("changed source")
    source.setForceGroup(7)
    source.setCutoffDistance(1.0)
    del source
    assert _definition(force) == saved


def test_custom_construction_rejects_interaction_groups():
    source = _source("custom")
    source.addInteractionGroup({0, 1, 2}, {1, 2, 3})
    with pytest.raises(Exception, match="interaction groups are not supported"):
        nv.NonbondedForce(source)


@pytest.mark.parametrize("force_class", [nv.NonbondedForce, nv.CustomNonbondedForce])
def test_custom_construction_preserves_explicit_switch_and_unswitched_expression(force_class):
    source = _source("custom")
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(.8)
    saved = mm.XmlSerializer.serialize(source)
    force = force_class(source)
    assert force.getUseSwitchingFunction()
    assert force.getSwitchingDistance() == .8
    assert force.getEnergyFunction() == EXPRESSION
    assert force.getCutoffInteraction().expression == EXPRESSION
    assert force.getCutoffInteraction().useSwitching
    assert force.getCutoffInteraction().switchingDistance == .8
    assert mm.XmlSerializer.serialize(force.getCutoffExpression()) == saved
    assert mm.XmlSerializer.serialize(source) == saved
    source.setSwitchingDistance(.6)
    assert force.getSwitchingDistance() == .8


@pytest.mark.parametrize("from_source", [False, True])
def test_custom_switch_setters_update_metadata_without_rewriting_expression(from_source):
    force = _convert("custom") if from_source else nv.NonbondedForce("r^2")
    saved = _definition(force)
    assert not force.getUseSwitchingFunction()
    force.setSwitchingDistance(.7)
    force.setUseSwitchingFunction(True)
    assert _definition(force) == dict(saved, switch=(True, .7))
    assert force.getCutoffInteraction().useSwitching
    assert force.getCutoffInteraction().switchingDistance == .7
    definition = force.getCutoffExpression()
    assert definition.getUseSwitchingFunction()
    assert definition.getSwitchingDistance().value_in_unit(unit.nanometer) == .7
    assert definition.getEnergyFunction() == saved["expression"]
    force.setUseSwitchingFunction(False)
    assert _definition(force) == dict(saved, switch=(False, .7))
    assert not force.getCutoffInteraction().useSwitching



@pytest.mark.parametrize("from_source", [False, True])
def test_custom_tail_setter_changes_only_correction_policy(from_source):
    force = _convert("custom") if from_source else nv.NonbondedForce("1/(1+r^6)")
    if from_source:
        force.setEnergyFunction(EXPRESSION.replace("1+r^2", "1+r^6"))
    before = _definition(force)
    force.setUseLongRangeCorrection(True)
    assert force.getUseLongRangeCorrection()
    assert _definition(force) == dict(before, long_range=True)
    force.setUseLongRangeCorrection(False)
    assert _definition(force) == before


def test_custom_conversion_preserves_owned_long_range_correction_definition():
    source = _source("custom")
    source.setEnergyFunction(EXPRESSION.replace("1+r^2", "1+r^6"))
    source.setUseLongRangeCorrection(True)
    before = mm.XmlSerializer.serialize(source)
    force = nv.NonbondedForce(source)
    assert force.getUseLongRangeCorrection()
    assert force.getEnergyFunction() == source.getEnergyFunction()
    assert mm.XmlSerializer.serialize(source) == before
    copied = _definition(force)
    source.setUseLongRangeCorrection(False)
    source.setEnergyFunction("0")
    assert _definition(force) == copied


@pytest.mark.parametrize("offset_kind", ["particle", "exception"])
def test_standard_conversion_preserves_parameter_offsets(offset_kind):
    source = _source("pme")
    source.addGlobalParameter("lambda", .3)
    if offset_kind == "particle":
        source.addParticleParameterOffset("lambda", 0, .1, 0, 0)
    else:
        source.addExceptionParameterOffset("lambda", 0, .1, 0, 0)
    before = mm.XmlSerializer.serialize(source)
    force = nv.NonbondedForce(source)
    assert force.getNumGlobalParameters() == 1
    assert force.getGlobalParameterName(0) == "lambda"
    assert force.getGlobalParameterDefaultValue(0) == .3
    if offset_kind == "particle":
        assert force.getNumParticleParameterOffsets() == 1
        assert _numbers(force.getParticleParameterOffset(0)) == _numbers(
            source.getParticleParameterOffset(0))
    else:
        assert force.getNumExceptionParameterOffsets() == 1
        assert _numbers(force.getExceptionParameterOffset(0)) == _numbers(
            source.getExceptionParameterOffset(0))
    assert mm.XmlSerializer.serialize(source) == before


def test_standard_conversion_preserves_inactive_exceptions():
    source = _source("pme")
    for index in (0, 1):
        source.setParticleParameters(index, 0, .3, 0)
    force = _convert("pme", source)
    assert force.getNumExceptions() == source.getNumExceptions()
    for index in range(source.getNumExceptions()):
        assert _numbers(force.getExceptionParameters(index)) == _numbers(
            source.getExceptionParameters(index))
    force.setParticleParameters(0, .5, .3, .2)
    assert tuple(force.getExceptionParameters(0)) == (0, 1, 0., .3, 0.)


def test_standard_exception_distance_defaults_to_nonperiodic():
    assert not mm.NonbondedForce().getExceptionsUsePeriodicBoundaryConditions()
    assert not nv.NonbondedForce().getExceptionsUsePeriodicBoundaryConditions()
    for method in (nv.NonbondedForce.PME, nv.NonbondedForce.Prolate):
        force = nv.NonbondedForce(method, 1., 1.)
        assert not force.getExceptionsUsePeriodicBoundaryConditions()


@pytest.mark.parametrize("mode", ["pme", "prolate"])
@pytest.mark.parametrize("periodic", [False, True])
def test_standard_exception_distance_convention_is_copied(mode, periodic):
    source = _source(mode)
    source.setExceptionsUsePeriodicBoundaryConditions(periodic)
    before = mm.XmlSerializer.serialize(source)
    force = _convert(mode, source)
    assert force.getExceptionsUsePeriodicBoundaryConditions() == periodic
    assert mm.XmlSerializer.serialize(source) == before
    definition = _definition(force)
    force.setExceptionsUsePeriodicBoundaryConditions(not periodic)
    assert _definition(force) == {**definition, "periodic_exceptions": not periodic}
    assert mm.XmlSerializer.serialize(source) == before
    force.setExceptionsUsePeriodicBoundaryConditions(periodic)
    source.setExceptionsUsePeriodicBoundaryConditions(not periodic)
    assert _definition(force) == definition


def test_custom_global_defaults_and_duplicate_names():
    force = _convert("custom")
    assert force.addGlobalParameter("other", .6) == 1
    force.setGlobalParameterDefaultValue(0, .8)
    force.setGlobalParameterName(1, "renamed")
    assert force.getGlobalParameterDefaultValue(0) == .8
    assert force.getGlobalParameterName(1) == "renamed"
    before = _definition(force)
    for operation in (lambda: force.addGlobalParameter("lambda", .7),
                      lambda: force.setGlobalParameterName(1, "lambda")):
        with pytest.raises(Exception, match="duplicate global parameter"):
            operation()
        assert _definition(force) == before


@pytest.mark.parametrize("mode", ["pme", "custom"])
def test_system_returns_typed_borrowed_plugin_force(mode):
    force = _convert(mode)
    before = _definition(force)
    system = _system(force)
    assert not force.thisown
    for borrowed in (system.getForce(0), system.getForces()[0]):
        assert isinstance(borrowed, nv.NonbondedForce)
        assert not borrowed.thisown
        assert _definition(borrowed) == before
    del force, borrowed
    gc.collect()
    assert _definition(system.getForce(0)) == before


@pytest.mark.parametrize("mode,cls", [("pme", mm.NonbondedForce),
                                     ("custom", mm.CustomNonbondedForce)])
def test_system_keeps_stock_force_type_and_ownership(mode, cls):
    system = _system(_source(mode))
    borrowed = system.getForce(0)
    assert type(borrowed) is cls
    assert not borrowed.thisown
    assert borrowed.getNumParticles() == len(XYZ)


@pytest.mark.parametrize("mode", ["pme", "prolate", "custom"])
def test_physical_components_match_conversion_and_compatibility_accessors(mode):
    force = _convert(mode)
    cutoff = force.getCutoffInteraction()
    assert force.usesPeriodicBoundaryConditions()
    assert force.hasLatticeSumInteraction() == (mode != "custom")
    if mode == "custom":
        assert cutoff.evaluator == nv.CutoffInteraction.Expression
        assert cutoff.expression == EXPRESSION
        assert cutoff.tail == nv.CutoffInteraction.NoTail
        assert not cutoff.useSwitching
        cutoff.expression = "1.7*(" + EXPRESSION.split(";", 1)[0] + ")"
        cutoff.useSwitching = True
        cutoff.switchingDistance = .75
        cutoff.tail = nv.CutoffInteraction.HomogeneousTail
        assert force.getEnergyFunction() == EXPRESSION  # Getter is an independent value.
        force.setCutoffInteraction(cutoff)
        assert force.getEnergyFunction() == cutoff.expression
        assert force.getUseSwitchingFunction()
        assert force.getSwitchingDistance() == .75
        assert force.getUseLongRangeCorrection()
        assert force.getCutoffInteraction().tail == nv.CutoffInteraction.HomogeneousTail
        assert force.getCutoffExpression().getUseSwitchingFunction()
        force.setUseLongRangeCorrection(False)
        assert force.getCutoffInteraction().tail == nv.CutoffInteraction.NoTail
        with pytest.raises(Exception, match="no LatticeSumInteraction"):
            force.getLatticeSumInteraction()
    else:
        assert cutoff.evaluator == nv.CutoffInteraction.BuiltinLJ
        assert cutoff.useSwitching == force.getUseSwitchingFunction()
        assert cutoff.switchingDistance == force.getSwitchingDistance()
        assert cutoff.tail == nv.CutoffInteraction.NoTail
        cutoff.combiningRule = nv.CutoffInteraction.Geometric
        cutoff.tail = nv.CutoffInteraction.HomogeneousTail
        cutoff.switchingDistance = .75
        assert force.getLJCombiningRule() == nv.NonbondedForce.LorentzBerthelot
        assert not force.getUseDispersionCorrection()
        force.setCutoffInteraction(cutoff)
        assert force.getLJCombiningRule() == nv.NonbondedForce.Geometric
        assert force.getUseDispersionCorrection()
        assert force.getSwitchingDistance() == .75
        split = force.getLatticeSumInteraction()
        expected = nv.LatticeSumInteraction.Ewald if mode == "pme" else nv.LatticeSumInteraction.Prolate
        assert split.splitting == expected
        split.splitting = nv.LatticeSumInteraction.Prolate if mode == "pme" else nv.LatticeSumInteraction.Ewald
        assert force.getLatticeSumInteraction().splitting == expected
        force.setLatticeSumInteraction(split)
        assert force.getNonbondedMethod() == (nv.NonbondedForce.Prolate if mode == "pme" else nv.NonbondedForce.PME)
        # Legacy setters update the same physical definition.
        force.setUseSwitchingFunction(False)
        force.setUseDispersionCorrection(False)
        force.setLJCombiningRule(nv.NonbondedForce.LorentzBerthelot)
        current = force.getCutoffInteraction()
        assert not current.useSwitching
        assert current.tail == nv.CutoffInteraction.NoTail
        assert current.combiningRule == nv.CutoffInteraction.LorentzBerthelot


@pytest.mark.parametrize("mode,field,value", [
    ("pme", "evaluator", 99),
    ("pme", "combiningRule", 99),
    ("pme", "tail", 99),
    ("pme", "expression", "r"),
    ("custom", "evaluator", 0),
    ("custom", "combiningRule", 1),
    ("custom", "tail", 2),
    ("pme", "tail", 2),
])
def test_invalid_cutoff_component_does_not_mutate_force(mode, field, value):
    force = _convert(mode)
    before = _definition(force)
    cutoff = force.getCutoffInteraction()
    cutoff.expression = "r^2" if mode == "custom" else ""
    setattr(cutoff, field, value)
    with pytest.raises(Exception, match="CutoffInteraction"):
        force.setCutoffInteraction(cutoff)
    assert _definition(force) == before


@pytest.mark.parametrize("mode", ["pme", "custom"])
def test_invalid_lattice_sum_component_does_not_mutate_force(mode):
    force = _convert(mode)
    before = _definition(force)
    split = nv.LatticeSumInteraction(nv.CoulombRadialKernel(), "q")
    split.splitting = 99
    with pytest.raises(Exception, match="splitting"):
        force.setLatticeSumInteraction(split)
    assert _definition(force) == before
