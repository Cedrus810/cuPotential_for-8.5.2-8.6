# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin objects retain their interfaces through workflow cloning boundaries."""

import copy
import gc
import xml.etree.ElementTree as ET

import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv


def _roundtrip(obj, operation):
    if operation == "deserialize":
        return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(obj))
    if operation == "clone":
        return mm.XmlSerializer.clone(obj)
    return copy.deepcopy(obj)


def _integrator(seed=815):
    integrator = nv.LangevinMiddleIntegrator(310*unit.kelvin, 2/unit.picosecond,
                                            2*unit.femtosecond, comFrequency=0)
    integrator.setRandomNumberSeed(seed)
    integrator.setConstraintTolerance(1e-7)
    integrator.setIntegrationForceGroups(3)
    return integrator


def _barostat():
    barostat = nv.MonteCarloBarostat(1.5*unit.bar, 310*unit.kelvin, frequency=17, seed=915)
    barostat.setForceGroup(2)
    barostat.setName("workflow pressure control")
    return barostat


def _nonbonded():
    definition = mm.CustomNonbondedForce("lambda_sterics*k1*k2/(1+r^6)")
    definition.addGlobalParameter("lambda_sterics", .4)
    definition.addPerParticleParameter("k")
    definition.addParticle([.2])
    definition.addParticle([.3])
    definition.setNonbondedMethod(definition.CutoffPeriodic)
    definition.setCutoffDistance(.9)
    return nv.NonbondedForce(definition)


def _system():
    system = mm.System()
    system.addParticle(12)
    system.addParticle(12)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0), mm.Vec3(0, 0, 3.2))
    system.addForce(_nonbonded())
    system.addForce(_barostat())
    return system


@pytest.mark.parametrize("operation", ["deserialize", "clone", "deepcopy"])
@pytest.mark.parametrize("seed", [0, 815])
def test_integrator_roundtrip_preserves_settings_units_and_ownership(operation, seed):
    original = _integrator(seed)
    xml = mm.XmlSerializer.serialize(original)
    restored = _roundtrip(original, operation)
    assert type(restored) is type(original)
    assert restored.thisown
    assert restored.getTemperature() == 310*unit.kelvin
    assert restored.getFriction() == 2/unit.picosecond
    assert restored.getStepSize() == 2*unit.femtosecond
    assert restored.getComFrequency() == 0
    assert restored.getConstraintTolerance() == 1e-7
    assert restored.getIntegrationForceGroups() == 3
    assert restored.getRandomNumberSeed() == seed
    assert mm.XmlSerializer.serialize(restored) == xml
    del original
    gc.collect()
    restored.setTemperature(320*unit.kelvin)
    assert restored.getTemperature() == 320*unit.kelvin


@pytest.mark.parametrize("operation", ["deserialize", "clone", "deepcopy"])
@pytest.mark.parametrize("factory", [_nonbonded, _barostat], ids=["nonbonded", "barostat"])
def test_force_roundtrip_returns_an_independent_owning_plugin_type(operation, factory):
    original = factory()
    xml = mm.XmlSerializer.serialize(original)
    restored = _roundtrip(original, operation)
    assert type(restored) is type(original)
    assert restored.thisown
    assert mm.XmlSerializer.serialize(restored) == xml
    if factory is _barostat:
        assert restored.getDefaultPressure() == 1.5*unit.bar
        assert restored.getDefaultTemperature() == 310*unit.kelvin
        assert restored.getFrequency() == 17
        assert restored.getRandomNumberSeed() == 915
        assert restored.getForceGroup() == 2
    else:
        assert restored.getNumGlobalParameters() == 1
        assert restored.getGlobalParameterName(0) == "lambda_sterics"
    del original
    gc.collect()
    restored.setName("independent clone")
    assert restored.getName() == "independent clone"


@pytest.mark.parametrize("operation", ["deserialize", "clone", "deepcopy"])
def test_system_returns_borrowed_plugin_types_and_retains_stock_types(operation):
    source = _system()
    source.addForce(mm.HarmonicBondForce())
    system = _roundtrip(source, operation)
    assert mm.XmlSerializer.serialize(system) == mm.XmlSerializer.serialize(source)
    for index in (0, 1):
        borrowed = system.getForce(index)
        assert type(borrowed) is type(source.getForce(index))
        assert not borrowed.thisown
        borrowed.setName("borrowed view " + str(index))
        del borrowed
        gc.collect()
        assert system.getForce(index).getName() == "borrowed view " + str(index)
    assert type(system.getForce(2)) is mm.HarmonicBondForce
    assert not system.getForce(2).thisown
    assert source.getForce(1).getName() == "workflow pressure control"


@pytest.mark.parametrize("operation", ["deserialize", "clone", "deepcopy"])
def test_compound_integrator_returns_borrowed_plugin_integrator(operation):
    source = mm.CompoundIntegrator()
    source.addIntegrator(_integrator())
    source.addIntegrator(mm.VerletIntegrator(.001))
    source.setCurrentIntegrator(1)
    restored = _roundtrip(source, operation)
    assert restored.getCurrentIntegrator() == 1
    borrowed = restored.getIntegrator(0)
    assert type(borrowed) is type(source.getIntegrator(0))
    assert not borrowed.thisown
    borrowed.setTemperature(320*unit.kelvin)
    del borrowed
    gc.collect()
    assert restored.getIntegrator(0).getTemperature() == 320*unit.kelvin
    assert source.getIntegrator(0).getTemperature() == 310*unit.kelvin
    assert type(restored.getIntegrator(1)) is mm.VerletIntegrator


@pytest.mark.parametrize("factory", [_integrator, _barostat], ids=["integrator", "barostat"])
def test_unknown_dynamics_serialization_version_is_rejected(factory):
    node = ET.fromstring(mm.XmlSerializer.serialize(factory()))
    node.set("version", "999")
    with pytest.raises(Exception, match="[Uu]nsupported.*serialization version"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


def test_openmmtools_state_clone_and_integrator_standardization():
    pytest.importorskip("openmmtools")
    from openmmtools import cache, states

    class AlchemicalState(states.GlobalParameterState):
        lambda_sterics = states.GlobalParameterState.GlobalParameter("lambda_sterics", standard_value=1.0)

    system = _system()
    state = states.CompoundThermodynamicState(
        states.ThermodynamicState(system, temperature=310*unit.kelvin, pressure=1.5*unit.bar),
        composable_states=[AlchemicalState.from_system(system)])
    restored = copy.deepcopy(state)
    assert restored.lambda_sterics == .4
    assert restored.temperature == 310*unit.kelvin
    assert restored.pressure == 1.5*unit.bar
    assert mm.XmlSerializer.serialize(restored.system) == mm.XmlSerializer.serialize(state.system)
    integrator = _integrator()
    assert restored._is_integrator_thermostated(integrator)
    standardized = cache.ContextCache._standardize_integrator(integrator)
    assert type(standardized) is type(integrator)
    assert standardized.thisown
    assert unit.is_quantity(standardized.getTemperature())
    assert unit.is_quantity(standardized.getFriction())
    assert standardized.getComFrequency() == integrator.getComFrequency()
    assert cache.ContextCache._generate_integrator_id(integrator) == cache.ContextCache._generate_integrator_id(copy.deepcopy(integrator))


@pytest.mark.parametrize("generic", [False, True], ids=["typed", "generic"])
def test_context_preserves_integrator_identity_ownership_and_thermodynamic_interface(generic):
    system = _system()
    integrator = _integrator()
    plugin_type = type(integrator)
    if generic:
        integrator = mm.XmlSerializer.deserialize.__wrapped__(mm.XmlSerializer.serialize(integrator))
        assert type(integrator) is mm.Integrator
    original_ownership = integrator.thisown
    assert original_ownership
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    try:
        view = context.getIntegrator()
        assert view is integrator
        assert type(view) is plugin_type
        assert view.thisown == original_ownership
        view.setTemperature(320*unit.kelvin)
        view.setFriction(3/unit.picosecond)
        del view
        gc.collect()
        assert integrator.thisown == original_ownership
        assert integrator.getTemperature() == 320*unit.kelvin
        assert integrator.getFriction() == 3/unit.picosecond
        context.setPositions([[.2, .2, .2], [.5, .2, .2]])
        assert context.getState(getEnergy=True).getPotentialEnergy().unit.is_compatible(unit.kilojoule_per_mole)
    finally:
        del context
