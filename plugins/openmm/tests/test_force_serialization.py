# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force definitions survive cloning, fresh-process loading and live updates."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

KINDS = ("pme", "prolate", "ljpme", "ljprolate", "ljmixed", "dispersion-only",
         "native-cutoff", "custom", "custom-p1", "custom-p6", "custom-mixed",
         "fused-pme", "fused-prolate", "fused-ljpme")


def _custom_definition():
    force = mm.CustomNonbondedForce("lambda*k1*k2/(1+r^6)")
    force.addGlobalParameter("lambda", .4)
    force.addPerParticleParameter("k")
    for value in (.1, .2, .3):
        force.addParticle([value])
    force.addExclusion(0, 1)
    force.addExclusion(1, 2)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(.9)
    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(.7)
    force.setUseLongRangeCorrection(True)
    return force


def _force(kind, periodic=True):
    if kind.startswith("custom"):
        definition = _custom_definition()
        if kind != "custom":
            terms = [".002/(1+r^6)"]
            if kind in ("custom-p1", "custom-mixed"):
                terms.append("1.7*lambda^2*k1*k2/r")
            if kind in ("custom-p6", "custom-mixed"):
                terms.append("-.6*k1*k2/r^6")
            definition.setEnergyFunction("+".join(terms))
            definition.setUseSwitchingFunction(False)
            definition.setUseLongRangeCorrection(False)
        force = nv.NonbondedForce(definition, nv.NonbondedForce.AutoBoxSafety)
        # Deliberately add P6 first: collection order and each independent
        # splitting, signed coupling and live weight expression must survive.
        if kind in ("custom-p6", "custom-mixed"):
            force.addLatticeSumInteraction(nv.LatticeSumInteraction(
                nv.DispersionRadialKernel(-.6), "k", nv.LatticeSumInteraction.Prolate))
        if kind in ("custom-p1", "custom-mixed"):
            force.addLatticeSumInteraction(nv.LatticeSumInteraction(
                nv.CoulombRadialKernel(1.7), "lambda*k", nv.LatticeSumInteraction.Ewald))
    else:
        source = mm.NonbondedForce()
        ljpme = "lj" in kind or kind == "dispersion-only"
        source.setNonbondedMethod(source.LJPME if ljpme else source.PME)
        source.setCutoffDistance(.9)
        source.setExceptionsUsePeriodicBoundaryConditions(periodic)
        source.setUseSwitchingFunction(True)
        source.setSwitchingDistance(.7)
        source.addGlobalParameter("lambda", .4)
        for charge in (.2, -.3, .1):
            source.addParticle(charge, .3, .2)
        source.addException(0, 1, -.03, .25, .07)
        # Keep an initially inactive replacement whose live offset activates it.
        source.addException(1, 2, 0., .28, 0.)
        source.addParticleParameterOffset("lambda", 2, .1, .02, .03)
        source.addExceptionParameterOffset("lambda", 0, .01, .02, .03)
        source.addExceptionParameterOffset("lambda", 1, .003, .01, .02)
        method = nv.NonbondedForce.Prolate if "prolate" in kind else nv.NonbondedForce.PME
        force = nv.NonbondedForce(source, nv.NonbondedForce.AutoBoxSafety, method)
        force.setDispersionCorrectionFollowsParameters(False)
        force.setLJCombiningRule(nv.NonbondedForce.Geometric)
        if kind == "ljmixed":
            dispersion = force.getLatticeSumInteraction(1)
            dispersion.splitting = nv.LatticeSumInteraction.Prolate
            force.setLatticeSumInteraction(1, dispersion)
        if kind in ("native-cutoff", "dispersion-only"):
            force.removeLatticeSumInteraction(0)
        if kind.startswith("fused"):
            definition = _custom_definition()
            if ljpme:
                definition.setUseSwitchingFunction(False)
                definition.setUseLongRangeCorrection(False)
            force.setCutoffInteraction(definition, not ljpme)
    if force.hasCutoffExpression():
        tail = _custom_definition()
        tail.addGlobalParameter("tail_only", .3)
        tail.setEnergyFunction("tail_only*k1*k2/(1+r^6)")
        tail.addInteractionGroup({0}, {1, 2})
        force.addExpressionTailCorrection(tail)
    force.setName("serialized " + kind)
    force.setForceGroup(3)
    return force


def _system(kind, periodic=True):
    system = mm.System()
    for _ in range(3):
        system.addParticle(12)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3.2, 0, 0), mm.Vec3(0, 3.2, 0), mm.Vec3(0, 0, 3.2))
    system.addForce(_force(kind, periodic))
    system.addForce(mm.MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
    return system


def _lattice_definitions(force):
    return [(term.kernel.power, term.kernel.coupling, term.weightExpression, term.splitting)
            for term in (force.getLatticeSumInteraction(i)
                         for i in range(force.getNumLatticeSumInteractions()))]


def _interaction_definition(term):
    return (term.evaluator, term.combiningRule, term.useSwitching,
            term.switchingDistance, term.tail, term.expression)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_system_clone_preserves_complete_force_definition(kind, periodic):
    source = _system(kind, periodic)
    xml = mm.XmlSerializer.serialize(source)
    restored = mm.XmlSerializer.deserialize(xml)
    assert mm.XmlSerializer.serialize(restored) == xml
    assert mm.XmlSerializer.serialize(copy.deepcopy(source)) == xml
    original, force = source.getForce(0), restored.getForce(0)
    assert isinstance(force, nv.NonbondedForce)
    assert force.getNumParticles() == 3
    assert force.getName() == original.getName()
    assert force.getForceGroup() == 3
    assert force.getBoxSafetyFactor() == nv.NonbondedForce.AutoBoxSafety
    assert force.getPairMode() == original.getPairMode()
    assert force.getCutoffDistance() == original.getCutoffDistance()
    assert _lattice_definitions(force) == _lattice_definitions(original)
    assert _interaction_definition(force.getCutoffInteraction()) == _interaction_definition(original.getCutoffInteraction())
    assert force.getNumGlobalParameters() == original.getNumGlobalParameters()
    for index in range(force.getNumGlobalParameters()):
        assert force.getGlobalParameterName(index) == original.getGlobalParameterName(index)
        assert force.getGlobalParameterDefaultValue(index) == original.getGlobalParameterDefaultValue(index)
    for index in range(force.getNumParticles()):
        assert force.getParticleParameters(index) == original.getParticleParameters(index)
    if force.hasCutoffExpression():
        assert mm.XmlSerializer.serialize(force.getCutoffExpression()) == mm.XmlSerializer.serialize(original.getCutoffExpression())
        assert force.getNumExpressionTailCorrections() == 1
        assert force.getExpressionTailCorrection(0).getNumInteractionGroups() == 1
        assert mm.XmlSerializer.serialize(force.getExpressionTailCorrection(0)) == mm.XmlSerializer.serialize(original.getExpressionTailCorrection(0))
    if force.getPairMode() == nv.NonbondedForce.Standard:
        assert force.getUseDispersionCorrection() == original.getUseDispersionCorrection()
        assert force.getExceptionsUsePeriodicBoundaryConditions() == periodic
        assert force.getNumExceptions() == 2
        for index in range(force.getNumExceptions()):
            assert force.getExceptionParameters(index) == original.getExceptionParameters(index)
        assert force.getNumParticleParameterOffsets() == 1
        assert force.getParticleParameterOffset(0) == original.getParticleParameterOffset(0)
        assert force.getNumExceptionParameterOffsets() == 2
        for index in range(force.getNumExceptionParameterOffsets()):
            assert force.getExceptionParameterOffset(index) == original.getExceptionParameterOffset(index)
        assert not force.getDispersionCorrectionFollowsParameters()
    else:
        assert force.getNumExclusions() == 2
        for index in range(force.getNumExclusions()):
            assert force.getExclusionParticles(index) == original.getExclusionParticles(index)


def _compare_contexts(contexts):
    for energy, forces in ((True, False), (False, True), (True, True)):
        observations = [c.getState(getEnergy=energy, getForces=forces) for c in contexts]
        if energy:
            np.testing.assert_allclose(
                observations[0].getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                observations[1].getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                rtol=1e-6, atol=1e-5)
        if forces:
            np.testing.assert_allclose(
                observations[0].getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer),
                observations[1].getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer),
                rtol=1e-6, atol=1e-5)


def _update_parameters(force, context):
    if force.getPairMode() == nv.NonbondedForce.Standard:
        force.setParticleParameters(2, -.15, .32, .24)
        force.setExceptionParameters(0, 0, 1, -.02, .27, .04)
        force.setParticleParameterOffset(0, "lambda", 2, .03, .01, .06)
        force.setExceptionParameterOffset(1, "lambda", 1, .005, .02, .01)
        force.setDispersionCorrectionFollowsParameters(True)
    if force.hasCutoffExpression():
        definition = force.getCutoffExpression()
        definition.setParticleParameters(2, [.35])
        preserve = force.getPairMode() == nv.NonbondedForce.Standard and force.getUseDispersionCorrection()
        force.setCutoffInteraction(definition, preserve)
        tail = force.getExpressionTailCorrection(0)
        tail.setParticleParameters(1, [.25])
        force.setExpressionTailCorrection(0, tail)
    force.updateParametersInContext(context)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("periodic", [False, True])
def test_roundtrip_preserves_live_energies_forces_and_molecule_groups(kind, periodic):
    source = _system(kind, periodic)
    restored = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(source))
    contexts, integrators = [], []
    try:
        for system in (source, restored):
            integrator = mm.LangevinMiddleIntegrator(300, 1, .001)
            integrators.append(integrator)
            context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                                 {"Precision": "mixed", "DeterministicForces": "true"})
            contexts.append(context)
            # The first exception crosses a box boundary. Its raw displacement
            # and minimum image give different physical replacement energies.
            context.setPositions([[.2, .2, .2], [3.1, .2, .2], [.7, .5, .3]])
        assert contexts[0].getMolecules() == contexts[1].getMolecules()
        for value in (0., .25, 1., .5, 0.):
            for context in contexts:
                context.setParameter("lambda", value)
                if source.getForce(0).hasCutoffExpression():
                    context.setParameter("tail_only", 1-value)
            _compare_contexts(contexts)
        for system, context in zip((source, restored), contexts):
            _update_parameters(system.getForce(0), context)
        for value in (.7, 0.):
            for context in contexts:
                context.setParameter("lambda", value)
            _compare_contexts(contexts)
        assert mm.XmlSerializer.serialize(source) == mm.XmlSerializer.serialize(restored)
    finally:
        contexts.clear()
        integrators.clear()


@pytest.mark.parametrize("kind", ["fused-ljpme", "custom-mixed"])
def test_fresh_process_can_restore_after_importing_plugin(tmp_path, kind):
    xml = mm.XmlSerializer.serialize(_system(kind, periodic=False))
    path = tmp_path / "system.xml"
    path.write_text(xml)
    code = """
import sys
from pathlib import Path
import openmm as mm
import nv_openmm_plugin as nv
xml = Path(sys.argv[1]).read_text()
system = mm.XmlSerializer.deserialize(xml)
assert isinstance(system.getForce(0), nv.NonbondedForce)
assert mm.XmlSerializer.serialize(system) == xml
"""
    # conftest adds the source package to pytest's sys.path, which a fresh
    # interpreter does not inherit. Load the same package in the child even
    # when it is not installed and the caller has no plugin PYTHONPATH.
    env = os.environ.copy()
    python_dir = str(Path(nv.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (python_dir, env.get("PYTHONPATH"))))
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True,
                            text=True, env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("attribute,value", [("version", 1), ("version", 999), ("mode", 9)])
def test_unknown_version_or_mode_is_rejected(attribute, value):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    node.set(attribute, str(value))
    with pytest.raises(Exception, match="Unsupported cuPO force serialization version|Invalid serialized cuPO force mode"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


@pytest.mark.parametrize("attribute", ["evaluator", "combining", "tail"])
def test_invalid_interaction_enum_is_rejected(attribute):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    node.find("Cutoff").set(attribute, "99")
    with pytest.raises(Exception, match="Invalid cuPO interaction enum"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


def test_schema_stores_all_lattice_terms_and_exception_policy():
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("ljmixed", periodic=False)))
    assert node.get("version") == "2"
    assert node.get("exceptionsUsePeriodicBoundaryConditions") == "0"
    assert "lattice" not in node.attrib and "splitting" not in node.attrib
    terms = node.findall("LatticeSums/LatticeSum")
    assert [(int(term.get("power")), int(term.get("splitting"))) for term in terms] == [(1, 0), (6, 1)]
    assert [term.get("weightExpression") for term in terms] == ["q", "2*sqrt(epsilon)*sigma^3"]
    assert [float(term.get("coupling")) for term in terms] == [nv.CoulombRadialKernel().coupling, -1.]


@pytest.mark.parametrize("attribute,value,message", [
    ("power", "8", "inverse powers"),
    ("coupling", "-1", "coupling"),
    ("coupling", "0", "coupling"),
    ("coupling", "1e100", "coupling"),
    ("splitting", "9", "Invalid serialized cuPO lattice interaction"),
    ("weightExpression", " \t", "Invalid serialized cuPO lattice interaction"),
])
def test_invalid_lattice_definition_is_rejected(attribute, value, message):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    node.find("LatticeSums/LatticeSum").set(attribute, value)
    with pytest.raises(Exception, match=message):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


def test_duplicate_lattice_power_is_rejected():
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    terms = node.find("LatticeSums")
    terms.append(copy.deepcopy(terms[0]))
    with pytest.raises(Exception, match="Duplicate lattice power"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


@pytest.mark.parametrize("missing", ["exceptionsUsePeriodicBoundaryConditions", "LatticeSums"])
def test_missing_required_v2_state_is_rejected(missing):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    if missing in node.attrib:
        del node.attrib[missing]
    else:
        node.remove(node.find(missing))
    with pytest.raises(Exception, match=missing):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


@pytest.mark.parametrize("value", [-1, 2])
def test_invalid_exception_periodicity_is_rejected(value):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    node.set("exceptionsUsePeriodicBoundaryConditions", str(value))
    with pytest.raises(Exception, match="Invalid serialized cuPO exception periodicity"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


@pytest.mark.parametrize("collection", ["ParticleOffsets", "ExceptionOffsets"])
@pytest.mark.parametrize("attribute", ["g", "index"])
@pytest.mark.parametrize("value", [-1, 99])
def test_invalid_offset_references_are_rejected(collection, attribute, value):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force("pme")))
    node.find(collection+"/Offset").set(attribute, str(value))
    with pytest.raises(Exception, match="Invalid (particle|exception) offset index"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


@pytest.mark.parametrize("kind,path", [
    ("custom", "CustomDefinition"),
    ("fused-pme", "CutoffExpression"),
    ("custom", "Tails/Tail"),
])
def test_wrong_nested_definition_type_is_rejected(kind, path):
    node = ET.fromstring(mm.XmlSerializer.serialize(_force(kind)))
    node.find(path).set("type", "NonbondedForce")
    with pytest.raises(Exception, match="Invalid custom definition type"):
        mm.XmlSerializer.deserialize(ET.tostring(node, encoding="unicode"))


@pytest.mark.parametrize("kind", ["custom", "custom-mixed"])
def test_changed_custom_cutoff_roundtrips(kind):
    force = _force(kind)
    force.setCutoffDistance(1.1)
    xml = mm.XmlSerializer.serialize(force)
    owner = mm.XmlSerializer.deserialize(xml)
    restored = nv.NonbondedForce.cast(owner)
    assert restored.getCutoffDistance() == 1.1
    assert float(ET.fromstring(xml).get("cutoff")) == 1.1
    assert mm.XmlSerializer.serialize(restored) == xml


def test_native_lj_roundtrip_after_removing_cutoff_expression():
    force = _force("pme")
    native = force.getCutoffInteraction()
    definition = _custom_definition()
    force.setCutoffInteraction(definition, True)
    force.setCutoffInteraction(interaction=native)
    xml = mm.XmlSerializer.serialize(force)
    owner = mm.XmlSerializer.deserialize(xml)
    restored = nv.NonbondedForce.cast(owner)
    assert not restored.hasCutoffExpression()
    assert mm.XmlSerializer.serialize(restored) == xml
    assert _interaction_definition(restored.getCutoffInteraction()) == _interaction_definition(native)
