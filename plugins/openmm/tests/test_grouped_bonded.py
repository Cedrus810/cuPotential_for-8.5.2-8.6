# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin routing, energy outputs, updates and live atom reorder for MIT bonded."""
import os
import numpy as np
import pytest
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv

MASK=(1<<0)|(1<<1)|(1<<31)


def system(periodic=False,reverse=False,multiple=False):
    s=mm.System()
    s.setDefaultPeriodicBoxVectors(mm.Vec3(4.1,0,0),mm.Vec3(0,4.2,0),mm.Vec3(0,0,4.3))
    bond,angle,torsion=mm.HarmonicBondForce(),mm.HarmonicAngleForce(),mm.PeriodicTorsionForce()
    for f,g in zip([bond,angle,torsion],[0,1,31]):
        f.setForceGroup(g); f.setUsesPeriodicBoundaryConditions(periodic)
    nb=mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.PME)
    nb.setCutoffDistance(.9)
    nb.setForceGroup(30)
    # An inert stock cutoff interaction makes OpenMM perform atom reordering.
    reorder=mm.CustomNonbondedForce("0")
    reorder.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    reorder.setCutoffDistance(.9)
    reorder.setForceGroup(29)
    xyz=[]
    for m in range(12):
        start=4*m
        origin=np.array([.5+(m%3),.5+(m//3)%2,.5+m//6])
        xyz.extend(origin+np.array([[0,0,0],[.15,0,0],[.23,.13,0],[.32,.16,.12]]))
        for _ in range(4):
            s.addParticle(12.); nb.addParticle(0.,.25,0.); reorder.addParticle([])
        for j in range(3): bond.addBond(start+j,start+j+1,.15,1000.+100*(m%2))
        angle.addAngle(start,start+1,start+2,1.8,150.)
        angle.addAngle(start+1,start+2,start+3,1.7,130.)
        torsion.addTorsion(start,start+1,start+2,start+3,3,.2,2.5)
    forces=[bond,angle,torsion]
    if multiple:
        # Separate objects of each kind exercise entry-to-batch bookkeeping.
        # Bonds share a group with the first object; the others use new groups.
        # Their combined 132 terms also span more than one grouped GPU batch.
        extra_bond=mm.HarmonicBondForce()
        extra_angle=mm.HarmonicAngleForce()
        extra_torsion=mm.PeriodicTorsionForce()
        for force,group in zip([extra_bond,extra_angle,extra_torsion],[0,4,5]):
            force.setForceGroup(group)
            force.setUsesPeriodicBoundaryConditions(periodic)
        for m in range(12):
            start=4*m
            extra_bond.addBond(start,start+2,.24,210.+m)
            extra_bond.addBond(start+1,start+3,.23,190.+m)
            extra_angle.addAngle(start,start+1,start+3,1.9,75.+m)
            extra_torsion.addTorsion(start,start+1,start+2,start+3,1,-.6,1.25)
            extra_torsion.addTorsion(start,start+1,start+2,start+3,5,.7,.75)
        # Interleave kinds instead of making force-object order match kind order.
        forces.extend([extra_torsion,extra_bond,extra_angle])
    if reverse:
        for force in forces:
            if isinstance(force,mm.HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    a,b,r,k=force.getBondParameters(i)
                    force.setBondParameters(i,b,a,r,k)
            elif isinstance(force,mm.HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    a,b,c,theta,k=force.getAngleParameters(i)
                    force.setAngleParameters(i,c,b,a,theta,k)
            else:
                for i in range(force.getNumTorsions()):
                    a,b,c,d,n,phase,k=force.getTorsionParameters(i)
                    force.setTorsionParameters(i,d,c,b,a,n,phase,k)
    for force in forces: s.addForce(force)
    s.addForce(nv.NonbondedForce(nb,1.))
    s.addForce(reorder)
    return s,np.array(xyz),forces


def make(grouped,precision="mixed",periodic=False,graph=True,reverse=False,multiple=False):
    os.environ["CUPO_PLUGIN_BONDED"]="1" if grouped else "0"
    os.environ["CUPO_PLUGIN_CUDA_GRAPH"]="1" if graph else "0"
    s,xyz,forces=system(periodic,reverse,multiple)
    integ=mm.VerletIntegrator(.0002)
    ctx=mm.Context(s,integ,mm.Platform.getPlatformByName("CUDA"),{"Precision":precision})
    ctx.setPositions(xyz)
    return s,integ,ctx,forces


def outputs(ctx,mask=MASK):
    selected={i for i in range(32) if mask & (1<<i)}
    state=ctx.getState(getEnergy=True,getForces=True,groups=selected if selected else 0)
    return (state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer),
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def assert_same_bits(actual,expected):
    # OpenMM exposes its fixed-point forces as exactly representable doubles.
    # Comparing the bytes catches even a single force-buffer bit or energy ULP.
    assert actual[0].tobytes()==expected[0].tobytes()
    assert np.float64(actual[1]).tobytes()==np.float64(expected[1]).tobytes()


def update_last_term(force,ctx):
    if isinstance(force,mm.HarmonicBondForce):
        index=force.getNumBonds()-1
        a,b,r,k=force.getBondParameters(index)
        force.setBondParameters(index,a,b,r*.97,k*1.1)
    elif isinstance(force,mm.HarmonicAngleForce):
        index=force.getNumAngles()-1
        a,b,c,theta,k=force.getAngleParameters(index)
        force.setAngleParameters(index,a,b,c,theta*.97,k*1.1)
    else:
        index=force.getNumTorsions()-1
        a,b,c,d,n,phase,k=force.getTorsionParameters(index)
        force.setTorsionParameters(index,a,b,c,d,n+1,phase+.2*unit.radian,k*1.1)
    force.updateParametersInContext(ctx)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE","0")
    monkeypatch.setenv("CUPO_PLUGIN_BONDED","1")
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH","1")


@pytest.mark.parametrize("precision",["single","mixed"])
@pytest.mark.parametrize("periodic",[False,True])
@pytest.mark.parametrize("graph",[False,True])
def test_bonded_plugin_outputs_updates(precision,periodic,graph):
    control=make(False,precision,periodic,graph)
    candidate=make(True,precision,periodic,graph)
    for iteration in range(3):
        for mask in [1,2,1<<31,MASK,0]:
            a,ea=outputs(control[2],mask)
            b,eb=outputs(candidate[2],mask)
            np.testing.assert_allclose(b,a,rtol=2e-5,atol=.003)
            np.testing.assert_allclose(eb,ea,rtol=3e-6,atol=.002)
        for _,_,ctx,forces in [control,candidate]:
            only=ctx.getState(getEnergy=True,groups={0,1,31}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            np.testing.assert_allclose(only,outputs(ctx)[1],rtol=1e-7,atol=1e-5)
            a,b,r,k=forces[0].getBondParameters(0)
            forces[0].setBondParameters(0,a,b,r,k*1.1)
            forces[0].updateParametersInContext(ctx)
            a,b,c,theta,k=forces[1].getAngleParameters(0)
            forces[1].setAngleParameters(0,a,b,c,theta,k*1.1)
            forces[1].updateParametersInContext(ctx)
            a,b,c,d,n,phase,k=forces[2].getTorsionParameters(0)
            forces[2].setTorsionParameters(0,a,b,c,d,n,phase,k*1.1)
            forces[2].updateParametersInContext(ctx)
    # Existing topology changes remain an explicit reinitialization requirement.
    if not periodic:
        a,b,r,k=candidate[3][0].getBondParameters(0)
        candidate[3][0].setBondParameters(0,a,b+1,r,k)
        with pytest.raises(Exception): candidate[3][0].updateParametersInContext(candidate[2])


@pytest.mark.parametrize("precision",["single","mixed"])
@pytest.mark.parametrize("graph",[False,True])
def test_reversed_interactions_are_bitwise_identical(precision,graph,monkeypatch,capfd):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE","1")
    forward=make(True,precision,graph=graph,multiple=True)
    reverse=make(True,precision,graph=graph,reverse=True,multiple=True)
    masks=[1,2,1<<31,1<<4,1<<5,MASK|(1<<4)|(1<<5),0xffffffff]
    # Capture requires 50 eager evaluations and two visits to each output mode.
    # Warm energy+force and force-only graphs before checking replay and updates.
    for _,integ,ctx,_ in [forward,reverse]:
        for _ in range(53):
            outputs(ctx,0xffffffff)
        integ.step(3)
    trace=capfd.readouterr().err
    captures=[line for line in trace.splitlines() if "graph capture #" in line]
    for marker in ["(energy=1 forces=1)","(energy=0 forces=1)"]:
        assert sum(marker in line for line in captures)==(2 if graph else 0),trace
    for iteration in range(3):
        for mask in masks:
            expected=outputs(forward[2],mask)
            for _ in range(2):
                assert_same_bits(outputs(reverse[2],mask),expected)
        if iteration<2:
            for _,integ,ctx,forces in [forward,reverse]:
                for force in forces:
                    update_last_term(force,ctx)
                integ.step(3)
    assert "graph capture #" not in capfd.readouterr().err


@pytest.mark.parametrize("precision",["single","mixed"])
def test_multiple_force_objects_update_only_their_entry(precision):
    control=make(False,precision,multiple=True)
    candidate=make(True,precision,multiple=True)
    masks=[1,2,1<<31,1<<4,1<<5,MASK|(1<<4)|(1<<5),0]

    def compare():
        result={}
        for mask in masks:
            a,ea=outputs(control[2],mask)
            b,eb=outputs(candidate[2],mask)
            result[mask]=(b,eb)
            np.testing.assert_allclose(b,a,rtol=2e-5,atol=.003)
            np.testing.assert_allclose(eb,ea,rtol=3e-6,atol=.002)
        return result

    before=compare()
    for index in range(len(candidate[3])):
        force=candidate[3][index]
        group=1<<force.getForceGroup()
        for _,_,ctx,forces in [control,candidate]:
            update_last_term(forces[index],ctx)
        after=compare()
        # A nonzero change proves the selected entry was uploaded; unrelated
        # force groups must retain their exact outputs after that partial update.
        assert after[group][1]!=before[group][1]
        for mask in masks:
            if not mask&group:
                assert_same_bits(after[mask],before[mask])
        before=after


def test_live_reordering_preserves_bonded_mapping(monkeypatch,capfd):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE","1")
    candidate=make(True)
    s,integ,ctx,_=candidate
    ctx.setVelocitiesToTemperature(30,171)
    # First force evaluations build all plans and then exercise their updates
    # after actual OpenMM molecule sorting, including a parameter-class change.
    for _ in range(3):
        integ.step(260)
        xyz=ctx.getState(getPositions=True).getPositions(asNumpy=True)
        # Normalize the same positions in both contexts after any wrapping.
        control=make(False)
        for c in [ctx,control[2]]:
            c.setPositions(xyz); c.setPositions(xyz)
        a,ea=outputs(control[2]); b,eb=outputs(ctx)
        np.testing.assert_allclose(b,a,rtol=3e-5,atol=.005)
        np.testing.assert_allclose(eb,ea,rtol=3e-6,atol=.003)
        del control
    assert "atom reorder #" in capfd.readouterr().err


def test_stock_context_is_unchanged_without_cupo_force():
    s=mm.System(); s.addParticle(12.); s.addParticle(12.)
    f=mm.HarmonicBondForce(); f.addBond(0,1,.15,1000.); s.addForce(f)
    integ=mm.VerletIntegrator(.001)
    ctx=mm.Context(s,integ,mm.Platform.getPlatformByName("CUDA"),{"Precision":"double"})
    ctx.setPositions([[0,0,0],[.16,0,0]])
    assert ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)==pytest.approx(.05,abs=1e-6)
