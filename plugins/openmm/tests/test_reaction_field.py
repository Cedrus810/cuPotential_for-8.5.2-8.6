# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reaction field is a Custom expression, not a Standard splitting method."""

import gc

import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest


CUTOFF = 0.9
COULOMB = 138.93545764498
FORCE_QUANTUM = float(np.float32(COULOMB)) * 2.0 ** -21
EXPRESSION = (
    "coulomb*q1*q2*(1/r+krf*r^2-crf)"
    "+4*epsilon*((sigma/r)^12-(sigma/r)^6);"
    "sigma=(sigma1+sigma2)/2;epsilon=sqrt(epsilon1*epsilon2)"
)


def _coefficients(dielectric):
    # This form also handles the conductor limit without inf/inf.
    inverse = 1.0 / dielectric
    return ((1-inverse)/((2+inverse)*CUTOFF**3),
            3/((2+inverse)*CUTOFF))


def _system(dielectric=78.3, with_lj=True, custom=False):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    if custom:
        force = mm.CustomNonbondedForce(EXPRESSION)
        for name in ("q", "sigma", "epsilon"):
            force.addPerParticleParameter(name)
        krf, crf = _coefficients(dielectric)
        for name, value in (("coulomb", COULOMB), ("krf", krf), ("crf", crf)):
            force.addGlobalParameter(name, value)
        # The automatic Custom switch would multiply RF as well as LJ.
        force.setUseSwitchingFunction(False)
        force.setUseLongRangeCorrection(False)
    else:
        force = mm.NonbondedForce()
        # Stock OpenMM evaluates inf/inf for an infinite dielectric. A very large
        # finite value gives the same conductor coefficients in double precision.
        force.setReactionFieldDielectric(1e100 if np.isinf(dielectric) else dielectric)
        force.setUseDispersionCorrection(False)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(CUTOFF)
    force.setForceGroup(5)
    # Deliberately non-neutral: RF must not add self/background corrections.
    rng = np.random.default_rng(731)
    positions = np.array([[x, y, z] for x in (0.2, 1.1, 2.0, 2.9)
                          for y in (0.2, 1.1, 2.0, 2.9)
                          for z in (0.2, 1.1, 2.0, 2.9)], dtype=float)
    positions += rng.uniform(-0.045, 0.045, positions.shape)
    # Exclusions across the periodic boundary and outside the ordinary cutoff.
    positions[0] = (0.2, 0.2, 0.2)
    positions[1] = (3.8, 0.2, 0.2)
    positions[2] = (1.7, 0.2, 0.2)
    particles = []
    for i in range(len(positions)):
        system.addParticle(18)
        parameters = (float(rng.uniform(-0.4, 0.5)),
                      float(rng.uniform(0.2, 0.27)),
                      float(rng.uniform(0.15, 0.4)) if with_lj else 0.0)
        particles.append(parameters)
        if custom:
            force.addParticle(parameters)
        else:
            force.addParticle(*parameters)
    # Custom exclusions remove pairs completely. Nonzero Standard exceptions
    # would require a separate force for their replacement interactions.
    for i, j in ((0, 1), (0, 2), (4, 5)):
        if custom:
            force.addExclusion(i, j)
        else:
            force.addException(i, j, 0.0, 0.2, 0.0)
    system.addForce(force)
    return system, force, positions, np.array(particles)


def _snapshot(context, energy=True, forces=True, groups=1 << 5):
    state = context.getState(getEnergy=energy, getForces=forces, groups=groups)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if energy else None,
        np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)).copy() if forces else None,
    )


def _run(plugin, dielectric, with_lj, update=False):
    system, force, positions, particles = _system(dielectric, with_lj, custom=plugin)
    if plugin:
        import nv_openmm_plugin as nv
        force = nv.CustomNonbondedForce(force)
        system.removeForce(0)
        system.addForce(force)
    # A force-independent kinetic expression permits genuine energy-only calls.
    integrator = mm.CustomIntegrator(0.001)
    integrator.setKineticEnergyExpression("0.5*m*v*v")
    platform = mm.Platform.getPlatformByName("CUDA" if plugin else "Reference")
    context = mm.Context(system, integrator, platform,
                         {"Precision": "mixed"} if plugin else {})
    try:
        context.setPositions(positions)
        samples = [_snapshot(context, energy, forces)
                   for energy, forces in ((False, True), (True, False), (True, True))]
        if update:
            context.setPeriodicBoxVectors(
                mm.Vec3(4.2, 0, 0), mm.Vec3(0, 4.2, 0), mm.Vec3(0, 0, 4.2))
            context.setPositions(positions * 1.05 + np.array([0.03, -0.02, 0.01]))
            samples.append(_snapshot(context))
            samples.append(_snapshot(context, groups=1 << 4))

            # Particle 16 retains ordinary neighbors after the box expansion.
            particles[16] *= (1.3, 1.05, 0.8)
            if plugin:
                force.setParticleParameters(16, particles[16])
            else:
                force.setParticleParameters(16, *particles[16])
            force.updateParametersInContext(context)
            samples.append(_snapshot(context))

            # The Custom coefficients are live globals. Stock NonbondedForce
            # requires reinitialization when its dielectric changes.
            if plugin:
                for name, value in zip(("krf", "crf"), _coefficients(3.0)):
                    context.setParameter(name, value)
            else:
                force.setReactionFieldDielectric(3.0)
                context.reinitialize(preserveState=True)
            samples.append(_snapshot(context))
        return samples
    finally:
        del context, integrator
        gc.collect()


def _assert_close(candidate, reference):
    assert len(candidate) == len(reference)
    for (ce, cf), (re, rf) in zip(candidate, reference):
        if re is not None:
            np.testing.assert_allclose(ce, re, rtol=2e-5, atol=2e-4)
        if rf is not None:
            # The shared direct-space traversal rounds each pair's force to
            # Force32. This fixture has at most six ordinary partners per atom.
            np.testing.assert_allclose(cf, rf, rtol=4e-5,
                                       atol=3*FORCE_QUANTUM + 3e-5)


@pytest.mark.parametrize("dielectric", [1.0, 2.0, 78.3, float("inf")])
@pytest.mark.parametrize("with_lj", [False, True])
def test_custom_reaction_field_matches_stock(plugin_dir, dielectric, with_lj):
    reference = _run(False, dielectric, with_lj)
    candidate = _run(True, dielectric, with_lj)
    _assert_close(candidate, reference)


def test_custom_reaction_field_box_particle_and_global_updates(plugin_dir):
    reference = _run(False, 78.3, True, update=True)
    candidate = _run(True, 78.3, True, update=True)
    _assert_close(candidate, reference)
    # The excluded force group must contribute neither energy nor force.
    assert candidate[-3][0] == 0.0
    assert np.all(candidate[-3][1] == 0.0)
    assert candidate[-2][0] != candidate[-4][0]
    assert candidate[-1][0] != candidate[-2][0]


def test_standard_rejects_reaction_field_conversion(plugin_dir):
    import nv_openmm_plugin as nv
    # The System owns the OpenMM force after addForce; keep it alive here.
    owner, source, positions, particles = _system(3.0)
    # PME-only settings must not make a cutoff model eligible for conversion.
    source.setEwaldErrorTolerance(1e-7)
    source.setPMEParameters(2.0, 32, 32, 32)
    with pytest.raises(Exception, match="CustomNonbondedForce"):
        nv.NonbondedForce(source)
    for method in (nv.NonbondedForce.PME, nv.NonbondedForce.Prolate):
        with pytest.raises(Exception, match="CustomNonbondedForce"):
            nv.NonbondedForce(source, 1.0, method)
    assert not hasattr(nv.NonbondedForce, "ReactionField")
    assert not hasattr(nv.NonbondedForce, "getReactionFieldDielectric")
    # The retired numeric enum cannot bypass the new two-method contract.
    with pytest.raises(Exception, match="(?i)method"):
        nv.NonbondedForce(2, CUTOFF)
