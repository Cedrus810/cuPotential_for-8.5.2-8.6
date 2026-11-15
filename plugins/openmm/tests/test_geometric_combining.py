# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometric LJ mixing with unchanged particle and explicit-exception formats."""

import gc

import numpy as np
import openmm as mm
import openmm.unit as unit
import pytest


CUTOFF = 0.9
BOX = 3.6
# The builtin direct-space accumulator rounds each force component per pair in
# Coulomb-reduced units with 21 fractional bits, then multiplies by this C.
FORCE_QUANTUM = float(np.float32(138.93545764498)) * 2.0 ** -21


def _fixture(charged=False):
    rng = np.random.default_rng(4913)
    positions = np.array([[x, y, z] for x in (0.2, 0.92, 1.64, 2.36)
                          for y in (0.2, 0.92, 1.64, 2.36)
                          for z in (0.2, 0.92, 1.64, 2.36)])
    positions += rng.uniform(-0.035, 0.035, positions.shape)
    positions[0] = (0.2, 0.2, 0.2)
    positions[1] = (3.4, 0.2, 0.2)  # Short periodic exception.
    positions[2] = (1.7, 0.5, 0.5)  # Exception beyond the ordinary cutoff.
    particles = np.column_stack((
        rng.uniform(-0.3, 0.3, len(positions)) if charged else np.zeros(len(positions)),
        rng.uniform(0.21, 0.41, len(positions)),
        rng.uniform(0.15, 0.8, len(positions)),
    ))
    particles[6, 1:] = (2.0, 0.0)   # Zero epsilon must suppress even a large sigma.
    particles[7, 1:] = (0.0, 0.6)   # Zero geometric sigma also has no ordinary LJ.
    exceptions = [
        [0, 1, -0.021 if charged else 0.0, 0.31, 0.2],
        [0, 2, 0.014 if charged else 0.0, 1.0, 0.1],
        [4, 5, 0.0, 0.3, 0.0],
    ]
    return positions, particles, exceptions


def _source(method, particles, exceptions, tail):
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(BOX, 0, 0), mm.Vec3(0, BOX, 0), mm.Vec3(0, 0, BOX))
    source = mm.NonbondedForce()
    source.setNonbondedMethod(mm.NonbondedForce.PME)
    source.setCutoffDistance(CUTOFF)
    source.setUseDispersionCorrection(tail)
    source.setExceptionsUsePeriodicBoundaryConditions(True)
    for q, sigma, epsilon in particles:
        system.addParticle(18.0)
        source.addParticle(q, sigma, epsilon)
    for i, j, q, sigma, epsilon in exceptions:
        source.addException(i, j, q, sigma, epsilon)
    return system, source


def _context(method, particles, exceptions, tail, rule="Geometric", integrator=None):
    import nv_openmm_plugin as nv

    system, source = _source(method, particles, exceptions, tail)
    force = nv.NonbondedForce(source, 1.1, getattr(nv.NonbondedForce, method))
    if rule is not None:
        force.setLJCombiningRule(getattr(nv.NonbondedForce, rule))
    system.addForce(force)
    if integrator is None:
        integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed", "DeterministicForces": "true"})
    return context, integrator, force


def _snapshot(context, energy=True, forces=True):
    state = context.getState(getEnergy=energy, getForces=forces)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if energy else None,
        np.asarray(state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer)).copy() if forces else None,
    )


def _evaluate(method, positions, particles, exceptions, tail, rule="Geometric"):
    context, integrator, force = _context(method, particles, exceptions, tail, rule)
    try:
        context.setPositions(positions)
        return _snapshot(context)
    finally:
        del context, integrator, force
        gc.collect()


def _lj_reference(positions, particles, exceptions, tail, box=BOX, geometric=True):
    """Continuous FP64 sum plus bounds for pair rounding and FP32 arithmetic."""
    energy = 0.0
    forces = np.zeros_like(positions)
    absolute_pair_forces = np.zeros_like(positions)
    incidence = np.zeros(len(particles), dtype=int)
    overrides = {(int(i), int(j)): (charge, sigma, epsilon)
                 for i, j, charge, sigma, epsilon in exceptions}
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            delta = positions[i] - positions[j]
            delta -= box * np.rint(delta / box)
            r2 = np.dot(delta, delta)
            original_charge_product = particles[i, 0] * particles[j, 0]
            if (i, j) in overrides:
                exception_charge, sigma, epsilon = overrides[i, j]
                charge_active = original_charge_product != 0 or exception_charge != 0
            else:
                if r2 >= CUTOFF ** 2:
                    continue
                sigma = (np.sqrt(particles[i, 1] * particles[j, 1]) if geometric
                         else (particles[i, 1] + particles[j, 1]) / 2)
                epsilon = np.sqrt(particles[i, 2] * particles[j, 2])
                charge_active = original_charge_product != 0
            sr6 = (sigma * sigma / r2) ** 3
            energy += 4 * epsilon * (sr6 * sr6 - sr6)
            pair_force = 24 * epsilon * (2 * sr6 * sr6 - sr6) * delta / r2
            forces[i] += pair_force
            forces[j] -= pair_force
            absolute_pair_forces[i] += np.abs(pair_force)
            absolute_pair_forces[j] += np.abs(pair_force)
            if charge_active or (epsilon != 0 and sigma != 0):
                incidence[i] += 1
                incidence[j] += 1
    if tail:
        # OpenMM averages all unordered type pairs INCLUDING self pairs. Using
        # particle pairs here avoids sharing the implementation's type grouping.
        tail_sum = 0.0
        for i in range(n):
            for j in range(i, n):
                sigma = (np.sqrt(particles[i, 1] * particles[j, 1]) if geometric
                         else (particles[i, 1] + particles[j, 1]) / 2)
                epsilon = np.sqrt(particles[i, 2] * particles[j, 2])
                tail_sum += epsilon * (sigma ** 12 / (9 * CUTOFF ** 9)
                                       - sigma ** 6 / (3 * CUTOFF ** 3))
        energy += 8 * np.pi * n * n * tail_sum / (n * (n + 1) / 2) / box ** 3
    # Each surviving pair component can round by at most half a quantum. The
    # ordinary term of an exception is undone bitwise, so count its correction
    # once, even outside the cutoff. Zero charge/LJ pairs add exactly zero.
    rounding_bound = 0.5 * FORCE_QUANTUM * incidence[:, None]
    return energy, forces, rounding_bound, absolute_pair_forces


def _assert_force_close(candidate, reference, rounding_bound, arithmetic_scale):
    # Sum absolute pair magnitudes instead of |net force|: cancellation must not
    # shrink the allowance for FP32 coordinate/parameter and pair arithmetic.
    arithmetic_bound = 64 * np.finfo(np.float32).eps * arithmetic_scale + 1e-8
    error = np.abs(candidate - reference)
    bound = rounding_bound + arithmetic_bound
    assert np.all(error <= bound), (
        f"force error exceeds pair-rounding + FP32 bound; "
        f"maximum excess={np.max(error - bound):.3g}, "
        f"error L2={np.linalg.norm(error):.3g}, bound L2={np.linalg.norm(bound):.3g}")


def _assert_close(candidate, reference):
    energy, forces = candidate
    expected_energy, expected_forces, rounding_bound, arithmetic_scale = reference
    if energy is not None:
        np.testing.assert_allclose(energy, expected_energy, rtol=2e-5, atol=3e-5)
    if forces is not None:
        _assert_force_close(forces, expected_forces, rounding_bound, arithmetic_scale)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("tail", [False, True])
def test_geometric_lj_matches_pair_reference(plugin_dir, method, tail):
    positions, particles, exceptions = _fixture()
    candidate = _evaluate(method, positions, particles, exceptions, tail)
    reference = _lj_reference(positions, particles, exceptions, tail)
    _assert_close(candidate, reference)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_geometric_mixing_preserves_electrostatics(plugin_dir, method):
    """Subtract two models with identical charges to isolate the changed LJ."""
    positions, particles, exceptions = _fixture(charged=True)
    geometric = _evaluate(method, positions, particles, exceptions, False)
    arithmetic = _evaluate(method, positions, particles, exceptions, False,
                           rule="LorentzBerthelot")
    geo_lj = _lj_reference(positions, particles, exceptions, False)
    lb_lj = _lj_reference(positions, particles, exceptions, False, geometric=False)
    delta = (geometric[0] - arithmetic[0], geometric[1] - arithmetic[1])
    expected = (geo_lj[0] - lb_lj[0], geo_lj[1] - lb_lj[1])
    # Two independently rounded direct-space evaluations contribute to this
    # difference. Include their larger Coulomb totals in the arithmetic scale;
    # the common reciprocal force is identical in both evaluations.
    np.testing.assert_allclose(delta[0], expected[0], rtol=3e-5, atol=2e-4)
    arithmetic_scale = (geo_lj[3] + lb_lj[3]
                        + np.abs(geometric[1]) + np.abs(arithmetic[1]))
    _assert_force_close(delta[1], expected[1], geo_lj[2] + lb_lj[2], arithmetic_scale)


def test_geometric_graph_modes_box_and_parameter_updates(plugin_dir, monkeypatch, capfd):
    import nv_openmm_plugin as nv

    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    positions, particles, exceptions = _fixture()
    # getState(getEnergy=True) also computes kinetic energy. Verlet's kinetic
    # energy needs forces for its velocity time shift, defeating energy-only
    # coverage. This force-free expression makes all three graph slots reachable.
    integrator = mm.CustomIntegrator(0.001)
    integrator.setKineticEnergyExpression("0.5*m*v*v")
    context, integrator, force = _context(
        "Prolate", particles, exceptions, True, integrator=integrator)
    try:
        context.setPositions(positions)
        reference = _lj_reference(positions, particles, exceptions, True)
        capfd.readouterr()
        for energy, forces in ((False, True), (True, False), (True, True)):
            for _ in range(56 if not energy else 4):
                sample = _snapshot(context, energy, forces)
            _assert_close(sample, reference)
        trace = capfd.readouterr().err
        for mode in ("energy=0 forces=1", "energy=1 forces=0", "energy=1 forces=1"):
            assert mode in trace, trace

        # Both direct geometry and the volume-dependent tail must see the new box.
        box = BOX * 1.04
        positions = positions * 1.04 + np.array([0.03, -0.02, 0.01])
        context.setPeriodicBoxVectors(
            mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box))
        context.setPositions(positions)
        _assert_close(_snapshot(context),
                      _lj_reference(positions, particles, exceptions, True, box))

        particles[0, 1:] *= (1.13, 1.4)
        force.setParticleParameters(0, *particles[0])
        # Turn a previously inactive particle on, requiring fresh type/tail sums.
        particles[6, 1:] = (0.34, 0.25)
        force.setParticleParameters(6, *particles[6])
        exceptions[1][3:] = [0.91, 0.17]
        force.setExceptionParameters(1, *exceptions[1])
        force.updateParametersInContext(context)
        expected = _lj_reference(positions, particles, exceptions, True, box)
        for energy, forces in ((False, True), (True, False), (True, True)):
            _assert_close(_snapshot(context, energy, forces), expected)

        # A rule change would require another kernel specialization and graph.
        force.setLJCombiningRule(nv.NonbondedForce.LorentzBerthelot)
        with pytest.raises(Exception, match="(?i)combining.*reinitial|reinitial.*combining"):
            force.updateParametersInContext(context)
        force.setLJCombiningRule(nv.NonbondedForce.Geometric)
        _assert_close(_snapshot(context), expected)
    finally:
        del context, integrator, force
        gc.collect()


def test_lj_combining_rule_conversion_and_validation(plugin_dir):
    import nv_openmm_plugin as nv

    _, particles, exceptions = _fixture()
    _, source = _source("PME", particles, exceptions, True)
    assert nv.NonbondedForce().getLJCombiningRule() == nv.NonbondedForce.LorentzBerthelot
    converted = nv.NonbondedForce(source)
    assert converted.getLJCombiningRule() == nv.NonbondedForce.LorentzBerthelot
    converted.setLJCombiningRule(nv.NonbondedForce.Geometric)
    assert converted.getLJCombiningRule() == nv.NonbondedForce.Geometric
    for i in range(source.getNumParticles()):
        np.testing.assert_array_equal(converted.getParticleParameters(i), particles[i])
    for i in range(source.getNumExceptions()):
        np.testing.assert_array_equal(converted.getExceptionParameters(i), exceptions[i])
    with pytest.raises(Exception, match="combining rule"):
        converted.setLJCombiningRule(999)
    assert converted.getLJCombiningRule() == nv.NonbondedForce.Geometric


def test_combining_rule_change_before_first_evaluation_requires_reinitialize(plugin_dir):
    import nv_openmm_plugin as nv

    positions, particles, exceptions = _fixture()
    context, integrator, force = _context(
        "Prolate", particles, exceptions, True, rule=None)
    try:
        assert force.getLJCombiningRule() == nv.NonbondedForce.LorentzBerthelot
        context.setPositions(positions)
        force.setLJCombiningRule(nv.NonbondedForce.Geometric)
        # No force/energy evaluation has occurred: lazy device setup must not
        # bypass the same structural contract enforced after graph capture.
        with pytest.raises(Exception, match="(?i)combining.*reinitial|reinitial.*combining"):
            force.updateParametersInContext(context)
        _assert_close(_snapshot(context),
                      _lj_reference(positions, particles, exceptions, True, geometric=False))

        context.reinitialize(preserveState=True)
        _assert_close(_snapshot(context),
                      _lj_reference(positions, particles, exceptions, True))
        restored = context.getState(getPositions=True).getPositions(asNumpy=True)
        np.testing.assert_allclose(restored.value_in_unit(unit.nanometer), positions,
                                   rtol=0, atol=1e-12)
        assert nv.NonbondedForce().getLJCombiningRule() == nv.NonbondedForce.LorentzBerthelot
    finally:
        del context, integrator, force
        gc.collect()
