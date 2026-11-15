# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenMM switching semantics for both ordinary-pair LJ combining rules."""

import gc

import numpy as np
import openmm as mm
import pytest

import test_geometric_combining as geo


SWITCH = 0.6


def _fixture(charged=False):
    positions, particles, exceptions = geo._fixture(charged)
    # An explicit LJ exception inside the switching interval complements the
    # fixture's exceptions below the interval and beyond the ordinary cutoff.
    exceptions.append([8, 9, -0.018 if charged else 0.0, 0.31, 0.2])
    return positions, particles, exceptions


def _switch(r, start):
    x = np.clip((r - start) / (geo.CUTOFF - start), 0.0, 1.0)
    # Independent FP64 polynomial and derivative, including both interval ends.
    value = 1 - 10*x**3 + 15*x**4 - 6*x**5
    derivative = -30*x*x*(1-x)**2 / (geo.CUTOFF - start)
    return value, derivative


def _reference(positions, particles, exceptions, tail, rule, box=geo.BOX,
               start=SWITCH, enabled=True):
    """Correct the continuous unswitched oracle, retaining its rounding bound."""
    geometric = rule == "Geometric"
    energy, forces, rounding, arithmetic = geo._lj_reference(
        positions, particles, exceptions, tail, box, geometric)
    if not enabled:
        return energy, forces, rounding, arithmetic
    excluded = {(int(i), int(j)) for i, j, *_ in exceptions}
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue  # Explicit exceptions are never switched.
            delta = positions[i] - positions[j]
            delta -= box * np.rint(delta / box)
            r = np.linalg.norm(delta)
            if not start < r < geo.CUTOFF:
                continue
            sigma = (np.sqrt(particles[i, 1]*particles[j, 1]) if geometric
                     else 0.5*(particles[i, 1]+particles[j, 1]))
            epsilon = np.sqrt(particles[i, 2]*particles[j, 2])
            sr6 = (sigma/r)**6
            potential = 4*epsilon*(sr6*sr6-sr6)
            force = 24*epsilon*(2*sr6*sr6-sr6)*delta/(r*r)
            switch, derivative = _switch(r, start)
            correction = (switch-1)*force - potential*derivative*delta/r
            energy += (switch-1)*potential
            forces[i] += correction
            forces[j] -= correction
            arithmetic[i] += np.abs(correction)
            arithmetic[j] += np.abs(correction)
    if tail:
        # Integrate the removed LJ in the switching interval directly; production
        # uses analytic moments. Particle pairs include self pairs, like OpenMM.
        nodes, weights = np.polynomial.legendre.leggauss(64)
        radii = start + (nodes+1)*(geo.CUTOFF-start)/2
        weights *= (geo.CUTOFF-start)/2
        removed = 1-_switch(radii, start)[0]
        lost_sum = 0.0
        for i in range(n):
            for j in range(i, n):
                sigma = (np.sqrt(particles[i, 1]*particles[j, 1]) if geometric
                         else 0.5*(particles[i, 1]+particles[j, 1]))
                epsilon = np.sqrt(particles[i, 2]*particles[j, 2])
                sr6 = (sigma/radii)**6
                lost_sum += np.dot(weights, removed*epsilon*(sr6*sr6-sr6)*radii**2)
        energy += 8*np.pi*n*n*lost_sum/(n*(n+1)/2)/box**3
    return energy, forces, rounding, arithmetic


def _context(method, particles, exceptions, tail, rule, start=SWITCH,
             enabled=True, integrator=None):
    import nv_openmm_plugin as nv

    system, source = geo._source(method, particles, exceptions, tail)
    source.setUseSwitchingFunction(enabled)
    source.setSwitchingDistance(start)
    force = nv.NonbondedForce(source, 1.1, getattr(nv.NonbondedForce, method))
    force.setLJCombiningRule(getattr(nv.NonbondedForce, rule))
    system.addForce(force)
    if integrator is None:
        integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed", "DeterministicForces": "true"})
    return context, integrator, force


def _evaluate(method, positions, particles, exceptions, tail, rule, **kwargs):
    context, integrator, force = _context(method, particles, exceptions, tail, rule,
                                         **kwargs)
    try:
        context.setPositions(positions)
        return geo._snapshot(context)
    finally:
        del context, integrator, force
        gc.collect()


@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("rule", ["LorentzBerthelot", "Geometric"])
def test_switched_lj_matches_pair_and_tail_reference(plugin_dir, method, rule):
    positions, particles, exceptions = _fixture()
    candidate = _evaluate(method, positions, particles, exceptions, True, rule)
    geo._assert_close(candidate, _reference(positions, particles, exceptions, True, rule))


@pytest.mark.parametrize("tail", [False, True])
def test_switched_lb_reference_matches_stock_openmm(plugin_dir, tail):
    """Confirm the independent oracle, including the switching tail convention."""
    positions, particles, exceptions = _fixture()
    system, source = geo._source("PME", particles, exceptions, tail)
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(SWITCH)
    system.addForce(source)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    try:
        context.setPositions(positions)
        energy, forces = geo._snapshot(context)
        reference = _reference(positions, particles, exceptions, tail, "LorentzBerthelot")
        np.testing.assert_allclose(energy, reference[0], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(forces, reference[1], rtol=1e-11, atol=1e-11)
    finally:
        del context, integrator
        gc.collect()


@pytest.mark.parametrize("rule", ["LorentzBerthelot", "Geometric"])
def test_switching_preserves_electrostatics(plugin_dir, rule):
    positions, particles, exceptions = _fixture(charged=True)
    switched = _evaluate("Prolate", positions, particles, exceptions, False, rule)
    plain = _evaluate("Prolate", positions, particles, exceptions, False, rule,
                      enabled=False)
    switched_lj = _reference(positions, particles, exceptions, False, rule)
    plain_lj = _reference(positions, particles, exceptions, False, rule, enabled=False)
    np.testing.assert_allclose(switched[0]-plain[0], switched_lj[0]-plain_lj[0],
                               rtol=3e-5, atol=2e-4)
    arithmetic = switched_lj[3]+plain_lj[3]+np.abs(switched[1])+np.abs(plain[1])
    geo._assert_force_close(switched[1]-plain[1], switched_lj[1]-plain_lj[1],
                            switched_lj[2]+plain_lj[2], arithmetic)


@pytest.mark.parametrize("rule", ["LorentzBerthelot", "Geometric"])
def test_switch_interval_boundaries_and_energy_derivative(plugin_dir, rule):
    positions, particles, _ = _fixture()
    particles[:] = 0
    particles[:2, 1:] = ((0.29, 0.4), (0.41, 0.7))
    context, integrator, force = _context("Prolate", particles, [], False, rule)
    try:
        positions[0] = (0.2, 0.2, 0.2)
        for distance in (SWITCH-1e-4, SWITCH, SWITCH+1e-4, 0.75,
                         geo.CUTOFF-1e-4, geo.CUTOFF, geo.CUTOFF+1e-4):
            positions[1] = positions[0]+(distance, 0, 0)
            context.setPositions(positions)
            geo._assert_close(geo._snapshot(context),
                              _reference(positions, particles, [], False, rule))
        distance, step = 0.75, 2e-4
        energies = []
        for r in (distance-step, distance+step, distance):
            positions[1] = positions[0]+(r, 0, 0)
            context.setPositions(positions)
            energy, forces = geo._snapshot(context)
            energies.append(energy)
        finite_difference = -(energies[1]-energies[0])/(2*step)
        np.testing.assert_allclose(forces[1, 0], finite_difference,
                                   rtol=2e-3, atol=geo.FORCE_QUANTUM)
    finally:
        del context, integrator, force
        gc.collect()


def test_switched_graph_modes_box_and_parameter_updates(plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    positions, particles, exceptions = _fixture()
    integrator = mm.CustomIntegrator(0.001)
    integrator.setKineticEnergyExpression("0.5*m*v*v")
    context, integrator, force = _context("Prolate", particles, exceptions, True,
                                          "Geometric", integrator=integrator)
    try:
        context.setPositions(positions)
        reference = _reference(positions, particles, exceptions, True, "Geometric")
        capfd.readouterr()
        for energy, forces in ((False, True), (True, False), (True, True)):
            for _ in range(56 if not energy else 4):
                sample = geo._snapshot(context, energy, forces)
            geo._assert_close(sample, reference)
        trace = capfd.readouterr().err
        for mode in ("energy=0 forces=1", "energy=1 forces=0", "energy=1 forces=1"):
            assert mode in trace, trace

        box = geo.BOX*1.04
        positions = positions*1.04 + (0.03, -0.02, 0.01)
        context.setPeriodicBoxVectors(mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0),
                                      mm.Vec3(0, 0, box))
        context.setPositions(positions)
        geo._assert_close(geo._snapshot(context),
                          _reference(positions, particles, exceptions, True, "Geometric", box))
        particles[6, 1:] = (0.34, 0.25)
        force.setParticleParameters(6, *particles[6])
        exceptions[1][3:] = [0.91, 0.17]
        force.setExceptionParameters(1, *exceptions[1])
        force.updateParametersInContext(context)
        reference = _reference(positions, particles, exceptions, True, "Geometric", box)
        for energy, forces in ((False, True), (True, False), (True, True)):
            geo._assert_close(geo._snapshot(context, energy, forces), reference)

        force.setSwitchingDistance(0.65)
        with pytest.raises(Exception, match="(?i)switch.*reinitial|reinitial.*switch"):
            force.updateParametersInContext(context)
        force.setSwitchingDistance(SWITCH)
        geo._assert_close(geo._snapshot(context), reference)
    finally:
        del context, integrator, force
        gc.collect()


def test_switching_conversion_defaults_and_active_validation(plugin_dir):
    import nv_openmm_plugin as nv

    assert not nv.NonbondedForce().getUseSwitchingFunction()
    assert nv.NonbondedForce().getSwitchingDistance() == -1
    _, particles, exceptions = _fixture()
    _, source = geo._source("PME", particles, exceptions, True)
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(SWITCH)
    converted = nv.NonbondedForce(source)
    assert converted.getUseSwitchingFunction()
    assert converted.getSwitchingDistance() == SWITCH
    # Each invalid active interval must be rejected before launching GPU work.
    for start in (-1.0, float("nan"), geo.CUTOFF, geo.CUTOFF+0.01,
                  np.nextafter(geo.CUTOFF, 0.0)):
        with pytest.raises(Exception, match="(?i)switch"):
            _context("Prolate", particles, exceptions, False, "Geometric", start=start)
        gc.collect()
    with pytest.raises(Exception, match="(?i)switch|dispersion"):
        _context("Prolate", particles, exceptions, True, "Geometric", start=0.0)
    gc.collect()


def test_zero_switch_distance_without_dispersion_correction(plugin_dir):
    positions, particles, exceptions = _fixture()
    candidate = _evaluate("Prolate", positions, particles, exceptions, False,
                          "Geometric", start=0.0)
    geo._assert_close(candidate, _reference(positions, particles, exceptions, False,
                                           "Geometric", start=0.0))


def test_invalid_switched_tail_update_preserves_context(plugin_dir):
    positions, particles, exceptions = _fixture()
    particles[:, 2] = 0
    for exception in exceptions:
        exception[4] = 0
    context, integrator, force = _context("Prolate", particles, exceptions, True,
                                          "Geometric", start=0.0)
    try:
        context.setPositions(positions)
        original = geo._snapshot(context)
        assert original[0] == 0
        np.testing.assert_array_equal(original[1], np.zeros_like(positions))

        # Two neighboring ordinary particles become active, so an upload before
        # tail validation would leave a measurable force after the failed update.
        for index in (10, 11):
            force.setParticleParameters(index, 0.0, 0.5, 0.7)
        with pytest.raises(Exception, match="(?i)positive switching distance"):
            force.updateParametersInContext(context)
        current = geo._snapshot(context)
        assert current[0] == original[0]
        np.testing.assert_array_equal(current[1], original[1])

        # Rebuilding the cell list must not expose partially updated atom data.
        context.setPositions(positions + (0.01, 0.0, 0.0))
        current = geo._snapshot(context)
        assert current[0] == original[0]
        np.testing.assert_array_equal(current[1], original[1])

        for index in (10, 11):
            force.setParticleParameters(index, *particles[index])
        force.updateParametersInContext(context)
        current = geo._snapshot(context)
        assert current[0] == original[0]
        np.testing.assert_array_equal(current[1], original[1])
    finally:
        del context, integrator, force
        gc.collect()


@pytest.mark.parametrize("change", ["enabled", "distance"])
def test_switch_changes_before_lazy_setup_require_reinitialize(plugin_dir, change):
    positions, particles, exceptions = _fixture()
    # No evaluation occurs before updateParametersInContext, exercising the
    # lazy-setup branch. A distance matters only when switching is active.
    originally_enabled = change == "distance"
    context, integrator, force = _context("Prolate", particles, exceptions, False,
                                          "LorentzBerthelot", enabled=originally_enabled)
    try:
        context.setPositions(positions)
        if change == "enabled":
            # An inactive distance does not enter the kernel or correction.
            force.setSwitchingDistance(0.65)
            force.updateParametersInContext(context)
            force.setSwitchingDistance(SWITCH)
            force.setUseSwitchingFunction(True)
        else:
            force.setSwitchingDistance(0.65)
        with pytest.raises(Exception, match="(?i)switch.*reinitial|reinitial.*switch"):
            force.updateParametersInContext(context)
        geo._assert_close(geo._snapshot(context),
                          _reference(positions, particles, exceptions, False,
                                     "LorentzBerthelot", enabled=originally_enabled))
        context.reinitialize(preserveState=True)
        geo._assert_close(geo._snapshot(context),
                          _reference(positions, particles, exceptions, False,
                                     "LorentzBerthelot", start=0.65 if change == "distance"
                                     else SWITCH))
    finally:
        del context, integrator, force
        gc.collect()
