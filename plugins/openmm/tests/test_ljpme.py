# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One direct traversal plus Coulomb/dispersion meshes, including live weights."""

import contextlib
import itertools
import math

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

RC = 1.0
BOX = 3.6
XYZ = np.array(list(itertools.product((.2, .8, 1.4, 2., 2.6, 3.2), repeat=3)))
XYZ += np.random.default_rng(1837).uniform(-.015, .015, XYZ.shape)
KJ = unit.kilojoule_per_mole
NM = unit.nanometer


def _source():
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.LJPME)
    force.setCutoffDistance(RC)
    force.setEwaldErrorTolerance(5e-4)
    force.setExceptionsUsePeriodicBoundaryConditions(True)
    force.setUseDispersionCorrection(False)
    force.addGlobalParameter("lam", .2)
    for i in range(len(XYZ)):
        force.addParticle((-1)**i*.025, .26+.008*(i % 5), .2+.015*(i % 4))
        if i % 7 == 0:
            force.addParticleParameterOffset("lam", i, .014, .018, .09)
    for i, j, q, s, e in ((0, 1, -.001, .31, .08), (4, 7, 0., .25, 0.), (12, 47, .002, .3, .07)):
        row = force.addException(i, j, q, s, e)
        force.addExceptionParameterOffset("lam", row, .002, .01, .03)
    force.setForceGroup(3)
    return force


def _custom(source, geometric=False):
    sigma = "sqrt(s1*s2)" if geometric else "0.5*(s1+s2)"
    force = mm.CustomNonbondedForce(
        "4*sqrt(e1*e2)*((sigma/r)^12-(sigma/r)^6);"
        f"sigma={sigma};s1=sigma1+lam*ds1;s2=sigma2+lam*ds2;"
        "e1=epsilon1+lam*de1;e2=epsilon2+lam*de2"
    )
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(RC)
    force.setForceGroup(3)
    force.addGlobalParameter("lam", .2)
    for name in ("sigma", "epsilon", "ds", "de"):
        force.addPerParticleParameter(name)
    for i in range(source.getNumParticles()):
        _, sigma, epsilon = source.getParticleParameters(i)
        force.addParticle([sigma.value_in_unit(NM), epsilon.value_in_unit(KJ),
                           .018 if i % 7 == 0 else 0., .09 if i % 7 == 0 else 0.])
    for row in range(source.getNumExceptions()):
        i, j, *_ = source.getExceptionParameters(row)
        force.addExclusion(i, j)
    return force


def _box(triclinic=False, scale=1.):
    return tuple(mm.Vec3(*v)*scale for v in
                 ((BOX, 0, 0), (.35 if triclinic else 0, BOX, 0),
                  (-.2 if triclinic else 0, .25 if triclinic else 0, BOX)))


def _converged_reference(force):
    # Set only AFTER plugin conversion: source meshes must otherwise be automatic.
    # Stock's default dispersion mesh is half the Coulomb size; use a converged
    # reference so its separate finite-mesh error does not set our tolerance.
    alpha = np.sqrt(-np.log(2*force.getEwaldErrorTolerance()))/RC
    force.setPMEParameters(alpha, 64, 64, 64)
    force.setLJPMEParameters(alpha, 64, 64, 64)


@contextlib.contextmanager
def _context(force, triclinic=False):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(*_box(triclinic))
    for item in force if isinstance(force, (tuple, list)) else [force]:
        system.addForce(item)
    integrator = mm.LangevinMiddleIntegrator(0, 1, .001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed", "DeterministicForces": "true"})
    context.setPositions(XYZ)
    try:
        yield context
    finally:
        del context, integrator


def _state(context, mode=(True, True), groups=-1):
    energy, forces = mode
    state = context.getState(getEnergy=energy, getForces=forces, groups=groups)
    return (state.getPotentialEnergy().value_in_unit(KJ) if energy else None,
            state.getForces(asNumpy=True).value_in_unit(KJ/NM) if forces else None)


def _ewald_omitted_real_space(delta, box, charges, weights, excluded, alpha, cutoff):
    """Restore Ewald's omitted Coulomb/dispersion SR terms beyond the cutoff.

    Prolate's SR is compact; Ewald's Gaussian SR has a small omitted tail.
    Enumerate images independently of the mesh and cuPo's splitting helpers.
    delta contains minimum-imaged x_i-x_j. At alpha >= sqrt(-log(.001)),
    exp(-(alpha*2.5)**2) < 2e-19, so the remaining tail is negligible here.
    """
    radius = 2.5
    assert alpha*radius > 6.5
    # The nearest image plus its 26 neighboring images covers this sphere.
    assert radius < np.min(1/np.linalg.norm(np.linalg.inv(box), axis=0))
    qprod = 138.93545764498*charges[:, None]*charges[None, :]
    c6 = weights[:, None]*weights[None, :]
    energy = 0.
    forces = np.zeros((len(charges), 3))
    for image in itertools.product((-1, 0, 1), repeat=3):
        displacement = delta + np.asarray(image)@box
        r2 = np.sum(displacement*displacement, axis=2)
        admitted = (r2 >= cutoff*cutoff) & (r2 <= radius*radius)
        if image == (0, 0, 0):
            # Exceptions replace the nearest pair only. Their periodic copies
            # and nonzero self images remain part of the lattice interaction.
            admitted &= ~excluded
        i, j = np.nonzero(admitted)
        rr = r2[i, j]
        r = np.sqrt(rr)
        ar = alpha*r
        y = ar*ar
        gaussian = np.exp(-y)
        erfc = np.fromiter((math.erfc(value) for value in ar), dtype=float, count=len(ar))
        polynomial = 1+y+.5*y*y
        charge = qprod[i, j]
        dispersion = c6[i, j]
        energy += .5*np.sum(charge*erfc/r-dispersion*gaussian*polynomial/rr**3)
        factor = (charge*(erfc+2*ar*gaussian/np.sqrt(np.pi))/(rr*r)
                  -dispersion*gaussian*(6*polynomial+y**3)/rr**4)
        np.add.at(forces, i, factor[:, None]*displacement[i, j])
    return energy, forces


def _reference_state(context, source, method, mode=(True, True)):
    energy, forces = _state(context, mode)
    if method != "Prolate":
        return energy, forces
    # Each splitter shifts its own direct LJ energy to zero at rc. Ewald's
    # nonzero SR6(rc) therefore adds a constant B*SR6(rc) to each ordinary pair
    # inside rc compared with Prolate. Account for that known energy convention,
    # then restore Ewald's omitted Gaussian real-space tail: Prolate's compact
    # split includes this contribution through its reciprocal complement.
    values = np.array([[q.value_in_unit(unit.elementary_charge), s.value_in_unit(NM),
                        e.value_in_unit(KJ)] for q, s, e in
                       (source.getParticleParameters(i) for i in range(source.getNumParticles()))])
    for index in range(source.getNumParticleParameterOffsets()):
        name, atom, dq, ds, de = source.getParticleParameterOffset(index)
        # OpenMM returns offset scales as raw MD-unit doubles.
        scale = [dq, ds, de]
        values[atom] += context.getParameter(name)*np.asarray(scale)
    weights = 2*np.sqrt(values[:, 2])*values[:, 1]**3
    state = context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(NM)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(NM)
    delta = positions[:, None, :]-positions[None, :, :]
    for axis in (2, 1, 0):
        delta -= np.rint(delta[:, :, axis]/box[axis, axis])[:, :, None]*box[axis]
    admitted = np.triu(np.sum(delta*delta, axis=2) < RC*RC, 1)
    excluded = np.zeros(admitted.shape, dtype=bool)
    for index in range(source.getNumExceptions()):
        i, j, *_ = source.getExceptionParameters(index)
        admitted[min(i, j), max(i, j)] = False
        excluded[i, j] = excluded[j, i] = True
    y = -np.log(2*source.getEwaldErrorTolerance())
    sr_at_cutoff = np.exp(-y)*(1+y+.5*y*y)/RC**6
    tail_energy, tail_forces = _ewald_omitted_real_space(
        delta, box, values[:, 0], weights, excluded, np.sqrt(y)/RC, RC)
    if energy is not None:
        energy += tail_energy-sr_at_cutoff*np.sum((weights[:, None]*weights[None, :])[admitted])
    if forces is not None:
        forces += tail_forces
    return energy, forces


def _close(actual, expected, *, energy=0.02, force=0.002):
    if actual[0] is not None:
        assert actual[0] == pytest.approx(expected[0], abs=energy, rel=2e-4)
    if actual[1] is not None:
        np.testing.assert_allclose(actual[1], expected[1], atol=force, rtol=2e-3)


def _reference_force_tolerance(method):
    # This fixture's existing, charge-only Prolate path has max error .00258
    # against the converged lattice reference, unchanged with LJPME disabled.
    # The new dispersion mesh alone is below .00042 at the shared Coulomb grid.
    return .003 if method == "Prolate" else .002


@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("triclinic", [False, True])
def test_ljpme_matches_converged_reference_live_parameters_and_box(method, custom, triclinic, monkeypatch):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    source = _source()
    plugin = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
    assert plugin.getUseLJPME()
    assert [plugin.getLatticeSumInteraction(i).kernel.power
            for i in range(plugin.getNumLatticeSumInteractions())] == [1, 6]
    if custom:
        plugin.setCutoffInteraction(_custom(source))
    _converged_reference(source)
    with _context(source, triclinic) as reference, _context(plugin, triclinic) as actual:
        # Cover independent output modes, lambda-updated self/reciprocal weights,
        # and a mesh-influence refresh without changing allocation dimensions.
        for lam, scale in ((.2, 1.), (.7, 1.), (.45, 1.03)):
            for ctx in (reference, actual):
                ctx.setParameter("lam", lam)
                ctx.setPeriodicBoxVectors(*_box(triclinic, scale))
            for mode in ((True, True), (False, True), (True, False)):
                _close(_state(actual, mode), _reference_state(reference, source, method, mode),
                       force=_reference_force_tolerance(method))
            absent = _state(actual, groups=1 << 2)
            assert absent[0] == 0.
            np.testing.assert_array_equal(absent[1], 0.)
        # Advance only one trajectory, then compare the two implementations at
        # exactly those coordinates. This exercises moving row sorts and both
        # reciprocal consumers during force-only integration.
        actual.setVelocitiesToTemperature(300, 1837)
        actual.getIntegrator().step(5)
        positions = actual.getState(getPositions=True)
        reference.setPositions(positions.getPositions())
        reference.setPeriodicBoxVectors(*positions.getPeriodicBoxVectors())
        _close(_state(actual), _reference_state(reference, source, method),
               force=_reference_force_tolerance(method))


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_ljpme_geometric_custom_native_and_graph_updates(method, monkeypatch):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    source = _source()
    native = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
    native.setLJCombiningRule(native.Geometric)
    custom = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
    custom.setCutoffInteraction(_custom(source, geometric=True))
    with _context(native) as reference, _context(custom) as actual:
        for _ in range(55):
            _state(actual, (False, True))
            _state(reference, (False, True))
        for lam in (.2, .8, .1):
            actual.setParameter("lam", lam)
            reference.setParameter("lam", lam)
            _close(_state(actual), _state(reference), energy=0.002, force=0.0003)
        checkpoint = actual.createCheckpoint()
        actual.setPositions(XYZ[::-1].copy())
        _state(actual)
        actual.loadCheckpoint(checkpoint)
        _close(_state(actual), _state(reference), energy=0.002, force=0.0003)


def test_ljpme_api_rejects_incompatible_switch_tail_and_pinned_mesh():
    source = _source()
    source.setUseSwitchingFunction(True)
    source.setSwitchingDistance(.8)
    source.setUseDispersionCorrection(True)
    plugin = nv.NonbondedForce(source)
    assert plugin.getUseLJPME()
    assert not plugin.getUseSwitchingFunction()
    assert not plugin.getUseDispersionCorrection()
    assert source.getUseSwitchingFunction()
    assert source.getUseDispersionCorrection()
    plugin.setUseSwitchingFunction(True)
    with pytest.raises(Exception, match="LJPME.*switching"):
        with _context(plugin):
            pass
    source.setLJPMEParameters(0, 32, 32, 32)
    with pytest.raises(Exception, match="pinned LJPME"):
        nv.NonbondedForce(source)


@pytest.mark.parametrize("custom", [False, True])
def test_ljpme_explicit_particle_updates_and_policy_are_live(custom, monkeypatch):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    source = _source()
    plugin = nv.NonbondedForce(source, 1.15, nv.NonbondedForce.PME)
    if custom:
        plugin.setCutoffInteraction(_custom(source))
    _converged_reference(source)
    with _context(source) as reference, _context(plugin) as actual:
        _close(_state(actual), _state(reference))
        # Both the ordinary mesh weight and the independent exception row change.
        for force in (source, plugin):
            force.setParticleParameters(0, -.017, .315, .24)
            force.setExceptionParameters(0, 0, 1, -.002, .29, .035)
        if custom:
            definition = plugin.getCutoffExpression()
            values = list(definition.getParticleParameters(0))
            values[:2] = [.315, .24]
            definition.setParticleParameters(0, values)
            plugin.setCutoffInteraction(definition)
        source.updateParametersInContext(reference)
        plugin.updateParametersInContext(actual)
        for lam in (.2, .8):
            actual.setParameter("lam", lam)
            reference.setParameter("lam", lam)
            _close(_state(actual), _state(reference))
        plugin.setUseLJPME(False)
        with pytest.raises(Exception, match="interaction policy changed"):
            plugin.updateParametersInContext(actual)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_ljpme_softcore_custom_matches_stock_residual_expression(method, monkeypatch):
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    source = _source()
    definition = _custom(source)
    # epsilon_i(coupling)=coupling*epsilon_i makes b6_i scale as sqrt(coupling),
    # exactly matching this softcore expression's asymptotic pair coefficient.
    # Retain live sigma/charge offsets independently, without a product-of-globals
    # parameter law that OpenMM's native linear offsets cannot represent.
    source.addGlobalParameter("coupling", .5)
    for i in range(source.getNumParticles()):
        q, sigma, epsilon = source.getParticleParameters(i)
        source.setParticleParameters(i, q, sigma, 0.)
        source.addParticleParameterOffset("coupling", i, 0., 0., epsilon)
        values = list(definition.getParticleParameters(i))
        values[3] = 0.
        definition.setParticleParameters(i, values)
    for i in range(source.getNumParticleParameterOffsets()):
        name, particle, dq, ds, de = source.getParticleParameterOffset(i)
        if name == "lam":
            source.setParticleParameterOffset(i, name, particle, dq, ds, 0.)
    _, intermediates = definition.getEnergyFunction().split(";", 1)
    definition.addGlobalParameter("coupling", .5)
    softcore = "4*sqrt(e1*e2)*coupling*x*(x-1)"
    ordinary = "4*sqrt(e1*e2)*coupling*((sigma/r)^12-(sigma/r)^6)"
    soft_at_cutoff = "4*sqrt(e1*e2)*coupling*x0*(x0-1)"
    ordinary_at_cutoff = f"4*sqrt(e1*e2)*coupling*((sigma/{RC})^12-(sigma/{RC})^6)"
    auxiliary = f";x=1/(0.5*(1-coupling)+(r/sigma)^6);x0=1/(0.5*(1-coupling)+({RC}/sigma)^6);"
    definition.setEnergyFunction(softcore+auxiliary+intermediates)
    plugin = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
    plugin.setCutoffInteraction(definition)
    _converged_reference(source)
    residual = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(definition))
    residual.setEnergyFunction(
        f"({softcore})-({ordinary})-({soft_at_cutoff})+({ordinary_at_cutoff})"+auxiliary+intermediates)
    with _context([source, residual]) as reference, _context(plugin) as actual:
        for lam, coupling in ((.2, .5), (.7, .1), (.4, 1.), (.8, 0.)):
            for ctx in (reference, actual):
                ctx.setParameter("lam", lam)
                ctx.setParameter("coupling", coupling)
            _close(_state(actual), _reference_state(reference, source, method),
                   force=_reference_force_tolerance(method))


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_ljpme_finite_softcore_at_close_separation(method, monkeypatch):
    # Only two particles carry dispersion weights, so a converged periodic
    # image sum is cheap. A stock LJ + (softcore-LJ) reference would itself lose
    # precision at these distances and cannot validate stable cancellation.
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")
    coupling = .5
    sigmas, epsilons = np.array([.30, .32]), np.array([.2, .25])
    weights = 2*np.sqrt(coupling*epsilons)*sigmas**3
    sigma, epsilon = sigmas.mean(), np.sqrt(np.prod(epsilons))
    source = mm.NonbondedForce()
    source.setNonbondedMethod(source.LJPME)
    source.setCutoffDistance(RC)
    source.setEwaldErrorTolerance(5e-4)
    source.addGlobalParameter("coupling", coupling)
    definition = mm.CustomNonbondedForce(
        "4*sqrt(epsilon1*epsilon2)*coupling*x*(x-1);"
        "x=1/(.5*(1-coupling)+(r/sigma)^6);sigma=.5*(sigma1+sigma2)")
    definition.setNonbondedMethod(definition.CutoffPeriodic)
    definition.setCutoffDistance(RC)
    definition.addGlobalParameter("coupling", coupling)
    definition.addPerParticleParameter("sigma")
    definition.addPerParticleParameter("epsilon")
    for atom in range(len(XYZ)):
        index = atom-20
        sig = sigmas[index] if index in (0, 1) else .3
        eps = epsilons[index] if index in (0, 1) else 0.
        source.addParticle(0., sig, 0.)
        definition.addParticle([sig, eps])
        if eps:
            source.addParticleParameterOffset("coupling", atom, 0., 0., eps)
    plugin = nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method))
    plugin.setCutoffInteraction(definition)
    images = np.array(list(itertools.product(range(-8, 9), repeat=3)), dtype=float)
    images = images[np.any(images != 0, axis=1)]*BOX
    x0 = 1/(.5*(1-coupling)+(RC/sigma)**6)
    u0 = 4*epsilon*coupling*x0*(x0-1)
    y = -np.log(2*source.getEwaldErrorTolerance())
    sr_cutoff = np.exp(-y)*(1+y+.5*y*y)/RC**6 if method == "PME" else 0.
    with _context(plugin) as context:
        for distance in (.03, .01):
            positions = XYZ.copy()
            positions[21] = positions[20]+[distance, 0, 0]
            context.setPositions(positions)
            x = 1/(.5*(1-coupling)+(distance/sigma)**6)
            energy = 4*epsilon*coupling*x*(x-1)-u0-np.prod(weights)*(RC**-6-sr_cutoff)
            forces = np.zeros_like(positions)
            delta = positions[20]-positions[21]
            forces[20] = 24*epsilon*coupling*(2*x-1)*x*x*distance**4/sigma**6*delta
            forces[21] = -forces[20]
            # Nearest-image LR cancels exactly against reciprocal space. Other
            # images are outside rc and carry geometric dispersion. Their Ewald
            # SR is <1e-30 here; the omitted image-shell energy is below 1e-8.
            for i, j in itertools.product(range(2), repeat=2):
                delta = positions[20+i]-positions[20+j]+images
                r2 = np.sum(delta*delta, axis=1)
                product = weights[i]*weights[j]
                energy -= .5*product*np.sum(r2**-3)
                forces[20+i] -= 6*product*np.sum(delta/r2[:, None]**4, axis=0)
            # For these exact positions, isolated Prolate reciprocal force
            # error falls from .000383 at 21^3 to .000041 at 32^3 and .000009
            # at 64^3. Keep the shared default Coulomb mesh in this test.
            force_tolerance = .0005 if method == "Prolate" else .0003
            _close(_state(context), (energy, forces), energy=.002, force=force_tolerance)
