# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom tails retain caller expression/region-mask semantics and live state."""

import contextlib
import itertools

import numpy as np
import openmm as mm
import pytest
import nv_openmm_plugin as nv

from test_custom_nonbonded import XYZ, compare, context, snapshot, switched_expression
def compare_modes(actual, reference):
    for energy, forces in ((False, True), (True, False), (True, True)):
        compare(snapshot(actual, energy, forces), snapshot(reference, energy, forces))


def source(switched=False, masked=False):
    radial = "lambda*(a1*a2/r^6-.02*(a1+a2)/r^8)"
    expression = (f"select(cross,{radial},0);"
                  "cross=alchemical1+alchemical2-2*alchemical1*alchemical2"
                  if masked else radial)
    if switched:
        expression = switched_expression(expression, .85, 1.25)
    force = mm.CustomNonbondedForce(expression)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.25)
    force.setUseLongRangeCorrection(True)
    force.addPerParticleParameter("a")
    if masked:
        force.addPerParticleParameter("alchemical")
    force.addGlobalParameter("lambda", .7)
    for atom in range(len(XYZ)):
        force.addParticle([.2+.1*(atom % 3)] + ([float(atom % 2)] if masked else []))
    force.addExclusion(0, 1)
    force.addExclusion(2, 3)
    return force


@pytest.mark.parametrize("switching", ["none", "manual", "explicit"])
@pytest.mark.parametrize("masked", [False, True])
def test_tail_matches_stock_after_lambda_particles_and_volume_updates(plugin_dir, switching, masked):
    stock = source(switching == "manual", masked)
    if switching == "explicit":
        stock.setUseSwitchingFunction(True)
        stock.setSwitchingDistance(.85)
    custom = nv.NonbondedForce(stock)
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        compare_modes(actual, reference)
        for value in (0., .4, 1.):
            for ctx in (reference, actual):
                ctx.setParameter("lambda", value)
            compare_modes(actual, reference)
        record = [.8]+([0.] if masked else [])
        stock.setParticleParameters(5, record)
        replacement = custom.getCutoffExpression()
        replacement.setParticleParameters(5, record)
        custom.setCutoffInteraction(replacement)
        for force, ctx in ((stock, reference), (custom, actual)):
            force.updateParametersInContext(ctx)
            ctx.setPeriodicBoxVectors(mm.Vec3(4.2, 0, 0), mm.Vec3(0, 4.2, 0),
                                      mm.Vec3(0, 0, 4.2))
        compare_modes(actual, reference)
        for force, ctx in ((stock, reference), (custom, actual)):
            force.setGlobalParameterDefaultValue(0, .25)
            force.updateParametersInContext(ctx)
            assert ctx.getParameter("lambda") == 1.
        # Edited defaults describe future Contexts. Explicit updates must keep
        # evaluating this Context's current lambda in the tail as well as pairs.
        compare_modes(actual, reference)


@contextlib.contextmanager
def tail_only_context(force, platform, triclinic=False):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(5, 0, 0), mm.Vec3(.4 if triclinic else 0, 5, 0),
                                        mm.Vec3(-.3 if triclinic else 0, .2 if triclinic else 0, 5))
    system.addForce(force)
    integrator = mm.VerletIntegrator(.001)
    properties = {"Precision": "mixed"} if platform == "CUDA" else {}
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    ctx.setPositions(np.array(list(itertools.product((.2, 1.7, 3.2), repeat=3))))
    try:
        yield ctx
    finally:
        del ctx, integrator


def test_complete_definition_update_preserves_additional_tail_globals_cuda(plugin_dir):
    definition = source()
    definition.setEnergyFunction("lambda*a1*a2/r^6")
    custom = nv.NonbondedForce(definition)
    tail = source()
    tail.addGlobalParameter("tail_scale", .3)
    tail.setEnergyFunction("tail_scale*a1*a2/r^6")
    custom.addExpressionTailCorrection(tail)
    stock = source()
    stock.addGlobalParameter("tail_scale", .3)
    stock.setEnergyFunction("(lambda+tail_scale)*a1*a2/r^6")

    # Every pair is beyond the cutoff: stock's sum is an independent oracle for
    # the plugin's implicit tail plus its separately owned additional tail.
    with tail_only_context(stock, "Reference") as reference, \
            tail_only_context(custom, "CUDA") as actual:
        for ctx in (reference, actual):
            ctx.setParameter("lambda", .31)
            ctx.setParameter("tail_scale", .81)
        compare_modes(actual, reference)
        for force in (stock, definition, tail):
            force.setParticleParameters(5, [.8])
        # The complete replacement has no tail_scale declaration. Its setter
        # must retain that existing global, including the Context's live value.
        custom.setCutoffInteraction(definition)
        custom.setExpressionTailCorrection(0, tail)
        stock.updateParametersInContext(reference)
        custom.updateParametersInContext(actual)
        assert actual.getParameter("lambda") == .31
        assert actual.getParameter("tail_scale") == .81
        compare_modes(actual, reference)
        for ctx in (reference, actual):
            ctx.setParameter("lambda", 0.)
            ctx.setParameter("tail_scale", 1.2)
        compare_modes(actual, reference)


def test_tail_uses_triclinic_volume_with_live_global_updates(plugin_dir):
    stock = source()
    stock.setEnergyFunction("lambda*a1*a2/r^6")
    for atom in range(len(XYZ)):
        stock.setParticleParameters(atom, [1.])
    custom = nv.NonbondedForce(stock, 1.15)
    with tail_only_context(stock, "Reference", True) as reference, \
            tail_only_context(custom, "CUDA", True) as actual:
        for length, lam in ((5., .7), (5.4, .2), (5.1, 1.)):
            box = (mm.Vec3(length, 0, 0), mm.Vec3(.4, length, 0), mm.Vec3(-.3, .2, length))
            for ctx in (reference, actual):
                ctx.setPeriodicBoxVectors(*box)
                ctx.setParameter("lambda", lam)
            compare_modes(actual, reference)
            expected = 2*np.pi*len(XYZ)**2*np.float32(lam)/(3*1.25**3*length**3)
            np.testing.assert_allclose(snapshot(actual)[0], expected, rtol=2e-5, atol=1e-8)


def test_tail_updates_preserve_stock_float_evaluation_of_double_globals(plugin_dir):
    stock = source()
    stock.setEnergyFunction("lambda*a1*a2/r^6")
    for atom in range(len(XYZ)):
        stock.setParticleParameters(atom, [1.])
    custom = nv.NonbondedForce(stock)
    # Every ordinary pair lies beyond the cutoff. Only C(lambda)/V remains.
    # The Context accepts doubles, while stock's tail evaluator rounds inputs
    # to float; a sub-ULP change must retain that source numerical behavior.
    for force in (stock, custom):
        force.setGlobalParameterDefaultValue(0, 1.)
    with tail_only_context(stock, "Reference") as reference, tail_only_context(custom, "CUDA") as actual:
        initial = [snapshot(ctx)[0] for ctx in (reference, actual)]
        analytic = 2*np.pi*len(XYZ)**2/(3*1.25**3*5**3)
        np.testing.assert_allclose(initial, analytic, rtol=1e-5, atol=1e-9)
        value = 1.+2**-26
        assert np.float32(value) == np.float32(1.)
        for ctx in (reference, actual):
            ctx.setParameter("lambda", value)
        changed = [snapshot(ctx)[0] for ctx in (reference, actual)]
        expected = changed[0]-initial[0]
        observed = changed[1]-initial[1]
        assert expected == 0.
        assert observed == 0.
        compare(snapshot(actual), snapshot(reference))
        for ctx in (reference, actual):
            ctx.setParameter("lambda", 1.+2**-18)
        compare(snapshot(actual), snapshot(reference))
        assert snapshot(actual)[0] != changed[1]


def test_force_only_defers_tail_integration_until_energy_is_requested(plugin_dir):
    stock = source()
    stock.setEnergyFunction("a1*a2*((1-lambda)/r^6+lambda/r^2)")
    stock.setGlobalParameterDefaultValue(0, 0.)
    for atom in range(len(XYZ)):
        stock.setParticleParameters(atom, [1.])
    custom = nv.NonbondedForce(stock)
    # The r^-2 term has finite cutoff forces but a divergent homogeneous tail.
    # A truncated Reference force independently supplies the direct-space oracle.
    stock.setUseLongRangeCorrection(False)
    with context(stock, "Reference", middle=True) as reference, \
            context(custom, "CUDA", middle=True) as actual:
        for _ in range(2):
            # Exercise both lazy initialization and an already prepared tail.
            for ctx in (reference, actual):
                ctx.setParameter("lambda", 1.)
            compare(snapshot(actual, False, True), snapshot(reference, False, True))
            with pytest.raises(Exception, match="Long range correction did not converge"):
                snapshot(actual, True, False)
            # Failed integration must not poison later energy/force evaluation.
            for ctx in (reference, actual):
                ctx.setParameter("lambda", 0.)
            energy, forces = snapshot(reference)
            coefficient = 2*np.pi*len(XYZ)**2/(3*1.25**3)
            compare(snapshot(actual), (energy+coefficient/4**3, forces))


def test_force_only_still_validates_unused_additional_tail_globals(plugin_dir):
    definition = source()
    definition.setUseLongRangeCorrection(False)
    custom = nv.NonbondedForce(definition)
    tail = source()
    tail.addGlobalParameter("tail_unused", 1.)
    custom.addExpressionTailCorrection(tail)
    with context(custom, "CUDA", middle=True) as actual:
        expected = snapshot(actual, False, True)
        for value in (float("nan"), float("inf"), 1e100):
            actual.setParameter("tail_unused", value)
            with pytest.raises(Exception, match="finite FP32"):
                snapshot(actual, False, True)
            actual.setParameter("tail_unused", 2.)
            compare(snapshot(actual, False, True), expected)
        # The unused value did not change the expression; energy preparation
        # after force-only requests must still work and include the extra tail.
        reference_force = source()
        with context(reference_force, "Reference", middle=True) as reference:
            compare(snapshot(actual), snapshot(reference))


def test_explicit_compact_switch_has_no_implicit_unswitched_tail(plugin_dir):
    stock = source(switched=True)
    custom = nv.NonbondedForce(stock)
    # The expression is zero beyond cutoff. The runtime must not infer a bare
    # potential or add the deficit inside the caller's switching interval.
    with tail_only_context(stock, "Reference") as reference, tail_only_context(custom, "CUDA") as actual:
        for value in (0., .4, 1.):
            for ctx in (reference, actual):
                ctx.setParameter("lambda", value)
                energy, forces = snapshot(ctx)
                assert energy == 0.
                np.testing.assert_array_equal(forces, 0.)


def test_automatic_switch_retains_the_unswitched_tail_and_switching_deficit(plugin_dir):
    stock = source()
    stock.setEnergyFunction("lambda*a1*a2/r^6")
    stock.setUseSwitchingFunction(True)
    stock.setSwitchingDistance(.85)
    for atom in range(len(XYZ)):
        stock.setParticleParameters(atom, [1.])
    custom = nv.NonbondedForce(stock)
    assert custom.getCutoffExpression().getEnergyFunction() == stock.getEnergyFunction()
    assert custom.getCutoffExpression().getUseSwitchingFunction()

    # Independently integrate the finite switching deficit with Gauss-Legendre
    # quadrature; the beyond-cutoff r^-6 integral is analytic. Every real pair
    # is beyond the cutoff, so the observed energy is this correction alone.
    points, weights = np.polynomial.legendre.leggauss(64)
    x = (points+1)/2
    r = .85 + (1.25-.85)*x
    removed = x**3*(10-15*x+6*x*x)
    deficit = (1.25-.85)/2*np.dot(weights, removed/r**4)
    far = 1/(3*1.25**3)
    assert deficit > 0.
    with tail_only_context(stock, "Reference") as reference, \
            tail_only_context(custom, "CUDA") as actual:
        for value in (.7, .25, 0., 1.):
            for ctx in (reference, actual):
                ctx.setParameter("lambda", value)
            compare_modes(actual, reference)
            energy, forces = snapshot(actual)
            expected = 2*np.pi*len(XYZ)**2*np.float32(value)*(far+deficit)/5**3
            np.testing.assert_allclose(energy, expected, rtol=2e-5, atol=1e-8)
            np.testing.assert_array_equal(forces, 0.)
