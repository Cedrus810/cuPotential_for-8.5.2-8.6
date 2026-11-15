# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing preserves expression semantics and live OpenMM state."""
import numpy as np
import openmm as mm
import pytest
import nv_openmm_plugin as nv

from test_custom_nonbonded import XYZ, compare, context, snapshot


OUTPUT_MODES = ((False, True), (True, False), (True, True))


def compare_modes(actual, reference):
    for energy, forces in OUTPUT_MODES:
        observed = snapshot(actual, energy, forces)
        expected = snapshot(reference, energy, forces)
        for result in (observed, expected):
            for value in result:
                if value is not None:
                    assert np.isfinite(value).all()
        compare(observed, expected)
    # Joint value/gradient lowering must agree with standalone force lowering.
    np.testing.assert_array_equal(snapshot(actual, False, True)[1], snapshot(actual)[1])


def test_unused_particle_records_preserve_updates_and_validation(plugin_dir):
    stock = mm.CustomNonbondedForce("scale/r^6")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addPerParticleParameter("unused")
    stock.addPerParticleParameter("also_unused")
    stock.addGlobalParameter("scale", .001)
    for atom in range(len(XYZ)):
        stock.addParticle([float(atom), -float(atom)])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        compare_modes(actual, reference)
        for force, ctx in ((stock, reference), (custom, actual)):
            force.setParticleParameters(5, [13., -7.])
            force.updateParametersInContext(ctx)
            ctx.setParameter("scale", .002)
            ctx.setPositions(XYZ[::-1])
        compare_modes(actual, reference)
        # Compaction does not remove the public parameter validation contract.
        custom.setParticleParameters(5, [float("nan"), -7.])
        with pytest.raises(Exception, match="finite FP32"):
            custom.updateParametersInContext(actual)
        compare_modes(actual, reference)


@pytest.mark.parametrize("transform,initial", [("sqrt(scale)", -1.), ("scale^2", 1e30)])
def test_derived_globals_preserve_inactive_domains_and_live_updates(plugin_dir, transform, initial):
    stock = mm.CustomNonbondedForce(
        f"select(active1+active2,select(mode,{transform},log(offset)),0)*r^2")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addPerParticleParameter("active")
    stock.addPerParticleParameter("unused")
    for name, value in (("mode", 1.), ("scale", initial), ("offset", -1.)):
        stock.addGlobalParameter(name, value)
    for atom in range(len(XYZ)):
        stock.addParticle([0., float(atom)])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference", True) as reference, context(custom, "CUDA", True) as actual:
        # A finite original global produces NaN or FP32 overflow, but every
        # pair selects zero. Hoisting must neither reject nor leak that value.
        compare_modes(actual, reference)
        for ctx in (reference, actual):
            ctx.setParameter("mode", 0.)
            ctx.setParameter("offset", 2.)
        for force, ctx in ((stock, reference), (custom, actual)):
            for atom in range(len(XYZ)):
                force.setParticleParameters(atom, [float(atom % 2), float(atom)])
            force.updateParametersInContext(ctx)
        compare_modes(actual, reference)
        for ctx in (reference, actual):
            ctx.setParameter("mode", 1.)
            ctx.setParameter("scale", .25)
            ctx.setParameter("offset", -1.)
            ctx.setPositions(XYZ[::-1])
        compare_modes(actual, reference)
        for value in (0., .81):
            for ctx in (reference, actual):
                ctx.setParameter("scale", value)
            compare_modes(actual, reference)


def test_derived_globals_preserve_exact_captured_exclusion_cancellation(plugin_dir, monkeypatch, capfd):
    from test_custom_pme import _capture_modes

    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    stock = mm.CustomNonbondedForce("p1*p2*(exp(-lambda)+sqrt(beta))*(r+r^2+r^3)")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addPerParticleParameter("p")
    stock.addGlobalParameter("lambda", .4)
    stock.addGlobalParameter("beta", .25)
    for atom in range(len(XYZ)):
        stock.addParticle([.1+.01*(atom % 5)])
        for other in range(atom):
            stock.addExclusion(other, atom)
    custom = nv.CustomNonbondedForce(stock)
    with context(custom, "CUDA", middle=True) as actual:
        _capture_modes(actual, capfd)
        for scale, beta in ((.4, .25), (-.2, 0.), (0., 4.)):
            actual.setParameter("lambda", scale)
            actual.setParameter("beta", beta)
            actual.setPositions(XYZ[::-1])
            for energy, forces in OUTPUT_MODES:
                observed = snapshot(actual, energy, forces)
                if energy:
                    assert observed[0] == 0.
                if forces:
                    np.testing.assert_array_equal(observed[1], 0.)
        assert "graph capture #" not in capfd.readouterr().err


def alchemical_force():
    # Both states differ, and dummy/REST2 flags vary independently. This exercises
    # the same one-particle charge transforms and conditional softcore potential
    # as an alchemical pair law with dummy atoms and REST2, at a modest scale
    # suitable for a reorder run.
    stock = mm.CustomNonbondedForce(
        "qeff1*qeff2/r+select(dummy1+dummy2,soft,hard);"
        "soft=vdwA*epsA*(vA^2-vA)+vdwB*epsB*(vB^2-vB);"
        "hard=epsA*(hA^2-hA);"
        "vA=sigA6/(.5*(1-vdwA)^2*sigA6+r^6);"
        "vB=sigB6/(.5*(1-vdwB)^2*sigB6+r^6);"
        "hA=sigA6/r^6;"
        "sigA6=((sigmaA1+sigmaA2)/2)^6;"
        "sigB6=((sigmaB1+sigmaB2)/2)^6;"
        "epsA=4*sqrt(epsilonA1*epsilonA2);"
        "epsB=4*sqrt(epsilonB1*epsilonB2);"
        "qeff1=(chargeA*qA1+chargeB*qB1+chargeC*(1-dummy1)*(qA1+qB1)/2)"
        "*sqrt(select(rest1,restScale,1));"
        "qeff2=(chargeA*qA2+chargeB*qB2+chargeC*(1-dummy2)*(qA2+qB2)/2)"
        "*sqrt(select(rest2,restScale,1))"
    )
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    for name in ("qA", "sigmaA", "epsilonA", "qB", "sigmaB", "epsilonB", "dummy", "rest"):
        stock.addPerParticleParameter(name)
    for name, value in (("chargeA", 1.), ("chargeB", 0.), ("chargeC", 0.),
                        ("restScale", 1.), ("vdwA", 1.), ("vdwB", 0.)):
        stock.addGlobalParameter(name, value)
    for atom in range(len(XYZ)):
        # Repeated parameter classes let OpenMM actually reorder equivalent
        # molecules; distinct A/B parameters still expose slot/order mistakes.
        group = atom % 4
        stock.addParticle([.2 if group % 2 else -.3, .38, .2,
                           -.1 if group % 2 else .25, .43, .15,
                           float(group == 0), float(group < 2)])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    return stock


def test_precomputed_parameters_follow_globals_updates_and_reordering(plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    stock = alchemical_force()
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference", True) as ref, context(custom, "CUDA", True) as actual:
        compare_modes(actual, ref)
        # No particle upload: every change must invalidate the precomputed
        # charge, including REST2's nonlinear dependence on a global parameter.
        for values in ((.35, .60, .05, .7, .5, .5), (0., 1., 0., .4, 0., 1.)):
            for ctx in (ref, actual):
                for name, value in zip(("chargeA", "chargeB", "chargeC", "restScale", "vdwA", "vdwB"), values):
                    ctx.setParameter(name, value)
            compare_modes(actual, ref)
        for force, ctx in ((stock, ref), (custom, actual)):
            force.setParticleParameters(5, [.4, .42, .3, -.2, .35, .25, 1., 1.])
            force.updateParametersInContext(ctx)
        compare_modes(actual, ref)

        # setPositions() alone does not establish that atom indices changed.
        # Require the plugin's observed order-change trace during integration,
        # then compare both implementations at the same resulting coordinates.
        moved = XYZ[np.random.default_rng(17).permutation(len(XYZ))]
        actual.setPositions(moved)
        actual.getIntegrator().setStepSize(1e-5)
        capfd.readouterr()
        actual.getIntegrator().step(270)
        assert "[CustomNonbondedForce] atom reorder #" in capfd.readouterr().err
        ref.setPositions(actual.getState(getPositions=True).getPositions())
        compare_modes(actual, ref)
        for ctx in (ref, actual):
            ctx.setParameter("chargeA", .4)
            ctx.setParameter("chargeB", .2)
            ctx.setParameter("restScale", .9)
        compare_modes(actual, ref)

        # A parameter upload after a real reorder must address original particle
        # indices, while a later position update retains the correct records.
        for force, ctx in ((stock, ref), (custom, actual)):
            force.setParticleParameters(19, [-.15, .4, .18, .3, .46, .2, 0., 1.])
            force.updateParametersInContext(ctx)
            ctx.setPositions(XYZ[::-1])
        compare_modes(actual, ref)


def test_nested_selection_signed_particle_powers_and_live_globals(plugin_dir):
    stock = mm.CustomNonbondedForce(
        "select(mode,select(step(r-threshold),sqrt(r-threshold),log(threshold-r)),r^2)"
        "*(u1*w2+u2*w1)+.125*(p1^4+p2^4)/(1+r^2);"
        "u1=select(flag1,sqrt(p1),p1^3);"
        "u2=select(flag2,sqrt(p2),p2^3);"
        "w1=exp(scale*p1)+sin(p1-offset);"
        "w2=exp(scale*p2)+sin(p2-offset)"
    )
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addPerParticleParameter("p")
    stock.addPerParticleParameter("flag")
    for name, value in (("mode", 0.), ("threshold", .8), ("scale", .4), ("offset", .25)):
        stock.addGlobalParameter(name, value)
    for atom in range(len(XYZ)):
        value = (-.5, 0., .25)[atom % 3]
        stock.addParticle([value, float(value > 0)])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        # The unselected radial branch takes sqrt/log of negative values;
        # particle transforms see negative and exactly zero integer-power bases.
        # u1*w2 and u2*w1 also require distinct per-side transforms and slots.
        for mode, threshold, scale, offset in ((0., .8, .4, .25), (1., .8, -.3, 0.),
                                                (-1., 1.1, .6, -.2), (0., .8, 0., .5)):
            for ctx in (ref, actual):
                for name, value in (("mode", mode), ("threshold", threshold),
                                    ("scale", scale), ("offset", offset)):
                    ctx.setParameter(name, value)
            compare_modes(actual, ref)
        for force, ctx in ((stock, ref), (custom, actual)):
            force.setParticleParameters(0, [.36, 1.])
            force.setParticleParameters(2, [-.4, 0.])
            force.updateParametersInContext(ctx)
            ctx.setParameter("mode", 1.)
            ctx.setPositions(XYZ[::-1])
        compare_modes(actual, ref)


def test_expensive_particle_primitives_preserve_signed_and_inactive_domains(plugin_dir):
    stock = mm.CustomNonbondedForce(
        "select(flag1*flag2,p1^.25*p2^.25,(p1*p2)^3)/(1+r^2)"
        "+.001*(sec(q1)*sec(q2)+csc(q1)*csc(q2)+cot(q1)*cot(q2))/(1+r)")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    for name in ("p", "flag", "q"):
        stock.addPerParticleParameter(name)
    for atom in range(len(XYZ)):
        p = (-.5, 0., .25)[atom % 3]
        q = (-.4, .3, .5)[atom % 3]
        stock.addParticle([p, float(p > 0.), q])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        # Fractional powers of negative inputs are stored as NaN, but only
        # positive-positive pairs select them. Integer powers retain signed
        # and zero inputs, and reciprocal trig transforms see both signs.
        compare_modes(actual, reference)
        for force, ctx in ((stock, reference), (custom, actual)):
            force.setParticleParameters(0, [.0625, 1., -.55])
            force.setParticleParameters(5, [-.36, 0., .45])
            force.updateParametersInContext(ctx)
            ctx.setPositions(XYZ[::-1])
        compare_modes(actual, reference)


def test_particle_preprocessing_ignores_radial_and_unused_global_changes(
        plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    stock = mm.CustomNonbondedForce(
        "u1*u2/(r+radial_scale);"
        "u1=particle_scale*sqrt(abs(p1));"
        "u2=particle_scale*sqrt(abs(p2))")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addPerParticleParameter("p")
    for name, value in (("particle_scale", .5), ("radial_scale", .2), ("unused", 1.)):
        stock.addGlobalParameter(name, value)
    for atom in range(len(XYZ)):
        stock.addParticle([(-1 if atom % 2 else 1)*(.1+.01*atom)])
    stock.addExclusion(0, 1)
    custom = nv.CustomNonbondedForce(stock)
    marker = "[cuPo expression] particle preprocessing #"
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        capfd.readouterr()
        compare(snapshot(actual), snapshot(reference))
        assert capfd.readouterr().err.count(marker) == 1
        for name, value, expected_runs in (("radial_scale", .4, 0),
                                           ("unused", 2., 0),
                                           ("particle_scale", .75, 1)):
            for ctx in (reference, actual):
                ctx.setParameter(name, value)
            for energy, forces in ((False, True), (True, False), (True, True)):
                compare(snapshot(actual, energy, forces), snapshot(reference, energy, forces))
            assert capfd.readouterr().err.count(marker) == expected_runs

        # Invalid unused globals still fail before uploading or preprocessing.
        actual.setParameter("unused", np.nan)
        with pytest.raises(Exception, match="finite FP32"):
            snapshot(actual)
        assert marker not in capfd.readouterr().err
        actual.setParameter("unused", 2.)
        compare(snapshot(actual), snapshot(reference))
        assert marker not in capfd.readouterr().err

        # Particle edits invalidate the transform independently of global keys.
        for force, ctx in ((stock, reference), (custom, actual)):
            force.setParticleParameters(5, [.9])
            force.updateParametersInContext(ctx)
        compare(snapshot(actual), snapshot(reference))
        assert capfd.readouterr().err.count(marker) == 1


def test_derived_globals_follow_captured_updates_and_actual_reordering(
        plugin_dir, monkeypatch, capfd):
    from test_custom_pme import _capture_modes
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    monkeypatch.setenv("CUPO_EXPRESSION_TRACE", "1")
    # Reuse the repeated particle classes that permit real OpenMM reordering.
    # Only two pairs are excluded: stale derived values cannot hide behind exact
    # cancellation, since the remaining interactions have nonzero forces.
    stock = alchemical_force()
    # Keep the coefficient next to r so particle extraction cannot absorb it
    # into a one-particle charge transform before global extraction runs.
    stock.setEnergyFunction("qA1*qA2/(r+(exp(-vdwA)+sqrt(restScale)))")
    custom = nv.CustomNonbondedForce(stock)
    marker = "[cuPo expression] global preprocessing #"
    with context(stock, "Reference", True, middle=True) as reference, \
            context(custom, "CUDA", True, middle=True) as actual:
        _capture_modes(actual, capfd)
        compare_modes(actual, reference)
        initial_force = snapshot(actual)[1].copy()
        assert np.linalg.norm(initial_force) > .01
        assert marker not in capfd.readouterr().err

        for ctx in (actual, reference):
            ctx.setParameter("vdwA", -.25)
            ctx.setParameter("restScale", .81)
        compare_modes(actual, reference)
        assert np.linalg.norm(snapshot(actual)[1]-initial_force) > .01
        trace = capfd.readouterr().err
        assert trace.count(marker) == 1
        assert "graph capture #" not in trace

        # Uploading a different original global must preserve the derived
        # suffix and must not rerun its kernel when that input is unrelated.
        for ctx in (actual, reference):
            ctx.setParameter("chargeA", .4)
        compare_modes(actual, reference)
        trace = capfd.readouterr().err
        assert marker not in trace
        assert "graph capture #" not in trace

        actual.setPositions(XYZ[np.random.default_rng(17).permutation(len(XYZ))])
        actual.getIntegrator().setStepSize(1e-5)
        capfd.readouterr()
        actual.getIntegrator().step(270)
        trace = capfd.readouterr().err
        assert "[CustomNonbondedForce] atom reorder #" in trace
        assert marker not in trace
        reference.setPositions(actual.getState(getPositions=True).getPositions())
        compare_modes(actual, reference)

        # A later update must reach every captured output mode after the input
        # particle order has changed, without capturing another pair graph.
        capfd.readouterr()
        for ctx in (actual, reference):
            ctx.setParameter("vdwA", .6)
            ctx.setParameter("restScale", 0.)
        compare_modes(actual, reference)
        trace = capfd.readouterr().err
        assert trace.count(marker) == 1
        assert "graph capture #" not in trace
