# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The caller expression compiler and generic runtime against stock Reference."""
import contextlib
import itertools

import numpy as np
import openmm as mm
from openmm import unit
import pytest
import nv_openmm_plugin as nv

XYZ = np.array(list(itertools.product((.2, .9, 1.6), repeat=3)))
FORCE_QUANTUM = float(np.float32(138.93545764498)) / 2**21
EXPRESSION = (
    "lambda*(a1*a2/r+sqrt(b1*b2)*(s/r)^12-c1*c2*(s/r)^6"
    "+.025*c1*c2*(s/r)^8);s=(s1+s2)/2"
)


def switched_expression(expression, switch=.8, cutoff=1.25):
    """Spell out a compact quintic switch inside the supplied expression."""
    energy, separator, definitions = expression.partition(";")
    return (f"({energy})*manual_switch;"
            "manual_switch=1-10*manual_switch_x^3+15*manual_switch_x^4-6*manual_switch_x^5;"
            f"manual_switch_x=min(1,max(0,(r-{switch})/({cutoff}-{switch})))"
            + (";"+definitions if separator else ""))


def source(expression=EXPRESSION, switched=False):
    force = mm.CustomNonbondedForce(expression)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.25)
    for name in ("a", "b", "c", "s"):
        force.addPerParticleParameter(name)
    force.addGlobalParameter("lambda", .8)
    for i in range(len(XYZ)):
        force.addParticle([(-1)**i*.3, .2+i*.001, .3, .45+i*.001])
    force.addExclusion(0, 1)
    force.addExclusion(2, 3)
    force.setUseSwitchingFunction(switched)
    force.setSwitchingDistance(.8)
    return force


@contextlib.contextmanager
def context(force, platform, inert_cutoff=False, middle=False):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4,0,0),mm.Vec3(0,4,0),mm.Vec3(0,0,4))
    system.addForce(force)
    if inert_cutoff:
        inert = mm.NonbondedForce()
        inert.setNonbondedMethod(inert.CutoffPeriodic)
        inert.setCutoffDistance(force.getCutoffDistance())
        inert.setUseDispersionCorrection(False)
        for _ in XYZ:
            inert.addParticle(0., .1, 0.)
        system.addForce(inert)
    # Middle velocities avoid hidden force work in energy-only graph checks.
    integrator = mm.LangevinMiddleIntegrator(0., 1., .001) if middle else mm.VerletIntegrator(.001)
    properties = {"Precision": "mixed"} if platform == "CUDA" else {}
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    ctx.setPositions(XYZ)
    try:
        yield ctx
    finally:
        del ctx, integrator


def snapshot(ctx, energy=True, forces=True):
    state = ctx.getState(getEnergy=energy, getForces=forces)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) if energy else None,
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        if forces else None,
    )


def compare(actual, expected):
    if expected[0] is not None:
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-5)
    if expected[1] is not None:
        # Any atom has at most N-1 partners. Add the worst-case half-quantum
        # rounding error per pair to the FP32 evaluator tolerance.
        force_atol = 3e-5 + .5 * FORCE_QUANTUM * (len(XYZ) - 1)
        np.testing.assert_allclose(actual[1], expected[1], rtol=4e-5, atol=force_atol)


@pytest.mark.parametrize("switching", ["none", "manual", "explicit"])
def test_parameters_globals_exclusions_and_output_modes(plugin_dir, switching):
    stock = source(switched=switching != "none")
    # Keep stock's automatic switch as an independent oracle for the explicit
    # expression used by cuPo.
    definition = source(
        expression=switched_expression(EXPRESSION) if switching == "manual" else EXPRESSION,
        switched=switching == "explicit")
    custom = nv.CustomNonbondedForce(definition)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for energy, forces in ((True, True), (True, False), (False, True)):
            compare(snapshot(actual, energy, forces), snapshot(ref, energy, forces))
        for lam in (0., .25, 1., .8):
            for ctx in (ref, actual):
                ctx.setParameter("lambda", lam)
            compare(snapshot(actual), snapshot(ref))
        for force, ctx in ((stock, ref), (custom, actual)):
            force.setParticleParameters(5, [.4, .3, .2, .5])
            force.updateParametersInContext(ctx)
            ctx.setPositions(XYZ + np.sin(np.arange(XYZ.size)).reshape(XYZ.shape)*.02)
        compare(snapshot(actual), snapshot(ref))


@pytest.mark.parametrize("excluded", [False, True])
def test_explicit_switch_boundaries_and_captured_exclusion_cancellation(
        plugin_dir, monkeypatch, capfd, excluded):
    from test_custom_pme import _capture_modes

    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    expression = "lambda*a1*a2*(1+r+r^2)"
    stock = source(expression, switched=True)
    # Only this pair contributes, making the switch derivative and endpoint
    # behavior visible without cancellation against unrelated pair forces.
    for index in range(len(XYZ)):
        stock.setParticleParameters(index, [float(index in (0, 2)), .2, .3, .45])
    if excluded:
        stock.addExclusion(0, 2)
    custom = nv.CustomNonbondedForce(stock)
    assert custom.getEnergyFunction() == expression
    assert custom.getUseSwitchingFunction()
    with context(stock, "Reference", middle=True) as reference, \
            context(custom, "CUDA", middle=True) as actual:
        _capture_modes(actual, capfd)
        for distance in (.79, .8, .81, 1., 1.24, 1.25, 1.26):
            positions = XYZ.copy()
            # Move an inactive atom out of the occupied cell before moving
            # the active partner in, keeping the small fixture's cell capacity.
            positions[1] = XYZ[2]
            positions[2] = positions[0] + [distance, 0., 0.]
            for ctx in (reference, actual):
                ctx.setPositions(positions)
            for energy, forces in ((True, False), (False, True), (True, True)):
                observed = snapshot(actual, energy, forces)
                compare(observed, snapshot(reference, energy, forces))
                if excluded or distance >= 1.25:
                    if energy:
                        assert observed[0] == 0.
                    if forces:
                        np.testing.assert_array_equal(observed[1], 0.)
        assert "graph capture #" not in capfd.readouterr().err


def test_three_parameter_records_and_live_updates(plugin_dir, monkeypatch):
    # Three FP32 parameters form a twelve-byte record. The forced feasible
    # layout selects inline records, exercising Lepton-generated generic loads
    # from local copies as well as updates to the sorted particle storage.
    for name, value in (
            ("CUPO_DSPACE_PIECE_CONFIG", "128-128-512-tc16"),
            ("CUPO_DSPACE_PIECE_THREADS", "128"),
            ("CUPO_DSPACE_CLEAVE_TOWER", "0"),
            ("CUPO_DSPACE_PIECE_SELECTOR", "auto"),
            ("CUPO_DSPACE_PAIR_KERNEL", "piece"),
            ("CUPO_DSPACE_AUTOTUNE", "0")):
        monkeypatch.setenv(name, value)
    stock = mm.CustomNonbondedForce(
        "lambda*(a1*a2*r^2-b1*b2*exp(-r)+c1*c2/(1+r^2))")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addGlobalParameter("lambda", .8)
    for name in ("a", "b", "c"):
        stock.addPerParticleParameter(name)
    for atom in range(len(XYZ)):
        stock.addParticle([(-1)**atom*.3, .2+.003*atom, .4-.002*atom])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for energy, forces in ((True, False), (False, True), (True, True)):
            compare(snapshot(actual, energy, forces), snapshot(ref, energy, forces))
        moved = XYZ[::-1] + np.sin(np.arange(XYZ.size)).reshape(XYZ.shape)*.02
        for force, ctx in ((stock, ref), (custom, actual)):
            force.setParticleParameters(5, [.7, -.25, .15])
            force.setParticleParameters(19, [-.4, .6, -.3])
            force.updateParametersInContext(ctx)
            ctx.setPositions(moved)
        for lam in (0., -.35, 1.1):
            for ctx in (ref, actual):
                ctx.setParameter("lambda", lam)
            for energy, forces in ((True, False), (False, True), (True, True)):
                compare(snapshot(actual, energy, forces), snapshot(ref, energy, forces))


def test_parameter_free_program(plugin_dir):
    stock = mm.CustomNonbondedForce("lambda*r^2")
    stock.addGlobalParameter("lambda", 1.)
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    for _ in XYZ:
        stock.addParticle([])
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for lam in (1., 0., -.4, 1.):
            for ctx in (ref, actual):
                ctx.setParameter("lambda", lam)
            compare(snapshot(actual), snapshot(ref))


def test_large_parameter_record_and_lambda_updates(plugin_dir):
    # Read both ends of a 260-byte record; unused fields must still contribute
    # to the stride when OpenMM updates parameters or the runtime sorts atoms.
    stock = mm.CustomNonbondedForce("lambda*(p01*p02+p641*p642)/(1+r^2)")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    stock.addGlobalParameter("lambda", .8)
    for index in range(65):
        stock.addPerParticleParameter(f"p{index}")
    for atom in range(len(XYZ)):
        stock.addParticle([.1 + .002*atom + .003*index for index in range(65)])
    stock.addExclusion(0, 1)
    stock.addExclusion(2, 3)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for lam in (.8, 0., -.25, 1.):
            for ctx in (ref, actual):
                ctx.setParameter("lambda", lam)
            compare(snapshot(actual), snapshot(ref))
        for force, ctx in ((stock, ref), (custom, actual)):
            parameters = list(force.getParticleParameters(5))
            parameters[0], parameters[64] = -.7, .9
            force.setParticleParameters(5, parameters)
            force.updateParametersInContext(ctx)
            ctx.setPositions(XYZ[::-1])
        for energy, forces in ((True, False), (False, True), (True, True)):
            compare(snapshot(actual, energy, forces), snapshot(ref, energy, forces))


def test_custom_records_and_exclusions_survive_real_reordering(plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    stock = source()
    # Repeated parameter classes permit OpenMM to permute equivalent molecules.
    for i in range(len(XYZ)):
        stock.setParticleParameters(i, [.2 if i % 2 else -.3, .2, .3, .45])
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference", True) as ref, context(custom, "CUDA", True) as actual:
        compare(snapshot(actual), snapshot(ref))
        for force, ctx in ((stock, ref), (custom, actual)):
            force.setParticleParameters(5, [.4, .25, .35, .5])
            force.updateParametersInContext(ctx)
        moved = XYZ[np.random.default_rng(17).permutation(len(XYZ))]
        actual.setPositions(moved)
        actual.getIntegrator().setStepSize(1e-5)
        capfd.readouterr()
        actual.getIntegrator().step(270)
        assert "[CustomNonbondedForce] atom reorder #" in capfd.readouterr().err
        ref.setPositions(actual.getState(getPositions=True).getPositions())
        compare(snapshot(actual), snapshot(ref))
        for ctx in (ref, actual):
            ctx.setParameter("lambda", .3)
        compare(snapshot(actual), snapshot(ref))


def test_globals_are_not_limited_to_packing_context_size(plugin_dir):
    stock = source("lambda*(g0+g1+g2+g3+g4)*a1*a2/r")
    for i in range(5):
        stock.addGlobalParameter(f"g{i}", .1*(i+1))
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for ctx in (ref, actual):
            ctx.setParameter("g4", .9)
        compare(snapshot(actual), snapshot(ref))


def test_wca_piecewise_program(plugin_dir):
    stock = source(
        "(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)*step(2^(1/6)*sigma-r);"
        "sigma=(s1+s2)/2;epsilon=sqrt(b1*b2)"
    )
    for i in range(len(XYZ)):
        stock.setParticleParameters(i, [.2,.5,.2,.65])
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        compare(snapshot(actual), snapshot(ref))


@pytest.mark.parametrize("powers", [(5, 6, 11, 12), (3, 7, 15, 16), (17, 31, 32, 64)])
def test_shared_integer_powers_at_zero_and_negative_bases(plugin_dir, powers):
    # Binary-exact separations make r-shift exactly zero for some pairs, while
    # other pairs see either sign. Reusing powers must not introduce division
    # by their base. Six live globals also exercise complete groups and tails.
    terms = "+".join(f"c{i}*x^{power}" for i, power in enumerate(powers))
    stock = mm.CustomNonbondedForce(f"lambda*({terms});x=r-shift")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    for name, value in (("lambda", .8), ("shift", .25),
                        ("c0", .5), ("c1", -.3), ("c2", .2), ("c3", -.1)):
        stock.addGlobalParameter(name, value)
    for _ in XYZ:
        stock.addParticle([])
    stock.addExclusion(0, 1)
    custom = nv.CustomNonbondedForce(stock)
    # Spread the grid over cells: this small system allocates eight slots per
    # cell. Retain exact .25 separations without exceeding that envelope.
    positions = np.array(list(itertools.product((0., .25, 1.5), repeat=3)))
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for ctx in (ref, actual):
            ctx.setPositions(positions)
        for shift, lam in ((.25, .8), (.5, -.35), (1., 0.), (.25, 1.1)):
            for ctx in (ref, actual):
                ctx.setParameter("shift", shift)
                ctx.setParameter("lambda", lam)
                ctx.setParameter("c0", .5 + shift)
                ctx.setParameter("c3", -.1 - shift)
            for energy, forces in ((True, False), (False, True), (True, True)):
                compare(snapshot(actual, energy, forces), snapshot(ref, energy, forces))
            force_only = snapshot(actual, False, True)[1]
            force_and_energy = snapshot(actual, True, True)[1]
            np.testing.assert_array_equal(force_only, force_and_energy)


@pytest.mark.parametrize("unsupported", ["nonperiodic", "no_cutoff", "derivatives"])
def test_unsupported_contract_is_explicit(plugin_dir, unsupported):
    stock = source()
    if unsupported == "nonperiodic":
        stock.setNonbondedMethod(stock.CutoffNonPeriodic)
    elif unsupported == "no_cutoff":
        stock.setNonbondedMethod(stock.NoCutoff)
    else:
        stock.addEnergyParameterDerivative("lambda")
    with pytest.raises(Exception, match="not supported|requires CutoffPeriodic"):
        nv.CustomNonbondedForce(stock)


def test_expression_and_topology_updates_require_reinitialize(plugin_dir):
    custom = nv.CustomNonbondedForce(source())
    with context(custom, "CUDA") as ctx:
        snapshot(ctx)
        custom.setEnergyFunction("r")
        with pytest.raises(Exception, match="reinitialize"):
            custom.updateParametersInContext(ctx)


def test_invalid_particle_update_preserves_existing_context(plugin_dir):
    stock = source()
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        before = snapshot(actual)
        # A valid edit before a later invalid record must not be partially
        # committed. Exercise both value and record-width validation.
        custom.setParticleParameters(0, [.4, .25, .35, .5])
        last = custom.getNumParticles()-1
        original = list(custom.getParticleParameters(last))
        for invalid, message in (([float("nan"), *original[1:]], "finite FP32"),
                                 (original[:-1], "wrong number")):
            custom.setParticleParameters(last, invalid)
            with pytest.raises(Exception, match=message):
                custom.updateParametersInContext(actual)
            unchanged = snapshot(actual)
            for observed, expected in zip(unchanged, before):
                np.testing.assert_array_equal(observed, expected)
        custom.setParticleParameters(last, original)
        custom.updateParametersInContext(actual)
        stock.setParticleParameters(0, [.4, .25, .35, .5])
        stock.updateParametersInContext(ref)
        compare(snapshot(actual), snapshot(ref))


def test_unsupported_tabulated_function_preserves_source(plugin_dir):
    stock = source("f(r)")
    stock.addTabulatedFunction("f", mm.Continuous1DFunction([0., 1., 4.], 0., 2.))
    with pytest.raises(Exception, match="tabulated functions are not supported"):
        nv.CustomNonbondedForce(stock)
    # The rejected adapter must leave the original force and its owned table usable.
    with context(stock, "Reference") as ref:
        energy, forces = snapshot(ref)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))


def test_invalid_box_reports_exception(plugin_dir):
    custom = nv.CustomNonbondedForce(source())
    with context(custom, "CUDA") as ctx:
        ctx.setPeriodicBoxVectors(mm.Vec3(3,0,0),mm.Vec3(0,3,0),mm.Vec3(0,0,3))
        # Shared cell-list validation reports the required stencil dimensions.
        with pytest.raises(Exception, match="cell-list box is too small"):
            snapshot(ctx)


def test_manual_lj_switch_preserves_coulomb_and_live_lambda(plugin_dir):
    lj = switched_expression(
        "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
        "sigma=(sigma1+sigma2)/2;epsilon=sqrt(epsilon1*epsilon2)")
    radial, definitions = lj.split(";", 1)
    force = nv.NonbondedForce(f"q1*q2/r+lambda*({radial});{definitions}")
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.25)
    force.addGlobalParameter("lambda", .7)
    for name in ("q", "sigma", "epsilon"):
        force.addPerParticleParameter(name)
    force.addParticle([.3, .45, .4])
    force.addParticle([-.2, .55, .9])
    system = mm.System()
    for _ in range(2):
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(4, 0, 0), mm.Vec3(0, 4, 0), mm.Vec3(0, 0, 4))
    system.addForce(force)
    integrator = mm.LangevinMiddleIntegrator(0., 1., .001)
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                     {"Precision": "mixed"})
    try:
        # An independent analytic oracle checks the product-rule derivative,
        # switch endpoints, and the unswitched Coulomb part at lambda=0.
        for lam in (0., .4, 1.):
            ctx.setParameter("lambda", lam)
            for r in (.65, .799, .8, .801, 1.05, 1.249, 1.251):
                ctx.setPositions([[.25, .25, .25], [.25+r, .25, .25]])
                x = min(1., max(0., (r-.8)/.45))
                switch = 1-10*x**3+15*x**4-6*x**5
                derivative = (-30*x**2+60*x**3-30*x**4)/.45
                sr6 = (.5/r)**6
                lj_energy = 4*.6*(sr6**2-sr6)
                lj_gradient = 24*.6*(sr6-2*sr6**2)/r
                expected_energy = -.06/r + lam*switch*lj_energy
                expected_gradient = .06/r**2 + lam*(switch*lj_gradient+derivative*lj_energy)
                if r >= 1.25:
                    expected_energy = expected_gradient = 0.
                expected_forces = np.array([[expected_gradient, 0., 0.],
                                            [-expected_gradient, 0., 0.]])
                for energy, forces in ((True, False), (False, True), (True, True)):
                    observed = snapshot(ctx, energy, forces)
                    if energy:
                        np.testing.assert_allclose(observed[0], expected_energy,
                                                   rtol=2e-5, atol=2e-6)
                    if forces:
                        np.testing.assert_allclose(observed[1], expected_forces,
                                                   rtol=4e-5, atol=FORCE_QUANTUM)
    finally:
        del ctx, integrator


@pytest.mark.parametrize("config", ["352-384-2048-tc16", "416-384-2432-packed-tc16",
                                    "256-304-1040-tc16",
                                    "352-360-2320-tc16"])
@pytest.mark.parametrize("excluded", [False, True])
@pytest.mark.parametrize("skip_zero", ["0", "1"])
def test_zero_parameters_preserve_custom_force(plugin_dir, monkeypatch, config, excluded, skip_zero):
    monkeypatch.setenv("CUPO_DSPACE_PIECE_CONFIG", config)
    monkeypatch.setenv("CUPO_DSPACE_SKIP_ZERO_PAIRS", skip_zero)
    stock = mm.CustomNonbondedForce("r")
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setCutoffDistance(1.25)
    for name in ("q", "sigma", "epsilon"):
        stock.addPerParticleParameter(name)
    for _ in XYZ:
        stock.addParticle([0., 0., 0.])
    if excluded:
        stock.addExclusion(0, 1)
    custom = nv.CustomNonbondedForce(stock)
    with context(stock, "Reference") as ref, context(custom, "CUDA") as actual:
        for energy, forces in [(False, True), (True, False), (True, True)]:
            compare(snapshot(actual, energy, forces), snapshot(ref, energy, forces))
