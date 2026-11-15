# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A full custom cutoff expression shares one traversal with split Coulomb."""

import contextlib
import ctypes as C
import itertools

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

from test_custom_nonbonded import switched_expression

BOX = 4.2
CUTOFF = 1.1
XYZ = np.array(list(itertools.product((.15, .95, 1.75, 2.55, 3.35), repeat=3)))
MODES = ((False, True), (True, False), (True, True))
KJ = unit.kilojoule_per_mole
NM = unit.nanometer


def _box(length):
    return (mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0), mm.Vec3(0, 0, length))


def _source(periodic=True, tail=True):
    base = mm.NonbondedForce()
    base.setNonbondedMethod(base.PME)
    base.setCutoffDistance(CUTOFF)
    base.setEwaldErrorTolerance(5e-4)
    base.setExceptionsUsePeriodicBoundaryConditions(periodic)
    base.setUseDispersionCorrection(tail)
    base.setUseSwitchingFunction(True)
    base.setSwitchingDistance(.85)
    for name, default in (("lambda_q", .2), ("lambda_s", .35), ("lambda_14", .1)):
        base.addGlobalParameter(name, default)
    for atom in range(len(XYZ)):
        base.addParticle((-1)**atom*(.11+.002*(atom % 7)), .28+.003*(atom % 5), .14+.01*(atom % 3))
        if atom % 3 == 0:
            base.addParticleParameterOffset("lambda_q", atom, .07-.01*(atom % 5), 0., 0.)
        if atom % 5 == 0:
            base.addParticleParameterOffset("lambda_s", atom, 0., .025, .08)
    # First pair crosses the periodic boundary; the second is within one image.
    # The third starts as an exclusion but has a nonzero live replacement.
    for first, second, qprod, sigma, epsilon in ((0, 4, -.012, .31, .09),
                                                (1, 2, .008, .29, .12),
                                                (10, 11, 0., .27, 0.)):
        index = base.addException(first, second, qprod, sigma, epsilon)
        base.addExceptionParameterOffset("lambda_14", index, .023, .018, .065)
        base.addExceptionParameterOffset("lambda_q", index, -.009, 0., 0.)

    base.setForceGroup(3)
    return base


def _pin_reference_mesh(base, length=BOX):
    # Reference uses the unrounded PME estimate while cuFFT rounds to a legal
    # transform size. Pin the stock reference AFTER converting the plugin copy
    # so this test measures live-parameter correctness, not different mesh errors.
    # Resolve the core symbols through the plugin's loaded CUDA dependency;
    # developer builds may link libcupo.so outside the package tree.
    lib = C.CDLL(str(nv._cuda_platform_dir(nv._PLUGIN_DIR) / "libOpenMMNvidiaCUDA.so"))
    lib.cupo_abi_version.restype = C.c_int
    assert lib.cupo_abi_version() == 32
    lib.cupo_splitting_plan_create_ewald.argtypes = [
        C.c_double, C.c_double, C.POINTER(C.c_void_p)]
    lib.cupo_splitting_plan_create_ewald.restype = C.c_int
    lib.cupo_plan_grid.argtypes = [
        C.c_void_p, C.POINTER(C.c_double), C.c_int, C.POINTER(C.c_int)]
    lib.cupo_plan_grid.restype = C.c_int
    lib.cupo_splitting_plan_destroy.argtypes = [C.c_void_p]
    lib.cupo_splitting_plan_destroy.restype = C.c_int
    plan = C.c_void_p()
    assert lib.cupo_splitting_plan_create_ewald(CUTOFF, 138.93545764498, C.byref(plan)) == 0
    dims = (C.c_int*3)()
    try:
        edges = (C.c_double*3)(length, length, length)
        assert lib.cupo_plan_grid(plan, edges, 0, dims) == 0
    finally:
        assert lib.cupo_splitting_plan_destroy(plan) == 0
    alpha = np.sqrt(-np.log(2*base.getEwaldErrorTolerance())) / CUTOFF
    base.setPMEParameters(alpha, *dims)


def _system(forces):
    system = mm.System()
    for _ in XYZ:
        system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(*_box(BOX))
    for force in forces:
        system.addForce(force)
    return system


@contextlib.contextmanager
def _context(system, platform):
    # Middle velocities keep energy-only evaluations free of hidden force work.
    integrator = mm.LangevinMiddleIntegrator(0., 1., .001)
    props = {"Precision": "mixed"} if platform == "CUDA" else {}
    ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), props)
    ctx.setPositions(XYZ)
    try:
        yield ctx
    finally:
        del ctx, integrator


def _snapshot(ctx, mode):
    energy, forces = mode
    state = ctx.getState(getEnergy=energy, getForces=forces)
    return (state.getPotentialEnergy().value_in_unit(KJ) if energy else None,
            state.getForces(asNumpy=True).value_in_unit(KJ/NM) if forces else None)


def _capture_modes(ctx, capfd):
    capfd.readouterr()
    for _ in range(53):
        _snapshot(ctx, MODES[0])
    for mode in MODES[1:]:
        for _ in range(3):
            _snapshot(ctx, mode)
    trace = capfd.readouterr().err
    for energy, forces in MODES:
        marker = f"(energy={int(energy)} forces={int(forces)})"
        captures = [line for line in trace.splitlines()
                    if "[nv-plugin cupo] graph capture #" in line and marker in line]
        assert len(captures) == 1, trace


def _clone(force):
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force))


def _definitions(periodic, switched=False, scenario="abfe"):
    base = _source(periodic, tail=False)
    base.setUseSwitchingFunction(False)
    # These ordinary sigma/epsilon values and their offsets are intentionally
    # nonzero. They must be replaced, while exception sigma/epsilon survive.
    if scenario == "abfe":
        selection = ""
        coupling = (
            ";coupling=1-(1-lambda_s)*cross;"
            "cross=region1+region2-2*region1*region2"
        )
        names = ("sigma", "epsilon", "region")
    else:
        selection = ";allowed=1-old1*new2-new1*old2"
        coupling = (
            ";coupling=1-lambda_s*oldcross-(1-lambda_s)*newcross;"
            "oldcross=old1+old2-2*old1*old2;newcross=new1+new2-2*new1*new2"
        )
        names = ("sigma", "epsilon", "old", "new")
    energy = "select(coupling,4*coupling*epsilon*x*(x-1),0)"
    if selection:
        energy = f"select(allowed,{energy},0)"
    custom = mm.CustomNonbondedForce(
        energy + ";x=1/(0.5*(1-coupling)+(r/sigma)^6);"
        "sigma=0.5*(sigma1+sigma2);epsilon=sqrt(epsilon1*epsilon2)" + coupling + selection
    )
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(CUTOFF)
    if switched:
        custom.setEnergyFunction(switched_expression(custom.getEnergyFunction(), .85, CUTOFF))
    custom.setUseLongRangeCorrection(False)
    custom.addGlobalParameter("lambda_s", .35)
    for name in names:
        custom.addPerParticleParameter(name)
    for atom in range(len(XYZ)):
        flags = [float(atom % 4 == 0)] if scenario == "abfe" else [float(atom % 4 == i) for i in (1, 2)]
        custom.addParticle([.30+.002*(atom % 4), .20+.01*(atom % 3), *flags])
    # Exact exclusions are necessary: both the injected Coulomb pair term and
    # the user's expression must omit precisely the Standard exception pairs.
    for index in reversed(range(base.getNumExceptions())):
        first, second, *_ = base.getExceptionParameters(index)
        custom.addExclusion(second, first)  # Reversed order is still the same set.
    custom.setForceGroup(base.getForceGroup())
    return base, custom


def _coulomb_only(base):
    reference = _clone(base)
    for atom in range(reference.getNumParticles()):
        charge, sigma, epsilon = reference.getParticleParameters(atom)
        reference.setParticleParameters(atom, charge, sigma, epsilon*0)
    for index in range(reference.getNumParticleParameterOffsets()):
        name, atom, charge, sigma, epsilon = reference.getParticleParameterOffset(index)
        reference.setParticleParameterOffset(index, name, atom, charge, 0., 0.)
    return reference


def _single(base, custom, method):
    force = nv.NonbondedForce(base, 1.15, getattr(nv.NonbondedForce, method))
    force.setCutoffInteraction(custom)
    return force


def _reference_forces(base, custom, method, length=BOX):
    coulomb = _coulomb_only(base)
    if method == "PME":
        # Independent stock forces are the physical oracle for the fused path.
        _pin_reference_mesh(coulomb, length)
        return [coulomb, custom], "Reference"
    # Stock has no Prolate implementation. Separate cuPo forces retain the
    # same splitting while giving Coulomb and sterics independent traversals.
    return [nv.NonbondedForce(coulomb, 1.15, nv.NonbondedForce.Prolate),
            nv.NonbondedForce(custom, 1.15)], "CUDA"


def _compare_snapshot(observed, reference):
    if reference[0] is not None:
        np.testing.assert_allclose(observed[0], reference[0], rtol=3e-5, atol=1e-3)
    if reference[1] is not None:
        # Joining Coulomb and sterics changes pair rounding, but must not
        # change the physical selection, exclusions or charge corrections.
        np.testing.assert_allclose(observed[1], reference[1], rtol=3e-4, atol=.006)


def _compare(actual, expected, first=MODES[0]):
    for mode in (first, *(item for item in MODES if item != first)):
        _compare_snapshot(_snapshot(actual, mode), _snapshot(expected, mode))


def _inert_cutoff_force():
    # Enable OpenMM's real reorder machinery: the plugin's private cell list
    # does not register a cutoff consumer with OpenMM NonbondedUtilities.
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(CUTOFF)
    force.setUseDispersionCorrection(False)
    for _ in XYZ:
        force.addParticle(0., .1, 0.)
    return force


def _repeated_classes(base, custom):
    # Reordering only permutes equivalent molecules. Four repeated, distinct
    # particle classes exercise real permutations while retaining differing
    # charges, expression records and their association with system atoms.
    repeated = mm.NonbondedForce()
    repeated.setNonbondedMethod(repeated.PME)
    repeated.setCutoffDistance(CUTOFF)
    repeated.setEwaldErrorTolerance(5e-4)
    repeated.setExceptionsUsePeriodicBoundaryConditions(base.getExceptionsUsePeriodicBoundaryConditions())
    repeated.setUseDispersionCorrection(False)
    repeated.setForceGroup(base.getForceGroup())
    for index in range(base.getNumGlobalParameters()):
        repeated.addGlobalParameter(base.getGlobalParameterName(index), base.getGlobalParameterDefaultValue(index))
    for atom in range(len(XYZ)):
        group = atom % 4
        repeated.addParticle(.12-.06*group, .28+.01*group, .14+.02*group)
        # Current ForceInfo conservatively pins every atom carrying an
        # offset. Keep one live-charge atom while allowing the others to move.
        if atom == 0:
            repeated.addParticleParameterOffset("lambda_q", atom, .02, 0., 0.)
        record = list(custom.getParticleParameters(atom))
        record[:2] = [.30+.015*group, .15+.02*group]
        custom.setParticleParameters(atom, record)
    for index in range(base.getNumExceptions()):
        repeated.addException(*base.getExceptionParameters(index))
    for index in range(base.getNumExceptionParameterOffsets()):
        repeated.addExceptionParameterOffset(*base.getExceptionParameterOffset(index))
    return repeated


@pytest.mark.parametrize("method,periodic", [("PME", True), ("Prolate", True)])
@pytest.mark.parametrize("switched", [False, True])
@pytest.mark.parametrize("scenario", ["abfe", "rbfe"])
def test_full_expression_matches_reference_with_live_charges_exceptions_and_switch(
        plugin_dir, method, periodic, switched, scenario):
    base, custom = _definitions(periodic, switched, scenario)
    combined = _single(base, custom, method)
    reference_forces, platform = _reference_forces(base, custom, method)
    with _context(_system([combined]), "CUDA") as actual, \
            _context(_system(reference_forces), platform) as reference:
        for mode, lam, charge, exception in zip(MODES, (0., .5, 1.), (.8, -.15, .45), (.7, .05, .3)):
            for ctx in (actual, reference):
                ctx.setParameter("lambda_s", lam)
                ctx.setParameter("lambda_q", charge)
                ctx.setParameter("lambda_14", exception)
            _compare(actual, reference, mode)
        # Shift one member of the cross-boundary exception by a whole box.
        # Periodic physical replacements are image invariant.
        shifted = XYZ.copy()
        shifted[4, 2] -= BOX
        for ctx in (actual, reference):
            ctx.setPositions(shifted)
        _compare(actual, reference)


def test_full_expression_captured_parameter_updates_reorders_and_box_changes(
        plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    base, custom = _definitions(True, switched=True, scenario="rbfe")
    base = _repeated_classes(base, custom)
    combined = _single(base, custom, "PME")
    reference_base = _coulomb_only(base)
    _pin_reference_mesh(reference_base)
    with _context(_system([combined, _inert_cutoff_force()]), "CUDA") as actual, \
            _context(_system([reference_base, custom, _inert_cutoff_force()]), "Reference") as reference:
        _capture_modes(actual, capfd)
        _compare(actual, reference)
        for ctx in (actual, reference):
            ctx.setParameter("lambda_q", .63)
            ctx.setParameter("lambda_s", .72)
            ctx.setParameter("lambda_14", .84)
        # Live charge, independent 1-4 parameters, and expression records must
        # all refresh around the current lambda rather than their defaults.
        for force, ctx in ((combined, actual), (reference_base, reference)):
            q, sigma, epsilon = force.getParticleParameters(5)
            force.setParticleParameters(5, .31, sigma, epsilon)
            force.setParticleParameterOffset(0, "lambda_q", 0, .12, 0., 0.)
            force.setExceptionParameters(1, 1, 2, -.013, .30, .14)
            force.setExceptionParameterOffset(0, "lambda_14", 0, .037, .021, .095)
        record = list(custom.getParticleParameters(19))
        record[1] *= 1.7
        record[2:] = [1., 0.]
        custom.setParticleParameters(19, record)
        combined.setCutoffInteraction(custom)
        combined.updateParametersInContext(actual)
        reference_base.updateParametersInContext(reference)
        custom.updateParametersInContext(reference)
        _compare(actual, reference, MODES[1])

        # Observe the FIRST captured output after an actual atom permutation
        # in every output mode. Re-evaluating first could repair stale payloads.
        rng = np.random.default_rng(19)
        capfd.readouterr()
        for index, mode in enumerate(MODES):
            for ctx in (actual, reference):
                ctx.setParameter("lambda_q", (.1, .5, .9)[index])
                ctx.setParameter("lambda_s", (.2, .6, .8)[index])
            requested = XYZ[rng.permutation(len(XYZ))] + np.array([BOX, -BOX, BOX])
            for _ in range(251):
                actual.setPositions(requested)
                observed = _snapshot(actual, mode)
                trace = capfd.readouterr().err
                assert "graph capture #" not in trace, trace
                if "[CustomNonbondedForce] atom reorder #" in trace:
                    assert "[nv-plugin cupo] atom reorder #" in trace, trace
                    break
            else:
                pytest.fail(f"no nontrivial Standard/Custom reorder after capture for mode {mode}")
            # Read the represented/wrapped inputs only after retaining that
            # first output; then give precisely those coordinates to Reference.
            positions = actual.getState(getPositions=True).getPositions(asNumpy=True)
            reference.setPositions(positions)
            _compare_snapshot(observed, _snapshot(reference, mode))
            _compare(actual, reference, mode)
        # Upload native and opaque records together AFTER a proven nontrivial
        # permutation. Both are indexed by logical atoms at the public boundary.
        for force in (combined, reference_base):
            _, sigma, epsilon = force.getParticleParameters(37)
            force.setParticleParameters(37, -.22, sigma, epsilon)
            force.setParticleParameterOffset(0, "lambda_q", 0, -.08, 0., 0.)
        record = list(custom.getParticleParameters(19))
        record[1] *= 1.35
        record[2:] = [0., 1.]
        custom.setParticleParameters(19, record)
        combined.setCutoffInteraction(custom)
        combined.updateParametersInContext(actual)
        reference_base.updateParametersInContext(reference)
        custom.updateParametersInContext(reference)
        _compare(actual, reference, MODES[0])
        for length, mode in zip((4.35, 4.65, BOX), MODES):
            for ctx in (actual, reference):
                ctx.setPeriodicBoxVectors(*_box(length))
                ctx.setPositions(positions*(length/BOX))
            _compare(actual, reference, mode)


def test_full_expression_triclinic_live_parameters_and_volume(plugin_dir):
    base, custom = _definitions(True, switched=True, scenario="rbfe")
    combined = _single(base, custom, "PME")
    reference_base = _coulomb_only(base)
    _pin_reference_mesh(reference_base)
    systems = (_system([combined]), _system([reference_base, custom]))

    def box(length):
        return (mm.Vec3(length, 0, 0), mm.Vec3(.35, length, 0), mm.Vec3(-.25, .3, length))

    for system in systems:
        system.setDefaultPeriodicBoxVectors(*box(BOX))
    with _context(systems[0], "CUDA") as actual, _context(systems[1], "Reference") as reference:
        for length, lam, mode in zip((BOX, 4.35, 4.65), (.2, .8, .4), MODES):
            vectors = box(length)
            moved = (XYZ/BOX)@np.asarray(vectors)
            for ctx in (actual, reference):
                ctx.setPeriodicBoxVectors(*vectors)
                ctx.setPositions(moved)
                ctx.setParameter("lambda_q", lam)
                ctx.setParameter("lambda_s", 1-lam)
                ctx.setParameter("lambda_14", .3+lam/2)
            _compare(actual, reference, mode)


@pytest.mark.parametrize("expression", ["0", "amplitude/(1+r^2)", "amplitude*shape; shape=r; r=2"])
def test_full_expression_without_particle_fields_keeps_coulomb(plugin_dir, expression):
    base, _ = _definitions(True)
    custom = mm.CustomNonbondedForce(expression)
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(CUTOFF)
    custom.setForceGroup(base.getForceGroup())
    custom.addGlobalParameter("amplitude", .47)
    # The local r=2 definition is valid OpenMM syntax. It applies only to the
    # supplied cutoff expression; managed Coulomb must still use pair distance.
    # An unused global must not collide with the plugin's single-particle
    # weight binding, even when its name never appears in the cutoff expression.
    custom.addGlobalParameter("__cupo_managed_charge", .3)
    for _ in XYZ:
        custom.addParticle([])
    for index in range(base.getNumExceptions()):
        first, second, *_ = base.getExceptionParameters(index)
        custom.addExclusion(first, second)
    combined = _single(base, custom, "PME")
    reference_base = _coulomb_only(base)
    _pin_reference_mesh(reference_base)
    with _context(_system([combined]), "CUDA") as actual, \
            _context(_system([reference_base, custom]), "Reference") as reference:
        for mode, charge, amplitude in zip(MODES, (0., .8, -.2), (.47, 0., .81)):
            for ctx in (actual, reference):
                ctx.setParameter("lambda_q", charge)
                ctx.setParameter("amplitude", amplitude)
            _compare(actual, reference, mode)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_full_expression_corrects_far_exceptions(plugin_dir, method):
    # Five cells per dimension permit a genuine failed stencil lookup, unlike
    # three-cell boxes where every periodic cell is a neighbor. Plant one pair
    # inside the cutoff, one outside it in adjacent cells, and one two cells away.
    length = 6.
    positions = np.array([
        [.25, .25, .25], [.80, .30, .25],
        [2.30, .40, .30], [3.50, .45, .30],
        [.35, 2.60, .40], [2.95, 2.60, .40],
        [.40, .75, .35], [3.10, 2.95, .45],
    ])
    exception_pairs = ((1, 0), (2, 3), (5, 4))
    cells_per_edge = int(length/CUTOFF)
    cells = np.floor(positions/(length/cells_per_edge)).astype(int)
    distances, cell_distances = [], []
    for first, second in exception_pairs:
        delta = positions[second]-positions[first]
        delta -= length*np.rint(delta/length)
        distances.append(np.linalg.norm(delta))
        cell_delta = np.abs(cells[second]-cells[first])
        cell_distances.append(np.max(np.minimum(cell_delta, cells_per_edge-cell_delta)))
    assert distances[0] < CUTOFF < distances[1] < distances[2]
    assert cell_distances == [0, 1, 2]

    base = mm.NonbondedForce()
    base.setNonbondedMethod(base.PME)
    base.setCutoffDistance(CUTOFF)
    base.setEwaldErrorTolerance(5e-4)
    base.setUseDispersionCorrection(False)
    base.addGlobalParameter("lambda_q", .2)
    base.addGlobalParameter("lambda_14", .1)
    for atom, charge in enumerate((.30, -.20, .40, -.35, .15, -.25, .18, -.09)):
        base.addParticle(charge, .29+.002*atom, .14+.01*atom)
    base.addParticleParameterOffset("lambda_q", 0, .08, 0., 0.)
    base.addParticleParameterOffset("lambda_q", 4, -.11, 0., 0.)
    for index, (first, second) in enumerate(exception_pairs):
        base.addException(first, second, -.012+.007*index, .31, .12)
        base.addExceptionParameterOffset("lambda_14", index, .017, .011, .06)

    custom = mm.CustomNonbondedForce("lambda_s*a1*a2/(1+r^2)")
    custom.addGlobalParameter("lambda_s", .35)
    custom.addPerParticleParameter("a")
    for atom in range(len(positions)):
        custom.addParticle([.4+.03*atom])
    for first, second in reversed(exception_pairs):
        custom.addExclusion(second, first)
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(CUTOFF)
    custom.setUseLongRangeCorrection(False)
    combined = _single(base, custom, method)
    reference_forces, platform = _reference_forces(base, custom, method, length)

    @contextlib.contextmanager
    def context(forces, platform):
        system = mm.System()
        for _ in positions:
            system.addParticle(39.9)
        system.setDefaultPeriodicBoxVectors(*_box(length))
        for force in forces:
            system.addForce(force)
        integrator = mm.LangevinMiddleIntegrator(0., 1., .001)
        props = {"Precision": "mixed"} if platform == "CUDA" else {}
        ctx = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), props)
        ctx.setPositions(positions)
        try:
            yield ctx
        finally:
            del ctx, integrator

    with context([combined], "CUDA") as actual, context(reference_forces, platform) as reference:
        for mode, charge, replacement in zip(MODES, (0., .6, 1.), (.1, .8, .3)):
            for ctx in (actual, reference):
                ctx.setParameter("lambda_q", charge)
                ctx.setParameter("lambda_14", replacement)
                ctx.setParameter("lambda_s", .2+.5*charge)
            _compare(actual, reference, mode)
        # Update both far original-charge products and the independent far 1-4
        # replacement after the sparse correction records have been allocated.
        for force, ctx in ((combined, actual), (reference_forces[0], reference)):
            charge, sigma, epsilon = force.getParticleParameters(5)
            force.setParticleParameters(5, .19, sigma, epsilon)
            force.setParticleParameterOffset(1, "lambda_q", 4, -.16, 0., 0.)
            force.setExceptionParameters(2, 5, 4, -.04, .33, .21)
            force.setExceptionParameterOffset(2, "lambda_14", 2, .025, .015, .08)
            force.updateParametersInContext(ctx)
        _compare(actual, reference, MODES[1])


def test_full_expression_long_range_correction_uses_only_custom_sterics(plugin_dir):
    # Keep a nonzero beyond-cutoff tail: a compact manual switch would make
    # the literal expression's long-range integral identically zero.
    base, custom = _definitions(True, switched=False, scenario="abfe")
    assert not base.getUseDispersionCorrection()
    custom.setUseLongRangeCorrection(True)
    combined = _single(base, custom, "PME")
    reference_base = _coulomb_only(base)
    _pin_reference_mesh(reference_base)
    with _context(_system([combined]), "CUDA") as actual, \
            _context(_system([reference_base, custom]), "Reference") as reference:
        for edited in (False, True):
            if edited:
                # Keep the current lambda while refreshing both the pair
                # records and the class averages used by the steric tail.
                record = list(custom.getParticleParameters(19))
                record[:2] = [.39, .52]
                custom.setParticleParameters(19, record)
                combined.setCutoffInteraction(custom)
                combined.updateParametersInContext(actual)
                custom.updateParametersInContext(reference)
                _compare(actual, reference, MODES[1])
            for mode, lam, charge in zip(MODES, (0., .4, 1.), (.65, -.2, .3)):
                for ctx in (actual, reference):
                    ctx.setParameter("lambda_s", lam)
                    ctx.setParameter("lambda_q", charge)
                _compare(actual, reference, mode)
            # The independent stock oracle integrates just the supplied LJ
            # expression. Folding injected Coulomb into that tail would either
            # produce a divergent integral or a spurious charge-dependent term.
        combined.setGlobalParameterDefaultValue(1, .6)  # lambda_s in the native force
        custom.setGlobalParameterDefaultValue(0, .6)
        combined.updateParametersInContext(actual)
        custom.updateParametersInContext(reference)
        assert actual.getParameter("lambda_s") == reference.getParameter("lambda_s") == 1.
        _compare(actual, reference, MODES[1])


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_explicit_cutoff_switch_preserves_coulomb_exceptions_and_native_tail(plugin_dir, method):
    base = _source(tail=True)
    sigma, epsilon = .3, .2
    for atom in range(base.getNumParticles()):
        charge, _, _ = base.getParticleParameters(atom)
        base.setParticleParameters(atom, charge, sigma, epsilon)
    for index in range(base.getNumParticleParameterOffsets()):
        name, atom, charge, _, _ = base.getParticleParameterOffset(index)
        base.setParticleParameterOffset(index, name, atom, charge, 0., 0.)

    # Keep an intermediate definition: adding managed bare Coulomb to the full
    # compiler input must leave the semicolon-defined expression scope intact.
    custom = mm.CustomNonbondedForce("amplitude*a1*a2*shape; shape=1/(1+r^6)")
    custom.setNonbondedMethod(custom.CutoffPeriodic)
    custom.setCutoffDistance(CUTOFF)
    custom.setUseSwitchingFunction(True)
    custom.setSwitchingDistance(.7)  # Distinct from the retained native tail's .85.
    custom.setUseLongRangeCorrection(True)
    custom.addGlobalParameter("amplitude", .6)
    custom.addPerParticleParameter("a")
    custom.setForceGroup(base.getForceGroup())
    for atom in range(len(XYZ)):
        custom.addParticle([.2+.01*(atom % 5)])
    for index in range(base.getNumExceptions()):
        first, second, *_ = base.getExceptionParameters(index)
        custom.addExclusion(first, second)
    combined = nv.NonbondedForce(base, 1.15, getattr(nv.NonbondedForce, method))
    combined.setCutoffInteraction(custom, preserveDispersionCorrection=True)
    assert combined.getUseSwitchingFunction()
    assert combined.getSwitchingDistance() == .7
    assert combined.getCutoffExpression().getEnergyFunction() == custom.getEnergyFunction()
    reference_forces, reference_platform = _reference_forces(base, custom, method)

    # The independent Custom force supplies its switched-expression correction.
    # The retained native tail is independently known for identical sigma/epsilon
    # records. Its original .85 switching interval must survive replacement by
    # an expression whose own switch starts at .7. Exclusions and charges do not
    # enter this mean-field coefficient.
    points, weights = np.polynomial.legendre.leggauss(64)
    x = (points+1)/2
    r = .85+(CUTOFF-.85)*x
    removed = x**3*(10-15*x+6*x*x)
    lj = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
    deficit = (CUTOFF-.85)/2*np.dot(weights, removed*lj*r*r)
    far = 4*epsilon*(sigma**12/(9*CUTOFF**9)-sigma**6/(3*CUTOFF**3))
    coefficient = 2*np.pi*len(XYZ)**2*(far+deficit)
    assert coefficient != 0.

    with _context(_system([combined]), "CUDA") as actual, \
            _context(_system(reference_forces), reference_platform) as reference:
        for length, amplitude, charge, exception in (
                (BOX, 0., .7, .4), (BOX, .6, -.2, .8), (4.35, 1., .45, .1)):
            # Zero cutoff amplitude isolates the canonical Coulomb extraction:
            # neither the switch nor the expression tail may act on Coulomb.
            for ctx in (actual, reference):
                ctx.setPeriodicBoxVectors(*_box(length))
                ctx.setPositions(XYZ*(length/BOX))
                ctx.setParameter("amplitude", amplitude)
                ctx.setParameter("lambda_q", charge)
                ctx.setParameter("lambda_14", exception)
            for mode in MODES:
                expected_energy, expected_force = _snapshot(reference, mode)
                if expected_energy is not None:
                    expected_energy += coefficient/length**3
                _compare_snapshot(_snapshot(actual, mode), (expected_energy, expected_force))
        # Retain switching metadata and the separate native reference when
        # refreshing expression records under the current live parameters.
        custom.setParticleParameters(19, [.8])
        combined.setCutoffInteraction(custom, preserveDispersionCorrection=True)
        combined.updateParametersInContext(actual)
        if method == "Prolate":
            reference_forces[1].setCutoffInteraction(custom)
            reference_forces[1].updateParametersInContext(reference)
        else:
            custom.updateParametersInContext(reference)
        energy, force = _snapshot(reference, (True, True))
        _compare_snapshot(_snapshot(actual, (True, True)),
                          (energy+coefficient/length**3, force))
