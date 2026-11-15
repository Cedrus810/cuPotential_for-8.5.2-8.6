# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent weighted lattice definitions share the nonbonded executor."""

import numpy as np
import openmm as mm
import pytest

import nv_openmm_plugin as nv
from test_ljpme import BOX, RC, XYZ, _context, _state
from test_custom_pme import _capture_modes


MODES = ((False, True), (True, False), (True, True))
COULOMB = 138.93545764498
EXCLUDED = ((0, 1), (4, 7), (12, 47))
PARTICLES = np.array([[(-1)**i*.025, .26+.008*(i % 5), .2+.015*(i % 4)]
                      for i in range(len(XYZ))])


@pytest.fixture(autouse=True)
def fixed_mesh_policy(monkeypatch):
    # Match a P6-only native force against P1+P6 without allowing the latter's
    # Coulomb compact planner to select a smaller shared allocation.
    monkeypatch.setenv("CUPO_COMPACT_CONVOLUTION", "off")
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")


def _stock(particles=PARTICLES, dispersion=True):
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.LJPME if dispersion else force.PME)
    force.setCutoffDistance(RC)
    force.setEwaldErrorTolerance(5e-4)
    force.setUseDispersionCorrection(False)
    force.setExceptionsUsePeriodicBoundaryConditions(True)
    for row in particles:
        force.addParticle(*row)
    for first, second in EXCLUDED:
        force.addException(first, second, 0., .3, 0.)
    return force


def _native(method, particles=PARTICLES, dispersion=True):
    return nv.NonbondedForce(_stock(particles, dispersion), 1.15,
                             getattr(nv.NonbondedForce, method))


def _term(power, weight, method, coupling=None):
    kernel = nv.RadialKernel.inversePower(power, coupling if coupling is not None else
                                          (COULOMB if power == 1 else -1.))
    splitting = getattr(nv.LatticeSumInteraction, "Ewald" if method == "PME" else "Prolate")
    return nv.LatticeSumInteraction(kernel, weight, splitting)


def _close(actual, expected, *, atol_energy=.002, atol_force=.0004):
    if actual[0] is not None:
        assert actual[0] == pytest.approx(expected[0], rel=3e-5, abs=atol_energy)
    if actual[1] is not None:
        np.testing.assert_allclose(actual[1], expected[1], rtol=3e-4, atol=atol_force)


def _compare(actual, reference, **tolerances):
    for mode in MODES:
        _close(_state(actual, mode), _state(reference, mode), **tolerances)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_dispersion_only_native_matches_zero_charge_coulomb_and_dispersion(method):
    particles = PARTICLES.copy()
    particles[:, 0] = 0.
    reference = _native(method, particles)
    only_dispersion = _native(method)
    only_dispersion.removeLatticeSumInteraction(0)
    assert only_dispersion.getNumLatticeSumInteractions() == 1
    assert only_dispersion.getLatticeSumInteraction(0).kernel.power == 6
    with _context(reference) as expected, _context(only_dispersion) as actual:
        _compare(actual, expected)
        for force, context, charge in ((reference, expected, 0.), (only_dispersion, actual, .09)):
            force.setParticleParameters(0, charge, .31, .4)
            force.updateParametersInContext(context)
        _compare(actual, expected)


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_standard_derived_charge_weight_matches_effective_native_charges(method):
    weighted = _native(method, dispersion=False)
    weighted.addGlobalParameter("lam", .2)
    weighted.setLatticeSumInteraction(_term(1, "lam*q", method))
    reference = _native(method, dispersion=False)
    with _context(reference) as expected, _context(weighted) as actual:
        for lam in (.2, .8, 0., 1.):
            actual.setParameter("lam", lam)
            for index, (q, sigma, epsilon) in enumerate(PARTICLES):
                reference.setParticleParameters(index, lam*q, sigma, epsilon)
            reference.updateParametersInContext(expected)
            _compare(actual, expected)
        # Exception replacement charge products remain independent of the
        # transformed ordinary weights, just as for native parameter offsets.
        lam = .4
        actual.setParameter("lam", lam)
        for index, (q, sigma, epsilon) in enumerate(PARTICLES):
            reference.setParticleParameters(index, lam*q, sigma, epsilon)
        for force, context in ((weighted, actual), (reference, expected)):
            force.setExceptionParameters(0, 0, 1, -.0007, .32, .08)
            force.updateParametersInContext(context)
        _compare(actual, expected)
        updated = PARTICLES[0].copy()
        updated[0] = -.037
        for force, context, scale in ((weighted, actual, 1.), (reference, expected, lam)):
            force.setParticleParameters(0, scale*updated[0], *updated[1:])
            force.updateParametersInContext(context)
        _compare(actual, expected)


@pytest.mark.parametrize("global_name", ["q", "sigma", "epsilon", "r"])
def test_native_weight_schema_preserves_existing_offset_global_names(global_name):
    def make(name):
        source = _stock()
        source.addGlobalParameter(name, .2)
        for index in range(0, len(XYZ), 7):
            source.addParticleParameterOffset(name, index, .014, .018, .09)
        source.addExceptionParameterOffset(name, 0, .0003, .012, .02)
        return nv.NonbondedForce(source, 1.15, nv.NonbondedForce.Prolate)

    with _context(make("lam")) as expected, _context(make(global_name)) as actual:
        for value in (.2, .75, 0.):
            actual.setParameter(global_name, value)
            expected.setParameter("lam", value)
            for mode in MODES:
                observed, reference = _state(actual, mode), _state(expected, mode)
                if mode[0]:
                    assert observed[0] == reference[0]
                if mode[1]:
                    np.testing.assert_array_equal(observed[1], reference[1])


def test_general_weight_rejects_a_used_ambiguous_global_name():
    force = _native("Prolate", dispersion=False)
    force.addGlobalParameter("q", .2)
    force.setLatticeSumInteraction(_term(1, "q+sigma", "Prolate"))
    with pytest.raises(Exception, match="ambiguous particle/global weight variable q"):
        with _context(force):
            pass


@pytest.mark.parametrize("charge_method,dispersion_method", [("PME", "Prolate"), ("Prolate", "PME")])
def test_independent_splittings_equal_sum_of_charge_and_dispersion_forces(charge_method, dispersion_method):
    combined = _native(charge_method)
    combined.setLatticeSumInteraction(1, _term(6, "2*sqrt(epsilon)*sigma^3", dispersion_method))
    charges = PARTICLES.copy()
    charges[:, 2] = 0.
    charge_only = _native(charge_method, charges, dispersion=False)
    charge_only.addLatticeSumInteraction(_term(6, "0", dispersion_method))
    dispersion_particles = PARTICLES.copy()
    dispersion_particles[:, 0] = 0.
    dispersion_only = _native(dispersion_method, dispersion_particles)
    dispersion_only.removeLatticeSumInteraction(0)
    dispersion_only.addLatticeSumInteraction(_term(1, "0", charge_method))
    with _context([charge_only, dispersion_only]) as expected, _context(combined) as actual:
        # Zero-weight partners retain the same conservative common mesh in both
        # independent controls. Only their physical P1/P6 contributions remain,
        # so differences cannot be attributed to different interpolation grids.
        displacement = XYZ[:, None, :]-XYZ[None, :, :]
        displacement -= np.rint(displacement/BOX)*BOX
        distance2 = np.sum(displacement*displacement, axis=2)
        neighbors = np.count_nonzero((distance2 > 0.) & (distance2 < RC*RC), axis=1)
        quantum = float(np.float32(COULOMB))/(1 << 21)
        for mode in MODES:
            observed, reference = _state(actual, mode), _state(expected, mode)
            _close((observed[0], None), (reference[0], None))
            if not mode[1]:
                continue
            difference = observed[1]-reference[1]
            # Each direct traversal rounds its pair forces to the default
            # int32 quantum. round(A)+round(B)-round(A+B) is bounded by one
            # quantum per pair/component; summing two traversals is therefore
            # not bitwise equivalent to the combined traversal. Counting even
            # excluded neighbors gives a conservative per-atom physical bound.
            rounding = 8*np.finfo(np.float32).eps*np.maximum(
                1., np.maximum(np.abs(observed[1]), np.abs(reference[1])))
            bound = neighbors[:, None]*quantum+rounding
            assert np.all(np.abs(difference) <= bound), np.max(np.abs(difference)/bound)
            # Matched meshes remove the earlier interpolation discrepancy:
            # the residual must consist of force quanta plus FP32 restoration
            # and final fixed-point conversion rounding.
            remainder = difference-quantum*np.rint(difference/quantum)
            assert np.all(np.abs(remainder) <= rounding), np.max(np.abs(remainder)/rounding)


def _full_custom(method, dispersion_coupling=-1.):
    strength = -dispersion_coupling
    source = mm.CustomNonbondedForce(
        f"{COULOMB:.17g}*charge1*charge2/r+"
        "4*sqrt(e1*e2)*(sig^12*(r^-12-1)-sig^6*(r^-6-1));"
        "charge1=(1-lam)*qA1+lam*qB1;charge2=(1-lam)*qA2+lam*qB2;"
        "sig=(sigma1+sigma2)/2;"
        f"e1={strength:.17g}*(1+lam)*epsilon1;e2={strength:.17g}*(1+lam)*epsilon2")
    source.setNonbondedMethod(source.CutoffPeriodic)
    source.setCutoffDistance(RC)
    source.addGlobalParameter("lam", .2)
    for name in ("qA", "qB", "sigma", "epsilon"):
        source.addPerParticleParameter(name)
    for q, sigma, epsilon in PARTICLES:
        source.addParticle([q, -.5*q+.003, sigma, epsilon])
    for first, second in EXCLUDED:
        source.addExclusion(first, second)
    custom = nv.NonbondedForce(source, 1.15)
    custom.addLatticeSumInteraction(_term(1, "(1-lam)*qA+lam*qB", method))
    custom.addLatticeSumInteraction(_term(6, "2*sqrt((1+lam)*epsilon)*sigma^3", method,
                                           dispersion_coupling))
    native_source = _stock()
    native_source.addGlobalParameter("lam", .2)
    for i, (q, sigma, epsilon) in enumerate(PARTICLES):
        native_source.setParticleParameters(i, q, sigma, strength*epsilon)
        native_source.addParticleParameterOffset("lam", i, -1.5*q+.003, 0., strength*epsilon)
    native = nv.NonbondedForce(native_source, 1.15, getattr(nv.NonbondedForce, method))
    return custom, native


@pytest.mark.parametrize("method", ["PME", "Prolate"])
@pytest.mark.parametrize("dispersion_coupling", [-1., -4.])
def test_standalone_custom_full_source_lattice_weights_update_and_replay(
        method, dispersion_coupling, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    custom, native = _full_custom(method, dispersion_coupling)
    strength = -dispersion_coupling
    with _context(native) as expected, _context(custom) as actual:
        _compare(actual, expected)
        _capture_modes(actual, capfd)
        for lam in (.2, .75, 0., 1.):
            for context in (actual, expected):
                context.setParameter("lam", lam)
            _compare(actual, expected)
            assert "graph capture #" not in capfd.readouterr().err
        # Both prepared charge and b6 fields must refresh after record changes,
        # while exclusion pairs keep their identities and captured pointers.
        qA, qB, sigma, epsilon = -.017, .011, .315, .24
        custom.setParticleParameters(0, [qA, qB, sigma, epsilon])
        native.setParticleParameters(0, qA, sigma, strength*epsilon)
        native.setParticleParameterOffset(0, "lam", 0, qB-qA, 0., strength*epsilon)
        custom.updateParametersInContext(actual)
        native.updateParametersInContext(expected)
        _compare(actual, expected)
        assert "graph capture #" not in capfd.readouterr().err
        checkpoint = actual.createCheckpoint()
        actual.setPositions(XYZ[::-1].copy())
        _state(actual, (False, True))
        actual.loadCheckpoint(checkpoint)
        _compare(actual, expected)
        assert "graph capture #" not in capfd.readouterr().err


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_standalone_custom_dispersion_only_refreshes_hidden_weights(method):
    custom, native = _full_custom(method)
    custom.removeLatticeSumInteraction(0)
    native.removeLatticeSumInteraction(0)
    # The full source contains a Coulomb factor, but both end-state charges are
    # zero here. Its P6 mesh and sparse corrections must work without a P1 slot.
    for index in range(len(XYZ)):
        values = list(custom.getParticleParameters(index))
        values[:2] = [0., 0.]
        custom.setParticleParameters(index, values)
    with _context(native) as expected, _context(custom) as actual:
        for lam in (.2, .8, 0., 1.):
            for context in (actual, expected):
                context.setParameter("lam", lam)
            _compare(actual, expected)


def test_standalone_custom_lattice_first_replay_after_real_atom_reorder(monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    custom, native = _full_custom("Prolate")
    # A stock cutoff consumer enables OpenMM's real spatial reorder machinery;
    # the plugin's own cell list does not register with NonbondedUtilities.
    inert = mm.CustomNonbondedForce("0")
    inert.setNonbondedMethod(inert.CutoffPeriodic)
    inert.setCutoffDistance(RC)
    for _ in XYZ:
        inert.addParticle([])
    for first, second in EXCLUDED:
        inert.addExclusion(first, second)
    with _context(native) as expected, _context([custom, inert]) as actual:
        _capture_modes(actual, capfd)
        requested = XYZ[np.random.default_rng(847).permutation(len(XYZ))]
        requested = requested + np.array([BOX, -BOX, BOX])
        capfd.readouterr()
        for _ in range(251):
            actual.setPositions(requested)
            # Retain the first output before reading back coordinates: another
            # evaluation or synchronization could hide stale ordering inputs.
            observed = _state(actual)
            trace = capfd.readouterr().err
            assert "graph capture #" not in trace, trace
            if "[CustomNonbondedForce] atom reorder #" in trace:
                assert "[nv-plugin cupo] atom reorder #" in trace, trace
                break
        else:
            pytest.fail("No nontrivial Custom/lattice atom reorder after capture")
        positions = actual.getState(getPositions=True).getPositions(asNumpy=True)
        expected.setPositions(positions)
        _close(observed, _state(expected))
        _compare(actual, expected)
        capfd.readouterr()
        for context in (actual, expected):
            context.setParameter("lam", .65)
        _compare(actual, expected)
        assert "graph capture #" not in capfd.readouterr().err
        custom.setParticleParameters(0, [-.017, .011, .315, .24])
        native.setParticleParameters(0, -.017, .315, .24)
        native.setParticleParameterOffset(0, "lam", 0, .028, 0., .24)
        custom.updateParametersInContext(actual)
        native.updateParametersInContext(expected)
        _compare(actual, expected)
        assert "graph capture #" not in capfd.readouterr().err


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_standalone_custom_raw_charge_switches_only_cutoff_residual(method):
    lj = "4*sqrt(epsilon1*epsilon2)*((sigma/r)^12-(sigma/r)^6);sigma=(sigma1+sigma2)/2"
    source = mm.CustomNonbondedForce(f"{COULOMB:.17g}*q1*q2/r+"+lj)
    switched_lj = mm.CustomNonbondedForce(lj)
    for force in (source, switched_lj):
        force.setNonbondedMethod(force.CutoffPeriodic)
        force.setCutoffDistance(RC)
        force.setUseSwitchingFunction(True)
        force.setSwitchingDistance(.8)
        names = ("q", "sigma", "epsilon") if force is source else ("sigma", "epsilon")
        for name in names:
            force.addPerParticleParameter(name)
        for row in PARTICLES:
            force.addParticle(row if force is source else row[1:])
        for first, second in EXCLUDED:
            force.addExclusion(first, second)
    custom = nv.NonbondedForce(source, 1.15)
    custom.addLatticeSumInteraction(_term(1, "q", method))
    charges = PARTICLES.copy()
    charges[:, 2] = 0.
    native = _native(method, charges, dispersion=False)
    with _context([native, switched_lj]) as expected, _context(custom) as actual:
        _compare(actual, expected)
        # Coordinate motion changes which ordinary LJ pairs enter the switch.
        moved = XYZ.copy()
        moved[3] += [.11, -.04, .03]
        for context in (actual, expected):
            context.setPositions(moved)
        _compare(actual, expected)


@pytest.mark.parametrize("switched", [False, True])
def test_standard_empty_lattice_falls_back_to_cutoff_lj(switched):
    particles = PARTICLES.copy()
    particles[:, 0] = 0.
    stock = _stock(particles, dispersion=False)
    stock.setNonbondedMethod(stock.CutoffPeriodic)
    stock.setUseSwitchingFunction(switched)
    stock.setSwitchingDistance(.8)
    plugin = _native("Prolate", dispersion=False)
    plugin.removeLatticeSumInteraction(0)
    plugin.setUseSwitchingFunction(switched)
    plugin.setSwitchingDistance(.8)
    with _context(stock) as expected, _context(plugin) as actual:
        _compare(actual, expected)


@pytest.mark.parametrize("weight", ["r", "0*r", "q1", "0*missing", "sqrt(-1)"])
def test_invalid_weight_definition_fails_before_execution(weight):
    force = _native("PME", dispersion=False)
    force.setLatticeSumInteraction(_term(1, weight, "PME"))
    with pytest.raises(Exception, match="weight|finite|Unknown|forbidden"):
        with _context(force):
            pass
