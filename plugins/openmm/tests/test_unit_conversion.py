# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A unit.Quantity must arrive as the number the C++ API expects.

The wrappers take Quantities so they can be drop-ins, which means each one has to
name the unit its C++ side reads. Inferring it instead -- via
value_in_unit_system(md_unit_system) -- is right for everything whose OpenMM API
happens to take MD units (nm, kJ/mol, Kelvin, 1/ps) and wrong for the one that
does not: pressure crosses in BAR, and the MD system renders 1 bar as 1e-25. A
barostat built with `1.0*unit.bar` therefore ran at 1e-25 bar with no error, no
NaN, and a P*dV acceptance term too small to disturb the trajectory visibly.

Every case is checked against stock OpenMM rather than against a literal, so the
tests say "the same number stock would use" instead of restating the conversion
and agreeing with themselves.
"""

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _units():
    import openmm.unit as unit
    return unit


# One bar spelled four ways. atm and pascal matter more than bar here: a helper
# that merely passed `_value` through would still get bar right by coincidence.
def _pressures():
    unit = _units()
    return [
        ("bar", 1.0 * unit.bar, 1.0),
        ("atmosphere", 1.0 * unit.atmosphere, 1.01325),
        ("pascal", 101325.0 * unit.pascal, 1.01325),
        ("kilopascal", 100.0 * unit.kilopascal, 1.0),
        ("bare float", 2.5, 2.5),
    ]


@pytest.mark.parametrize("name", [p[0] for p in _pressures()])
def test_barostat_pressure_matches_stock(plugin_dir, name):
    """The reported bug: `1.0*unit.bar` used to arrive as 1e-25."""
    import openmm as mm
    import nv_openmm_plugin as nv
    unit = _units()

    quantity, expected_bar = next((q, e) for n, q, e in _pressures() if n == name)
    plugin = nv.MonteCarloBarostat(quantity, 300 * unit.kelvin, frequency=25)
    stock = mm.MonteCarloBarostat(quantity, 300 * unit.kelvin, 25)

    got = plugin.getDefaultPressure().value_in_unit(unit.bar)
    want = stock.getDefaultPressure().value_in_unit(unit.bar)
    assert got == pytest.approx(want, rel=1e-12), (
        "%s: plugin got %r bar, stock uses %r bar" % (name, got, want))
    assert got == pytest.approx(expected_bar, rel=1e-12)


def test_barostat_temperature_matches_stock(plugin_dir):
    """Correct before the fix too -- Kelvin is already an MD unit -- so this is
    the control that says the fix did not move what was already right."""
    import openmm as mm
    import nv_openmm_plugin as nv
    unit = _units()

    plugin = nv.MonteCarloBarostat(1.0 * unit.bar, 310.0 * unit.kelvin, frequency=25)
    stock = mm.MonteCarloBarostat(1.0 * unit.bar, 310.0 * unit.kelvin, 25)
    assert plugin.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(
        stock.getDefaultTemperature().value_in_unit(unit.kelvin), rel=1e-12)


def test_integrator_arguments_match_stock(plugin_dir):
    """The integrator already named its units; this pins that it still does.

    Friction is the interesting one: 1/ps is an MD unit, but 1/fs is not, so a
    helper that stopped naming the unit would be caught here.
    """
    import openmm as mm
    import nv_openmm_plugin as nv
    unit = _units()

    temperature = 310.0 * unit.kelvin
    friction = 0.002 / unit.femtosecond          # == 2.0 / picosecond
    step = 2.0 * unit.femtosecond                # == 0.002 picosecond

    plugin = nv.LangevinMiddleIntegrator(temperature, friction, step)
    stock = mm.LangevinMiddleIntegrator(temperature, friction, step)
    assert plugin.getTemperature().value_in_unit(unit.kelvin) == pytest.approx(
        stock.getTemperature().value_in_unit(unit.kelvin), rel=1e-12)
    assert plugin.getFriction().value_in_unit(unit.picosecond**-1) == pytest.approx(
        stock.getFriction().value_in_unit(unit.picosecond**-1), rel=1e-12)
    assert plugin.getStepSize().value_in_unit(unit.picosecond) == pytest.approx(
        stock.getStepSize().value_in_unit(unit.picosecond), rel=1e-12)
    assert plugin.getFriction().value_in_unit(unit.picosecond**-1) == pytest.approx(2.0, rel=1e-12)


def test_a_bare_number_is_taken_as_already_converted(plugin_dir):
    """A bare number stays legal, for two reasons that both point outward.

    Stock takes one -- mm.MonteCarloBarostat(1.0, 300, 25) is valid -- and these
    wrappers claim to be drop-ins for it. And a bare float in bar was the
    documented workaround while the conversion was broken, so refusing it now
    would break whoever followed our own advice.
    """
    import nv_openmm_plugin as nv
    unit = _units()

    barostat = nv.MonteCarloBarostat(1.0, 300, frequency=25)
    assert barostat.getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(1.0, rel=1e-12)
    assert barostat.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(300.0, rel=1e-12)
    assert nv.MonteCarloBarostat(2, 300).getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(2.0)


@pytest.mark.parametrize("bad", ["1.0", None, [1.0], object()])
def test_a_non_number_is_refused_by_the_same_rule_as_stock(plugin_dir, bad):
    """Refused, and refused by SWIG rather than by a rule of our own.

    The wrapper hands a non-Quantity straight on, exactly as stock's generated
    prologue does, so what this plugin accepts is decided by the same typemap that
    decides it for OpenMM. A float() here would have been a second, looser rule --
    float("1.0") is 1.0 -- and worse, a Quantity from some OTHER unit library
    answers openmm's is_quantity() with False and would then unwrap to whatever
    magnitude it happens to carry.
    """
    import openmm as mm
    import openmm.unit as unit
    import nv_openmm_plugin as nv

    with pytest.raises(TypeError):
        nv.MonteCarloBarostat(bad, 300 * unit.kelvin, frequency=25)
    # ... and stock refuses it too, which is the point.
    with pytest.raises(TypeError):
        mm.MonteCarloBarostat(bad, 300 * unit.kelvin, 25)


# ── The number has to reach the physics, not just the C++ setter ──────────────
#
# Everything above checks what arrives at getDefaultPressure(). That is necessary
# and not sufficient: a barostat that stored the right number and then ignored it
# would pass every one of those tests.
#
# No LIQUID system can close that gap. Water at 1 bar and at 1e-25 bar produces a
# bitwise identical trajectory -- 1 bar compresses water by 4.5e-5, which never
# flips a Metropolis decision -- so the whole NPT suite is blind to a pressure
# that has been scaled away. A GAS is not: with no cohesion to resist it, the
# entropic term in the acceptance is unopposed and the box expands without bound.
#
# Argon at 300 K, well above its 150.7 K critical point, at ~1e-3 of liquid
# density, with the standard Lennard-Jones parameters for argon.

_AR_SIGMA = 0.3405          # nm
_AR_EPSILON = 0.238 * 4.184  # kcal/mol -> kJ/mol
_AR_MASS = 39.948            # g/mol
_GAS_N = 64
_GAS_T = 300.0               # K
_GAS_EDGE = 16.0             # nm; larger than the 1-bar equilibrium, so 1 bar COMPRESSES
_BAR_IN_MD = 6.02214076e23 * 1e-25   # kJ/mol/nm^3 per bar


def _argon_gas_system():
    """N argon atoms, uncharged, in a box too big for their number at 1 bar."""
    import numpy as np
    import openmm as mm
    import openmm.unit as unit

    system = mm.System()
    edge = _GAS_EDGE
    system.setDefaultPeriodicBoxVectors(mm.Vec3(edge, 0, 0), mm.Vec3(0, edge, 0),
                                        mm.Vec3(0, 0, edge))
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.9 * unit.nanometer)
    nonbonded.setUseSwitchingFunction(False)
    nonbonded.setUseDispersionCorrection(False)
    for _ in range(_GAS_N):
        system.addParticle(_AR_MASS * unit.amu)
        nonbonded.addParticle(0.0, _AR_SIGMA, _AR_EPSILON)
    system.addForce(nonbonded)
    positions = np.random.default_rng(7).uniform(0, edge, size=(_GAS_N, 3))
    return system, positions


def _gas_volume_after(pressure, steps):
    """Box volume in nm^3 after `steps` of NPT at `pressure`.

    `pressure` is a unit.Quantity on purpose. A bare float was converted correctly
    even under the bug -- only Quantities went through the inferring helper -- so a
    float-only version of this test would have passed while 1 bar was 1e-25.
    """
    import numpy as np
    import openmm as mm
    import openmm.unit as unit
    import nv_openmm_plugin as nv

    system, positions = _argon_gas_system()
    for i, force in enumerate(system.getForces()):
        if isinstance(force, mm.NonbondedForce):
            converted = nv.NonbondedForce(force, 3.0)   # room for the box to grow
            system.removeForce(i)
            system.addForce(converted)
            break
    system.addForce(nv.MonteCarloBarostat(pressure, _GAS_T * unit.kelvin,
                                          frequency=25, seed=12345))
    integrator = mm.LangevinMiddleIntegrator(
        _GAS_T * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    integrator.setRandomNumberSeed(4242)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    context.setPositions(positions)
    try:
        integrator.step(steps)
        box = context.getState().getPeriodicBoxVectors()
        return (box[0][0] * box[1][1] * box[2][2]).value_in_unit(unit.nanometer**3)
    finally:
        del context, integrator


def test_a_gas_tells_one_bar_from_vacuum(plugin_dir):
    """1 bar must compress the gas; ~0 bar must let it run away.

    This is the test the liquid suite cannot be: it fails if pressure is scaled,
    dropped, or ignored anywhere between the Python call and the acceptance
    criterion. Under the unit bug it would have seen 1e-25 bar in BOTH arms and
    the two volumes would have agreed.
    """
    unit = _units()
    steps = 6000
    at_one_bar = _gas_volume_after(1.0 * unit.bar, steps)
    at_vacuum = _gas_volume_after(1e-25 * unit.bar, steps)

    assert at_vacuum > 1.3 * at_one_bar, (
        "1 bar and ~0 bar gave indistinguishable volumes for a GAS "
        "(%.0f vs %.0f nm^3); the pressure is not reaching the barostat."
        % (at_one_bar, at_vacuum))
    # 1 bar compresses below the starting box; vacuum expands above it.
    assert at_one_bar < _GAS_EDGE**3 < at_vacuum


def test_the_compressed_gas_lands_near_the_ideal_gas_volume(plugin_dir):
    """And it compresses to the RIGHT volume, not merely to a smaller one.

    V = NkT/P for an ideal gas. Loose tolerance on purpose: 64 atoms fluctuate by
    1/sqrt(N) ~ 12%, and this is checking the order of magnitude of the pressure
    rather than the equation of state.
    """
    kt = 0.00831446261815324 * _GAS_T                 # kJ/mol
    ideal = _GAS_N * kt / _BAR_IN_MD                  # nm^3 at 1 bar
    volume = _gas_volume_after(1.0 * _units().bar, 12000)
    assert volume == pytest.approx(ideal, rel=0.45), (
        "settled at %.0f nm^3, ideal gas at 1 bar is %.0f nm^3" % (volume, ideal))
