# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only lattice definition tests; run with --noconftest to avoid GPU setup."""

import gc

import pytest

import nv_openmm_plugin as nv


def _force(custom):
    return nv.NonbondedForce("0") if custom else nv.NonbondedForce()


def _term(power=1, coupling=1., weight="charge", splitting=None):
    return nv.LatticeSumInteraction(nv.RadialKernel.inversePower(power, coupling), weight,
                                    nv.LatticeSumInteraction.Prolate if splitting is None else splitting)


def test_kernel_and_weight_require_explicit_selection():
    with pytest.raises(TypeError):
        nv.RadialKernel()
    with pytest.raises(TypeError):
        nv.RadialKernel(1)
    with pytest.raises(TypeError):
        nv.LatticeSumInteraction()
    with pytest.raises(TypeError):
        nv.LatticeSumInteraction(nv.CoulombRadialKernel())


@pytest.mark.parametrize("factory,power,coupling", [
    (nv.CoulombRadialKernel, 1, 138.93545764498),
    (nv.DispersionRadialKernel, 6, -1.),
])
def test_named_radial_factories_produce_value_descriptors(factory, power, coupling):
    kernel = factory()
    assert isinstance(kernel, nv.RadialKernel)
    assert kernel.power == power
    assert kernel.coupling == coupling
    term = nv.LatticeSumInteraction(kernel, "weight")
    kernel.coupling *= 2
    assert term.kernel.coupling == coupling
    assert term.weightExpression == "weight"
    assert term.splitting == nv.LatticeSumInteraction.Prolate
    custom = factory(coupling*3)
    assert custom.power == power
    assert custom.coupling == coupling*3
    with pytest.raises(Exception, match="RadialKernel"):
        factory(-coupling)


def _snapshot(force):
    values = []
    for index in range(force.getNumLatticeSumInteractions()):
        term = force.getLatticeSumInteraction(index)
        values.append((term.kernel.power, term.kernel.coupling,
                       term.weightExpression, term.splitting))
    return values


@pytest.mark.parametrize("custom", [False, True])
def test_lattice_collection_supports_empty_and_dispersion_only(custom):
    force = _force(custom)
    assert force.getNumLatticeSumInteractions() == (0 if custom else 1)
    if not custom:
        initial = force.getLatticeSumInteraction()
        assert initial.kernel.power == 1
        assert initial.kernel.coupling == pytest.approx(138.93545764498)
        assert initial.weightExpression == "q"
        force.removeLatticeSumInteraction(0)
    assert not force.hasLatticeSumInteraction()
    assert force.getNonbondedMethod() == nv.NonbondedForce.CutoffPeriodic

    term = _term(6, -2., "sqrt(lambda)*b6", nv.LatticeSumInteraction.Ewald)
    assert force.addLatticeSumInteraction(term) == 0
    assert force.getUseLJPME()
    assert force.getNonbondedMethod() == nv.NonbondedForce.PME
    assert _snapshot(force) == [(6, -2., "sqrt(lambda)*b6", nv.LatticeSumInteraction.Ewald)]
    with pytest.raises(Exception, match="no LatticeSumInteraction"):
        force.getLatticeSumInteraction()

    # Stored definitions and nested kernels are independent of returned copies.
    term.weightExpression = "other"
    copied = force.getLatticeSumInteraction(0)
    copied.kernel.coupling = -3.
    assert force.getLatticeSumInteraction(0).kernel.coupling == -2.
    borrowed_kernel = force.getLatticeSumInteraction(0).kernel
    gc.collect()
    assert borrowed_kernel.power == 6
    assert borrowed_kernel.coupling == -2.
    force.setLatticeSumInteraction(0, copied)
    assert force.getLatticeSumInteraction(0).kernel.coupling == -3.
    force.removeLatticeSumInteraction(0)
    assert not force.getUseLJPME()


@pytest.mark.parametrize("custom", [False, True])
def test_lattice_legacy_coulomb_accessors_preserve_independent_dispersion(custom):
    force = _force(custom)
    force.setLatticeSumInteraction(_term(1, 2., "q_eff", nv.LatticeSumInteraction.Ewald))
    assert force.getNumLatticeSumInteractions() == 1
    dispersion = _term(6, -1., "b6", nv.LatticeSumInteraction.Prolate)
    force.addLatticeSumInteraction(dispersion)
    with pytest.raises(Exception, match="different splittings"):
        force.getNonbondedMethod()
    coulomb = force.getLatticeSumInteraction()
    coulomb.splitting = nv.LatticeSumInteraction.Prolate
    force.setLatticeSumInteraction(coulomb)
    assert force.getNonbondedMethod() == nv.NonbondedForce.Prolate
    assert force.getLatticeSumInteraction(1).weightExpression == "b6"
    with pytest.raises(Exception, match="Coulomb/P1"):
        force.setLatticeSumInteraction(dispersion)


@pytest.mark.parametrize("power,coupling", [
    (8, -1.), (0, 1.), (-1, 1.), (1, -1.), (6, 1.), (1, 0.), (6, 0.),
    (1, float("nan")), (6, float("inf")), (1, 1e100), (6, -1e-100),
])
@pytest.mark.parametrize("factory", [nv.RadialKernel, nv.RadialKernel.inversePower])
def test_radial_kernel_rejects_unsupported_or_unrepresentable_values(power, coupling, factory):
    with pytest.raises(Exception, match="RadialKernel"):
        factory(power, coupling)


@pytest.mark.parametrize("field,value", [
    ("power", 8), ("coupling", -1.), ("splitting", 99), ("weight", " \t"),
])
def test_invalid_lattice_update_is_transactional(field, value):
    force = _force(False)
    before = _snapshot(force)
    term = force.getLatticeSumInteraction(0)
    if field in ("power", "coupling"):
        setattr(term.kernel, field, value)
    elif field == "weight":
        term.weightExpression = value
    else:
        setattr(term, field, value)
    with pytest.raises(Exception, match="RadialKernel|LatticeSumInteraction"):
        force.setLatticeSumInteraction(0, term)
    assert _snapshot(force) == before


def test_duplicate_powers_and_invalid_indices_are_rejected():
    force = _force(False)
    force.addLatticeSumInteraction(_term(6, -1., "b6"))
    before = _snapshot(force)
    with pytest.raises(Exception, match="one term per inverse power"):
        force.addLatticeSumInteraction(_term())
    with pytest.raises(Exception, match="one term per inverse power"):
        force.setLatticeSumInteraction(1, _term())
    for index in (-1, 2):
        for operation in (lambda: force.getLatticeSumInteraction(index),
                          lambda: force.setLatticeSumInteraction(index, _term()),
                          lambda: force.removeLatticeSumInteraction(index)):
            with pytest.raises(Exception):
                operation()
    assert _snapshot(force) == before


@pytest.mark.parametrize("custom", [False, True])
def test_ljpme_convenience_updates_collection_and_cutoff_policies(custom):
    force = _force(custom)
    force.setUseSwitchingFunction(True)
    if custom:
        force.setUseLongRangeCorrection(True)
    else:
        force.setUseDispersionCorrection(True)
    force.setUseLJPME(True)
    assert not force.getUseSwitchingFunction()
    assert not (force.getUseLongRangeCorrection() if custom else force.getUseDispersionCorrection())
    assert force.getNumLatticeSumInteractions() == (1 if custom else 2)
    term = force.getLatticeSumInteraction(force.getNumLatticeSumInteractions()-1)
    assert term.kernel.power == 6
    assert term.kernel.coupling == -1.
    assert term.weightExpression == "2*sqrt(epsilon)*sigma^3"
    force.setUseLJPME(True)
    assert force.getNumLatticeSumInteractions() == (1 if custom else 2)
    force.setUseLJPME(False)
    assert force.getNumLatticeSumInteractions() == (0 if custom else 1)


def test_custom_cutoff_compatibility_selector_clears_lattice_terms():
    force = _force(True)
    force.setLatticeSumInteraction(_term())
    force.setUseLJPME(True)
    force.setNonbondedMethod(nv.NonbondedForce.CutoffPeriodic)
    assert not force.hasLatticeSumInteraction()
    assert force.getNonbondedMethod() == nv.NonbondedForce.CutoffPeriodic
