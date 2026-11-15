# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller expressions select alchemical regions without interaction-group APIs."""

import itertools

import numpy as np
import openmm as mm
import pytest
import nv_openmm_plugin as nv

from test_custom_nonbonded import XYZ, compare, context, snapshot, switched_expression
from test_custom_precompute import OUTPUT_MODES, compare_modes


def selected_force(selection, *, oracle=False, radial="lambda*a1*a2/(1+r^2)"):
    """Use stock groups only as an independent direct-space selection oracle."""
    if selection == "abfe_cross":
        names = ("alchemical",)
        records = [[float(atom % 3 == 0)] for atom in range(len(XYZ))]
        mask = "alchemical1+alchemical2-2*alchemical1*alchemical2"
        ligand = {atom for atom, record in enumerate(records) if record[0]}
        groups = [(ligand, set(range(len(XYZ)))-ligand)]
    elif selection == "rbfe":
        # Environment, unique old, unique new, and common core. Omit env-env
        # (handled by a separate component) and old-new interactions.
        names = ("env", "old", "new")
        records = [[float(atom % 4 == region) for region in range(3)]
                   for atom in range(len(XYZ))]
        mask = "1-env1*env2-old1*new2-new1*old2"
        regions = [{atom for atom in range(len(XYZ)) if atom % 4 == region}
                   for region in range(4)]
        groups = [(regions[first], regions[second])
                  for first, second in itertools.combinations_with_replacement(range(4), 2)
                  if (first, second) not in ((0, 0), (1, 2))]
    else:
        raise ValueError(selection)
    expression = radial if oracle else f"select(allowed,{radial},0);allowed={mask}"
    force = mm.CustomNonbondedForce(expression)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.25)
    force.addPerParticleParameter("a")
    for name in (() if oracle else names):
        force.addPerParticleParameter(name)
    force.addGlobalParameter("lambda", .7)
    for atom, record in enumerate(records):
        force.addParticle([.2+.1*(atom % 3)] + ([] if oracle else record))
    if oracle:
        for first, second in groups:
            force.addInteractionGroup(first, second)
    force.addExclusion(0, 1)
    force.addExclusion(2, 3)
    return force


@pytest.mark.parametrize("selection", ["abfe_cross", "rbfe"])
@pytest.mark.parametrize("switched", [False, True])
def test_region_masks_match_grouped_oracle_with_exclusions_switching_and_updates(
        plugin_dir, selection, switched):
    stock = selected_force(selection, oracle=True)
    expression = selected_force(selection)
    assert expression.getNumInteractionGroups() == 0
    stock.setUseSwitchingFunction(switched)
    stock.setSwitchingDistance(.85)
    if switched:
        expression.setEnergyFunction(switched_expression(expression.getEnergyFunction(), .85, 1.25))
    custom = nv.NonbondedForce(expression)
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        compare_modes(actual, reference)
        for value in (0., .3, 1.):
            for ctx in (reference, actual):
                ctx.setParameter("lambda", value)
            compare_modes(actual, reference)
        for force, ctx in ((stock, reference), (custom, actual)):
            for atom, value in ((5, .8), (21, .15)):
                record = list(force.getParticleParameters(atom))
                record[0] = value
                force.setParticleParameters(atom, record)
            force.updateParametersInContext(ctx)
            ctx.setPositions(XYZ[::-1])
        compare_modes(actual, reference)


def test_abfe_pair_lambda_preserves_ligand_and_environment_internal_interactions(plugin_dir):
    expression = selected_force("abfe_cross")
    expression.setEnergyFunction(
        "coupling*a1*a2/(1+r^2);coupling=1-(1-lambda)*cross;"
        "cross=alchemical1+alchemical2-2*alchemical1*alchemical2")
    # Independent reference decomposition: all interactions at full strength,
    # plus a cross-region correction. At lambda=0 only the cross pairs vanish.
    full = selected_force("abfe_cross")
    full.setEnergyFunction("a1*a2/(1+r^2)")
    cross = selected_force("abfe_cross", oracle=True, radial="(lambda-1)*a1*a2/(1+r^2)")
    custom = nv.NonbondedForce(expression)
    with context(full, "Reference") as full_reference, \
            context(cross, "Reference") as cross_reference, context(custom, "CUDA") as actual:
        for value in (0., .3, 1.):
            for ctx in (full_reference, cross_reference, actual):
                ctx.setParameter("lambda", value)
            for energy, forces in OUTPUT_MODES:
                parts = [snapshot(ctx, energy, forces) for ctx in (full_reference, cross_reference)]
                expected = tuple(parts[0][i]+parts[1][i] if parts[0][i] is not None else None
                                 for i in range(2))
                compare(snapshot(actual, energy, forces), expected)


def test_region_memberships_can_be_updated_as_particle_parameters(plugin_dir):
    stock = selected_force("rbfe")
    custom = nv.NonbondedForce(stock)
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        compare_modes(actual, reference)
        for force, ctx in ((stock, reference), (custom, actual)):
            # Move an old-only atom into the environment and a new-only atom
            # into the core, without any interaction-group metadata update.
            force.setParticleParameters(5, [.3, 1., 0., 0.])
            force.setParticleParameters(6, [.2, 0., 0., 0.])
            force.updateParametersInContext(ctx)
            ctx.setParameter("lambda", .4)
        compare_modes(actual, reference)


def test_explicit_mask_skips_invalid_radial_arithmetic(plugin_dir):
    # An unexcluded close pair has r < .5, where sqrt(r-.5) and its
    # derivative are undefined. Lazy select must skip that arithmetic.
    stock = mm.CustomNonbondedForce("lambda*sqrt(r-.5)")
    expression = mm.CustomNonbondedForce("select(active1*active2,lambda*sqrt(r-.5),0)")
    expression.addPerParticleParameter("active")
    for force in (stock, expression):
        force.setNonbondedMethod(force.CutoffPeriodic)
        force.setCutoffDistance(1.25)
        force.addGlobalParameter("lambda", .7)
        for atom in range(len(XYZ)):
            force.addParticle([] if force is stock else [float(atom >= 3)])
        force.addExclusion(0, 2)
    stock.addInteractionGroup(set(range(3, len(XYZ))), set(range(3, len(XYZ))))
    custom = nv.NonbondedForce(expression)
    positions = XYZ.copy()
    # Move atom 1 inside its existing coarse cell: moving atom 2 from the
    # neighboring cell would exceed this deliberately sparse fixture's capacity.
    positions[1] = positions[0]+[.1, 0, 0]
    with context(stock, "Reference") as reference, context(custom, "CUDA") as actual:
        for ctx in (reference, actual):
            ctx.setPositions(positions)
        compare_modes(actual, reference)


def test_region_parameters_follow_reordering_and_parameter_updates(plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    stock = selected_force("rbfe", oracle=True)
    expression = selected_force("rbfe")
    # Equal interaction strengths with different region flags must remain
    # distinct particle types in molecule/reordering bookkeeping.
    for force in (stock, expression):
        for atom in range(len(XYZ)):
            record = list(force.getParticleParameters(atom))
            record[0] = .2
            force.setParticleParameters(atom, record)
    custom = nv.NonbondedForce(expression)
    with context(stock, "Reference", True) as reference, context(custom, "CUDA", True) as actual:
        compare_modes(actual, reference)
        actual.setPositions(XYZ[np.random.default_rng(19).permutation(len(XYZ))])
        actual.getIntegrator().setStepSize(1e-5)
        capfd.readouterr()
        actual.getIntegrator().step(270)
        assert "[CustomNonbondedForce] atom reorder #" in capfd.readouterr().err
        reference.setPositions(actual.getState(getPositions=True).getPositions())
        compare_modes(actual, reference)
        for force, ctx in ((stock, reference), (custom, actual)):
            record = list(force.getParticleParameters(19))
            record[0] = .55
            force.setParticleParameters(19, record)
            force.updateParametersInContext(ctx)
            ctx.setParameter("lambda", .4)
        compare_modes(actual, reference)
