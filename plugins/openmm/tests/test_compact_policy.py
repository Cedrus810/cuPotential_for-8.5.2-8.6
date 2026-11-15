# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Short integration check: automatic compression survives OpenMM box replay."""
import numpy as np
import re
import openmm as mm
import openmm.unit as unit
import pytest


@pytest.mark.parametrize('precision', ['single', 'mixed'])
@pytest.mark.parametrize('autotune', [False, True])
def test_compact_auto_box_graph(plugin_dir, monkeypatch, capfd, precision, autotune):
    import nv_openmm_plugin as nv
    if autotune:
        monkeypatch.delenv('CUPO_COMPACT_AUTOTUNE', raising=False)  # default is on
    else:
        monkeypatch.setenv('CUPO_COMPACT_AUTOTUNE', '0')

    initial = 8.8546
    def vectors(length):
        return [mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0), mm.Vec3(0, 0, length)]

    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*vectors(initial))
    source = mm.NonbondedForce()
    source.setNonbondedMethod(mm.NonbondedForce.PME)
    source.setCutoffDistance(.9)
    source.setUseDispersionCorrection(False)
    rng = np.random.default_rng(318)
    charges = rng.normal(size=64)
    charges -= charges.mean()
    for q in charges:
        system.addParticle(40.)
        source.addParticle(float(q), .3, 0.)
    system.addForce(nv.NonbondedForce(source, 1.2))
    coordinates = np.array(list(np.ndindex(4, 4, 4)), float)
    coordinates = (coordinates+.5+rng.uniform(-.04, .04, (64, 3)))*initial/4
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': precision, 'DeviceIndex': '0', 'DeterministicForces': 'true'}
    contexts, integrators = [], []
    try:
        for graph in [True, False]:
            # Unset/default and explicit auto must select the same policy.
            monkeypatch.delenv('CUPO_COMPACT_CONVOLUTION', raising=False)
            monkeypatch.delenv('CUPO_EXPERIMENT_COMPACT_CONVOLUTION', raising=False)
            if not graph:
                monkeypatch.setenv('CUPO_COMPACT_CONVOLUTION', 'auto')
            monkeypatch.setenv('CUPO_PLUGIN_CUDA_GRAPH', '1' if graph else '0')
            monkeypatch.setenv('CUPO_PLUGIN_GRAPH_TRACE', '1')
            integrators.append(mm.VerletIntegrator(.001))
            context = mm.Context(system, integrators[-1], platform, properties)
            context.setPositions(coordinates)
            contexts.append(context)
            # Device setup (and backend selection) is deferred to first use.
            context.getState(getForces=True)
        setup = capfd.readouterr().err
        configurations = re.findall(
            r'\[nv-plugin cupo\] convolution mesh=.*? compact=(\d) mode=(\w+)', setup)
        assert len(configurations) == 2, setup
        if any(compact != '1' for compact, _ in configurations):
            pytest.skip('runtime does not support compact convolution on this GPU')
        assert all(mode == 'auto' for _, mode in configurations)
        choices = re.findall(r'convolution selection potential=(\w+) energy=(\w+) both=(\w+)', setup)
        assert len(choices) == (2 if autotune else 0)
        graph_context, eager_context = contexts
        def evaluate(context):
            force = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole/unit.nanometer)
            energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            return force, energy
        for _ in range(55):
            evaluate(graph_context)
        warm = capfd.readouterr().err
        assert warm.count('graph capture #') == 2, warm
        for length in [8.7, 9.05, 8.8546]:
            for context in contexts:
                context.setPeriodicBoxVectors(*vectors(length))
                context.setPositions(coordinates*(length/initial))
            actual, expected = evaluate(graph_context), evaluate(eager_context)
            # Separate contexts can time different winners. Identical backends
            # retain the original bitwise graph/eager check; different FFT
            # algorithms use the same 1e-5 numerical criterion as the grid oracle.
            if not autotune or choices[0][0] == choices[1][0]:
                np.testing.assert_array_equal(actual[0], expected[0])
            else:
                assert np.linalg.norm(actual[0]-expected[0])/np.linalg.norm(expected[0]) < 1e-5
            if not autotune or choices[0][1] == choices[1][1]:
                assert actual[1] == expected[1]
            else:
                assert abs(actual[1]-expected[1])/max(abs(expected[1]),1.) < 1e-5
        updates = capfd.readouterr().err
        assert 'graph capture #' not in updates
        assert 'convolution selection ' not in updates
    finally:
        contexts.clear()
        # Local references otherwise keep a Context alive after its integrator.
        context = graph_context = eager_context = None
        integrators.clear()
