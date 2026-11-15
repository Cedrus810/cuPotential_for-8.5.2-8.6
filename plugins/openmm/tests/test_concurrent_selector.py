# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenMM invokes the joined selector once, before any force/energy graph."""
import ctypes
import json
import numpy as np
import pytest
import openmm as mm
import openmm.unit as unit


@pytest.mark.parametrize('reciprocal_mode', ['coulomb', 'ljpme', 'dispersion'])
@pytest.mark.parametrize('convolution', ['off', 'auto'])
@pytest.mark.parametrize('shape', ['cube', 'rhombic_dodecahedron', 'truncated_octahedron'])
@pytest.mark.parametrize('method', ['PME', 'Prolate'])
def test_concurrent_selector_first_energy_and_replay(plugin_dir, monkeypatch, capfd, shape, method, convolution, reciprocal_mode):
    import nv_openmm_plugin as nv
    from test_triclinic import _shapes
    for key in ['CUPO_DSPACE_AUTOTUNE', 'CUPO_DSPACE_PIECE_CONFIG',
                'CUPO_DSPACE_PIECE_THREADS', 'CUPO_DSPACE_CLEAVE_TOWER',
                'CUPO_DSPACE_PIECE_SELECTOR', 'CUPO_DSPACE_AUTOTUNE_OBJECTIVE']:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('CUPO_PIECE_REPORT_CONFIG','1')
    monkeypatch.setenv('CUPO_PLUGIN_GRAPH_TRACE','1')
    monkeypatch.setenv('CUPO_PLUGIN_CUDA_GRAPH','1')
    # Hold the reciprocal backend fixed across the two direct-space objectives.
    # Separate compact-policy tests cover reciprocal backend autotuning.
    monkeypatch.setenv('CUPO_COMPACT_CONVOLUTION', convolution)
    monkeypatch.setenv('CUPO_COMPACT_AUTOTUNE', '0')
    vectors = _shapes(mm.Vec3)[shape]
    system = mm.System()
    system.setDefaultPeriodicBoxVectors(*vectors)
    source = mm.NonbondedForce()
    source.setNonbondedMethod(mm.NonbondedForce.PME if reciprocal_mode=='coulomb' else mm.NonbondedForce.LJPME)
    source.setCutoffDistance(1.2)
    source.setUseDispersionCorrection(False)
    source.setExceptionsUsePeriodicBoundaryConditions(True)
    for i in range(216):
        system.addParticle(40.)
        source.addParticle(.1 if i%2 else -.1,.25,.1)
    source.addException(0,1,0.,.25,0.)
    force = nv.NonbondedForce(source,1.2,getattr(nv.NonbondedForce,method))
    if reciprocal_mode=='dispersion':
        force.removeLatticeSumInteraction(0)
    system.addForce(force)
    coords = ((np.array(list(np.ndindex(6,6,6)))+.5)/6) @ np.array(vectors)
    properties = {'Precision':'mixed','DeviceIndex':'0','DeterministicForces':'true'}
    contexts, integrators = [], []
    context = None
    try:
        for objective in ['concurrent','direct']:
            monkeypatch.setenv('CUPO_DSPACE_AUTOTUNE_OBJECTIVE',objective)
            integrators.append(mm.VerletIntegrator(.001))
            context = mm.Context(system,integrators[-1],mm.Platform.getPlatformByName('CUDA'),properties)
            contexts.append(context)
            context.setPositions(coords)
            # First call is ENERGY ONLY, but layout is chosen for MD force steps.
            context.getState(getEnergy=True)
            ctypes.CDLL(None).fflush(None)
            captured = capfd.readouterr()
            if convolution == 'auto' and reciprocal_mode != 'dispersion' and 'compact convolution mesh=' not in captured.err:
                pytest.skip('compact convolution unavailable for this build/GPU/grid')
            report = captured.out
            choices = [json.loads(line.split(' ',1)[1]) for line in report.splitlines()
                       if line.startswith('PIECE_AUTOTUNE ')]
            assert len(choices)==1, report
            assert choices[0]['objective']==('direct+kspace' if objective=='concurrent' else 'direct')
            expected_count = (2 if reciprocal_mode=='ljpme' else 1) if objective=='concurrent' else 0
            assert choices[0]['reciprocal_workspaces']==expected_count

        def evaluate(ctx):
            state=ctx.getState(getEnergy=True,getForces=True)
            return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))

        for _ in range(55):
            actual, expected = map(evaluate, contexts)
        assert actual[0]==expected[0]
        np.testing.assert_array_equal(actual[1],expected[1])
        ctypes.CDLL(None).fflush(None)
        report=capfd.readouterr()
        assert 'graph capture #' in report.err
        assert 'PIECE_AUTOTUNE ' not in report.out
        for context in contexts:
            context.setPeriodicBoxVectors(*(v*1.015 for v in vectors))
            context.setPositions(coords*1.015)
        actual, expected = map(evaluate,contexts)
        assert actual[0]==expected[0]
        np.testing.assert_array_equal(actual[1],expected[1])
        ctypes.CDLL(None).fflush(None)
        assert 'PIECE_AUTOTUNE ' not in capfd.readouterr().out
    finally:
        context = None
        contexts.clear()
        integrators.clear()
