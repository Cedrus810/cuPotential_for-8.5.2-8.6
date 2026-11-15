# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused integration against the separate BAOAB/SETTLE/SHAKE implementation.

Synthetic nonzero forces isolate integration from pair-kernel error. The fixed
Philox seed tests stage ordering and graph replay as well as constraint solving.
Single-precision tolerance allows FP32 contraction differences amplified by the
constraint displacement / dt velocity correction; mixed must remain bitwise equal.

Mixed precision computes in float-float on most GPUs and in native double on GPUs
with fast FP64 (A100, H100, B200; CUPO_PLUGIN_MIXED_COMPUTE overrides the choice).
'mixed' pins float-float, so its bitwise checks run on every GPU; 'mixed-double'
pins double, where the fused SETTLE and the separate kernels may differ by FP64
contraction, a few ulps.
"""
import os
from unittest.mock import patch
import pytest
import numpy as np
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv

def system_for(case, water_count=47):
    sys = mm.System()
    xyz = []
    rng = np.random.default_rng(371)
    def add(masses, pos, edges):
        start = len(xyz)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pos = np.array(pos) @ rotation + rng.uniform(-1, 1, 3)
        for m, p in zip(masses, pos):
            sys.addParticle(m); xyz.append(p)
        for a, b in edges:
            sys.addConstraint(start+a, start+b, float(np.linalg.norm(pos[a]-pos[b])))
    if 'water' in case or case == 'combined':
        for i in range(water_count):
            # Keep identical distances in the parameter table for the constants arm.
            start = len(xyz)
            length = .09572 * (1.01 if case == 'water_geometry' and i % 2 else 1)
            angle = np.deg2rad(104.52)
            masses = [16., 1., 1.]
            if case == 'water_mass' and i % 2: masses = [18., 2., 2.]
            add(masses, [[0,0,0], [length,0,0], [length*np.cos(angle),length*np.sin(angle),0]], [])
            sys.addConstraint(start, start+1, length)
            sys.addConstraint(start, start+2, length)
            sys.addConstraint(start+1, start+2, 2*length*np.sin(angle/2))
    if case.startswith('shake') or case == 'combined':
        sizes = [int(case[-1])] if case.startswith('shake') else [2,3,4]
        for size in sizes:
            for _ in range(17):
                add([12.] + [1.]*(size-1), [[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]][:size],
                    [(0, k) for k in range(1, size)])
    if case in ('free', 'combined', 'water_free'):
        for i in range(3 if case == 'water_free' else 137):
            add([0. if i % 5 == 0 else 12.], [[0,0,0]], [])
    if case == 'water_virtual':
        for parent in [0, 3]:
            atom = sys.addParticle(0.)
            sys.setVirtualSite(atom, mm.ThreeParticleAverageSite(parent, parent+1, parent+2, .8, .1, .1))
            xyz.append(.8*xyz[parent]+.1*xyz[parent+1]+.1*xyz[parent+2])
    force = mm.CustomExternalForce('0.5*k*(x*x+y*y+z*z)')
    force.addGlobalParameter('k', 3.)
    for i in range(len(xyz)): force.addParticle(i, [])
    sys.addForce(force)
    return sys, np.array(xyz), rng.normal(scale=.1, size=(len(xyz),3))

# The test precisions: OpenMM's Precision property and the mixed compute type.
# 'mixed-env' leaves CUPO_PLUGIN_MIXED_COMPUTE as the caller set it.
PRECISIONS = {'single': ('single', 'auto'), 'mixed': ('mixed', 'fltflt'), 'mixed-double': ('mixed', 'double'),
              'mixed-env': ('mixed', None)}

def run(case, precision, fused, graph, steps=(1, 4, 20), update=False, water_count=47):
    precision, compute = PRECISIONS[precision]
    sys, xyz, vel = system_for(case, water_count)
    integ = nv.LangevinMiddleIntegrator(300., 1., .001, 0)
    integ.setRandomNumberSeed(58392)
    integ.setConstraintTolerance(1e-5)
    # The fused step skips the integrator graph by default; graph=True asks for it
    # back so the capture and replay stay covered.
    env = {'CUPO_PLUGIN_FUSED_INTEGRATOR':str(int(fused)), 'CUPO_PLUGIN_CUDA_GRAPH':str(int(graph)),
           'CUPO_PLUGIN_INTEGRATOR_GRAPH':str(int(graph))}
    if compute is not None:
        env['CUPO_PLUGIN_MIXED_COMPUTE'] = compute
    with patch.dict(os.environ, env):
        ctx = mm.Context(sys, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision':precision})
    ctx.setPositions(xyz); ctx.setVelocities(vel)
    results = []
    for i, n in enumerate(steps):
        if update and i == 2:
            integ.setTemperature(275.)
            integ.setFriction(2.)
            integ.setStepSize(.0015)
        integ.step(n)
        state = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)
        errors = []
        for j in range(sys.getNumConstraints()):
            a,b,d = sys.getConstraintParameters(j); d = d.value_in_unit(unit.nanometer)
            errors.append(abs(np.linalg.norm(pos[a]-pos[b])/d - 1))
        results.append({'pos':pos, 'vel':v, 'constraint':max(errors, default=0),
                        'energy':state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)})
    del ctx, integ
    return results


@pytest.mark.parametrize('precision', ['single', 'mixed', 'mixed-double'])
@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('case', ['free', 'shake2', 'shake3', 'shake4', 'water',
    'water_mass', 'water_geometry', 'water_free', 'water_virtual', 'combined'])
def test_fused_matches_separate(case, precision, graph):
    separate = run(case, precision, False, graph, update=True)
    fused = run(case, precision, True, graph, update=True)
    for reference, actual in zip(separate, fused):
        assert np.isfinite(actual['pos']).all()
        assert np.isfinite(actual['vel']).all()
        assert actual['constraint'] < 1e-5
        if precision == 'mixed':
            np.testing.assert_array_equal(actual['pos'], reference['pos'])
            np.testing.assert_array_equal(actual['vel'], reference['vel'])
        elif precision == 'mixed-double':
            # Measured on an RTX 6000 Ada: at most 2.2e-16 apart after 25 steps.
            np.testing.assert_allclose(actual['pos'], reference['pos'], rtol=0, atol=1e-13)
            np.testing.assert_allclose(actual['vel'], reference['vel'], rtol=0, atol=1e-10)
        else:
            np.testing.assert_allclose(actual['pos'], reference['pos'], rtol=0, atol=5e-6)
            np.testing.assert_allclose(actual['vel'], reference['vel'], rtol=0, atol=5e-4)


@pytest.mark.parametrize('precision', ['single', 'mixed', 'mixed-double'])
def test_fused_graph_matches_eager(precision):
    eager = run('combined', precision, True, False, update=True)
    graph = run('combined', precision, True, True, update=True)
    for a, b in zip(eager, graph):
        np.testing.assert_array_equal(a['pos'], b['pos'])
        np.testing.assert_array_equal(a['vel'], b['vel'])


@pytest.mark.parametrize('precision', ['single', 'mixed', 'mixed-double'])
@pytest.mark.parametrize('graph', [False, True])
def test_grid_preserves_trajectory(precision, graph, capfd):
    # 1201 waters span 31 SETTLE chunks, plus packed SHAKE/free-atom blocks. A tiny
    # blocks-per-SM value gives one block, which takes every unit through the dynamic
    # handout; 0 gives one block per unit and no handout; the default sits between.
    # The trajectory, the last-block RNG counter and graph replay must not notice.
    import json
    snapshots = []
    records = []
    for cap in ['0.01', '2.5', '5', '0']:
        with patch.dict(os.environ, {'CUPO_PLUGIN_FUSED_BLOCKS_PER_SM':cap,
                                    'CUPO_PLUGIN_FUSED_REPORT_CONFIG':'1'}):
            snapshots.append(run('combined', precision, True, graph,
                                 steps=(1, 4, 20), update=True, water_count=1201))
        lines = capfd.readouterr().out.splitlines()
        record = next(json.loads(line.split(' ', 1)[1]) for line in lines
                      if line.startswith('FUSED_INTEGRATOR_CONFIG '))
        units = record['units']
        expected = units if float(cap) == 0 else min(units, max(1, int(record['sms']*float(cap))))
        assert record['settle_chunks'] == 31
        assert record['packed_blocks'] > 0
        assert units == record['settle_chunks'] + record['packed_blocks']
        assert record['grid_blocks'] == expected
        records.append(record)
    assert records[0]['grid_blocks'] == 1
    assert records[0]['grid_blocks'] < records[-1]['grid_blocks'] == records[-1]['units']
    for actual in snapshots[1:]:
        for a, b in zip(snapshots[0], actual):
            np.testing.assert_array_equal(a['pos'], b['pos'])
            np.testing.assert_array_equal(a['vel'], b['vel'])
            assert a['energy'] == b['energy']


@pytest.mark.parametrize('precision', ['single', 'mixed', 'mixed-double'])
@pytest.mark.parametrize('graph', [False, True])
def test_heavy_units_preserve_trajectory(precision, graph, capfd):
    # The SHAKE units with 3- and 4-atom clusters get SMs of their own (heavy_start in
    # FusedChains.cuh), in mixed precision: the SM's first block takes one, the others
    # exit, and every other unit is handed out at run time. Off, on with the default
    # grid, on with a one-block grid (every heavy unit through the late handout) and
    # on with one block per unit must give the same trajectory. Single precision has
    # no heavy units.
    import json
    snapshots, heavy = [], []
    for env in [{'CUPO_PLUGIN_FUSED_HEAVY_SIZE':'0'}, {}, {'CUPO_PLUGIN_FUSED_HEAVY_SIZE':'2'},
                {'CUPO_PLUGIN_FUSED_BLOCKS_PER_SM':'0.01'}, {'CUPO_PLUGIN_FUSED_BLOCKS_PER_SM':'0'}]:
        with patch.dict(os.environ, {**env, 'CUPO_PLUGIN_FUSED_REPORT_CONFIG':'1'}):
            snapshots.append(run('combined', precision, True, graph,
                                 steps=(1, 4, 20), update=True, water_count=1201))
        lines = capfd.readouterr().out.splitlines()
        record = next(json.loads(line.split(' ', 1)[1]) for line in lines
                      if line.startswith('FUSED_INTEGRATOR_CONFIG '))
        heavy.append(record['heavy_units'])
    if precision == 'single':
        assert heavy == [0] * len(heavy), heavy
    else:
        assert heavy[0] == 0 and min(heavy[1:]) > 0, heavy
    for actual in snapshots[1:]:
        for a, b in zip(snapshots[0], actual):
            np.testing.assert_array_equal(a['pos'], b['pos'])
            np.testing.assert_array_equal(a['vel'], b['vel'])
            assert a['energy'] == b['energy']


@pytest.mark.parametrize('precision', ['mixed', 'mixed-double'])
def test_heavy_units_checkpoint_replay(precision, capfd):
    # Loading a checkpoint sets the step number back, so the steps after it run with
    # step numbers the fused kernel has seen before. With heavy units on, the steps
    # after a load must repeat the steps after the checkpoint exactly.
    import json
    compute = PRECISIONS[precision][1]
    sys, xyz, vel = system_for('combined', 1201)
    integ = nv.LangevinMiddleIntegrator(300., 1., .001, 0)
    integ.setRandomNumberSeed(58392)
    integ.setConstraintTolerance(1e-5)
    with patch.dict(os.environ, {'CUPO_PLUGIN_MIXED_COMPUTE':compute, 'CUPO_PLUGIN_FUSED_REPORT_CONFIG':'1'}):
        ctx = mm.Context(sys, integ, mm.Platform.getPlatformByName('CUDA'), {'Precision':'mixed'})
    ctx.setPositions(xyz); ctx.setVelocities(vel)
    integ.step(3)
    record = next(json.loads(line.split(' ', 1)[1]) for line in capfd.readouterr().out.splitlines()
                  if line.startswith('FUSED_INTEGRATOR_CONFIG '))
    assert record['heavy_units'] > 0, record
    checkpoint = ctx.createCheckpoint()
    runs = []
    for _ in range(3):
        ctx.loadCheckpoint(checkpoint)
        integ.step(7)
        state = ctx.getState(getPositions=True, getVelocities=True)
        runs.append((state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                     state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond)))
    del ctx, integ
    for pos, v in runs[1:]:
        np.testing.assert_array_equal(pos, runs[0][0])
        np.testing.assert_array_equal(v, runs[0][1])


@pytest.mark.parametrize('value', ['', '-1', 'nan', 'inf', '33', '5x'])
def test_invalid_blocks_per_sm(value):
    with patch.dict(os.environ, {'CUPO_PLUGIN_FUSED_BLOCKS_PER_SM':value}):
        with pytest.raises(mm.OpenMMException, match='CUPO_PLUGIN_FUSED_BLOCKS_PER_SM'):
            run('water', 'mixed', True, False)


@pytest.mark.parametrize('compute', ['fltflt', 'double'])
def test_mixed_compute_override(compute, capfd):
    import json
    with patch.dict(os.environ, {'CUPO_PLUGIN_MIXED_COMPUTE':compute, 'CUPO_PLUGIN_FUSED_REPORT_CONFIG':'1'}):
        run('water', 'mixed-env', True, False, steps=(1,))
    lines = capfd.readouterr().out.splitlines()
    record = next(json.loads(line.split(' ', 1)[1]) for line in lines
                  if line.startswith('FUSED_INTEGRATOR_CONFIG '))
    assert record['mixed_compute'] == compute


def test_invalid_mixed_compute():
    with patch.dict(os.environ, {'CUPO_PLUGIN_MIXED_COMPUTE':'float'}):
        with pytest.raises(mm.OpenMMException, match='CUPO_PLUGIN_MIXED_COMPUTE'):
            run('water', 'mixed-env', True, False)
