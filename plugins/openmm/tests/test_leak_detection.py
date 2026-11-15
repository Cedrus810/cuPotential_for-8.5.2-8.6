# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that plugin context create-destroy cycles don't leak GPU memory.

Each test creates and destroys OpenMM Contexts in a loop (10 iterations).
The first 2 are warmup (absorb JIT/module loads). Iterations 3-10
must show ~0 memory growth.
"""
import numpy as np
import random

import pytest

from mem_tracker import MemoryTracker


def _build_water_system(barostat_bar=None):
    """TIP3P water box with NvNonbondedForce (plugin's prolate splitting)."""
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from conftest import replace_nonbonded

    ff = app.ForceField("tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    np.random.seed(42)
    random.seed(42)
    modeller.addSolvent(ff, model="tip3p",
                        boxSize=mm.Vec3(3.0, 3.0, 3.0) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    replace_nonbonded(system, npt_box_safety_factor=1.15)
    if barostat_bar is not None:
        baro = mm.MonteCarloBarostat(
            barostat_bar * unit.bar, 300 * unit.kelvin, 25)
        baro.setRandomNumberSeed(777)
        system.addForce(baro)
    positions = modeller.positions
    return system, positions


@pytest.mark.parametrize("graph,barostat_bar,steps,minimize", [
    pytest.param("0", None, 5, False, id="eager-nvt"),
    pytest.param("1", None, 60, False, id="graph-nvt"),
    pytest.param("1", 1.0, 100, True, id="graph-npt"),
])
def test_plugin_context_does_not_leak(plugin_dir, graph, barostat_bar,
                                      steps, minimize):
    """Plugin context: create/step/destroy cycle must not leak."""
    import openmm as mm
    import openmm.unit as unit
    import os

    os.environ["CUPO_PLUGIN_CUDA_GRAPH"] = graph
    system, positions = _build_water_system(barostat_bar=barostat_bar)
    platform = mm.Platform.getPlatformByName("CUDA")
    props = {"Precision": "mixed"}

    with MemoryTracker(warmup=2) as mt:
        for _ in range(10):
            integ = mm.LangevinMiddleIntegrator(
                300 * unit.kelvin, 1.0 / unit.picosecond,
                0.002 * unit.picoseconds)
            integ.setRandomNumberSeed(4242)
            ctx = mm.Context(system, integ, platform, props)
            ctx.setPositions(positions)
            if minimize:
                mm.LocalEnergyMinimizer.minimize(ctx, maxIterations=50)
            integ.step(steps)
            del ctx, integ
            mt.tick()
