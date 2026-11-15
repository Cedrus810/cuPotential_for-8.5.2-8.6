# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host runtime errors preserve a live Standard OpenMM Context and its graph."""
import itertools
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import openmm as mm
from openmm import unit
import pytest


def _cube(length):
    return (mm.Vec3(length, 0, 0), mm.Vec3(0, length, 0), mm.Vec3(0, 0, length))


def _snapshot(context):
    state = context.getState(getEnergy=True, getForces=True)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))


def _exercise_box_recovery(method):
    import nv_openmm_plugin as nv

    positions = np.array(list(itertools.product((.2, .9, 1.6), repeat=3)))
    system, source = mm.System(), mm.NonbondedForce()
    system.setDefaultPeriodicBoxVectors(*_cube(4.0))
    source.setNonbondedMethod(source.PME)
    source.setCutoffDistance(.9)
    source.setUseDispersionCorrection(False)
    for atom in range(len(positions)):
        source.addParticle((-1)**atom*.03, .3, .2)
        system.addParticle(39.9)
    system.addForce(nv.NonbondedForce(source, 1.15, getattr(nv.NonbondedForce, method)))
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed", "DeterministicForces": "true"})
    try:
        context.setPositions(positions)
        for _ in range(55):
            expected_energy, expected_forces = _snapshot(context)
        assert np.isfinite(expected_energy) and abs(expected_energy) > 0
        assert np.isfinite(expected_forces).all() and np.linalg.norm(expected_forces) > 0
        for _ in range(2):
            context.setPeriodicBoxVectors(*_cube(8.0))
            with pytest.raises(mm.OpenMMException, match="(?i)envelope"):
                _snapshot(context)
            context.setPeriodicBoxVectors(*_cube(4.0))
            energy, forces = _snapshot(context)
            assert energy == expected_energy
            np.testing.assert_array_equal(forces, expected_forces)
    finally:
        del context, integrator
    print("BOX_RECOVERY_PASSED")


@pytest.mark.parametrize("method", ["PME", "Prolate"])
def test_box_expansion_error_preserves_context(plugin_dir, method):
    # An abort or C++ exception escaping the C boundary must fail this test,
    # without terminating the entire pytest process.
    env = os.environ.copy()
    from conftest import PLUGIN_PATH
    env["PYTHONPATH"] = os.pathsep.join(filter(None, PLUGIN_PATH + [env.get("PYTHONPATH")]))
    env["CUPO_PLUGIN_CUDA_GRAPH"] = "1"
    env["CUPO_PLUGIN_GRAPH_TRACE"] = "1"
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), method],
                            env=env, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "BOX_RECOVERY_PASSED" in result.stdout, output
    assert "graph capture #" in result.stderr, output


if __name__ == "__main__":
    _exercise_box_recovery(sys.argv[1])
