# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process XML reconstruction preserves checkpointed NVT/NPT state."""
import gc
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import openmm as mm
import pytest

import nv_openmm_plugin as nv
import nvplugin

from test_barostat_checkpoint import (
    _advance_npt, _assert_continues, _assert_restored, _npt_context, _observation,
    _system as _npt_system,
)
from test_barostat_checkpoint import _WARMUP as _NPT_WARMUP
from test_integrator_checkpoint import _assert_equal, _context, _environment
from test_integrator_checkpoint import _WARMUP as _NVT_WARMUP
from test_whole_step_graph import nonbonded_system


def _resume_in_fresh_process(directory, ensemble, integrator_kind, barostat_kind):
    """Called only by the child interpreter; inherit no live parent objects."""
    system_xml = (directory/"system.xml").read_text()
    integrator_xml = (directory/"integrator.xml").read_text()
    system = mm.XmlSerializer.deserialize(system_xml)
    integrator = mm.XmlSerializer.deserialize(integrator_xml)
    assert mm.XmlSerializer.serialize(system) == system_xml
    assert mm.XmlSerializer.serialize(integrator) == integrator_xml
    integrator_type = (nvplugin.LangevinMiddleIntegrator if integrator_kind == "plugin"
                       else mm.LangevinMiddleIntegrator)
    assert isinstance(integrator, integrator_type)
    assert integrator.getRandomNumberSeed() == 0
    if integrator_kind == "plugin":
        assert integrator.getComFrequency() == 7
    assert isinstance(system.getForce(0), nv.NonbondedForce)
    if ensemble == "npt":
        assert system.getNumForces() == 2
        barostat = system.getForce(1)
        barostat_type = (nvplugin.MonteCarloBarostat if barostat_kind == "plugin"
                         else mm.MonteCarloBarostat)
        assert isinstance(barostat, barostat_type)
        assert barostat.getRandomNumberSeed() == 0
        assert barostat.getFrequency() == 5
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed", "DeterministicForces": "true"})
    try:
        # Do not reseed or reconstruct RNG state in Python. XML retains seed
        # zero, while the checkpoint restores the resolved integrator RNG and
        # OpenMM's shared random stream. Barostat bookkeeping is not restored.
        context.loadCheckpoint((directory/"context.chk").read_bytes())
        if ensemble == "npt":
            assert context.getParameter(barostat.Pressure()) == 77.
            assert context.getParameter(barostat.Temperature()) == 305.
            snapshots = [_observation(context)]
            _assert_continues(context, integrator)
        else:
            snapshots = _advance_npt(context, integrator)
        np.savez(directory/"resumed.npz", **{
            f"state_{index}_{field}": value
            for index, snapshot in enumerate(snapshots)
            for field, value in enumerate(snapshot)
        })
    finally:
        del context, integrator


@pytest.mark.parametrize("ensemble,integrator_kind,barostat_kind", [
    pytest.param("nvt", "plugin", "none", id="nvt-plugin"),
    pytest.param("npt", "plugin", "plugin", id="npt-plugin-plugin"),
    pytest.param("npt", "stock", "plugin", id="npt-stock-plugin"),
    pytest.param("npt", "plugin", "stock", id="npt-plugin-stock"),
    pytest.param("npt", "stock", "stock", id="npt-stock-stock"),
])
def test_checkpoint_restart_in_fresh_process(
        monkeypatch, tmp_path, ensemble, integrator_kind, barostat_kind):
    _environment(monkeypatch, "component")
    monkeypatch.delenv("CUPO_OPENMM_PLUGIN_RNG_SEED", raising=False)
    if ensemble == "npt":
        system, positions, velocities = _npt_system(0, barostat_kind)
        context, integrator = _npt_context(
            system, positions, velocities, 0, "mixed", integrator_kind)
        warmup = _NPT_WARMUP
    else:
        system, positions, velocities = nonbonded_system(nv.NonbondedForce.Prolate)
        context, integrator = _context(system, positions, velocities, 0, "mixed")
        warmup = _NVT_WARMUP
    try:
        integrator.step(warmup)
        if ensemble == "npt":
            barostat = system.getForce(1)
            context.setParameter(barostat.Pressure(), 77.)
            context.setParameter(barostat.Temperature(), 305.)
        (tmp_path/"system.xml").write_text(mm.XmlSerializer.serialize(system))
        (tmp_path/"integrator.xml").write_text(mm.XmlSerializer.serialize(integrator))
        (tmp_path/"context.chk").write_bytes(context.createCheckpoint())
        expected = ([_observation(context)] if ensemble == "npt"
                    else _advance_npt(context, integrator))
    finally:
        del context, integrator
    # Release the parent Context before starting another CUDA process.
    gc.collect()

    environment = os.environ.copy()
    plugin_python = str(Path(nv.__file__).resolve().parent.parent)
    tests = str(Path(__file__).resolve().parent)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        plugin_python, tests, environment.get("PYTHONPATH"),
    )))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), ensemble,
         integrator_kind, barostat_kind],
        env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(tmp_path/"resumed.npz", allow_pickle=False) as restored:
        observed = [
            tuple(restored[f"state_{index}_{field}"] for field in range(len(snapshot)))
            for index, snapshot in enumerate(expected)
        ]
        if ensemble == "npt":
            _assert_restored(expected[0], observed[0])
        else:
            _assert_equal(expected, observed)


if __name__ == "__main__":
    _resume_in_fresh_process(Path(sys.argv[1]), *sys.argv[2:])
