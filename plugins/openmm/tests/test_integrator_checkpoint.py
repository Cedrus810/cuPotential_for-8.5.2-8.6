# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binary checkpoints continue private Philox noise across Contexts and graphs."""
import struct

import numpy as np
import openmm as mm
from openmm import unit
import pytest

import nv_openmm_plugin as nv

from test_fused_integrator import system_for
from test_whole_step_graph import nonbonded_system


_RNG_MAGIC = 0x4E564C4D
_WARMUP = 90


def _environment(monkeypatch, path):
    monkeypatch.delenv("CUPO_PLUGIN_BONDED_SCHEDULE", raising=False)
    for name, value in {
        "CUPO_COMPACT_AUTOTUNE": "0",
        "CUPO_PLUGIN_CUDA_GRAPH": "0" if path == "eager" else "1",
        "CUPO_PLUGIN_INTEGRATOR_GRAPH": "1",
        "CUPO_PLUGIN_FUSED_INTEGRATOR": "0" if path == "separate" else "1",
        "CUPO_PLUGIN_WHOLE_STEP_GRAPH": "1" if path == "whole-batch" else "0",
        "CUPO_PLUGIN_WHOLE_STEP_BATCH": "8",
    }.items():
        monkeypatch.setenv(name, value)


def _context(system, positions, velocities, seed, precision="mixed"):
    integrator = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
    integrator.setConstraintTolerance(1e-6)
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision, "DeterministicForces": "true"})
    context.setPositions(positions)
    context.setVelocities(velocities)
    return context, integrator


def _snapshot(context):
    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
    return (
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).copy(),
        state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond).copy(),
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer).copy(),
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getTime().value_in_unit(unit.picosecond),
        state.getStepCount(),
    )


def _advance(context, integrator):
    snapshots = []
    for steps in (37, 43):
        integrator.step(steps)
        snapshots.append(_snapshot(context))
    return snapshots


def _assert_equal(expected, observed):
    for before, after in zip(expected, observed):
        for reference, actual in zip(before, after):
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual, reference)


def _rng_offset(checkpoint):
    offset = checkpoint.rfind(struct.pack("=2I", _RNG_MAGIC, 1))
    assert offset >= 0
    return offset


@pytest.mark.parametrize("path", ["eager", "component", "separate", "whole-batch"])
@pytest.mark.parametrize("seed", [8731, 0])
@pytest.mark.parametrize("precision", ["single", "mixed"])
def test_checkpoint_continues_same_fresh_and_captured_context(monkeypatch, path, seed, precision):
    _environment(monkeypatch, path)
    system, positions, velocities = system_for("combined", water_count=9)
    context, integrator = _context(system, positions, velocities, seed, precision)
    try:
        integrator.step(_WARMUP)
        context.setParameter("k", 4.25)
        # The public step count is independently mutable. It cannot substitute
        # for the private Philox counter when saving or restoring a checkpoint.
        context.setStepCount(10003)
        checkpoint = context.createCheckpoint()
        offset = _rng_offset(checkpoint)
        magic, version, resolved_seed, counter = struct.unpack_from("=4I", checkpoint, offset)
        assert (magic, version, counter) == (_RNG_MAGIC, 1, _WARMUP)
        assert resolved_seed != 0
        if seed:
            assert resolved_seed == seed
        assert len(checkpoint) == offset+16
        expected = _advance(context, integrator)

        context.loadCheckpoint(checkpoint)
        _assert_equal(expected, _advance(context, integrator))
        for captured in (False, True):
            other, other_integrator = _context(system, positions, velocities, 9817, precision)
            try:
                if captured:
                    other_integrator.step(_WARMUP)
                other.loadCheckpoint(checkpoint)
                assert other.getParameter("k") == 4.25
                _assert_equal(expected, _advance(other, other_integrator))
                if path == "whole-batch":
                    assert other_integrator.getWholeStepGraphReplayCount() > 0
            finally:
                del other, other_integrator
    finally:
        del context, integrator


@pytest.mark.parametrize("path", ["component", "whole-batch"])
def test_checkpoint_restores_nonbonded_atom_order(monkeypatch, path):
    _environment(monkeypatch, path)
    system, positions, velocities = nonbonded_system(nv.NonbondedForce.Prolate)
    context, integrator = _context(system, positions, velocities, 731)
    try:
        integrator.step(260)
        checkpoint = context.createCheckpoint()
        expected = _advance(context, integrator)
        context.loadCheckpoint(checkpoint)
        _assert_equal(expected, _advance(context, integrator))
        other, other_integrator = _context(system, positions, velocities, 912)
        try:
            other_integrator.step(320)
            other.loadCheckpoint(checkpoint)
            _assert_equal(expected, _advance(other, other_integrator))
        finally:
            del other, other_integrator
    finally:
        del context, integrator


def test_reinitialize_preserves_noise_when_barostat_is_added_or_removed(monkeypatch):
    _environment(monkeypatch, "component")
    system, positions, velocities = nonbonded_system(nv.NonbondedForce.Prolate)
    context, integrator = _context(system, positions, velocities, 731)
    try:
        integrator.step(_WARMUP)
        checkpoint = context.createCheckpoint()
        expected = _advance(context, integrator)

        context.loadCheckpoint(checkpoint)
        index = system.addForce(nv.MonteCarloBarostat(1., 300., 0))
        context.reinitialize(preserveState=True)
        _assert_equal(expected, _advance(context, integrator))

        context.loadCheckpoint(checkpoint)
        system.removeForce(index)
        context.reinitialize(preserveState=True)
        _assert_equal(expected, _advance(context, integrator))
    finally:
        del context, integrator


def test_checkpoint_rejects_missing_truncated_and_unknown_private_state(monkeypatch):
    _environment(monkeypatch, "component")
    system, positions, velocities = system_for("water", water_count=2)
    context, integrator = _context(system, positions, velocities, 731)
    try:
        integrator.step(_WARMUP)
        checkpoint = context.createCheckpoint()
        rng = _rng_offset(checkpoint)
        malformed = [checkpoint[:rng], checkpoint[:rng+12]]
        for offset, value in ((rng, 0), (rng+4, 99), (rng+8, 0)):
            altered = bytearray(checkpoint)
            struct.pack_into("=I", altered, offset, value)
            malformed.append(bytes(altered))
        for invalid in malformed:
            with pytest.raises(Exception, match="checkpoint state"):
                context.loadCheckpoint(invalid)
        # A rejected payload must not prevent a subsequent valid restoration.
        context.loadCheckpoint(checkpoint)
        expected = _advance(context, integrator)
        context.loadCheckpoint(checkpoint)
        _assert_equal(expected, _advance(context, integrator))
    finally:
        del context, integrator
