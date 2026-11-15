# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NPT restart restores physical state after a live box change."""

import pytest

from test_barostat_checkpoint import (
    _assert_continues, _assert_restored, _npt_context, _observation, _system, _WARMUP,
)
from test_integrator_checkpoint import _environment


@pytest.mark.parametrize("restart", ["fresh", "reinitialize"])
def test_checkpoint_after_box_expansion_restores_physical_state(monkeypatch, restart):
    _environment(monkeypatch, "component")
    monkeypatch.setenv("CUPO_COMPACT_CONVOLUTION", "auto")
    system, xyz, velocities = _system(719)
    context, integrator = _npt_context(system, xyz, velocities, 817)
    try:
        integrator.step(_WARMUP)
        barostat = system.getForce(1)
        context.setParameter(barostat.Pressure(), 77.)
        context.setParameter(barostat.Temperature(), 305.)
        # Device allocation started at the original 3.2-nm box. This expansion
        # fits its 1.5 safety factor, but rebuilding the envelope at the new box
        # can change the automatic compact support and force rounding.
        # Keep the System defaults unchanged, as an ordinary NPT move does.
        box = system.getDefaultPeriodicBoxVectors()
        context.setPeriodicBoxVectors(*(1.2*vector for vector in box))
        integrator.step(1)
        checkpoint = context.createCheckpoint()
        expected = _observation(context)

        if restart == "reinitialize":
            context.reinitialize(preserveState=True)
            _assert_restored(expected, _observation(context))
            assert context.getParameter(barostat.Pressure()) == 77.
            assert context.getParameter(barostat.Temperature()) == 305.
            _assert_continues(context, integrator)
        else:
            other_system, other_xyz, other_velocities = _system(913)
            other, other_integrator = _npt_context(
                other_system, other_xyz, other_velocities, 9817)
            try:
                other.loadCheckpoint(checkpoint)
                _assert_restored(expected, _observation(other))
                assert other.getParameter(barostat.Pressure()) == 77.
                assert other.getParameter(barostat.Temperature()) == 305.
                _assert_continues(other, other_integrator)
            finally:
                del other, other_integrator
    finally:
        del context, integrator
