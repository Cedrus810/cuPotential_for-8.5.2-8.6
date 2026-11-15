# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replacement records and live charges survive setup, capture, and updates."""

import numpy as np

import nv_openmm_plugin as nv

from test_custom_pme import (
    BOX, MODES, XYZ, _box, _capture_modes, _context, _system,
)
from test_custom_pme import _compare, _coulomb_only, _definitions, _single


def test_replacement_payload_updates_before_setup_and_after_capture(
        plugin_dir, monkeypatch, capfd):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    base, replacement = _definitions(True, switched=True, scenario="rbfe")
    combined = _single(base, replacement, "Prolate")

    with _context(_system([combined]), "CUDA") as actual:
        initial_globals = dict(lambda_q=.63, lambda_s=.72, lambda_14=.84)
        for name, value in initial_globals.items():
            actual.setParameter(name, value)

        # No force/energy evaluation has allocated the cell list or payloads.
        # Change charges, independent exceptions, and the replacement records
        # together; first setup must upload these values around the live lambda.
        for force in (base, combined):
            _, sigma, epsilon = force.getParticleParameters(5)
            force.setParticleParameters(5, .31, sigma, epsilon)
            force.setParticleParameterOffset(0, "lambda_q", 0, .12, 0., 0.)
            force.setExceptionParameters(1, 1, 2, -.013, .30, .14)
            force.setExceptionParameterOffset(0, "lambda_14", 0, .037, .021, .095)
        record = list(replacement.getParticleParameters(19))
        record[1] *= 1.7
        record[2:] = [1., 0.]
        replacement.setParticleParameters(19, record)
        combined.setCutoffInteraction(replacement)
        combined.updateParametersInContext(actual)

        # The oracle uses two independent cell lists/forces. Construct it from
        # the edited definitions so it cannot share a dropped pre-setup update.
        # Keep its graphs disabled: only the combined force's captures are under
        # test, and Prolate uses the same splitting in both contexts.
        reference_base = nv.NonbondedForce(
            _coulomb_only(base), 1.15, nv.NonbondedForce.Prolate)
        reference_replacement = nv.NonbondedForce(replacement, 1.15)
        with monkeypatch.context() as eager:
            eager.setenv("CUPO_PLUGIN_CUDA_GRAPH", "0")
            with _context(_system([reference_base, reference_replacement]),
                          "CUDA") as reference:
                for name, value in initial_globals.items():
                    reference.setParameter(name, value)
                _compare(actual, reference, MODES[2])
                _capture_modes(actual, capfd)

                # Each mode's first captured output sees new effective charges,
                # replacement globals, and the expression's records.
                for index, (mode, q, steric, exception) in enumerate(zip(
                        MODES, (.1, .5, .9), (.2, .6, .8), (.7, .05, .3))):
                    for ctx in (actual, reference):
                        for name, value in dict(lambda_q=q, lambda_s=steric,
                                                lambda_14=exception).items():
                            ctx.setParameter(name, value)
                    if index == 0:
                        for force in (combined, reference_base):
                            _, sigma, epsilon = force.getParticleParameters(37)
                            force.setParticleParameters(37, -.22, sigma, epsilon)
                            force.setParticleParameterOffset(0, "lambda_q", 0, -.08, 0., 0.)
                        record = list(replacement.getParticleParameters(19))
                        record[1] *= 1.35
                        record[2:] = [0., 1.]
                        replacement.setParticleParameters(19, record)
                        combined.setCutoffInteraction(replacement)
                        combined.updateParametersInContext(actual)
                        reference_base.updateParametersInContext(reference)
                        reference_replacement.setParticleParameters(19, record)
                        reference_replacement.updateParametersInContext(reference)
                    _compare(actual, reference, mode)

                # Refresh all payload mappings when atom positions change cells,
                # including a cell-grid band crossing within the original arena.
                length = 4.65
                moved = np.roll(XYZ, 19, axis=0)*(length/BOX)
                for ctx in (actual, reference):
                    ctx.setPeriodicBoxVectors(*_box(length))
                    ctx.setPositions(moved)
                _compare(actual, reference, MODES[1])
                assert "graph capture #" not in capfd.readouterr().err
