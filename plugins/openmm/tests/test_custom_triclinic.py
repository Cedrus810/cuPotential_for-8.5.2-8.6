# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom triclinic forces, live records and captured box changes against OpenMM."""
import contextlib
import itertools

import numpy as np
import openmm as mm
from openmm import unit
import pytest
import nv_openmm_plugin as nv

FRACTIONAL = np.array(list(itertools.product((.03, .27, .51, .76), repeat=3)))
MODES = ((False, True), (True, False), (True, True))
# Fixed internal force increment in kJ/mol/nm.
QUANTUM = float(np.float32(138.93545764498)) / 2**21


def _source(wide):
    expression = ("lambda*(p01*p02*r^2+p641*p642*exp(-r))" if wide else
                  "lambda*(a1*a2*r^2+b1*b2*exp(-r)+c1*c2/(1+r^2))")
    force = mm.CustomNonbondedForce(expression)
    force.setNonbondedMethod(force.CutoffPeriodic)
    force.setCutoffDistance(1.1)
    force.addGlobalParameter("lambda", .7)
    names = [f"p{i}" for i in range(65)] if wide else ["a", "b", "c"]
    for name in names:
        force.addPerParticleParameter(name)
    for atom in range(len(FRACTIONAL)):
        force.addParticle([.1 + .002*atom + .001*field for field in range(len(names))])
    # Include nearest neighbours across the c and b faces, as well as local pairs.
    for pair in ((0, 3), (0, 12), (0, 1), (20, 21)):
        force.addExclusion(*pair)
    return force


def _vectors(box):
    return tuple(mm.Vec3(*row) for row in box)


@contextlib.contextmanager
def _context(force, platform, box):
    system = mm.System()
    for _ in FRACTIONAL:
        system.addParticle(40.)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(*_vectors(box))
    # Middle velocities avoid a force evaluation for energy-only observations.
    integrator = mm.LangevinMiddleIntegrator(0., 1., 1e-5)
    properties = {"Precision": "mixed"} if platform == "CUDA" else {}
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    context.setPositions(FRACTIONAL @ box)
    try:
        yield context
    finally:
        del context, integrator


def _snapshot(context, mode):
    energy, forces = mode
    state = context.getState(getEnergy=energy, getForces=forces)
    return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) if energy else None,
            state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            if forces else None)


def _compare(actual, reference):
    for mode in MODES:
        observed, expected = _snapshot(actual, mode), _snapshot(reference, mode)
        if mode[0]:
            np.testing.assert_allclose(observed[0], expected[0], rtol=3e-5, atol=3e-5)
        if mode[1]:
            np.testing.assert_allclose(observed[1], expected[1], rtol=5e-5,
                                       atol=.5*QUANTUM*(len(FRACTIONAL)-1)+5e-5)


@pytest.mark.parametrize("wide", [False, True], ids=["three_fields", "wide_record"])
@pytest.mark.parametrize("shear", [1., -1.])
def test_custom_triclinic_captured_updates(plugin_dir, monkeypatch, capfd, wide, shear):
    monkeypatch.setenv("CUPO_PLUGIN_CUDA_GRAPH", "1")
    monkeypatch.setenv("CUPO_PLUGIN_GRAPH_TRACE", "1")
    box = np.array([[5., 0., 0.], [shear*1.2, 5., 0.], [-shear*1.0, shear*.8, 5.]])
    stock = _source(wide)
    custom = nv.NonbondedForce(stock, 1.4)
    with _context(stock, "Reference", box) as reference, _context(custom, "CUDA", box) as actual:
        _compare(actual, reference)
        capfd.readouterr()
        for _ in range(53):
            _snapshot(actual, MODES[0])
        for mode in MODES[1:]:
            for _ in range(3):
                _snapshot(actual, mode)
        captures = capfd.readouterr().err
        assert captures.count("graph capture #") == 3, captures
        _compare(actual, reference)

        # Change shear and cell counts, then pass through a rectangular box using
        # the same triclinic allocation. All steps replay the captured kernels.
        boxes = (np.array([[5.3, 0., 0.], [-shear*.8, 5.1, 0.], [shear*.9, -.6, 5.2]]),
                 np.diag([5.4, 5.3, 5.4]), box)
        for step, updated in enumerate(boxes):
            fractional = FRACTIONAL + np.sin(np.arange(FRACTIONAL.size)).reshape(FRACTIONAL.shape)*.002
            positions = fractional @ updated
            # Equivalent images exercise wrapping with opaque records attached.
            positions[::5] += updated[1] - updated[2]
            for context in (actual, reference):
                context.setPeriodicBoxVectors(*_vectors(updated))
                context.setPositions(positions)
                context.setParameter("lambda", (.25, 1., .7)[step])
            for force, context in ((custom, actual), (stock, reference)):
                parameters = list(force.getParticleParameters(5))
                parameters[0] += .03
                parameters[-1] -= .02
                force.setParticleParameters(5, parameters)
                force.updateParametersInContext(context)
            _compare(actual, reference)
        assert "graph capture #" not in capfd.readouterr().err
        actual.getIntegrator().step(5)
        reference.setPositions(actual.getState(getPositions=True).getPositions())
        _compare(actual, reference)
