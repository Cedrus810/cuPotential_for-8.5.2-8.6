# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangevinMiddleIntegrator.step() releases the GIL while it integrates.

The SWIG interface names the method to release the GIL around; with the wrong
namespace the directive applied to nothing and step() held the GIL for its whole
duration, so no other Python thread ran until it returned.
"""
import threading
import time

import openmm as mm
import pytest

import nv_openmm_plugin as nv

from test_whole_step_graph import nonbonded_system


def _context(seed):
    system, xyz, vel = nonbonded_system(nv.NonbondedForce.Prolate)
    integrator = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
    integrator.setRandomNumberSeed(seed)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    context.setPositions(xyz)
    context.setVelocities(vel)
    return context, integrator


def test_step_releases_the_gil(plugin_dir):
    context, integrator = _context(11)
    integrator.step(50)  # warm-up: kernel compilation and the first tuning pass
    begin = time.perf_counter()
    integrator.step(200)
    per_step = (time.perf_counter() - begin) / 200
    steps = max(500, int(1.0 / per_step))  # about a second of integration

    ticks = []
    def count():
        # A thread that cannot take the GIL does not wake from sleep until step()
        # returns; with the GIL released it keeps ticking the whole time.
        while not done.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.005)
    done = threading.Event()
    counter = threading.Thread(target=count)
    counter.start()
    begin = time.perf_counter()
    integrator.step(steps)
    elapsed = time.perf_counter() - begin
    done.set()
    counter.join()

    during = sum(begin < t < begin + elapsed for t in ticks)
    expected = elapsed / 0.005
    assert during > expected / 4, (
        f"{during} ticks in {elapsed:.2f}s of step(); the GIL was held through step()")
    del context


def test_exception_from_step_restores_the_thread_state(plugin_dir):
    """The error path must take the GIL back before raising, not crash.

    step() on an integrator with no Context throws from C++ before any GPU work.
    """
    integrator = nv.LangevinMiddleIntegrator(300., 1., .001, 7)
    with pytest.raises(Exception, match="not bound to a context"):
        integrator.step(1)
    # The interpreter is intact: Python code runs and the error is cleared.
    assert threading.current_thread() is threading.main_thread()
    with pytest.raises(Exception, match="not bound to a context"):
        integrator.step(1)
