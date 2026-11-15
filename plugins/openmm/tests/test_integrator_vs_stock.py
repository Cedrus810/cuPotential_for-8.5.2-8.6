# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nv.LangevinMiddleIntegrator against stock openmm.LangevinMiddleIntegrator.

The plugin's other integrator tests compare the plugin only with itself (fused
against separate kernels, graph against eager, checkpoints). These compare its
dynamics with stock OpenMM's, within relative errors rather than bit for bit:

  * test_deterministic: friction 0, so there is no noise and both integrators are
    velocity Verlet with constraints. The same start, the same steps, and the
    positions, velocities and energies compared by relative error, with a
    tolerance taken from stock's own spread.
  * test_statistics: friction 1/ps at 300 K. The random streams differ (the
    plugin draws from Philox), so the comparison is of averages, with error bars
    from block averages.
  * test_com_removal: the plugin removes the centre-of-mass motion itself every
    comFrequency steps. The total momentum must be what the Langevin noise
    builds up between removals, and with removal every step it must be as small
    as stock's with a CMMotionRemover.

The forces. Both sides run the same System, with the plugin's NonbondedForce
(nv.NonbondedForce, converted from the force field's NonbondedForce) and stock
bonded forces. The plugin's integrator runs with stock forces as well, but its
default path, the fused step inside a whole-step CUDA graph, is only taken when
the nonbonded force is the plugin's (README, "Whole-step CUDA graphs"). With a
stock NonbondedForce the default path would not be tested. The force is the same
on both sides, so only the integrator differs.

The plugin's mean volume under nv.MonteCarloBarostat is compared with stock's
barostat by test_nv_barostat.py, so it is not repeated here.
"""
import copy
import functools
import os
from unittest.mock import patch

import numpy as np
import pytest

import openmm as mm
import openmm.app as app
from openmm import unit

from conftest import record_observed, replace_nonbonded

TEMPERATURE = 300.0          # K
FRICTION = 1.0               # 1/ps, for the statistical tests
CONSTRAINT_TOL = 1e-5        # both integrators, set explicitly
KB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

# The plugin's execution switches, read when a Context is created. The tests set
# each one they mean, and clear the rest, so that a value from the environment
# (run_docker forwards them) does not change what "default" means here.
_SWITCHES = ("CUPO_PLUGIN_FUSED_INTEGRATOR", "CUPO_PLUGIN_CUDA_GRAPH",
             "CUPO_PLUGIN_INTEGRATOR_GRAPH", "CUPO_PLUGIN_WHOLE_STEP_GRAPH",
             "CUPO_PLUGIN_WHOLE_STEP_BATCH", "CUPO_PLUGIN_BONDED_SCHEDULE")

# The paths of the plugin's integrator compared with stock:
#   default   the fused step, inside a whole-step CUDA graph;
#   separate  the separate integration kernels (still inside the whole-step graph);
#   eager     the fused step with every plugin CUDA graph off.
PATHS = {
    "default": {},
    "separate": {"CUPO_PLUGIN_FUSED_INTEGRATOR": "0"},
    "eager": {"CUPO_PLUGIN_CUDA_GRAPH": "0"},
}


@pytest.fixture(autouse=True)
def _environment(monkeypatch):
    for name in _SWITCHES:
        monkeypatch.delenv(name, raising=False)
    # The nonbonded force picks its FFT by timing unless told not to. Off, so the
    # stock and the plugin Contexts compute the same forces from the same positions.
    monkeypatch.setenv("CUPO_COMPACT_AUTOTUNE", "0")


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _water():
    """About 890 rigid TIP3P waters in a 3 nm box (SETTLE). 3 nm is just above
    three times the 0.9 nm cutoff, the smallest box the nonbonded force takes."""
    from test_npt import _build_water
    system, positions = _build_water(3.0)
    return system, positions


def _dhfr():
    """DHFR in water, 23,558 atoms, HBonds constraints: SETTLE for the water and
    SHAKE for the protein's bonds to hydrogen. The smallest protein in data/."""
    from data.datasets import load_dataset, template_pins
    dataset = load_dataset("dhfr")
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = ff.createSystem(dataset.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True,
                             residueTemplates=template_pins(dataset.topology))
    return system, dataset.coords


def _lj():
    """256 charged Lennard-Jones particles of argon's mass, no bonds and no
    constraints."""
    from test_plugin_vs_stock import _create_lj_charged_box
    system, positions = _create_lj_charged_box()
    return system, np.array([p.value_in_unit(unit.nanometer) for p in positions])


_BUILDERS = {"water": _water, "dhfr": _dhfr, "lj": _lj}


def _has_dhfr():
    from data.datasets import available_datasets
    return "dhfr" in available_datasets()


def _skip_missing(name):
    if name == "dhfr" and not _has_dhfr():
        pytest.skip("data/dhfr is not in this tree")


def _without_cm_remover(system):
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), mm.CMMotionRemover):
            system.removeForce(i)
    return system


def _masses(system):
    return np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                     for i in range(system.getNumParticles())])


@functools.lru_cache(maxsize=None)
def _prepared(name):
    """(System, positions, velocities) of an equilibrated start, shared by every
    test of the system.

    The System has nv.NonbondedForce and no CMMotionRemover. The start is
    minimised and equilibrated at 300 K with stock's integrator, its constraints
    are applied to 1e-10 and its total momentum is zero, so both integrators
    start from one state that satisfies both of their constraint tolerances.
    """
    system, positions = _BUILDERS[name]()
    _without_cm_remover(system)
    replace_nonbonded(system)
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, 0.002)
    integrator.setRandomNumberSeed(9137)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": "mixed"})
    context.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(context, 10.0, 500)
    context.setVelocitiesToTemperature(TEMPERATURE, 3301)
    integrator.step(2000 if name == "water" else 500)
    context.applyConstraints(1e-10)
    context.applyVelocityConstraints(1e-10)
    state = context.getState(getPositions=True, getVelocities=True)
    x = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    v = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    m = _masses(system)
    v = v - (m[:, None] * v).sum(axis=0) / m.sum()
    del context, integrator
    return system, np.array(x), np.array(v)


def _constraints(system):
    """(i, j, distance) arrays of the System's constraints."""
    rows = [system.getConstraintParameters(k) for k in range(system.getNumConstraints())]
    if not rows:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    i, j, d = zip(*rows)
    return np.array(i), np.array(j), np.array([q.value_in_unit(unit.nanometer) for q in d])


def _constraint_deviation(constraints, x):
    """The largest |r/d - 1| over the constraints, 0 with none."""
    i, j, d = constraints
    if len(d) == 0:
        return 0.0
    return float(np.abs(np.linalg.norm(x[i] - x[j], axis=1) / d - 1).max())


def _degrees_of_freedom(system):
    """3 per massive particle, less one per constraint, less 3 for the centre of
    mass, which both sides remove."""
    m = _masses(system)
    return 3 * int((m > 0).sum()) - system.getNumConstraints() - 3


def _context(system, integrator, precision, env=None):
    with patch.dict(os.environ, env or {}):
        return mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                          {"Precision": precision})


def _plugin_integrator(friction, dt, com_frequency, seed=0):
    import nv_openmm_plugin as nv
    integrator = nv.LangevinMiddleIntegrator(TEMPERATURE, friction, dt, com_frequency, seed)
    integrator.setConstraintTolerance(CONSTRAINT_TOL)
    return integrator


def _stock_integrator(friction, dt, seed=0):
    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, friction, dt)
    integrator.setConstraintTolerance(CONSTRAINT_TOL)
    if seed:
        integrator.setRandomNumberSeed(seed)
    return integrator


# ---------------------------------------------------------------------------
# Deterministic: friction 0
# ---------------------------------------------------------------------------

# 100 steps in one step() call: the whole-step graph is captured after 55 warmup
# steps (README, "Whole-step CUDA graphs"), so 45 of them are graph replays.
STEPS = 100

# The stock spread. Runs of stock at the same precision, one from the start and
# six from the start with every coordinate multiplied by 1 +- 2^-24 (a random sign
# per coordinate, a new draw of the signs each run): one float32 rounding of the
# positions. That is the smallest difference two correct implementations can
# have once any of their arithmetic differs, since both compute the forces in
# float32 at either precision; the trajectories then diverge by however much 100
# steps amplify it. The spread of a quantity is the largest of its six errors.
#
# The tolerances, as multiples of the spread:
#   positions, velocities  3. Two runs that each lie within the spread of a
#       reference lie within twice the spread of each other, and the third is
#       for scatter. These errors are sums over every atom, and their spread
#       changes little from one draw of the signs to the next.
#   energies  5. Each is one number, which can come out near zero in one run
#       and not in the next. If the plugin's error is one more sample of the
#       same distribution as the six, 5 times their largest is exceeded with a
#       probability of 2e-4 (for a normal distribution).
# On an RTX 6000 Ada the plugin's errors were at most half of these.
_PERTURBATION_SEEDS = (1, 2, 3, 4, 5, 6)
_SPREAD_FACTOR = {"positions": 3.0, "velocities": 3.0, "potential": 5.0, "kinetic": 5.0}


def _trajectory(system, x, v, precision, dt, plugin, env=None):
    """The state after STEPS steps at friction 0, and the plugin's whole-step
    graph (captures, replays)."""
    if plugin:
        integrator = _plugin_integrator(0.0, dt, 0)
    else:
        integrator = _stock_integrator(0.0, dt)
    context = _context(system, integrator, precision, env)
    context.setPositions(x)
    context.setVelocities(v)
    integrator.step(STEPS)
    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    result = {
        "x": np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
        "v": np.array(state.getVelocities(asNumpy=True)
                      .value_in_unit(unit.nanometer / unit.picosecond)),
        "pe": state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        "ke": state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole),
    }
    graph = ((integrator.getWholeStepGraphCaptureCount(),
              integrator.getWholeStepGraphReplayCount()) if plugin else None)
    del context, integrator
    return result, graph


def _errors(got, ref, x0):
    """Relative errors of a final state against a reference one.

    Positions: the RMS difference over the RMS displacement from the start, so
    the error is measured against how far the atoms moved, not against where the
    box happens to sit. Velocities: the RMS difference over the RMS velocity.
    Energies: the difference over the reference kinetic energy, the scale of the
    energy's fluctuations; the potential energy's own magnitude is set by its
    zero and would hide differences.
    """
    return {
        "positions": np.linalg.norm(got["x"] - ref["x"]) / np.linalg.norm(ref["x"] - x0),
        "velocities": np.linalg.norm(got["v"] - ref["v"]) / np.linalg.norm(ref["v"]),
        "potential": abs(got["pe"] - ref["pe"]) / ref["ke"],
        "kinetic": abs(got["ke"] - ref["ke"]) / ref["ke"],
    }


@pytest.mark.parametrize("precision", ["mixed", "single"])
@pytest.mark.parametrize("dt_fs", [2, 4])
@pytest.mark.parametrize("name", ["water", "dhfr", "lj"])
def test_deterministic(request, plugin_dir, name, dt_fs, precision):
    _skip_missing(name)
    system, x, v = _prepared(name)
    dt = dt_fs * 1e-3
    constraints = _constraints(system)

    ref, _ = _trajectory(system, x, v, precision, dt, plugin=False)
    spread = {}
    for seed in _PERTURBATION_SEEDS:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=x.shape)
        perturbed, _ = _trajectory(system, x * (1 + signs * 2.0**-24), v, precision, dt,
                                   plugin=False)
        for key, e in _errors(perturbed, ref, x).items():
            spread[key] = max(spread.get(key, 0.0), e)
    tolerance = {key: _SPREAD_FACTOR[key] * s for key, s in spread.items()}
    print(f"{name} {dt_fs} fs {precision}: stock spread "
          + " ".join(f"{k} {s:.2e}" for k, s in spread.items()))

    # Stock's own constraint deviation after the run, for the report.
    print(f"  stock constraint deviation {_constraint_deviation(constraints, ref['x']):.2e}")
    failures = []
    worst = dict.fromkeys(tolerance, 0.0)
    for path, env in PATHS.items():
        got, graph = _trajectory(system, x, v, precision, dt, plugin=True, env=env)
        errors = _errors(got, ref, x)
        deviation = _constraint_deviation(constraints, got["x"])
        print(f"  {path:8s} " + " ".join(f"{k} {e:.2e} ({e / tolerance[k]:.2f} of tol)"
                                         for k, e in errors.items())
              + f" constraint {deviation:.2e} graph {graph}")
        for key, e in errors.items():
            worst[key] = max(worst[key], e)
            if not e <= tolerance[key]:
                failures.append(f"{path}: {key} relative error {e:.3e} > {tolerance[key]:.3e}")
        # The constraints hold to the integrator's tolerance at both precisions:
        # a float32 rounding of a coordinate is at most 4e-7 nm in these boxes,
        # 4e-6 of a 0.1 nm bond.
        if not deviation <= CONSTRAINT_TOL:
            failures.append(f"{path}: constraint deviation {deviation:.3e} > {CONSTRAINT_TOL}")
        # The paths are what they say they are: the graph replays where the
        # graphs are on, and does not where they are off.
        replays = graph[1]
        if path == "eager" and replays != 0:
            failures.append(f"{path}: {replays} whole-step graph replays with graphs off")
        if path != "eager" and replays < STEPS - 55:
            failures.append(f"{path}: only {replays} whole-step graph replays")
    for key, e in worst.items():
        record_observed(request, f"{key}, worst of the paths", tolerance[key], e)
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Statistical: friction 1/ps, 300 K
# ---------------------------------------------------------------------------

# Steps between samples. 100, so that each step() call is long enough for the
# whole-step graph to be captured and replayed (55 warmup steps), and the plugin's
# default path is the one that samples.
_SAMPLE = 100

# (equilibration steps, production steps). 2 fs, so 2 ps of equilibration from
# the shared start, then 200 ps of water and 80 ps of DHFR, in 20 blocks: 10 ps
# and 4 ps, much longer than the kinetic energy's correlation time at friction
# 1/ps (1/(2 gamma) = 0.5 ps).
_RUNS = {"water": (1000, 100000), "dhfr": (1000, 40000)}
_BLOCKS = 20

# The threshold for a difference of two means: 4 standard errors of the
# difference, with each standard error from the block averages. With 20 blocks
# the standard error is itself uncertain by about a sixth, and 4 keeps the chance
# that a correct integrator fails one of the comparisons near 1e-3.
_SIGMAS = 4.0


def _block_mean(series, blocks=_BLOCKS):
    """(mean, standard error) of a series from the means of `blocks` blocks."""
    s = np.asarray(series, dtype=float)
    s = s[:len(s) // blocks * blocks].reshape(blocks, -1)
    means = s.mean(axis=1)
    return means.mean(), means.std(ddof=1) / np.sqrt(blocks)


def _block_variance(series, blocks=_BLOCKS):
    """(variance, standard error) of a series: the block means of the squared
    deviation from the mean of the whole series, so that slow fluctuations
    count."""
    s = np.asarray(series, dtype=float)
    s = s[:len(s) // blocks * blocks]
    return _block_mean((s - s.mean()) ** 2, blocks)


def _sample(system, x, v, plugin, equilibration, production, sample=_SAMPLE,
            com_frequency=100):
    """Kinetic and potential energies, constraint deviations and total momenta,
    every `sample` steps of a production run at 300 K and friction 1/ps, with
    fixed seeds on both sides.

    The plugin removes the centre-of-mass motion itself every `com_frequency`
    steps (100 is its default). Stock has a CMMotionRemover, as the force field
    makes one; on the CUDA platform of OpenMM 8.4 it removes the motion every
    step whatever its frequency (test_com_removal).
    """
    dt = 0.002
    if plugin:
        integrator = _plugin_integrator(FRICTION, dt, com_frequency, seed=24601)
    else:
        system = copy.deepcopy(system)
        system.addForce(mm.CMMotionRemover(1))
        integrator = _stock_integrator(FRICTION, dt, seed=24601)
    context = _context(system, integrator, "mixed")
    context.setPositions(x)
    context.setVelocities(v)
    integrator.step(equilibration)
    constraints = _constraints(system)
    m = _masses(system)
    out = {"ke": [], "pe": [], "constraint": [], "momentum": []}
    for _ in range(production // sample):
        integrator.step(sample)
        state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        xs = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vs = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        out["ke"].append(state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole))
        out["pe"].append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        out["constraint"].append(_constraint_deviation(constraints, np.asarray(xs)))
        out["momentum"].append((m[:, None] * np.asarray(vs)).sum(axis=0))
    replays = integrator.getWholeStepGraphReplayCount() if plugin else None
    del context, integrator
    return {k: np.array(s) for k, s in out.items()}, replays


def _compare(request, failures, label, a, b, expected=None):
    """Plugin `a` against stock `b`, each (value, standard error): the
    difference must be within _SIGMAS standard errors of the difference."""
    threshold = _SIGMAS * np.hypot(a[1], b[1])
    difference = a[0] - b[0]
    line = (f"  {label:22s} plugin {a[0]:.6g} +- {a[1]:.2g}  stock {b[0]:.6g} +- {b[1]:.2g}"
            f"  difference {difference:+.3g} ({abs(difference) / threshold:.2f} of the"
            f" threshold {threshold:.3g})")
    if expected is not None:
        line += f"  expected {expected:.6g}"
    print(line)
    record_observed(request, f"{label}, plugin - stock", threshold, abs(difference))
    if not abs(difference) <= threshold:
        failures.append(f"{label}: plugin {a[0]:.6g} +- {a[1]:.2g}, stock {b[0]:.6g} "
                        f"+- {b[1]:.2g}, difference {difference:.3g} > {threshold:.3g}")


@pytest.mark.parametrize("name", ["water", "dhfr"])
def test_statistics(request, plugin_dir, name):
    _skip_missing(name)
    system, x, v = _prepared(name)
    equilibration, production = _RUNS[name]
    stock, _ = _sample(system, x, v, False, equilibration, production)
    plugin, replays = _sample(system, x, v, True, equilibration, production)
    dof = _degrees_of_freedom(system)
    kt = KB * TEMPERATURE
    print(f"{name}: {dof} degrees of freedom, {len(plugin['ke'])} samples, {_BLOCKS} blocks,"
          f" plugin whole-step graph replays {replays}")

    failures = []
    temperature = {side: _block_mean(2 * s["ke"] / (dof * KB))
                   for side, s in (("plugin", plugin), ("stock", stock))}
    _compare(request, failures, "temperature (K)", temperature["plugin"],
             temperature["stock"], TEMPERATURE)
    _compare(request, failures, "kinetic energy mean", _block_mean(plugin["ke"]),
             _block_mean(stock["ke"]), dof * kt / 2)
    _compare(request, failures, "kinetic energy var", _block_variance(plugin["ke"]),
             _block_variance(stock["ke"]), dof * kt * kt / 2)
    _compare(request, failures, "potential energy mean", _block_mean(plugin["pe"]),
             _block_mean(stock["pe"]))

    # Equipartition, with the constraints counted: each side's temperature is
    # 300 K within its error bars. A wrong count of the degrees of freedom is off
    # by the constraints' share of them, a third for rigid water.
    for side, t in temperature.items():
        if not abs(t[0] - TEMPERATURE) <= _SIGMAS * t[1]:
            failures.append(f"{side} temperature {t[0]:.2f} +- {t[1]:.2f} K is not "
                            f"{TEMPERATURE} K within {_SIGMAS} standard errors")

    worst = {side: s["constraint"].max() for side, s in (("plugin", plugin), ("stock", stock))}
    print(f"  constraint deviation   plugin {worst['plugin']:.2e}  stock {worst['stock']:.2e}"
          f"  tolerance {CONSTRAINT_TOL}")
    record_observed(request, "constraint deviation, plugin", CONSTRAINT_TOL, worst["plugin"])
    if not worst["plugin"] <= CONSTRAINT_TOL:
        failures.append(f"plugin constraint deviation {worst['plugin']:.3e} > {CONSTRAINT_TOL}")
    if replays < (production // _SAMPLE) * (_SAMPLE - 55):
        failures.append(f"only {replays} whole-step graph replays")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Centre-of-mass removal
# ---------------------------------------------------------------------------

def test_com_removal(request, plugin_dir):
    """The plugin keeps the total momentum P as small as its removal frequency
    allows, and as small as stock's CMMotionRemover does at the same frequency.

    The forces sum to zero, so between removals only the Langevin noise changes
    P: each step multiplies it by a = exp(-gamma dt) and adds noise of variance
    (1 - a^2) M kT per component, for total mass M. k steps after a removal,
    <|P|^2> = 3 M kT (1 - a^(2k)), and with no removal it grows to 3 M kT.

    Every 100 steps, the plugin's default: sampled every 37 steps, which is prime
    to 100, the samples fall evenly over k = 1..100, and <|P|^2> must be the mean
    of 3 M kT (1 - a^(2k)) over them, about 0.18 * 3 M kT. Stock cannot be run
    at this frequency: on the CUDA platform of OpenMM 8.4, CMMotionRemover removes
    the motion every step whatever its frequency (measured: <|P|^2> 437 at
    frequencies 1, 10, 100 and 1000 on this box, where the CPU platform gives 475
    at 1 and 18913 at 100).

    Every step: the plugin with comFrequency 1 against stock with
    CMMotionRemover(1).
    """
    system, x, v = _prepared("water")
    m = _masses(system)
    kt = KB * TEMPERATURE
    a2 = np.exp(-2 * FRICTION * 0.002)
    unremoved = 3 * m.sum() * kt
    failures = []

    def p2(samples):
        return _block_mean((samples["momentum"] ** 2).sum(axis=1))

    every100, _ = _sample(system, x, v, True, 1000, 37 * 1000, sample=37, com_frequency=100)
    got = p2(every100)
    expected = unremoved * np.mean([1 - a2 ** k for k in range(1, 101)])
    print(f"every 100 steps: plugin <|P|^2> {got[0]:.4g} +- {got[1]:.2g} (amu nm/ps)^2,"
          f" expected {expected:.4g}, {abs(got[0] - expected) / (_SIGMAS * got[1]):.2f}"
          f" of the threshold; with no removal {unremoved:.4g}")
    record_observed(request, "mean P^2 every 100 steps, plugin - expected",
                    _SIGMAS * got[1], abs(got[0] - expected))
    if not abs(got[0] - expected) <= _SIGMAS * got[1]:
        failures.append(f"every 100 steps: plugin <|P|^2> {got[0]:.4g} +- {got[1]:.2g},"
                        f" expected {expected:.4g}")

    stock, _ = _sample(system, x, v, False, 1000, 37 * 1000, sample=37)
    every1, _ = _sample(system, x, v, True, 1000, 37 * 1000, sample=37, com_frequency=1)
    print(f"every step (expected at most {unremoved * (1 - a2):.4g}):")
    _compare(request, failures, "mean P^2 every step", p2(every1), p2(stock))
    assert not failures, "\n".join(failures)
