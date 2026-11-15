#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked GPU clocks for the benchmarks, and what the GPU did while they ran.

The benchmarks lock the clocks themselves, at the values in tests/hw_config.py, for
the whole session (the `gpu_hw` fixture), and reset them at the end.

A lock is a ceiling, not a floor: under load a card at its power or temperature
limit runs below it. So the lock is checked with the GPU idle, where a locked GPU
reads its lock: after locking, the GPU is left idle for LOCK_SETTLE_SECS, and the
clocks it then reads, once they stop changing, are what every measurement is held
to; before every measurement
the GPU is given time to get back to them (MEASURE_SETTLE_SECS). What the
clocks did during a measurement is recorded, not checked. A benchmark brackets its
timed work with two calls:

    gpu_hw.wait_for_locked_clocks()
    ...timed work...
    telemetry = gpu_hw.get_telemetry()

While a session runs, one nvidia-smi process samples the GPU every 200 ms: graphics
and memory clock, power, temperature, clock event reasons, utilization. A
measurement's statistics are over the samples between its two calls, so they
describe the measurement and not the setup around it. Each measurement's statistics -- min, max, median and mean of clock, power and
temperature -- go into its golden.

Shared by the core benchmarks (tests/conftest.py imports the fixtures) and the
OpenMM plugin's (plugins/openmm/tests/conftest.py, and its CLI, load this file by
path). Standard library and pytest only; pytest only for the fixture.

Also a command line, for measuring by hand with the clocks the benchmarks use:

    tests/gpu_clocks.py lock            # lock, wait for the lock, show the GPU
    tests/gpu_clocks.py lock --load 10  # ... and 10 s under load (needs cupy)
    tests/gpu_clocks.py status          # clocks, power, temperature now
    tests/gpu_clocks.py unlock          # give the GPU its clocks back

The GPU is device 0, the only one the job's container sees in CI. Locking needs
`sudo -n nvidia-smi`; the dev image's sudoers allows it.
"""

import dataclasses
import os
import importlib.util
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import pytest
except ImportError:         # the command line, on a machine without pytest
    pytest = None

GPU_INDEX = "0"

# For a GPU missing from the table: the clocks are locked at this much of its maximum.
FALLBACK_FRAC = 0.95

# A sample counts as taken under load when its utilization (nvidia-smi's
# utilization.gpu: the share of the sample period a kernel was running) is at least
# this. A window covers a whole measurement, setup included -- loading a system,
# making a context, compiling -- and the GPU is idle or half idle for parts of that.
# The statistics are given over every sample ("all") and again over the samples under
# load ("under_load"), which show the clocks the measured work actually ran at.
UNDER_LOAD_UTILIZATION_PCT = 50

SAMPLE_MS = 200

# How far a clock may read from the one it is held to. A -lgc request snaps to the
# nearest supported clock step (1510 -> 1515, 1610 -> 1620, under 1%); an unlocked
# clock, or one the card cannot reach, is further off than this.
CLOCK_TOLERANCE_FRAC = 0.02

# A lock is a ceiling, not a floor. Under load a card at its power or temperature
# limit can run well below it, and when the work stops the clock goes back up to
# the lock: an idle locked GPU reads its locked clock. So a lock is checked with the GPU idle, and the
# GPU is given time to get there: right after locking (LOCK_SETTLE_SECS), and before
# each measurement, which may follow the previous one by milliseconds
# (MEASURE_SETTLE_SECS, BenchClocks.wait_for_locked_clocks).
LOCK_SETTLE_SECS = 5
MEASURE_SETTLE_SECS = 30
POLL_SECS = 0.5

# Patched by the tests, which have no GPU to wait for.
_sleep = time.sleep
_monotonic = time.monotonic

# What nvidia-smi is asked for, and the name each field goes by in a Sample.
_QUERY = (
    ("timestamp", "timestamp"),
    ("clocks.gr", "graphics_mhz"),
    ("clocks.mem", "memory_mhz"),
    ("power.draw", "power_w"),
    ("temperature.gpu", "temperature_c"),
    ("clocks_event_reasons.active", "event_reasons"),
    ("utilization.gpu", "utilization_pct"),
)

# The fields a measurement's statistics are made of.
STAT_FIELDS = ("graphics_mhz", "memory_mhz", "power_w", "temperature_c")


def _load_sibling(name):
    """A module next to this file, loaded once and shared, whoever asks first."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / ("%s.py" % name))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


hw_config = _load_sibling("hw_config")


# ── nvidia-smi ───────────────────────────────────────────────────────────────

def _smi(*args):
    return subprocess.check_output(["nvidia-smi", "-i", GPU_INDEX] + list(args),
                                   text=True, timeout=30).strip()


def gpu_name():
    return _smi("--query-gpu=name", "--format=csv,noheader").splitlines()[0].strip()


def max_clocks():
    """(graphics, memory) maximum clocks in MHz."""
    out = _smi("--query-gpu=clocks.max.graphics,clocks.max.memory",
               "--format=csv,noheader,nounits").splitlines()[0]
    gr, mem = (int(float(v)) for v in out.split(","))
    return gr, mem


def current_clocks():
    """(graphics, memory) clocks the GPU reads now, in MHz."""
    out = _smi("--query-gpu=clocks.gr,clocks.mem",
               "--format=csv,noheader,nounits").splitlines()[0]
    gr, mem = (int(float(v)) for v in out.split(","))
    return gr, mem


def supported_clocks():
    """{memory MHz: [graphics MHz, ...]}: the clock pairs the driver offers
    (nvidia-smi --query-supported-clocks), or None when it does not say."""
    try:
        out = _smi("--query-supported-clocks=memory,graphics",
                   "--format=csv,noheader,nounits")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    pairs = {}
    for line in out.splitlines():
        try:
            mem, gr = (int(float(v)) for v in line.split(","))
        except ValueError:
            continue
        pairs.setdefault(mem, []).append(gr)
    return {mem: sorted(grs) for mem, grs in pairs.items()} or None


def offered_for(requested, supported):
    """The clocks the driver settles on for `requested` (a LockedClocks): for memory
    and then for graphics among the clocks offered with it, the lowest offered at or
    above the request, or the highest if none is. That is what the driver does:
    asked for a clock between two steps it runs at the higher one."""
    def up(want, offered):
        above = [c for c in offered if c >= want]
        return min(above) if above else max(offered)
    mem = up(requested.memory_mhz, supported)
    return LockedClocks(up(requested.graphics_mhz, supported[mem]), mem)


def _sudo_smi(what, *args):
    """Run `sudo -n nvidia-smi -i 0 args`, and return what it said; what ("lock",
    "reset") names it in the error."""
    cmd = ["sudo", "-n", "nvidia-smi", "-i", GPU_INDEX] + list(args)
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if cp.returncode != 0:
        said = (cp.stderr or cp.stdout).strip()
        raise RuntimeError(
            "cannot %s the GPU clocks: `%s` failed (exit %d)%s\n"
            "The benchmarks set the clocks themselves. That takes passwordless sudo for\n"
            "nvidia-smi -- the dev image has it (Dockerfile); elsewhere add\n"
            "  <user> ALL=(root) NOPASSWD: %s\n"
            "to /etc/sudoers.d/ -- and, in a container, --cap-add=SYS_ADMIN, without\n"
            "which even root may not change clocks. ./run_docker does both."
            % (what, " ".join(cmd), cp.returncode, ": " + said if said else "",
               shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"))
    return cp.stdout + cp.stderr


def lock(graphics_mhz, memory_mhz):
    _sudo_smi("lock", "-lgc", "%d,%d" % (graphics_mhz, graphics_mhz))
    _sudo_smi("lock", "-lmc", "%d,%d" % (memory_mhz, memory_mhz))


def unlock():
    """Give the GPU its clocks back. Best effort: a teardown that raises would hide
    whatever failed before it."""
    for flag in ("-rgc", "-rmc"):
        try:
            _sudo_smi("reset", flag)
        except Exception as e:  # noqa: BLE001 -- teardown, reported not raised
            print("gpu_clocks: could not reset clocks (%s): %s" % (flag, e),
                  file=sys.stderr)


# ── Telemetry ────────────────────────────────────────────────────────────────

Sample = dataclasses.make_dataclass("Sample", [name for _, name in _QUERY])


def parse_sample(line):
    """One line of nvidia-smi's csv,noheader,nounits output -> Sample, or None for a
    line that is not one (a warning, a blank). A field nvidia-smi cannot read
    ([N/A]) is None."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(_QUERY):
        return None
    values = {}
    for (_, name), raw in zip(_QUERY, parts):
        if name in ("timestamp", "event_reasons"):
            values[name] = raw
            continue
        try:
            values[name] = float(raw)
        except ValueError:
            values[name] = None
    if values["graphics_mhz"] is None and values["memory_mhz"] is None:
        return None
    return Sample(**values)


def stats(samples):
    """{field: {min, max, median, mean}} over the samples, for each of STAT_FIELDS,
    plus how many samples there were. A field with no readings is left out."""
    out = {"samples": len(samples)}
    for field in STAT_FIELDS:
        vals = [getattr(s, field) for s in samples if getattr(s, field) is not None]
        if vals:
            out[field] = {"min": min(vals), "max": max(vals),
                          "median": round(statistics.median(vals), 2),
                          "mean": round(statistics.fmean(vals), 1)}
    return out


class GpuTelemetry:
    """One nvidia-smi sampling the GPU for as long as this lives.

    A thread reads its output the whole time -- left unread, the pipe would fill and
    nvidia-smi would stop -- and keeps every sample, with the time it arrived, so
    that the samples of any stretch of the session can be asked for afterwards
    (since). Five samples a second is a few MB over an hour's session.
    """

    def __init__(self, command=None):
        self.command = command or [
            "nvidia-smi", "-i", GPU_INDEX,
            "--query-gpu=" + ",".join(q for q, _ in _QUERY),
            "--format=csv,noheader,nounits", "-lms", str(SAMPLE_MS)]
        self._proc = subprocess.Popen(self.command, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
        self._samples = []          # (time.monotonic() on arrival, Sample)
        self._guard = threading.Lock()
        self.unparsed = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self._proc.stdout:
            sample = parse_sample(line)
            if sample is None:
                if line.strip():
                    self.unparsed = (self.unparsed + [line.strip()])[-5:]
                continue
            with self._guard:
                self._samples.append((time.monotonic(), sample))

    def since(self, start):
        """The samples that arrived at or after `start` (a time.monotonic())."""
        with self._guard:
            return [sample for t, sample in self._samples if t >= start]

    def running(self):
        return self._proc.poll() is None

    def close(self):
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()


# ── Locked clocks ─────────────────────────────────────────────────────────────

# Set to anything but "" or "0", the benchmarks run when the clocks cannot be
# locked: at whatever clocks the GPU has, marked UNLOCKED, and compared with nothing.
ALLOW_UNLOCKED_VAR = "CUPO_ALLOW_UNLOCKED_CLOCKS"
# The warning such a run's measurements carry.
UNLOCKED = "clocks not locked"
# True once a session's BenchClocks runs unlocked (BenchClocks.run_unlocked): what
# the benchmarks then measure is compared with nothing.
SESSION_UNLOCKED = False


def allow_unlocked(environ=None):
    """Whether CUPO_ALLOW_UNLOCKED_CLOCKS is set (ALLOW_UNLOCKED_VAR)."""
    environ = os.environ if environ is None else environ
    return environ.get(ALLOW_UNLOCKED_VAR, "") not in ("", "0")


def lock_failure_message(error):
    """What a benchmark says when the clocks cannot be locked: why, how to fix it,
    and the variable that runs the benchmarks unlocked instead."""
    return ("cannot lock the GPU clocks the benchmarks run at (%s). Locking takes "
            "passwordless sudo for nvidia-smi and, in a container, the SYS_ADMIN "
            "capability, which ./run_docker gives it: run the benchmarks through "
            "./run_docker or ./run_test. Or set %s=1 to run them at whatever clocks "
            "the GPU has, measured and printed, and not compared with reference "
            "numbers." % (str(error).splitlines()[0], ALLOW_UNLOCKED_VAR))


@dataclasses.dataclass(frozen=True)
class LockedClocks:
    """The clocks the GPU is locked at, and what was not as asked when it was locked
    (warnings, strings; recorded with every measurement in the golden)."""
    graphics_mhz: int
    memory_mhz: int
    warnings: tuple = ()


def off_by(clocks, held):
    """The clocks of `clocks` (graphics, memory) further than CLOCK_TOLERANCE_FRAC
    from `held` (a LockedClocks), as [(name, read, held)]."""
    return [(name, read, want)
            for name, read, want in (("graphics", clocks[0], held.graphics_mhz),
                                     ("memory", clocks[1], held.memory_mhz))
            if abs(read - want) > CLOCK_TOLERANCE_FRAC * want]


def wait_for_clocks(held, secs):
    """Poll the clocks until they are within CLOCK_TOLERANCE_FRAC of `held`, for at
    most `secs` (0: read once). Returns (clocks, []) once they are, or the last
    reading and what was off in it when the time ran out."""
    deadline = _monotonic() + secs
    while True:
        clocks = current_clocks()
        off = off_by(clocks, held)
        if not off or _monotonic() >= deadline:
            return clocks, off
        _sleep(POLL_SECS)


def wait_for_stable_clocks(secs, same=4):
    """Poll the clocks every POLL_SECS for `secs`, and after that until `same`
    readings in a row agree, for at most `secs` more. Returns (clocks, stable): the
    last reading, and whether the last `same` agreed.

    The whole of `secs` is waited even when the readings agree sooner. Right after
    nvidia-smi -lmc a card can still read its idle memory clock before going to the
    locked one, and how long a card takes to apply a lock is not known; a few
    seconds of agreeing readings could all be from before it."""
    start = _monotonic()
    readings = [current_clocks()]
    while True:
        elapsed = _monotonic() - start
        agree = len(readings) >= same and len(set(readings[-same:])) == 1
        if elapsed >= secs and agree:
            return readings[-1], True
        if elapsed >= 2 * secs:
            return readings[-1], False
        _sleep(POLL_SECS)
        readings.append(current_clocks())


def describe(off):
    return ", ".join("%s %d MHz, held to %d" % o for o in off)


def table_line(name, clocks):
    return "    %r: %r," % (name, tuple(clocks))


# The tests/hw_config.py lines this session worked out, by GPU name, for the summary
# at the end of the run (report_suggestions). They are printed where they are worked
# out too, but that is inside a fixture, and pytest captures a fixture's output with
# its test's and shows it only for a test that fails -- which in CI is never.
SUGGESTED = {}


def report_suggestions(terminalreporter):
    """For pytest_terminal_summary: the tests/hw_config.py lines this session worked
    out, where nothing captures them."""
    if SUGGESTED:
        terminalreporter.write_sep("-", "lines for tests/hw_config.py")
        for line in SUGGESTED.values():
            terminalreporter.write_line(line)


class BenchClocks:
    """The session's GPU, the clocks it is locked at, and its telemetry."""

    def __init__(self, telemetry=None, name=None):
        self.gpu_name = name or gpu_name()
        self.telemetry = telemetry or GpuTelemetry()
        self._requested = None
        self.locked = None
        # When the last wait_for_locked_clocks() returned: where get_telemetry()
        # starts.
        self._measure_start = None
        # Set by the first measurement whose wait for the clocks ran out. Every later
        # one reads the clocks once instead: a GPU that did not come back to its lock
        # in MEASURE_SETTLE_SECS is not going to, and each benchmark finding that out
        # again would add half a minute to a job that is failing anyway.
        self.settle_failed = False
        # True after run_unlocked(): the measurements are at whatever clocks the GPU
        # has, and compared with nothing.
        self.unlocked = False

    def run_unlocked(self):
        """In place of lock(), when the clocks cannot be locked and
        CUPO_ALLOW_UNLOCKED_CLOCKS is set: `locked` becomes the clocks the GPU reads
        now, with the UNLOCKED warning, which every measurement records, and the
        waits before measurements do not wait for a lock."""
        gr, mem = current_clocks()
        global SESSION_UNLOCKED
        self.locked = LockedClocks(gr, mem, (UNLOCKED,))
        self.unlocked = SESSION_UNLOCKED = True
        print("\ngpu_clocks: %s: %s; running at %d/%d MHz, and comparing with nothing"
              % (self.gpu_name, UNLOCKED, gr, mem), flush=True)
        return self.locked

    def requested(self):
        """The clocks to lock at: the table's, or FALLBACK_FRAC of the maximum for a
        GPU the table does not have."""
        if self._requested is None:
            entry = hw_config.CLOCKS.get(self.gpu_name)
            if entry:
                gr, mem = entry
            else:
                max_gr, max_mem = max_clocks()
                gr, mem = int(max_gr * FALLBACK_FRAC), int(max_mem * FALLBACK_FRAC)
                SUGGESTED[self.gpu_name] = table_line(self.gpu_name, (gr, mem))
                print("\ngpu_clocks: %r is not in tests/hw_config.py; locking at %d%% "
                      "of its maximum clocks (%d/%d MHz). Add:\n%s"
                      % (self.gpu_name, FALLBACK_FRAC * 100, max_gr, max_mem,
                         SUGGESTED[self.gpu_name]), flush=True)
            self._requested = LockedClocks(int(gr), int(mem))
        return self._requested

    def lock(self):
        """Lock the GPU at requested(), wait until the idle clocks stop changing, and
        return what they read: the clocks every measurement is then held to.

        Two things are checked, apart, against the clocks the driver offers
        (supported_clocks):

        - The table. Each clock of the request is matched to the one the driver
          settles on (offered_for). One more than CLOCK_TOLERANCE_FRAC away is a
          warning, with the table line to use: the table asks for a clock this card
          does not have. Not a failure, because cards of one model do not all offer
          the same clocks.
        - The card. What it reads once the clocks settle is compared with those
          offered clocks. A clock more than CLOCK_TOLERANCE_FRAC off is
          measured all the same, with a warning, which goes into the golden with
          every measurement and the plots mark: a locked clock is not always
          held. A graphics clock above it is a GPU that is not locked, and fails.

        When the driver does not list its clocks, the request stands in for them,
        and the table is not checked.

        A lock that nvidia-smi refused fails in lock(), before any of this.
        """
        requested = self.requested()
        supported = supported_clocks()
        target = offered_for(requested, supported) if supported else requested
        warnings = []
        if supported:
            warnings += ["tests/hw_config.py asks for %s %d MHz, the card offers %d"
                         % (name, want, read)
                         for name, read, want in
                         off_by((target.graphics_mhz, target.memory_mhz), requested)]
            if warnings:
                SUGGESTED[self.gpu_name] = table_line(
                    self.gpu_name, (target.graphics_mhz, target.memory_mhz))
        else:
            print("gpu_clocks: the driver does not list its supported clocks; checking "
                  "against the request", flush=True)
        print("\ngpu_clocks: %s: locking at %d/%d MHz (offered: %d/%d)"
              % (self.gpu_name, requested.graphics_mhz, requested.memory_mhz,
                 target.graphics_mhz, target.memory_mhz), flush=True)
        lock(requested.graphics_mhz, requested.memory_mhz)
        (gr, mem), stable = wait_for_stable_clocks(LOCK_SETTLE_SECS)
        if not stable:
            raise RuntimeError(
                "the GPU clocks did not settle after locking: still changing after %d s "
                "idle, last %d/%d MHz. The GPU is busy with something else."
                % (2 * LOCK_SETTLE_SECS, gr, mem))
        if gr > target.graphics_mhz * (1 + CLOCK_TOLERANCE_FRAC):
            raise RuntimeError(
                "the GPU graphics clock reads %d MHz after locking at %d: above the "
                "lock, so the lock did not take." % (gr, target.graphics_mhz))
        asked = {"graphics": requested.graphics_mhz, "memory": requested.memory_mhz}
        warnings += ["%s clock %d MHz, %d%% %s the %d MHz %s"
                     % (name, read, round(100.0 * abs(read - want) / want),
                        "below" if read < want else "above", want,
                        "it offers for the %d requested" % asked[name]
                        if supported else "requested")
                     for name, read, want in off_by((gr, mem), target)]
        self.locked = LockedClocks(gr, mem, tuple(warnings))
        print("gpu_clocks: locked, reads %d/%d MHz" % (gr, mem), flush=True)
        for warning in warnings:
            print("gpu_clocks: WARNING: %s; measuring at %d/%d MHz" % (warning, gr, mem),
                  flush=True)
        if self.gpu_name in SUGGESTED:
            print("gpu_clocks: the clocks this card offers, for tests/hw_config.py:\n%s"
                  % SUGGESTED[self.gpu_name], flush=True)
        return self.locked

    def wait_for_locked_clocks(self):
        """Call right before a measurement. Waits, up to MEASURE_SETTLE_SECS, for the
        GPU to read its locked clocks again, and raises if it does not; after one wait
        has run out, reads once. Marks where the measurement's telemetry starts.
        Unlocked (run_unlocked), nothing is waited for."""
        if self.unlocked:
            self._measure_start = time.monotonic()
            return
        secs = 0 if self.settle_failed else MEASURE_SETTLE_SECS
        clocks, off = wait_for_clocks(self.locked, secs)
        if off:
            waited = not self.settle_failed
            self.settle_failed = True
            raise RuntimeError(
                "the GPU clocks are not at their lock before the measurement: %s%s. "
                "A locked GPU reads its lock when idle; one that does not is still "
                "busy, or lost the lock."
                % (describe(off),
                   ", still after %d s" % MEASURE_SETTLE_SECS if waited else
                   " (not waited for: an earlier measurement's wait already ran out)"))
        self._measure_start = time.monotonic()

    def get_telemetry(self):
        """Call right after the measurement. The statistics of the telemetry since
        the last wait_for_locked_clocks(), as two blocks side by side: "all", over
        every sample, and "under_load", over the samples taken under load
        (under_load_samples), which also says the utilization it took to count.

        Telemetry runs for the whole session, so nothing is left open by a
        measurement that raises between the two calls: its samples are simply never
        asked for."""
        if self._measure_start is None:
            raise RuntimeError("get_telemetry() without wait_for_locked_clocks() first")
        samples = self.telemetry.since(self._measure_start)
        if not samples:
            raise RuntimeError(
                "no GPU telemetry while the measurement ran (nvidia-smi %s%s)"
                % ("running" if self.telemetry.running() else "exited",
                   ": " + " | ".join(self.telemetry.unparsed)
                   if self.telemetry.unparsed else ""))
        loaded, threshold = under_load_samples(samples)
        return {"all": stats(samples),
                "under_load": dict(stats(loaded), min_utilization_pct=threshold)}

    def close(self):
        self.telemetry.close()


def under_load_samples(samples):
    """The samples taken under load (UNDER_LOAD_UTILIZATION_PCT), and the utilization
    threshold that picked them. All of them, and a threshold of None, when nvidia-smi
    cannot read the utilization: then there is no telling idle from loaded."""
    if any(s.utilization_pct is None for s in samples):
        return list(samples), None
    return ([s for s in samples if s.utilization_pct >= UNDER_LOAD_UTILIZATION_PCT],
            UNDER_LOAD_UTILIZATION_PCT)


# ── pytest ───────────────────────────────────────────────────────────────────

def gpu_hw():
    """The session's BenchClocks, with the GPU locked; clocks reset and telemetry
    stopped at the end."""
    clocks = BenchClocks()
    try:
        clocks.lock()
        yield clocks
    finally:
        unlock()
        clocks.close()


if pytest is not None:
    gpu_hw = pytest.fixture(scope="session")(gpu_hw)


# ── Command line ─────────────────────────────────────────────────────────────

# Bits of clocks_event_reasons.active that say why the clock is below what it may
# run at, as nvidia-smi --help-query-gpu names them.
_EVENT_REASONS = ((0x1, "idle"), (0x2, "application clocks"), (0x4, "power cap"),
                  (0x8, "hardware slowdown"), (0x10, "sync boost"),
                  (0x20, "thermal (software)"), (0x40, "thermal (hardware)"),
                  (0x80, "power brake"), (0x100, "display clocks"))


def event_reasons(text):
    """clocks_event_reasons.active ("0x0000000000000004") as words."""
    try:
        bits = int(text, 16)
    except (TypeError, ValueError):
        return text or "n/a"
    return ", ".join(name for bit, name in _EVENT_REASONS if bits & bit) or "none"


def print_status(heading="now"):
    out = _smi("--query-gpu=name,driver_version,clocks.gr,clocks.mem,clocks.max.graphics,"
               "clocks.max.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,"
               "clocks_event_reasons.active", "--format=csv,noheader,nounits")
    (name, driver, gr, mem, max_gr, max_mem, power, limit, temp, util,
     reasons) = [v.strip() for v in out.splitlines()[0].split(",")]
    table = hw_config.CLOCKS.get(name)
    print("%s, driver %s" % (name, driver))
    print("  %-10s graphics %s / memory %s MHz, %s W, %s C, utilization %s%%, "
          "held back by: %s" % (heading, gr, mem, power, temp, util, event_reasons(reasons)))
    print("  %-10s graphics %s / memory %s MHz, power limit %s W"
          % ("maximum", max_gr, max_mem, limit))
    print("  %-10s %s" % ("table", "graphics %d / memory %d MHz (tests/hw_config.py)" % table
                          if table else "none: %s is not in tests/hw_config.py" % name))


def _show_load(clocks, secs):
    """An 8192 x 8192 fp32 matrix multiply for `secs`, and what the clocks did. The
    heaviest load there is: it takes a card to its power limit, which a lock does not
    lift."""
    import cupy as cp
    a = cp.random.rand(8192, 8192, dtype=cp.float32)
    clocks.wait_for_locked_clocks()
    end = time.monotonic() + secs
    while time.monotonic() < end:
        a @ a
        cp.cuda.runtime.deviceSynchronize()
    t = clocks.get_telemetry()["under_load"]
    if not t["samples"]:
        print("  under load: no samples with the GPU at work")
        return
    print("  under load, %g s of matrix multiply (%d samples):" % (secs, t["samples"]))
    for field, unit in (("graphics_mhz", "MHz"), ("memory_mhz", "MHz"),
                        ("power_w", "W"), ("temperature_c", "C")):
        st = t.get(field)
        if st:
            print("    %-14s min %7.1f  median %7.1f  max %7.1f %s"
                  % (field.split("_")[0], st["min"], st["median"], st["max"], unit))
    below = (clocks.locked.graphics_mhz - t["graphics_mhz"]["median"]) / clocks.locked.graphics_mhz
    if below > CLOCK_TOLERANCE_FRAC:
        print("  under load the graphics clock is %.0f%% below the lock: the card is at "
              "its power or temperature limit, and the lock is a ceiling it cannot reach"
              % (below * 100))


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog="gpu_clocks.py",
        description="Lock GPU 0 at the clocks the benchmarks use (tests/hw_config.py), "
                    "release it, or show what it is doing.")
    sub = p.add_subparsers(dest="command", required=True)
    lk = sub.add_parser("lock", help="lock at the table's clocks and wait until the "
                                     "idle GPU reads them; they stay locked until unlock")
    lk.add_argument("--load", type=float, metavar="SECS", default=0,
                    help="then run a matrix multiply for SECS and show the clocks under "
                         "load (needs cupy)")
    sub.add_parser("unlock", help="reset the graphics and memory clocks")
    sub.add_parser("status", help="clocks, power, temperature and what holds them back")
    args = p.parse_args(argv)

    try:
        if args.command == "status":
            print_status()
        elif args.command == "unlock":
            unlock()
            print_status("unlocked")
        else:
            clocks = BenchClocks()
            try:
                clocks.lock()
                print_status("locked")
                if args.load:
                    _show_load(clocks, args.load)
            finally:
                clocks.close()
            print("Clocks stay locked. Release them with: %s unlock" % __file__)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit("gpu_clocks: %s" % e)


if __name__ == "__main__":
    main()
