# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""How long each part of a run took: the stages, and what they report about
themselves. STDLIB ONLY, like the rest of ci/lib.

A run is everything one ./run_test (or ci/scripts/stages.py) does, including the
stages it hands to the dev container: those run in another process, inside
./run_docker. So the timings are kept in files, in TIMING_DIR under logs/, which is
in the tree and so is the same directory on both sides. CUPO_RUN_ID says which run
a process belongs to. The process that starts a run sets it and empties TIMING_DIR,
and ./run_docker passes it into the container, whose process then adds to the
files rather than starting over.

The process that started the run is the one that says how it went: the stages with
their times, the path of the full report, and PASS or FAIL, in that order and last.

Files in TIMING_DIR:
  stages.jsonl      one line per stage that ran: name, seconds, exit code
  report.txt        the full report, written at the end of the run (write_report)

Other parts of a run add their own files there and a section of the report: see
SECTIONS at the bottom.
"""
import json
import os
import pathlib
import time

from . import proc

repo_dir = pathlib.Path(__file__).resolve().parents[2]

TIMING_DIR = repo_dir / "logs" / "timing"
STAGES_FILE = TIMING_DIR / "stages.jsonl"
REPORT_FILE = TIMING_DIR / "report.txt"

RUN_ID_VAR = "CUPO_RUN_ID"

# For the programs a stage runs -- a compiler launcher, pytest -- to find TIMING_DIR
# and to say which stage their timings belong to. Set in every process of the run,
# because the directory's path is not the same inside ./run_docker (CI mounts the
# tree at /tmp/home).
DIR_VAR = "CUPO_TIMING_DIR"
STAGE_VAR = "CUPO_STAGE"

# Whether this process started the run. Set by begin().
_started_here = False


def begin():
    """Join the run this process belongs to, or start one.

    A process with no CUPO_RUN_ID in its environment is the start of a run: it
    empties TIMING_DIR, so the report says what this run did and nothing an earlier
    one did, and puts its own id in the environment for the processes it starts.
    """
    global _started_here
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    os.environ[DIR_VAR] = str(TIMING_DIR)
    if os.environ.get(RUN_ID_VAR):
        return
    _started_here = True
    os.environ[RUN_ID_VAR] = "%d-%d" % (os.getpid(), int(time.time()))
    for f in TIMING_DIR.iterdir():
        if f.is_file():
            f.unlink()


def started_here():
    return _started_here


def stage_started(name):
    """Say, in the environment, which stage the programs started from now on run in."""
    os.environ[STAGE_VAR] = name


def stage_ended():
    os.environ.pop(STAGE_VAR, None)


def record_stage(name, secs, code):
    """One stage that ran, how long it took, and its exit code (None for a stage
    that returned nothing, which is success)."""
    with STAGES_FILE.open("a") as f:
        f.write(json.dumps({"stage": name, "secs": round(secs, 1),
                            "code": code}) + "\n")


def stages():
    """The stages of this run, in the order they ran: [(name, secs, code)]."""
    if not STAGES_FILE.is_file():
        return []
    out = []
    for line in STAGES_FILE.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.append((r["stage"], r["secs"], r["code"]))
    return out


def elapsed(secs):
    """A duration for a person: "45s", "7m30s", "1h02m"."""
    secs = int(round(secs))
    if secs < 60:
        return "%ds" % secs
    if secs < 3600:
        return "%dm%02ds" % divmod(secs, 60)
    return "%dh%02dm" % divmod(secs // 60, 60)


def ran_line():
    """The stages with their times, for the line above the verdict:
    "cupo_build (7m30s), cupo_unit (4m02s)"."""
    return ", ".join("%s (%s)" % (name, elapsed(secs)) for name, secs, _ in stages())


# ── Log sections ──────────────────────────────────────────────────────────────
#
# In the CI's job log, the lines between a section_start and a section_end marker
# fold into one, and the log shows how long the section took beside its header. The
# markers are escape sequences a terminal does not show, but they are printed in CI
# only: the CI's log is the one reader that makes use of them.

def _in_ci():
    return bool(os.environ.get("CI"))


def section_start(name, header, collapsed=False):
    """header, as the first line of a log section named after the stage; header
    alone outside CI."""
    if not _in_ci():
        return header
    return "\033[0Ksection_start:%d:stage_%s[collapsed=%s]\r\033[0K%s" % (
        int(time.time()), name, "true" if collapsed else "false", header)


def section_end(name):
    """The line that closes the section section_start opened, or None outside CI."""
    if not _in_ci():
        return None
    return "\033[0Ksection_end:%d:stage_%s\r\033[0K" % (int(time.time()), name)


def print_report(path):
    """In CI, print the report into the job log, folded: the log is what a reader
    of a CI job has at hand, and the artifact is a download away. Outside CI the
    path is enough -- the file is right there."""
    if not _in_ci() or not path:
        return
    print(section_start("timing_report", "=== Timings (%s) ===" % path.name,
                        collapsed=True))
    print(path.read_text(), end="")
    print(section_end("timing_report"), flush=True)


# ── The report ────────────────────────────────────────────────────────────────

def _stage_section():
    rows = stages()
    if not rows:
        return []
    total = sum(secs for _, secs, _ in rows) or 1
    width = max(len(name) for name, _, _ in rows)
    lines = ["Stages, in the order they ran (wall time):", ""]
    for name, secs, code in rows:
        result = "ok" if code in (None, 0, proc.PYTEST_NO_TESTS) else "exit %s" % code
        lines.append("  %-*s  %8s  %3d%%  %s"
                     % (width, name, elapsed(secs), round(100 * secs / total), result))
    lines.append("  %-*s  %8s" % (width, "total", elapsed(total)))
    return lines


# Each a function returning the lines of one section of the report, or [] when it
# has nothing to say for this run. Other modules add theirs to this list.
SECTIONS = [_stage_section]


def write_report():
    """Write report.txt from everything the run left in TIMING_DIR, and return its
    path, or None when there was nothing to report."""
    parts = []
    for section in SECTIONS:
        lines = section()
        if lines:
            parts.append("\n".join(lines))
    if not parts:
        return None
    REPORT_FILE.write_text("\n\n".join(parts) + "\n")
    return REPORT_FILE
