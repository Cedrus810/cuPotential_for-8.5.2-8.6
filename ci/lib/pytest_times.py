# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""How long each test took: the report's section on the stages that run pytest.

While a stage runs, PYTEST_ADDOPTS asks every pytest it starts for a JUnit XML file
in the timing directory (ci/lib/timing.py), named after the stage. That covers the
stages that run pytest through stages.run_pytest and any whose own script runs it.
A stage that ran pytest twice keeps the second file only.

A test's time in the file is its setup, call and teardown together, so a test that
builds a module-scoped fixture carries that fixture's time.
"""
import collections
import os
import xml.etree.ElementTree as ET

from . import timing

# How many of the slowest tests, and of the slowest test files, the report lists for
# each stage.
SLOWEST = 15

_ADDOPTS = "PYTEST_ADDOPTS"


def junit_path(stage):
    return timing.TIMING_DIR / ("pytest-%s.xml" % stage)


def stage_started(stage):
    """Ask the pytests this stage runs for their per-test times. Returns what
    PYTEST_ADDOPTS was before, for stage_ended to put back."""
    before = os.environ.get(_ADDOPTS)
    os.environ[_ADDOPTS] = ((before + " ") if before else "") + \
        "--junitxml=%s" % junit_path(stage)
    return before


def stage_ended(before):
    if before is None:
        os.environ.pop(_ADDOPTS, None)
    else:
        os.environ[_ADDOPTS] = before


def tests(path):
    """[(test id, seconds, outcome)] from a JUnit XML file; outcome is passed,
    failed, error or skipped. [] for a file that cannot be read."""
    try:
        root = ET.parse(str(path)).getroot()
    except (OSError, ET.ParseError):
        return []
    out = []
    for case in root.iter("testcase"):
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed"}.get(child.tag, child.tag)
        name = "%s::%s" % (case.get("classname", ""), case.get("name", ""))
        out.append((name, float(case.get("time") or 0), outcome))
    return out


def _secs(secs):
    """A test's time: hundredths under ten seconds, where most tests are, and
    timing.elapsed above."""
    return "%.2fs" % secs if secs < 10 else timing.elapsed(secs)


def _file_of(test_id):
    """The test file a JUnit classname belongs to: "tests.test_kspace.TestP1" ->
    "tests/test_kspace.py". pytest writes the module's dotted path, then the
    class, if any."""
    dotted = test_id.split("::")[0]
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith("test_") or parts[i - 1].endswith("_test"):
            return "/".join(parts[:i]) + ".py"
    return dotted


def _section():
    lines = []
    for name, _, _ in timing.stages():
        rows = tests(junit_path(name))
        if not rows:
            continue
        total = sum(secs for _, secs, _ in rows)
        counts = collections.Counter(outcome for _, _, outcome in rows)
        lines += ["", "%s: %d tests, %s in tests (%s)" % (
            name, len(rows), _secs(total),
            ", ".join("%d %s" % (n, o) for o, n in sorted(counts.items())))]
        lines.append("  slowest tests:")
        for test_id, secs, outcome in sorted(rows, key=lambda r: -r[1])[:SLOWEST]:
            lines.append("    %8s  %s%s" % (_secs(secs), test_id,
                                           "" if outcome == "passed" else
                                           "  (%s)" % outcome))
        by_file = collections.Counter()
        for test_id, secs, _ in rows:
            by_file[_file_of(test_id)] += secs
        lines.append("  slowest test files:")
        for path, secs in by_file.most_common(SLOWEST):
            lines.append("    %8s  %3d%%  %s" % (_secs(secs),
                                                 round(100 * secs / (total or 1)),
                                                 path))
    return ["Tests, per stage that ran pytest (setup + call + teardown):"] + lines \
        if lines else []


timing.SECTIONS.append(_section)
