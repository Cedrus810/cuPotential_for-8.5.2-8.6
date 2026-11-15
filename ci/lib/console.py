# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coloured output, in the one place all of ci/ reads it from.

The codes are the ones the shell scripts used, kept so that a log from before the
rewrite and a log from after it look the same: 33 for a stage banner, 34 for a job
banner, 90 for a skip, '1;32' for a pass and '1;31' for a failure.
"""
import sys

# Names for the codes that had a meaning, so a call site says what it means rather
# than what it prints. The raw code still works -- color("1;35", ...) is fine.
STAGE = "33"
JOB = "34"
SKIP = "90"
PASS = "1;32"
FAIL = "1;31"


def color(code, text, file=None):
    """Print text in a colour. Unconditional: these logs are read in a terminal or
    in the CI's log view, and both render ANSI."""
    print("\033[%sm%s\033[0m" % (code, text), file=file or sys.stdout, flush=True)


def section(title, file=None):
    """The heavy banner setup-node used for its dumps of nvidia-smi and friends."""
    color("1;36", "═" * 22 + " " + title, file=file)


def banner(code, title):
    """A stage's or job's own three-line box."""
    line = "═" * 62
    color(code, line)
    color(code, "  " + title)
    color(code, line)
