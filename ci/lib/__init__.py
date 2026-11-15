# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared code for the scripts under ci/.

Every executable here is a standalone script rather than an installed package, so
each one puts the ci/ directory on sys.path and imports `lib`:

    repo_dir = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_dir / "ci"))
    from lib import proc

Stdlib only, deliberately. These scripts run on a CI shell runner, on the machine
that hands work to the GPU nodes, on a bare GPU node and inside three different
container images; the one thing all of them have is a python3. Adding a dependency means
adding an install step to each of those places, and the install step is the part
that breaks.

Importing this package puts the process in line-buffered mode -- see below.
"""
import os
import sys


def line_buffered():
    """Stop Python from holding CI output in a buffer, ours and our children's.

    Python picks its buffering from what stdout is attached to, and under CI stdout is
    a pipe: 8 KB accumulate before anything is written. Everything these scripts run
    writes to that same pipe directly and is not buffered by us, so a run's narration
    arrives in a block long after the commands it introduces -- a banner printed
    before a twenty-minute build shows up after it. Two lines fix it for the whole
    tree:

      * our own streams go line-buffered, which also empties the buffer before each
        child starts, since every print ends in a newline;
      * PYTHONUNBUFFERED is set in our environment, so every Python we start -- pytest,
        stages.py in the container, a script calling another script -- inherits it
        without being passed anything.

    A CI script that wants its output held back does not exist, so this runs on import
    rather than waiting to be called. That makes it an import side effect, which is
    worth it here: it is the whole reason there is no line to forget in a new script.
    A script that is piped to another machine as source, with no package around it,
    is the one kind this cannot reach, and prints with flush=True instead.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"


line_buffered()
