# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running commands, and the trace bash gave for free.

Every script under ci/ ran with `set -x` and PS4='+[$(date +%T)] ', and that was not
decoration. What these scripts do IS the commands they run -- docker invocations,
ssh, tar, pytest -- and when one goes red in a CI log the first question is always
which command, with which arguments, at what time. Python prints nothing on its own,
so the trace has to be built: every command started through this module announces
itself in the same format the shell used, and a log from before the rewrite lines up
with a log from after it.

`set -e` becomes check=True, which is the default here for the same reason it was
the default there: a CI script that keeps going after a failed step reports the wrong
verdict. A call that is genuinely allowed to fail says so with check=False, which is
the Python for `|| true` and just as visible.
"""
import contextlib
import os
import shlex
import subprocess
import sys
import time

_TRACE = True


class Fatal(Exception):
    """An error whose message is the whole of what the user needs to read.

    Raise it for a condition the script itself detected -- a missing artifact, an
    unparseable config, an argument that names nothing. main() prints the message
    and exits 1, with no traceback: a traceback through this module's frames tells
    the reader nothing they did not already know from the message.
    """


def set_trace(on):
    """Turn the command trace on or off, as `set -x` / `set +x` did.

    Off for --list, which is asked a question and should answer with the answer and
    nothing else, and off around a script's final verdict, where the trace would put
    a line of shell between the reader and the word PASS.
    """
    global _TRACE
    _TRACE = bool(on)


def tracing():
    return _TRACE


@contextlib.contextmanager
def untraced():
    """Silence the trace for a block, then put it back the way it was.

    For loops that run the same command every few seconds -- polling a job's state,
    draining its log -- where tracing every call buries the output the loop exists
    to show.
    """
    was = _TRACE
    set_trace(False)
    try:
        yield
    finally:
        set_trace(was)


def fmt(cmd):
    """A command as a line you could paste back into a shell."""
    if isinstance(cmd, str):
        return cmd
    return " ".join(shlex.quote(str(c)) for c in cmd)


def trace(cmd):
    """Announce a command the way PS4='+[$(date +%T)] ' did.

    On stderr, where the shell put it: a caller that captures stdout is capturing a
    value, and the trace is not part of the value.

    A traced line is what ran, and the way to keep it that way is to trace the same
    variable the subprocess call takes -- which is what _announce below does for
    everything in this module. The one licensed exception writes the program's name
    between tildes, `~ssh~ host 'cmd'`: what ran, with a constant part left out. The
    tildes are there so nobody pastes the line into a shell expecting it to work. The
    CI's SSH session code is the only place that spells one.
    """
    if _TRACE:
        sys.stderr.write("+[%s] %s\n" % (time.strftime("%H:%M:%S"), fmt(cmd)))
        sys.stderr.flush()


def _announce(cmd, kw):
    """Trace a command and settle the one keyword its form implies. Returns the kw.

    It starts nothing -- the caller does that on the next line, with the same `cmd`
    this one printed, which is the whole arrangement: a trace built beside the call it
    describes cannot end up describing something else.

    The keyword is shell=True for a string and nothing for a list -- the distinction
    the shell made between a command line and an expanded array, and the reason a list
    is preferred everywhere a list will do.
    """
    trace(cmd)
    if isinstance(cmd, str):
        kw.setdefault("shell", True)
    return kw


def run(cmd, *, check=True, **kw):
    """Run a command with its output going wherever ours goes. Returns CompletedProcess.

    check=True raises CalledProcessError, which main() reports as the failing command
    and its exit code.
    """
    kw = _announce(cmd, kw)
    return subprocess.run(cmd, check=check, **kw)


def rc(cmd, **kw):
    """Run a command and return its exit code, never raising. `|| true`, with the
    code kept."""
    return run(cmd, check=False, **kw).returncode


def popen(cmd, **kw):
    """Start a command and hand back the Popen, without waiting for it.

    What run() cannot do, and what a pipe needs: one end of a pipe has to keep
    running while the other end is fed. Before this existed, building a pipe meant
    calling subprocess.Popen directly, outside the trace -- and a command that starts
    outside the trace gets described by hand, in a line that agrees with the code only
    until one of the two is edited. The CI's upload over SSH is one such pipe.
    """
    kw = _announce(cmd, kw)
    return subprocess.Popen(cmd, **kw)


def ok(cmd, **kw):
    """Whether a command succeeded, with its output discarded. The `if cmd >/dev/null
    2>&1` of the shell scripts -- used for probes (does this image exist, is this
    tool installed), where the output is noise and only the answer is wanted."""
    kw.setdefault("stdout", subprocess.DEVNULL)
    kw.setdefault("stderr", subprocess.DEVNULL)
    return rc(cmd, **kw) == 0


def capture(cmd, *, check=True, strip=True, stderr=None, **kw):
    """Run a command and return its stdout as text -- `$(...)`.

    stderr is left alone by default, so a command's diagnostics reach the log rather
    than disappearing into a variable the caller only reads for its value.
    """
    kw = _announce(cmd, kw)
    cp = subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=stderr,
                        text=True, **kw)
    out = cp.stdout or ""
    return out.strip() if strip else out


def tee(cmd, logfile, **kw):
    """Run a command, streaming its merged output to our stdout AND to a file.

    `cmd 2>&1 | tee log`, and live: the reader of a CI log wants the test that is
    running now, not the whole run at the end. Returns the exit code -- ours, not
    tee's, which is the bug `set -o pipefail` existed to paper over in the shell
    version.
    """
    kw = _announce(cmd, kw)
    os.makedirs(os.path.dirname(os.path.abspath(logfile)), exist_ok=True)
    with open(logfile, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, **kw)
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        p.stdout.close()
        return p.wait()


# pytest's "no tests collected". Under a -k filter that is the normal answer from
# every test file that does not contain the test being chased.
PYTEST_NO_TESTS = 5


def env_with(**overrides):
    """os.environ plus overrides, with a None meaning "leave it out".

    The shell wrote `VAR=value cmd`, which is a copy of the environment with one
    entry changed and does not touch the caller's own. Same here: this returns a
    dict to hand to run(env=...), and os.environ is untouched.
    """
    e = dict(os.environ)
    for k, v in overrides.items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = str(v)
    return e


def path_prepend(env, var, *entries):
    """Put entries at the front of a colon-path in an env dict, keeping what was
    there. The `PYTHONPATH="new${PYTHONPATH:+:$PYTHONPATH}"` idiom, which is fiddly
    in shell precisely because of the empty case."""
    parts = [str(e) for e in entries if e]
    old = env.get(var)
    if old:
        parts.append(old)
    env[var] = ":".join(parts)
    return env


def main(fn):
    """Run a script's main function and turn its failures into exit codes.

    Three kinds, all of which the shell scripts handled with `set -e` and an
    explicit exit:

      Fatal              -- a condition this script diagnosed. Its message is the
                            whole report; a traceback would add frames from this
                            module and nothing else.
      CalledProcessError -- a command failed. Report which one and with what code,
                            and exit with that same code, so a caller further up
                            sees the failure the command actually produced.
      KeyboardInterrupt  -- 130, as a shell does.
    """
    try:
        code = fn()
    except Fatal as e:
        sys.stderr.write("ERROR: %s\n" % e)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.stderr.write("ERROR: command failed (exit %d): %s\n"
                         % (e.returncode, fmt(e.cmd)))
        sys.exit(e.returncode or 1)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        sys.exit(130)
    sys.exit(code or 0)
