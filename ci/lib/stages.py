# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a stage is, what STAGES means, and how a dependency crosses a script.

A stage is a decorated function, and the function's own name IS the stage name:

    from lib.stages import stage

    @stage
    def cupo_build():
        proc.run([...])

    @stage(depends=cupo_build)          # one, or a list
    def cupo_unit():
        ...

That is the name in STAGES, in the CI's configuration and on the ./run_test command line,
so `git grep cupo_unit` finds the stage, the job that asks for it and the README
that describes it -- the three things that go out of step with each other.

There is one registry, here, and one file writes stages:

    ci/scripts/stages.py    the builds, the tests, the release tarballs and the
                            packaging tests

./run_test is that file under the name people type -- a symlink, so the two names are
one program. CI runs it under its own name, because a job whose whole work is to
build release tarballs reads badly as a job running run_test.

One file rather than several because nearly every edge crosses what the boundary used
to be: a packaging test needs the release tarball, and the tarball needs the build. In
one file each of those is a function reference, checked where it is written.

Which of those environments a stage needs is the stage's own business, and is written
on the stage:

    @stage(where="container")   the default: the dev image, where the toolchain is
    @stage(where="host")        the docker the dev image has none of

The host is the larger of the two: it can start the container, and nothing inside the
container can get back out of it. So a run starts on the host, and a group of stages
that wants the container is re-run through ./run_docker -- `./run_test cupo_unit`
enters the container by itself, and `./run_docker ./run_test cupo_unit` is the same
run with the entering done by hand. The other direction is not a thing that can be
arranged, so asking for a host stage from inside the container is an error that says
which command to type instead.

What a stage runs after is what the stage says it runs after. The run order is a
topological sort of those edges, so where in the file a stage is written settles
nothing about when it runs. Two stages with no edge between them have no opinion
about each other, and the order they were written in breaks that tie: it is the only
thing left to break it with, and it makes a listing read the way the file reads.

There are two kinds of edge, and the difference is whether the other stage has to
run at all:

    @stage(depends=cupo_build)      needs it. `./run_test cupo_unit` runs cupo_build
                                    as well, because cupo_unit cannot work without
                                    it.
    @stage(after=cupo_tarball)      only follows it. collect_artifacts copies
                                    whatever tarballs the run produced and asks for
                                    none of them, but it cannot run before the stage
                                    that makes one. `./run_test collect_artifacts`
                                    builds no tarball.

Either kind is written as the other stage's function, never as its name, and is
checked as it is written: the other stage has to be registered already. A typo is an
error before anything runs, a stage can only wait on one written above it, and a cycle
is not a thing that can be spelled.

Choosing what runs. The two ways of naming stages differ exactly on dependencies:

  ./run_test cupo_unit   runs cupo_unit and everything it depends on. That is what
                         a person means: run this, and whatever it needs.
  STAGES="cupo_unit"     runs cupo_unit and nothing else. That is what CI means: the
                         dependency ran in another job on another machine and arrived
                         here as an artifact, and following the edge would rebuild it.

And naming nothing at all:

  ./run_test             runs the stages marked autorun, and everything those need.
                         That is the short list a person means by "test this tree":
                         the unit tests, the benchmarks, the sample. A stage that is
                         not marked runs when it is named, and only then -- which is
                         what keeps a bare ./run_test from building release tarballs
                         and starting docker containers.

Each STAGES entry is matched as a shell pattern, so a family of stages can be asked
for by what they are named after rather than one by one -- STAGES="*customer_env*"
is every customer-environment stage, and a route added there needs no edit at any
call site. A bare stage name is a pattern that matches only itself, so spelling the
list out still works.

Patterns at call sites are meant to be the family's own word, spelled the way the
stages spell it, so that `git grep` finds every one of them from the pattern.
"""
import argparse
import fnmatch
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from . import compile_timing, console, proc, pytest_times, timing

# Which shard of a split CI test job this run is (ci_shard). A tree without the
# module runs unsplit, as when CI_NODE_TOTAL is unset.
try:
    from . import shards
except ImportError:
    shards = None

repo_dir = pathlib.Path(__file__).resolve().parents[2]

# The two environments a stage can ask for. Strings rather than an enum because they
# are written at every call site -- @stage(where="host") -- and read best there.
HOST = "host"
CONTAINER = "container"


def here():
    """Which of the two this process is in.

    The dev container is the one ./run_docker starts, and ./run_docker says so: it
    sets CUPO_DEV_CONTAINER in it (as does plugins/openmm/samples/run_docker, the
    run_docker of the source tarball). Anything else is the host, including other
    containers: a CI job's own image, a dev box, the shell this was typed into.

    A host with no docker installed is still the host -- it simply cannot run the
    stages that need docker, and those stages say so themselves, when they are asked
    for. The stages that want the dev container run in place there, as in it
    (run()).
    """
    return CONTAINER if os.environ.get("CUPO_DEV_CONTAINER") else HOST


# ── Patterns ──────────────────────────────────────────────────────────────────

def wanted(patterns, names):
    """The names some patterns asked for, in the order the names were given.

    `patterns` is the split value of STAGES; an empty or missing value means all of
    them, which is what an unset STAGES has always meant. Matching is fnmatch, which
    is `case` with the same four constructs -- *, ?, [seq], [!seq].

    A name matched by nothing is simply not returned. That is not the same as an
    error, and this function is deliberately not the place that decides which it is --
    a caller filtering a list it already has is asking a different question from a
    caller checking what a person typed. check_patterns() is where a pattern that
    matches no stage at all is an error.
    """
    if isinstance(patterns, str):
        patterns = patterns.split()
    patterns = list(patterns or [])
    if not patterns:
        return list(names)
    return [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in patterns)]


def from_env(env_var="STAGES"):
    """The patterns in an environment variable, split on whitespace as the shell
    did. Empty and unset are the same thing here, as they were there."""
    return os.environ.get(env_var, "").split()


def _widen(pattern, names):
    """The stages a command-line stage name asks for.

    A shell pattern, as in STAGES -- and, when a plain word matches no stage exactly,
    the same word read as *word*. `./run_test customer_env` is a person naming a
    family by the word its stages share, and that should find them rather than
    nothing.
    """
    hits = wanted([pattern], names)
    if not hits and not any(c in pattern for c in "*?["):
        hits = wanted(["*%s*" % pattern], names)
    return hits


# ── The registry ──────────────────────────────────────────────────────────────

class _Stage:
    """One stage: what to run, where it runs, what it needs, and what surrounds it."""

    def __init__(self, fn, where, autorun, depends, after, pre, post, skip_on, needs_files):
        self.fn = fn
        self.where = where
        self.autorun = autorun
        self.depends = depends
        self.after = after
        self.pre = pre
        self.post = post
        self.skip_on = skip_on
        self.needs_files = needs_files

    def unavailable(self):
        """Why this tree cannot run the stage, or "" when it can: the first of its
        needs_files that is not here."""
        for rel in self.needs_files:
            if not (repo_dir / rel).exists():
                return "not available in this tree (no %s)" % rel
        return ""


class _Script:
    """The stages one file writes, in the order it writes them.

    One file writes stages, so there is one of these. It is still a thing rather than
    a bare dictionary because a run that cannot finish here has to say what to start
    instead, and that is this file's own path and its own arguments -- see
    _run_script.

    `argv` is what to put after the path when the stages are re-run in the other
    environment: the pytest arguments, for one, since a filter typed on the host is
    meant for the pytest that ends up running in the container. It is a callable
    rather than a list because the caller does not know the answer until it has
    parsed its own command line.
    """

    def __init__(self, path):
        self.path = path                # absolute, as the file was loaded
        self.stages = {}                # name -> _Stage, in written order
        self.argv = None

    @property
    def label(self):
        """What a listing calls this script: its path from the repo root."""
        try:
            return str(pathlib.Path(self.path).relative_to(repo_dir))
        except ValueError:
            return self.path

    def extra_argv(self):
        return [str(a) for a in (self.argv() if callable(self.argv) else self.argv or [])]


_scripts = {}                           # path -> _Script
_order = []                             # _Script, in registration order


def _script_at(path):
    """The _Script for a file, made on the file's first stage."""
    path = os.path.realpath(path)
    script = _scripts.get(path)
    if script is None:
        script = _scripts[path] = _Script(path)
        _order.append(script)
    return script


def _own():
    """The script being run.

    The file Python was started on, which is the file that writes the stages -- under
    whichever of its two names it was started. Looked up rather than made, so that a
    script asking for stages it never wrote says so instead of quietly having none.
    """
    path = getattr(sys.modules["__main__"], "__file__", "") or ""
    script = _scripts.get(os.path.realpath(path))
    if script is None:
        raise proc.Fatal("%s writes no stages" % (path or "<no __main__ file>"))
    return script


def stage(fn=None, *, where=CONTAINER, autorun=False, depends=(), after=(),
          pre=None, post=None, skip_on=None, needs_files=()):
    """Register a stage. Works bare (@stage) and with arguments (@stage(...)).

    Bare, Python hands the decorator the function; with arguments it hands it the
    arguments and expects a decorator back. Both spellings read the same at the call
    site, and a stage with no dependencies should not have to write an empty argument
    list to say so.

    depends and after both mean "not before that stage", and differ on whether that
    stage has to run: what a stage depends on is fetched along when the stage is asked
    for by name, and what it is after is not. Each takes one stage or a list of them,
    written as a function or as a name.

    pre and post are callables run around the stage body, and post runs even when the
    body fails -- the compiling stages start sccache in one and report its statistics
    in the other, and a build that fails still reports them. They are the stage's own
    business, written on the stage, so that running one stage by name does the same
    around it as running all of them.

    skip_on is a callable that returns why this stage should not run, or "" for run
    it, or a list of such callables, of which the first to give a reason decides.
    Also the stage's own business: `./run_test -k something` cannot filter a
    sample program that takes no filter, so that stage says so itself rather than
    every caller remembering to.

    needs_files is the files and directories, relative to the repository root, that
    the stage cannot run without. The plugin's source tree on GitHub is a part of this
    repository: it has the plugin and its tests, and not cupo's sources, the release
    scripts or the CI jobs. A stage whose files are not there is skipped, saying "not
    available in this tree (no <file>)", rather than failing on a command that is not
    there. Skipped like a skip_on, so a stage that depends on it still runs: the
    plugin's build depends on cupo's, and builds against the libraries the tree has.

    where is which environment the stage needs, "container" or "host". The default is
    the container, because that is where the toolchain and the GPU tests are; "host"
    is for the stages that drive docker, of which the container has none. A stage that
    asks for an environment the run is not in is taken there, or refused -- see run().

    autorun is whether a bare run includes this stage. Off by default, so a stage has
    to say that it is one of the ones ./run_test means when it is typed with nothing
    after it: the unit tests, the benchmarks, the sample. Everything else -- a release
    tarball, a packaging test, unpacking CI artifacts over the tree -- runs when it is
    named, and only then. The build stages are not marked either, and do not need to
    be: the marked stages depend on them, and a bare run follows dependencies.
    """
    # Bare @stage: Python hands over the function and there is nothing else to read,
    # so the arguments here are their own defaults. Written as the arguments rather
    # than as those defaults spelled out again, so that a caller which passes both --
    # a wrapper like compiling_stage(), handing on a function and the hooks it adds --
    # gets what it asked for instead of silently getting the defaults.
    if fn is not None:
        return _add(fn, where, autorun, depends, after, pre, post, skip_on, needs_files)

    def deco(f):
        return _add(f, where, autorun, depends, after, pre, post, skip_on, needs_files)
    return deco


def _edges(spec, script, name, word):
    """The stage names one edge argument asks for. One stage function, or a list.

    A function rather than a name, because whether it is a stage is knowable here and
    now -- so it is answered here and now, rather than by something that goes looking
    when the edge is followed.
    """
    if callable(spec):
        spec = [spec]
    out = []
    for other in spec or ():
        if not callable(other) or other.__name__ not in script.stages:
            raise ValueError("stage %s %s %r, which is not a stage defined above it"
                             % (name, word, getattr(other, "__name__", other)))
        out.append(other.__name__)
    return out


def _add(fn, where, autorun, depends, after, pre, post, skip_on, needs_files=()):
    script = _script_at(fn.__code__.co_filename)
    name = fn.__name__
    if name in script.stages:
        raise ValueError("duplicate stage name: %s" % name)
    if where not in (HOST, CONTAINER):
        raise ValueError("stage %s runs where=%r, which is neither %r nor %r"
                         % (name, where, HOST, CONTAINER))
    script.stages[name] = _Stage(fn, where, autorun,
                                 _edges(depends, script, name, "depends on"),
                                 _edges(after, script, name, "runs after"),
                                 pre, post, skip_on, tuple(needs_files))
    return fn


def own_argv(args):
    """Say what follows this script's own path when its stages are re-run in the other
    environment.

    The pytest arguments, in practice: a filter typed on the host is meant for the
    pytest that ends up running in the container. A callable, because the caller says
    this before it knows the answer -- see _Script.
    """
    _own().argv = args


# ── Asking ────────────────────────────────────────────────────────────────────

def names():
    """The stages of the script being run, in run order.

    The same list as all_names(), since one file writes the stages. Both names are
    kept because they answer different questions -- "mine" and "every one there is" --
    and the day a second file writes stages they stop being the same list.
    """
    own = _own().stages
    return [n for n in all_names() if n in own]


def all_names():
    """Every stage there is, in run order.

    Run order comes from the edges: a stage runs after everything it depends on and
    everything it says it is after. Two stages with no edge between them are left in
    the order they were registered in, which is all there is to break the tie with.

    Written as the plain repeated sweep for the next stage that can go. There are
    twenty stages; the sweep is the definition read out loud, and it cannot be subtly
    wrong the way a hand-rolled queue can.
    """
    registered = [n for script in _order for n in script.stages]
    known = set(registered)
    waits = {n: (set(_stage_of(n).depends) | set(_stage_of(n).after)) & known
             for n in registered}

    done, order = set(), []
    while len(order) < len(registered):
        nxt = next((n for n in registered if n not in done and waits[n] <= done), None)
        if nxt is None:
            raise proc.Fatal("these stages wait on each other, so none can go first: %s"
                             % " ".join(n for n in registered if n not in done))
        done.add(nxt)
        order.append(nxt)
    return order


def _script_of(name):
    """The script that writes a stage."""
    for script in _order:
        if name in script.stages:
            return script
    raise proc.Fatal("no such stage: %s" % name)


def _stage_of(name):
    return _script_of(name).stages[name]


def selected(patterns=None):
    """The stages of ours a run with no stage names on the command line wants, in run
    order. Dependencies are not followed here; the caller decides whether to.

    With patterns -- STAGES -- it is the stages they match, autorun or not: naming a
    stage is asking for it, and CI names the stages it wants one job at a time. With
    none, it is the autorun stages, which is the bare ./run_test list.
    """
    if patterns is None:
        patterns = from_env()
    if not patterns:
        return [n for n in names() if _stage_of(n).autorun]
    return wanted(patterns, names())


def check_patterns(patterns):
    """Fail unless every pattern names at least one stage, anywhere in the registry.

    A pattern that matches nothing runs nothing, and a run that runs nothing passes:
    that is how a stage name misspelled in the CI's configuration becomes a green job that
    tested nothing at all. The check belongs here rather than in the caller because
    "nobody's stage" is a question about the whole registry, and here is the only
    place that can see the whole registry.
    """
    unmatched = [p for p in patterns if not wanted([p], all_names())]
    if not unmatched:
        return
    raise proc.Fatal(
        "no stage matches: %s\n"
        "\n"
        "STAGES entries are shell patterns, and every one of them has to match at\n"
        "least one stage. A pattern that matches nothing runs nothing, and the run\n"
        "would otherwise pass having tested nothing at all.\n"
        "\n"
        "The stages there are:\n%s"
        % (" ".join(unmatched), "".join("  %s\n" % n for n in all_names())))


def match(pattern):
    """The stages a command-line stage name asks for."""
    return _widen(pattern, all_names())


def with_depends(asked):
    """Those stages plus everything they need, in run order."""
    rank = {n: i for i, n in enumerate(all_names())}
    need = set()

    def visit(name):
        if name in need:
            return
        need.add(name)
        for dep in _stage_of(name).depends:
            visit(dep)

    for name in asked:
        visit(name)
    return sorted(need, key=lambda n: rank[n])


def _split_by_where(script, names_to_run):
    """The stages, in the order given, gathered into runs that want the same
    environment.

    A run is one invocation in one place, which is what the caller wants: a group that
    has to be taken into the container is one ./run_docker, not one per stage.
    """
    out = []
    for name in names_to_run:
        where = script.stages[name].where
        if out and out[-1][0] == where:
            out[-1][1].append(name)
        else:
            out.append((where, [name]))
    return out


def list_stages():
    """Print every stage there is, script by script, with what each one waits on.

    "needs" and "after" are the two kinds of edge, printed apart because they answer
    different questions: needs is what naming this stage also runs, after is only
    where it lands among stages that were going to run anyway.

    A stage this environment cannot run is still listed -- it is a stage, and a listing
    that hid it would leave a reader wondering where a name they had seen went. It is
    listed commented out, in grey, with the reason: the shape of a line the run cannot
    take. Inside the dev container that is every host stage, and the reason is the same
    one every time, so the line says the fix rather than the diagnosis.

    The marker in front of a name says how the stage is reached: a star for the ones a
    bare run does, a hash for the ones this environment cannot do at all, and nothing
    for the ones that wait to be named.
    """
    order = all_names()
    for script in _order:
        console.color(console.STAGE, "\n%s:" % script.label)
        for name in (n for n in order if n in script.stages):
            st = script.stages[name]
            missing = st.unavailable()
            out_of_reach = not missing and st.where == HOST and here() == CONTAINER
            edges = ["needs %s" % ", ".join(st.depends)] if st.depends else []
            edges += ["after %s" % ", ".join(st.after)] if st.after else []
            # Only the host ones say so, and only where the word is news: the container
            # is the default, and in the container the comment below already says it.
            edges += ["on the host"] if st.where == HOST and not out_of_reach else []
            # The marker goes in the indent, not in front of the name, so the edges
            # stay in the one column whatever the marker is.
            mark = "# " if out_of_reach or missing else ("* " if st.autorun else "  ")
            line = ("%s%-32s%s" % (mark, name,
                                   "  (%s)" % "; ".join(edges) if edges else "")).rstrip()
            if missing:
                console.color(console.SKIP, "%s  # %s" % (line, missing))
            elif out_of_reach:
                console.color(console.SKIP,
                              "%s  # must run outside of ./run_docker" % line)
            else:
                print(line)
    print()
    print("  * runs by default; every other stage runs when it is named")
    print()


# ── Running ───────────────────────────────────────────────────────────────────
#
# The verdict is kept here rather than by each script, because here is where the
# stages run, and what a stage returns is an exit code.

_keep_going = False
_rc = 0
_ran_here = False
_dispatched = False


def set_keep_going(on):
    """Run every stage that was asked for, even after one of them fails.

    Off by default: the first failure ends the run. What a reader wants from a red log
    is the failure, and a run that carries on past one spends its remaining minutes --
    a build, a benchmark, a container -- producing output that pushes the failure off
    the screen. Output about nothing, usually: the stages after a failed build all
    test the last build that worked.

    On is the other answer, and there is a real question behind it: a nightly wants
    every stage's verdict in one pass rather than one failure per run. ./run_test
    --keep-going asks for that.
    """
    global _keep_going
    _keep_going = on


def rc():
    """The worst exit code the stages have returned."""
    return _rc


def ran_here():
    """Whether any stage ran in this process, rather than being handed to another.

    False makes this process a dispatcher: it decided what to run and where, and the
    processes it started did the running and printed the summary of their logs. A
    dispatcher printing that summary on top of theirs is the same log twice. The
    verdict is a different matter: see the end of _main.
    """
    return _ran_here


def dispatched():
    """Whether this process handed any stage to another process to run.

    True means some other process ran stages out of this tree, and printed whatever it
    prints about them -- the pytest log summary among it, read out of the same logs
    this process would read. Printing that summary again here is those logs twice, once
    per path spelling: /tmp/home/logs/cupo_unit.log inside the container, and the host's
    own name for the same file.
    """
    return _dispatched


def _record(code):
    """Take one stage's exit code. True to carry on, False to end the run.

    pytest's 5, "no tests collected", is not a failure: a targeted `-k` run collects
    nothing from most test files, and counting that would make every targeted run red.
    """
    global _rc
    if code in (None, 0, proc.PYTEST_NO_TESTS):
        return True
    _rc = code
    return _keep_going


def _cannot_leave(names_to_run):
    """What to say when the stages asked for want the host and this is the container.

    The only direction that cannot be arranged. The message is the command to type,
    because the mistake is nearly always one word too many at the front of the one
    that was typed.
    """
    return ("%s runs on the host, and this is the dev container.\n"
            "  drop the ./run_docker:  ./run_test %s"
            % (" ".join(names_to_run), " ".join(names_to_run)))


def must_run_here(names_to_run):
    """Raise unless every one of those stages of ours can run in this environment.

    For a caller that wants the answer before it starts arranging things: what to
    skip and what to refuse are questions worth asking separately from "am I even in
    the right place".
    """
    script = _own()
    for where, chunk in _split_by_where(script, names_to_run):
        if where == HOST and here() == CONTAINER and not all(
                script.stages[n].unavailable() for n in chunk):
            raise proc.Fatal(_cannot_leave(chunk))


def _run_script(script, names_to_run):
    """Re-run this same script inside the dev container, for the stages that want it.
    Returns that process's exit code.

    The one reason left to start a process: this one is on the host and those stages
    need the container. The other direction is not a thing that can be arranged, which
    is why a run starts on the host -- see _cannot_leave.

    The names go through STAGES, which is exactly the "these and no dependencies"
    request that variable exists for: the dependencies are in the same list already,
    resolved here, and letting the other process follow them again would run them
    twice.

    The path is written from the repo root rather than absolutely, because under
    ./run_docker the tree is not always mounted where it lives -- CI mounts it at
    /tmp/home -- and the path has to mean the same thing on both sides of that.
    """
    global _dispatched
    _dispatched = True

    console.color(console.JOB, "\n=== entering the dev container for: %s ===\n"
                  % " ".join(names_to_run))
    return proc.rc([str(repo_dir / "run_docker"), "./" + script.label]
                   + script.extra_argv(),
                   env=proc.env_with(STAGES=" ".join(names_to_run)),
                   cwd=str(repo_dir))


def run(names_to_run):
    """Run our own stages, in the order given, in the environments they ask for.
    Returns the worst exit code any of them has produced.

    A stage that wants the container while this runs on the host is not run here: the
    stages that want it are handed back to this same script through ./run_docker, and
    that process runs them. Which is why a run starts on the host -- the container
    cannot do the reverse, and a stage that wants the host from inside it is refused
    with the command to type instead.

    A host without docker has no ./run_docker to hand them to. There the stages that
    want the container run in place, as if this were it (CUPO_DEV_CONTAINER=1): a
    user who installed the plugin with pip or conda, outside Docker, runs
    `CUPO_TEST_INSTALLED=1 ./run_test openmm_unit` in the environment that has it.
    """
    script = _own()
    must_run_here(names_to_run)
    choose_cuda_architectures(names_to_run)
    total, first = len(names_to_run), 1
    for where, chunk in _split_by_where(script, names_to_run):
        # A group of stages this tree has none of the files for is only so many
        # skips, and those are said here: entering the container would build its
        # image to say them.
        if where == here() or all(script.stages[n].unavailable() for n in chunk):
            _run_here(script, chunk, first, total)
        elif where == CONTAINER and not shutil.which("docker"):
            console.color(console.JOB, "=== no docker on this machine: running %s here, "
                          "as in the dev container ===" % " ".join(chunk))
            _run_here(script, chunk, first, total)
        elif where == CONTAINER:
            _record(_run_script(script, chunk))
        else:
            raise proc.Fatal(_cannot_leave(chunk))
        first += len(chunk)
        if _rc and not _keep_going:
            break
    return _rc


def _run_here(script, names_to_run, first, total):
    """Run stages in this process, each behind its own banner.

    The banner is the stage's own name and its place in the run, printed here rather
    than by each stage: a stage that prints its own is a stage that can print somebody
    else's, which is what happened when a shared helper hardcoded one -- three of four
    routes announced themselves as the fourth, in the log where that line is the only
    thing saying which route failed.

    A stage says how it went by what it returns: an exit code, or nothing. Both are
    normal. A stage that runs pytest has a code worth keeping even though pytest
    itself ran fine, and returns it; a stage that builds something has nothing to add
    beyond the build having worked, and the command it ran raises if it did not.
    """
    global _ran_here, _current
    _ran_here = True
    for i, name in enumerate(names_to_run, first):
        st = script.stages[name]
        print()
        console.color(console.STAGE, timing.section_start(
            name, "=== [%d/%d] %s ===" % (i, total, name)))
        reason = st.unavailable() or _skip_reason(st.skip_on)
        if reason:
            console.color(console.SKIP, "--- skipping %s: %s ---" % (name, reason))
            _end_section(name)
            continue
        _current = name
        # The stage's time is its pre, body and post: what a reader waits for.
        started = time.monotonic()
        timing.stage_started(name)
        addopts = pytest_times.stage_started(name)
        code = None
        try:
            # pre inside the try, so that a pre that fails is this stage failing: it
            # gets the banner below, it counts towards the verdict, and --keep-going
            # means what it says. The body is skipped, which is the point of a pre
            # that can fail: the body after it would not have what it needs.
            if st.pre:
                st.pre()
            code = st.fn()
        except subprocess.CalledProcessError as e:
            # The command already said what it was and how it failed; what this adds
            # is which stage it failed in, which is what a reader of a long log is
            # looking for.
            console.color(console.FAIL, "--- stage %s failed (exit %d): %s ---"
                          % (name, e.returncode, proc.fmt(e.cmd)))
            code = e.returncode or 1
        except (OSError, proc.Fatal) as e:
            # A command that could not be started at all -- a script this tree does
            # not have, a tool not installed -- or a condition the stage found and
            # said in words. A stage failing, like any other, and not the end of the
            # run: with --keep-going the stages after it still run.
            console.color(console.FAIL, "--- stage %s failed: %s ---" % (name, e))
            code = 1
        finally:
            if st.post:
                st.post()
            _current = None
            pytest_times.stage_ended(addopts)
            timing.stage_ended()
            timing.record_stage(name, time.monotonic() - started, code)
            _end_section(name)
        if not _record(code):
            break


def _skip_reason(skip_on):
    """Why a stage does not run, from its skip_on: a callable, a list of them, or
    None. "" when it runs."""
    for check in (skip_on if isinstance(skip_on, (list, tuple)) else [skip_on]):
        reason = check() if check else ""
        if reason:
            return reason
    return ""


def _end_section(name):
    end = timing.section_end(name)
    if end:
        print(end, flush=True)


def run_with_depends(patterns):
    """Run what the command line named, and everything those stages need. Returns the
    worst exit code any of them produced.

    Every pattern has to name something. A name that matches nothing is a typo, and
    the listing is printed with the error because the answer to a mistyped stage name
    is nearly always visible in the list of the ones there are.

    What comes back is run order, and run() takes it from there: it is run() that
    knows which stages want the container and hands those over in one go.
    """
    asked, unknown = [], []
    for pattern in patterns:
        hits = match(pattern)
        if not hits:
            unknown.append(pattern)
        asked += hits

    if unknown:
        list_stages()
        raise proc.Fatal("no stage matches: " + " ".join(unknown))

    # Every stage can be got to, asked before any of them runs -- run() asks the same
    # question, but asking it here is asking it about the whole list. A run that builds
    # for ten minutes and then says the stage it was building for needed another
    # machine has spent the ten minutes on something it knew at the start.
    plan = with_depends(asked)
    must_run_here(plan)
    return run(plan)


# ── Logs ──────────────────────────────────────────────────────────────────────
#
# A stage that runs a test program tees its output to a log, and summary() prints the
# interesting lines of each at the end -- for a reader who scrolled past them while
# they streamed.
#
# Which logs to print is not a list anyone writes down: it is the logs this run wrote.
# A stage asks log_path() for its log and gets it recorded in the same breath. logs/
# is never emptied, so a written-down list printed the last run's numbers under this
# run's summary -- a `./run_test cupo_unit` reported on benchmarks it never ran.

logdir = repo_dir / "logs"

_logs = []

# The stage running right now, or None between stages. Kept by _run_here, and read by
# log_path() below, which is the only thing that wants it: a stage's log is named
# after the stage, and the stage should not have to say its own name to get that.
_current = None


def log_path():
    """Where the running stage's log goes, recorded as one this run wrote.

    Named after the stage -- cupo_unit writes logs/cupo_unit.log -- so the log a reader
    is pointed at says which stage wrote it, and a stage renamed keeps the two in step.

    A string, because every caller hands it straight to proc.tee.
    """
    if not _current:
        raise proc.Fatal("log_path() called outside a stage")
    path = logdir / (_current + ".log")
    if path not in _logs:
        _logs.append(path)
    return str(path)


def summary():
    """The last few interesting lines of each log this run wrote, so a reader who
    scrolled past the output still gets the counts.

    Nothing at all when the run wrote no logs. The stages that write none are the
    builds and the packaging stages, and a summary of nothing under a heading reads
    like an answer.
    """
    if not _logs:
        return
    interesting = re.compile(r"pass|error|fail|skip|deselected", re.I)

    console.color(console.JOB, "\n=== Summary ===\n")
    for log in _logs:
        if not log.is_file():
            continue
        console.color(console.STAGE, "--- %s ---" % log)
        hits = [line for line in log.read_text(errors="replace").splitlines()
                if interesting.search(line)]
        for line in hits[-10:]:
            print(line)
        print()


# ── What the run is passing through to pytest ─────────────────────────────────
#
# Kept here rather than by the script that parses it, because two things want it and
# neither is the parser: the stages that run pytest, and filtered() below, which is
# how a stage says a filter makes it pointless.

_pytest_args = []


def set_pytest_args(args):
    """Take the arguments this run passes through to pytest -- -k, -m, -x and friends,
    everything the command line did not claim for itself."""
    global _pytest_args
    _pytest_args = list(args)


def pytest_args():
    """Those arguments, for a stage that ends in a pytest command."""
    return list(_pytest_args)


def _split_mark(args):
    """Pull a -m expression out of the arguments this run passes to pytest, returning
    what is left and the expression, or None if none was typed.

    A stage's own -m and a -m on the command line are two halves of one question --
    "the benchmark tests, of the ones this stage runs" -- and pytest has no way to be
    asked that twice: its -m is a store option, so a second one wins outright and the
    stage's own half is silently lost. `./run_test -m benchmark` used to make
    cupo_unit, which asks for "not benchmark", run the benchmarks instead. Taking the
    typed half out here is what lets run_pytest below put the two together itself.

    Both spellings pytest takes: -m EXPR and -mEXPR.
    """
    rest, expr, it = [], None, iter(args)
    for a in it:
        if a == "-m":
            expr = next(it, "")
        elif a.startswith("-m"):
            expr = a[2:]
        else:
            rest.append(a)
    return rest, expr


def run_pytest(paths, mark, env=None, shard=False):
    """Run pytest over `paths` with a -m expression, teeing to the stage's log.
    Returns pytest's exit code, which is the stage's.

    shard=True is for a stage whose tests a CI job splits between its shards
    (ci_shard): pytest then loads the CI's shard plugin, which keeps this shard's
    share. Outside such a job it changes nothing.

    Here rather than with the stages that call it because the log is half of it:
    log_path() names the log after the stage and tells summary() it belongs to this
    run, and neither is something a stage should have to say.

    The stage's mark and a typed one are ANDed, so that a typed one narrows the stage
    rather than replacing it. Under `-m benchmark` that leaves cupo_unit asking for
    "(not benchmark) and (benchmark)", which selects nothing -- and selecting nothing
    is the filter working, not the stage failing, so pytest's exit 5 counts as success
    whenever a filter was typed at all.
    """
    extra, typed = _split_mark(pytest_args())
    if typed:
        mark = "(%s) and (%s)" % (mark, typed)
    # --capture=sys, not pytest's default fd. cupo reports a fatal to stderr and then
    # calls abort(), and fd-level capture holds that message in a buffer the dying
    # process never gets to report -- leaving "Fatal Python error: Aborted" and a
    # Python traceback that stops at the call into C, with the one line saying why
    # thrown away. Under sys-level capture the C write goes straight to the log.
    # Python-level output is still captured per test, and the capfd fixture still
    # captures at fd level for the tests that ask for it.
    cmd = ["python3", "-m", "pytest"] + [str(p) for p in paths] + \
          ["-m", mark, "-v", "--capture=sys"] + extra
    if shard and ci_shard():
        # Appended to PYTHONPATH rather than put first: ci/ holds directories named
        # lib, scripts and tests, which should not hide anything the tests import.
        env = dict(env if env is not None else os.environ)
        env["PYTHONPATH"] = ":".join(
            p for p in (env.get("PYTHONPATH"), str(repo_dir / "ci")) if p)
        cmd += ["-p", "lib.pytest_shard"]
    code = proc.tee(cmd, log_path(), env=env)
    return 0 if code == 5 and pytest_args() else code


def ci_shard():
    """(k, N) when this run is shard k of a CI test job split N ways, else None.
    The CI sets CI_NODE_INDEX and CI_NODE_TOTAL only for a job whose tests are
    split. A local run has neither, and a tree without the shards module is never
    a shard.
    """
    if shards is None:
        return None
    try:
        return shards.shard()
    except ValueError as e:
        raise proc.Fatal(str(e))


def later_shard():
    """Why a stage that runs once per CI job does not run here, or "" if it does.

    In a job split into shards (ci_shard) the stages that are not split -- the
    samples, the benchmarks, the LTO-IR programs -- run in shard 1 and in no other,
    so that they run once per job. Written as a stage's own skip_on, the way
    filtered() is.
    """
    which = ci_shard()
    if not which or which[0] == 1:
        return ""
    return "runs in shard 1 of %d only" % which[1]


def filtered():
    """Why a pytest filter makes a stage pointless, or "" if it does not.

    A stage that runs a program rather than a test file takes no -k and no -m, so
    under either it would run in full on every iteration of what is meant to be a
    fast, targeted loop. Written as a stage's own skip_on rather than as a check
    inside each such stage.

    -m counts for the same reason -k does, and used not to: `./run_test -m benchmark`
    ran openmm_sample and every packaging stage in full.
    """
    return ("pytest filter given"
            if any(a.startswith("-k") or a.startswith("-m") for a in _pytest_args)
            else "")


# ── Packaging stages ──────────────────────────────────────────────────────────
#
# What ./run_test does to a release tarball, rather than to the tree it was built
# from. Here rather than with the stages themselves because what makes one of these a
# packaging stage is the same three answers every time, and a stage should not have to
# remember all three.

def commit_sha_short():
    """The short commit hash the tarball names end in.

    Whoever knows it can pass it in -- CI does, and so does the host side of a run
    that dispatches into the dev container -- and this is the fallback for a caller
    that did not. Asking git has to survive git not being there at all: check=False
    covers a git that answers badly, and OSError a git that cannot be run, which is
    the case inside our own image.

    Answered once and put in the environment, because make_release_tarball reads it
    from there.
    """
    if not os.environ.get("COMMIT_SHA_SHORT"):
        with proc.untraced():
            try:
                sha = proc.capture(["git", "-C", str(repo_dir),
                                    "rev-parse", "--short", "HEAD"],
                                   check=False, stderr=subprocess.DEVNULL)
            except OSError:
                sha = ""
        os.environ["COMMIT_SHA_SHORT"] = sha or "unknown"
    return os.environ["COMMIT_SHA_SHORT"]


def artifacts_dir():
    """Where the release tarballs are.

    ARTIFACTS_DIR when a caller says so -- a CI job downloads the build jobs' artifacts
    into its own working directory -- and otherwise the artifacts/ directory the
    collect_artifacts stage fills, which is where a build on this machine puts them.

    This is where tarballs are read from. It is deliberately not where collect_artifacts
    writes them: that stage empties the directory first, and in CI this one holds the
    artifacts of the jobs upstream.
    """
    return pathlib.Path(os.environ.get("ARTIFACTS_DIR") or (repo_dir / "artifacts"))


def no_artifacts():
    """Why a stage that unpacks has nothing to unpack, or "" if it has.

    ARTIFACTS_DIR is how a job says it handed something over. Without it we are a
    working tree that built its own, and unpacking there would drop whatever tarballs a
    local `./run_test cupo_tarball` left in artifacts/ on top of the tree that built
    them. Written as a stage's own skip_on, the way filtered() is.
    """
    if os.environ.get("ARTIFACTS_DIR"):
        return ""
    return "nothing was handed to this tree (no ARTIFACTS_DIR)"


def unpack(pattern, subdir):
    """Unpack every tarball matching pattern into repo_dir/subdir.

    What a job does with the tarballs an upstream job handed it. Here rather than with
    the stages that call it because the two halves belong together: this reads the
    directory artifacts_dir() above names.

    The pattern may match nothing, and often does -- a stage asks for what it needs and
    takes whatever of it the upstream job actually produced.
    """
    for tarball in sorted(artifacts_dir().glob(pattern)):
        target = repo_dir / subdir
        print("  unpacking %s -> %s/" % (tarball.name, subdir))
        target.mkdir(parents=True, exist_ok=True)
        # --strip-components=1: every tarball has a single top-level directory named
        # after the release, and the tree wants its contents, not it.
        proc.run(["tar", "-xzf", str(tarball), "-C", str(target),
                  "--strip-components=1"])


def needs_docker():
    """Refuse, rather than skip, when a packaging stage cannot be run here.

    A pre hook rather than a skip_on, and that difference is the whole point. None of
    these stages runs by default, so reaching one means somebody named it -- in STAGES
    or on the command line -- and a named stage that quietly does not run is how a CI
    job passes having tested nothing. A skip would be exactly that. This is an error.
    """
    if not shutil.which("docker"):
        raise proc.Fatal("no docker on this host, and every packaging stage hands a "
                         "release tarball to one")
    if not artifacts_dir().is_dir():
        raise proc.Fatal("no artifacts directory at %s -- collect_artifacts makes "
                         "one, or point ARTIFACTS_DIR at the tarballs"
                         % artifacts_dir())


def packaging_stage(**kw):
    """A stage that takes a release tarball out of the artifacts directory and hands
    it to docker: what we ship, rather than the tree we built it from.

    Three things are true of every one of them, so they are written once here instead
    of seven times over:

      where="host"  there is no docker inside the dev container -- nor, for the
                    customer-environment routes, may there be: what those test is that
                    the artifact works somewhere that is *not* our devel image.
      pre           a host that cannot run them says so and stops. See needs_docker.
      skip_on       these stages take no -k, so under one they would run in full on
                    every iteration of what is meant to be a fast, targeted loop.

    autorun is not among them, and that is the fourth thing they share: naming one is
    the only way to run one.
    """
    return stage(where=HOST, pre=needs_docker, skip_on=filtered, **kw)


# ── Stages that compile ───────────────────────────────────────────────────────
#
# nvcc through sccache, which is a server: it has to be up before the first compile,
# and what it did is worth reading after every stage that compiled. Written as hooks
# a stage opts into, so a run that compiles nothing neither starts the server nor
# reports on it.

_sccache_started = False


def sccache_start():
    """Have the sccache server up before the first compile, say what we are
    compiling for, and zero the server's statistics.

    The server once per run, however many compiling stages it asks for: the first
    compile would start the server anyway, and paying for that inside a timed build
    step is the whole of what this avoids. run_docker provides SCCACHE_DIR and
    USE_SCCACHE through the environment and a volume mount (host ~/.sccache ->
    container /tmp/sccache).

    The statistics at every stage: they count from the server's start, and each
    stage's should be its own.
    """
    global _sccache_started
    if _sccache_started:
        proc.rc(["sccache", "--zero-stats"])
        return
    _sccache_started = True

    # flush: everything around this line is written by a child process straight to
    # the same fd, so unflushed the line lands after the compile it announces.
    print("=== CUDA_ARCHITECTURES=%s ==="
          % os.environ.get("CUDA_ARCHITECTURES", "<cmake default>"), flush=True)

    sccache_dir = os.environ.get("SCCACHE_DIR")
    if sccache_dir:
        proc.rc(["ls", "-al", sccache_dir])
        proc.rc(["du", "-sh", sccache_dir])
    # The server's own log, with the run's timings (ci/lib/timing.py), so in CI it
    # is in the job's timing/ artifact. Without it a server that stops halfway
    # through a build says why nowhere: the compiles after it only print "The
    # server looks like it shut down unexpectedly, compiling locally instead".
    # "info" is a few lines each time a server starts and stops, and its warnings,
    # errors and panics. A server a later compile starts again appends to the same
    # file, so the one that stopped keeps its lines.
    os.environ.setdefault("SCCACHE_ERROR_LOG",
                          str(timing.TIMING_DIR / compile_timing.SCCACHE_LOG_NAME))
    os.environ.setdefault("SCCACHE_LOG", "info")
    # Never idle out. The server stops itself after SCCACHE_IDLE_TIMEOUT (600 s by
    # default) with no new request, even with compiles still running: in
    # the CI's cupo build, one source file takes 16 minutes for eight architectures and the
    # compact device links do not go through sccache, so ten minutes passed without
    # a new compile. The compiles still running then ran again without the cache,
    # and the next ones raced each other to start a new server ("Address in use"),
    # so 95 of 105 compiles missed it. The server ends with the container anyway.
    os.environ.setdefault("SCCACHE_IDLE_TIMEOUT", "0")
    proc.rc(["sccache", "--start-server"])
    proc.rc(["sccache", "--zero-stats"])


def sccache_stats():
    """What the cache did for the stage that just compiled.

    Per stage rather than once at the end of the run, because that is the question
    worth asking: a build that took twenty minutes and a build that took two differ
    by their hit rate, and one number covering every stage cannot say which stage
    missed.

    Best effort. It runs in the post hook, which runs whether the stage passed or
    failed -- a failed build is exactly when the hit rate is worth seeing -- and no
    sccache on this machine is not a second failure to report.

    The same statistics as JSON go to the timing directory, for the report
    (ci/lib/compile_timing.py).
    """
    if proc.rc(["sccache", "--show-stats"]) != 0:
        return
    stats = proc.capture(["sccache", "--show-stats", "--stats-format=json"],
                         check=False)
    if stats:
        compile_timing.sccache_file(_current).write_text(stats + "\n")


def compiling_stage(fn=None, **kw):
    """A stage that has a chance of running nvcc or a C++ compiler.

    Which is more stages than the ones named after building: a test stage whose
    script builds what it is about to test compiles too, and its hit rate counts the
    same.

    Bare and with arguments both, as @stage is, because most of these stages have
    nothing else to say.
    """
    return stage(fn, pre=sccache_start, post=sccache_stats, **kw)


# The stages that build a release tarball. A run with one of these in it builds for
# every architecture: see cuda_architectures.
_tarball_stages = set()


def tarball_stage(fn=None, **kw):
    """A compiling stage whose build is a release tarball, so it is for every GPU we
    ship for and not only the ones in this machine."""
    def deco(f):
        _tarball_stages.add(f.__name__)
        return compiling_stage(f, **kw)
    return deco(fn) if fn is not None else deco


def gpu_compute_caps():
    """The compute capability of each GPU this run uses, as nvidia-smi gives it:
    "8.9". NV_GPU says which GPUs, as it does for ./run_docker.

    An empty list when there is no nvidia-smi or it finds no GPU.
    """
    cmd = ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    gpu = os.environ.get("NV_GPU") or "all"
    if gpu != "all":
        cmd += ["-i", gpu]
    with proc.untraced():
        try:
            out = proc.capture(cmd, check=False, stderr=subprocess.DEVNULL)
        except OSError:
            return []
    return re.findall(r"^\d+\.\d+$", out, re.M)


def cuda_architectures(env, gpus, makes_tarball):
    """What a run should build for: (CUDA_ARCHITECTURES, why).

    A local run builds for the GPUs of this machine only, since it only ever runs
    on those. On an RTX 6000 Ada, cupo_build took 2m41s for sm_89 and about 14
    minutes for the full list of eight.

    The full list -- None here, which leaves the CMake default -- in CI, for a
    release tarball, and when there is no GPU to build for. A CUDA_ARCHITECTURES
    already in the environment is the caller's choice and is kept.

    `gpus` is a callable that returns the compute capabilities ("8.9"), so that
    nvidia-smi is only asked when the answer is needed.
    """
    if env.get("CUDA_ARCHITECTURES"):
        return env["CUDA_ARCHITECTURES"], "from the environment"
    if env.get("CI"):
        return None, "CI builds every architecture"
    if makes_tarball:
        return None, "a release tarball is for every architecture"
    if env.get("NO_GPU"):
        return None, "NO_GPU is set"
    caps = gpus()
    if not caps:
        return None, "no GPU found"
    archs = sorted({c.replace(".", "") for c in caps}, key=int)
    return ";".join(archs), ("this machine's GPUs; set CUDA_ARCHITECTURES to "
                             "override")


def choose_cuda_architectures(names_to_run):
    """Decide what a run that compiles builds for, and say so.

    Once per run, by the process that started it: it is put in the environment, and
    ./run_docker passes it into the dev container, whose process keeps it. The whole
    run has to agree, because every build of it shares the same build directories.
    """
    if not timing.started_here():
        return
    if not any(_stage_of(n).pre is sccache_start for n in names_to_run):
        return
    archs, why = cuda_architectures(os.environ, gpu_compute_caps,
                                    any(n in _tarball_stages for n in names_to_run))
    if archs:
        os.environ["CUDA_ARCHITECTURES"] = archs
    console.color(console.JOB, "=== CUDA_ARCHITECTURES=%s (%s) ==="
                  % (archs or "<cmake default>", why))


# ── Command line ──────────────────────────────────────────────────────────────
#
# The whole of it, here rather than in the script that writes the stages: what the
# command line does is choose stages and run them, and both of those are this file.
# A stage script keeps what is its own -- its stages, and the examples in its --help.

EPILOG = """\
Any argument this script does not recognise is a pytest argument, so -k, -m, -x and
friends mean there what they mean there. Stage names come first on the line, before
the first option, so that `./run_test cupo_unit -k test_p1` can tell the stage from
the filter.

Typed with no stage name, this runs the stages marked * in --list-stages: the unit
tests, the benchmarks and the sample, plus the builds those need. Every other stage
runs when it is named, and only then -- which is what keeps a bare ./run_test from
building release tarballs and starting docker containers.

A stage name is a shell pattern, and a plain word that matches no stage exactly is
tried again as *word*: `./run_test customer_env` asks for the whole
customer-environment family.

Most stages want the dev container and this script enters it for them, so
`./run_docker ./run_test cupo_unit` and `./run_test cupo_unit` are the same run. The
packaging stages want the host, where there is a docker -- ./run_docker in front of
one of those is one word too many, and is refused saying so.

For CI:
  STAGES="cupo_unit" ./run_test     exactly those stages, no dependencies

STAGES takes the same shell patterns and does not follow dependencies, because in CI
each one ran in another job on another machine and arrived here as an artifact. A
stage named on the command line does follow them. Every STAGES pattern has to match
some stage: one that matches nothing would run nothing and pass. See ci/lib/stages.py.
"""


def parse_args(argv, usage):
    """The command line: stage names, our own options, and pytest's.

    Stage names are the words before the first option, taken here rather than by
    argparse, because argparse cannot tell a positional of ours from the value of an
    option it has never heard of -- and every option it has never heard of is
    pytest's. `./run_test cupo_unit -k test_p1` has to mean one stage and one filter,
    not two stages.

    `usage` is the calling script's own examples, which name its own stages. The only
    part of this that is not the same for every stage script there could be.
    """
    names, rest = [], list(argv)
    while rest and not rest[0].startswith("-"):
        names.append(rest.pop(0))

    parser = argparse.ArgumentParser(
        prog=sys.argv[0], allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build and test cupo and its plugins, and the tarballs they ship "
                    "in.\n\n" + usage,
        epilog=EPILOG,
        usage="%(prog)s [options] [STAGE ...] [pytest arguments ...]")
    parser.add_argument("--stage", action="append", nargs="?", const="", default=[],
                        metavar="NAME",
                        help="a stage to run, as a positional name does. Bare, it "
                             "lists the stages, as --list-stages does.")
    parser.add_argument("--keep-going", action="store_true",
                        help="run every stage asked for even after one fails, and "
                             "report the worst code at the end. Off by default: the "
                             "first failure ends the run.")
    parser.add_argument("--list-stages", action="store_true",
                        help="print every stage there is, with what each runs after, "
                             "and exit")
    opts, extra = parser.parse_known_args(rest)

    opts.list_stages = opts.list_stages or "" in opts.stage
    opts.stages = names + [s for s in opts.stage if s]
    opts.pytest_args = extra
    return opts


def _main(usage):
    opts = parse_args(sys.argv[1:], usage)
    set_pytest_args(opts.pytest_args)
    set_keep_going(opts.keep_going)

    # What to put after the script's path when stages are re-run inside the dev
    # container. A filter typed on the host is meant for the pytest that ends up
    # running in there, and so is --keep-going, since the run it governs is that one.
    own_argv(lambda: (["--keep-going"] if opts.keep_going else [])
             + opts.pytest_args)

    # A listing answers a question and should answer with the answer alone: no trace.
    if opts.list_stages:
        proc.set_trace(False)
        list_stages()
        return 0

    os.environ["PYTHONUNBUFFERED"] = "1"  # stream Python output live through tee
    logdir.mkdir(parents=True, exist_ok=True)
    timing.begin()

    # Answered here, on whichever machine started the run, and put in the environment
    # for anything this run hands off to -- the container it may enter has no git.
    commit_sha_short()

    patterns = from_env()
    if opts.stages:
        # Named on the command line: those stages and everything they need.
        code = run_with_depends(opts.stages)
    elif patterns:
        # STAGES: exactly those stages, no dependencies, because in CI each dependency
        # ran in another job and arrived here as an artifact. Checked first, against
        # every stage there is: a pattern that matches nothing runs nothing, and a run
        # that runs nothing passes, which is how a name misspelled in the CI's configuration
        # becomes a green job that tested nothing at all.
        check_patterns(patterns)
        code = run(selected(patterns))
    else:
        # Nothing named at all: the autorun stages, and what those need.
        code = run_with_depends(selected())

    proc.set_trace(False)

    # The log summary belongs to whoever ran the stages that wrote the logs. A run
    # that handed a group to the dev container gets it from that process, out of the
    # same files -- so this process reading them again would print them twice.
    if ran_here() and not dispatched():
        summary()

    # How the run went is said once, by the process that started it, and last: the
    # stages that ran with their times -- those the dev container ran included, since
    # they are in the same files (ci/lib/timing.py) -- then where the full report is,
    # then the verdict. What ran is said above the verdict because PASS says the
    # stages that ran were fine and nothing about which ones those were, and a stage
    # skipped by its skip_on, or a STAGES pattern that matched less than the reader
    # thought, both end in a PASS that looks complete.
    if timing.started_here():
        report = timing.write_report()
        print()
        timing.print_report(report)
        console.color(console.JOB, "Ran stages: %s  (See all with ./run_test "
                      "--list-stages)" % (timing.ran_line() or "none"))
        failed = failed_line()
        if failed:
            console.color(console.FAIL, "Failed stages: %s" % failed)
        if report:
            console.color(console.JOB, "Timings: %s" % os.path.relpath(report))
        print()
        console.color(console.PASS if code == 0 else console.FAIL,
                      "PASS" if code == 0 else "FAIL")
    return code


def failed_line():
    """The stages of this run that failed, with their exit codes, for the line
    under the stages that ran: "github_pip_via_pypi (exit 1), cupo_unit (exit 2)".
    "" when none did. More than one only with --keep-going."""
    return ", ".join("%s (exit %d)" % (name, code) for name, _secs, code in timing.stages()
                     if code not in (None, 0, proc.PYTEST_NO_TESTS))


def main(usage):
    """Run the whole command line and exit with its code.

    What a stage script's `if __name__ == "__main__"` calls, and it does not return:
    proc.main turns a Fatal, a failed command or a Ctrl-C into the exit code a caller
    further up expects to see.
    """
    proc.main(lambda: _main(usage))
