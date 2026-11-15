# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""What the compiling stages of a run compiled, and how long it took. STDLIB ONLY,
like the rest of ci/lib.

Two sources, both files in timing.TIMING_DIR:
  compile.jsonl         one line per file compiled, from the compile wrapper,
                        which the build scripts put in front of the compiler: stage,
                        file, seconds, exit code
  sccache-<stage>.json  sccache's statistics for one stage, saved by the stage's post
                        hook (sccache_stats in ci/lib/stages.py)

sccache counts hits and misses, and not the time of each file. timed-compile has the
time of each file, and not whether it came from the cache. The report section has
both, per stage.
"""
import json
import re

from . import timing

COMPILES_FILE_NAME = "compile.jsonl"

# How many of a stage's files the report names, slowest first.
SLOWEST = 15


# The sccache server's log (stages.sccache_start).
SCCACHE_LOG_NAME = "sccache-server.log"


def sccache_server_starts():
    """How many times an sccache server started in this run, by its log."""
    path = timing.TIMING_DIR / SCCACHE_LOG_NAME
    if not path.is_file():
        return 0
    return path.read_text(errors="replace").count("server started, listening")


def sccache_log_lines():
    """The sccache server log's warnings, errors and panics, as written."""
    path = timing.TIMING_DIR / SCCACHE_LOG_NAME
    if not path.is_file():
        return []
    return [line for line in path.read_text(errors="replace").splitlines()
            if re.search(r"\b(WARN|ERROR)\b|panicked", line)]


def sccache_file(stage):
    return timing.TIMING_DIR / ("sccache-%s.json" % stage)


def compiles(links=False):
    """Every file compiled in this run, in the order they finished:
    [(stage, file, secs, code)]. With links=True, every link instead, with the file
    it made."""
    path = timing.TIMING_DIR / COMPILES_FILE_NAME
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("link", False) == links:
                out.append((r["stage"], r["file"], r["secs"], r["code"]))
    return out


def _secs(duration):
    """sccache writes a duration as {"secs": ..., "nanos": ...}."""
    return duration["secs"] + duration["nanos"] / 1e9


# The names sccache counts the phases of an nvcc compile under, besides the compile
# itself ("CUDA"): cudafe++, cicc and ptxas.
NVCC_PHASES = ("CUDA (Device code)", "PTX", "CUBIN")


def sccache_totals(stage):
    """The numbers worth reading from sccache's statistics for one stage, or None
    when the stage saved none.

    sccache splits one nvcc compile into its phases and looks each up in the cache
    on its own: a file that misses can still find its ptxas output there. So the
    hits and misses are given twice, for the files and for the phases. Every nvcc
    compile also counts as one non-cacheable compilation (its phases are what is
    cached), and every phase sccache ran as one compiler run, which is what the
    average compile time is of.
    """
    path = sccache_file(stage)
    if not path.is_file():
        return None
    s = json.loads(path.read_text())["stats"]
    hits, misses = s["cache_hits"]["counts"], s["cache_misses"]["counts"]

    def count(counts, phases):
        return sum(n for kind, n in counts.items() if (kind in NVCC_PHASES) == phases)

    runs = s["compilations"]
    return {
        "requests": s["compile_requests"],
        "hits": count(hits, False),
        "misses": count(misses, False),
        "phase_hits": count(hits, True),
        "phase_misses": count(misses, True),
        "non_cacheable": s["non_cacheable_compilations"],
        "compiler_runs": runs,
        "average_compile": _secs(s["compiler_write_duration"]) / runs if runs else None,
    }


def _rate(hits, misses):
    return "%.0f%%" % (100 * hits / (hits + misses)) if hits + misses else "-"


def _sccache_lines(t, compiled=None):
    """The sccache totals of a stage. compiled is how many files the stage
    compiled (timed-compile's count): sccache's statistics only know the compiles
    that reached it, and a server that stopped and started again knows only those
    since, so "10 hits, 100%" alone can be 10 of 105 compiles."""
    # A stage whose build does not go through sccache, because USE_SCCACHE is
    # unset, still starts the server and reads its stats.
    if not t["requests"]:
        return ["sccache: not used"]
    average = ("%.1fs" % t["average_compile"] if t["average_compile"] is not None
               else "-")
    if compiled and compiled > t["requests"]:
        seen = "%d of the %d compiles" % (t["requests"], compiled)
    else:
        seen = "%d requests" % t["requests"]
    return [
        "sccache: %s; files: %d hits, %d misses (hit rate %s)"
        % (seen, t["hits"], t["misses"], _rate(t["hits"], t["misses"])),
        "         nvcc phases: %d hits, %d misses (hit rate %s); %d non-cacheable"
        % (t["phase_hits"], t["phase_misses"],
           _rate(t["phase_hits"], t["phase_misses"]), t["non_cacheable"]),
        "         %d compiler runs, %s on average"
        % (t["compiler_runs"], average),
    ]


# How many of the slowest links the report lists for each stage. Most links take
# under a second; the few that do not are what this is for.
LINKS_SHOWN = 5


# A file built once per GPU architecture names the architecture in its path: the
# compact convolution's per-architecture sources, and its cupo_compact_sm89.dir device
# link. Listed one by one, a build for eight architectures fills the list with the
# same file eight times. So the report lists such a file once, its architecture
# written sm*, with the time of all of them together.
_ARCH_RE = re.compile(r"(?<=sm)\d+|(?<=compute_)\d+")


def _grouped(rows):
    """[(name, count, total secs, slowest secs, worst exit code)], one per file
    with its architecture taken out (_ARCH_RE), the most time first. A file built
    for one architecture only keeps its own name."""
    groups = {}
    for file, secs, code in rows:
        name = _ARCH_RE.sub("*", file)
        first, count, total, slowest, worst = groups.get(name, (file, 0, 0.0, 0.0, 0))
        groups[name] = (first, count + 1, total + secs, max(slowest, secs),
                        worst or code)
    out = [(name if count > 1 else first, count, total, slowest, worst)
           for name, (first, count, total, slowest, worst) in groups.items()]
    return sorted(out, key=lambda g: -g[2])


def _secs_shown(secs):
    return "%.1fs" % secs if secs < 100 else timing.elapsed(secs)


def _slowest(rows, count=None):
    out = []
    for name, n, total, slowest, code in _grouped(rows)[:count or SLOWEST]:
        failed = "  (exit %d)" % code if code else ""
        several = "  (%d of them, the slowest %s)" % (n, _secs_shown(slowest)) \
            if n > 1 else ""
        out.append("      %8s  %s%s%s" % (_secs_shown(total), name, several, failed))
    return out


def _section():
    by_stage, links_by_stage = {}, {}
    for stage, file, secs, code in compiles():
        by_stage.setdefault(stage, []).append((file, secs, code))
    for stage, file, secs, code in compiles(links=True):
        links_by_stage.setdefault(stage, []).append((file, secs, code))
    # In the order the stages ran; files compiled outside any stage last.
    order = [name for name, _, _ in timing.stages()]
    order += sorted(set(by_stage) - set(order))
    lines = []
    for stage in order:
        files = by_stage.get(stage, [])
        links = links_by_stage.get(stage, [])
        totals = sccache_totals(stage)
        if not files and not links and not (totals and totals["requests"]):
            continue
        lines += ["", "  %s: %d files compiled, %s summed over the files"
                  % (stage or "(no stage)", len(files),
                     timing.elapsed(sum(secs for _, secs, _ in files)))]
        if totals:
            lines += ["    " + line for line in _sccache_lines(totals, len(files))]
        if files:
            lines.append("    slowest:")
            lines += _slowest(files)
        if links:
            lines.append("    links: %d, %s summed; slowest:"
                         % (len(links), timing.elapsed(sum(s for _, s, _ in links))))
            lines += _slowest(links, LINKS_SHOWN)
    if not lines:
        return []
    starts = sccache_server_starts()
    if starts > 1:
        lines += ["", "  the sccache server started %d times (%s): the statistics above "
                  "are the last server's only, and the compiles while none was up "
                  "missed the cache" % (starts, SCCACHE_LOG_NAME)]
    trouble = sccache_log_lines()
    if trouble:
        lines += ["", "  the sccache server logged %d warnings or errors (%s); the last:"
                  % (len(trouble), SCCACHE_LOG_NAME)]
        lines += ["    " + line for line in trouble[-5:]]
    return ["Compiles, per stage (wall time of each file; the build runs several "
            "at once):"] + lines


timing.SECTIONS.append(_section)
