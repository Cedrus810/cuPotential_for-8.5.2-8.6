# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A golden is many runs. This folds them into the numbers a run is compared to.

    golden/NVIDIA_L40S/
        1001/nvt.json           one run, in the format a single golden always had
        1002/nvt.json

Every file is a measurement. There is no committed summary: the central value is
computed from the run directories whenever it is wanted, which costs milliseconds
against a benchmark that runs for twenty minutes and removes a second copy of the
same fact that could disagree with the first.

Stdlib only, and no knowledge of OpenMM or of CI. The benchmark suite imports it,
and so do the scripts that collect and plot runs, by file path, because this file
has no importable home from either of those places.
"""
import json
import pathlib

# ── Benchmark groups ──────────────────────────────────────────────────────────
#
# The benchmark systems are split into groups, each run by a job of its own so
# that the groups measure in parallel (the openmm_bench and openmm_bench_triclinic
# stages of ./run_test). Each group keeps its goldens in a tree of its own, with
# the same shape:
#
#     plugins/openmm/tests/golden/<GPU>/<run>/{nvt,npt}.json             main
#     plugins/openmm/tests/golden-triclinic/<GPU>/<run>/{nvt,npt}.json   triclinic
#
# and a job's artifacts carry them as golden/<artifact dir>/<GPU>/<run>/... (and
# golden-baseline/... for the merge base measured on the same node).
#
# The trees are never merged into one file. A run's file is one job's measurement on
# one node, and its config and per-system telemetry describe that job alone; folding
# two jobs into one file would leave one node's clocks standing for the other's
# numbers. Readers that want a GPU's whole picture take the runs of every tree, which
# fold() handles as it is -- it works per system -- and count runs per group
# (run_count) so that two groups' single runs do not read as a golden of two.
#
# (group, committed tree under plugins/openmm/tests, artifact subdirectory). main
# first: it is the tree that predates groups, and the fallback a newer group's
# systems are compared against until that group has a golden of its own.
GROUPS = (("main", "golden", "plugin"),
          ("triclinic", "golden-triclinic", "plugin-triclinic"))
MAIN_GROUP = "main"


def group_tree(group):
    """The committed tree of a group, relative to plugins/openmm/tests."""
    return next(tree for name, tree, _ in GROUPS if name == group)


def group_artifact_dir(group):
    """The subdirectory of golden/ (and golden-baseline/) a group's job writes."""
    return next(art for name, _, art in GROUPS if name == group)


def group_of_artifact_dir(artifact_dir):
    """The group an artifact subdirectory belongs to, or None for one that is not a
    group's."""
    return next((name for name, _, art in GROUPS if art == artifact_dir), None)


def group_of(data):
    """The group a run belongs to: config.bench_group, which the benchmark writes and
    the readers set on a run they load from a group's tree. A run with neither
    predates groups, which makes it main's."""
    return (data.get("config") or {}).get("bench_group") or MAIN_GROUP


def tag_group(runs, group):
    """Mark every run as group's, in place, unless it already says. Returns runs."""
    for _, data in runs:
        config = data.setdefault("config", {})
        if isinstance(config, dict):
            config.setdefault("bench_group", group)
    return runs


def run_count(runs):
    """How many runs a golden has: the most any one group contributed.

    Runs of different groups measured different systems, so they are not more runs of
    the same thing -- one main run and one triclinic run is a golden of one run.
    """
    counts = {}
    for _, data in runs:
        group = group_of(data)
        counts[group] = counts.get(group, 0) + 1
    return max(counts.values(), default=0)


# Points further than this many robust deviations from the median are rejected.
# 3.0 reads as "three sigma" for well-behaved data because of the 1.4826 below.
REJECT_AT = 3.0

# Makes the median absolute deviation a consistent estimator of the standard
# deviation for normal data, so REJECT_AT can be read as a number of sigma.
_MAD_TO_SIGMA = 1.4826

# The smallest run count at which rejecting anything is meaningful. The median of
# two points is their midpoint and the deviation of each is half their distance, so
# the rule would reject one of any two runs that disagree -- which is the case worth
# reporting, not the case worth deleting.
MIN_RUNS_TO_REJECT = 4

# A floor on the scale, as a fraction of the median. Without it, runs that agree to
# four significant figures give a deviation of nearly zero, and the next run is an
# outlier for being half a percent away.
SCALE_FLOOR_FRAC = 0.02


def sig(x, n=4):
    """Round to n significant digits, as a golden has always stored numbers."""
    if not x:
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + n - 1)


def median(values):
    s = sorted(values)
    n = len(s)
    if not n:
        raise ValueError("median of nothing")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def keep(values):
    """Split values into (kept, rejected) by distance from the median.

    Returns everything as kept when there are too few points for the question to
    mean anything; see MIN_RUNS_TO_REJECT.
    """
    values = list(values)
    if len(values) < MIN_RUNS_TO_REJECT:
        return values, []
    med = median(values)
    mad = median([abs(v - med) for v in values])
    scale = max(_MAD_TO_SIGMA * mad, SCALE_FLOOR_FRAC * abs(med))
    if not scale:
        return values, []
    limit = REJECT_AT * scale
    kept = [v for v in values if abs(v - med) <= limit]
    rejected = [v for v in values if abs(v - med) > limit]
    # Never reject everything: a scale of zero with a median nothing sits on would
    # otherwise empty the list and leave nothing to compare against.
    return (kept, rejected) if kept else (values, [])


def central(values):
    """The number a run is compared against: the mean of what survives `keep`."""
    kept, _ = keep(values)
    return sig(sum(kept) / len(kept))


def run_dirs(gpu_dir):
    """The run directories under one GPU, sorted. Empty when there are none."""
    gpu_dir = pathlib.Path(gpu_dir)
    if not gpu_dir.is_dir():
        return []
    return sorted(d for d in gpu_dir.iterdir() if d.is_dir())


def load_runs(gpu_dir, filename):
    """[(run_id, measurement)] for one ensemble, oldest directory name first.

    A directory without this ensemble's file is skipped rather than failing: a run
    that measured NVT and died before NPT still has an NVT number worth having.
    """
    out = []
    for d in run_dirs(gpu_dir):
        path = d / filename
        if not path.is_file():
            continue
        try:
            out.append((d.name, json.loads(path.read_text())))
        except (OSError, ValueError):
            continue
    return out


def metric_values(runs, precision, system, metric):
    """Every run's value for one cell, in run order. Runs missing it are skipped."""
    out = []
    for _, data in runs:
        cell = (data.get(precision) or {}).get(system)
        if isinstance(cell, dict) and isinstance(cell.get(metric), (int, float)):
            out.append(float(cell[metric]))
    return out


def metric_points(runs, precision, system, metric):
    """[(node, value)] for one cell, in run order.

    metric_values answers "what were the numbers"; this also answers "which
    machine produced each", which is what lets a plot show that three of four
    runs shared a node. The node is None for a run recorded before config.node
    existed.
    """
    out = []
    for _, data in runs:
        cell = (data.get(precision) or {}).get(system)
        if isinstance(cell, dict) and isinstance(cell.get(metric), (int, float)):
            out.append(((data.get("config") or {}).get("node"),
                        float(cell[metric])))
    return out


def paired_points(runs_a, runs_b, precision, system, metric):
    """[(node, value_a, value_b, run id)] for the runs of a and b with the same run id.

    A benchmark job that measures the merge base on its own node first writes both
    sides under its own CI job id, so a run id present on both sides is
    one node measured twice, an hour apart. Pairing them is what lets a gain be
    taken against that node's own baseline instead of the baseline's centre. Runs
    with no partner, or missing the cell, are left out.
    """
    a = {run_id: data for run_id, data in runs_a}
    out = []
    for run_id, data_b in runs_b:
        data_a = a.get(run_id)
        if data_a is None:
            continue
        cells = [(d.get(precision) or {}).get(system) for d in (data_a, data_b)]
        if not all(isinstance(c, dict) and isinstance(c.get(metric), (int, float))
                   for c in cells):
            continue
        out.append(((data_b.get("config") or {}).get("node"),
                    float(cells[0][metric]), float(cells[1][metric]), run_id))
    return out


def baseline_first(runs_a, runs_b):
    """{run id: whether its a side finished first}, for the run ids on both sides.

    From config.finished_at, which the CI puts on every golden a benchmark job
    writes. A pair without both times is taken as a first: that is the
    order every job ran in before it alternated.
    """
    def when(data):
        return (data.get("config") or {}).get("finished_at")
    a = {run_id: when(data) for run_id, data in runs_a}
    out = {}
    for run_id, data_b in runs_b:
        if run_id in a:
            ta, tb = a[run_id], when(data_b)
            out[run_id] = ta < tb if ta and tb else True
    return out


def gain_cell(runs_a, runs_b, precision, system, metric, higher_is_better,
              paired, first=None):
    """The gain of b over a for one cell, as the plots draw it, or None when a side
    has nothing for it.

    Returns {"gain": %, "points": [(node, value_b, pct, ratio)], "kept": [ratio],
    "rejected": [ratio], "order_effect": % or None, "sides": [bool or None]}.
    The ratio is the gain factor (b over a for a metric that is better higher, a
    over b for one better lower); pct is 100 * (ratio - 1).

    paired: each run of b against its own run of a, by run id (the merge base
    measured on the same node). The gain is then the mean of the two order groups'
    gains -- `first` says which pairs ran a first (baseline_first) -- so that an
    order effect cancels whatever the split; half their difference is returned as
    the order effect. Unpaired: every run of b against the centre of a.
    """
    if paired:
        pairs = paired_points(runs_a, runs_b, precision, system, metric)
        if not pairs:
            return None
        first = first or {}
        # Baseline-first pairs first.
        pairs.sort(key=lambda p: not first.get(p[3], True))
        ratios = [(v_b / v_a) if higher_is_better else (v_a / v_b)
                  for _, v_a, v_b, _ in pairs]
        points = [(node, v_b, 100.0 * (r - 1.0), r)
                  for (node, _, v_b, _), r in zip(pairs, ratios)]
        sides = [first.get(run_id, True) for *_, run_id in pairs]
        # Outliers are judged on the ratios, which sit near 1, not on the
        # percentages, which sit near 0: keep()'s scale floor is a fraction of the
        # median, and a median near 0 would take the floor away.
        kept, rejected = keep(ratios)
        groups = [[r for r, s in zip(ratios, sides) if s is side and r in kept]
                  for side in (True, False)]
        order_effect = None
        if all(groups):
            g_first, g_second = (100.0 * (central(g) - 1.0) for g in groups)
            gain = (g_first + g_second) / 2
            order_effect = (g_first - g_second) / 2
        else:
            gain = 100.0 * (central(ratios) - 1.0)
        return {"gain": gain, "points": points, "kept": kept, "rejected": rejected,
                "order_effect": order_effect, "sides": sides}
    base = metric_values(runs_a, precision, system, metric)
    mine = metric_points(runs_b, precision, system, metric)
    if not base or not mine:
        return None
    centre = central(base)
    if not centre:
        return None

    def factor(v):
        return v / centre if higher_is_better else centre / v
    ratios = [factor(v) for _, v in mine]
    points = [(node, v, 100.0 * (r - 1.0), r) for (node, v), r in zip(mine, ratios)]
    kept_values, rejected_values = keep([v for _, v in mine])
    return {"gain": 100.0 * (factor(central(kept_values)) - 1.0), "points": points,
            "kept": [factor(v) for v in kept_values],
            "rejected": [factor(v) for v in rejected_values],
            "order_effect": None, "sides": [None] * len(points)}


def fold(runs):
    """{precision: {system: {metric: central value}}} across every run.

    The shape a single golden file has always had, so a caller that used to read one
    file can use this instead and change nothing else.
    """
    out = {}
    for _, data in runs:
        for precision, systems in data.items():
            if not isinstance(systems, dict):
                continue
            for system, cell in systems.items():
                if not isinstance(cell, dict):
                    continue
                for metric, value in cell.items():
                    if isinstance(value, (int, float)):
                        out.setdefault(precision, {}).setdefault(
                            system, {}).setdefault(metric, [])
    for precision, systems in out.items():
        for system, metrics in systems.items():
            for metric in metrics:
                metrics[metric] = central(
                    metric_values(runs, precision, system, metric))
    return out


def warnings_by_node(runs):
    """{node: [warning, ...]} for the runs whose config carries warnings -- what was
    not as asked when the clocks were locked, such as a card that tops out below the
    requested clock (tests/gpu_clocks.py). Each warning once per node, in run order.
    A run recorded before config.node existed is filed under None."""
    out = {}
    for _, data in runs:
        cfg = data.get("config") or {}
        for warning in cfg.get("warnings") or ():
            seen = out.setdefault(cfg.get("node"), [])
            if warning not in seen:
                seen.append(warning)
    return out


def nodes(runs):
    """The distinct machines that contributed, for saying how thin a golden is."""
    seen = []
    for _, data in runs:
        node = (data.get("config") or {}).get("node")
        if node and node not in seen:
            seen.append(node)
    return seen
