# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What golden_runs.py does with a directory of runs. No GPU, no CI, no OpenMM."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import golden_runs as gr  # noqa: E402


def write_run(gpu_dir, run_id, single, node=None):
    d = pathlib.Path(gpu_dir) / run_id
    d.mkdir(parents=True)
    body = {"config": {"node": node or ("node-" + run_id)}, "single": single}
    (d / "nvt.json").write_text(json.dumps(body))
    return d


# ── keep(): which points survive ──────────────────────────────────────────────

def test_agreeing_runs_all_survive():
    kept, rejected = gr.keep([100.0, 101.0, 99.0, 100.5, 100.2])
    assert rejected == []
    assert len(kept) == 5


def test_one_wild_run_is_rejected():
    """The case the rule exists for: a GPU that fell off the bus, or a host that
    was uniformly slow, sitting in a set that otherwise agrees."""
    kept, rejected = gr.keep([100.0, 101.0, 99.0, 100.5, 30.0])
    assert rejected == [30.0]
    assert 30.0 not in kept


def test_nothing_is_rejected_below_four_runs():
    """The median of two points is their midpoint and each sits half the distance
    away, so any two runs that disagree would lose one. Two runs that disagree is
    the thing worth reporting."""
    for values in ([100.0, 30.0], [100.0, 30.0, 101.0]):
        kept, rejected = gr.keep(values)
        assert rejected == [], values
        assert kept == values


def test_identical_runs_do_not_make_the_next_one_an_outlier():
    """Four runs agreeing to four significant figures give a deviation of nearly
    zero. Without a floor on the scale, anything at all is then an outlier."""
    kept, rejected = gr.keep([100.0, 100.0, 100.0, 100.0, 101.0])
    assert rejected == []
    assert 101.0 in kept


def test_a_split_set_keeps_everything():
    """Two clusters three apart is not an outlier problem, it is two populations --
    two nodes, or a driver that changed. Deleting half of it would hide that."""
    kept, rejected = gr.keep([100.0, 101.0, 130.0, 131.0])
    assert rejected == []
    assert len(kept) == 4


# ── central(): the number a run is compared against ───────────────────────────

def test_central_is_the_mean_of_survivors():
    assert gr.central([100.0, 102.0]) == 101.0


def test_central_ignores_the_rejected_point():
    """Without rejection the wild run drags the centre down by a seventh, and
    every later comparison is against a number nothing measured."""
    values = [100.0, 101.0, 99.0, 100.5, 30.0]
    assert gr.central(values) == 100.1                 # mean of the four good ones
    assert abs(sum(values) / len(values) - 86.1) < 0.05   # what it would have been


def test_central_rounds_the_way_a_golden_always_has():
    """Goldens store four significant figures. A mean that did not round would
    warn about rounding noise on every comparison."""
    assert gr.central([1.0 / 3, 1.0 / 3, 1.0 / 3]) == 0.3333


# ── the directory ─────────────────────────────────────────────────────────────

def test_fold_has_the_shape_of_a_single_golden(tmp_path):
    """So a caller that used to read one file can call this and change nothing."""
    gpu = tmp_path / "NVIDIA_L40S"
    write_run(gpu, "1", {"apoa1": {"plugin_ns_per_day": 100.0}})
    write_run(gpu, "2", {"apoa1": {"plugin_ns_per_day": 102.0}})
    folded = gr.fold(gr.load_runs(gpu, "nvt.json"))
    assert folded == {"single": {"apoa1": {"plugin_ns_per_day": 101.0}}}


def test_a_metric_missing_from_one_run_still_folds(tmp_path):
    """A golden written before a metric existed, or a run that could not measure
    it. The metric folds over the runs that have it rather than vanishing."""
    gpu = tmp_path / "NVIDIA_L40S"
    write_run(gpu, "1", {"apoa1": {"plugin_ns_per_day": 100.0, "speedup_step": 2.0}})
    write_run(gpu, "2", {"apoa1": {"plugin_ns_per_day": 102.0}})
    folded = gr.fold(gr.load_runs(gpu, "nvt.json"))
    assert folded["single"]["apoa1"]["plugin_ns_per_day"] == 101.0
    assert folded["single"]["apoa1"]["speedup_step"] == 2.0


def test_no_runs_folds_to_nothing(tmp_path):
    """A GPU nobody has measured. The caller treats an empty fold the way it
    treats a missing golden today: record, compare against nothing."""
    assert gr.fold(gr.load_runs(tmp_path / "NVIDIA_NEW", "nvt.json")) == {}


def test_a_run_without_this_ensemble_is_skipped(tmp_path):
    """A run that measured NVT and died before NPT still has an NVT number."""
    gpu = tmp_path / "NVIDIA_L40S"
    write_run(gpu, "1", {"apoa1": {"plugin_ns_per_day": 100.0}})
    (gpu / "2").mkdir()
    assert len(gr.load_runs(gpu, "nvt.json")) == 1
    assert gr.load_runs(gpu, "npt.json") == []


def test_unreadable_run_does_not_take_the_golden_with_it(tmp_path):
    """Half a file from a job that was killed mid-write."""
    gpu = tmp_path / "NVIDIA_L40S"
    write_run(gpu, "1", {"apoa1": {"plugin_ns_per_day": 100.0}})
    bad = gpu / "2"
    bad.mkdir()
    (bad / "nvt.json").write_text('{"single": {"apoa1"')
    runs = gr.load_runs(gpu, "nvt.json")
    assert len(runs) == 1
    assert gr.fold(runs)["single"]["apoa1"]["plugin_ns_per_day"] == 100.0


def test_nodes_are_reported_and_deduplicated(tmp_path):
    """Two runs on one machine is a thinner golden than two runs on two, and the
    difference is not visible in the numbers."""
    gpu = tmp_path / "NVIDIA_L40S"
    write_run(gpu, "1", {"apoa1": {"plugin_ns_per_day": 100.0}}, node="a")
    write_run(gpu, "2", {"apoa1": {"plugin_ns_per_day": 102.0}}, node="a")
    write_run(gpu, "3", {"apoa1": {"plugin_ns_per_day": 101.0}}, node="b")
    runs = gr.load_runs(gpu, "nvt.json")
    assert gr.nodes(runs) == ["a", "b"]


def test_metric_values_keeps_run_order(tmp_path):
    """The plots draw a dot per run and colour it by node, so the values and the
    runs they came from have to stay in step."""
    gpu = tmp_path / "NVIDIA_L40S"
    for i, v in enumerate([100.0, 102.0, 101.0], start=1):
        write_run(gpu, str(i), {"apoa1": {"plugin_ns_per_day": v}})
    runs = gr.load_runs(gpu, "nvt.json")
    assert gr.metric_values(runs, "single", "apoa1", "plugin_ns_per_day") == [
        100.0, 102.0, 101.0]
    assert [r for r, _ in runs] == ["1", "2", "3"]


def test_the_config_block_is_not_mistaken_for_measurements(tmp_path):
    """`config` sits beside the precisions and holds numbers -- clocks, cpu MHz.
    Folding it would invent a precision called config."""
    gpu = tmp_path / "NVIDIA_L40S"
    d = gpu / "1"
    d.mkdir(parents=True)
    (d / "nvt.json").write_text(json.dumps({
        "config": {"graphics_clock_mhz": 2394, "node": "a"},
        "single": {"apoa1": {"plugin_ns_per_day": 100.0}}}))
    folded = gr.fold(gr.load_runs(gpu, "nvt.json"))
    assert "graphics_clock_mhz" not in json.dumps(folded)


# ── What a run records about itself ───────────────────────────────────────────

def test_the_node_comes_from_outside_the_container(monkeypatch):
    """The benchmark runs inside docker, where gethostname() is a container id and
    the job scheduler's variables are absent. The CI captures the real node outside
    and run_docker forwards it, so CUPO_NODE_NAME has to win over everything else.

    Getting this wrong is silent: the golden records a plausible-looking name and
    every run of the pipeline appears to be on a different machine.
    """
    import importlib.machinery
    import importlib.util
    src = pathlib.Path(__file__).resolve().parent / "test_bench_plugin.py"
    # Read the function out of the source rather than importing the module, which
    # pulls in OpenMM and a GPU.
    text = src.read_text()
    start = text.index("def _node_name(")
    end = text.index("def _gpu_config_block(")
    ns = {"os": __import__("os"), "socket": __import__("socket")}
    exec(compile(text[start:end], str(src), "exec"), ns)  # noqa: S102

    monkeypatch.setenv("CUPO_NODE_NAME", "gpu-node-17")
    monkeypatch.setenv("SLURMD_NODENAME", "should-not-win")
    assert ns["_node_name"]() == "gpu-node-17"

    monkeypatch.delenv("CUPO_NODE_NAME")
    assert ns["_node_name"]() == "should-not-win"


def _load_fn(name, upto):
    """One function out of test_bench_plugin.py, without importing the module --
    which would pull in OpenMM and want a GPU."""
    import subprocess
    src = (pathlib.Path(__file__).resolve().parent / "test_bench_plugin.py").read_text()
    body = src[src.index("def %s(" % name):src.index("def %s(" % upto)]
    # A copy of the subprocess namespace, not the module: the tests below replace its
    # check_output, and on the module itself that would outlive the test and reach
    # every later test in the session that runs a command.
    import types
    ns = {"subprocess": types.SimpleNamespace(**vars(subprocess)), "os": __import__("os")}
    exec(compile(body, "test_bench_plugin.py", "exec"), ns)  # noqa: S102
    return ns[name], ns


def test_the_card_is_recorded_not_just_the_machine(monkeypatch):
    """`node` names the machine, which is the card only on a one-GPU machine. A job
    given one card of several gets the one the job scheduler chooses, different run
    to run -- so without this two runs on one machine read as the same hardware when
    they may be different cards."""
    fn, ns = _load_fn("_gpu_identity", "_gpu_config_block")
    ns["subprocess"].check_output = lambda *a, **k: (
        "GPU-00000000-0000-0000-0000-000000000001, 00000000:01:00.0\n")
    assert fn() == {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001",
                    "gpu_pci_bus_id": "00000000:01:00.0"}


def test_the_allocated_card_is_the_one_recorded():
    """nvidia-smi inside the container sees only the GPU the job was given, so the
    first row is the card that measured. If a second ever appears, taking the first
    is still the right answer -- it is the one the benchmark uses, the same row
    _gpu_name() reads the product name from."""
    fn, ns = _load_fn("_gpu_identity", "_gpu_config_block")
    ns["subprocess"].check_output = lambda *a, **k: (
        "GPU-aaaa, 00000000:01:00.0\nGPU-bbbb, 00000000:02:00.0\n")
    assert fn()["gpu_uuid"] == "GPU-aaaa"


@pytest.mark.parametrize("output", ["", "\n", "nonsense", ", 00000000:01:00.0"])
def test_unreadable_card_identity_is_a_gap_not_a_failure(output):
    """A provenance field that cannot be read is worth a hole in the record, not a
    benchmark that refuses to run."""
    fn, ns = _load_fn("_gpu_identity", "_gpu_config_block")
    ns["subprocess"].check_output = lambda *a, **k: output
    assert fn() == {}


def test_no_nvidia_smi_is_a_gap_not_a_failure():
    fn, ns = _load_fn("_gpu_identity", "_gpu_config_block")
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    ns["subprocess"].check_output = boom
    assert fn() == {}


def test_metric_points_carries_the_node_with_the_value(tmp_path):
    """A plot that marks which runs shared a machine needs the pairing, and
    metric_values throws it away."""
    gpu = tmp_path / "NVIDIA_L40S"
    write_run(gpu, "1", {"apoa1": {"plugin_ns_per_day": 100.0}}, node="a")
    write_run(gpu, "2", {"apoa1": {"plugin_ns_per_day": 102.0}}, node="a")
    write_run(gpu, "3", {"apoa1": {"plugin_ns_per_day": 101.0}}, node="b")
    runs = gr.load_runs(gpu, "nvt.json")
    assert gr.metric_points(runs, "single", "apoa1", "plugin_ns_per_day") == [
        ("a", 100.0), ("a", 102.0), ("b", 101.0)]


def test_metric_points_survives_a_run_with_no_node(tmp_path):
    """A golden carried in from the single-file format has no config.node."""
    gpu = tmp_path / "NVIDIA_L40S"
    d = gpu / "legacy"
    d.mkdir(parents=True)
    (d / "nvt.json").write_text(json.dumps(
        {"single": {"apoa1": {"plugin_ns_per_day": 100.0}}}))
    runs = gr.load_runs(gpu, "nvt.json")
    assert gr.metric_points(runs, "single", "apoa1", "plugin_ns_per_day") == [
        (None, 100.0)]


def test_paired_points_match_runs_by_id():
    """A job that measured its merge base on the same node writes both sides under
    one run id; those, and only those, are pairs."""
    def run(run_id, ms, node):
        return (run_id, {"config": {"node": node},
                         "mixed": {"dhfr": {"plugin_step_ms": ms}}})
    base = [run("1", 1.00, "a"), run("2", 1.10, "b"), run("3", 0.90, "c")]
    mine = [run("2", 1.21, "b"), run("1", 0.98, "a"), run("4", 1.00, "d")]
    assert gr.paired_points(base, mine, "mixed", "dhfr", "plugin_step_ms") == [
        ("b", 1.10, 1.21, "2"), ("a", 1.00, 0.98, "1")]
    assert gr.paired_points(base, mine, "single", "dhfr", "plugin_step_ms") == []


def test_which_side_ran_first_comes_from_the_times():
    def run(run_id, when):
        return (run_id, {"config": {"finished_at": when} if when else {}})
    base = [run("1", "2026-10-01T10:00:00Z"), run("2", "2026-10-01T11:30:00Z"),
            run("3", None)]
    mine = [run("1", "2026-10-01T10:40:00Z"), run("2", "2026-10-01T10:50:00Z"),
            run("3", "2026-10-01T12:00:00Z"), run("4", "2026-10-01T12:00:00Z")]
    # No time on one side: taken as the baseline first, the order before alternation.
    assert gr.baseline_first(base, mine) == {"1": True, "2": False, "3": True}


def test_warnings_by_node():
    runs = [("1", {"config": {"node": "n1", "warnings": ["graphics clock low"]}}),
            ("2", {"config": {"node": "n2", "warnings": []}}),
            ("3", {"config": {"node": "n1", "warnings": ["graphics clock low",
                                                         "memory clock odd"]}}),
            ("4", {"config": {"node": "n3"}})]
    assert gr.warnings_by_node(runs) == {"n1": ["graphics clock low", "memory clock odd"]}


def _run(run_id, node, step_ms, finished=None):
    cfg = {"node": node}
    if finished:
        cfg["finished_at"] = finished
    return (run_id, {"config": cfg, "mixed": {"dhfr": {"plugin_step_ms": step_ms}}})


def test_gain_cell_paired_takes_the_order_effect_out():
    """Two replicas a first, two b first; b is 1% faster, and the second run of
    every pair 0.5% slower. The gain is the 1%, the order effect -0.5%."""
    a, b = [], []
    for i, b_first in enumerate((False, False, True, True)):
        t_a, t_b = ("2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z") if b_first else \
                   ("2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z")
        slow_a, slow_b = (1.005, 1.0) if b_first else (1.0, 1.005)
        a.append(_run(str(i), "n%d" % i, 10.0 * slow_a, t_a))
        b.append(_run(str(i), "n%d" % i, 10.0 / 1.01 * slow_b, t_b))
    cell = gr.gain_cell(a, b, "mixed", "dhfr", "plugin_step_ms", False, True,
                        gr.baseline_first(a, b))
    assert abs(cell["gain"] - 1.0) < 0.01
    assert abs(cell["order_effect"] + 0.5) < 0.01


def test_gain_cell_unpaired_is_against_the_centre():
    a = [_run(str(i), "a%d" % i, 10.0) for i in range(3)]
    b = [_run(str(i + 10), "b%d" % i, 9.0) for i in range(3)]
    cell = gr.gain_cell(a, b, "mixed", "dhfr", "plugin_step_ms", False, False)
    assert abs(cell["gain"] - 100.0 * (10.0 / 9.0 - 1)) < 0.01
    assert cell["order_effect"] is None
    assert gr.gain_cell(a, [], "mixed", "dhfr", "plugin_step_ms", False, False) is None
