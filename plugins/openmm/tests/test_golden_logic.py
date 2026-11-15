# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test that the golden regression logic actually catches regressions.

Every test here drives _get_golden_entry the way the benchmarks do and asks what a
regression check would then see. The failures they guard against are the ones that
made every benchmark on every GPU unfailable: comparing a run against a file the
run itself had just written.
"""

import json

import pytest

import test_bench_plugin as tbp

GOOD = {
    "plugin_step_ms": 1.000,
    "plugin_force_and_energy_ms": 0.800,
    "plugin_ns_per_day": 100.0,
    "stock_step_ms": 2.000,
    "stock_force_and_energy_ms": 1.600,
    "stock_ns_per_day": 50.0,
    "speedup_step": 2.0,
    "speedup_force_and_energy": 2.0,
    # Energy-only: smaller than force+energy, which carries OpenMM's per-atom
    # round trip on top of the same evaluation.
    "plugin_energy_ms": 0.400,
    "stock_energy_ms": 0.800,
    "speedup_energy": 2.0,
}

# The plugin is 30% slower; stock is untouched, so the speedups fall with it.
REGRESSED = {
    "plugin_step_ms": 1.300,
    "plugin_force_and_energy_ms": 1.040,
    "plugin_ns_per_day": 76.92,
    "stock_step_ms": 2.000,
    "stock_force_and_energy_ms": 1.600,
    "stock_ns_per_day": 50.0,
    "speedup_step": 2.000 / 1.300,
    "speedup_force_and_energy": 1.600 / 1.040,
    "plugin_energy_ms": 0.520,
    "stock_energy_ms": 0.800,
    "speedup_energy": 0.800 / 0.520,
}


@pytest.fixture
def golden(tmp_path, monkeypatch):
    """A committed golden holding GOOD, and a knob for where a run writes.

    The golden is a directory of runs, so the committed one here is a single run
    directory. Returns (ref_path, set_output): ref_path is that run's file, and
    set_output(path) sends the run's own measurements somewhere else, which is what
    GOLDEN_OUTPUT_DIR does in CI.
    """
    gpu_dir = tmp_path / "ref" / "gpu"
    ref_path = gpu_dir / "committed" / "nvt.json"
    ref_path.parent.mkdir(parents=True)
    ref_path.write_text(json.dumps({"mixed": {"water_100k": GOOD}}) + "\n")

    monkeypatch.setattr(tbp, "_golden_gpu_dir", lambda group=None: gpu_dir)
    monkeypatch.setattr(tbp, "_ensemble", lambda: "nvt")
    monkeypatch.setattr(tbp, "_golden_out_path", lambda group=None: ref_path)
    monkeypatch.setattr(tbp, "_reference", {})
    monkeypatch.setattr(tbp, "_reference_runs", {})

    def set_output(path):
        monkeypatch.setattr(tbp, "_golden_out_path", lambda group=None: path)

    return ref_path, set_output


def _regression_frac(entry, key, measured):
    """What _check_regression would compute for one higher-is-worse metric."""
    return (tbp._sig(measured[key]) - entry[key]) / entry[key]


def test_regression_detected(golden):
    """A 30% regression is a 30% regression, and the check must see it."""
    _ref_path, _set_output = golden

    entry = tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")

    frac = _regression_frac(entry, "plugin_step_ms", REGRESSED)
    assert frac > tbp._FAIL_THRESHOLD, (
        "30%% regression not detected (regression_frac=%.4f): _get_golden_entry "
        "returned the measurement instead of the committed golden." % frac)


def test_regression_detected_with_separate_output_dir(golden, tmp_path):
    """The CI shape: measurements go to a scratch directory, and the comparison
    still has to be against the committed golden.

    This is the failure that made every benchmark on every GPU unfailable --
    GOLDEN_OUTPUT_DIR moved the read as well as the write, so each run compared
    itself against the empty file it had just started writing.
    """
    ref_path, set_output = golden
    out_path = tmp_path / "out" / "gpu" / "nvt.json"
    set_output(out_path)

    entry = tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")

    frac = _regression_frac(entry, "plugin_step_ms", REGRESSED)
    assert frac > tbp._FAIL_THRESHOLD, (
        "30%% regression not detected with GOLDEN_OUTPUT_DIR set "
        "(regression_frac=%.4f): the run read its own output." % frac)

    # The committed golden is a reference, so a run must not touch it...
    assert json.loads(ref_path.read_text())["mixed"]["water_100k"] == GOOD
    # ...and the scratch file holds what this run measured, for CI to collect.
    written = json.loads(out_path.read_text())["mixed"]["water_100k"]
    assert written["plugin_step_ms"] == tbp._sig(REGRESSED["plugin_step_ms"])


def test_every_metric_is_checked_not_just_the_first(golden, tmp_path):
    """All six benchmark tests share one measurement and each asks for the entry.

    The reference has to be the same on the sixth call as on the first. When it was
    re-read from disk, the first call's write became the second call's golden and
    five of the six metrics compared a number against itself.
    """
    _ref_path, set_output = golden
    set_output(tmp_path / "out" / "gpu" / "nvt.json")

    entries = [tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")
               for _ in range(6)]

    assert all(e == entries[0] for e in entries), (
        "the golden entry moved between calls within one run: %r" % (entries,))
    for key in ("plugin_step_ms", "plugin_force_and_energy_ms"):
        frac = _regression_frac(entries[-1], key, REGRESSED)
        assert frac > tbp._FAIL_THRESHOLD, (
            "%s: the last of six calls no longer sees the regression "
            "(regression_frac=%.4f)" % (key, frac))


def test_speedup_regression_detected(golden, tmp_path):
    """speedup_step is lower-is-worse, and it fell from 2.0x to 1.54x."""
    _ref_path, set_output = golden
    set_output(tmp_path / "out" / "gpu" / "nvt.json")

    entry = tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")

    golden_val = entry["speedup_step"]
    measured = tbp._sig(REGRESSED["speedup_step"])
    frac = (golden_val - measured) / golden_val
    assert frac > tbp._FAIL_THRESHOLD, (
        "speedup regression not detected (regression_frac=%.4f)" % frac)


def test_new_system_has_nothing_to_compare_against(golden, tmp_path):
    """A system the committed golden has never seen records and passes."""
    _ref_path, set_output = golden
    out_path = tmp_path / "out" / "gpu" / "nvt.json"
    set_output(out_path)

    entry = tbp._get_golden_entry(REGRESSED, "brand_new_system", "mixed")

    assert entry["plugin_step_ms"] == tbp._sig(REGRESSED["plugin_step_ms"])
    assert _regression_frac(entry, "plugin_step_ms", REGRESSED) == 0.0
    assert "brand_new_system" in json.loads(out_path.read_text())["mixed"]


def test_developer_golden_is_rewritten_on_drift(golden):
    """With no GOLDEN_OUTPUT_DIR the output IS the committed golden, and a run that
    has moved rewrites it -- after returning the old numbers to be checked."""
    ref_path, _set_output = golden

    entry = tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")

    assert entry["plugin_step_ms"] == GOOD["plugin_step_ms"]
    on_disk = json.loads(ref_path.read_text())["mixed"]["water_100k"]
    assert on_disk["plugin_step_ms"] == tbp._sig(REGRESSED["plugin_step_ms"])


def test_developer_golden_records_even_without_drift(golden, tmp_path):
    """A run records what it measured whether or not the number moved.

    It used to write only past an update threshold, so a golden could hold this
    run's measurement for one system and a months-old one for the next, with
    nothing in the file saying which. A small move is still this run's number.
    """
    _ref_path, set_output = golden
    out_path = tmp_path / "out" / "gpu" / "nvt.json"
    set_output(out_path)

    nudged = dict(GOOD, plugin_step_ms=GOOD["plugin_step_ms"] * 1.01)  # under any warn
    entry = tbp._get_golden_entry(nudged, "water_100k", "mixed")

    # Compared against the committed golden...
    assert entry["plugin_step_ms"] == GOOD["plugin_step_ms"]
    # ...and the 1% move is recorded anyway.
    assert json.loads(out_path.read_text())["mixed"]["water_100k"]["plugin_step_ms"] == \
        tbp._sig(nudged["plugin_step_ms"])


def test_nvt_and_npt_do_not_share_a_reference(tmp_path, monkeypatch):
    """One session benches NVT and then NPT, and the ensemble picks the file.

    BAROSTAT is a module global the ensemble fixture flips, so the ensemble names
    which file inside each run directory is folded -- nvt.json for the first half of
    a run and npt.json for the second. A reference cached as a single dict would
    measure the NPT numbers against the NVT golden.
    """
    ensemble = ["nvt"]
    run = tmp_path / "gpu" / "committed"
    run.mkdir(parents=True)
    (run / "nvt.json").write_text(json.dumps({"mixed": {"water_100k": GOOD}}) + "\n")
    # The NPT golden already says what this run measures, so NPT must come out clean.
    npt_golden = {k: tbp._sig(v) for k, v in REGRESSED.items()}
    (run / "npt.json").write_text(
        json.dumps({"mixed": {"water_100k": npt_golden}}) + "\n")

    monkeypatch.setattr(tbp, "_golden_gpu_dir", lambda group=None: tmp_path / "gpu")
    monkeypatch.setattr(tbp, "_ensemble", lambda: ensemble[0])
    monkeypatch.setattr(tbp, "_golden_out_path",
                        lambda group=None: tmp_path / "out" / ("%s.json" % ensemble[0]))
    monkeypatch.setattr(tbp, "_reference", {})
    monkeypatch.setattr(tbp, "_reference_runs", {})

    nvt_entry = tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")
    ensemble[0] = "npt"
    npt_entry = tbp._get_golden_entry(REGRESSED, "water_100k", "mixed")

    assert _regression_frac(nvt_entry, "plugin_step_ms", REGRESSED) > tbp._FAIL_THRESHOLD
    assert _regression_frac(npt_entry, "plugin_step_ms", REGRESSED) == 0.0, (
        "NPT was measured against the NVT golden: %r" % (npt_entry,))


def test_a_benchmark_with_no_golden_skips_saying_so(golden, tmp_path):
    """With nothing committed to compare against, a benchmark records what it
    measured and is skipped, "measured, not compared", rather than passing against
    its own number."""
    _ref_path, set_output = golden
    out_path = tmp_path / "out" / "gpu" / "nvt.json"
    set_output(out_path)
    with pytest.raises(pytest.skip.Exception, match="measured, not compared"):
        tbp._compared_entry(REGRESSED, "brand_new_system", "mixed")
    assert "brand_new_system" in json.loads(out_path.read_text())["mixed"]


def test_with_unlocked_clocks_a_benchmark_with_a_golden_is_not_compared(golden, tmp_path,
                                                                       monkeypatch):
    """With CUPO_ALLOW_UNLOCKED_CLOCKS=1 and clocks that could not be locked, a
    benchmark records what it measured and is skipped, "clocks not locked", even
    where a committed golden exists: an unlocked number is compared with nothing."""
    _ref_path, set_output = golden
    out_path = tmp_path / "out" / "gpu" / "nvt.json"
    set_output(out_path)
    monkeypatch.setattr(tbp.gpu_clocks, "SESSION_UNLOCKED", True)
    with pytest.raises(pytest.skip.Exception,
                       match="measured, not compared: clocks not locked "
                             r"\(CUPO_ALLOW_UNLOCKED_CLOCKS=1\)"):
        tbp._compared_entry(REGRESSED, "water_100k", "mixed")
    assert "water_100k" in json.loads(out_path.read_text())["mixed"]
