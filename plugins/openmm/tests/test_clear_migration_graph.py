# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clear/migration overlap is active only inside multi-step graphs."""
import numpy as np
import pytest
from test_whole_step_graph import trajectory
from test_whole_step_batch import reports


@pytest.mark.parametrize('precision', ['single', 'mixed'])
@pytest.mark.parametrize('case', ['prolate', 'ewald', 'triclinic'])
@pytest.mark.parametrize('mode', ['eager', 'component', 'whole', 'batch'])
@pytest.mark.parametrize('setting', ['default', 'batched'])
def test_clear_preserves_states(monkeypatch, capfd, precision, case, mode, setting):
    monkeypatch.setenv('CUPO_COMPACT_AUTOTUNE', '0')
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH', '25' if mode == 'batch' else '1')
    monkeypatch.setenv('CUPO_PLUGIN_WHOLE_STEP_BATCH_REPORT', '1')
    args = dict(graph=mode != 'eager', mutate=True)
    whole = mode in ('whole', 'batch')
    monkeypatch.setenv('CUPO_PLUGIN_CLEAR_PARALLEL', 'off')
    expected, ec = trajectory(case, precision, whole, **args)
    baseline_logs = reports(capfd)
    assert all(r['clear_migration_overlaps'] == r['clear_grid_resizes'] == 0 for r in baseline_logs)
    if setting == 'default':
        monkeypatch.delenv('CUPO_PLUGIN_CLEAR_PARALLEL')
    else:
        monkeypatch.setenv('CUPO_PLUGIN_CLEAR_PARALLEL', setting)
    actual, ac = trajectory(case, precision, whole, **args)
    logs = reports(capfd)
    assert ec == ac
    for reference, observed in zip(expected, actual):
        for x, y in zip(reference, observed):
            np.testing.assert_array_equal(x, y)
    if mode == 'batch':
        assert any(r['clear_migration_overlaps'] > 0 and r['clear_grid_resizes'] > 0 for r in logs)
        assert all(r['clear_migration_overlaps'] <= r['batched_steps'] for r in logs)
    else:
        assert all(r['clear_migration_overlaps'] == r['clear_grid_resizes'] == 0 for r in logs)
