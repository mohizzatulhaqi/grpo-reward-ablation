"""Tests for runner.py job-state detection — the logic that makes weekly resume
correct (never recompute a finished job, never skip an unfinished one)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import runner


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_pending_when_nothing_exists(tmp_path):
    res = tmp_path / "results"
    runs = tmp_path / "runs"
    res.mkdir(); runs.mkdir()
    assert runner.job_state(str(res), str(runs), "A0", 0) == "pending"


def test_needs_eval_when_adapter_checkpoint_present(tmp_path):
    res = tmp_path / "results"
    runs = tmp_path / "runs"
    res.mkdir()
    _touch(str(runs / "A0_seed0" / "adapter_config.json"))
    assert runner.job_state(str(res), str(runs), "A0", 0) == "needs_eval"


def test_needs_eval_with_full_checkpoint(tmp_path):
    res = tmp_path / "results"
    runs = tmp_path / "runs"
    res.mkdir()
    _touch(str(runs / "A1_seed2" / "model.safetensors"))
    assert runner.job_state(str(res), str(runs), "A1", 2) == "needs_eval"


def test_done_only_when_both_datasets_present(tmp_path):
    res = tmp_path / "results"
    runs = tmp_path / "runs"
    res.mkdir()
    _touch(str(runs / "A0_seed0" / "adapter_config.json"))
    _touch(str(res / "A0_seed0_gsm8k.json"))
    # only one eval done -> still needs_eval, not done
    assert runner.job_state(str(res), str(runs), "A0", 0) == "needs_eval"
    _touch(str(res / "A0_seed0_math500.json"))
    assert runner.job_state(str(res), str(runs), "A0", 0) == "done"


def test_pending_jobs_excludes_done(tmp_path):
    res = tmp_path / "results"
    runs = tmp_path / "runs"
    res.mkdir(); runs.mkdir()
    # mark A0_seed0 fully done
    _touch(str(res / "A0_seed0_gsm8k.json"))
    _touch(str(res / "A0_seed0_math500.json"))
    pend = runner.pending_jobs(str(res), str(runs))
    assert ("A0", 0, "done") not in pend
    assert all(st != "done" for _, _, st in pend)
    # grid is 5 variants x 5 seeds = 25; one done -> 24 pending
    assert len(pend) == 24


def test_job_grid_size():
    assert len(runner.job_grid()) == 25
