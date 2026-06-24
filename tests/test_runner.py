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


# ---------------- command builders ----------------

def _fake_args(**over):
    import types
    base = dict(model="Qwen/Qwen2.5-1.5B-Instruct", max_steps=250, num_generations=6,
                use_qlora=True, use_vllm=False, vllm_gpu_mem=0.25,
                runs="runs", results="results", n_samples=4, eval_limit=300)
    base.update(over)
    return types.SimpleNamespace(**base)


def test_train_cmd_includes_qlora_and_core_flags():
    cmd = runner._train_cmd(_fake_args(), "A3", 1)
    assert "--use-qlora" in cmd
    assert "--variant" in cmd and "A3" in cmd
    assert "--seed" in cmd and "1" in cmd
    assert "--use-vllm" not in cmd  # off by default in this fake


def test_train_cmd_adds_vllm_when_enabled():
    cmd = runner._train_cmd(_fake_args(use_vllm=True), "A0", 0)
    assert "--use-vllm" in cmd
    assert "--vllm-gpu-mem" in cmd and "0.25" in cmd


def test_eval_cmd_passes_limit_and_samples():
    cmd = runner._eval_cmd(_fake_args(), "A2", 4, "math500")
    assert "--dataset" in cmd and "math500" in cmd
    assert "--limit" in cmd and "300" in cmd
    assert "--n-samples" in cmd and "4" in cmd
    assert os.path.join("runs", "A2_seed4") in cmd  # checkpoint path


def test_eval_cmd_omits_limit_when_none():
    cmd = runner._eval_cmd(_fake_args(eval_limit=None), "A0", 0, "gsm8k")
    assert "--limit" not in cmd
