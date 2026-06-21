"""Resume-friendly driver for the ablation grid — built for Kaggle's weekly quota.

A full sweep (5 variants x 5 seeds x 2 eval sets) does not fit in one 9-hour
session. This driver checkpoints at the *job* level: each Kaggle session you run

    python runner.py run --minutes 500

and it works through the pending jobs, persists each result, and stops before the
session limit. Next session, it picks up exactly where it left off — nothing is
recomputed.

A (variant, seed) job has three states:
  pending   -> no checkpoint yet              -> train, then eval both datasets
  needs_eval-> checkpoint exists, evals don't -> skip training, just eval
  done      -> both eval JSONs exist          -> skip

The state functions are pure filesystem checks, so they're unit-tested in
tests/test_runner.py. `status` and `run` are the CLI entry points.
"""
import argparse
import os
import subprocess
import sys
import time

from config import VARIANTS, SEEDS

DATASETS = ["gsm8k", "math500"]
_CKPT_MARKERS = ("config.json", "adapter_config.json",
                 "model.safetensors", "adapter_model.safetensors")


def job_grid(variants=None, seeds=None):
    variants = variants or list(VARIANTS)
    seeds = seeds or SEEDS
    return [(v, s) for v in variants for s in seeds]


def result_path(results_dir, v, s, dataset):
    return os.path.join(results_dir, f"{v}_seed{s}_{dataset}.json")


def _has_checkpoint(runs_dir, v, s):
    ckpt = os.path.join(runs_dir, f"{v}_seed{s}")
    return os.path.isdir(ckpt) and any(
        os.path.exists(os.path.join(ckpt, m)) for m in _CKPT_MARKERS
    )


def job_state(results_dir, runs_dir, v, s):
    """Return 'done' | 'needs_eval' | 'pending' for one (variant, seed)."""
    if all(os.path.exists(result_path(results_dir, v, s, d)) for d in DATASETS):
        return "done"
    if _has_checkpoint(runs_dir, v, s):
        return "needs_eval"
    return "pending"


def pending_jobs(results_dir, runs_dir, variants=None, seeds=None):
    """List of (v, s, state) for every job that is not yet 'done'."""
    out = []
    for v, s in job_grid(variants, seeds):
        st = job_state(results_dir, runs_dir, v, s)
        if st != "done":
            out.append((v, s, st))
    return out


# ----------------------------- CLI -----------------------------

def _train(args, v, s):
    cmd = [sys.executable, "train_grpo.py", "--variant", v, "--seed", str(s),
           "--model", args.model, "--max-steps", str(args.max_steps),
           "--num-generations", str(args.num_generations)]
    if args.use_qlora:
        cmd.append("--use-qlora")
    print(f"\n>>> TRAIN {v} seed {s}\n    {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _eval(args, v, s, dataset):
    ckpt = os.path.join(args.runs, f"{v}_seed{s}")
    cmd = [sys.executable, "eval_grpo.py", "--model", ckpt, "--variant", v,
           "--seed", str(s), "--dataset", dataset,
           "--n-samples", str(args.n_samples), "--out-dir", args.results]
    if args.eval_limit is not None:
        cmd += ["--limit", str(args.eval_limit)]
    print(f"\n>>> EVAL  {v} seed {s} ({dataset})")
    subprocess.run(cmd, check=True)


def cmd_status(args):
    grid = job_grid()
    states = {(v, s): job_state(args.results, args.runs, v, s) for v, s in grid}
    done = sum(1 for st in states.values() if st == "done")
    print(f"Progress: {done}/{len(grid)} jobs done\n")
    header = "      " + "  ".join(f"seed{s}" for s in SEEDS)
    print(header)
    symbol = {"done": "  ok ", "needs_eval": " ev? ", "pending": "  .. "}
    for v in VARIANTS:
        row = "  ".join(symbol[states[(v, s)]] for s in SEEDS)
        print(f"{v:>4}  {row}")
    print("\nlegend: ok=done  ev?=trained, needs eval  ..=pending")


def cmd_run(args):
    deadline = time.time() + args.minutes * 60
    ran = 0
    while True:
        pend = pending_jobs(args.results, args.runs)
        if not pend:
            print("\nAll jobs complete. Run: python -m analysis.aggregate ...")
            break
        if time.time() >= deadline:
            print(f"\nTime budget reached after {ran} job(s). "
                  f"{len(pend)} remaining — re-run next session to continue.")
            break
        v, s, st = pend[0]
        if st == "pending":
            _train(args, v, s)
        for dataset in DATASETS:
            if not os.path.exists(result_path(args.results, v, s, dataset)):
                _eval(args, v, s, dataset)
        ran += 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", default="results")
    common.add_argument("--runs", default="runs")

    ps = sub.add_parser("status", parents=[common], help="show grid completion")
    ps.set_defaults(func=cmd_status)

    pr = sub.add_parser("run", parents=[common],
                        help="run pending jobs within a time budget")
    pr.add_argument("--minutes", type=int, default=500,
                    help="stop before this many minutes (keep < session limit)")
    pr.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    pr.add_argument("--use-qlora", action=argparse.BooleanOptionalAction, default=True,
                    help="QLoRA on by default (required for 1.5B on a 16GB T4)")
    pr.add_argument("--max-steps", type=int, default=500)
    pr.add_argument("--num-generations", type=int, default=8)
    pr.add_argument("--n-samples", type=int, default=8)
    pr.add_argument("--eval-limit", type=int, default=None,
                    help="subsample eval set for speed (None = full)")
    pr.set_defaults(func=cmd_run)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
