  # GRPO Reward-Design Ablation for Small-Model Math Reasoning

  An empirical study isolating **which reward components actually help** when
  post-training a small LLM with GRPO on math reasoning — and which are redundant
  or invite reward hacking. The contribution is not a new algorithm; it's a
  *rigorous, multi-seed, statistically-tested* decomposition of the reward, which
  the RLVR literature largely lacks.

  > Status: core logic (rewards, verifier, pass@k, statistics, aggregation) is
  > unit-tested and runs on CPU. Training/eval scripts require a CUDA box.

  ---

  ## The experiment

  Cumulative ablation ladder — each variant adds one reward component:

  | variant | reward components | studies |
  |---|---|---|
  | **A0** | correctness only | pure RLVR baseline (everything is measured against this) |
  | **A1** | + format | does enforcing structure help, or just change surface form? |
  | **A2** | + length control | brevity vs the short-wrong-answer failure mode |
  | **A3** | + partial credit | does a process/credit-assignment signal help a *small* model? |
  | **A4** | full, tuned weights | best achievable combination |

  Plus two sub-experiments when you have compute:
  - **Leave-one-out on A4** — drop one component to measure its marginal contribution.
  - **`length` vs `length_gated`** — unconditional vs correctness-gated length reward.

  **Models:** `Qwen2.5-1.5B-Instruct` (primary; deliberately *not* the Math variant,
  so there's headroom for the reward to move accuracy). Prototype on `Qwen2.5-0.5B`.
  **Train:** GSM8K (~7.5k). **Eval:** GSM8K test (1319) + MATH-500 (out-of-distribution).
  **Algorithm:** GRPO (no value network → light) via TRL.

  ---

  ## Setup

  ```bash
  pip install -r requirements.txt
  pytest -q                      # verify the CPU core (26 tests)
  ```

  ## Run order

  ```bash
  # 1. Data (needs internet)
  python prepare_data.py --split train --out data/gsm8k_train

  # 2. Validate the pipeline cheaply BEFORE the full sweep
  python train_grpo.py --variant A0 --seed 0 --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 50
  #    -> confirm reward curve climbs and nothing crashes

  # 3. Single-seed sweep (week 3-6): pick reward weights
  SEEDS=(0) bash run_ablation.sh     # edit the SEEDS line in the script

  # 4. Full multi-seed sweep (week 7-10)
  bash run_ablation.sh

  # 5. Aggregate -> comparison table
  python -m analysis.aggregate --results results --metric "pass@1" --baseline A0 --dataset gsm8k
  ```

  ## Free-tier mode (Kaggle, $0)

  The full scope (1.5B, A0–A4, 5 seeds, GSM8K + MATH-500) runs on a **free 16 GB T4**
  via **QLoRA** (4-bit base + LoRA adapter), spread across Kaggle's ~30 GPU-h/week
  quota. Two pieces make this work:

  - `--use-qlora` on `train_grpo.py` — fits 1.5B in 16 GB. `eval_grpo.py` auto-detects
    adapter checkpoints (loads base + adapter, merges for inference).
  - `runner.py` — a resume-friendly driver. Each session it runs pending jobs within a
    time budget and stops before the session cap; next session it continues. Job state
    is tracked on disk (`pending` → `needs_eval` → `done`), so nothing is recomputed.

  ```bash
  python runner.py status                       # show the 5x5 completion grid
  python runner.py run --minutes 480            # work through pending jobs (QLoRA on by default)
  ```

  Use `kaggle_run.ipynb` for the ready-made notebook (handles clone, install, cross-session
  result persistence via a Kaggle Dataset, and the weekly loop). QLoRA is a valid choice
  for a reward ablation as long as it's **identical across all variants** — declare the
  LoRA config in your writeup and note full-FT replication as future work. Trade-off: T4
  is slow, so the full sweep is ~5–8 weeks part-time; measure one run first to plan.

  ---

  ## Metrics

  - **pass@1** — primary, averaged `c/n` over questions (low variance).
  - **pass@n** — headroom metric. If A-variants lift pass@1 but not pass@n, that's
    evidence RLVR is *sharpening the distribution* rather than expanding capability
    — the open debate this design lets you probe.
  - **mean_len, format_rate, training stability** — behavioural signals.
  - **reward-hacking diagnostic** — TRL logs each reward component separately
    (`train/rewards/<fn>`); plot each component's reward vs held-out accuracy. A
    component whose reward rises while accuracy stalls is being gamed.
  - **significance** — every variant compared to A0 via paired **Wilcoxon** (same
    seeds), with **Cliff's delta** effect size. See `analysis/stats.py`.

  ---

  ## Design notes & known pitfalls (read before tuning)

  - **Seeds and p-values.** With 5 paired seeds the smallest possible Wilcoxon
    p-value is **0.0625** — so you literally cannot reach p < 0.05 with 5 seeds.
    Use **≥6 seeds** if you need that threshold; otherwise lead with effect sizes
    (Cliff's delta) and report p alongside. Don't over-claim significance.
  - **Length reward is a trap.** Applied unconditionally it can reward short
    *wrong* answers. `length_gated` only rewards length on correct answers —
    comparing the two is itself a finding. Watch mean_len **and** accuracy together.
  - **Keep partial credit simple.** The overlap-based signal is intentionally
    crude. Only invest in a smarter process reward if A3 shows the simple one
    already helps.
  - **Reward scales.** All components are ~[0,1] so `weights` are clean relative
    importances. If you change a component's range, rebalance weights.
  - **Model choice matters.** A math-tuned base can saturate GSM8K and hide reward
    effects; that's why the primary model is the general Instruct model, with
    MATH-500 for extra headroom.
  - **Small-model GRPO instability.** `beta` (KL) and `lr` are the knobs that
    matter most; an over-weighted format reward can cause collapse. Validate at
    50 steps on 0.5B before committing GPU.

  ---

  ## Compute

  One GRPO run (1.5B / GSM8K, ~500 steps) ≈ 6–18 h on a single A100.
  Full grid: 5 variants × 5 seeds = 25 runs (+ prototyping) ≈ ~400 GPU-hours.
  At ~$1.5–2/h on rented A100 (cheaper on spot) ≈ **$300–600**. Reduce with 3
  seeds, 0.5B for parts of the sweep, or shorter training.

  ---

  ## Layout

  ```
  verify.py            answer extraction + matching            [tested]
  rewards.py           4 reward components + TRL builder        [tested]
  config.py            A0-A4 variant definitions + prompt
  prepare_data.py      GSM8K -> TRL format                      [GPU box / internet]
  train_grpo.py        GRPO training (+ --use-qlora)            [GPU]
  eval_grpo.py         sampling eval -> results JSON            [GPU]
  runner.py            resume-friendly sweep driver             [tested]
  run_ablation.sh      full sweep orchestration
  kaggle_run.ipynb     ready-made Kaggle free-tier notebook
  analysis/
    passk.py           unbiased pass@k                          [tested]
    stats.py           Wilcoxon / Mann-Whitney / Cliff's delta  [tested]
    aggregate.py       results JSONs -> markdown table          [tested]
  tests/
    test_core.py       26 tests for the CPU core
    test_runner.py     6 tests for resume / job-state logic
  ```

  ## Pointers

  GRPO/DeepSeek-R1 (RLVR), TRL `GRPOTrainer` docs, and the pass@k estimator
  (Chen et al., 2021). When you write up, the related work to position against is
  the credit-assignment / process-reward line and the pass@1-vs-pass@n debate.
