#!/usr/bin/env bash
# Full ablation sweep: A0-A4 x seeds 0-4, eval on GSM8K + MATH-500, then aggregate.
# This is the heavy compute block (week 7-10). Edit VARIANTS/SEEDS to subset.
#
# Tip: for the week 3-6 single-seed sweep, run with SEEDS=(0) first to validate
# every variant trains and to pick reward weights before spending GPU on 5 seeds.
set -euo pipefail

VARIANTS=(A0 A1 A2 A3 A4)
SEEDS=(0 1 2 3 4)

for v in "${VARIANTS[@]}"; do
  for s in "${SEEDS[@]}"; do
    echo "================ TRAIN $v seed $s ================"
    python train_grpo.py --variant "$v" --seed "$s"

    echo "================ EVAL  $v seed $s (gsm8k) ========"
    python eval_grpo.py --model "runs/${v}_seed${s}" --variant "$v" --seed "$s" --dataset gsm8k

    echo "================ EVAL  $v seed $s (math500) ======"
    python eval_grpo.py --model "runs/${v}_seed${s}" --variant "$v" --seed "$s" --dataset math500
  done
done

echo "================ AGGREGATE ================"
python -m analysis.aggregate --results results --metric "pass@1" --baseline A0 --dataset gsm8k
python -m analysis.aggregate --results results --metric "pass@1" --baseline A0 --dataset math500
