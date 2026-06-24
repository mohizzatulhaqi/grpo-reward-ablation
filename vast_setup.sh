#!/usr/bin/env bash
# One-shot setup for a fresh Vast.ai (or any CUDA) instance.
#
# Run it directly from GitHub once your repo is up to date:
#   curl -sL https://raw.githubusercontent.com/mohizzatulhaqi/grpo-reward-ablation/main/vast_setup.sh | bash
#
# ...or, if you've already cloned the repo, just: bash vast_setup.sh
set -e

REPO="https://github.com/mohizzatulhaqi/grpo-reward-ablation.git"
DIR="grpo-reward-ablation"

if [ -f "train_grpo.py" ] && [ -f "runner.py" ]; then
  echo "==> Already inside the repo."
elif [ -d "$DIR/.git" ]; then
  echo "==> Repo exists; updating."
  git -C "$DIR" pull --ff-only || true
  cd "$DIR"
else
  echo "==> Cloning repo."
  git clone "$REPO"
  cd "$DIR"
fi

echo "==> Installing requirements (a few minutes)..."
pip install -q -r requirements.txt

echo "==> Installing vLLM (fits a 24GB GPU)..."
pip install -q vllm || echo "[warn] vllm install failed; training auto-falls back to standard generation"

echo "==> Preparing GSM8K data..."
if [ -d data/gsm8k_train ]; then
  echo "    data/gsm8k_train already present"
else
  python prepare_data.py --split train --out data/gsm8k_train
fi

cat <<'EOF'

================================================================
Setup complete. NEXT — measure ONE run to get the per-step time:

  python train_grpo.py --variant A0 --seed 0 --use-qlora --use-vllm --max-steps 20

Watch the final "wall time:" line (and check whether vLLM ran or fell
back). Report that number, then launch the full sweep with runner.py:

  python runner.py run --minutes 999 --use-qlora --use-vllm \
    --max-steps 250 --num-generations 6 --eval-limit 300 --n-samples 4

When done: download results/*.json, then DESTROY the instance.
================================================================
EOF
