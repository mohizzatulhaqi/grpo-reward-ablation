"""Prepare GSM8K into the format TRL's GRPOTrainer expects.

Output columns:
  - prompt        : conversational [{system}, {user}]  (TRL builds completions from this)
  - answer        : float gold answer            (consumed by correctness/length_gated reward)
  - gold_numbers  : list[float] intermediate nums (consumed by partial reward)

Run on the machine with internet + datasets installed:
    python prepare_data.py --split train --out data/gsm8k_train
    python prepare_data.py --split test  --out data/gsm8k_test
"""
import argparse

from datasets import load_dataset, Dataset

from verify import extract_gold_answer, extract_numbers
from config import SYSTEM_PROMPT


def build(split):
    raw = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for ex in raw:
        gold = extract_gold_answer(ex["answer"])
        if gold is None:
            continue
        # intermediate numbers = numbers in the worked solution, minus the final line
        solution_body = ex["answer"].split("####")[0]
        nums = extract_numbers(solution_body)
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer": gold,
            "gold_numbers": nums,
        })
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default="data/gsm8k_train")
    args = ap.parse_args()
    ds = build(args.split)
    ds.save_to_disk(args.out)
    print(f"Saved {len(ds)} examples -> {args.out}")
