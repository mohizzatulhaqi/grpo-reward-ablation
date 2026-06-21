"""Evaluate a trained checkpoint and write one results JSON.

Samples n completions per question (temperature > 0) and reports:
  - pass@1     : low-variance, averaged c/n over questions
  - pass@<n>   : headroom metric (the "expand vs sharpen" question)
  - mean_len   : mean completion length in characters
  - format_rate: fraction of completions that hit the structural format
  - n_questions

Example:
    python eval_grpo.py --model runs/A3_seed0 --variant A3 --seed 0 --dataset gsm8k

For speed at scale, swap the HF generate loop for vLLM — the scoring logic
(verify.extract_answer / answers_match) is identical.
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # pin to one GPU (matches training)

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from verify import extract_answer, answers_match, extract_gold_answer
from analysis.passk import pass_at_k
from rewards import format_reward
from config import SYSTEM_PROMPT


def load_eval_set(name, limit=None):
    """Return list of {'question','gold'} for gsm8k test or MATH-500."""
    if name == "gsm8k":
        raw = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"question": e["question"], "gold": extract_gold_answer(e["answer"])}
                 for e in raw]
    elif name == "math500":
        raw = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": e["problem"], "gold": e["answer"]} for e in raw]
    else:
        raise ValueError(f"unknown dataset {name}")
    items = [it for it in items if it["gold"] is not None]
    return items[:limit] if limit else items


def load_model_and_tok(path):
    """Load a checkpoint for inference, handling both full and LoRA-adapter dirs.

    A QLoRA run saves only the adapter (adapter_config.json + adapter weights).
    We detect that, load the base model in bf16, attach the adapter, and merge it
    for fast clean generation. Full-FT checkpoints load directly.
    """
    adapter_cfg = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        import json
        from peft import PeftModel
        with open(adapter_cfg) as f:
            base = json.load(f)["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)            # checkpoint dir or base model
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    model, tok = load_model_and_tok(args.model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    items = load_eval_set(args.dataset, args.limit)
    n = args.n_samples

    p1_sum, pk_sum, len_sum, fmt_hits, gen_count = 0.0, 0.0, 0, 0, 0

    for it in items:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": it["question"]},
        ]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok([text] * n, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.95,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        completions = tok.batch_decode(
            out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )

        c = sum(1 for comp in completions
                if answers_match(extract_answer(comp), it["gold"]))
        p1_sum += c / n                       # unbiased pass@1
        pk_sum += pass_at_k(n, c, n)          # pass@n (headroom)
        len_sum += sum(len(x) for x in completions)
        fmt_hits += sum(1 for s in format_reward(completions) if s >= 1.0)
        gen_count += n

    n_q = len(items)
    record = {
        "variant": args.variant,
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model,
        "n_samples": n,
        "pass@1": round(p1_sum / n_q, 4),
        f"pass@{n}": round(pk_sum / n_q, 4),
        "mean_len": round(len_sum / gen_count, 1),
        "format_rate": round(fmt_hits / gen_count, 3),
        "n_questions": n_q,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir,
                        f"{args.variant}_seed{args.seed}_{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()