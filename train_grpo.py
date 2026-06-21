"""GRPO training for one (variant, seed). Runs on a CUDA box.

Example:
    python train_grpo.py --variant A3 --seed 0

The reward function list + weights come straight from config.VARIANTS, so the
ablation is fully specified by --variant. TRL logs each reward component
separately under train/rewards/<fn_name> — keep those logs, they are your
reward-hacking diagnostics.

Free-tier mode: pass --use-qlora to fit Qwen2.5-1.5B on a 16 GB T4 (Kaggle/Colab).
    python train_grpo.py --variant A3 --seed 0 --use-qlora

API note: written against trl>=0.12 (GRPOConfig/GRPOTrainer, `beta` = KL coeff,
`processing_class` for the tokenizer). If your TRL version differs, adjust the
config kwargs — the reward/data plumbing stays the same.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 4-bit bitsandbytes breaks under
# multi-GPU DataParallel (e.g. Kaggle's T4 x2). Pin to one GPU unless explicitly
# overridden (set CUDA_VISIBLE_DEVICES yourself + use accelerate launch for real DDP).

import argparse
import dataclasses
import random

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rewards import build_reward_funcs
from config import VARIANTS


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _supported_kwargs(config_cls, kwargs):
    """Drop kwargs the installed TRL/GRPOConfig version doesn't accept.

    TRL's config API drifts between releases (e.g. max_prompt_length was removed).
    Rather than pin a version, we keep only fields the installed GRPOConfig
    actually declares (inherited TrainingArguments fields included) and warn about
    the rest. Keeps this script runnable across TRL versions on Kaggle/Colab.
    """
    valid = {f.name for f in dataclasses.fields(config_cls)}
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"[warn] this TRL's GRPOConfig ignores unsupported args: {dropped}")
    return {k: v for k, v in kwargs.items() if k in valid}


def build_policy(args):
    """Load tokenizer + model. Returns (model, tokenizer, peft_config).

    --use-qlora: 4-bit NF4 base + a LoRA adapter. peft_config is handed to
    GRPOTrainer, which wraps the model and disables the adapter to get the KL
    reference logprobs — so there is NO second full-size model copy. That, plus
    4-bit weights and adapter-only optimizer state, is what lets 1.5B fit in
    16 GB. Without the flag: ordinary full-precision (bf16) fine-tuning.

    Keep QLoRA settings identical across all variants — the ablation compares the
    *relative* contribution of reward components, which stays valid as long as the
    training setup is held constant.
    """
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    peft_config = None
    if args.use_qlora:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, torch_dtype=torch.bfloat16,
        )
        model.config.use_cache = False
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        print(f"[qlora] 4-bit NF4 + LoRA r={args.lora_rank} alpha={args.lora_alpha}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16,
        )
    return model, tok, peft_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/gsm8k_train")
    ap.add_argument("--output", default=None)
    # GRPO knobs — the ones that actually matter for small-model stability
    ap.add_argument("--num-generations", type=int, default=8)   # group size G
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)         # KL-to-reference coeff
    ap.add_argument("--max-completion-length", type=int, default=640)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--per-device-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    # Free-tier / memory options
    ap.add_argument("--use-qlora", action="store_true",
                    help="4-bit NF4 base + LoRA adapter; fits 1.5B on a 16GB T4")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                    default=True, help="trade compute for memory (on by default)")
    args = ap.parse_args()

    set_seed(args.seed)
    out = args.output or f"runs/{args.variant}_seed{args.seed}"
    cfg = VARIANTS[args.variant]
    reward_funcs, reward_weights = build_reward_funcs(cfg)
    print(f"[{args.variant}] rewards={cfg['rewards']} weights={reward_weights}")

    ds = load_from_disk(args.data)

    model, tok, peft_config = build_policy(args)

    cfg_kwargs = dict(
        output_dir=out,
        seed=args.seed,
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=args.max_steps,
        reward_weights=reward_weights,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        report_to="none",
    )
    grpo_cfg = GRPOConfig(**_supported_kwargs(GRPOConfig, cfg_kwargs))

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_funcs,
        args=grpo_cfg,
        train_dataset=ds,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(out)
    print(f"Saved checkpoint -> {out}")


if __name__ == "__main__":
    main()