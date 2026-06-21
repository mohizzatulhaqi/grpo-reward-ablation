"""Reward components for the GRPO reward-design ablation.

Each component is an independent, TRL-compatible reward function with the
signature `fn(completions, **kwargs) -> list[float]`. TRL passes every
non-prompt/-completion dataset column through **kwargs (e.g. `answer`,
`gold_numbers`), aligned one-per-completion.

Keeping each component separate (rather than one fused reward) is deliberate:
TRL logs each reward function's value independently, which is exactly what you
need for the reward-hacking diagnostics (watch a component's reward rise while
held-out accuracy stays flat).

All components are scaled to roughly [0, 1] so the weights in config.py read as
clean relative importances.
"""
import re
from verify import extract_answer, answers_match, extract_numbers, to_number

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _get_text(completion):
    """Completion may be a plain string or TRL conversational [{'role','content'}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [m.get("content", "") for m in completion if isinstance(m, dict)]
        return parts[-1] if parts else ""
    return str(completion)


def correctness_reward(completions, answer, **kwargs):
    """1.0 if the extracted final answer matches the gold answer, else 0.0.
    This is the only component used in A0 (pure RLVR)."""
    texts = [_get_text(c) for c in completions]
    return [1.0 if answers_match(extract_answer(t), g) else 0.0
            for t, g in zip(texts, answer)]


def format_reward(completions, **kwargs):
    """Rewards *structure*, independent of whether the value is correct.

    0.5 for a reasoning block, +0.5 for a parseable \\boxed{} answer. Orthogonal
    to correctness by design: a model can score format=1 with a wrong number,
    which is what lets you separate "follows the format" from "gets it right".
    """
    out = []
    for c in completions:
        t = _get_text(c)
        score = 0.0
        if ("<reasoning>" in t and "</reasoning>" in t) or \
           ("<think>" in t and "</think>" in t):
            score += 0.5
        mb = _BOXED_RE.search(t)
        if mb is not None and to_number(mb.group(1)) is not None:
            score += 0.5
        out.append(score)
    return out


def length_reward(completions, target_len=512, hard_len=None, unit="char", **kwargs):
    """Soft brevity reward: 1.0 within budget, linear decay to 0 at hard_len.

    KNOWN PITFALL (study this!): applied unconditionally, a length reward can
    incentivise short *wrong* answers. A correctness-gated variant lives in
    `length_reward_gated`; comparing the two is a clean sub-experiment for A2.
    """
    if hard_len is None:
        hard_len = 2 * target_len
    out = []
    for c in completions:
        t = _get_text(c)
        L = len(t) if unit == "char" else len(t.split())
        if L <= target_len:
            out.append(1.0)
        elif L >= hard_len:
            out.append(0.0)
        else:
            out.append(1.0 - (L - target_len) / (hard_len - target_len))
    return out


def length_reward_gated(completions, answer, target_len=512, hard_len=None,
                        unit="char", **kwargs):
    """Length reward that only applies to correct answers (0 reward if wrong)."""
    if hard_len is None:
        hard_len = 2 * target_len
    texts = [_get_text(c) for c in completions]
    base = length_reward(completions, target_len=target_len, hard_len=hard_len, unit=unit)
    out = []
    for t, g, b in zip(texts, answer, base):
        out.append(b if answers_match(extract_answer(t), g) else 0.0)
    return out


def partial_credit_reward(completions, gold_numbers, **kwargs):
    """Process signal: fraction of gold intermediate numbers present in the output.

    Intentionally simple (overlap-based, not a learned PRM) — start here and
    only add complexity if A3 shows the signal is worth it. This is the
    component most aligned with the credit-assignment theme.
    """
    out = []
    for c, golds in zip(completions, gold_numbers):
        t = _get_text(c)
        if not golds:
            out.append(0.0)
            continue
        pred = extract_numbers(t)
        hit = sum(
            1 for g in set(golds)
            if any(abs(g - p) <= 1e-4 * max(1.0, abs(g)) for p in pred)
        )
        out.append(hit / len(set(golds)))
    return out


REWARD_FUNCS = {
    "correctness": correctness_reward,
    "format": format_reward,
    "length": length_reward,
    "length_gated": length_reward_gated,
    "partial": partial_credit_reward,
}


def build_reward_funcs(variant_cfg):
    """Return (funcs, weights) aligned lists for GRPOConfig(reward_weights=...)."""
    names = variant_cfg["rewards"]
    funcs = [REWARD_FUNCS[n] for n in names]
    weights = [variant_cfg["weights"][n] for n in names]
    return funcs, weights
