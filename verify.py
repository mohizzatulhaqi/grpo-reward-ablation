"""Answer extraction and verification for math-reasoning RLVR.

Pure Python (re + fractions only) so it can be imported by reward functions
*and* unit-tested without torch/transformers. This is the backbone of the
correctness reward: if extraction is sloppy, your reward signal is noise.
"""
import re
from fractions import Fraction

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_RE = re.compile(r"####\s*([^\n]+)")
_ANS_RE = re.compile(
    r"(?:final answer|answer)\s*(?:is|:)?\s*\$?(-?[\d,]+(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def _clean_num(s):
    s = str(s).strip().strip("$").replace(",", "").replace(" ", "")
    return s.rstrip(".")


def to_number(s):
    """Parse a string into a float, supporting simple fractions. None on failure."""
    if s is None:
        return None
    s = _clean_num(s)
    if s == "":
        return None
    try:
        if "/" in s:
            return float(Fraction(s))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text):
    """Extract a model's final numeric answer from generated text.

    Priority: \\boxed{} > '#### N' > 'answer is N' > last number in text.
    Returns a float, or None if nothing parseable is found.
    """
    if not text:
        return None
    for rx in (_BOXED_RE, _HASH_RE, _ANS_RE):
        m = list(rx.finditer(text))
        if m:
            v = to_number(m[-1].group(1))
            if v is not None:
                return v
    m = list(_NUM_RE.finditer(text))
    if m:
        return to_number(m[-1].group(0))
    return None


def answers_match(pred, gold, tol=1e-4):
    """True if pred and gold are numerically equal within a relative tolerance."""
    p = pred if isinstance(pred, (int, float)) else to_number(pred)
    g = gold if isinstance(gold, (int, float)) else to_number(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def extract_gold_answer(gsm8k_answer):
    """GSM8K gold answers end with '#### <number>'. Returns the float."""
    m = _HASH_RE.search(gsm8k_answer or "")
    return to_number(m.group(1)) if m else None


def extract_numbers(text):
    """Every number in a text, as a list of floats (for partial-credit overlap)."""
    if not text:
        return []
    out = []
    for mt in _NUM_RE.finditer(text):
        v = to_number(mt.group(0))
        if v is not None:
            out.append(v)
    return out
