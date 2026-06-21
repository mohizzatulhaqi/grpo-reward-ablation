"""Experiment configuration: the A0-A4 ablation ladder.

Cumulative design — each variant adds one reward component on top of the
previous. Weights are *starting points*; the week 3-6 sweep is exactly about
tuning them. A4 holds your chosen-best weights after that sweep.

To run the length sub-experiment, swap "length" -> "length_gated" in A2/A3/A4
and compare. To run leave-one-out on A4, copy A4 and drop one component.
"""

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Reason step by step inside "
    "<reasoning> and </reasoning> tags, then give ONLY the final numeric "
    "answer as \\boxed{...}."
)

VARIANTS = {
    # A0 — pure RLVR. The internal baseline every other variant is measured against.
    "A0": {
        "desc": "correctness only (pure RLVR baseline)",
        "rewards": ["correctness"],
        "weights": {"correctness": 1.0},
    },
    # A1 — + structural format reward.
    "A1": {
        "desc": "+ format (reasoning block + boxed answer)",
        "rewards": ["correctness", "format"],
        "weights": {"correctness": 1.0, "format": 0.2},
    },
    # A2 — + length control. Watch for the short-wrong-answer failure mode.
    "A2": {
        "desc": "+ length control (soft brevity penalty)",
        "rewards": ["correctness", "format", "length"],
        "weights": {"correctness": 1.0, "format": 0.2, "length": 0.1},
    },
    # A3 — + partial credit (process / credit-assignment signal).
    "A3": {
        "desc": "+ partial credit (intermediate-number overlap)",
        "rewards": ["correctness", "format", "length", "partial"],
        "weights": {"correctness": 1.0, "format": 0.2, "length": 0.1, "partial": 0.3},
    },
    # A4 — full reward, weights tuned from the A0-A3 sweep. Update these after week 3-6.
    "A4": {
        "desc": "full reward with tuned weights",
        "rewards": ["correctness", "format", "length", "partial"],
        "weights": {"correctness": 1.0, "format": 0.3, "length": 0.15, "partial": 0.4},
    },
}

SEEDS = [0, 1, 2, 3, 4]
