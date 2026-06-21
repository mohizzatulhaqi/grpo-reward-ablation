"""Tests for the GPU-free core: verifier, reward components, pass@k, stats.

These cover the logic that actually determines your reward signal and your
reported numbers. Run with: pytest -q
"""
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verify
import rewards
from analysis.passk import pass_at_k
from analysis import stats
from analysis.aggregate import build_table
from config import VARIANTS


# ---------------- verifier ----------------

def test_extract_boxed():
    assert verify.extract_answer(r"reasoning... \boxed{72}") == 72.0

def test_extract_boxed_takes_last():
    assert verify.extract_answer(r"\boxed{1} then \boxed{42}") == 42.0

def test_extract_hash():
    assert verify.extract_answer("work\n#### 18") == 18.0

def test_extract_answer_phrase_with_commas_and_dollar():
    assert verify.extract_answer("The final answer is $1,200.") == 1200.0

def test_extract_last_number_fallback():
    assert verify.extract_answer("first 5 then 10 and 15") == 15.0

def test_extract_none_when_empty():
    assert verify.extract_answer("no digits here") is None
    assert verify.extract_answer("") is None

def test_to_number_fraction():
    assert verify.to_number("3/4") == 0.75

def test_answers_match():
    assert verify.answers_match(72.0, 72) is True
    assert verify.answers_match(72.0, 73) is False
    assert verify.answers_match(None, 5) is False

def test_extract_gold_answer():
    assert verify.extract_gold_answer("Janet sells... #### 18") == 18.0

def test_extract_numbers():
    assert verify.extract_numbers("2 + 3 = 5") == [2.0, 3.0, 5.0]


# ---------------- reward components ----------------

def test_correctness_reward_string_completions():
    comps = [r"\boxed{72}", r"\boxed{8}"]
    assert rewards.correctness_reward(comps, answer=[72.0, 9.0]) == [1.0, 0.0]

def test_correctness_reward_conversational():
    comps = [[{"role": "assistant", "content": r"\boxed{72}"}]]
    assert rewards.correctness_reward(comps, answer=[72.0]) == [1.0]

def test_format_reward_levels():
    full = r"<reasoning>steps</reasoning> \boxed{5}"
    half_struct = r"<reasoning>steps</reasoning> no box"
    half_box = r"just \boxed{5}"
    none = "plain text 5"
    out = rewards.format_reward([full, half_struct, half_box, none])
    assert out == [1.0, 0.5, 0.5, 0.0]

def test_length_reward_bounds():
    short = "a" * 100
    longer = "a" * 1024
    out = rewards.length_reward([short, longer], target_len=512, hard_len=1024)
    assert out[0] == 1.0 and out[1] == 0.0

def test_length_reward_gated_zeros_wrong():
    comps = [r"short \boxed{72}", r"short \boxed{0}"]
    out = rewards.length_reward_gated(comps, answer=[72.0, 72.0], target_len=512)
    assert out[0] == 1.0 and out[1] == 0.0  # second is wrong -> 0 despite being short

def test_partial_credit_full_and_half():
    full = rewards.partial_credit_reward(["chain 2, 12, 72"],
                                         gold_numbers=[[2.0, 12.0, 72.0]])
    half = rewards.partial_credit_reward(["only 2 and 12"],
                                         gold_numbers=[[2.0, 12.0, 72.0, 6.0]])
    assert full == [1.0]
    assert math.isclose(half[0], 0.5)

def test_build_reward_funcs_a3():
    funcs, weights = rewards.build_reward_funcs(VARIANTS["A3"])
    assert len(funcs) == 4
    assert weights == [1.0, 0.2, 0.1, 0.3]


# ---------------- pass@k ----------------

def test_passk_extremes():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 4, 1) == 0.5

def test_passk_saturates_when_enough_correct():
    assert pass_at_k(4, 1, 4) == 1.0  # n-c=3 < k=4

def test_passk_known_value():
    # n=10, c=5, k=2 -> 1 - (5/10)(4/9) = 1 - 0.2222...
    assert math.isclose(pass_at_k(10, 5, 2), 1 - (5/10) * (4/9))

def test_passk_rejects_bad_k():
    try:
        pass_at_k(4, 1, 5)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------- stats ----------------

def test_cliffs_delta_direction():
    d = stats.cliffs_delta([3, 4, 5], [1, 2, 3])
    assert d > 0
    assert stats.interpret_cliffs(d) in {"medium", "large"}

def test_cliffs_delta_symmetry():
    d1 = stats.cliffs_delta([3, 4, 5], [1, 2, 3])
    d2 = stats.cliffs_delta([1, 2, 3], [3, 4, 5])
    assert math.isclose(d1, -d2)

def test_compare_paired_runs_and_has_keys():
    a = [0.50, 0.52, 0.55, 0.53, 0.51]
    b = [0.40, 0.42, 0.39, 0.45, 0.41]
    res = stats.compare(a, b, paired=True)
    assert res["test"] == "wilcoxon_signed_rank"
    assert res["cliffs_delta"] > 0
    assert "p_value" in res and not math.isnan(res["p_value"])

def test_compare_unequal_length_paired_skips():
    res = stats.compare([1, 2, 3], [1, 2], paired=True)
    assert res["test"] == "skipped"


# ---------------- aggregate table ----------------

def test_build_table_basic():
    records = []
    for seed in range(3):
        records.append({"variant": "A0", "seed": seed, "pass@1": 0.40 + 0.01 * seed})
        records.append({"variant": "A1", "seed": seed, "pass@1": 0.50 + 0.01 * seed})
    rows = build_table(records, metric="pass@1", baseline="A0")
    by = {r["variant"]: r for r in rows}
    assert by["A0"]["delta_vs_base"] == 0.0
    assert by["A1"]["delta_vs_base"] > 0.09  # ~0.10 improvement
    assert by["A1"]["cliffs"] > 0
