"""Statistical comparison across seeds — the part most RLVR papers skip.

Use Wilcoxon signed-rank (paired) when the same seeds are used across variants
(the default in this repo: each variant is run on seeds 0-4). Use Mann-Whitney U
(unpaired) only if seed sets differ. Cliff's delta reports effect size and
direction independent of the significance test.
"""
import numpy as np
from scipy import stats as ss


def cliffs_delta(a, b):
    """Cliff's delta in [-1, 1]. Positive => values in `a` tend to exceed `b`."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    diff = a[:, None] - b[None, :]
    gt = int(np.sum(diff > 0))
    lt = int(np.sum(diff < 0))
    return (gt - lt) / (a.size * b.size)


def interpret_cliffs(d):
    """Romano et al. thresholds."""
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def compare(a, b, paired=True):
    """Compare two variants' per-seed metric arrays.

    Returns a dict with means/stds, the significance test result, Cliff's delta,
    and its qualitative label. Gracefully degrades (test='skipped') when a test
    is undefined (e.g. all-zero paired differences, unequal lengths).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    res = {
        "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(np.mean(a)) if a.size else float("nan"),
        "mean_b": float(np.mean(b)) if b.size else float("nan"),
        "std_a": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "std_b": float(np.std(b, ddof=1)) if b.size > 1 else 0.0,
        "cliffs_delta": cliffs_delta(a, b),
    }
    res["effect"] = interpret_cliffs(res["cliffs_delta"])
    try:
        if paired:
            if a.size != b.size:
                raise ValueError("paired test requires equal-length samples")
            stat, p = ss.wilcoxon(a, b)
            res["test"] = "wilcoxon_signed_rank"
        else:
            stat, p = ss.mannwhitneyu(a, b, alternative="two-sided")
            res["test"] = "mann_whitney_u"
        res["statistic"] = float(stat)
        res["p_value"] = float(p)
    except ValueError as e:
        res["test"] = "skipped"
        res["statistic"] = float("nan")
        res["p_value"] = float("nan")
        res["note"] = str(e)
    return res
