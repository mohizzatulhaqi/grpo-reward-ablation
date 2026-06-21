"""Unbiased pass@k estimator (Chen et al., 2021, the Codex paper).

pass@k = 1 - C(n-c, k) / C(n, k)

computed in product form to avoid overflow on large binomials. pass@1 from this
estimator is just c/n averaged over questions — the standard low-variance way to
report pass@1 under sampling. pass@k (k>1) is your headroom metric for the
"does RLVR expand reasoning or just sharpen the distribution" question.
"""


def pass_at_k(n, c, k):
    """n = samples drawn, c = number correct, k = budget. Returns a probability."""
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c < 0 or c > n:
        raise ValueError(f"c ({c}) must be in [0, n]")
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod
