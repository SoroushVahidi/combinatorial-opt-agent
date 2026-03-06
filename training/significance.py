"""
Paired bootstrap significance tests for retrieval metrics.

paired_bootstrap_test(metric_per_query_A, metric_per_query_B, B=1000, seed=0) -> p_value
"""
from __future__ import annotations

from typing import List
import random


def paired_bootstrap_test(
    metric_per_query_A: List[float],
    metric_per_query_B: List[float],
    B: int = 1000,
    seed: int = 0,
) -> float:
    """
    Paired bootstrap test for difference in mean (A - B).
    Returns two-sided p-value: proportion of bootstrap samples where
    sign of difference is opposite to observed (or crosses zero), doubled.
    """
    if not metric_per_query_A or not metric_per_query_B:
        return 1.0
    if len(metric_per_query_A) != len(metric_per_query_B):
        n = min(len(metric_per_query_A), len(metric_per_query_B))
        metric_per_query_A = metric_per_query_A[:n]
        metric_per_query_B = metric_per_query_B[:n]
    n = len(metric_per_query_A)
    diffs = [a - b for a, b in zip(metric_per_query_A, metric_per_query_B)]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    count_opposite = 0
    for _ in range(max(1, B)):
        idxs = [rng.randrange(n) for _ in range(n)]
        boot_diffs = [diffs[i] for i in idxs]
        d = sum(boot_diffs) / n
        # count samples where sign differs from observed or crosses zero
        if obs > 0 and d <= 0:
            count_opposite += 1
        elif obs < 0 and d >= 0:
            count_opposite += 1
        elif obs == 0:
            if d != 0:
                count_opposite += 1
    p = (count_opposite / max(1, B)) * 2.0
    return min(1.0, max(0.0, p))


