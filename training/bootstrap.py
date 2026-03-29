"""
Bootstrap utilities for retrieval metrics.

bootstrap_ci(metric_fn, items, B=1000, seed=0) -> (mean, lo, hi)
  - metric_fn: callable that takes a list of items (e.g. per-query metric
    contributions or (ranked_names, expected_name) tuples) and returns a
    single float metric.
  - items: list of items (length N).
  - B: number of bootstrap resamples.
  - seed: RNG seed for reproducibility.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Tuple
import random


def bootstrap_ci(
    metric_fn: Callable[[List], float],
    items: List,
    B: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Generic non-parametric bootstrap for a scalar metric.

    Returns (mean, lo, hi) where lo/hi are (1-alpha) central interval bounds.
    """
    if not items:
        return 0.0, 0.0, 0.0
    n = len(items)
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(max(1, B)):
        idxs = [rng.randrange(n) for _ in range(n)]
        resampled = [items[i] for i in idxs]
        samples.append(float(metric_fn(resampled)))
    samples.sort()
    mean = sum(samples) / len(samples)
    lo_idx = int((alpha / 2.0) * len(samples))
    hi_idx = max(0, int((1.0 - alpha / 2.0) * len(samples)) - 1)
    lo = samples[lo_idx]
    hi = samples[hi_idx]
    return mean, lo, hi


