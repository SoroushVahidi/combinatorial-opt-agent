"""
Retrieval metrics: P@k, MRR@k, nDCG@k, Coverage@k.
Expects ranked list of (problem, score) and expected problem id/name for each instance.
"""

from __future__ import annotations


def precision_at_k(ranked_names: list[str], expected_name: str, k: int) -> float:
    """1.0 if expected is in top-k, else 0.0. k is 1-based (top 1 = P@1)."""
    if k < 1 or not expected_name:
        return 0.0
    top_k = ranked_names[:k]
    return 1.0 if expected_name in top_k else 0.0


def reciprocal_rank_at_k(ranked_names: list[str], expected_name: str, k: int) -> float:
    """1/rank of first occurrence of expected in top-k, else 0.0. k is max position (e.g. 10 for MRR@10)."""
    if k < 1 or not expected_name:
        return 0.0
    for i, name in enumerate(ranked_names[:k], start=1):
        if name == expected_name:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_names: list[str], expected_name: str, k: int) -> float:
    """nDCG@k: relevance 1 for expected, 0 else. DCG / IDCG; IDCG = 1 so nDCG = DCG."""
    if k < 1 or not expected_name:
        return 0.0
    rel = [1.0 if name == expected_name else 0.0 for name in ranked_names[:k]]
    if not rel or sum(rel) == 0:
        return 0.0

    def dcg(gains: list[float]) -> float:
        s = 0.0
        for i, g in enumerate(gains, start=1):
            s += g / (__import__("math").log2(i + 1))
        return s

    dcg_val = dcg(rel)
    # Ideal: 1 at position 1, 0 elsewhere
    ideal = [1.0] + [0.0] * (k - 1)
    idcg_val = dcg(ideal)
    if idcg_val <= 0:
        return 0.0
    return dcg_val / idcg_val


def coverage_at_k(ranked_names: list[str], expected_name: str, k: int) -> float:
    """1.0 if expected appears in top-k, else 0.0. Same as P@k for a single instance."""
    return 1.0 if expected_name in (ranked_names[:k]) else 0.0


def compute_metrics(
    results: list[tuple[list[str], str]],
    k: int = 10,
) -> dict[str, float]:
    """
    results: list of (ranked list of problem names, expected problem name) per instance.
    Returns dict with P@1, P@5, MRR@k, nDCG@k, Coverage@k (and k in key where relevant).
    """
    if not results:
        return {
            "P@1": 0.0,
            "P@5": 0.0,
            f"MRR@{k}": 0.0,
            f"nDCG@{k}": 0.0,
            f"Coverage@{k}": 0.0,
        }
    n = len(results)
    p1 = sum(precision_at_k(names, exp, 1) for names, exp in results) / n
    p5 = sum(precision_at_k(names, exp, 5) for names, exp in results) / n
    mrr = sum(reciprocal_rank_at_k(names, exp, k) for names, exp in results) / n
    ndcg = sum(ndcg_at_k(names, exp, k) for names, exp in results) / n
    cov = sum(coverage_at_k(names, exp, k) for names, exp in results) / n
    return {
        "P@1": p1,
        "P@5": p5,
        f"MRR@{k}": mrr,
        f"nDCG@{k}": ndcg,
        f"Coverage@{k}": cov,
    }
