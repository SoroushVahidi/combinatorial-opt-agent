"""
Retrieval metrics: P@k, MRR@k, nDCG@k, Coverage@k.
Also provides failure-mode diagnosis to distinguish retrieval misses from instances
where the correct problem was retrieved but the query contains specific numeric
parameter values that would need a follow-up parameter-extraction step.
Expects ranked list of (problem, score) and expected problem id/name for each instance.
"""

from __future__ import annotations

import re


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


# ---------------------------------------------------------------------------
# Failure-mode diagnosis
# ---------------------------------------------------------------------------

def has_numeric_parameters(query: str) -> bool:
    """Return True if the query contains specific numeric values (integers or decimals).

    Used as a heuristic to detect queries that describe a *concrete* problem
    instance with specific parameter values (e.g. "capacity = 50", "5 items
    with weights 3, 4, 2").  Such queries require a parameter-extraction step
    *beyond* retrieval to fully instantiate the formulation template.

    Returns False for purely conceptual queries ("minimize shipping cost") that
    contain no numeric constants.
    """
    if not isinstance(query, str) or not query:
        return False
    return bool(re.search(r"\b\d+(?:\.\d+)?\b", query))


def classify_failure_mode(
    query: str,
    ranked_names: list[str],
    expected_name: str,
    k: int,
) -> str:
    """Classify the failure mode for one retrieval instance.

    Distinguishes between two types of shortcomings:

    * **"retrieval_miss"** – the correct problem was *not* found in the top-k
      results; the system cannot connect the query to the right catalog entry.

    * **"param_extraction_needed"** – the correct problem *was* retrieved (it
      appears in the top-k), but the query contains specific numeric values
      (e.g. capacities, weights, costs) that would need a follow-up
      parameter-extraction step to fill them into the symbolic formulation
      template.

    * **"success"** – the correct problem is in the top-k and either no
      specific numeric parameters are present, or the retrieval step is
      sufficient for the user's intent.

    Note: "param_extraction_needed" signals a retrieval *success* paired with
    an unmet need for parameter instantiation — it does *not* mean the
    retrieval failed.

    Args:
        query: the original natural-language query.
        ranked_names: problem names in rank order (position 0 = best match).
        expected_name: canonical name of the ground-truth problem.
        k: top-k cutoff — only the first k elements of ranked_names are used.

    Returns:
        One of ``"success"``, ``"param_extraction_needed"``, or
        ``"retrieval_miss"``.
    """
    if not expected_name:
        return "success"
    if expected_name in ranked_names[:k]:
        if has_numeric_parameters(query):
            return "param_extraction_needed"
        return "success"
    return "retrieval_miss"


def diagnose_failures(
    instances: list[tuple[str, list[str], str]],
    k: int = 10,
) -> dict[str, int | float]:
    """Aggregate failure-mode breakdown across all eval instances.

    For each instance the mode is one of: "success", "retrieval_miss", or
    "param_extraction_needed" (see :func:`classify_failure_mode`).

    Args:
        instances: list of ``(query, ranked_names, expected_name)`` per
            instance, where *ranked_names* is ordered from best to worst match.
        k: top-k cutoff applied to each instance.

    Returns:
        A dict with the following keys:

        * ``n_total`` – total number of instances.
        * ``n_success`` – instances where retrieval succeeded with no
          outstanding parameter-extraction need.
        * ``n_retrieval_miss`` – instances where the correct problem was not
          in the top-k.
        * ``n_param_extraction_needed`` – instances where retrieval succeeded
          but specific numeric parameter values require extraction.
        * ``frac_retrieval_miss`` – fraction of instances that are retrieval
          misses (0.0 when n_total == 0).
        * ``frac_param_extraction_needed`` – fraction of instances needing
          parameter extraction (0.0 when n_total == 0).
    """
    counts: dict[str, int] = {
        "success": 0,
        "retrieval_miss": 0,
        "param_extraction_needed": 0,
    }
    for query, ranked_names, expected_name in instances:
        mode = classify_failure_mode(query, ranked_names, expected_name, k)
        counts[mode] += 1
    n = len(instances)
    return {
        "n_total": n,
        "n_success": counts["success"],
        "n_retrieval_miss": counts["retrieval_miss"],
        "n_param_extraction_needed": counts["param_extraction_needed"],
        "frac_retrieval_miss": counts["retrieval_miss"] / n if n > 0 else 0.0,
        "frac_param_extraction_needed": counts["param_extraction_needed"] / n if n > 0 else 0.0,
    }
