"""
Unit tests for training/metrics.py: precision_at_k, reciprocal_rank_at_k, ndcg_at_k,
coverage_at_k, and compute_metrics.
"""
from __future__ import annotations

import pytest


def test_precision_at_k_hit():
    """P@k returns 1.0 when the expected name is in the top-k."""
    from training.metrics import precision_at_k
    assert precision_at_k(["A", "B", "C"], "A", 1) == 1.0
    assert precision_at_k(["A", "B", "C"], "B", 2) == 1.0
    assert precision_at_k(["A", "B", "C"], "C", 3) == 1.0


def test_precision_at_k_miss():
    """P@k returns 0.0 when the expected name is outside the top-k window."""
    from training.metrics import precision_at_k
    assert precision_at_k(["A", "B", "C"], "C", 2) == 0.0
    assert precision_at_k(["A", "B", "C"], "D", 3) == 0.0


def test_precision_at_k_edge_cases():
    """P@k returns 0.0 for k<1 and empty expected_name."""
    from training.metrics import precision_at_k
    assert precision_at_k(["A"], "A", 0) == 0.0
    assert precision_at_k(["A"], "", 1) == 0.0
    assert precision_at_k([], "A", 1) == 0.0


def test_reciprocal_rank_exact_positions():
    """MRR returns 1/rank for the first matching item."""
    from training.metrics import reciprocal_rank_at_k
    assert reciprocal_rank_at_k(["A", "B", "C"], "A", 3) == pytest.approx(1.0)
    assert reciprocal_rank_at_k(["A", "B", "C"], "B", 3) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["A", "B", "C"], "C", 3) == pytest.approx(1 / 3)


def test_reciprocal_rank_miss():
    """MRR returns 0.0 when expected is not in top-k."""
    from training.metrics import reciprocal_rank_at_k
    assert reciprocal_rank_at_k(["A", "B", "C"], "D", 3) == 0.0
    assert reciprocal_rank_at_k(["A", "B", "C"], "C", 2) == 0.0


def test_reciprocal_rank_edge_cases():
    from training.metrics import reciprocal_rank_at_k
    assert reciprocal_rank_at_k([], "A", 5) == 0.0
    assert reciprocal_rank_at_k(["A"], "", 5) == 0.0
    assert reciprocal_rank_at_k(["A"], "A", 0) == 0.0


def test_ndcg_at_k_perfect():
    """nDCG@k is 1.0 when expected is the first result."""
    from training.metrics import ndcg_at_k
    assert ndcg_at_k(["A", "B", "C"], "A", 3) == pytest.approx(1.0)


def test_ndcg_at_k_lower_positions():
    """nDCG@k decreases as expected result moves down."""
    from training.metrics import ndcg_at_k
    import math
    # Expected at position 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
    expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert ndcg_at_k(["A", "B", "C"], "B", 3) == pytest.approx(expected)


def test_ndcg_at_k_miss():
    """nDCG@k is 0.0 when expected is not in top-k."""
    from training.metrics import ndcg_at_k
    assert ndcg_at_k(["A", "B", "C"], "D", 3) == 0.0
    assert ndcg_at_k(["A", "B", "C"], "C", 2) == 0.0


def test_ndcg_at_k_edge_cases():
    from training.metrics import ndcg_at_k
    assert ndcg_at_k([], "A", 5) == 0.0
    assert ndcg_at_k(["A"], "", 5) == 0.0
    assert ndcg_at_k(["A"], "A", 0) == 0.0


def test_coverage_at_k_hit():
    """coverage_at_k returns 1.0 when expected is in top-k."""
    from training.metrics import coverage_at_k
    assert coverage_at_k(["A", "B", "C"], "A", 1) == 1.0
    assert coverage_at_k(["A", "B", "C"], "C", 3) == 1.0


def test_coverage_at_k_miss():
    """coverage_at_k returns 0.0 when expected is not in top-k."""
    from training.metrics import coverage_at_k
    assert coverage_at_k(["A", "B", "C"], "C", 2) == 0.0
    assert coverage_at_k(["A", "B", "C"], "D", 3) == 0.0


def test_coverage_at_k_edge_cases():
    """coverage_at_k handles k<1 and empty expected_name safely."""
    from training.metrics import coverage_at_k
    assert coverage_at_k(["A"], "A", 0) == 0.0
    assert coverage_at_k(["A"], "", 1) == 0.0
    assert coverage_at_k([], "A", 1) == 0.0


def test_compute_metrics_all_correct():
    """compute_metrics returns perfect scores when every query is answered correctly."""
    from training.metrics import compute_metrics
    results = [
        (["A", "B", "C"], "A"),
        (["X", "Y", "Z"], "X"),
    ]
    m = compute_metrics(results, k=5)
    assert m["P@1"] == pytest.approx(1.0)
    assert m["P@5"] == pytest.approx(1.0)
    assert m["MRR@5"] == pytest.approx(1.0)
    assert m["nDCG@5"] == pytest.approx(1.0)
    assert m["Coverage@5"] == pytest.approx(1.0)


def test_compute_metrics_all_wrong():
    """compute_metrics returns zero scores when nothing is found."""
    from training.metrics import compute_metrics
    results = [
        (["A", "B", "C"], "D"),
        (["X", "Y", "Z"], "W"),
    ]
    m = compute_metrics(results, k=5)
    assert m["P@1"] == 0.0
    assert m["MRR@5"] == 0.0
    assert m["Coverage@5"] == 0.0


def test_compute_metrics_empty():
    """compute_metrics handles an empty results list without error."""
    from training.metrics import compute_metrics
    m = compute_metrics([], k=10)
    assert m["P@1"] == 0.0
    assert "MRR@10" in m
    assert "nDCG@10" in m
    assert "Coverage@10" in m


def test_compute_metrics_partial():
    """compute_metrics averages over mixed hits and misses."""
    from training.metrics import compute_metrics
    results = [
        (["A", "B"], "A"),  # hit at 1
        (["A", "B"], "B"),  # hit at 2
        (["A", "B"], "C"),  # miss
    ]
    m = compute_metrics(results, k=5)
    # P@1: 1 hit out of 3 → 1/3
    assert m["P@1"] == pytest.approx(1 / 3)
    # P@5: 2 hits out of 3 → 2/3
    assert m["P@5"] == pytest.approx(2 / 3)
