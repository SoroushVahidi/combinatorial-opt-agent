from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as u


def test_score_formula_and_normalization_are_frozen():
    assert u._normalize_retrieval_scores([2.0, 1.0, 0.0]) == [1.0, 0.5, 0.0]
    assert u._normalize_retrieval_scores([3.0, 3.0]) == [1.0, 1.0]
    assert u._selective_grounding_consistency_score(1.0, 0.8, 0.6) == pytest.approx(
        0.50 * 1.0 + 0.25 * 0.8 + 0.25 * 0.6
    )


def test_high_margin_returns_plain_tfidf_top1():
    catalog, _ = u._load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    tfidf = get_baseline("tfidf")
    tfidf.fit(catalog)
    query = "Completely distinctive query text"
    gold_by_id = {"gold": {"parameters": {}, "problem_info": {}}}
    rank_fn = u.make_selective_grounding_rerank_rank_fn(
        lambda _q, top_k=1: [("a", 1.0), ("b", 0.0), ("c", 0.0), ("d", 0.0), ("e", 0.0)][:top_k],
        gold_by_id,
        {query: "gold"},
        variant="orig",
    )

    assert rank_fn(query, top_k=1)[0][0] == "a"
    assert getattr(rank_fn, "_selective_grounding_rerank_diagnostics")[-1]["triggered"] is False


def test_tie_breaking_is_deterministic():
    query = "There are 5 units."
    gold_by_id = {
        "gold": {"parameters": {"X": 5}, "problem_info": {"parameters": {"X": {"shape": []}}}},
        "a": {"parameters": {"X": 5}, "problem_info": {"parameters": {"X": {"shape": []}}}},
        "b": {"parameters": {"X": 5}, "problem_info": {"parameters": {"X": {"shape": []}}}},
    }
    rank_fn = u.make_selective_grounding_rerank_rank_fn(
        lambda _q, top_k=1: [("a", 1.0), ("b", 1.0)][:top_k],
        gold_by_id,
        {query: "gold"},
        variant="orig",
        k_retrieval=2,
    )

    assert rank_fn(query, top_k=1)[0][0] == "a"
    assert rank_fn(query, top_k=1)[0][0] == "a"


def test_rank_fn_does_not_mutate_base_rank_results():
    query = "There are 5 units."
    ranked = [("a", 1.0), ("b", 0.99)]
    gold_by_id = {"gold": {"parameters": {}, "problem_info": {}}}
    rank_fn = u.make_selective_grounding_rerank_rank_fn(
        lambda _q, top_k=1: ranked[:top_k],
        gold_by_id,
        {query: "gold"},
        variant="orig",
        k_retrieval=2,
    )

    _ = rank_fn(query, top_k=1)

    assert ranked == [("a", 1.0), ("b", 0.99)]


def test_production_candidate_regression(tmp_path):
    if os.environ.get("PYTHONHASHSEED") != "0":
        pytest.skip("aggregate regression is defined for PYTHONHASHSEED=0")
    os.environ.setdefault(
        "NLP4LP_GOLD_CACHE",
        str(ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"),
    )

    assert u.run_single_setting("orig", "tfidf_selective_grounding_rerank", "typed", tmp_path)
    import json

    with open(tmp_path / "nlp4lp_downstream_orig_tfidf_selective_grounding_rerank.json") as f:
        agg = json.load(f)["aggregate"]

    assert agg["instantiation_ready"] == 265 / 331
    assert agg["schema_R1"] == 303 / 331
