from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from retrieval.baselines import get_baseline
from tools import nlp4lp_downstream_utility as u
from tools.topk_schema_rerank_stage_a import (
    CandidateGrounding,
    exact_mcnemar_p,
    run_diagnostic,
    select_by_rule,
)


def _candidate(schema_id: str, rank: int, score: float, coverage: float, type_match: float) -> CandidateGrounding:
    return CandidateGrounding(
        query_id="q",
        gold_schema="gold",
        schema_id=schema_id,
        rank=rank,
        retrieval_score=score,
        retrieval_margin=0.01,
        schema_hit=schema_id == "gold",
        n_expected_scalar=5,
        n_filled=int(round(coverage * 5)),
        coverage=coverage,
        type_match=type_match,
        ready=coverage >= 0.8 and type_match >= 0.8,
        key_overlap=coverage,
        extracted_number_count=6,
        unmatched_mention_count=1,
        incompatible_assignment_count=0,
        null_slot_count=5 - int(round(coverage * 5)),
        min_assignment_margin=1.0,
        mean_assignment_margin=1.0,
        lexical_overlap_query_schema=0.1,
    )


def test_tfidf_topk_retrieval_is_deterministic():
    eval_items = u._load_eval(ROOT / "data" / "processed" / "nlp4lp_eval_orig.jsonl")
    catalog, _ = u._load_catalog_as_problems(ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl")
    tfidf = get_baseline("tfidf")
    tfidf.fit(catalog)

    first = tfidf.rank(eval_items[0]["query"], top_k=5)
    second = tfidf.rank(eval_items[0]["query"], top_k=5)

    assert first == second
    assert len(first) == 5


def test_reranker_tie_breaking_prefers_higher_retrieval_score():
    cands = [
        _candidate("wrong_low", 1, 0.5, 0.8, 0.8),
        _candidate("wrong_high", 2, 0.7, 0.8, 0.8),
    ]

    selected = select_by_rule(cands, "R3_ready_cov_type_tfidf")

    assert selected.schema_id == "wrong_high"


def test_small_consistency_score_can_preserve_schema_on_retrieval_strength():
    cands = [
        _candidate("gold", 1, 1.0, 0.8, 0.8),
        _candidate("wrong", 2, 0.0, 1.0, 1.0),
    ]

    selected = select_by_rule(cands, "R5_small_consistency_score")

    assert selected.schema_id == "gold"


def test_exact_mcnemar_helper():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(0, 8) < 0.01
    assert exact_mcnemar_p(3, 3) == 1.0


def test_full_diagnostic_preserves_current_baseline(tmp_path):
    if os.environ.get("PYTHONHASHSEED") != "0":
        pytest.skip("baseline regression is defined for PYTHONHASHSEED=0")
    os.environ.setdefault("NLP4LP_GOLD_CACHE", str(ROOT / "results" / "eswa_revision" / "00_env" / "nlp4lp_gold_cache.json"))

    summary = run_diagnostic(tmp_path)

    assert summary["current_ready"] == 257
    assert summary["schema_misses"] == 30
    assert summary["current_instantiation_ready"] == 257 / 331
    assert summary["true_rescues_by_k"]["3"] >= 7
    assert summary["decision"] == "TOP2_GO"
