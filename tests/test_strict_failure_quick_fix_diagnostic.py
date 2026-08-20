from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.strict_failure_quick_fix_diagnostic import (  # noqa: E402
    _multiplicative_ratio_word_tokens,
    _text_exposes_missing_gold,
    generate,
)


def test_ratio_word_tokens_are_deterministic_and_ratio_typed():
    toks = _multiplicative_ratio_word_tokens("At least twice as many A as B; triple if needed.")
    assert [(t.raw, t.value, t.kind) for t in toks] == [
        ("RATIO_WORD:twice", 2.0, "percent"),
        ("RATIO_WORD:triple", 3.0, "percent"),
    ]


def test_text_exposed_missing_gold_is_narrow():
    assert _text_exposes_missing_gold("at least twice as many", 2.0)
    assert _text_exposes_missing_gold("three times as many", 3.0)
    assert not _text_exposes_missing_gold("there are several products", 5.0)


def test_generate_quick_fix_regression_counts(tmp_path: Path):
    summary = generate(tmp_path)
    assert summary["baseline_strict_ready"] == 247
    assert summary["schema_correct_not_ready"] == 54
    assert summary["oracle_schema_not_ready"] == 58
    assert summary["decision"] == "QUICK_FIX_GO"
    assert summary["resubmission_recommendation"] == "IMPLEMENT_ONE_QUICK_FIX_THEN_FREEZE_METHOD"
    assert summary["ratio_word_prototype"]["prototype_strict_ready"] == 255
    assert summary["ratio_word_prototype"]["gains"] == 8
    assert summary["ratio_word_prototype"]["losses"] == 0
    assert summary["candidate_fixes"][0]["candidate"] == "multiplicative_ratio_word_extraction"
    assert summary["candidate_fixes"][0]["confidence"] == "HIGH"


def test_oracle_bounds_are_leakage_controlled(tmp_path: Path):
    summary = generate(tmp_path)
    interventions = {r["intervention"]: r for r in summary["oracle_interventions"]}
    assert interventions["perfect_numeric_extraction_only"]["rescued_queries"] == 7
    assert interventions["multiplicative_ratio_word_extraction_prototype"]["rescued_queries"] == 8
    assert summary["root_cause_query_counts"]["SCHEMA_SLOT_REPRESENTATION_MISMATCH"] == 21
