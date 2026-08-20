from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.strict_instantiation_ready_diagnostic import (
    generate,
    ordinary_ready,
    strict_ready,
    summarize_method,
    threshold_diagnostic,
    transition_rows,
)


def test_strict_ready_definition_gates_wrong_schema():
    row = {"schema_hit": "0", "param_coverage": "1.0", "type_match": "1.0"}
    assert ordinary_ready(row)
    assert not strict_ready(row)


def test_strict_ready_keeps_correct_schema_ready():
    row = {"schema_hit": "1", "param_coverage": "0.8", "type_match": "0.8"}
    assert ordinary_ready(row)
    assert strict_ready(row)


def test_threshold_handling_is_inclusive():
    row = {"schema_hit": "1", "param_coverage": "0.899999", "type_match": "1.0"}
    assert strict_ready(row, coverage_threshold=0.8)
    assert not strict_ready(row, coverage_threshold=0.9)


def test_method_summary_counts_false_ready_and_strict_ready():
    rows = [
        {"query_id": "a", "schema_correct": 1, "ordinary_ready": 1, "strict_ready": 1, "false_ready": 0,
         "param_coverage": 1.0, "type_match": 1.0, "exact5": "1.0", "exact20": "1.0"},
        {"query_id": "b", "schema_correct": 0, "ordinary_ready": 1, "strict_ready": 0, "false_ready": 1,
         "param_coverage": 1.0, "type_match": 1.0, "exact5": "", "exact20": ""},
    ]
    summary = summarize_method("m", rows)
    assert summary["ordinary_ready_count"] == 2
    assert summary["strict_ready_count"] == 1
    assert summary["false_ready_count"] == 1


def test_transition_analysis_reports_selective_only_strict_ids():
    a_rows = [
        {"query_id": "nlp4lp_test_1", "strict_ready": 1, "schema_correct": 1, "ordinary_ready": 1,
         "predicted_doc_id": "g1", "gold_doc_id": "g1"},
        {"query_id": "nlp4lp_test_2", "strict_ready": 0, "schema_correct": 0, "ordinary_ready": 0,
         "predicted_doc_id": "x", "gold_doc_id": "g2"},
    ]
    b_rows = [
        {"query_id": "nlp4lp_test_1", "strict_ready": 1, "schema_correct": 1, "ordinary_ready": 1,
         "predicted_doc_id": "g1", "gold_doc_id": "g1"},
        {"query_id": "nlp4lp_test_2", "strict_ready": 1, "schema_correct": 1, "ordinary_ready": 1,
         "predicted_doc_id": "g2", "gold_doc_id": "g2"},
    ]
    summary, details = transition_rows("a", a_rows, "b", b_rows, "strict_ready")
    assert summary["both"] == 1
    assert summary["b_only"] == 1
    assert summary["b_only_ids"] == "nlp4lp_test_2"
    assert details[0]["query_id"] == "nlp4lp_test_2"


def test_threshold_diagnostic_preserves_false_ready_across_stricter_thresholds():
    rows = {
        "m": [
            {"schema_correct": 0, "param_coverage": 1.0, "type_match": 1.0},
            {"schema_correct": 1, "param_coverage": 0.85, "type_match": 0.85},
        ]
    }
    out = threshold_diagnostic(rows)
    at_1 = next(r for r in out if r["coverage_threshold"] == 1.0 and r["type_match_threshold"] == 1.0)
    assert at_1["ordinary_ready_count"] == 1
    assert at_1["strict_ready_count"] == 0
    assert at_1["false_ready_count"] == 1


def test_generate_regression_counts(tmp_path: Path):
    summary = generate(tmp_path)
    assert summary["baseline"]["ordinary_ready_count"] == 257
    assert summary["baseline"]["strict_ready_count"] == 247
    assert summary["baseline"]["false_ready_count"] == 10
    assert summary["selective"]["ordinary_ready_count"] == 265
    assert summary["selective"]["strict_ready_count"] == 249
    assert summary["selective"]["false_ready_count"] == 16
    assert summary["oracle"]["strict_ready_count"] == 273
    assert summary["strict_transition"]["tfidf_selective_grounding_rerank_only_ids"] == "nlp4lp_test_222 nlp4lp_test_268"

    method_summary = list(csv.DictReader((tmp_path / "method_summary.csv").open(newline="", encoding="utf-8")))
    baseline = next(r for r in method_summary if r["method"] == "tfidf_typed_greedy")
    assert baseline["ordinary_ready_count"] == "257"
    assert baseline["strict_ready_count"] == "247"

    loaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert loaded["metric_definitions"]["strict_instantiation_ready"].startswith("SchemaCorrect")
