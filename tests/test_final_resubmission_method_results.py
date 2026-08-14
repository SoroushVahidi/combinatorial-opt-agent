from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results" / "final_resubmission_method"


def test_final_resubmission_method_metrics_are_frozen():
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))

    assert summary["decision"] == "QUICK_FIX_VALIDATED"
    assert summary["method_freeze_state"] == "FROZEN_FOR_RESUBMISSION"
    assert summary["baseline"]["StrictInstantiationReady_count"] == 247
    assert summary["patched"]["StrictInstantiationReady_count"] == 255
    assert summary["baseline"]["InstantiationReady_count"] == 257
    assert summary["patched"]["InstantiationReady_count"] == 265
    assert summary["patched"]["schema_correct_count"] == 301
    assert summary["strict_transition"]["prepatch_only"] == 0
    assert summary["strict_transition"]["patched_only"] == 8
    assert summary["strict_transition"]["mcnemar_p"] == 0.0078125


def test_final_resubmission_changed_query_audit_matches_expected_ids():
    summary = json.loads((RESULT_DIR / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "nlp4lp_test_47",
        "nlp4lp_test_98",
        "nlp4lp_test_116",
        "nlp4lp_test_128",
        "nlp4lp_test_156",
        "nlp4lp_test_195",
        "nlp4lp_test_245",
        "nlp4lp_test_261",
    }
    assert set(summary["strict_transition"]["patched_only_ids"].split()) == expected

    rows = list(csv.DictReader((RESULT_DIR / "changed_queries.csv").open(newline="", encoding="utf-8")))
    rescue_rows = [r for r in rows if r["classification"] == "CORRECT_MULTIPLICATIVE_RESCUE"]
    assert {r["query_id"] for r in rescue_rows} == expected
    assert all(r["new_strict_ready"] == "1" for r in rescue_rows)
