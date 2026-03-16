"""Tests for the Copilot-vs-our-model benchmark infrastructure.

Tests verify:
  - benchmark_cases.jsonl has expected structure and content
  - our_model_outputs.jsonl has expected structure
  - copilot_outputs.jsonl has expected structure
  - score_comparison.py runs without error and produces valid CSV
  - the 4 hand-crafted cases have full gold_param_values
  - the scorer computes non-trivial our-model scores (schema >0.5 on NLP4LP cases)
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT   = Path(__file__).resolve().parent.parent
BENCH  = ROOT / "artifacts" / "copilot_vs_model" / "benchmark_cases.jsonl"
OUR    = ROOT / "artifacts" / "copilot_vs_model" / "our_model_outputs.jsonl"
COP    = ROOT / "artifacts" / "copilot_vs_model" / "copilot_outputs.jsonl"
CSV_   = ROOT / "artifacts" / "copilot_vs_model" / "comparison_summary.csv"
SCORER = ROOT / "artifacts" / "copilot_vs_model" / "score_comparison.py"
RUNNER = ROOT / "artifacts" / "copilot_vs_model" / "run_our_model.py"


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


# ── benchmark_cases.jsonl ────────────────────────────────────────────────────

class TestBenchmarkCases:
    def test_file_exists(self):
        assert BENCH.exists(), "benchmark_cases.jsonl must exist"

    def test_has_30_cases(self):
        cases = _load_jsonl(BENCH)
        assert len(cases) == 30, f"Expected 30 cases, got {len(cases)}"

    def test_required_fields(self):
        cases = _load_jsonl(BENCH)
        required = {"case_id", "input_text", "source_split", "gold_schema_id",
                    "gold_scalar_slots", "difficulty_tags"}
        for case in cases:
            missing = required - set(case.keys())
            assert not missing, f"Case {case.get('case_id')} missing fields: {missing}"

    def test_case_ids_unique(self):
        cases = _load_jsonl(BENCH)
        ids = [c["case_id"] for c in cases]
        assert len(ids) == len(set(ids)), "Duplicate case_ids found"

    def test_4_handcrafted_with_full_gold(self):
        cases = _load_jsonl(BENCH)
        hc = [c for c in cases if c["case_id"].startswith("handcrafted")]
        assert len(hc) == 4, f"Expected 4 hand-crafted cases, got {len(hc)}"
        for c in hc:
            gp = c.get("gold_param_values")
            assert isinstance(gp, dict) and len(gp) >= 4, (
                f"Hand-crafted case {c['case_id']} must have full gold_param_values"
            )

    def test_all_cases_have_input_text(self):
        cases = _load_jsonl(BENCH)
        for c in cases:
            assert c.get("input_text"), f"Case {c['case_id']} has empty input_text"

    def test_all_cases_have_slots(self):
        cases = _load_jsonl(BENCH)
        for c in cases:
            assert c.get("gold_scalar_slots"), f"Case {c['case_id']} has no gold_scalar_slots"

    def test_categories_covered(self):
        cases = _load_jsonl(BENCH)
        all_tags = set()
        for c in cases:
            all_tags.update(c.get("difficulty_tags", []))
        required_cats = {"percent", "bounds", "float_heavy", "general"}
        missing = required_cats - all_tags
        assert not missing, f"Missing difficulty categories: {missing}"

    def test_includes_noisy_and_short_variants(self):
        cases = _load_jsonl(BENCH)
        all_tags = set()
        for c in cases:
            all_tags.update(c.get("difficulty_tags", []))
        assert "noisy" in all_tags, "Should include noisy variants"
        assert "short" in all_tags or any("_short" in c["case_id"] for c in cases), (
            "Should include short variants"
        )


# ── our_model_outputs.jsonl ──────────────────────────────────────────────────

class TestOurModelOutputs:
    def test_file_exists(self):
        assert OUR.exists(), "our_model_outputs.jsonl must exist"

    def test_has_30_outputs(self):
        outs = _load_jsonl(OUR)
        assert len(outs) == 30, f"Expected 30 outputs, got {len(outs)}"

    def test_required_fields(self):
        outs = _load_jsonl(OUR)
        required = {"case_id", "input_text", "predicted_schema_id", "schema_correct",
                    "predicted_slots", "slot_value_assignments", "method"}
        for o in outs:
            missing = required - set(o.keys())
            assert not missing, f"Output {o.get('case_id')} missing: {missing}"

    def test_method_is_known(self):
        """Method field must name a valid grounding strategy."""
        valid_methods = {
            "global_consistency_grounding",
            "slot_aware_extraction",
            "none",
        }
        outs = _load_jsonl(OUR)
        for o in outs:
            method = o.get("method", "")
            # Method string may be composite, e.g. "hybrid_bm25_tfidf+slot_aware_extraction"
            has_valid = any(vm in method for vm in valid_methods)
            assert has_valid, (
                f"Output {o.get('case_id')} has unrecognised method: {method!r}"
            )

    def test_schema_accuracy_above_60pct(self):
        """Our model must achieve at least 60% schema retrieval on these 30 cases."""
        outs = _load_jsonl(OUR)
        acc = sum(o["schema_correct"] for o in outs) / len(outs)
        assert acc >= 0.60, f"Schema accuracy {acc:.1%} below threshold 60%"

    def test_schema_accuracy_nlp4lp_above_75pct(self):
        """On NLP4LP in-distribution cases the bar is higher."""
        outs = _load_jsonl(OUR)
        nlp4lp = [o for o in outs if not o["case_id"].startswith("handcrafted")]
        acc = sum(o["schema_correct"] for o in nlp4lp) / len(nlp4lp)
        assert acc >= 0.75, f"NLP4LP schema accuracy {acc:.1%} below threshold 75%"

    def test_slot_value_assignments_is_dict(self):
        outs = _load_jsonl(OUR)
        for o in outs:
            assert isinstance(o.get("slot_value_assignments"), dict), (
                f"Case {o['case_id']} slot_value_assignments is not a dict"
            )

    def test_no_negative_values(self):
        """Our grounding should not assign negative values in these LP problems."""
        outs = _load_jsonl(OUR)
        for o in outs:
            for slot, val in (o.get("slot_value_assignments") or {}).items():
                if val is not None:
                    try:
                        assert float(val) >= 0, (
                            f"Negative value {val} for slot {slot} in {o['case_id']}"
                        )
                    except (TypeError, ValueError):
                        pass


# ── copilot_outputs.jsonl ─────────────────────────────────────────────────────

class TestCopilotOutputs:
    def test_file_exists(self):
        assert COP.exists(), "copilot_outputs.jsonl must exist"

    def test_has_30_entries(self):
        entries = _load_jsonl(COP)
        assert len(entries) == 30, f"Expected 30 copilot entries, got {len(entries)}"

    def test_required_fields(self):
        entries = _load_jsonl(COP)
        for e in entries:
            assert "case_id" in e, "copilot entry must have case_id"
            assert "model" in e, "copilot entry must have model"

    def test_4_handcrafted_are_filled(self):
        """The 4 simulated hand-crafted responses should be parsed (not PENDING)."""
        entries = {e["case_id"]: e for e in _load_jsonl(COP)}
        hc_ids = [c["case_id"] for c in _load_jsonl(BENCH) if c["case_id"].startswith("handcrafted")]
        assert len(hc_ids) == 4, f"Expected 4 hand-crafted cases in benchmark, got {len(hc_ids)}"
        for case_id in hc_ids:
            e = entries.get(case_id)
            assert e is not None, f"Missing copilot entry for {case_id}"
            assert e.get("parsed") is not None, (
                f"Copilot entry for {case_id} is not filled in (parsed is None)"
            )
            n_slots = len((e["parsed"] or {}).get("slot_value_assignments", {}))
            assert n_slots >= 4, (
                f"Expected ≥4 slot assignments for {case_id}, got {n_slots}"
            )

    def test_all_26_nlp4lp_are_filled(self):
        """All 26 NLP4LP cases now have simulated LLM responses (no longer PENDING)."""
        entries = _load_jsonl(COP)
        pending = [e for e in entries if e.get("parse_error") == "PENDING"]
        assert len(pending) == 0, (
            f"Expected 0 pending entries after LLM simulation, got {len(pending)}"
        )
        nlp4lp = [e for e in entries if e["case_id"].startswith("nlp4lp")]
        assert len(nlp4lp) == 26, f"Expected 26 NLP4LP entries, got {len(nlp4lp)}"
        for e in nlp4lp:
            assert e.get("parsed") is not None, (
                f"NLP4LP entry {e['case_id']} has no parsed output"
            )
            assert e.get("model") == "gpt4-claude-simulated", (
                f"NLP4LP entry {e['case_id']} has unexpected model: {e.get('model')}"
            )


# ── score_comparison.py ───────────────────────────────────────────────────────

class TestScorer:
    def test_scorer_runs(self):
        result = subprocess.run(
            [sys.executable, str(SCORER)],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        assert result.returncode == 0, (
            f"Scorer exited with code {result.returncode}\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

    def test_csv_exists_after_scorer(self):
        assert CSV_.exists(), "comparison_summary.csv must be created by scorer"

    def test_csv_has_30_rows(self):
        assert CSV_.exists()
        with CSV_.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 30, f"Expected 30 CSV rows, got {len(rows)}"

    def test_csv_has_required_columns(self):
        assert CSV_.exists()
        with CSV_.open() as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
        required = {
            "case_id", "category", "our_model_schema_correct",
            "our_model_overall", "copilot_overall", "winner",
        }
        missing = required - cols
        assert not missing, f"CSV missing columns: {missing}"

    def test_our_model_overall_scores_are_numeric(self):
        assert CSV_.exists()
        with CSV_.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            try:
                v = float(row["our_model_overall"])
                assert 0.0 <= v <= 1.0, f"our_model_overall out of range: {v}"
            except (ValueError, KeyError) as e:
                pytest.fail(f"Could not parse our_model_overall for {row.get('case_id')}: {e}")

    def test_our_model_median_overall_above_threshold(self):
        assert CSV_.exists()
        with CSV_.open() as f:
            rows = list(csv.DictReader(f))
        scores = sorted(float(r["our_model_overall"]) for r in rows)
        median = scores[len(scores) // 2]
        assert median >= 0.50, f"Median our_model_overall {median:.3f} below 0.50"


# ── runner script ─────────────────────────────────────────────────────────────

class TestRunner:
    def test_runner_file_exists(self):
        assert RUNNER.exists()

    def test_runner_is_valid_python(self):
        import ast
        try:
            ast.parse(RUNNER.read_text())
        except SyntaxError as e:
            pytest.fail(f"run_our_model.py has syntax error: {e}")


# ── comparison results (LLM simulation vs our model) ─────────────────────────

class TestComparisonResults:
    """Verify key findings from the LLM-simulation vs our-model comparison."""

    def _load_csv(self):
        assert CSV_.exists(), "comparison_summary.csv must exist; run score_comparison.py"
        with CSV_.open() as f:
            return list(csv.DictReader(f))

    def test_all_30_cases_are_scored(self):
        """After LLM simulation, every case should be scored (no pending)."""
        rows = self._load_csv()
        pending = [r for r in rows if r["winner"] == "pending"]
        assert len(pending) == 0, (
            f"Expected 0 pending rows, got {len(pending)}: {[r['case_id'] for r in pending]}"
        )

    def test_llm_grounding_coverage_beats_our_model(self):
        """LLM simulation achieves higher grounding coverage than our TF-IDF model."""
        rows = self._load_csv()
        llm_cov = sum(float(r["copilot_grounding_coverage"]) for r in rows) / len(rows)
        our_cov = sum(float(r["our_model_grounding_coverage"]) for r in rows) / len(rows)
        assert llm_cov > our_cov, (
            f"LLM grounding coverage {llm_cov:.3f} should exceed our model {our_cov:.3f}"
        )

    def test_our_model_hallucination_rate_not_worse_than_llm(self):
        """Our model should not hallucinate more than the LLM simulation."""
        rows = self._load_csv()
        our_h = sum(float(r["our_model_no_hallucination"]) for r in rows) / len(rows)
        llm_h = sum(float(r["copilot_no_hallucination"]) for r in rows) / len(rows)
        # Both should be high; our model is template-based so never introduces new numbers
        assert our_h >= llm_h - 0.10, (
            f"Our model hallucination score {our_h:.3f} is unexpectedly much worse than LLM {llm_h:.3f}"
        )

    def test_llm_overall_score_above_threshold(self):
        """LLM simulation overall score (average across 30 cases) should be ≥0.70."""
        rows = self._load_csv()
        avg = sum(float(r["copilot_overall"]) for r in rows) / len(rows)
        assert avg >= 0.70, f"LLM average overall score {avg:.3f} below threshold 0.70"

    def test_handcrafted_value_exact_match_both_systems(self):
        """Both systems should achieve ≥0.85 exact-value match on the 4 hand-crafted cases."""
        rows = self._load_csv()
        hc_rows = [r for r in rows if r["has_gold_values"] == "1"]
        assert len(hc_rows) == 4, f"Expected 4 gold-value rows, got {len(hc_rows)}"
        for system, col in [("our_model", "our_model_value_exact"),
                             ("copilot",   "copilot_value_exact")]:
            scores = [float(r[col]) for r in hc_rows if r[col] != ""]
            if scores:
                avg = sum(scores) / len(scores)
                assert avg >= 0.85, (
                    f"{system} value exact match {avg:.3f} below threshold 0.85 on hand-crafted cases"
                )

    def test_llm_objective_direction_perfect(self):
        """LLM simulation should identify objective direction correctly on all full-text cases."""
        rows = self._load_csv()
        full_text = [r for r in rows
                     if not any(tag in r["category"] for tag in ["short", "noisy"])]
        scores = [float(r["copilot_objective_dir"]) for r in full_text]
        avg = sum(scores) / len(scores) if scores else 0.0
        assert avg >= 0.95, (
            f"LLM objective direction {avg:.3f} below 0.95 on full-text cases"
        )

    def test_llm_type_correctness_above_threshold(self):
        """LLM simulation type correctness should be ≥0.80 (handles % → fraction correctly)."""
        rows = self._load_csv()
        avg = sum(float(r["copilot_type_correct"]) for r in rows) / len(rows)
        assert avg >= 0.80, (
            f"LLM type correctness {avg:.3f} below threshold 0.80"
        )

    def test_csv_winner_column_valid_values(self):
        """All winner column values must be one of the expected categories."""
        rows = self._load_csv()
        valid = {"our_model", "copilot", "tie"}
        for row in rows:
            assert row["winner"] in valid, (
                f"Unexpected winner value {row['winner']!r} for case {row['case_id']}"
            )
