"""Smoke tests for evaluate_unified_grounding_status.

Verifies that:
1. The unified evaluation runs without errors
2. All 9 taxonomy categories are present in the output
3. Fix rates for hard families are correct (100% in full mode)
4. No regressions are detected
5. Artifact files are written correctly
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest


class TestUnifiedEvalRuns:
    """Smoke tests: the script runs and returns sensible output."""

    def test_returns_expected_keys(self) -> None:
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        assert "statuses" in result
        assert "residuals" in result
        assert "g1_overall" in result
        assert "g3_overall" in result

    def test_all_taxonomy_categories_present(self) -> None:
        from tools.evaluate_unified_grounding_status import (
            TAXONOMY,
            run_unified_evaluation,
        )

        result = run_unified_evaluation(verbose=False)
        statuses = result["statuses"]
        for cat in TAXONOMY:
            assert cat in statuses, f"Missing taxonomy category: {cat}"

    def test_hard_families_solved_in_full_mode(self) -> None:
        """hard_wrong_assignment and hard_swapped_quantities must be 100% in full mode."""
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        g1 = result["g1_overall"]
        g3 = result["g3_overall"]
        assert g1["full"] == pytest.approx(1.0), f"G1 full fix rate: {g1['full']}"
        assert g3["full"] == pytest.approx(1.0), f"G3 full fix rate: {g3['full']}"

    def test_no_regressions_on_targeted_sets(self) -> None:
        """Zero regressions must be reported for both hard-family evaluators."""
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        statuses = result["statuses"]
        for cat in ("hard_wrong_assignment", "hard_swapped_quantities"):
            s = statuses[cat]
            assert s.targeted_regression_count == 0, (
                f"Unexpected regression in {cat}: {s.targeted_regression_count}"
            )

    def test_no_residuals_on_targeted_set(self) -> None:
        """All targeted cases must pass in full mode — zero residuals."""
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        assert len(result["residuals"]) == 0, (
            f"Expected 0 residuals but got {len(result['residuals'])}: "
            + ", ".join(r.case_id for r in result["residuals"])
        )

    def test_hard_wrong_assignment_has_eval_cases(self) -> None:
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        s = result["statuses"]["hard_wrong_assignment"]
        assert s.targeted_eval_cases >= 14, (
            f"Expected ≥14 eval cases for hard_wrong_assignment, got {s.targeted_eval_cases}"
        )

    def test_hard_swapped_quantities_has_eval_cases(self) -> None:
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        s = result["statuses"]["hard_swapped_quantities"]
        assert s.targeted_eval_cases >= 3, (
            f"Expected ≥3 eval cases for hard_swapped_quantities, got {s.targeted_eval_cases}"
        )

    def test_g1_basic_is_lower_than_full(self) -> None:
        """Group 1 basic mode must be strictly worse than full mode."""
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        g1 = result["g1_overall"]
        assert g1["basic"] < g1["full"], (
            f"Expected basic < full for G1, got basic={g1['basic']} full={g1['full']}"
        )

    def test_g3_basic_is_lower_than_full(self) -> None:
        """Group 3 basic mode must be strictly worse than full mode."""
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        result = run_unified_evaluation(verbose=False)
        g3 = result["g3_overall"]
        assert g3["basic"] < g3["full"], (
            f"Expected basic < full for G3, got basic={g3['basic']} full={g3['full']}"
        )


class TestUnifiedEvalArtifacts:
    """Tests that the artifact files are written correctly."""

    def test_writes_unified_csv(self) -> None:
        from tools.evaluate_unified_grounding_status import TAXONOMY, run_unified_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            run_unified_evaluation(verbose=False, output_dir=Path(tmp))
            csv_path = Path(tmp) / "unified_summary.csv"
            assert csv_path.exists()
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            categories_in_csv = {r["category"] for r in rows}
            for cat in TAXONOMY:
                assert cat in categories_in_csv, f"Missing category in CSV: {cat}"

    def test_writes_unified_json(self) -> None:
        from tools.evaluate_unified_grounding_status import TAXONOMY, run_unified_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            run_unified_evaluation(verbose=False, output_dir=Path(tmp))
            json_path = Path(tmp) / "unified_summary.json"
            assert json_path.exists()
            with json_path.open() as f:
                data = json.load(f)
            assert "taxonomy" in data
            assert "ablation_summary" in data
            for cat in TAXONOMY:
                assert cat in data["taxonomy"], f"Missing category in JSON: {cat}"

    def test_writes_residual_audit_csv(self) -> None:
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            run_unified_evaluation(verbose=False, output_dir=Path(tmp))
            audit_path = Path(tmp) / "residual_audit.csv"
            assert audit_path.exists()

    def test_writes_final_report_md(self) -> None:
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            run_unified_evaluation(verbose=False, output_dir=Path(tmp))
            report_path = Path(tmp) / "final_report.md"
            assert report_path.exists()
            content = report_path.read_text()
            assert "Unified Grounding Status" in content
            assert "hard_wrong_assignment" in content
            assert "hard_swapped_quantities" in content

    def test_writes_executive_summary_md(self) -> None:
        from tools.evaluate_unified_grounding_status import run_unified_evaluation

        with tempfile.TemporaryDirectory() as tmp:
            run_unified_evaluation(verbose=False, output_dir=Path(tmp))
            exec_path = Path(tmp) / "executive_summary.md"
            assert exec_path.exists()
            content = exec_path.read_text()
            assert "Executive Summary" in content
            assert "Group 1" in content
            assert "Group 3" in content


class TestUnifiedTaxonomyIntegrity:
    """Tests that taxonomy constants are internally consistent."""

    def test_taxonomy_has_nine_categories(self) -> None:
        from tools.evaluate_unified_grounding_status import TAXONOMY

        assert len(TAXONOMY) == 9

    def test_baseline_counts_cover_taxonomy(self) -> None:
        from tools.evaluate_unified_grounding_status import TAXONOMY, _BASELINE_COUNTS

        for cat in TAXONOMY:
            assert cat in _BASELINE_COUNTS, f"Missing baseline count for {cat}"

    def test_implemented_fixes_cover_taxonomy(self) -> None:
        from tools.evaluate_unified_grounding_status import TAXONOMY, _IMPLEMENTED_FIX_COUNTS

        for cat in TAXONOMY:
            assert cat in _IMPLEMENTED_FIX_COUNTS, f"Missing fix count for {cat}"

    def test_assign_status_no_eval_data(self) -> None:
        from tools.evaluate_unified_grounding_status import _assign_status

        status, rec = _assign_status("under_specified_template", 0.0, 0.0, 0, 9, 0, 0)
        assert status == "under_evaluated"
        assert rec == "stop_inherently_ambiguous"

    def test_assign_status_solved(self) -> None:
        from tools.evaluate_unified_grounding_status import _assign_status

        status, rec = _assign_status("hard_wrong_assignment", 1.0, 0.5, 0, 67, 6, 19)
        assert status == "solved"
        assert rec == "stop"
