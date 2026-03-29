"""Tests for the large failure audit infrastructure.

Covers:
- build_large_stress_cases: case counts, structure, category coverage
- run_large_failure_audit: taxonomy labeling, aggregation, artifact generation
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# build_large_stress_cases tests
# ---------------------------------------------------------------------------

class TestBuildLargeStressCases:
    """Tests for the synthetic stress-case generator."""

    def test_total_case_count_gte_60(self):
        from tools.build_large_stress_cases import get_all_stress_cases
        cases = get_all_stress_cases()
        assert len(cases) >= 60, f"Expected >= 60 cases, got {len(cases)}"

    def test_runnable_cases_have_slots_and_expected(self):
        from tools.build_large_stress_cases import get_runnable_cases
        for c in get_runnable_cases():
            assert c.slots, f"{c.id}: runnable case has no slots"
            assert c.expected, f"{c.id}: runnable case has no expected values"

    def test_all_cases_have_required_fields(self):
        from tools.build_large_stress_cases import get_all_stress_cases
        for c in get_all_stress_cases():
            assert c.id, "case missing id"
            assert c.category, f"{c.id}: missing category"
            assert c.query, f"{c.id}: missing query"
            assert isinstance(c.secondary_tags, list), f"{c.id}: secondary_tags not a list"

    def test_all_categories_covered(self):
        from tools.build_large_stress_cases import get_all_stress_cases, TAXONOMY_GROUPS
        covered = {c.category for c in get_all_stress_cases()}
        # All 9 taxonomy categories should be represented
        for cat in TAXONOMY_GROUPS:
            # easy_retrieval has static-only cases — still counts
            assert cat in covered, f"Category {cat!r} has no stress cases"

    def test_get_cases_by_category(self):
        from tools.build_large_stress_cases import get_cases_by_category
        pct_cases = get_cases_by_category("easy_percent_type")
        assert len(pct_cases) >= 5

    def test_unique_ids(self):
        from tools.build_large_stress_cases import get_all_stress_cases
        ids = [c.id for c in get_all_stress_cases()]
        assert len(ids) == len(set(ids)), "Duplicate case IDs found"

    def test_to_dict_serialisable(self):
        from tools.build_large_stress_cases import get_all_stress_cases
        for c in get_all_stress_cases():
            d = c.to_dict()
            json.dumps(d)  # must be JSON-serialisable

    def test_main_writes_json(self, tmp_path):
        from tools.build_large_stress_cases import get_all_stress_cases, get_runnable_cases
        out = tmp_path / "stress.json"
        cases = get_all_stress_cases()
        runnable = get_runnable_cases()
        data = {
            "total": len(cases),
            "runnable": len(runnable),
            "cases": [c.to_dict() for c in cases],
        }
        out.write_text(json.dumps(data))
        loaded = json.loads(out.read_text())
        assert loaded["total"] == len(cases)
        assert loaded["runnable"] == len(runnable)

    def test_hard_wrong_assignment_has_sibling_variants(self):
        from tools.build_large_stress_cases import get_cases_by_category
        cases = get_cases_by_category("hard_wrong_assignment")
        # Should include sibling_swap and wrong_entity_family patterns
        all_tags = {t for c in cases for t in c.secondary_tags}
        assert "sibling_swap" in all_tags or "wrong_entity_family" in all_tags

    def test_hard_swapped_quantities_has_sibling_swap_tag(self):
        from tools.build_large_stress_cases import get_cases_by_category
        cases = get_cases_by_category("hard_swapped_quantities")
        assert all("sibling_swap" in c.secondary_tags for c in cases), (
            "All hard_swapped_quantities cases should have sibling_swap tag"
        )

    def test_mixed_cases_have_multiple_tags(self):
        from tools.build_large_stress_cases import get_cases_by_category
        cases = get_cases_by_category("mixed_or_other")
        multi_tag = [c for c in cases if len(c.secondary_tags) >= 2]
        assert len(multi_tag) >= 3, "Expected at least 3 mixed cases with >= 2 tags"


# ---------------------------------------------------------------------------
# TAXONOMY_GROUPS constant
# ---------------------------------------------------------------------------

class TestTaxonomyGroups:
    def test_all_nine_groups_present(self):
        from tools.run_large_failure_audit import TAXONOMY_GROUPS
        assert len(TAXONOMY_GROUPS) == 9
        assert "easy_percent_type" in TAXONOMY_GROUPS
        assert "hard_swapped_quantities" in TAXONOMY_GROUPS
        assert "under_specified_template" in TAXONOMY_GROUPS
        assert "mixed_or_other" in TAXONOMY_GROUPS

    def test_secondary_tags_list_nonempty(self):
        from tools.run_large_failure_audit import SECONDARY_TAGS
        assert len(SECONDARY_TAGS) >= 15
        assert "sibling_swap" in SECONDARY_TAGS
        assert "percent_normalization_error" in SECONDARY_TAGS
        assert "lower_upper_reversal" in SECONDARY_TAGS


# ---------------------------------------------------------------------------
# _label_failure heuristics
# ---------------------------------------------------------------------------

class TestLabelFailureHeuristics:
    """Tests for the deterministic failure-labeling heuristics."""

    def _label(self, *args, **kwargs):
        from tools.run_large_failure_audit import _label_failure
        return _label_failure(*args, **kwargs)

    def test_passed_returns_passed_note(self):
        tags, note = self._label("id", "hard_wrong_assignment", [], {}, {}, passed=True)
        assert note == "passed"

    def test_missing_slots_detected(self):
        gold = {"SlotA": 5.0, "SlotB": 10.0}
        pred = {"SlotA": 5.0}  # SlotB missing
        tags, note = self._label("id", "mixed_or_other", [], gold, pred, passed=False)
        assert "missing_value" in tags
        assert "SlotB" in note

    def test_exact_sibling_swap_detected(self):
        gold = {"SlotA": 3.0, "SlotB": 7.0}
        pred = {"SlotA": 7.0, "SlotB": 3.0}
        tags, note = self._label("id", "hard_swapped_quantities", [], gold, pred, passed=False)
        assert "sibling_swap" in tags

    def test_no_swap_when_not_swapped(self):
        gold = {"SlotA": 3.0, "SlotB": 7.0}
        pred = {"SlotA": 3.0, "SlotB": 7.0}
        tags, note = self._label("id", "hard_swapped_quantities", [], gold, pred, passed=True)
        assert note == "passed"

    def test_percent_100x_error_detected(self):
        gold = {"MaxFrac": 0.30}
        pred = {"MaxFrac": 30.0}  # 100× too large
        tags, note = self._label("id", "easy_percent_type", [], gold, pred, passed=False)
        assert "percent_normalization_error" in tags

    def test_lower_upper_reversal_detected(self):
        gold = {"MinUnits": 10.0, "MaxUnits": 50.0}
        pred = {"MinUnits": 50.0, "MaxUnits": 10.0}  # swapped
        tags, note = self._label("id", "easy_bounds_minmax", [], gold, pred, passed=False)
        assert "lower_upper_reversal" in tags

    def test_total_in_coeff_slot_detected(self):
        gold = {"LaborHoursPerProduct": 3.0}
        pred = {"LaborHoursPerProduct": 3000.0}  # 1000× larger
        tags, note = self._label(
            "id", "easy_total_vs_perunit", [], gold, pred, passed=False
        )
        assert "total_to_coeff_confusion" in tags

    def test_initial_tags_preserved(self):
        gold = {"A": 1.0}
        pred = {"A": 2.0}
        tags, note = self._label(
            "id", "hard_wrong_assignment", ["distractor_number"], gold, pred, passed=False
        )
        assert "distractor_number" in tags

    def test_fallback_uncategorized(self):
        # Unusual mismatch with no pattern
        gold = {"Slot": 42.0}
        pred = {"Slot": 99.0}
        tags, note = self._label("id", "mixed_or_other", [], gold, pred, passed=False)
        assert isinstance(tags, list)
        assert isinstance(note, str)


# ---------------------------------------------------------------------------
# AuditRecord
# ---------------------------------------------------------------------------

class TestAuditRecord:
    def test_to_dict_serialisable(self):
        from tools.run_large_failure_audit import AuditRecord
        rec = AuditRecord(
            case_id="test_01",
            source="stress",
            family="hard_wrong_assignment",
            secondary_tags=["sibling_swap", "wrong_entity_family"],
            query="Feed A has 10 protein.",
            mode="full",
            gold_summary="ProteinFeedA=10",
            predicted_summary="ProteinFeedA=10",
            passed=True,
            type_match=1.0,
            exact20=1.0,
            diagnostic_note="passed",
        )
        d = rec.to_dict()
        assert d["case_id"] == "test_01"
        assert d["secondary_tags"] == "sibling_swap;wrong_entity_family"

    def test_to_dict_failed_case(self):
        from tools.run_large_failure_audit import AuditRecord
        rec = AuditRecord(
            case_id="test_02",
            source="group1",
            family="easy_percent_type",
            secondary_tags=["percent_normalization_error"],
            query="At least 40% of production must be A.",
            mode="basic",
            gold_summary="MinFractionA=0.4",
            predicted_summary="MinFractionA=40.0",
            passed=False,
            type_match=0.0,
            exact20=0.0,
            diagnostic_note="percent_x100",
        )
        d = rec.to_dict()
        assert d["passed"] is False
        assert "percent_normalization_error" in d["secondary_tags"]


# ---------------------------------------------------------------------------
# Artifact generation
# ---------------------------------------------------------------------------

class TestArtifactGeneration:
    """Tests that artifact generators produce valid files."""

    def _make_records(self):
        from tools.run_large_failure_audit import AuditRecord
        return [
            AuditRecord(
                case_id=f"case_{i:02d}",
                source="stress",
                family=fam,
                secondary_tags=tags,
                query=f"Query {i}",
                mode=mode,
                gold_summary="A=1",
                predicted_summary="A=1" if ok else "A=2",
                passed=ok,
                type_match=1.0 if ok else 0.0,
                exact20=1.0 if ok else 0.0,
                diagnostic_note="passed" if ok else "mismatch",
            )
            for i, (fam, tags, mode, ok) in enumerate([
                ("hard_wrong_assignment", ["sibling_swap"], "full", True),
                ("hard_swapped_quantities", ["sibling_swap"], "full", False),
                ("easy_percent_type", ["percent_normalization_error"], "full", True),
                ("easy_bounds_minmax", ["lower_upper_reversal"], "full", False),
                ("mixed_or_other", ["distractor_number"], "basic", True),
                ("mixed_or_other", ["distractor_number"], "full", True),
            ])
        ]

    def test_failure_audit_csv_written(self, tmp_path):
        from tools.run_large_failure_audit import _write_failure_audit_csv
        records = self._make_records()
        _write_failure_audit_csv(records, tmp_path)
        p = tmp_path / "failure_audit.csv"
        assert p.exists()
        rows = list(csv.DictReader(p.open()))
        assert len(rows) == len(records)

    def test_group_summary_csv_written(self, tmp_path):
        from tools.run_large_failure_audit import _write_group_summary_csv
        records = self._make_records()
        _write_group_summary_csv(records, tmp_path)
        p = tmp_path / "failure_summary_by_group.csv"
        assert p.exists()
        rows = list(csv.DictReader(p.open()))
        # One row per taxonomy group
        from tools.run_large_failure_audit import TAXONOMY_GROUPS
        assert len(rows) == len(TAXONOMY_GROUPS)

    def test_tag_summary_csv_written(self, tmp_path):
        from tools.run_large_failure_audit import _write_tag_summary_csv
        records = self._make_records()
        _write_tag_summary_csv(records, tmp_path)
        p = tmp_path / "failure_summary_by_tag.csv"
        assert p.exists()
        content = p.read_text()
        assert "secondary_tag" in content

    def test_examples_by_group_md_written(self, tmp_path):
        from tools.run_large_failure_audit import _write_examples_by_group_md
        records = self._make_records()
        _write_examples_by_group_md(records, tmp_path)
        p = tmp_path / "failure_examples_by_group.md"
        assert p.exists()
        content = p.read_text()
        assert "## hard_swapped_quantities" in content

    def test_examples_by_tag_md_written(self, tmp_path):
        from tools.run_large_failure_audit import _write_examples_by_tag_md
        records = self._make_records()
        _write_examples_by_tag_md(records, tmp_path)
        p = tmp_path / "failure_examples_by_tag.md"
        assert p.exists()

    def test_overall_report_md_written(self, tmp_path):
        from tools.run_large_failure_audit import _write_overall_report_md
        records = self._make_records()
        _write_overall_report_md(records, {"stress": 6}, tmp_path)
        p = tmp_path / "overall_audit_report.md"
        assert p.exists()
        content = p.read_text()
        assert "Total runnable tested" in content
        assert "Per-Group Results" in content

    def test_group_summary_csv_counts_correct(self, tmp_path):
        from tools.run_large_failure_audit import _write_group_summary_csv
        records = self._make_records()
        _write_group_summary_csv(records, tmp_path)
        rows = {
            r["group"]: r
            for r in csv.DictReader((tmp_path / "failure_summary_by_group.csv").open())
        }
        # hard_swapped_quantities: 1 full-mode case, 0 passed, 1 failed
        swapped = rows["hard_swapped_quantities"]
        assert int(swapped["total_tested"]) == 1
        assert int(swapped["passed"]) == 0
        assert int(swapped["failed"]) == 1

    def test_all_six_artifacts_present(self, tmp_path):
        from tools.run_large_failure_audit import (
            _write_failure_audit_csv,
            _write_group_summary_csv,
            _write_tag_summary_csv,
            _write_examples_by_group_md,
            _write_examples_by_tag_md,
            _write_overall_report_md,
        )
        records = self._make_records()
        _write_failure_audit_csv(records, tmp_path)
        _write_group_summary_csv(records, tmp_path)
        _write_tag_summary_csv(records, tmp_path)
        _write_examples_by_group_md(records, tmp_path)
        _write_examples_by_tag_md(records, tmp_path)
        _write_overall_report_md(records, {"stress": 6}, tmp_path)
        expected = {
            "failure_audit.csv",
            "failure_summary_by_group.csv",
            "failure_summary_by_tag.csv",
            "failure_examples_by_group.md",
            "failure_examples_by_tag.md",
            "overall_audit_report.md",
        }
        actual = {p.name for p in tmp_path.iterdir()}
        assert expected == actual


# ---------------------------------------------------------------------------
# run_large_failure_audit integration (light)
# ---------------------------------------------------------------------------

class TestRunLargeFailureAuditIntegration:
    """Light integration tests — skip expensive end-to-end grounding in CI."""

    def test_returns_expected_keys(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        result = run_large_failure_audit(out_dir=tmp_path, verbose=False)
        assert "records" in result
        assert "sources" in result
        assert "out_dir" in result

    def test_records_have_all_modes(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        result = run_large_failure_audit(
            out_dir=tmp_path,
            modes=("basic", "full"),
            verbose=False,
        )
        modes_seen = {r.mode for r in result["records"]}
        assert "basic" in modes_seen
        assert "full" in modes_seen

    def test_all_six_artifacts_written(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        run_large_failure_audit(out_dir=tmp_path, verbose=False)
        expected = {
            "failure_audit.csv",
            "failure_summary_by_group.csv",
            "failure_summary_by_tag.csv",
            "failure_examples_by_group.md",
            "failure_examples_by_tag.md",
            "overall_audit_report.md",
        }
        actual = {p.name for p in tmp_path.iterdir()}
        assert expected == actual

    def test_full_mode_pass_rate_high(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        result = run_large_failure_audit(
            out_dir=tmp_path,
            modes=("full",),
            verbose=False,
        )
        full_records = [r for r in result["records"] if r.mode == "full" and r.source != "stress_static"]
        passed = sum(1 for r in full_records if r.passed)
        rate = passed / len(full_records) if full_records else 0.0
        # Full mode should pass at least 80% of runnable cases
        assert rate >= 0.80, f"Full mode pass rate {rate:.2f} is too low"

    def test_group1_group3_perfect_in_full_mode(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        result = run_large_failure_audit(
            out_dir=tmp_path,
            modes=("full",),
            verbose=False,
        )
        targeted = [
            r for r in result["records"]
            if r.mode == "full" and r.source in ("group1", "group3")
        ]
        passed = sum(1 for r in targeted if r.passed)
        assert passed == len(targeted), (
            f"Expected all targeted cases to pass in full mode, "
            f"but {len(targeted) - passed} failed"
        )

    def test_failure_audit_csv_has_correct_columns(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        run_large_failure_audit(out_dir=tmp_path, modes=("full",), verbose=False)
        rows = list(csv.DictReader((tmp_path / "failure_audit.csv").open()))
        assert rows, "failure_audit.csv is empty"
        required_cols = {
            "case_id", "source", "family", "secondary_tags",
            "query", "mode", "gold_summary", "predicted_summary",
            "passed", "type_match", "exact20", "diagnostic_note",
        }
        assert required_cols <= set(rows[0].keys())

    def test_sources_dict_contains_known_sources(self, tmp_path):
        from tools.run_large_failure_audit import run_large_failure_audit
        result = run_large_failure_audit(out_dir=tmp_path, modes=("full",), verbose=False)
        assert "group1" in result["sources"]
        assert "group3" in result["sources"]
        assert "stress" in result["sources"]
