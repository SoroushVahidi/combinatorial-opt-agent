"""Tests for the experiment rerun pipeline components.

Covers:
- run_confidence_intervals new method keys and paired comparisons
- run_error_analysis COMPARISON_METHODS list and new method_comparison_table output
- generate_revision_report produces the expected markdown file
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_per_query_csv(tmp_path: Path, variant: str, method: str, rows: list[dict]) -> Path:
    fields = [
        "query_id", "variant", "baseline", "predicted_doc_id", "gold_doc_id",
        "schema_hit", "n_expected_scalar", "n_filled", "param_coverage",
        "type_match", "exact5", "exact20", "key_overlap",
    ]
    p = tmp_path / f"nlp4lp_downstream_per_query_{variant}_{method}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _make_downstream_json(tmp_path: Path, variant: str, method: str, agg: dict) -> Path:
    p = tmp_path / f"nlp4lp_downstream_{variant}_{method}.json"
    data = {"config": {"variant": variant, "baseline": method}, "aggregate": agg}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return p


def _sample_rows(n: int = 10) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "query_id": f"q{i}", "variant": "orig", "baseline": "tfidf",
            "predicted_doc_id": f"d{i}", "gold_doc_id": f"d{i}",
            "schema_hit": 1, "n_expected_scalar": 4, "n_filled": 4,
            "param_coverage": 1.0, "type_match": 0.8,
            "exact5": 0.5, "exact20": 0.5, "key_overlap": 1.0,
        })
    return rows


# ── confidence intervals ──────────────────────────────────────────────────────

class TestCINewMethods:
    """Verify new method families are registered in MAIN_METHODS."""

    def test_gcg_keys_present(self):
        from tools.run_confidence_intervals import MAIN_METHODS
        assert "gcg_local" in MAIN_METHODS
        assert "gcg_pairwise" in MAIN_METHODS
        assert "gcg_full" in MAIN_METHODS

    def test_ral_keys_present(self):
        from tools.run_confidence_intervals import MAIN_METHODS
        assert "ral_basic" in MAIN_METHODS
        assert "ral_ops" in MAIN_METHODS
        assert "ral_semantic" in MAIN_METHODS
        assert "ral_full" in MAIN_METHODS

    def test_aag_keys_present(self):
        from tools.run_confidence_intervals import MAIN_METHODS
        assert "aag_greedy" in MAIN_METHODS
        assert "aag_beam" in MAIN_METHODS
        assert "aag_abstain" in MAIN_METHODS
        assert "aag_full" in MAIN_METHODS

    def test_gcg_maps_to_correct_filenames(self):
        from tools.run_confidence_intervals import MAIN_METHODS
        assert MAIN_METHODS["gcg_local"] == "tfidf_global_compat_local"
        assert MAIN_METHODS["gcg_pairwise"] == "tfidf_global_compat_pairwise"
        assert MAIN_METHODS["gcg_full"] == "tfidf_global_compat_full"

    def test_ral_maps_to_correct_filenames(self):
        from tools.run_confidence_intervals import MAIN_METHODS
        assert MAIN_METHODS["ral_basic"] == "tfidf_relation_aware_basic"
        assert MAIN_METHODS["ral_full"] == "tfidf_relation_aware_full"

    def test_aag_maps_to_correct_filenames(self):
        from tools.run_confidence_intervals import MAIN_METHODS
        assert MAIN_METHODS["aag_beam"] == "tfidf_ambiguity_aware_beam"
        assert MAIN_METHODS["aag_abstain"] == "tfidf_ambiguity_aware_abstain"
        assert MAIN_METHODS["aag_full"] == "tfidf_ambiguity_aware_full"

    def test_new_paired_comparisons_present(self):
        from tools.run_confidence_intervals import PAIRED_COMPARISONS
        labels = [c[0] for c in PAIRED_COMPARISONS]
        assert any("GCG" in lbl for lbl in labels)
        assert any("RAL" in lbl for lbl in labels)
        assert any("AAG" in lbl for lbl in labels)

    def test_paired_comparisons_reference_valid_keys(self):
        from tools.run_confidence_intervals import MAIN_METHODS, PAIRED_COMPARISONS
        for label, method_a, method_b, metric in PAIRED_COMPARISONS:
            assert method_a in MAIN_METHODS, f"Unknown method_A key '{method_a}' in '{label}'"
            assert method_b in MAIN_METHODS, f"Unknown method_B key '{method_b}' in '{label}'"

    def test_ci_runner_handles_new_methods_gracefully(self, tmp_path):
        """CI runner silently skips missing method files (new method may not have data yet)."""
        import importlib
        import tools.run_confidence_intervals as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR
        orig_methods = mod.MAIN_METHODS
        orig_variants = mod.VARIANTS
        orig_comps = mod.PAIRED_COMPARISONS

        try:
            rows = _sample_rows(30)
            _make_per_query_csv(tmp_path, "orig", "tfidf", rows)
            out_dir = tmp_path / "sig_out"
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = out_dir
            # Include both existing tfidf and a new method key that has no file
            mod.MAIN_METHODS = {
                "tfidf": "tfidf",
                "gcg_full": "tfidf_global_compat_full",  # no file in tmp_path
            }
            mod.VARIANTS = ["orig"]
            mod.PAIRED_COMPARISONS = []

            # Should not raise even though gcg_full file is absent
            mod.main(B=20, seed=42)
            assert (out_dir / "confidence_intervals.csv").exists()
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out
            mod.MAIN_METHODS = orig_methods
            mod.VARIANTS = orig_variants
            mod.PAIRED_COMPARISONS = orig_comps


# ── error analysis ─────────────────────────────────────────────────────────

class TestErrorAnalysisNewMethods:
    """Verify COMPARISON_METHODS is updated with new method families."""

    def test_comparison_methods_includes_gcg(self):
        from tools.run_error_analysis import COMPARISON_METHODS
        assert "tfidf_global_compat_local" in COMPARISON_METHODS
        assert "tfidf_global_compat_pairwise" in COMPARISON_METHODS
        assert "tfidf_global_compat_full" in COMPARISON_METHODS

    def test_comparison_methods_includes_ral(self):
        from tools.run_error_analysis import COMPARISON_METHODS
        assert "tfidf_relation_aware_basic" in COMPARISON_METHODS
        assert "tfidf_relation_aware_ops" in COMPARISON_METHODS
        assert "tfidf_relation_aware_semantic" in COMPARISON_METHODS
        assert "tfidf_relation_aware_full" in COMPARISON_METHODS

    def test_comparison_methods_includes_aag(self):
        from tools.run_error_analysis import COMPARISON_METHODS
        assert "tfidf_ambiguity_candidate_greedy" in COMPARISON_METHODS
        assert "tfidf_ambiguity_aware_beam" in COMPARISON_METHODS
        assert "tfidf_ambiguity_aware_abstain" in COMPARISON_METHODS
        assert "tfidf_ambiguity_aware_full" in COMPARISON_METHODS

    def test_comparison_methods_still_includes_baselines(self):
        from tools.run_error_analysis import COMPARISON_METHODS
        assert "tfidf" in COMPARISON_METHODS
        assert "bm25" in COMPARISON_METHODS
        assert "oracle" in COMPARISON_METHODS

    def test_method_comparison_table_written(self, tmp_path):
        """main() produces method_comparison_table.csv when data is present."""
        import tools.run_error_analysis as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR
        orig_eval = mod.EVAL_ORIG
        orig_catalog = mod.CATALOG_PATH
        orig_primary = mod.PRIMARY_METHOD

        try:
            rows = _sample_rows(10)
            _make_per_query_csv(tmp_path, "orig", "tfidf", rows)
            _make_downstream_json(tmp_path, "orig", "tfidf", {
                "schema_R1": 0.9, "param_coverage": 0.87, "type_match": 0.75,
                "instantiation_ready": 0.53, "n": 10,
            })

            out_dir = tmp_path / "err_out"
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = out_dir
            mod.EVAL_ORIG = tmp_path / "nonexistent.jsonl"
            mod.CATALOG_PATH = tmp_path / "nonexistent_catalog.jsonl"
            mod.PRIMARY_METHOD = "tfidf"

            # Patch COMPARISON_METHODS to only check tfidf to avoid missing files
            orig_cmp = mod.COMPARISON_METHODS
            mod.COMPARISON_METHODS = ["tfidf"]

            mod.main()

            cmp_path = out_dir / "method_comparison_table.csv"
            assert cmp_path.exists(), "method_comparison_table.csv should be created"
            with open(cmp_path) as f:
                cmp_rows = list(csv.DictReader(f))
            assert len(cmp_rows) >= 1
            assert "method" in cmp_rows[0]
            assert "InstReady" in cmp_rows[0]
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out
            mod.EVAL_ORIG = orig_eval
            mod.CATALOG_PATH = orig_catalog
            mod.PRIMARY_METHOD = orig_primary
            mod.COMPARISON_METHODS = orig_cmp


# ── revision report generator ─────────────────────────────────────────────

class TestRevisionReportGenerator:
    """Tests for generate_revision_report.py."""

    def test_imports_cleanly(self):
        import tools.generate_revision_report as m
        assert callable(m.main)

    def test_produces_markdown_file(self, tmp_path):
        import tools.generate_revision_report as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_sig = mod.SIG_DIR
        orig_err = mod.ERR_DIR
        orig_ovl = mod.OVL_DIR
        orig_out = mod.OUT_DIR

        try:
            # Create minimal input files
            _make_downstream_json(tmp_path, "orig", "tfidf", {
                "schema_R1": 0.91, "param_coverage": 0.86, "type_match": 0.75,
                "instantiation_ready": 0.53, "n": 10,
            })
            _make_downstream_json(tmp_path, "orig", "oracle", {
                "schema_R1": 1.0, "param_coverage": 0.92, "type_match": 0.80,
                "instantiation_ready": 0.57, "n": 10,
            })
            _make_downstream_json(tmp_path, "orig", "tfidf_relation_aware_basic", {
                "schema_R1": 0.91, "param_coverage": 0.82, "type_match": 0.72,
                "instantiation_ready": 0.50, "n": 10,
            })

            out_dir = tmp_path / "reports"
            mod.DOWNSTREAM_DIR = tmp_path
            mod.SIG_DIR = tmp_path / "sig"
            mod.ERR_DIR = tmp_path / "err"
            mod.OVL_DIR = tmp_path / "ovl"
            mod.OUT_DIR = out_dir

            mod.main()

            md_path = out_dir / "FINAL_REVISION_EXPERIMENT_SUMMARY.md"
            assert md_path.exists()
            content = md_path.read_text(encoding="utf-8")
            assert "Executive Summary" in content
            assert "Downstream Comparison" in content
            assert "Statistical Significance" in content
            assert "Conclusions" in content
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.SIG_DIR = orig_sig
            mod.ERR_DIR = orig_err
            mod.OVL_DIR = orig_ovl
            mod.OUT_DIR = orig_out

    def test_downstream_table_includes_new_methods(self, tmp_path):
        import tools.generate_revision_report as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR

        try:
            for mkey in ["tfidf", "oracle", "tfidf_global_compat_full",
                         "tfidf_relation_aware_basic", "tfidf_ambiguity_aware_beam"]:
                _make_downstream_json(tmp_path, "orig", mkey, {
                    "schema_R1": 0.9, "param_coverage": 0.85, "type_match": 0.75,
                    "instantiation_ready": 0.5, "n": 10,
                })
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = tmp_path / "out"

            table = mod._downstream_table("orig")
            assert "GCG" in table
            assert "RAL" in table
            assert "AAG" in table
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out

    def test_cross_variant_table_uses_all_three_variants(self, tmp_path):
        import tools.generate_revision_report as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR

        try:
            for v in ["orig", "noisy", "short"]:
                for m in ["tfidf", "oracle", "tfidf_acceptance_rerank",
                          "tfidf_global_compat_full", "tfidf_relation_aware_basic",
                          "tfidf_ambiguity_aware_beam"]:
                    _make_downstream_json(tmp_path, v, m, {
                        "schema_R1": 0.9, "param_coverage": 0.85, "type_match": 0.75,
                        "instantiation_ready": 0.5, "n": 10,
                    })
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = tmp_path / "out"

            table = mod._cross_variant_table()
            assert "orig" in table
            assert "noisy" in table
            assert "short" in table
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out
