"""
Tests for the three reviewer-response analysis tools (1B, 1C, 1D).

Tests run offline using tiny in-memory fixtures. No network, no torch.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ────────────────────────────────────────────────────────────────────────────

def _make_per_query_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a per-query CSV with the same schema as downstream_postfix files."""
    p = tmp_path / "per_query.csv"
    fields = [
        "query_id", "variant", "baseline", "predicted_doc_id", "gold_doc_id",
        "schema_hit", "n_expected_scalar", "n_filled", "param_coverage",
        "type_match", "exact5", "exact20", "key_overlap",
    ]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _sample_rows() -> list[dict]:
    return [
        {
            "query_id": "q0", "variant": "orig", "baseline": "tfidf",
            "predicted_doc_id": "d0", "gold_doc_id": "d0",
            "schema_hit": 1, "n_expected_scalar": 4, "n_filled": 4,
            "param_coverage": 1.0, "type_match": 1.0,
            "exact5": 0.5, "exact20": 0.5, "key_overlap": 1.0,
        },
        {
            "query_id": "q1", "variant": "orig", "baseline": "tfidf",
            "predicted_doc_id": "d1", "gold_doc_id": "d1",
            "schema_hit": 1, "n_expected_scalar": 2, "n_filled": 2,
            "param_coverage": 0.5, "type_match": 0.5,
            "exact5": 0.0, "exact20": 0.0, "key_overlap": 0.5,
        },
        {
            "query_id": "q2", "variant": "orig", "baseline": "tfidf",
            "predicted_doc_id": "wrong_d", "gold_doc_id": "d2",
            "schema_hit": 0, "n_expected_scalar": 3, "n_filled": 0,
            "param_coverage": 0.0, "type_match": 0.0,
            "exact5": 0.0, "exact20": 0.0, "key_overlap": 0.0,
        },
        {
            "query_id": "q3", "variant": "orig", "baseline": "tfidf",
            "predicted_doc_id": "d3", "gold_doc_id": "d3",
            "schema_hit": 1, "n_expected_scalar": 5, "n_filled": 4,
            "param_coverage": 0.8, "type_match": 0.8,
            "exact5": 0.25, "exact20": 0.5, "key_overlap": 0.8,
        },
    ]


# ────────────────────────────────────────────────────────────────────────────
# Part 1B: bootstrap CI and paired significance
# ────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_perfect_metric(self):
        """CI around 1.0 should be [1.0, 1.0]."""
        from tools.run_confidence_intervals import bootstrap_ci
        vals = [1.0] * 50
        obs, lo, hi = bootstrap_ci(vals, B=200, seed=42)
        assert obs == pytest.approx(1.0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_zero_metric(self):
        """CI around 0.0 should be [0.0, 0.0]."""
        from tools.run_confidence_intervals import bootstrap_ci
        vals = [0.0] * 50
        obs, lo, hi = bootstrap_ci(vals, B=200, seed=42)
        assert obs == pytest.approx(0.0)
        assert hi == pytest.approx(0.0)

    def test_mixed_binary(self):
        """CI on mixed binary values: mean should be ~0.5, CI should bracket it."""
        from tools.run_confidence_intervals import bootstrap_ci
        vals = [1.0, 0.0] * 50
        obs, lo, hi = bootstrap_ci(vals, B=500, seed=42)
        assert obs == pytest.approx(0.5)
        assert lo < 0.5 < hi

    def test_empty_returns_nan(self):
        import math
        from tools.run_confidence_intervals import bootstrap_ci
        obs, lo, hi = bootstrap_ci([], B=100, seed=42)
        assert math.isnan(obs)

    def test_reproducible(self):
        """Same seed → same results."""
        from tools.run_confidence_intervals import bootstrap_ci
        vals = [float(i % 2) for i in range(40)]
        r1 = bootstrap_ci(vals, B=200, seed=7)
        r2 = bootstrap_ci(vals, B=200, seed=7)
        assert r1 == r2

    def test_different_seeds_differ(self):
        """Different seeds → (usually) different bootstrap samples, so CIs differ."""
        from tools.run_confidence_intervals import bootstrap_ci
        vals = [float(i % 3) / 2 for i in range(60)]
        r1 = bootstrap_ci(vals, B=200, seed=1)
        r2 = bootstrap_ci(vals, B=200, seed=99)
        # Observed means must be equal
        assert r1[0] == r2[0]
        # CIs should differ (bootstrap is stochastic w.r.t. seed)
        assert r1[1:] != r2[1:]


class TestPairedBootstrap:
    def test_identical_distributions(self):
        """Paired test on identical lists: obs_diff=0, CI contains 0, not significant."""
        from tools.run_confidence_intervals import paired_bootstrap_test
        vals = [float(i % 2) for i in range(50)]
        diff, lo, hi, p = paired_bootstrap_test(vals, vals, B=500, seed=42)
        assert diff == pytest.approx(0.0)
        assert lo <= 0 <= hi  # CI brackets zero → not significant

    def test_clear_difference(self):
        """A consistently better A should give a small p-value."""
        from tools.run_confidence_intervals import paired_bootstrap_test
        a = [1.0] * 60
        b = [0.0] * 60
        diff, lo, hi, p = paired_bootstrap_test(a, b, B=500, seed=42)
        assert diff == pytest.approx(1.0)
        assert lo > 0
        assert hi > 0
        assert p < 0.05

    def test_empty_returns_nan(self):
        import math
        from tools.run_confidence_intervals import paired_bootstrap_test
        diff, lo, hi, p = paired_bootstrap_test([], [], B=100, seed=42)
        assert math.isnan(diff)

    def test_p_in_range(self):
        """p-value must be in [0, 1]."""
        from tools.run_confidence_intervals import paired_bootstrap_test
        import random
        rng = random.Random(42)
        a = [rng.random() for _ in range(40)]
        b = [rng.random() for _ in range(40)]
        _, _, _, p = paired_bootstrap_test(a, b, B=200, seed=42)
        assert 0.0 <= p <= 1.0


class TestCIRunner:
    def test_ci_runner_produces_files(self, tmp_path):
        """run_confidence_intervals.main() writes the three expected files."""
        # Patch DOWNSTREAM_DIR and OUT_DIR via monkeypatching the module
        import importlib
        import tools.run_confidence_intervals as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR
        try:
            # Create a tiny per-query CSV for one method+variant
            rows = _sample_rows()
            for variant in ["orig"]:
                for method_suffix in ["tfidf"]:
                    fname = f"nlp4lp_downstream_per_query_{variant}_{method_suffix}.csv"
                    _make_per_query_csv(tmp_path, rows)
                    (tmp_path / fname).unlink(missing_ok=True)
                    src = tmp_path / "per_query.csv"
                    dst = tmp_path / fname
                    src.rename(dst)
            out_dir = tmp_path / "sig_out"
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = out_dir
            # Restrict to only tfidf so we don't look for missing files
            orig_methods = mod.MAIN_METHODS
            mod.MAIN_METHODS = {"tfidf": "tfidf"}
            orig_variants = mod.VARIANTS
            mod.VARIANTS = ["orig"]
            orig_comps = mod.PAIRED_COMPARISONS
            mod.PAIRED_COMPARISONS = []

            mod.main(B=50, seed=42)

            assert (out_dir / "confidence_intervals.csv").exists()
            assert (out_dir / "paired_significance.csv").exists()
            assert (out_dir / "SIGNIFICANCE_SUMMARY.md").exists()

            with open(out_dir / "confidence_intervals.csv") as f:
                ci_rows = list(csv.DictReader(f))
            assert len(ci_rows) > 0
            assert "observed" in ci_rows[0]
            assert "ci_lo_95" in ci_rows[0]
            assert "ci_hi_95" in ci_rows[0]
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out
            mod.MAIN_METHODS = orig_methods
            mod.VARIANTS = orig_variants
            mod.PAIRED_COMPARISONS = orig_comps


# ────────────────────────────────────────────────────────────────────────────
# Part 1C: error analysis
# ────────────────────────────────────────────────────────────────────────────

class TestBuildDiagnostics:
    def test_inst_ready_logic(self):
        """inst_ready=1 iff coverage>=0.8 AND type_match>=0.8."""
        from tools.run_error_analysis import _build_diagnostics
        rows = [
            {"query_id": "a", "schema_hit": "1", "n_expected_scalar": "4", "n_filled": "4",
             "param_coverage": "1.0", "type_match": "1.0", "exact20": "0.5", "key_overlap": "1.0"},
            {"query_id": "b", "schema_hit": "1", "n_expected_scalar": "4", "n_filled": "3",
             "param_coverage": "0.75", "type_match": "1.0", "exact20": "0.0", "key_overlap": "0.5"},
            {"query_id": "c", "schema_hit": "0", "n_expected_scalar": "2", "n_filled": "0",
             "param_coverage": "0.0", "type_match": "0.0", "exact20": "", "key_overlap": "0.0"},
        ]
        diags = _build_diagnostics(rows, {}, "tfidf", "orig")
        assert diags[0]["inst_ready"] == 1  # cov=1, tm=1
        assert diags[1]["inst_ready"] == 0  # cov=0.75 < 0.8
        assert diags[2]["inst_ready"] == 0  # schema miss

    def test_ambiguity_buckets(self):
        """Ambiguity bucket is assigned from numeric mention count."""
        from tools.run_error_analysis import _build_diagnostics, _count_numeric_mentions
        rows = [
            {"query_id": "q1", "schema_hit": "1", "n_expected_scalar": "2", "n_filled": "2",
             "param_coverage": "1.0", "type_match": "1.0", "exact20": "0.5", "key_overlap": "1.0"},
        ]
        # 1 mention → low; 4 → medium; 8 → high
        assert _count_numeric_mentions("buy 1 unit") == 1
        assert _count_numeric_mentions("a 1 b 2 c 3 d 4 e") == 4
        assert _count_numeric_mentions("1 2 3 4 5 6 7 8") == 8

        qt = {"q1": "invest 100 dollars"}
        diags = _build_diagnostics(rows, qt, "tfidf", "orig")
        assert diags[0]["ambiguity_bucket"] in ("low", "medium", "high")

    def test_slot_count_buckets(self):
        """Slot count bucket groups n_expected_scalar correctly."""
        from tools.run_error_analysis import _build_diagnostics
        cases = [
            ("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("5", "4+"),
        ]
        for n_exp, expected_bucket in cases:
            rows = [{"query_id": "x", "schema_hit": "1", "n_expected_scalar": n_exp,
                     "n_filled": "0", "param_coverage": "0.0", "type_match": "0.0",
                     "exact20": "", "key_overlap": "0.0"}]
            diags = _build_diagnostics(rows, {}, "tfidf", "orig")
            assert diags[0]["slot_count_bucket"] == expected_bucket

    def test_schema_hit_miss_breakdown(self):
        """Hit group should have higher InstReady than miss group."""
        from tools.run_error_analysis import _build_diagnostics, schema_hit_miss_breakdown
        rows = [
            {"query_id": "h", "schema_hit": "1", "n_expected_scalar": "3", "n_filled": "3",
             "param_coverage": "1.0", "type_match": "1.0", "exact20": "0.5", "key_overlap": "1.0"},
            {"query_id": "m", "schema_hit": "0", "n_expected_scalar": "3", "n_filled": "0",
             "param_coverage": "0.0", "type_match": "0.0", "exact20": "", "key_overlap": "0.0"},
        ]
        diags = _build_diagnostics(rows, {}, "tfidf", "orig")
        breakdown = schema_hit_miss_breakdown(diags)
        hit_row = next(r for r in breakdown if "hit" in r["group"] and "miss" not in r["group"])
        miss_row = next(r for r in breakdown if "miss" in r["group"])
        assert float(hit_row["InstReady_rate"]) > float(miss_row["InstReady_rate"])

    def test_error_analysis_runner(self, tmp_path):
        """run_error_analysis.main() writes expected output files."""
        import tools.run_error_analysis as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR
        orig_eval = mod.EVAL_ORIG
        orig_catalog = mod.CATALOG_PATH
        orig_method = mod.PRIMARY_METHOD
        orig_variant = mod.PRIMARY_VARIANT

        try:
            rows = _sample_rows()
            per_q = tmp_path / f"nlp4lp_downstream_per_query_orig_tfidf.csv"
            _make_per_query_csv(tmp_path, rows)
            (tmp_path / "per_query.csv").rename(per_q)

            # Write tiny eval and catalog
            eval_path = tmp_path / "eval.jsonl"
            with open(eval_path, "w") as f:
                for i in range(4):
                    f.write(json.dumps({"query_id": f"q{i}", "query": f"buy {i+1} items", "relevant_doc_id": f"d{i}"}) + "\n")

            cat_path = tmp_path / "catalog.jsonl"
            with open(cat_path, "w") as f:
                for i in range(4):
                    f.write(json.dumps({"doc_id": f"d{i}", "text": f"Maximize item{i} profit"}) + "\n")

            out_dir = tmp_path / "err_out"
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = out_dir
            mod.EVAL_ORIG = eval_path
            mod.CATALOG_PATH = cat_path

            mod.main()

            assert (out_dir / "per_instance_diagnostics.csv").exists()
            assert (out_dir / "schema_hit_miss_breakdown.csv").exists()
            assert (out_dir / "slot_count_breakdown.csv").exists()
            assert (out_dir / "ambiguity_breakdown.csv").exists()
            assert (out_dir / "ERROR_EXAMPLES.md").exists()

            with open(out_dir / "per_instance_diagnostics.csv") as f:
                diag_rows = list(csv.DictReader(f))
            assert len(diag_rows) == 4
            assert "inst_ready" in diag_rows[0]
            assert "slot_count_bucket" in diag_rows[0]
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out
            mod.EVAL_ORIG = orig_eval
            mod.CATALOG_PATH = orig_catalog
            mod.PRIMARY_METHOD = orig_method
            mod.PRIMARY_VARIANT = orig_variant


# ────────────────────────────────────────────────────────────────────────────
# Part 1D: overlap analysis
# ────────────────────────────────────────────────────────────────────────────

class TestTokenization:
    def test_base_tokenize(self):
        from tools.run_overlap_analysis import _tokenize
        assert _tokenize("Hello World") == ["hello", "world"]
        assert _tokenize("buy 3 items for $100") == ["buy", "items", "for"]

    def test_no_numbers(self):
        from tools.run_overlap_analysis import _tokenize_no_numbers
        toks = _tokenize_no_numbers("buy 100 items")
        assert "100" not in toks
        assert "buy" in toks

    def test_no_stopwords(self):
        from tools.run_overlap_analysis import _tokenize_no_stopwords
        toks = _tokenize_no_stopwords("buy the items for a profit")
        assert "the" not in toks
        assert "a" not in toks
        assert "buy" in toks
        assert "items" in toks

    def test_jaccard_identical(self):
        from tools.run_overlap_analysis import _token_jaccard
        toks = ["a", "b", "c"]
        assert _token_jaccard(toks, toks) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        from tools.run_overlap_analysis import _token_jaccard
        assert _token_jaccard(["a", "b"], ["c", "d"]) == pytest.approx(0.0)

    def test_jaccard_partial(self):
        from tools.run_overlap_analysis import _token_jaccard
        # shared = {a}, union = {a,b,c}
        j = _token_jaccard(["a", "b"], ["a", "c"])
        assert j == pytest.approx(1 / 3)

    def test_unigram_overlap_ratio(self):
        from tools.run_overlap_analysis import _unigram_overlap_ratio
        # "a" is in doc, "b" is not
        ratio = _unigram_overlap_ratio(["a", "b"], ["a", "c"])
        assert ratio == pytest.approx(0.5)


class TestOverlapRow:
    def test_compute_overlap_row_structure(self):
        from tools.run_overlap_analysis import compute_overlap_row
        row = compute_overlap_row(
            "q0",
            "buy 5 units for $200",
            "MaxUnits Cost Maximize",
            schema_hit=1,
        )
        assert "jaccard_base" in row
        assert "overlap_bucket" in row
        assert row["overlap_bucket"] in ("low", "medium", "high")
        assert row["schema_hit"] == 1

    def test_overlap_bucket_thresholds(self):
        from tools.run_overlap_analysis import compute_overlap_row
        # Ensure bucket assignment matches code's documented thresholds
        # jaccard < 0.05 → low
        low = compute_overlap_row("q", "aaa bbb ccc", "xxx yyy zzz", 0)
        assert low["overlap_bucket"] == "low"
        # jaccard = 1.0 → high
        high = compute_overlap_row("q", "abc def ghi", "abc def ghi", 1)
        assert high["overlap_bucket"] == "high"


class TestOverlapRunner:
    def test_overlap_runner_produces_files(self, tmp_path):
        """run_overlap_analysis.main() writes expected output files."""
        import tools.run_overlap_analysis as mod

        orig_downstream = mod.DOWNSTREAM_DIR
        orig_out = mod.OUT_DIR
        orig_eval = mod.EVAL_ORIG
        orig_catalog = mod.CATALOG_PATH

        try:
            # Write tiny eval JSONL
            eval_path = tmp_path / "eval.jsonl"
            with open(eval_path, "w") as f:
                for i in range(5):
                    f.write(json.dumps({
                        "query_id": f"q{i}",
                        "query": f"maximize profit with {i+1} units and budget of {(i+1)*100}",
                        "relevant_doc_id": f"d{i}",
                    }) + "\n")

            # Write catalog JSONL
            cat_path = tmp_path / "catalog.jsonl"
            with open(cat_path, "w") as f:
                for i in range(5):
                    f.write(json.dumps({"doc_id": f"d{i}", "text": f"MaxUnits_{i} Budget_{i} Maximize profit"}) + "\n")

            # Write per-query CSV for schema_hit loading
            rows = [
                {"query_id": f"q{i}", "variant": "orig", "baseline": "tfidf",
                 "predicted_doc_id": f"d{i}", "gold_doc_id": f"d{i}",
                 "schema_hit": 1, "n_expected_scalar": 2, "n_filled": 2,
                 "param_coverage": 1.0, "type_match": 1.0,
                 "exact5": 0.5, "exact20": 0.5, "key_overlap": 1.0}
                for i in range(5)
            ]
            per_q = tmp_path / "nlp4lp_downstream_per_query_orig_tfidf.csv"
            _make_per_query_csv(tmp_path, rows)
            (tmp_path / "per_query.csv").rename(per_q)

            out_dir = tmp_path / "overlap_out"
            mod.DOWNSTREAM_DIR = tmp_path
            mod.OUT_DIR = out_dir
            mod.EVAL_ORIG = eval_path
            mod.CATALOG_PATH = cat_path

            mod.main()

            assert (out_dir / "lexical_overlap_stats.csv").exists()
            assert (out_dir / "overlap_stratified_retrieval.csv").exists()
            assert (out_dir / "retrieval_overlap_ablation.csv").exists()
            assert (out_dir / "OVERLAP_ANALYSIS.md").exists()

            with open(out_dir / "lexical_overlap_stats.csv") as f:
                stats = list(csv.DictReader(f))
            assert len(stats) == 5
            assert "jaccard_base" in stats[0]
            assert "overlap_bucket" in stats[0]
        finally:
            mod.DOWNSTREAM_DIR = orig_downstream
            mod.OUT_DIR = orig_out
            mod.EVAL_ORIG = orig_eval
            mod.CATALOG_PATH = orig_catalog
