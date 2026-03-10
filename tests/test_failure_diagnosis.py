"""
Tests for failure-mode diagnosis in training/metrics.py.

The system distinguishes between two shortcomings when a query is not fully
answered:

  retrieval_miss        – the correct problem was not found in the top-k results
                          at all (the model cannot connect the query to the right
                          catalog entry).

  param_extraction_needed – the correct problem *was* retrieved but the query
                          contains specific numeric parameter values (e.g.
                          capacities, weights, costs) that would require a
                          follow-up extraction step to instantiate the symbolic
                          formulation template.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# has_numeric_parameters
# ---------------------------------------------------------------------------

class TestHasNumericParameters:
    """has_numeric_parameters returns True iff the query contains standalone numbers."""

    def test_query_with_integer_returns_true(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters("I have 5 items") is True

    def test_query_with_decimal_returns_true(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters("capacity is 12.5 kg") is True

    def test_query_with_multiple_numbers_returns_true(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters("weights 3, 4, 2 values 10 8 6 capacity 7") is True

    def test_conceptual_query_returns_false(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters("minimize total shipping cost from warehouses to customers") is False

    def test_empty_string_returns_false(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters("") is False

    def test_none_returns_false(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters(None) is False

    def test_non_string_returns_false(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters(123) is False

    def test_letters_only_returns_false(self):
        from training.metrics import has_numeric_parameters
        assert has_numeric_parameters("knapsack optimization problem formulation") is False


# ---------------------------------------------------------------------------
# classify_failure_mode
# ---------------------------------------------------------------------------

class TestClassifyFailureMode:
    """classify_failure_mode returns the correct label for each scenario."""

    def test_success_when_in_top_k_no_numbers(self):
        from training.metrics import classify_failure_mode
        # Correct problem at rank 1, no specific numbers in query
        result = classify_failure_mode(
            query="minimize total shipping cost from warehouses to customers",
            ranked_names=["Facility Location", "Set Cover"],
            expected_name="Facility Location",
            k=3,
        )
        assert result == "success"

    def test_param_extraction_needed_when_in_top_k_with_numbers(self):
        from training.metrics import classify_failure_mode
        # Correct problem retrieved but query has concrete numeric values
        result = classify_failure_mode(
            query="I have 5 items with weights 3 4 2 and values 10 8 6 capacity 7",
            ranked_names=["Knapsack", "Set Cover"],
            expected_name="Knapsack",
            k=3,
        )
        assert result == "param_extraction_needed"

    def test_retrieval_miss_when_not_in_top_k(self):
        from training.metrics import classify_failure_mode
        # Correct problem not in top-k at all
        result = classify_failure_mode(
            query="minimize total shipping cost from warehouses to customers",
            ranked_names=["Knapsack", "TSP"],
            expected_name="Facility Location",
            k=3,
        )
        assert result == "retrieval_miss"

    def test_retrieval_miss_takes_precedence_over_numbers_in_query(self):
        from training.metrics import classify_failure_mode
        # Query has numbers but correct problem not retrieved → retrieval_miss
        result = classify_failure_mode(
            query="I have 5 items with weights 3 4 2 and values 10 8 6 capacity 7",
            ranked_names=["TSP", "Set Cover"],
            expected_name="Knapsack",
            k=3,
        )
        assert result == "retrieval_miss"

    def test_success_when_correct_at_boundary_k(self):
        from training.metrics import classify_failure_mode
        # Correct problem exactly at position k (1-indexed == len[:k])
        result = classify_failure_mode(
            query="route vehicles to minimize distance",
            ranked_names=["TSP", "VRP", "Knapsack"],
            expected_name="Knapsack",
            k=3,
        )
        assert result == "success"

    def test_retrieval_miss_when_correct_just_outside_k(self):
        from training.metrics import classify_failure_mode
        # Correct problem at position k+1 — just outside the cutoff
        result = classify_failure_mode(
            query="route vehicles to minimize distance",
            ranked_names=["TSP", "VRP", "Knapsack", "Set Cover"],
            expected_name="Set Cover",
            k=3,
        )
        assert result == "retrieval_miss"

    def test_empty_expected_name_returns_success(self):
        from training.metrics import classify_failure_mode
        # Edge-case: unknown expected name — treat as success (no ground truth)
        result = classify_failure_mode(
            query="some query",
            ranked_names=["Knapsack"],
            expected_name="",
            k=5,
        )
        assert result == "success"


# ---------------------------------------------------------------------------
# diagnose_failures
# ---------------------------------------------------------------------------

class TestDiagnoseFailures:
    """diagnose_failures returns correct aggregate counts and fractions."""

    def _make_instances(self):
        """Five instances: 2 success, 2 retrieval_miss, 1 param_extraction_needed."""
        return [
            # success – correct at rank 1, no numbers
            ("minimize shipping cost", ["Facility Location", "TSP"], "Facility Location"),
            # success – correct at rank 2, no numbers
            ("cover all elements with minimum sets", ["Set Cover", "Knapsack"], "Set Cover"),
            # retrieval_miss – correct problem not in top-2
            ("minimize shipping cost", ["Knapsack", "TSP"], "Facility Location"),
            # retrieval_miss – correct problem not in top-2
            ("route vehicles", ["Set Cover", "Facility Location"], "TSP"),
            # param_extraction_needed – correct at rank 1 but query has numbers
            ("5 items weights 3 4 2 capacity 7", ["Knapsack", "Set Cover"], "Knapsack"),
        ]

    def test_counts_are_correct(self):
        from training.metrics import diagnose_failures
        result = diagnose_failures(self._make_instances(), k=2)
        assert result["n_total"] == 5
        assert result["n_success"] == 2
        assert result["n_retrieval_miss"] == 2
        assert result["n_param_extraction_needed"] == 1

    def test_fractions_sum_correctly(self):
        from training.metrics import diagnose_failures
        result = diagnose_failures(self._make_instances(), k=2)
        total_frac = result["frac_retrieval_miss"] + result["frac_param_extraction_needed"]
        # 2/5 + 1/5 = 0.6
        assert abs(total_frac - 0.6) < 1e-9

    def test_frac_retrieval_miss(self):
        from training.metrics import diagnose_failures
        result = diagnose_failures(self._make_instances(), k=2)
        assert abs(result["frac_retrieval_miss"] - 0.4) < 1e-9

    def test_frac_param_extraction_needed(self):
        from training.metrics import diagnose_failures
        result = diagnose_failures(self._make_instances(), k=2)
        assert abs(result["frac_param_extraction_needed"] - 0.2) < 1e-9

    def test_empty_instances_returns_zero_counts(self):
        from training.metrics import diagnose_failures
        result = diagnose_failures([], k=10)
        assert result["n_total"] == 0
        assert result["n_success"] == 0
        assert result["n_retrieval_miss"] == 0
        assert result["n_param_extraction_needed"] == 0
        assert result["frac_retrieval_miss"] == 0.0
        assert result["frac_param_extraction_needed"] == 0.0

    def test_all_success_no_numbers(self):
        from training.metrics import diagnose_failures
        instances = [
            ("minimize cost", ["Facility Location"], "Facility Location"),
            ("cover all elements", ["Set Cover"], "Set Cover"),
        ]
        result = diagnose_failures(instances, k=1)
        assert result["n_success"] == 2
        assert result["n_retrieval_miss"] == 0
        assert result["n_param_extraction_needed"] == 0

    def test_all_retrieval_miss(self):
        from training.metrics import diagnose_failures
        instances = [
            ("minimize cost", ["TSP", "Knapsack"], "Facility Location"),
            ("cover all elements", ["TSP", "Knapsack"], "Set Cover"),
        ]
        result = diagnose_failures(instances, k=2)
        assert result["n_retrieval_miss"] == 2
        assert result["frac_retrieval_miss"] == 1.0

    def test_result_keys_present(self):
        from training.metrics import diagnose_failures
        result = diagnose_failures([], k=5)
        for key in (
            "n_total",
            "n_success",
            "n_retrieval_miss",
            "n_param_extraction_needed",
            "frac_retrieval_miss",
            "frac_param_extraction_needed",
        ):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Integration: run_baselines JSON output includes failure_modes
# ---------------------------------------------------------------------------

class TestRunBaselinesIncludesFailureModes:
    """run_baselines subprocess test: output JSON must contain failure_modes."""

    def test_failure_modes_in_json_output(self, tmp_path):
        import json
        import subprocess
        from pathlib import Path

        cat = [
            {
                "id": "p1",
                "name": "Knapsack",
                "aliases": ["0-1 knapsack"],
                "description": "Select items with weights and values to maximize value in capacity.",
            },
            {
                "id": "p2",
                "name": "Set Cover",
                "aliases": [],
                "description": "Choose minimum subsets to cover all elements.",
            },
            {
                "id": "p3",
                "name": "Vertex Cover",
                "aliases": [],
                "description": "Minimum vertices to cover every edge.",
            },
        ]

        catalog_path = tmp_path / "catalog.json"
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(cat, f)

        # Build a small eval file with both conceptual and numeric queries
        eval_path = tmp_path / "eval.jsonl"
        eval_lines = [
            json.dumps({"query": "maximize value within weight capacity", "problem_id": "p1"}),
            json.dumps({"query": "5 items weights 3 4 2 capacity 7", "problem_id": "p1"}),
            json.dumps({"query": "cover all elements with fewest subsets", "problem_id": "p2"}),
        ]
        eval_path.write_text("\n".join(eval_lines) + "\n", encoding="utf-8")

        results_dir = tmp_path / "results"
        results_dir.mkdir()

        proc = subprocess.run(
            [
                "python", "-m", "training.run_baselines",
                "--catalog", str(catalog_path),
                "--eval-file", str(eval_path),
                "--baselines", "bm25",
                "--results-dir", str(results_dir),
                "--k", "3",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)

        json_files = list(results_dir.glob("baselines*.json"))
        assert len(json_files) == 1

        with open(json_files[0], encoding="utf-8") as fh:
            data = json.load(fh)

        assert "failure_modes" in data, "output JSON must contain 'failure_modes'"
        assert "bm25" in data["failure_modes"]
        fm = data["failure_modes"]["bm25"]
        for key in (
            "n_total",
            "n_success",
            "n_retrieval_miss",
            "n_param_extraction_needed",
            "frac_retrieval_miss",
            "frac_param_extraction_needed",
        ):
            assert key in fm, f"Missing key in failure_modes['bm25']: {key}"
        assert fm["n_total"] == 3
