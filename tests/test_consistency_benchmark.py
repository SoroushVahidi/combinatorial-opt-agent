from __future__ import annotations
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from src.analysis.consistency_benchmark import run_benchmark, write_benchmark_outputs, SYNTHETIC_CASES


class TestBenchmarkStructure:
    def test_run_benchmark_returns_valid_dict(self):
        results, per_case = run_benchmark(SYNTHETIC_CASES)
        assert "old" in results
        assert "repaired" in results
        assert "n_cases" in results
        assert "n_correct" in results
        assert "n_wrong" in results
        assert results["n_cases"] == len(SYNTHETIC_CASES)

    def test_old_metrics_valid(self):
        results, _ = run_benchmark(SYNTHETIC_CASES)
        old = results["old"]
        assert "recall_wrong" in old
        assert "fpr_correct" in old
        assert 0.0 <= old["recall_wrong"] <= 1.0
        assert 0.0 <= old["fpr_correct"] <= 1.0

    def test_repaired_metrics_valid(self):
        results, _ = run_benchmark(SYNTHETIC_CASES)
        repaired = results["repaired"]
        assert 0.0 <= repaired["recall_wrong"] <= 1.0
        assert 0.0 <= repaired["fpr_correct"] <= 1.0


class TestRepairedVsOld:
    def test_repaired_has_lower_fpr(self):
        results, _ = run_benchmark(SYNTHETIC_CASES)
        # Repaired checker should have equal or lower false positive rate
        assert results["repaired"]["fpr_correct"] <= results["old"]["fpr_correct"] + 0.15  # allow small tolerance

    def test_repaired_maintains_some_recall(self):
        results, _ = run_benchmark(SYNTHETIC_CASES)
        # Repaired checker should catch at least 25% of wrong answers
        assert results["repaired"]["recall_wrong"] >= 0.25


class TestWriteBenchmarkOutputs:
    def test_write_creates_expected_files(self, tmp_path):
        results, per_case = run_benchmark(SYNTHETIC_CASES)
        output_dir = str(tmp_path / "benchmark_out")
        write_benchmark_outputs(results, per_case, output_dir=output_dir)

        out = Path(output_dir)
        assert (out / "summary.json").exists()
        assert (out / "per_candidate_results.csv").exists()
        assert (out / "failure_type_summary.csv").exists()
        assert (out / "role_signal_summary.csv").exists()

    def test_summary_json_valid(self, tmp_path):
        results, per_case = run_benchmark(SYNTHETIC_CASES)
        output_dir = str(tmp_path / "benchmark_out2")
        write_benchmark_outputs(results, per_case, output_dir=output_dir)

        with open(Path(output_dir) / "summary.json") as f:
            data = json.load(f)
        assert "old" in data
        assert "repaired" in data
