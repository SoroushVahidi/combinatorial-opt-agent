"""
Tests for retrieval baselines: unified interface, ranking shape, runner outputs.
"""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest


def _tiny_catalog() -> list[dict]:
    return [
        {"id": "p1", "name": "Knapsack", "aliases": ["0-1 knapsack"], "description": "Select items with weights and values to maximize value in capacity."},
        {"id": "p2", "name": "Set Cover", "aliases": [], "description": "Choose minimum subsets to cover all elements."},
        {"id": "p3", "name": "Vertex Cover", "aliases": [], "description": "Minimum vertices to cover every edge."},
    ]


def test_bm25_returns_top_k_and_scores():
    """BM25 baseline rank() returns exactly top_k items with (problem_id, score)."""
    from retrieval.baselines import BM25Baseline
    cat = _tiny_catalog()
    bl = BM25Baseline()
    bl.fit(cat)
    out = bl.rank("knapsack problem", top_k=2)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        pid, score = item
        assert pid in ("p1", "p2", "p3")
        assert isinstance(score, (int, float))


def test_tfidf_returns_top_k_and_scores():
    """TF-IDF baseline rank() returns exactly top_k items with (problem_id, score)."""
    from retrieval.baselines import TfidfBaseline
    cat = _tiny_catalog()
    bl = TfidfBaseline()
    bl.fit(cat)
    out = bl.rank("set cover", top_k=2)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        pid, score = item
        assert pid in ("p1", "p2", "p3")
        assert isinstance(score, (int, float))


def test_sbert_returns_top_k_and_scores():
    """SBERT baseline rank() returns exactly top_k items with (problem_id, score)."""
    pytest.importorskip("torch", reason="SBERT requires torch")
    from retrieval.baselines import SBERTBaseline
    cat = _tiny_catalog()
    bl = SBERTBaseline(model_path="sentence-transformers/all-MiniLM-L6-v2")
    try:
        bl.fit(cat)
    except (OSError, RuntimeError, ImportError) as exc:
        pytest.skip(f"Skipping SBERT test: model could not be loaded ({exc})")
    out = bl.rank("knapsack", top_k=2)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        pid, score = item
        assert pid in ("p1", "p2", "p3")
        assert isinstance(score, (int, float))


def test_get_baseline():
    """get_baseline(name) returns correct type for bm25, tfidf, sbert."""
    from retrieval.baselines import get_baseline, BM25Baseline, TfidfBaseline, SBERTBaseline
    assert isinstance(get_baseline("bm25"), BM25Baseline)
    assert isinstance(get_baseline("tfidf"), TfidfBaseline)
    assert isinstance(get_baseline("sbert"), SBERTBaseline)
    with pytest.raises(ValueError, match="Unknown baseline"):
        get_baseline("unknown")


def test_runner_produces_json_and_csv():
    """run_baselines with tiny catalog and eval file produces baselines_*.json and baselines_*.csv."""
    from training.run_baselines import _load_catalog, _generate_eval_instances
    cat = _tiny_catalog()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        catalog_path = tmp / "catalog.json"
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(cat, f)
        eval_path = tmp / "eval.jsonl"
        pairs = _generate_eval_instances(cat, seed=42, num_instances=5, problem_ids=None)
        with open(eval_path, "w", encoding="utf-8") as f:
            for q, pid in pairs:
                f.write(json.dumps({"query": q, "problem_id": pid}) + "\n")
        results_dir = tmp / "results"
        results_dir.mkdir()

        # Run via subprocess to avoid import side effects
        import subprocess
        proc = subprocess.run(
            [
                "python", "-m", "training.run_baselines",
                "--catalog", str(catalog_path),
                "--eval-file", str(eval_path),
                "--baselines", "bm25", "tfidf",
                "--results-dir", str(results_dir),
                "--k", "5",
                "--num", "5",
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)

        json_files = list(results_dir.glob("baselines*.json"))
        csv_files = list(results_dir.glob("baselines*.csv"))
        assert len(json_files) == 1, json_files
        assert len(csv_files) == 1, csv_files
        with open(json_files[0], encoding="utf-8") as f:
            data = json.load(f)
        assert "config" in data and "baselines" in data
        assert "bm25" in data["baselines"] and "tfidf" in data["baselines"]
        assert "P@1" in data["baselines"]["bm25"]
        with open(csv_files[0], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2  # bm25, tfidf
        assert rows[0]["baseline"] in ("bm25", "tfidf")
