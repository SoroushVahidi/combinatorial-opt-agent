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


@pytest.mark.requires_network
def test_sbert_returns_top_k_and_scores():
    """SBERT baseline rank() returns exactly top_k items with (problem_id, score)."""
    pytest.importorskip("torch", reason="SBERT requires torch")
    from retrieval.baselines import SBERTBaseline
    cat = _tiny_catalog()
    bl = SBERTBaseline(model_path="sentence-transformers/all-MiniLM-L6-v2")
    bl.fit(cat)
    out = bl.rank("knapsack", top_k=2)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        pid, score = item
        assert pid in ("p1", "p2", "p3")
        assert isinstance(score, (int, float))


def test_get_baseline():
    """get_baseline(name) returns correct type for bm25, tfidf, sbert, e5, bge."""
    from retrieval.baselines import get_baseline, BM25Baseline, TfidfBaseline, SBERTBaseline, E5Baseline, BGEBaseline
    assert isinstance(get_baseline("bm25"), BM25Baseline)
    assert isinstance(get_baseline("tfidf"), TfidfBaseline)
    assert isinstance(get_baseline("sbert"), SBERTBaseline)
    assert isinstance(get_baseline("e5"), E5Baseline)
    assert isinstance(get_baseline("bge"), BGEBaseline)
    with pytest.raises(ValueError, match="Unknown baseline"):
        get_baseline("unknown")


@pytest.mark.requires_network
def test_e5_returns_top_k_and_scores():
    """E5 baseline rank() returns exactly top_k items with (problem_id, score)."""
    pytest.importorskip("torch", reason="E5 requires torch")
    from retrieval.baselines import E5Baseline
    cat = _tiny_catalog()
    bl = E5Baseline(model_name="intfloat/e5-base-v2")
    bl.fit(cat)
    out = bl.rank("knapsack weight capacity", top_k=2)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        pid, score = item
        assert pid in ("p1", "p2", "p3")
        assert isinstance(score, (int, float))


@pytest.mark.requires_network
def test_bge_returns_top_k_and_scores():
    """BGE baseline rank() returns exactly top_k items with (problem_id, score)."""
    pytest.importorskip("torch", reason="BGE requires torch")
    from retrieval.baselines import BGEBaseline
    cat = _tiny_catalog()
    bl = BGEBaseline(model_name="BAAI/bge-large-en-v1.5")
    bl.fit(cat)
    out = bl.rank("cover all elements with minimum subsets", top_k=2)
    assert len(out) == 2
    for item in out:
        assert isinstance(item, tuple) and len(item) == 2
        pid, score = item
        assert pid in ("p1", "p2", "p3")
        assert isinstance(score, (int, float))


def test_e5_prefix_applied():
    """E5Baseline correctly prepends 'passage: ' to corpus and 'query: ' to queries (offline check)."""
    from retrieval.baselines import E5Baseline

    class _FakeModel:
        """Records inputs passed to encode()."""

        def __init__(self):
            self.encoded: list[list[str]] = []

        def encode(self, texts, *, show_progress_bar=False, convert_to_numpy=False):
            import numpy as np
            self.encoded.append(list(texts))
            return np.ones((len(texts), 4), dtype="float32")

    cat = [{"id": "x", "name": "Foo", "aliases": [], "description": "bar baz"}]
    bl = E5Baseline.__new__(E5Baseline)
    bl._model_name = "intfloat/e5-base-v2"
    bl._problem_ids = []
    bl._embeddings = None
    fake = _FakeModel()
    bl._model = fake

    import numpy as np
    bl._problem_ids = ["x"]
    raw = np.ones((1, 4), dtype="float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    bl._embeddings = raw / norms

    bl.rank("my query", top_k=1)
    # The query passed to encode() should start with "query: "
    assert fake.encoded and fake.encoded[0][0].startswith("query: ")


def test_bge_instruction_applied():
    """BGEBaseline correctly prepends instruction prefix to queries (offline check)."""
    from retrieval.baselines import BGEBaseline

    class _FakeModel:
        def __init__(self):
            self.encoded: list[list[str]] = []

        def encode(self, texts, *, show_progress_bar=False, convert_to_numpy=False):
            import numpy as np
            self.encoded.append(list(texts))
            return np.ones((len(texts), 4), dtype="float32")

    cat = [{"id": "x", "name": "Foo", "aliases": [], "description": "bar baz"}]
    bl = BGEBaseline.__new__(BGEBaseline)
    bl._model_name = "BAAI/bge-large-en-v1.5"
    bl._problem_ids = ["x"]
    fake = _FakeModel()
    bl._model = fake

    import numpy as np
    raw = np.ones((1, 4), dtype="float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    bl._embeddings = raw / norms

    bl.rank("my query", top_k=1)
    assert fake.encoded and fake.encoded[0][0].startswith(
        "Represent this sentence for searching relevant passages: "
    )


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
