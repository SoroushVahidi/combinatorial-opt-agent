"""
Unit tests for training/splits.py: disjoint train/dev/test, reproducibility, stratification.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_build_splits_disjoint():
    """Train, dev, test must be disjoint and cover all problems with id."""
    from training.splits import build_splits, load_catalog

    catalog = load_catalog()
    if len(catalog) < 10:
        pytest.skip("Catalog too small")
    splits = build_splits(catalog, seed=42, train_ratio=0.70, dev_ratio=0.15, test_ratio=0.15)
    train = set(splits["train"])
    dev = set(splits["dev"])
    test = set(splits["test"])
    assert train & dev == set(), "train and dev must be disjoint"
    assert train & test == set(), "train and test must be disjoint"
    assert dev & test == set(), "dev and test must be disjoint"
    ids_in_catalog = {p["id"] for p in catalog if p.get("id")}
    all_split_ids = train | dev | test
    assert all_split_ids <= ids_in_catalog, "split IDs must be from catalog"
    assert len(all_split_ids) == len(ids_in_catalog), "every catalog id must be in exactly one split"


def test_build_splits_reproducibility():
    """Same seed must yield same splits."""
    from training.splits import build_splits, load_catalog

    catalog = load_catalog()
    if len(catalog) < 5:
        pytest.skip("Catalog too small")
    a = build_splits(catalog, seed=123)
    b = build_splits(catalog, seed=123)
    assert a["train"] == b["train"] and a["dev"] == b["dev"] and a["test"] == b["test"]


def test_build_splits_different_seeds_different():
    """Different seeds should usually yield different splits (at least one different)."""
    from training.splits import build_splits, load_catalog

    catalog = load_catalog()
    if len(catalog) < 20:
        pytest.skip("Catalog too small")
    a = build_splits(catalog, seed=1)
    b = build_splits(catalog, seed=2)
    # At least one split should differ
    assert (a["train"] != b["train"]) or (a["dev"] != b["dev"]) or (a["test"] != b["test"])


def test_write_and_load_splits():
    """Write splits to JSON and load back; content must match."""
    from training.splits import build_splits, load_splits, write_splits, load_catalog

    catalog = load_catalog()
    if len(catalog) < 5:
        pytest.skip("Catalog too small")
    splits = build_splits(catalog, seed=99)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        write_splits(splits, path)
        loaded = load_splits(path)
        assert loaded["train"] == splits["train"]
        assert loaded["dev"] == splits["dev"]
        assert loaded["test"] == splits["test"]
    finally:
        path.unlink(missing_ok=True)


def test_get_problem_ids_for_split():
    """get_problem_ids_for_split returns the correct list and rejects invalid split name."""
    from training.splits import get_problem_ids_for_split

    splits = {"train": ["a", "b"], "dev": ["c"], "test": ["d"]}
    assert get_problem_ids_for_split(splits, "train") == ["a", "b"]
    assert get_problem_ids_for_split(splits, "dev") == ["c"]
    assert get_problem_ids_for_split(splits, "test") == ["d"]
    with pytest.raises(ValueError, match="split must be one of"):
        get_problem_ids_for_split(splits, "invalid")
