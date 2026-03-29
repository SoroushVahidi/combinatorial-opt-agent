"""
Tests to ensure no train/eval leakage: train and test problem sets are disjoint,
and generate_samples with --split train only produces pairs for train problems.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_splits_train_test_disjoint():
    """After building splits, train and test problem IDs must be disjoint."""
    from training.splits import build_splits, load_catalog

    catalog = load_catalog()
    if len(catalog) < 10:
        pytest.skip("Catalog too small")
    splits = build_splits(catalog, seed=42)
    train_ids = set(splits["train"])
    test_ids = set(splits["test"])
    assert train_ids & test_ids == set(), "Train and test must be disjoint (no problem in both)"


def test_generate_samples_respects_split():
    """generate_all_samples with split_problem_ids only produces passages for those problems."""
    from training.splits import load_catalog
    from training.generate_samples import generate_all_samples, searchable_text

    catalog = load_catalog()
    if len(catalog) < 15:
        pytest.skip("Catalog too small")
    # Use a small subset as "train"
    train_ids = [p["id"] for p in catalog if p.get("id")][:5]
    pairs = generate_all_samples(
        seed=42,
        instances_per_problem=5,
        include_real_world=False,
        split_problem_ids=train_ids,
    )
    # Every passage should be searchable_text of a problem in train_ids
    train_passages = set()
    for p in catalog:
        if p.get("id") in train_ids:
            train_passages.add(searchable_text(p))
    for _q, passage in pairs:
        assert passage in train_passages, "Every pair's passage must be from a train-set problem"


def test_eval_instances_only_for_split():
    """When generating eval instances with problem_ids, only those problems appear as expected."""
    from training.splits import load_catalog
    from training.evaluate_retrieval import _generate_eval_instances

    catalog = load_catalog()
    if len(catalog) < 10:
        pytest.skip("Catalog too small")
    test_ids = [p["id"] for p in catalog if p.get("id")][5:10]
    pairs = _generate_eval_instances(
        catalog,
        seed=999,
        num_instances=50,
        problem_ids=test_ids,
    )
    expected_ids = {pid for _q, pid in pairs}
    assert expected_ids <= set(test_ids), "All expected problem_ids in eval must be in the requested split"


def test_full_pipeline_no_overlap():
    """Build splits; train IDs and test IDs have no overlap."""
    from training.splits import build_splits, load_catalog, write_splits, load_splits, get_problem_ids_for_split
    from training.generate_samples import generate_all_samples, searchable_text

    catalog = load_catalog()
    if len(catalog) < 20:
        pytest.skip("Catalog too small")
    splits = build_splits(catalog, seed=42)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "splits.json"
        write_splits(splits, path)
        loaded = load_splits(path)
        train_ids = set(get_problem_ids_for_split(loaded, "train"))
        test_ids = set(get_problem_ids_for_split(loaded, "test"))
        assert train_ids & test_ids == set()

        # Training pairs: only train problems
        train_pairs = generate_all_samples(
            seed=42,
            instances_per_problem=3,
            include_real_world=False,
            split_problem_ids=list(train_ids),
        )
        # All passages come from train problems
        train_passages = {searchable_text(p) for p in catalog if p.get("id") in train_ids}
        for _q, passage in train_pairs:
            assert passage in train_passages
