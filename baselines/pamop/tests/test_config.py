"""Tests for config loading and the paper_faithful vs reconstructed_default
distinction (baselines/pamop/config.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.config import (
    UnspecifiedPaperDetailError,
    load_config,
    paper_faithful_path,
    reconstructed_default_path,
)


def test_paper_faithful_loads_and_has_paper_specified_values():
    cfg = load_config(paper_faithful_path())
    assert cfg.config_kind == "paper_faithful"
    # PAPER-SPECIFIED values must be present.
    assert cfg.llm.temperature == 0.2
    assert cfg.llm.max_correction_iterations == 5
    assert cfg.execution.generation_target == "AMPL"
    assert cfg.execution.solver_backend == "gurobi_via_ampl"


def test_paper_faithful_unspecified_fields_are_null():
    cfg = load_config(paper_faithful_path())
    assert cfg.partitioning.epsilon is None
    assert cfg.partitioning.tfidf_top_k is None
    assert cfg.partitioning.leaf_stop_min_constraints is None
    assert cfg.partitioning.similarity_weights_by_layer is None


def test_paper_faithful_require_raises_for_unspecified_field():
    cfg = load_config(paper_faithful_path())
    with pytest.raises(UnspecifiedPaperDetailError):
        cfg.require("partitioning", "epsilon")


def test_paper_faithful_require_succeeds_for_specified_field():
    cfg = load_config(paper_faithful_path())
    assert cfg.require("llm", "temperature") == 0.2


def test_reconstructed_default_has_no_unspecified_partitioning_fields():
    cfg = load_config(reconstructed_default_path())
    for name in (
        "tfidf_top_k",
        "epsilon",
        "similarity_weights_by_layer",
        "clustering_algorithm",
        "independent_set_algorithm",
        "bipartite_edge_confidence_threshold",
        "leaf_stop_min_constraints",
        "leaf_stop_similarity_threshold",
    ):
        assert cfg.require("partitioning", name) is not None, name


def test_reconstructed_default_similarity_weights_has_root_and_default_layers():
    cfg = load_config(reconstructed_default_path())
    weights = cfg.partitioning.similarity_weights_by_layer
    assert "root" in weights
    assert "default" in weights
    for layer_weights in weights.values():
        assert set(layer_weights) == {"adjacency", "keyword", "vector"}


def test_unknown_config_field_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "config_kind: bad\ncitation: x\npartitioning:\n  not_a_real_field: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad)
