"""Handcrafted structured features for (slot, mention) pairs."""

from __future__ import annotations

from typing import Any


def row_to_feature_vector(row: dict[str, Any]) -> list[float]:
    """Build a small float vector from a pairwise ranker data row."""
    return [
        float(row.get("feat_type_match", 0)),
        float(row.get("feat_operator_cue_match", 0)),
        float(row.get("feat_total_like", 0)),
        float(row.get("feat_per_unit_like", 0)),
        float(row.get("feat_slot_mention_overlap", 0)),
    ]


FEATURE_DIM = 5
