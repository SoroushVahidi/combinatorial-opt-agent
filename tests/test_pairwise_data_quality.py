from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.learning.check_nlp4lp_pairwise_data_quality import _check_split


def test_check_split_computes_feature_means_from_aligned_rows() -> None:
    rows = [
        {"label": 1, "group_id": "g1", "slot_name": "s1", "mention_surface": "m1"},
        {"label": 0, "group_id": "g1", "slot_name": "s1", "mention_surface": "m2", "type_match": 0.2},
        {"label": 1, "group_id": "g2", "slot_name": "s2", "mention_surface": "m3", "type_match": 0.8},
        {"label": 0, "group_id": "g2", "slot_name": "s2", "mention_surface": "m4", "type_match": 0.4},
    ]

    result = _check_split("train", rows)

    assert result["feature_stats"]["type_match"]["mean_on_positive"] == 0.8
    assert result["feature_stats"]["type_match"]["mean_on_negative"] == 0.3


def test_check_split_ignores_null_feature_values_in_means() -> None:
    rows = [
        {"label": 1, "group_id": "g1", "slot_name": "s1", "mention_surface": "m1", "type_match": None},
        {"label": 1, "group_id": "g1", "slot_name": "s1", "mention_surface": "m2", "type_match": 1.0},
        {"label": 0, "group_id": "g2", "slot_name": "s2", "mention_surface": "m3", "type_match": None},
        {"label": 0, "group_id": "g2", "slot_name": "s2", "mention_surface": "m4", "type_match": 0.5},
    ]

    result = _check_split("train", rows)

    assert result["feature_stats"]["type_match"]["coverage"] == 0.5
    assert result["feature_stats"]["type_match"]["mean_on_positive"] == 1.0
    assert result["feature_stats"]["type_match"]["mean_on_negative"] == 0.5
