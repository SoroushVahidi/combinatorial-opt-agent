"""Tests for the NLP4LP subset selector and its post-PaMOP-id guardrails.

No network access required -- these only test the id-range logic in
baselines/pamop/data.py, not actual dataset fetches.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop import data


def test_pamop_possible_269_selector_returns_exactly_1_to_269():
    ids = data.list_ids_for_subset(data.SUBSET_POSSIBLE_269)
    assert ids == list(range(1, 270))
    assert len(ids) == 269


def test_unknown_subset_rejected():
    with pytest.raises(data.UnknownSubsetError):
        data.list_ids_for_subset("nope")


def test_pamop_67_is_never_a_valid_subset_name():
    """Guardrail: the exact PaMOP set is unresolved (see report section 13);
    a subset literally named "pamop_67" would misrepresent an unverified
    superset as PaMOP's confirmed evaluation set, so it must never exist."""
    assert "pamop_67" not in data._VALID_SUBSETS
    with pytest.raises(data.UnknownSubsetError):
        data.list_ids_for_subset("pamop_67")


@pytest.mark.parametrize("problem_id", [270, 271, 292, 293, 300, 354, 355, 361])
def test_post_pamop_ids_rejected(problem_id):
    with pytest.raises(data.PostPamopIdError):
        data.assert_not_post_pamop(problem_id)


@pytest.mark.parametrize("problem_id", [1, 2, 100, 268, 269])
def test_pre_pamop_ids_accepted(problem_id):
    data.assert_not_post_pamop(problem_id)  # must not raise


def test_load_problem_record_rejects_post_pamop_id_before_any_network_call():
    """The guard must fire before hf_hub_download is even attempted."""
    with pytest.raises(data.PostPamopIdError):
        data.load_problem_record(300)


def test_alignment_manifest_matches_selector():
    manifest = data.load_alignment_manifest()
    possible = [r for r in manifest if r["mapping_confidence"] == "POSSIBLE_MATCH"]
    no_match = [r for r in manifest if r["mapping_confidence"] == "NO_MATCH"]
    assert len(manifest) == 331
    assert len(possible) == 269
    assert len(no_match) == 62
    possible_hf_ids = sorted(r["current_nlp4lp_hf_problem_id"] for r in possible)
    assert possible_hf_ids == data.list_ids_for_subset(data.SUBSET_POSSIBLE_269)
    for row in no_match:
        assert row["current_nlp4lp_hf_problem_id"] >= data.POST_PAMOP_HF_ID_MIN
    for row in manifest:
        # guardrail against ever leaking gated text into the committed manifest
        assert "description" not in row
        assert "text" not in row
