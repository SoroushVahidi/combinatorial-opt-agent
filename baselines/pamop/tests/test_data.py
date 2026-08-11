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


def test_missing_structured_data_error_is_a_file_not_found_error():
    """MissingStructuredDataError should be catchable as FileNotFoundError
    too, so generic file-handling code doesn't need a special case."""
    assert issubclass(data.MissingStructuredDataError, FileNotFoundError)


def test_resolve_problem_info_path_uses_bare_id_when_present(monkeypatch):
    calls = []

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        calls.append(filename)
        if filename == "data/5/problem_info.json":
            return "/fake/path/problem_info.json"
        raise _entry_not_found()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    path = data._resolve_problem_info_path(5, token=None)
    assert path == "/fake/path/problem_info.json"
    assert calls == ["data/5/problem_info.json"]  # never had to list the repo


def test_resolve_problem_info_path_falls_back_to_suffixed_variant(monkeypatch):
    """Regression test for the loader gap: an id with no bare
    problem_info.json but a suffixed sibling (e.g. "28-unsolved") that DOES
    have one should still resolve, without a network call for every id."""

    def fake_hf_hub_download(repo_id, filename, **kwargs):
        if filename == "data/28-unsolved/problem_info.json":
            return "/fake/path/28-unsolved/problem_info.json"
        raise _entry_not_found()

    def fake_list_repo_files(self, repo_id, **kwargs):
        return [
            "data/28-unsolved/description.txt",
            "data/28-unsolved/problem_info.json",
            "data/29/description.txt",
        ]

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", fake_list_repo_files)

    path = data._resolve_problem_info_path(28, token=None)
    assert path == "/fake/path/28-unsolved/problem_info.json"


def test_resolve_problem_info_path_raises_missing_structured_data_when_nothing_matches(monkeypatch):
    def fake_hf_hub_download(repo_id, filename, **kwargs):
        raise _entry_not_found()

    def fake_list_repo_files(self, repo_id, **kwargs):
        return ["data/28/description.txt", "data/28/metadata.json"]  # no problem_info.json anywhere

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", fake_list_repo_files)

    with pytest.raises(data.MissingStructuredDataError):
        data._resolve_problem_info_path(28, token=None)


def test_load_problem_text_reads_description_txt_not_structured_metadata(monkeypatch, tmp_path):
    description = tmp_path / "description.txt"
    description.write_text("A safe synthetic LP has 3 widgets and 7 hours.\n", encoding="utf-8")
    calls = []

    def fake_resolve(problem_id, filename, token):
        calls.append((problem_id, filename, token))
        return str(description)

    monkeypatch.setattr(data, "_get_hf_token", lambda: "hf-token")
    monkeypatch.setattr(data, "_resolve_problem_file_path", fake_resolve)

    assert data.load_problem_text(14) == "A safe synthetic LP has 3 widgets and 7 hours.\n"
    assert calls == [(14, "description.txt", "hf-token")]


def _entry_not_found():
    from huggingface_hub.errors import EntryNotFoundError

    return EntryNotFoundError("404 (mocked)")


@pytest.mark.requires_network
@pytest.mark.parametrize("problem_id", [28, 51, 57, 123, 126, 135])
def test_known_missing_ids_raise_missing_structured_data_error_live(problem_id):
    """Live regression test for the exact ids discovered to have no
    problem_info.json anywhere (bare or suffixed) in the current HF
    snapshot -- see MissingStructuredDataError's docstring."""
    with pytest.raises(data.MissingStructuredDataError):
        data.load_problem_record(problem_id)


@pytest.mark.requires_network
def test_known_good_id_still_loads_live():
    record = data.load_problem_record(1)
    assert "parametrized_description" in record


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
