"""Tests for the LLM-based structured extraction stage (G_extr).

Uses fake in-process providers only -- never a real network/API call. See
tests/test_data.py / test_llm.py for the network-marked live regression
tests that are skipped by default.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.config import load_config, reconstructed_default_path
from baselines.pamop.extraction import (
    ExtractionValidationError,
    extract_structured_problem,
    validate_extraction,
)
from baselines.pamop.llm.base import LLMProvider


def _valid_payload():
    return {
        "global_summary": "A toy factory production problem.",
        "objective_text": "Maximize profit from producing two products.",
        "constraints": [
            {"description": "Labor hours used must not exceed available labor.", "vagueness_score": 0.1},
            {"description": "Material used must not exceed available material.", "vagueness_score": 0.2},
        ],
        "variables": [
            {"name": "labor_hours", "description": "Available labor hours", "type": "parameter"},
            {"name": "produce_a", "description": "Units of product A produced", "type": "continuous"},
        ],
    }


class _ScriptedProvider(LLMProvider):
    """Returns each entry of ``responses`` in order, one per call."""

    name = "scripted"

    def __init__(self, responses: list[str], **kwargs):
        super().__init__(**kwargs)
        self._responses = list(responses)
        self.calls = 0

    def _call(self, prompt, config):
        text = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return {"text": text, "finish_reason": "stop"}


@pytest.fixture(scope="module")
def config():
    return load_config(reconstructed_default_path())


# ---------------------------------------------------------------------
# validate_extraction
# ---------------------------------------------------------------------


def test_validate_extraction_accepts_a_well_formed_payload():
    validate_extraction(_valid_payload())  # must not raise


def test_validate_extraction_rejects_non_dict():
    with pytest.raises(ExtractionValidationError):
        validate_extraction(["not", "a", "dict"])


@pytest.mark.parametrize(
    "field", ["global_summary", "objective_text", "constraints", "variables"]
)
def test_validate_extraction_rejects_missing_required_field(field):
    payload = _valid_payload()
    del payload[field]
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)


def test_validate_extraction_rejects_empty_constraints_list():
    payload = _valid_payload()
    payload["constraints"] = []
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)


@pytest.mark.parametrize("bad_score", [-0.1, 1.1, "high", None, True])
def test_validate_extraction_rejects_out_of_range_or_wrong_type_vagueness_score(bad_score):
    payload = _valid_payload()
    payload["constraints"][0]["vagueness_score"] = bad_score
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)


def test_validate_extraction_rejects_invalid_variable_type():
    payload = _valid_payload()
    payload["variables"][0]["type"] = "not_a_real_type"
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)


def test_validate_extraction_rejects_non_identifier_variable_name():
    payload = _valid_payload()
    payload["variables"][0]["name"] = "not a valid identifier!"
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)


def test_validate_extraction_rejects_duplicate_variable_names():
    payload = _valid_payload()
    payload["variables"].append(dict(payload["variables"][0]))
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)


def test_validate_extraction_never_mutates_or_fills_in_missing_content():
    """The validator must reject bad input, not repair it -- PaMOP's own
    repair mechanisms are a separate, later, not-yet-implemented stage."""
    payload = _valid_payload()
    del payload["objective_text"]
    original = json.loads(json.dumps(payload))  # deep copy for comparison
    with pytest.raises(ExtractionValidationError):
        validate_extraction(payload)
    assert payload == original  # unchanged, nothing silently added


# ---------------------------------------------------------------------
# extract_structured_problem (full flow, scripted/fake provider)
# ---------------------------------------------------------------------


def test_extract_structured_problem_succeeds_on_first_valid_response(config):
    provider = _ScriptedProvider([json.dumps(_valid_payload())])
    result = extract_structured_problem("p1", "some raw NL problem text", provider, config)
    assert result.validation_attempts == 1
    assert len(result.structured_problem.constraints) == 2
    assert len(result.structured_problem.variables) == 2
    assert result.structured_problem.source == "llm_extraction:scripted:gpt-4o"


def test_extract_structured_problem_retries_on_malformed_json_then_succeeds(config):
    provider = _ScriptedProvider(["not json at all", json.dumps(_valid_payload())])
    result = extract_structured_problem("p2", "text", provider, config)
    assert result.validation_attempts == 2
    assert provider.calls == 2


def test_extract_structured_problem_retries_on_schema_violation_then_succeeds(config):
    bad = _valid_payload()
    del bad["objective_text"]
    provider = _ScriptedProvider([json.dumps(bad), json.dumps(_valid_payload())])
    result = extract_structured_problem("p3", "text", provider, config)
    assert result.validation_attempts == 2


def test_extract_structured_problem_gives_up_after_extraction_max_retries(config):
    always_bad = json.dumps({"global_summary": "s"})  # missing required fields
    provider = _ScriptedProvider([always_bad] * 10)
    with pytest.raises(ExtractionValidationError):
        extract_structured_problem("p4", "text", provider, config)
    # extraction_max_retries=2 in reconstructed_default.yaml -> 3 attempts total
    assert provider.calls == config.llm.extraction_max_retries + 1


def test_extract_structured_problem_tolerates_a_markdown_code_fence(config):
    fenced = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    provider = _ScriptedProvider([fenced])
    result = extract_structured_problem("p5", "text", provider, config)
    assert result.validation_attempts == 1


def test_extract_structured_problem_records_prompt_hash_matching_loaded_template(config):
    from baselines.pamop.prompts import load_prompt

    provider = _ScriptedProvider([json.dumps(_valid_payload())])
    result = extract_structured_problem("p6", "raw text", provider, config)
    rendered = load_prompt("extraction_v1.txt").render(problem_text="raw text")
    from baselines.pamop.llm.base import prompt_hash

    assert result.llm_response.prompt_hash == prompt_hash(rendered)


def test_extract_structured_problem_is_deterministic_given_fixed_provider_output(config):
    def make_result():
        provider = _ScriptedProvider([json.dumps(_valid_payload())])
        return extract_structured_problem("p7", "raw text", provider, config)

    r1, r2 = make_result(), make_result()
    assert r1.structured_problem.to_dict() if hasattr(r1.structured_problem, "to_dict") else True
    assert [c.description for c in r1.structured_problem.constraints] == [
        c.description for c in r2.structured_problem.constraints
    ]
    assert [v.name for v in r1.structured_problem.variables] == [v.name for v in r2.structured_problem.variables]


def test_extraction_wires_into_partition_tree(config):
    """Integration check: G_extr output feeds directly into the already
    implemented, independent partitioning stage."""
    from baselines.pamop.partition import build_partition_tree

    provider = _ScriptedProvider([json.dumps(_valid_payload())])
    result = extract_structured_problem("p8", "raw text", provider, config)
    tree = build_partition_tree(result.structured_problem, config)
    seen = sorted(idx for leaf in tree.leaves() for idx in leaf.constraint_group)
    assert seen == list(range(len(result.structured_problem.constraints)))
