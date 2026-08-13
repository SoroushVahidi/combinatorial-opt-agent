"""Lightweight ORLM tests; no GPU, COPT, network, or paid API required."""
from __future__ import annotations

import json

from baselines.orlm.config import ORLM_PROMPT_TEMPLATE, OrlmConfig
from baselines.orlm.data_adapter import adapt_record, build_orlm_prompt, load_jsonl_records
from baselines.orlm.evaluator import evaluate_results
from baselines.orlm.execution_harness import execute_coptpy
from baselines.orlm.output_normalizer import parse_orlm_output
from baselines.orlm.pipeline import JsonlResultStore, run_one
from baselines.orlm.result_schema import OrlmResult
from baselines.orlm.runner import OrlmRunner
from baselines.orlm.static_validation import validate_coptpy_code


class FakeBackend:
    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt, config):
        assert "# Question:" in prompt
        return self.output, {"prompt_tokens": 4, "completion_tokens": 12, "total_tokens": 16}


VALID_CODE = """import coptpy\nenv = coptpy.Envr()\nmodel = env.createModel('x')\nx = model.addVar(lb=0)\nmodel.setObjective(x, coptpy.COPT.MAXIMIZE)\nmodel.addConstr(x <= 1)\nmodel.solve()\n"""


def test_official_prompt_snapshot():
    assert build_orlm_prompt("  A\nproblem. ") == ORLM_PROMPT_TEMPLATE.format(Question="A\nproblem.") .strip()
    assert "# Response:" in build_orlm_prompt("problem")


def test_adapter_preserves_id_text_gold_and_stable_hash():
    result = adapt_record({"doc_id": "nlp4lp_test_4", "text": "Line 1\nLine 2", "meta": {"kind": "LP"}}, source="fixture")
    assert result.supported
    assert result.record.problem_id == "nlp4lp_test_4"
    assert result.record.raw_text == "Line 1\nLine 2"
    assert result.record.gold_metadata["meta"]["kind"] == "LP"
    assert result.record.to_upstream_example()["en_question"] == "Line 1\nLine 2"


def test_adapter_rejects_missing_fields_without_dropping_reason():
    assert adapt_record({"text": "x"}).reason == "missing_problem_id"
    assert adapt_record({"doc_id": "x"}).reason == "missing_or_empty_problem_text"


def test_jsonl_adapter_keeps_malformed_rows(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"doc_id":"1","text":"ok"}\nnot-json\n\n', encoding="utf-8")
    results = load_jsonl_records(path)
    assert [r.supported for r in results] == [True, False, False]
    assert results[1].reason == "invalid_json:2"
    assert results[2].reason == "blank_line:3"


def test_output_normalizer_handles_multiple_blocks_and_prose():
    raw = "Description\n```text\nnot code\n```\n```python\n" + VALID_CODE + "```\nAfter"
    parsed = parse_orlm_output(raw)
    assert parsed.coptpy_code == VALID_CODE.strip()
    assert parsed.code_blocks_seen == 2
    assert "Description" in parsed.model_description


def test_output_normalizer_handles_unfenced_and_empty():
    parsed = parse_orlm_output("Model.\n" + VALID_CODE)
    assert parsed.parser_status == "UNFENCED_CODE_EXTRACTED"
    assert parse_orlm_output("").parser_status == "EMPTY"


def test_static_validation_and_security_checks():
    valid = validate_coptpy_code(VALID_CODE)
    assert valid.status == "STATIC_VALID"
    unsafe = validate_coptpy_code("import coptpy\nimport os\nmodel.solve()\nos.system('x')")
    assert unsafe.status == "STATIC_INVALID"
    assert "dangerous_operation_present" in unsafe.errors
    assert validate_coptpy_code("def broken(").status == "PYTHON_SYNTAX_FAILURE"


def test_mock_end_to_end_and_result_round_trip(tmp_path):
    record = adapt_record({"doc_id": "1", "text": "Maximize x."}, source="fixture").record
    runner = OrlmRunner(OrlmConfig(), FakeBackend("Model\n```python\n" + VALID_CODE + "```"))
    result = run_one(record, runner, git_sha="test-sha")
    assert result.parsed is not None
    assert result.static_validation.status == "STATIC_VALID"
    restored = OrlmResult.from_dict(json.loads(result.to_json()))
    assert restored.problem_id == "1"
    assert restored.static_validation.status == "STATIC_VALID"
    assert evaluate_results([result.to_dict()])["static_valid_code_rate"] == 1.0


def test_runner_returns_structured_generation_result():
    result = OrlmRunner(backend=FakeBackend(VALID_CODE)).generate(build_orlm_prompt("prompt"))
    assert result.status == "COMPLETED"
    assert result.token_counts["total_tokens"] == 16


def test_result_store_resumes_by_problem_id(tmp_path):
    record = adapt_record({"doc_id": "1", "text": "x"}, source="fixture").record
    store = JsonlResultStore(tmp_path / "results.jsonl")
    runner = OrlmRunner(backend=FakeBackend("empty"))
    first = store.append_unfinished([record], runner)
    second = store.append_unfinished([record], runner)
    assert len(first) == 1
    assert second == []


def test_execution_harness_is_dry_run_by_default():
    result = execute_coptpy(VALID_CODE)
    assert result.attempted is False
    assert result.status == "DRY_RUN"
