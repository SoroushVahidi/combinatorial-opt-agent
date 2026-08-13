"""Lightweight OR-R1 tests; no GPU, coptpy, vLLM, or network required."""
from __future__ import annotations

import json

from baselines.orr1.config import ORR1_PROMPT_TEMPLATE, OrR1Config, pass1_config, pass8_config
from baselines.orr1.data_adapter import adapt_record, build_orr1_prompt, load_jsonl_records
from baselines.orr1.evaluator import compute_solving_accuracy, evaluate_results
from baselines.orr1.execution_harness import execute_coptpy
from baselines.orr1.output_normalizer import parse_output
from baselines.orr1.pipeline import run_mock_pipeline
from baselines.orr1.result_schema import OrR1Result
from baselines.orr1.rollout import group_majority_vote, majority_voting, score_group
from baselines.orr1.runner import OrR1Runner
from baselines.orr1.static_validation import validate_code
from baselines.orr1.tgrpo_controller import (
    CheckpointState, assert_no_cross_group_isolation_violation, mock_merge_step, mock_sft_step,
    mock_tgrpo_step, reward_component_breakdown,
)

VALID_CODE = (
    "import coptpy\nfrom coptpy import COPT\nenv = coptpy.Envr()\n"
    "model = env.createModel('x')\nx = model.addVar(lb=0)\n"
    "model.setObjective(x, COPT.MAXIMIZE)\nmodel.addConstr(x <= 1)\nmodel.solve()\n"
)


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, prompt, config):
        assert "# Question:" in prompt
        return self.outputs, {"prompt_tokens": 4, "rollout_count": len(self.outputs)}


# --- data adapter / prompt --------------------------------------------------

def test_official_prompt_uses_replace_not_format_and_matches_template():
    out = build_orr1_prompt("A problem with {braces}.")
    assert "A problem with {braces}." in out
    assert out == ORR1_PROMPT_TEMPLATE.replace("{Question}", "A problem with {braces}.").strip()
    assert "# Response:" in out


def test_adapter_preserves_id_text_and_stable_hash():
    result = adapt_record({"problem_id": 14, "question": "line 1\nline 2", "answer": 67})
    assert result.supported
    assert result.record.problem_id == "14"
    assert result.record.raw_text == "line 1\nline 2"
    assert result.record.gold_metadata["gold_objective"] == 67
    assert len(result.record.input_sha256) == 64
    assert result.record.to_upstream_example() == {"question": "line 1\nline 2", "answer": 67}


def test_adapter_rejects_missing_fields_without_dropping_reason():
    assert adapt_record({"question": "x"}).reason == "missing_problem_id"
    assert adapt_record({"problem_id": "x"}).reason == "missing_or_empty_problem_text"


def test_jsonl_adapter_keeps_malformed_rows(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"problem_id":"1","question":"ok","answer":1}\nnot-json\n\n', encoding="utf-8")
    results = load_jsonl_records(path)
    assert [r.supported for r in results] == [True, False, False]
    assert results[1].reason == "invalid_json:2"
    assert results[2].reason == "blank_line:3"


# --- output normalizer / static validation ----------------------------------

def test_output_normalizer_first_block_only_and_format_reward():
    raw = ("## Mathematical Model:\nx\n## Decision Variables:\nx\n## Objective Function:\nx\n"
           "## Constraints:\nx\n## Python Code Solution Using `coptpy`:\n```python\n" + VALID_CODE + "```\nignored second ```python\nmore\n```")
    parsed = parse_output(raw)
    assert parsed.coptpy_code == VALID_CODE.strip()
    assert parsed.format_reward == 1.0
    assert parsed.parser_status == "CODE_EXTRACTED"


def test_output_normalizer_empty_and_no_code():
    assert parse_output("").parser_status == "EMPTY"
    assert parse_output("prose only").parser_status == "NO_CODE"


def test_static_validation_requires_literal_model_variable():
    valid = validate_code(VALID_CODE)
    assert valid.status == "STATIC_VALID" and valid.model_variable_present
    renamed = VALID_CODE.replace("model", "mdl")
    invalid = validate_code(renamed)
    assert "missing_model_variable" in invalid.errors
    assert validate_code("def broken(").status == "SYNTAX_INVALID"
    unsafe = validate_code("import coptpy\nimport os\nmodel = 1\nmodel.solve()\nos.system('x')")
    assert "dangerous_operation_present" in unsafe.errors


# --- rollout / majority voting ----------------------------------------------

def test_majority_voting_exact_port():
    assert majority_voting([1, 1, 2]) == 1
    assert group_majority_vote(["1", "1", "2", None]) == 1


def test_score_group_no_solution_and_zero_gold_cases():
    assert score_group(["No Best Solution", "No Best Solution"], "No Best Solution").pass_at_k is True
    assert score_group([0, 0.01, 5], 0).pass_at_k is True  # abs tolerance branch when gold == 0
    assert score_group([None, None], 10).pass_at_k is False


# --- TGRPO controller --------------------------------------------------------

def test_reward_never_uses_ground_truth():
    rewards = reward_component_breakdown(format_reward=1.0, execution_best_solution="3", group_pred_answers=["3", "3", "5"], own_pred_answer="3")
    assert rewards.valid_code_reward == 1.0 and rewards.voting_reward == 1.0 and rewards.total == 3.0
    dissent = reward_component_breakdown(format_reward=1.0, execution_best_solution="5", group_pred_answers=["3", "3", "5"], own_pred_answer="5")
    assert dissent.voting_reward == 0.0


def test_checkpoint_state_machine_and_isolation_guard():
    state = CheckpointState("BASE", base_model="Qwen/Qwen3-8B")
    state = mock_sft_step(state, output_dir="./output/sft")
    state = mock_tgrpo_step(state, output_dir="./output/lora_grpo", group_id="g1")
    state = mock_merge_step(state, output_dir="./output/full_grpo")
    assert state.stage == "MERGED" and state.adaptation_scope == "PER_PROBLEM_GROUP"
    assert_no_cross_group_isolation_violation([state])
    conflicting = CheckpointState("GRPO_LORA", base_model="Qwen/Qwen3-8B", lora_adapter_path="other", adaptation_scope="PER_PROBLEM_GROUP", owning_group_id="g1")
    try:
        assert_no_cross_group_isolation_violation([state, conflicting])
        assert False, "expected leakage assertion"
    except AssertionError:
        pass


def test_tgrpo_stage_order_is_enforced():
    base = CheckpointState("BASE", base_model="Qwen/Qwen3-8B")
    try:
        mock_tgrpo_step(base, output_dir="x")
        assert False, "TGRPO without SFT should fail"
    except ValueError:
        pass


# --- runner / result schema / pipeline --------------------------------------

def test_runner_returns_one_output_per_rollout():
    config = pass8_config(model_id="test-model")
    result = OrR1Runner(config, FakeBackend(["out"] * 8)).generate(build_orr1_prompt("q"))
    assert result.status == "COMPLETED" and len(result.raw_outputs) == 8


def test_runner_fails_without_checkpoint():
    class RaisingBackend:
        def generate(self, prompt, config):
            raise RuntimeError("checkpoint_unavailable")
    result = OrR1Runner(OrR1Config(), RaisingBackend()).generate(build_orr1_prompt("q"))
    assert result.status == "FAILED" and result.error_category == "checkpoint_unavailable"


def test_mock_end_to_end_pipeline_pass8():
    outcome = run_mock_pipeline({"problem_id": "14", "question": "Maximize x subject to a bound.", "answer": 1})
    assert len(outcome["records"]) == 8
    restored = [OrR1Result.from_dict(r) for r in outcome["records"]]
    assert all(r.static_validation["status"] == "STATIC_VALID" for r in restored)
    assert outcome["group_score"]["k"] == 8
    assert outcome["group_score"]["pass_at_k"] is True  # majority answer 1 matches gold 1
    assert json.loads(restored[0].to_json())["problem_id"] == "14"


def test_evaluator_offline_metrics_and_official_accuracy():
    outcome = run_mock_pipeline({"problem_id": "23", "question": "Maximize y.", "answer": 1})
    offline = evaluate_results(outcome["records"])
    assert offline["static_valid_code_rate"] == 1.0
    accuracy = compute_solving_accuracy(outcome["records"])
    assert accuracy["n_problems"] == 1
    assert accuracy["rollout_group_size"] == 8


def test_execution_harness_is_dry_run_by_default():
    result = execute_coptpy(VALID_CODE)
    assert result.attempted is False and result.status == "DRY_RUN"
