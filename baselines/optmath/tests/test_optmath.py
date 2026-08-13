from __future__ import annotations

import json

from baselines.optmath.config import OPTMATH_SYSTEM_PROMPT, OPTMATH_USER_TEMPLATE, OptmathConfig
from baselines.optmath.data_adapter import adapt_record, load_jsonl_records
from baselines.optmath.evaluator import evaluate_results
from baselines.optmath.execution_harness import execute_gurobi
from baselines.optmath.output_normalizer import parse_output
from baselines.optmath.pipeline import JsonlResultStore, run_one
from baselines.optmath.prompt import build_prompt
from baselines.optmath.result_schema import OptmathResult
from baselines.optmath.runner import OptmathRunner
from baselines.optmath.static_validation import validate_code


VALID = """import gurobipy as gp\nfrom gurobipy import GRB\nmodel = gp.Model()\nx = model.addVar(lb=0)\nmodel.setObjective(x, GRB.MAXIMIZE)\nmodel.addConstr(x <= 1)\nmodel.optimize()\nprint(model.objVal)\n"""


class FakeBackend:
    def generate(self, prompt, config):
        assert prompt.system == OPTMATH_SYSTEM_PROMPT
        assert "# Question:" in prompt.user
        return "Explanation\n```python\n" + VALID + "```", {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}


def test_official_prompt_snapshot():
    bundle = build_prompt("  Maximize x. \n")
    assert bundle.system == OPTMATH_SYSTEM_PROMPT
    assert bundle.user == OPTMATH_USER_TEMPLATE.format(question="Maximize x.")
    assert "Start your code with: import gurobipy as gp" in bundle.user


def test_adapter_preserves_ids_text_and_gold_metadata():
    result = adapt_record({"doc_id": "nlp4lp_test_4", "text": "Line 1\nLine 2", "en_answer": 12.0, "meta": {"kind": "LP"}})
    assert result.supported and result.record.problem_id == "nlp4lp_test_4"
    assert result.record.raw_text == "Line 1\nLine 2"
    assert result.record.gold_metadata["gold_objective"] == 12.0


def test_adapter_rejects_and_jsonl_retains_bad_rows(tmp_path):
    assert adapt_record({"text": "x"}).reason == "missing_problem_id"
    path = tmp_path / "records.jsonl"
    path.write_text('{"doc_id":"1","text":"x"}\nnot-json\n', encoding="utf-8")
    rows = load_jsonl_records(path)
    assert [row.supported for row in rows] == [True, False]
    assert rows[1].reason == "invalid_json:2"


def test_output_parser_and_static_validation():
    parsed = parse_output("Model description\n```python\n" + VALID + "```\n")
    assert parsed.status == "CODE_EXTRACTED"
    assert validate_code(parsed.generated_code).status == "STATIC_VALID"
    assert validate_code("import gurobipy\nmodel.optimize()").status == "STATIC_INVALID"


def test_mock_end_to_end_result_roundtrip_and_evaluator():
    record = adapt_record({"doc_id": "1", "text": "Maximize x.", "en_answer": 1.0}).record
    result = run_one(record, OptmathRunner(backend=FakeBackend()), git_sha="test-sha")
    assert result.static_validation.status == "STATIC_VALID"
    restored = OptmathResult.from_dict(json.loads(result.to_json()))
    assert restored.problem_id == "1"
    metrics = evaluate_results([result.to_dict()])
    assert metrics["static_valid_code_rate"] == 1.0


def test_result_store_and_execution_dry_run(tmp_path):
    store = JsonlResultStore(tmp_path / "results.jsonl")
    assert store.completed_ids() == set()
    assert execute_gurobi(VALID).status == "DRY_RUN"
