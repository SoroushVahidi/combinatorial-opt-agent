"""Network-free tests for AMPL rendering, validation, and execution parsing."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.pamop.ampl.executor import AmplExecutor
from baselines.pamop.ampl.renderer import render_merged_model
from baselines.pamop.ampl.types import DiagnosticSeverity, ErrorCategory
from baselines.pamop.ampl.validator import validate_ampl_model
from baselines.pamop.llm.types import LLMResponse
from baselines.pamop.modeling import MergedModel
from baselines.pamop.prompts import load_prompt


def _fake_response() -> LLMResponse:
    return LLMResponse(
        text="",
        provider="fake",
        model="fake-model",
        timestamp="2026-01-01T00:00:00+00:00",
        temperature=0.2,
        top_p=None,
        max_tokens=None,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_seconds=0.0,
        retry_count=0,
        prompt_hash="deadbeef",
    )


def _merged_model(
    *,
    params: str = "param cap default 4;",
    variables: str = "var x >= 0;",
    objective: str = "maximize profit: 3*x;",
    constraints: str = "subject to cap_limit: x <= cap;",
) -> MergedModel:
    return MergedModel(
        problem_id="synthetic",
        parameters_text=params,
        variables_text=variables,
        objective_text=objective,
        constraints_text=constraints,
        leaf_results=(),
        root_llm_response=_fake_response(),
        root_prompt_template=load_prompt("modeling_root_v1.txt"),
        symbol_conflicts=(),
        config_hash="hash",
        provenance={},
    )


def test_ampl_renderer_outputs_valid_lp_model():
    rendered = render_merged_model(_merged_model())
    assert rendered.valid
    assert "param cap default 4;" in rendered.model_text
    assert "maximize profit" in rendered.model_text


def test_ampl_validator_accepts_integer_and_binary_declarations():
    text = """
    var x >= 0 integer;
    var y binary;
    maximize obj: x + y;
    subject to c1: x + y <= 1;
    """
    diagnostics = validate_ampl_model(text)
    assert not [d for d in diagnostics if d.severity == DiagnosticSeverity.ERROR]


def test_ampl_validator_flags_duplicate_symbols():
    diagnostics = validate_ampl_model(
        "param cap; var cap >= 0; maximize obj: cap; subject to c: cap <= 1;"
    )
    assert any(d.code == "duplicate_symbol" for d in diagnostics)


def test_ampl_validator_flags_unresolved_symbols():
    diagnostics = validate_ampl_model("var x; maximize obj: x + y; subject to c: x <= 1;")
    assert any(d.code == "unresolved_symbol" and d.symbol == "y" for d in diagnostics)


def test_ampl_validator_flags_malformed_constraint_expression():
    diagnostics = validate_ampl_model("var x; maximize obj: x; subject to c: x <= ;")
    assert any(d.code == "malformed_constraint_expression" for d in diagnostics)


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_executor_parses_successful_solve(monkeypatch):
    def fake_run(*args, **kwargs):
        return _Proc(0, "solve_result = solved\nprofit = 12\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = AmplExecutor(python_executable="python").execute(render_merged_model(_merged_model()).model_text)
    assert result.success
    assert result.solver_status == "solved"
    assert result.objective_value == 12


def test_executor_parses_infeasible_result(monkeypatch):
    def fake_run(*args, **kwargs):
        return _Proc(0, "solve_result = infeasible\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = AmplExecutor(python_executable="python").execute(render_merged_model(_merged_model()).model_text)
    assert not result.success
    assert result.solver_status == "infeasible"
    assert result.error_category == ErrorCategory.MODEL_ERROR


def test_executor_classifies_environment_error(monkeypatch):
    def fake_run(*args, **kwargs):
        return _Proc(1, "", "license checkout failed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = AmplExecutor(python_executable="python").execute(render_merged_model(_merged_model()).model_text)
    assert not result.success
    assert result.error_category == ErrorCategory.ENVIRONMENT_ERROR


def test_executor_does_not_invoke_subprocess_when_static_validation_fails(monkeypatch):
    def fake_run(*args, **kwargs):
        raise AssertionError("subprocess should not run")

    monkeypatch.setattr(subprocess, "run", fake_run)
    bad = render_merged_model(_merged_model(constraints="subject to c: x <= missing;"))
    result = AmplExecutor(python_executable="python").execute(bad.model_text)
    assert not result.success
    assert result.error_category == ErrorCategory.MODEL_ERROR
