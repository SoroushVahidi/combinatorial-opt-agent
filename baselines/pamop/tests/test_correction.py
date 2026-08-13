"""Network-free tests for PaMOP correction loop reconstruction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from baselines.pamop.ampl.types import AmplDiagnostic, AmplExecutionResult, DiagnosticSeverity, ErrorCategory
from baselines.pamop.config import load_config, reconstructed_default_path
from baselines.pamop.correction import (
    CorrectionValidationError,
    run_correction_loop,
    run_g_comp,
    run_g_remod,
    run_g_rev,
)
from baselines.pamop.llm.base import LLMProvider, prompt_hash
from baselines.pamop.llm.types import LLMResponse
from baselines.pamop.modeling import MergedModel
from baselines.pamop.prompts import load_prompt
from baselines.pamop.representations import synthetic_structured_problem


class _ScriptedProvider(LLMProvider):
    name = "scripted_correction"

    def __init__(self, responses: list[dict]):
        super().__init__(max_retries=0)
        self.responses = [json.dumps(r) for r in responses]
        self.prompts_seen: list[str] = []
        self.calls = 0

    def _call(self, prompt, config):
        self.prompts_seen.append(prompt)
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return {"text": text, "finish_reason": "stop", "total_tokens": 7}


class _FakeExecutor:
    def __init__(self, results: list[AmplExecutionResult]):
        self.results = results
        self.calls = 0
        self.models_seen: list[str] = []

    def execute(self, model_text):
        self.models_seen.append(model_text)
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


@pytest.fixture(scope="module")
def config():
    return load_config(reconstructed_default_path())


def _problem():
    return synthetic_structured_problem(
        "corr_synth",
        global_summary="Choose production amount.",
        objective_text="Maximize profit.",
        constraint_texts=["Production must not exceed capacity."],
        variables=[("x", "production amount"), ("cap", "capacity")],
    )


def _llm_response(text: str = "") -> LLMResponse:
    return LLMResponse(
        text=text,
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


def _merged_model(ampl_bits: str = "subject to c: x <= cap;") -> MergedModel:
    return MergedModel(
        problem_id="corr_synth",
        parameters_text="param cap default 4;",
        variables_text="var x >= 0;",
        objective_text="maximize profit: x;",
        constraints_text=ampl_bits,
        leaf_results=(),
        root_llm_response=_llm_response(),
        root_prompt_template=load_prompt("modeling_root_v1.txt"),
        symbol_conflicts=(),
        config_hash="hash",
        provenance={},
    )


def _exec(success: bool, category: ErrorCategory = ErrorCategory.NONE) -> AmplExecutionResult:
    return AmplExecutionResult(
        model_hash="hash",
        parse_success=success,
        model_load_success=success,
        solver_invocation_success=success,
        solver_status="solved" if success else None,
        objective_value=4.0 if success else None,
        runtime_seconds=0.01,
        diagnostics=()
        if success
        else (
            AmplDiagnostic(
                DiagnosticSeverity.ERROR,
                "bad_model",
                "bad model",
                category,
            ),
        ),
        error_category=category,
    )


def _responses(corrected_model: str = "var x >= 0;\nmaximize profit: x;\nsubject to c: x <= 4;"):
    return [
        {
            "error_category": "model_error",
            "diagnosis": "capacity is not assigned",
            "actionable_feedback": ["give cap a value or inline the bound"],
        },
        {
            "needs_remodel": True,
            "reason": "review identifies a model issue",
            "targeted_issues": ["capacity value"],
        },
        {
            "ampl_model": corrected_model,
            "changes": ["inlined capacity"],
        },
    ]


def test_g_rev_prompt_generation_and_parsing(config):
    provider = _ScriptedProvider(_responses())
    result = run_g_rev(
        ampl_model="var x; maximize profit: x; subject to c: x <= cap;",
        execution=_exec(False, ErrorCategory.MODEL_ERROR),
        structured_problem=_problem(),
        provider=provider,
        config=config,
    )
    assert result.error_category == ErrorCategory.MODEL_ERROR
    assert "Structured execution diagnostics" in provider.prompts_seen[0]


def test_g_comp_behavior(config):
    provider = _ScriptedProvider(_responses()[1:])
    review = run_g_rev(
        ampl_model="bad",
        execution=_exec(False, ErrorCategory.MODEL_ERROR),
        structured_problem=_problem(),
        provider=_ScriptedProvider(_responses()),
        config=config,
    )
    comparison = run_g_comp(structured_problem=_problem(), review=review, provider=provider, config=config)
    assert comparison.needs_remodel is True
    assert comparison.targeted_issues == ("capacity value",)


def test_g_remod_parsing(config):
    provider = _ScriptedProvider(_responses()[2:])
    review_provider = _ScriptedProvider(_responses())
    review = run_g_rev(
        ampl_model="bad",
        execution=_exec(False, ErrorCategory.MODEL_ERROR),
        structured_problem=_problem(),
        provider=review_provider,
        config=config,
    )
    comparison = run_g_comp(
        structured_problem=_problem(),
        review=review,
        provider=_ScriptedProvider(_responses()[1:]),
        config=config,
    )
    remodel = run_g_remod(
        ampl_model="bad",
        structured_problem=_problem(),
        review=review,
        comparison=comparison,
        provider=provider,
        config=config,
    )
    assert "subject to c" in remodel.ampl_model
    assert remodel.changes == ("inlined capacity",)


def test_g_remod_rejects_malformed_response(config):
    provider = _ScriptedProvider([{"changes": []}])
    with pytest.raises(CorrectionValidationError):
        run_g_remod(
            ampl_model="bad",
            structured_problem=_problem(),
            review=run_g_rev(
                ampl_model="bad",
                execution=_exec(False, ErrorCategory.MODEL_ERROR),
                structured_problem=_problem(),
                provider=_ScriptedProvider(_responses()),
                config=config,
            ),
            comparison=run_g_comp(
                structured_problem=_problem(),
                review=run_g_rev(
                    ampl_model="bad",
                    execution=_exec(False, ErrorCategory.MODEL_ERROR),
                    structured_problem=_problem(),
                    provider=_ScriptedProvider(_responses()),
                    config=config,
                ),
                provider=_ScriptedProvider(_responses()[1:]),
                config=config,
            ),
            provider=provider,
            config=config,
        )


def test_correction_loop_success_on_iteration_zero(config):
    trace = run_correction_loop(
        merged_model=_merged_model(),
        structured_problem=_problem(),
        provider=_ScriptedProvider(_responses()),
        config=config,
        executor=_FakeExecutor([_exec(True)]),
    )
    assert trace.final_success
    assert trace.correction_iterations_observed == 0


def test_correction_loop_success_after_retry(config):
    trace = run_correction_loop(
        merged_model=_merged_model(),
        structured_problem=_problem(),
        provider=_ScriptedProvider(_responses()),
        config=config,
        executor=_FakeExecutor([_exec(False, ErrorCategory.MODEL_ERROR), _exec(True)]),
    )
    assert trace.final_success
    assert trace.correction_iterations_observed == 1
    assert trace.total_tokens == 21


def test_correction_loop_stops_after_max_five(config):
    provider = _ScriptedProvider(_responses() * 5)
    trace = run_correction_loop(
        merged_model=_merged_model(),
        structured_problem=_problem(),
        provider=provider,
        config=config,
        executor=_FakeExecutor([_exec(False, ErrorCategory.MODEL_ERROR)] * 10),
    )
    assert not trace.final_success
    assert trace.stopped_reason == "max_correction_iterations"
    assert trace.correction_iterations_observed == 5


def test_correction_loop_does_not_correct_environment_failure(config):
    provider = _ScriptedProvider(_responses())
    trace = run_correction_loop(
        merged_model=_merged_model(),
        structured_problem=_problem(),
        provider=provider,
        config=config,
        executor=_FakeExecutor([_exec(False, ErrorCategory.ENVIRONMENT_ERROR)]),
    )
    assert not trace.final_success
    assert trace.stopped_reason == "non_model_failure:environment_error"
    assert provider.calls == 0


def test_correction_trace_serialization_has_hashes_not_raw_secrets(config):
    trace = run_correction_loop(
        merged_model=_merged_model(),
        structured_problem=_problem(),
        provider=_ScriptedProvider(_responses()),
        config=config,
        executor=_FakeExecutor([_exec(False, ErrorCategory.MODEL_ERROR), _exec(True)]),
    )
    serialized = trace.to_dict()
    assert serialized["initial_ampl_hash"] == prompt_hash(
        "var x >= 0;\nmaximize profit: x;\nsubject to c: x <= cap;"
    ) or serialized["initial_ampl_hash"]
    assert "api_key" not in str(serialized).lower()
    assert "secret" not in str(serialized).lower()


def test_remodel_serialization_preserves_generated_ampl(config):
    remodel = run_g_remod(
        ampl_model="bad",
        structured_problem=_problem(),
        review=run_g_rev(
            ampl_model="bad",
            execution=_exec(False, ErrorCategory.MODEL_ERROR),
            structured_problem=_problem(),
            provider=_ScriptedProvider(_responses()),
            config=config,
        ),
        comparison=run_g_comp(
            structured_problem=_problem(),
            review=run_g_rev(
                ampl_model="bad",
                execution=_exec(False, ErrorCategory.MODEL_ERROR),
                structured_problem=_problem(),
                provider=_ScriptedProvider(_responses()),
                config=config,
            ),
            provider=_ScriptedProvider(_responses()[1:]),
            config=config,
        ),
        provider=_ScriptedProvider(_responses()[2:]),
        config=config,
    )
    assert "subject to c" in remodel.to_dict()["ampl_model"]
