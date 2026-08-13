"""PaMOP execution/correction loop (`G_exe`, `G_rev`, `G_comp`, `G_remod`).

The paper specifies the correction architecture and equations, but not the
prompt text or response schema. This module therefore uses reconstructed,
versioned JSON prompts and keeps every correction attempt in a serializable
trace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .ampl.executor import AmplExecutor
from .ampl.renderer import render_merged_model
from .ampl.types import AmplExecutionResult, ErrorCategory
from .config import PamopConfig
from .llm.base import LLMProvider, prompt_hash
from .llm.types import LLMResponse
from .llm.types import ModelConfig as LLMModelConfig
from .modeling import MergedModel
from .prompts import PromptTemplate, load_prompt
from .representations import StructuredProblem


class CorrectionValidationError(ValueError):
    """Raised when a correction-stage LLM response is malformed."""


@dataclass(frozen=True)
class ReviewResult:
    diagnosis: str
    error_category: ErrorCategory
    actionable_feedback: tuple[str, ...]
    llm_response: LLMResponse
    prompt_template: PromptTemplate

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "error_category": self.error_category,
            "actionable_feedback": list(self.actionable_feedback),
            "prompt_hash": self.llm_response.prompt_hash,
            "provider": self.llm_response.provider,
            "model": self.llm_response.model,
            "underlying_model": self.llm_response.underlying_model,
            "total_tokens": self.llm_response.total_tokens,
            "latency_seconds": self.llm_response.latency_seconds,
        }


@dataclass(frozen=True)
class ComparisonResult:
    needs_remodel: bool
    reason: str
    targeted_issues: tuple[str, ...]
    llm_response: LLMResponse
    prompt_template: PromptTemplate

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_remodel": self.needs_remodel,
            "reason": self.reason,
            "targeted_issues": list(self.targeted_issues),
            "prompt_hash": self.llm_response.prompt_hash,
            "provider": self.llm_response.provider,
            "model": self.llm_response.model,
            "underlying_model": self.llm_response.underlying_model,
            "total_tokens": self.llm_response.total_tokens,
            "latency_seconds": self.llm_response.latency_seconds,
        }


@dataclass(frozen=True)
class RemodelResult:
    ampl_model: str
    changes: tuple[str, ...]
    llm_response: LLMResponse
    prompt_template: PromptTemplate

    def to_dict(self) -> dict[str, Any]:
        return {
            "ampl_hash": prompt_hash(self.ampl_model),
            # Generated AMPL is the scientifically relevant raw artifact of
            # G_remod. Preserve it in local traces so a failed correction can
            # be audited without replaying an API call. It contains no API
            # credentials; callers must still apply the repository's policy
            # against storing gated problem text or prompts.
            "ampl_model": self.ampl_model,
            "changes": list(self.changes),
            "prompt_hash": self.llm_response.prompt_hash,
            "provider": self.llm_response.provider,
            "model": self.llm_response.model,
            "underlying_model": self.llm_response.underlying_model,
            "total_tokens": self.llm_response.total_tokens,
            "latency_seconds": self.llm_response.latency_seconds,
        }


@dataclass(frozen=True)
class CorrectionIteration:
    iteration_number: int
    ampl_hash: str
    execution: AmplExecutionResult
    review: ReviewResult | None = None
    comparison: ComparisonResult | None = None
    remodel: RemodelResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "ampl_hash": self.ampl_hash,
            "execution": self.execution.to_dict(),
            "review": self.review.to_dict() if self.review else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "remodel": self.remodel.to_dict() if self.remodel else None,
        }


@dataclass(frozen=True)
class CorrectionTrace:
    problem_id: str
    initial_ampl_hash: str
    final_ampl_hash: str
    final_success: bool
    stopped_reason: str
    iterations: tuple[CorrectionIteration, ...] = field(default_factory=tuple)

    @property
    def correction_iterations_observed(self) -> int:
        return sum(1 for item in self.iterations if item.remodel is not None)

    @property
    def total_tokens(self) -> int:
        total = 0
        for iteration in self.iterations:
            for obj in (iteration.review, iteration.comparison, iteration.remodel):
                if obj is not None and obj.llm_response.total_tokens is not None:
                    total += obj.llm_response.total_tokens
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "initial_ampl_hash": self.initial_ampl_hash,
            "final_ampl_hash": self.final_ampl_hash,
            "final_success": self.final_success,
            "stopped_reason": self.stopped_reason,
            "correction_iterations_observed": self.correction_iterations_observed,
            "total_tokens": self.total_tokens,
            "iterations": [i.to_dict() for i in self.iterations],
        }


def run_correction_loop(
    *,
    merged_model: MergedModel,
    structured_problem: StructuredProblem,
    provider: LLMProvider,
    config: PamopConfig,
    executor: AmplExecutor,
) -> CorrectionTrace:
    """Render, execute, and correct an AMPL model until success or budget."""
    render_result = render_merged_model(merged_model)
    current_ampl = render_result.model_text
    initial_hash = render_result.model_hash
    max_corrections = config.require("llm", "max_correction_iterations")
    iterations: list[CorrectionIteration] = []
    corrections_used = 0

    while True:
        execution = executor.execute(current_ampl)
        current_hash = prompt_hash(current_ampl)
        if execution.success:
            iterations.append(CorrectionIteration(len(iterations), current_hash, execution))
            return CorrectionTrace(
                problem_id=merged_model.problem_id,
                initial_ampl_hash=initial_hash,
                final_ampl_hash=current_hash,
                final_success=True,
                stopped_reason="execution_success",
                iterations=tuple(iterations),
            )

        if execution.error_category in {ErrorCategory.ENVIRONMENT_ERROR, ErrorCategory.DATA_ERROR}:
            iterations.append(CorrectionIteration(len(iterations), current_hash, execution))
            return CorrectionTrace(
                problem_id=merged_model.problem_id,
                initial_ampl_hash=initial_hash,
                final_ampl_hash=current_hash,
                final_success=False,
                stopped_reason=f"non_model_failure:{execution.error_category}",
                iterations=tuple(iterations),
            )

        if corrections_used >= max_corrections:
            iterations.append(CorrectionIteration(len(iterations), current_hash, execution))
            return CorrectionTrace(
                problem_id=merged_model.problem_id,
                initial_ampl_hash=initial_hash,
                final_ampl_hash=current_hash,
                final_success=False,
                stopped_reason="max_correction_iterations",
                iterations=tuple(iterations),
            )

        review = run_g_rev(
            ampl_model=current_ampl,
            execution=execution,
            structured_problem=structured_problem,
            provider=provider,
            config=config,
        )
        comparison = run_g_comp(
            structured_problem=structured_problem,
            review=review,
            provider=provider,
            config=config,
        )
        if not comparison.needs_remodel:
            iterations.append(
                CorrectionIteration(len(iterations), current_hash, execution, review, comparison, None)
            )
            return CorrectionTrace(
                problem_id=merged_model.problem_id,
                initial_ampl_hash=initial_hash,
                final_ampl_hash=current_hash,
                final_success=False,
                stopped_reason="comparison_declined_remodel",
                iterations=tuple(iterations),
            )

        remodel = run_g_remod(
            ampl_model=current_ampl,
            structured_problem=structured_problem,
            review=review,
            comparison=comparison,
            provider=provider,
            config=config,
        )
        iterations.append(
            CorrectionIteration(len(iterations), current_hash, execution, review, comparison, remodel)
        )
        current_ampl = remodel.ampl_model
        corrections_used += 1


def run_g_rev(
    *,
    ampl_model: str,
    execution: AmplExecutionResult,
    structured_problem: StructuredProblem,
    provider: LLMProvider,
    config: PamopConfig,
) -> ReviewResult:
    template = load_prompt("correction_review_v1.txt")
    prompt = template.render(
        global_summary=structured_problem.global_summary,
        objective_text=structured_problem.objective_text,
        constraints="\n".join(c.description for c in structured_problem.constraints),
        ampl_model=ampl_model,
        execution_diagnostics=json.dumps(execution.to_dict(), indent=2),
    )
    response = provider.generate(prompt, _model_config(config))
    raw = _parse_json(response.text)
    category = ErrorCategory(raw.get("error_category", ErrorCategory.MODEL_ERROR))
    feedback = raw.get("actionable_feedback", [])
    if not isinstance(raw.get("diagnosis"), str) or not isinstance(feedback, list):
        raise CorrectionValidationError("G_rev response must include diagnosis and actionable_feedback")
    return ReviewResult(
        diagnosis=raw["diagnosis"],
        error_category=category,
        actionable_feedback=tuple(str(item) for item in feedback),
        llm_response=response,
        prompt_template=template,
    )


def run_g_comp(
    *,
    structured_problem: StructuredProblem,
    review: ReviewResult,
    provider: LLMProvider,
    config: PamopConfig,
) -> ComparisonResult:
    template = load_prompt("correction_compare_v1.txt")
    prompt = template.render(
        global_summary=structured_problem.global_summary,
        objective_text=structured_problem.objective_text,
        constraints="\n".join(c.description for c in structured_problem.constraints),
        review=json.dumps(review.to_dict(), indent=2),
    )
    response = provider.generate(prompt, _model_config(config))
    raw = _parse_json(response.text)
    issues = raw.get("targeted_issues", [])
    if not isinstance(raw.get("needs_remodel"), bool) or not isinstance(raw.get("reason"), str):
        raise CorrectionValidationError("G_comp response must include needs_remodel and reason")
    if not isinstance(issues, list):
        raise CorrectionValidationError("G_comp targeted_issues must be a list")
    return ComparisonResult(
        needs_remodel=raw["needs_remodel"],
        reason=raw["reason"],
        targeted_issues=tuple(str(item) for item in issues),
        llm_response=response,
        prompt_template=template,
    )


def run_g_remod(
    *,
    ampl_model: str,
    structured_problem: StructuredProblem,
    review: ReviewResult,
    comparison: ComparisonResult,
    provider: LLMProvider,
    config: PamopConfig,
) -> RemodelResult:
    template = load_prompt("correction_remodel_v1.txt")
    prompt = template.render(
        global_summary=structured_problem.global_summary,
        objective_text=structured_problem.objective_text,
        constraints="\n".join(c.description for c in structured_problem.constraints),
        current_ampl_model=ampl_model,
        review=json.dumps(review.to_dict(), indent=2),
        comparison=json.dumps(comparison.to_dict(), indent=2),
    )
    response = provider.generate(prompt, _model_config(config))
    raw = _parse_json(response.text)
    changes = raw.get("changes", [])
    if not isinstance(raw.get("ampl_model"), str) or not raw["ampl_model"].strip():
        raise CorrectionValidationError("G_remod response must include a non-empty ampl_model")
    if not isinstance(changes, list):
        raise CorrectionValidationError("G_remod changes must be a list")
    return RemodelResult(
        ampl_model=raw["ampl_model"].strip(),
        changes=tuple(str(item) for item in changes),
        llm_response=response,
        prompt_template=template,
    )


def _model_config(config: PamopConfig) -> LLMModelConfig:
    return LLMModelConfig(
        provider=config.require("llm", "provider"),
        model=config.require("llm", "model"),
        temperature=config.require("llm", "temperature"),
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
    )


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    raw = json.loads(stripped)
    if not isinstance(raw, dict):
        raise CorrectionValidationError("correction response must be a JSON object")
    return raw
