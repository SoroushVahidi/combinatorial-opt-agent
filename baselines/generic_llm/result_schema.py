"""Result schema for the GENERAL_PURPOSE_LLM_BASELINE.

Mirrors the OptMATH/ORLM result shape so the shared comparison harness can
consume it via the same `_adapt_coptpy_gurobi_family` adapter family. Parse
and static validation reuse the OptMATH gurobipy tools so metrics are
identical in meaning across OptMATH and this baseline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from baselines.generic_llm.runner import GenerationResult
from baselines.optmath.output_normalizer import ParsedOutput
from baselines.optmath.static_validation import StaticValidation


@dataclass
class GenericLLMResult:
    problem_id: str
    dataset: str
    input_sha256: str
    checkpoint: str
    checkpoint_revision: str | None
    prompt_version: str
    prompt_sha256: str
    generation: GenerationResult
    parsed: ParsedOutput | None = None
    static_validation: StaticValidation | None = None
    execution_attempted: bool = False
    execution: dict[str, Any] = field(default_factory=dict)
    gold_objective: float | str | None = None
    objective_value: float | None = None
    objective_proxy_status: str = "NOT_EVALUABLE"
    semantic_evaluation_status: str = "NOT_EVALUABLE"
    error_category: str | None = None
    git_sha: str | None = None
    timestamp_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "dataset": self.dataset,
            "input_sha256": self.input_sha256,
            "checkpoint": self.checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "prompt_version": self.prompt_version,
            "prompt_sha256": self.prompt_sha256,
            "generation": self.generation.to_dict(),
            "parsed": self.parsed.to_dict() if self.parsed else None,
            "static_validation": self.static_validation.to_dict() if self.static_validation else None,
            "execution_attempted": self.execution_attempted,
            "execution": self.execution,
            "gold_objective": self.gold_objective,
            "objective_value": self.objective_value,
            "objective_proxy_status": self.objective_proxy_status,
            "semantic_evaluation_status": self.semantic_evaluation_status,
            "error_category": self.error_category,
            "git_sha": self.git_sha,
            "timestamp_utc": self.timestamp_utc,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)