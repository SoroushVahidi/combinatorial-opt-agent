"""Deterministic JSON-friendly OptMATH per-instance result schema."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from baselines.optmath.output_normalizer import ParsedOutput
from baselines.optmath.runner import GenerationResult
from baselines.optmath.static_validation import StaticValidation


@dataclass
class OptmathResult:
    problem_id: str
    dataset: str
    input_sha256: str
    checkpoint: str
    checkpoint_revision: str | None
    upstream_revision: str
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
        return {"problem_id": self.problem_id, "dataset": self.dataset, "input_sha256": self.input_sha256, "checkpoint": self.checkpoint, "checkpoint_revision": self.checkpoint_revision, "upstream_revision": self.upstream_revision, "prompt_version": self.prompt_version, "prompt_sha256": self.prompt_sha256, "generation": self.generation.to_dict(), "parsed": self.parsed.to_dict() if self.parsed else None, "static_validation": self.static_validation.to_dict() if self.static_validation else None, "execution_attempted": self.execution_attempted, "execution": self.execution, "gold_objective": self.gold_objective, "objective_value": self.objective_value, "objective_proxy_status": self.objective_proxy_status, "semantic_evaluation_status": self.semantic_evaluation_status, "error_category": self.error_category, "git_sha": self.git_sha, "timestamp_utc": self.timestamp_utc}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OptmathResult":
        generation = GenerationResult(**value["generation"])
        parsed_data = value.get("parsed")
        parsed = ParsedOutput(**{**parsed_data, "warnings": tuple(parsed_data.get("warnings", []))}) if parsed_data else None
        validation_data = value.get("static_validation")
        validation = StaticValidation(**{**validation_data, "dangerous_operations": tuple(validation_data.get("dangerous_operations", [])), "unsupported_imports": tuple(validation_data.get("unsupported_imports", [])), "possible_undefined_names": tuple(validation_data.get("possible_undefined_names", [])), "warnings": tuple(validation_data.get("warnings", [])), "errors": tuple(validation_data.get("errors", []))}) if validation_data else None
        return cls(value["problem_id"], value["dataset"], value["input_sha256"], value["checkpoint"], value.get("checkpoint_revision"), value["upstream_revision"], value["prompt_version"], value["prompt_sha256"], generation, parsed, validation, value.get("execution_attempted", False), value.get("execution", {}), value.get("gold_objective"), value.get("objective_value"), value.get("objective_proxy_status", "NOT_EVALUABLE"), value.get("semantic_evaluation_status", "NOT_EVALUABLE"), value.get("error_category"), value.get("git_sha"), value.get("timestamp_utc"))
