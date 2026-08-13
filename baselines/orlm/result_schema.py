"""Stable JSON-friendly ORLM per-instance result schema."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from baselines.orlm.config import OrlmConfig
from baselines.orlm.output_normalizer import OrlmParsedOutput
from baselines.orlm.runner import GenerationResult
from baselines.orlm.static_validation import StaticValidationResult


@dataclass
class OrlmResult:
    problem_id: str
    dataset: str
    raw_problem_text_sha256: str
    prompt_version: str
    prompt_sha256: str
    generation: GenerationResult
    parsed: OrlmParsedOutput | None = None
    static_validation: StaticValidationResult | None = None
    execution_attempted: bool = False
    execution: dict[str, Any] = field(default_factory=dict)
    gold_objective: float | None = None
    objective_value: float | None = None
    objective_proxy_status: str = "NOT_EVALUABLE"
    semantic_evaluation_status: str = "NOT_EVALUABLE"
    error_category: str | None = None
    git_sha: str | None = None
    timestamp_utc: str | None = None

    @classmethod
    def from_generation(cls, problem_id: str, dataset: str, text_hash: str, prompt_version: str, generation: GenerationResult) -> "OrlmResult":
        return cls(problem_id, dataset, text_hash, prompt_version, generation.prompt_sha256, generation, error_category=generation.error_category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "dataset": self.dataset,
            "raw_problem_text_sha256": self.raw_problem_text_sha256,
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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OrlmResult":
        generation_data = value["generation"]
        generation = GenerationResult(**generation_data)
        parsed_data = value.get("parsed")
        parsed = None
        if parsed_data:
            parsed = OrlmParsedOutput(
                raw_output=parsed_data["raw_output"], model_description=parsed_data["model_description"],
                coptpy_code=parsed_data.get("coptpy_code"), code_block_found=parsed_data.get("code_block_found", False),
                code_blocks_seen=parsed_data.get("code_blocks_seen", 0), selected_block_index=parsed_data.get("selected_block_index"),
                warnings=tuple(parsed_data.get("warnings", [])), parser_status=parsed_data.get("parser_status", "EMPTY"),
            )
        static_data = value.get("static_validation")
        static = None
        if static_data:
            static = StaticValidationResult(
                status=static_data["status"], python_syntax_valid=static_data["python_syntax_valid"],
                coptpy_import_present=static_data["coptpy_import_present"], model_creation_present=static_data["model_creation_present"],
                objective_present=static_data["objective_present"], optimize_call_present=static_data["optimize_call_present"],
                constraint_signal_present=static_data["constraint_signal_present"], suspicious_empty_model=static_data["suspicious_empty_model"],
                dangerous_operations=tuple(static_data.get("dangerous_operations", [])), unsupported_imports=tuple(static_data.get("unsupported_imports", [])),
                possible_undefined_names=tuple(static_data.get("possible_undefined_names", [])),
                warnings=tuple(static_data.get("warnings", [])), errors=tuple(static_data.get("errors", [])),
            )
        return cls(
            problem_id=value["problem_id"], dataset=value["dataset"], raw_problem_text_sha256=value["raw_problem_text_sha256"],
            prompt_version=value["prompt_version"], prompt_sha256=value["prompt_sha256"], generation=generation,
            parsed=parsed, static_validation=static,
            execution_attempted=value.get("execution_attempted", False), execution=value.get("execution", {}),
            gold_objective=value.get("gold_objective"), objective_value=value.get("objective_value"),
            objective_proxy_status=value.get("objective_proxy_status", "NOT_EVALUABLE"),
            semantic_evaluation_status=value.get("semantic_evaluation_status", "NOT_EVALUABLE"),
            error_category=value.get("error_category"), git_sha=value.get("git_sha"), timestamp_utc=value.get("timestamp_utc"),
        )
