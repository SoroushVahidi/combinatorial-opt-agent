"""Stable JSON-friendly per-rollout OR-R1 result record.

One record is emitted per rollout (matching upstream `generate.py`, which
writes one `generated.jsonl` line per sampled completion); group-level
Pass@k/mj@k aggregation is computed separately by `evaluator.py` over all
records sharing a `problem_id`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from baselines.orr1.config import UPSTREAM_REVISION
from baselines.orr1.output_normalizer import OrR1ParsedOutput
from baselines.orr1.runner import GenerationResult
from baselines.orr1.static_validation import StaticValidationResult
from baselines.orr1.tgrpo_controller import RewardComponents, TGRPOTrainingConfig


@dataclass
class OrR1Result:
    problem_id: str
    dataset: str
    input_sha256: str
    model_id: str | None
    model_revision: str | None
    checkpoint_stage: str
    official_repo_sha: str
    prompt_version: str
    prompt_sha256: str
    generation_config: dict[str, Any]
    tgrpo_config: dict[str, Any]
    rollout_index: int
    rollout_count: int
    raw_output: str
    parsed: dict[str, Any] | None = None
    static_validation: dict[str, Any] | None = None
    rewards: dict[str, Any] | None = None
    execution_attempted: bool = False
    execution: dict[str, Any] = field(default_factory=dict)
    solver_status: str | None = None
    objective: float | None = None
    gold_objective: Any = None
    solving_accuracy_status: str = "NOT_EVALUABLE"  # official pass@k semantics: PASS | FAIL | NOT_EVALUABLE
    objective_proxy_status: str = "NOT_EVALUABLE"  # this repo's tolerance proxy, kept distinct from (A)
    runtime_seconds: float | None = None
    token_counts: dict[str, Any] = field(default_factory=dict)
    tgrpo_steps_applied: int = 0
    failure_category: str | None = None
    git_sha: str | None = None
    timestamp_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OrR1Result":
        return cls(**value)

    @classmethod
    def from_generation(
        cls, *, problem_id: str, dataset: str, input_sha256: str, config, rollout_index: int,
        rollout_count: int, raw_output: str, generation: GenerationResult, git_sha: str | None = None,
        timestamp_utc: str | None = None,
    ) -> "OrR1Result":
        return cls(
            problem_id=problem_id, dataset=dataset, input_sha256=input_sha256,
            model_id=config.model_id, model_revision=config.model_revision,
            checkpoint_stage=config.checkpoint_stage, official_repo_sha=UPSTREAM_REVISION,
            prompt_version=config.prompt_version, prompt_sha256=generation.prompt_sha256,
            generation_config=config.generation_dict(), tgrpo_config=TGRPOTrainingConfig().to_dict(),
            rollout_index=rollout_index, rollout_count=rollout_count, raw_output=raw_output,
            runtime_seconds=generation.runtime_seconds, token_counts=dict(generation.token_counts),
            failure_category=generation.error_category, git_sha=git_sha, timestamp_utc=timestamp_utc,
        )
