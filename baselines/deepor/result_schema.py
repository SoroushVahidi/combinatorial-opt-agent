"""Stable JSON-friendly DeepOR result record."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any
from .runner import GenerationResult
from .output_normalizer import ParsedOutput
from .static_validation import StaticValidation

@dataclass
class DeepORResult:
    problem_id: str; dataset: str; input_sha256: str; model_id: str | None; model_revision: str | None; paper_revision: str; prompt_version: str; prompt_sha256: str; generation: GenerationResult; parsed: ParsedOutput | None = None; static_validation: StaticValidation | None = None; execution_attempted: bool = False; execution: dict[str, Any] = field(default_factory=dict); gold_objective: Any = None; objective: float | None = None; objective_proxy_status: str = "NOT_EVALUABLE"; semantic_evaluation_status: str = "NOT_EVALUABLE"; failure_category: str | None = None; git_sha: str | None = None; timestamp_utc: str | None = None
    def to_dict(self):
        d=self.__dict__.copy(); d["generation"]=self.generation.to_dict(); d["parsed"]=self.parsed.to_dict() if self.parsed else None; d["static_validation"]=self.static_validation.to_dict() if self.static_validation else None; return d
    def to_json(self): return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
    @classmethod
    def from_dict(cls, d):
        return cls(**{**d, "generation": GenerationResult(**d["generation"])})
