"""Deterministic NLP4LP-to-OptMATH input adaptation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from baselines.optmath.config import OptmathConfig
from baselines.optmath.prompt import PromptBundle, build_prompt


@dataclass(frozen=True)
class OptmathInputRecord:
    problem_id: str
    dataset: str
    raw_text: str
    source_metadata: dict[str, Any] = field(default_factory=dict)
    gold_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_text_sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()

    def prompt(self, config: OptmathConfig | None = None) -> PromptBundle:
        return build_prompt(self.raw_text, config)

    def to_official_input(self, config: OptmathConfig | None = None) -> dict[str, Any]:
        prompt = self.prompt(config)
        return {
            "problem_id": self.problem_id,
            "dataset": self.dataset,
            "en_question": self.raw_text,
            "en_answer": self.gold_metadata.get("gold_objective"),
            "prompt": prompt.user,
            "system_prompt": prompt.system,
            "raw_text_sha256": self.raw_text_sha256,
            "source_metadata": self.source_metadata,
            "gold_metadata": self.gold_metadata,
        }


@dataclass(frozen=True)
class AdapterResult:
    supported: bool
    record: OptmathInputRecord | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "reason": self.reason, "record": self.record.to_official_input() if self.record else None}


def adapt_record(record: Mapping[str, Any], *, dataset: str = "nlp4lp", problem_id: str | int | None = None) -> AdapterResult:
    candidate_id = problem_id or record.get("problem_id") or record.get("doc_id") or record.get("query_id")
    text = record.get("en_question") or record.get("text") or record.get("query")
    if candidate_id is None:
        return AdapterResult(False, reason="missing_problem_id")
    if not isinstance(text, str) or not text.strip():
        return AdapterResult(False, reason="missing_or_empty_problem_text")
    reserved = {"en_question", "text", "query", "problem_id", "doc_id", "query_id"}
    gold = {str(k): v for k, v in record.items() if k not in reserved}
    if "en_answer" in record and "gold_objective" not in gold:
        gold["gold_objective"] = record["en_answer"]
    return AdapterResult(True, OptmathInputRecord(str(candidate_id), dataset, text.strip(), {"input_keys": sorted(map(str, record))}, gold))


def load_jsonl_records(path: str | Path, *, dataset: str = "nlp4lp") -> list[AdapterResult]:
    results: list[AdapterResult] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                results.append(AdapterResult(False, reason=f"blank_line:{line_no}"))
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                results.append(AdapterResult(False, reason=f"invalid_json:{line_no}"))
                continue
            if not isinstance(value, dict):
                results.append(AdapterResult(False, reason=f"record_not_object:{line_no}"))
                continue
            results.append(adapt_record(value, dataset=dataset))
    return results
