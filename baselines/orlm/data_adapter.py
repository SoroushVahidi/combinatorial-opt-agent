"""Deterministic NLP4LP-to-ORLM records and prompt preparation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from baselines.orlm.config import OrlmConfig


@dataclass(frozen=True)
class OrlmInputRecord:
    problem_id: str
    source: str
    raw_text: str
    gold_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def raw_text_sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()

    def to_upstream_example(self, config: OrlmConfig | None = None) -> dict[str, Any]:
        config = config or OrlmConfig()
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "en_question": self.raw_text,
            "prompt": build_orlm_prompt(self.raw_text, config),
            "raw_text_sha256": self.raw_text_sha256,
            "gold_metadata": self.gold_metadata,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class AdapterResult:
    supported: bool
    record: OrlmInputRecord | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "record": self.record.to_upstream_example() if self.record else None,
        }


def adapt_record(
    record: Mapping[str, Any],
    *,
    source: str = "unknown",
    problem_id: str | int | None = None,
) -> AdapterResult:
    """Convert common NLP4LP/catalog shapes without silently dropping rows."""
    candidate_id = problem_id or record.get("problem_id") or record.get("doc_id") or record.get("query_id")
    text = record.get("en_question") or record.get("text") or record.get("query")
    if candidate_id is None:
        return AdapterResult(False, reason="missing_problem_id")
    if not isinstance(text, str) or not text.strip():
        return AdapterResult(False, reason="missing_or_empty_problem_text")
    reserved = {"text", "query", "en_question", "problem_id", "doc_id", "query_id"}
    gold = {str(k): v for k, v in record.items() if k not in reserved}
    return AdapterResult(
        True,
        OrlmInputRecord(
            problem_id=str(candidate_id),
            source=source,
            raw_text=text.strip(),
            gold_metadata=gold,
            source_metadata={"input_keys": sorted(str(k) for k in record)},
        ),
    )


def load_jsonl_records(path: str | Path, *, source: str | None = None) -> list[AdapterResult]:
    """Load rows in stable order, retaining an explicit result for every row."""
    path = Path(path)
    results: list[AdapterResult] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                results.append(AdapterResult(False, reason=f"blank_line:{line_number}"))
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                results.append(AdapterResult(False, reason=f"invalid_json:{line_number}"))
                continue
            if not isinstance(value, dict):
                results.append(AdapterResult(False, reason=f"record_not_object:{line_number}"))
                continue
            results.append(adapt_record(value, source=source or str(path)))
    return results


def build_orlm_prompt(nlp4lp_query: str, config: OrlmConfig | None = None) -> str:
    """Use the exact upstream ``eval/generate.py`` prompt structure."""
    config = config or OrlmConfig()
    if not isinstance(nlp4lp_query, str) or not nlp4lp_query.strip():
        raise ValueError("ORLM question must be a non-empty string")
    question = nlp4lp_query.strip()
    return config.prompt_template.format(Question=question, question=question).strip()
