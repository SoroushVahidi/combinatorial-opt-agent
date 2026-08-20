"""Deterministic NLP4LP adaptation; no records are silently discarded."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class DeepORInputRecord:
    problem_id: str
    dataset: str
    raw_text: str
    source_metadata: dict[str, Any] = field(default_factory=dict)
    gold_metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def input_sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()
    def to_dict(self) -> dict[str, Any]:
        return {"problem_id": self.problem_id, "dataset": self.dataset,
                "raw_text": self.raw_text, "input_sha256": self.input_sha256,
                "source_metadata": self.source_metadata,
                "gold_metadata": self.gold_metadata}

@dataclass(frozen=True)
class AdapterResult:
    supported: bool
    record: DeepORInputRecord | None = None
    reason: str | None = None
    def to_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "reason": self.reason,
                "record": self.record.to_dict() if self.record else None}

def adapt_record(record: Mapping[str, Any], *, dataset: str = "nlp4lp",
                 problem_id: str | int | None = None) -> AdapterResult:
    if not isinstance(record, Mapping):
        return AdapterResult(False, reason="record_not_mapping")
    pid = problem_id if problem_id is not None else record.get("problem_id", record.get("doc_id", record.get("query_id")))
    text = record.get("en_question", record.get("text", record.get("query")))
    if pid is None: return AdapterResult(False, reason="missing_problem_id")
    if not isinstance(text, str) or not text.strip():
        return AdapterResult(False, reason="missing_or_empty_problem_text")
    reserved = {"problem_id", "doc_id", "query_id", "en_question", "text", "query"}
    gold = {str(k): v for k, v in record.items() if k not in reserved}
    if "en_answer" in record: gold.setdefault("gold_objective", record["en_answer"])
    return AdapterResult(True, DeepORInputRecord(str(pid), dataset, text,
        {"input_keys": sorted(map(str, record.keys()))}, gold))

def load_jsonl(path: str | Path, *, dataset: str = "nlp4lp") -> list[AdapterResult]:
    out = []
    for no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): out.append(AdapterResult(False, reason=f"blank_line:{no}")); continue
        try: value = json.loads(line)
        except json.JSONDecodeError: out.append(AdapterResult(False, reason=f"invalid_json:{no}")); continue
        out.append(adapt_record(value, dataset=dataset))
    return out
