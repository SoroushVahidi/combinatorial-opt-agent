"""Deterministic NLP4LP-to-OR-R1 record and prompt preparation.

Upstream OR-R1 examples are flat `{"question": ..., "answer": ...}` objects
(see `datasets/trainset/train_all.jsonl` and `datasets/testset/nlp4lp.jsonl`,
which additionally carry `ori`/`index` provenance fields). This adapter
preserves the original NLP4LP id, raw text verbatim, and a stable SHA-256
input hash; it never silently drops a record.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from baselines.orr1.config import OrR1Config, ORR1_PROMPT_TEMPLATE


@dataclass(frozen=True)
class OrR1InputRecord:
    problem_id: str
    dataset: str
    raw_text: str
    gold_metadata: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()

    def to_upstream_example(self) -> dict[str, Any]:
        """The flat `{"question", "answer"}` shape used by upstream jsonl files."""
        return {
            "question": self.raw_text,
            "answer": self.gold_metadata.get("gold_objective"),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "dataset": self.dataset,
            "raw_text": self.raw_text,
            "input_sha256": self.input_sha256,
            "gold_metadata": self.gold_metadata,
            "source_metadata": self.source_metadata,
        }


@dataclass(frozen=True)
class AdapterResult:
    supported: bool
    record: OrR1InputRecord | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "record": self.record.to_dict() if self.record else None,
        }


def adapt_record(
    record: Mapping[str, Any],
    *,
    dataset: str = "nlp4lp",
    problem_id: str | int | None = None,
) -> AdapterResult:
    if not isinstance(record, Mapping):
        return AdapterResult(False, reason="record_not_mapping")
    candidate_id = problem_id if problem_id is not None else record.get("problem_id", record.get("doc_id", record.get("query_id")))
    text = record.get("question", record.get("en_question", record.get("text", record.get("query"))))
    if candidate_id is None:
        return AdapterResult(False, reason="missing_problem_id")
    if not isinstance(text, str) or not text.strip():
        return AdapterResult(False, reason="missing_or_empty_problem_text")
    reserved = {"problem_id", "doc_id", "query_id", "question", "en_question", "text", "query"}
    gold = {str(k): v for k, v in record.items() if k not in reserved}
    if "answer" in record:
        gold.setdefault("gold_objective", record["answer"])
    elif "en_answer" in record:
        gold.setdefault("gold_objective", record["en_answer"])
    return AdapterResult(
        True,
        OrR1InputRecord(
            problem_id=str(candidate_id),
            dataset=dataset,
            raw_text=text.strip(),
            gold_metadata=gold,
            source_metadata={"input_keys": sorted(str(k) for k in record)},
        ),
    )


def load_jsonl_records(path: str | Path, *, dataset: str | None = None) -> list[AdapterResult]:
    """Load rows in stable order; every row gets an explicit result, none dropped."""
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
            results.append(adapt_record(value, dataset=dataset or str(path)))
    return results


def build_orr1_prompt(nlp4lp_query: str, config: OrR1Config | None = None) -> str:
    """The literal `TEMPLATE_q2mc_en` structure from `02_grpo_train.py` / `eval/generate.py`.

    Upstream substitutes with `str.replace("{Question}", ...)`, not `str.format`,
    so a question containing literal `{`/`}` characters is inserted verbatim
    rather than raising or being reinterpreted; this mirrors that exactly.
    Upstream additionally wraps this string with `tokenizer.apply_chat_template`
    before generation; that step requires an actual tokenizer/chat template and
    is applied by the runner, not here (see `OrR1Config.requires_chat_template`).
    """
    config = config or OrR1Config()
    if not isinstance(nlp4lp_query, str) or not nlp4lp_query.strip():
        raise ValueError("OR-R1 question must be a non-empty string")
    question = nlp4lp_query.strip()
    return config.prompt_template.replace("{Question}", question).strip()
