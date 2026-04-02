"""NLP4LP adapter backed by local processed files and catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]


class NLP4LPAdapter:
    name = "nlp4lp"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=True,
        supports_scalar_instantiation=True,
        supports_solver_eval=False,
        supports_full_formulation=False,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data")

    def list_splits(self) -> list[str]:
        splits: list[str] = []
        for split in ("orig", "noisy", "short"):
            if (self.data_root / "processed" / f"nlp4lp_eval_{split}.jsonl").exists():
                splits.append(split)
        return splits

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        p = self.data_root / "processed" / f"nlp4lp_eval_{split_name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
        rows: list[dict[str, Any]] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def iter_examples(self, split_name: str) -> Iterable[dict[str, Any]]:
        for row in self.load_split(split_name):
            yield row

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        return InternalExample(
            id=str(example.get("query_id") or ""),
            source_dataset=self.name,
            split=split_name,
            nl_query=(example.get("query") or "").strip(),
            schema_id=example.get("relevant_doc_id"),
            schema_text=None,
            candidate_schemas=None,
            scalar_gold_params=None,
            structured_gold_params=None,
            formulation_text=None,
            solver_artifact_path=None,
            metadata={"raw": example},
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        p = self.data_root / "catalogs" / "nlp4lp_catalog.jsonl"
        if not p.exists():
            return []
        out: list[dict[str, Any]] = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        targets: dict[str, dict[str, Any]] = {}
        for row in self.load_split(split_name):
            qid = str(row.get("query_id") or "")
            if qid:
                targets[qid] = {"schema_id": row.get("relevant_doc_id")}
        return targets

