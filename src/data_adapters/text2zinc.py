"""Text2Zinc adapter (supports manual or scripted local snapshots)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]


class Text2ZincAdapter:
    name = "text2zinc"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=False,
        supports_scalar_instantiation=True,
        supports_solver_eval=True,
        supports_full_formulation=True,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "external" / "text2zinc")

    def _split_path(self, split_name: str) -> Path:
        return self.data_root / f"{split_name}.jsonl"

    def list_splits(self) -> list[str]:
        return [s for s in ("train", "validation", "test") if self._split_path(s).exists()]

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        p = self._split_path(split_name)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing Text2Zinc split at {p}. Run scripts/get_text2zinc.py and follow gated access instructions."
            )
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
        input_obj = example.get("input") if isinstance(example.get("input"), dict) else {}
        meta = input_obj.get("metadata") if isinstance(input_obj.get("metadata"), dict) else {}
        ex_id = str(meta.get("identifier") or example.get("id") or "")
        nl = (input_obj.get("description") or example.get("description") or "").strip()
        scalar = example.get("scalar_gold_params")
        if not isinstance(scalar, dict):
            scalar = None
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=str(meta.get("source")) if meta.get("source") else None,
            schema_text=meta.get("name"),
            candidate_schemas=None,
            scalar_gold_params=scalar,
            structured_gold_params=input_obj if input_obj else None,
            formulation_text=example.get("model_mzn"),
            solver_artifact_path=example.get("data_dzn_path"),
            metadata={"raw": example},
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        return []

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ex in self.load_split(split_name):
            iex = self.to_internal_example(ex, split_name)
            out[iex.id] = {
                "scalar_gold_params": iex.scalar_gold_params,
                "formulation_text": iex.formulation_text,
            }
        return out

