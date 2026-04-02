"""ComplexOR adapter (closest public variant supported)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]


class ComplexORAdapter:
    name = "complexor"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=False,
        supports_scalar_instantiation=True,
        supports_solver_eval=True,
        supports_full_formulation=True,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "external" / "complexor")

    def _split_path(self, split_name: str) -> Path:
        return self.data_root / f"{split_name}.jsonl"

    def list_splits(self) -> list[str]:
        return [s for s in ("train", "validation", "test") if self._split_path(s).exists()]

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        p = self._split_path(split_name)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing ComplexOR split at {p}. Run scripts/get_complexor.py and follow source guidance."
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
        ex_id = str(example.get("id") or example.get("instance_id") or "")
        nl = (example.get("nl_query") or example.get("description") or example.get("text") or "").strip()
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=example.get("problem_family"),
            schema_text=example.get("problem_family"),
            candidate_schemas=None,
            scalar_gold_params=example.get("scalar_gold_params"),
            structured_gold_params=example.get("structured_gold_params"),
            formulation_text=example.get("formulation_text") or example.get("model_mzn"),
            solver_artifact_path=example.get("solver_artifact_path"),
            metadata={"raw": example},
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        return []

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ex in self.load_split(split_name):
            ex_id = str(ex.get("id") or ex.get("instance_id") or "")
            if not ex_id:
                continue
            out[ex_id] = {
                "scalar_gold_params": ex.get("scalar_gold_params"),
                "formulation_text": ex.get("formulation_text") or ex.get("model_mzn"),
            }
        return out

