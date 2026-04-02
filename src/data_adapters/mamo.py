"""MAMO adapter (FreedomIntelligence/Mamo).

This adapter expects local JSONL split files under ``data/external/mamo/``.
It is intentionally permissive about raw row keys because upstream formats can vary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]
_SOURCE_URL = "https://github.com/FreedomIntelligence/Mamo"
_KNOWN_SPLITS = ("train", "validation", "dev", "test", "benchmark")


class MAMOAdapter:
    name = "mamo"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=True,
        supports_scalar_instantiation=True,
        supports_solver_eval=False,
        supports_full_formulation=True,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "external" / "mamo")

    def _split_path(self, split_name: str) -> Path:
        return self.data_root / f"{split_name}.jsonl"

    def list_splits(self) -> list[str]:
        return [s for s in _KNOWN_SPLITS if self._split_path(s).exists()]

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        path = self._split_path(split_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing MAMO split at {path}. Run scripts/get_mamo.py or place split JSONL files manually."
            )
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def iter_examples(self, split_name: str) -> Iterable[dict[str, Any]]:
        for row in self.load_split(split_name):
            yield row

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        ex_id = str(example.get("id") or example.get("instance_id") or example.get("uid") or "")
        nl = (
            example.get("nl_query")
            or example.get("query")
            or example.get("problem")
            or example.get("question")
            or ""
        ).strip()
        schema = example.get("schema_id") or example.get("problem_type") or example.get("task")
        formulation = example.get("formulation_text") or example.get("target_model") or example.get("lp")
        scalar = example.get("scalar_gold_params") if isinstance(example.get("scalar_gold_params"), dict) else None
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=schema,
            schema_text=example.get("schema_text") or (str(schema) if schema else None),
            candidate_schemas=example.get("candidate_schemas"),
            scalar_gold_params=scalar,
            structured_gold_params=example.get("structured_gold_params"),
            formulation_text=formulation,
            solver_artifact_path=example.get("solver_artifact_path"),
            metadata={"source_url": _SOURCE_URL, "raw": example},
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        path = self.data_root / "schema_candidates.jsonl"
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self.load_split(split_name):
            ex_id = str(row.get("id") or row.get("instance_id") or row.get("uid") or "")
            if not ex_id:
                continue
            out[ex_id] = {
                "schema_id": row.get("schema_id") or row.get("problem_type") or row.get("task"),
                "scalar_gold_params": row.get("scalar_gold_params"),
                "formulation_text": row.get("formulation_text") or row.get("target_model") or row.get("lp"),
            }
        return out
