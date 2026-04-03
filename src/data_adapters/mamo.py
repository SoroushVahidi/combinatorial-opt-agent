"""MAMO adapter (FreedomIntelligence/Mamo).

This adapter expects local JSONL split files under ``data/external/mamo/``.
Run ``scripts/get_mamo.py`` to download.

Source: https://github.com/FreedomIntelligence/Mamo
Splits: train, validation, test (benchmark directory)

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
        p = self._split_path(split_name)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing MAMO split at {p}. Run scripts/get_mamo.py to download."
            )
        rows: list[dict[str, Any]] = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def iter_examples(self, split_name: str) -> Iterable[dict[str, Any]]:
        for row in self.load_split(split_name):
            yield row

    @staticmethod
    def _parse_answer(answer: Any) -> dict[str, Any] | None:
        """Try to extract a numeric objective value from the answer field."""
        if answer is None:
            return None
        if isinstance(answer, (int, float)):
            return {"objective_value": float(answer)}
        if isinstance(answer, str):
            val = answer.strip()
            for converter in (int, float):
                try:
                    return {"objective_value": float(converter(val))}
                except (ValueError, TypeError):
                    pass
        return None

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        ex_id = str(example.get("id") or example.get("instance_id") or example.get("uid") or "")
        nl = (
            example.get("nl_query")
            or example.get("query")
            or example.get("problem")
            or example.get("question")
            or example.get("en_question")
            or ""
        ).strip()
        schema = example.get("schema_id") or example.get("problem_type") or example.get("task")
        formulation = example.get("formulation_text") or example.get("target_model") or example.get("lp")
        if isinstance(example.get("scalar_gold_params"), dict):
            scalar = example["scalar_gold_params"]
        else:
            scalar = self._parse_answer(example.get("en_answer"))
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=str(schema) if schema is not None else None,
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
            scalar = (
                row["scalar_gold_params"]
                if isinstance(row.get("scalar_gold_params"), dict)
                else self._parse_answer(row.get("en_answer"))
            )
            out[ex_id] = {
                "schema_id": row.get("schema_id") or row.get("problem_type") or row.get("task"),
                "scalar_gold_params": scalar,
                "formulation_text": row.get("formulation_text") or row.get("target_model") or row.get("lp"),
            }
        return out
