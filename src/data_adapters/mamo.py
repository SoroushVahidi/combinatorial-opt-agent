"""MAMO adapter – CardinalOperations/MAMO HuggingFace dataset.

Splits: easy_lp, complex_lp.
Data files expected at data/external/mamo/ (JSONL, one row per line).
Run scripts/get_mamo.py to download.

Source: https://huggingface.co/datasets/CardinalOperations/MAMO
License: CC-BY-NC-4.0 (non-commercial use only)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]

_KNOWN_SPLITS = ("easy_lp", "complex_lp")
_SOURCE_URL = "https://huggingface.co/datasets/CardinalOperations/MAMO"


class MAMOAdapter:
    """Adapter for the MAMO LP benchmark (EasyLP and ComplexLP subsets).

    Each example contains a natural-language optimization problem
    (``nl_query``) and an answer string (``en_answer``) representing the
    optimal objective value.  Scalar gold params are populated from the
    answer when it parses as a number; otherwise the raw string is preserved
    in ``metadata``.
    """

    name = "mamo"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=False,
        supports_scalar_instantiation=True,
        supports_solver_eval=False,
        supports_full_formulation=False,
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
        ex_id = str(example.get("id") or "")
        nl = (example.get("nl_query") or example.get("en_question") or "").strip()
        answer = example.get("en_answer")
        scalar = self._parse_answer(answer)
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=None,
            schema_text=None,
            candidate_schemas=None,
            scalar_gold_params=scalar,
            structured_gold_params=None,
            formulation_text=None,
            solver_artifact_path=None,
            metadata={
                "source_url": _SOURCE_URL,
                "license": "CC-BY-NC-4.0",
                "en_answer_raw": answer,
                "mamo_split": split_name,
            },
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        return []

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ex in self.load_split(split_name):
            ex_id = str(ex.get("id") or "")
            if not ex_id:
                continue
            out[ex_id] = {
                "scalar_gold_params": self._parse_answer(ex.get("en_answer")),
            }
        return out
