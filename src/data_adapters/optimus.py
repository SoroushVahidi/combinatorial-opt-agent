"""OptiMUS adapter – NL → MILP dataset (NLP4LP / OptiMUS, 269 problems).

Data files are expected at ``data/external/optimus/`` as JSONL splits
(e.g. ``train.jsonl``, ``test.jsonl``).  When no files are present the
adapter reports zero available splits and returns empty results gracefully.

Source: https://github.com/teshnizi/OptiMUS
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]

_SOURCE_URL = "https://github.com/teshnizi/OptiMUS"
_KNOWN_SPLITS = ("train", "validation", "test")


class OptiMUSAdapter:
    """Adapter for the OptiMUS NL-to-MILP benchmark dataset.

    Reads JSONL files from ``data/external/optimus/``.  Falls back
    gracefully to zero examples when no data files are present.
    """

    name = "optimus"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=True,
        supports_scalar_instantiation=True,
        supports_solver_eval=False,
        supports_full_formulation=True,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "external" / "optimus")

    def _split_path(self, split_name: str) -> Path:
        return self.data_root / f"{split_name}.jsonl"

    def list_splits(self) -> list[str]:
        return [s for s in _KNOWN_SPLITS if self._split_path(s).exists()]

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        p = self._split_path(split_name)
        if not p.exists():
            raise FileNotFoundError(
                f"Missing OptiMUS split at {p}. "
                "Download the dataset from https://github.com/teshnizi/OptiMUS "
                "and place JSONL files under data/external/optimus/."
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

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        ex_id = str(
            example.get("id")
            or example.get("instance_id")
            or ""
        )
        nl = (
            example.get("nl_query")
            or example.get("problem")
            or example.get("text")
            or ""
        ).strip()
        formulation = (
            example.get("formulation_text")
            or example.get("target_model")
            or None
        )
        scalar_params: dict[str, Any] | None = example.get("scalar_gold_params") or None
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=example.get("problem_type") or None,
            schema_text=example.get("problem_type") or None,
            candidate_schemas=None,
            scalar_gold_params=scalar_params,
            structured_gold_params=example.get("structured_gold_params") or None,
            formulation_text=formulation,
            solver_artifact_path=example.get("solver_artifact_path") or None,
            metadata={
                "source_url": _SOURCE_URL,
                "raw": example,
            },
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
                "formulation_text": ex.get("formulation_text") or ex.get("target_model"),
            }
        return out
