"""Pyomo Examples catalog adapter (catalog-only, no labeled queries)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]

_SOURCE_URL = "https://github.com/Pyomo/pyomo"


class PyomoExamplesAdapter:
    """Catalog adapter for the Pyomo example scripts.

    Each entry corresponds to one example in the Pyomo repository.
    All entries are emitted under the ``"catalog"`` split; no NL-query /
    gold-parameter benchmarking is supported.
    """

    name = "pyomo_examples"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=False,
        supports_scalar_instantiation=False,
        supports_solver_eval=False,
        supports_full_formulation=False,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "sources")
        self._source_path = self.data_root / "pyomo_examples.json"

    def _load_manifest(self) -> dict[str, Any]:
        with open(self._source_path, encoding="utf-8") as fh:
            return json.load(fh)

    def list_splits(self) -> list[str]:
        return ["catalog"]

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        if split_name != "catalog":
            raise ValueError(f"Unknown split '{split_name}'. Only 'catalog' is available.")
        manifest = self._load_manifest()
        return [
            {"example": example, "source_file": str(self._source_path)}
            for example in manifest.get("examples", [])
        ]

    def iter_examples(self, split_name: str) -> Iterable[dict[str, Any]]:
        for row in self.load_split(split_name):
            yield row

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        ex_name: str = example["example"]
        human_name = ex_name.replace("_", " ").replace("-", " ").title()
        return InternalExample(
            id=f"pyomo_examples_{ex_name}",
            source_dataset=self.name,
            split="catalog",
            nl_query=human_name,
            schema_id=ex_name,
            schema_text=f"Pyomo example: {human_name}",
            candidate_schemas=None,
            scalar_gold_params=None,
            structured_gold_params=None,
            formulation_text=None,
            solver_artifact_path=None,
            metadata={
                "source_url": _SOURCE_URL,
                "source_file": example.get("source_file", str(self._source_path)),
                "catalog_only": True,
                "entry_type": "python_example",
            },
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        return []

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        return {}
