"""Gurobi OptiMods catalog adapter (catalog-only, no labeled queries)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]

_SOURCE_URL = "https://github.com/Gurobi/gurobi-optimods"


class GurobiOptimodsAdapter:
    """Catalog adapter for the Gurobi OptiMods library.

    Each entry corresponds to one optimization module (mod) in the library.
    All entries are emitted under the ``"catalog"`` split; no NL-query /
    gold-parameter benchmarking is supported.
    """

    name = "gurobi_optimods"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=False,
        supports_scalar_instantiation=False,
        supports_solver_eval=False,
        supports_full_formulation=False,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "sources")
        self._source_path = self.data_root / "gurobi_optimods.json"

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
            {"mod": mod, "source_file": str(self._source_path)}
            for mod in manifest.get("mods", [])
        ]

    def iter_examples(self, split_name: str) -> Iterable[dict[str, Any]]:
        for row in self.load_split(split_name):
            yield row

    def to_internal_example(self, example: dict[str, Any], split_name: str) -> InternalExample:
        mod: str = example["mod"]
        human_name = mod.replace("_", " ").replace("-", " ").title()
        return InternalExample(
            id=f"gurobi_optimods_{mod}",
            source_dataset=self.name,
            split="catalog",
            nl_query=human_name,
            schema_id=mod,
            schema_text=f"Gurobi OptiMod: {human_name}",
            candidate_schemas=None,
            scalar_gold_params=None,
            structured_gold_params=None,
            formulation_text=None,
            solver_artifact_path=None,
            metadata={
                "source_url": _SOURCE_URL,
                "source_file": example.get("source_file", str(self._source_path)),
                "catalog_only": True,
                "entry_type": "optimod",
            },
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        return []

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        return {}
