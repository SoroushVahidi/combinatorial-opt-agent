"""CP-Bench adapter for DCP-Bench-Open style exports (CPMPy reference code).

In this repository, **cp_bench** refers to the public **DCP-Bench-Open** corpus
(https://github.com/DCP-Bench/DCP-Bench-Open), Apache-2.0. Upstream uses the
**DCP-Bench** name; we keep the registry key ``cp_bench`` for short, stable imports.

The bundled ``sample_test.jsonl`` (and typical JSONL exports) contain **problem id**
and **reference CPMPy Python**, not natural-language descriptions comparable to NLP4LP.
Do not assume the NLP4LP retrieval + scalar-grounding pipeline applies without
additional problem formalization work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .base import DatasetCapabilities, InternalExample

ROOT = Path(__file__).resolve().parents[2]


class CPBenchAdapter:
    name = "cp_bench"
    capabilities = DatasetCapabilities(
        supports_schema_retrieval=False,
        supports_scalar_instantiation=False,
        supports_solver_eval=False,
        supports_full_formulation=True,
    )

    def __init__(self, data_root: Path | None = None) -> None:
        self.data_root = data_root or (ROOT / "data" / "external" / "cp_bench")

    def _jsonl_paths(self) -> list[Path]:
        if not self.data_root.is_dir():
            return []
        return sorted(self.data_root.glob("*.jsonl"))

    def list_splits(self) -> list[str]:
        return [p.stem for p in self._jsonl_paths()]

    def load_split(self, split_name: str) -> list[dict[str, Any]]:
        p = self.data_root / f"{split_name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing CP-Bench (DCP-Bench-Open) split at {p}. "
                f"Run: python scripts/datasets/get_cp_bench_open.py"
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
        ex_id = str(example.get("id") or example.get("problem_id") or "")
        model = example.get("model") or example.get("reference_code") or ""
        model = model if isinstance(model, str) else ""
        nl = (
            example.get("natural_language")
            or example.get("nl")
            or example.get("question")
            or example.get("description")
        )
        nl = nl.strip() if isinstance(nl, str) else ""
        return InternalExample(
            id=ex_id,
            source_dataset=self.name,
            split=split_name,
            nl_query=nl,
            schema_id=None,
            schema_text=None,
            candidate_schemas=None,
            scalar_gold_params=None,
            structured_gold_params={k: v for k, v in example.items() if k not in ("id", "model")},
            formulation_text=model or None,
            solver_artifact_path=None,
            metadata={
                "upstream": "DCP-Bench-Open",
                "upstream_url": "https://github.com/DCP-Bench/DCP-Bench-Open",
                "has_natural_language": bool(nl),
                "raw": example,
            },
        )

    def get_schema_candidates(self) -> list[dict[str, Any]]:
        return []

    def get_gold_targets(self, split_name: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for ex in self.load_split(split_name):
            iex = self.to_internal_example(ex, split_name)
            out[iex.id] = {
                "formulation_text": iex.formulation_text,
            }
        return out
