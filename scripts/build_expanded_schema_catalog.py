#!/usr/bin/env python3
"""Build expanded schema catalog across benchmark, catalog-only, and source-only entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_adapters.registry import create_adapter, list_datasets

# Source URLs for all registered adapters; used to generate dynamic source-only
# fallback catalog entries when no local splits are available for a dataset.
_ADAPTER_SOURCE_URLS: dict[str, str | None] = {
    "cardinal_nl4opt": "https://github.com/CardinalOperations/NL4OPT",
    "complexor": None,
    "gams_models": "https://www.gams.com/latest/gamslib_ml/libhtml/",
    "gurobi_modeling_examples": "https://github.com/Gurobi/modeling-examples",
    "gurobi_optimods": "https://github.com/Gurobi/gurobi-optimods",
    "industryor": "https://github.com/CardinalOperations/IndustryOR",
    "mamo": "https://github.com/FreedomIntelligence/Mamo",
    "miplib": "https://miplib.zib.de/",
    "nl4opt": None,
    "nlp4lp": None,
    "optimus": "https://github.com/teshnizi/OptiMUS",
    "optmath": None,
    "or_library": "http://people.brunel.ac.uk/~mastjjb/jeb/info.html",
    "pyomo_examples": "https://github.com/Pyomo/pyomo",
    "structuredor": "https://github.com/CardinalOperations/StructuredOR",
    "text2zinc": None,
    "cp_bench": "https://github.com/DCP-Bench/DCP-Bench-Open",
}

# Fixed manifest for source-only entries that are *not* registered as adapters
# (e.g., corpus aliases, legacy dataset names).  Registered adapters with no
# local data now get a dynamic fallback entry generated at build time instead.
SOURCE_ONLY_MANIFEST = [
    {
        "id": "source_only_gams_model_library",
        "source_dataset": "gams_model_library",
        "schema_text": "GAMS model library collection; currently tracked via source manifest and catalog metadata.",
        "source_metadata": {
            "source_url": "https://www.gams.com/latest/gamslib_ml/libhtml/",
            "notes": "No raw model corpus vendored.",
        },
        "entry_status": "source-only",
        "benchmark_labeled": False,
    },
    {
        "id": "source_only_miplib_2017_instances",
        "source_dataset": "miplib_2017_instances",
        "schema_text": "MIPLIB 2017 instance corpus (.mps files) tracked as external source; not normalized locally.",
        "source_metadata": {
            "source_url": "https://miplib.zib.de/",
            "notes": "Large raw files not vendored.",
        },
        "entry_status": "source-only",
        "benchmark_labeled": False,
    },
]


def _entry_status(metadata: dict, benchmark_labeled: bool) -> str:
    if metadata.get("catalog_only"):
        return "catalog-only"
    if benchmark_labeled:
        return "benchmark-ready"
    return "source-only"


def collect_schema_entries(dataset_name: str) -> list[dict]:
    adapter = create_adapter(dataset_name)
    entries: list[dict] = []

    for split in adapter.list_splits():
        try:
            for idx, raw_ex in enumerate(adapter.iter_examples(split)):
                ie = adapter.to_internal_example(raw_ex, split)
                if not ie.schema_id and not ie.schema_text:
                    continue

                benchmark_labeled = (
                    ie.scalar_gold_params is not None
                    or ie.formulation_text is not None
                    or ie.solver_artifact_path is not None
                )
                entry = {
                    "id": ie.id or f"{dataset_name}:{split}:{idx}",
                    "source_dataset": ie.source_dataset,
                    "schema_id": ie.schema_id,
                    "schema_text": ie.schema_text,
                    "source_metadata": {
                        "split": split,
                        "source_url": ie.metadata.get("source_url"),
                        "catalog_only": bool(ie.metadata.get("catalog_only", False)),
                        "raw_keys": sorted(raw_ex.keys()) if isinstance(raw_ex, dict) else None,
                    },
                    "benchmark_labeled": benchmark_labeled,
                    "entry_status": _entry_status(ie.metadata, benchmark_labeled),
                    "nl_query": ie.nl_query,
                    "metadata": ie.metadata,
                }
                entries.append(entry)
        except FileNotFoundError:
            continue

    return entries


def _make_dynamic_source_only_entry(dataset_name: str) -> dict:
    """Generate a source-only catalog entry for a registered adapter with no local data."""
    source_url = _ADAPTER_SOURCE_URLS.get(dataset_name)
    return {
        "id": f"source_only_{dataset_name}",
        "source_dataset": dataset_name,
        "schema_text": (
            f"{dataset_name} source registered; local splits not available in this environment."
        ),
        "source_metadata": {"source_url": source_url},
        "entry_status": "source-only",
        "benchmark_labeled": False,
    }


def build_catalog(out_path: Path) -> int:
    all_entries: list[dict] = []
    seen_ids: set[str] = set()

    for dataset_name in list_datasets():
        entries = collect_schema_entries(dataset_name)
        if not entries:
            # No local data: emit a dynamic source-only fallback so every
            # registered adapter appears in the catalog.
            fallback = _make_dynamic_source_only_entry(dataset_name)
            fid = fallback["id"]
            if fid not in seen_ids:
                seen_ids.add(fid)
                all_entries.append(fallback)
            continue

        for entry in entries:
            uid = entry["id"]
            if uid in seen_ids:
                uid = f"{uid}__{entry['source_dataset']}"
                entry["id"] = uid
            seen_ids.add(uid)
            all_entries.append(entry)

    # Append fixed manifest entries for sources not covered by registered adapters.
    for manifest_entry in SOURCE_ONLY_MANIFEST:
        mid = manifest_entry["id"]
        if mid not in seen_ids:
            seen_ids.add(mid)
            all_entries.append(manifest_entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for entry in all_entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(all_entries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build expanded schema catalog from adapters and source-only manifests.")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_schema_catalog.jsonl",
        help="Output JSONL path.",
    )
    args = ap.parse_args()

    print(f"Building expanded schema catalog -> {args.out}")
    n = build_catalog(args.out)
    print(f"Done. Wrote {n} entries.")


if __name__ == "__main__":
    main()
