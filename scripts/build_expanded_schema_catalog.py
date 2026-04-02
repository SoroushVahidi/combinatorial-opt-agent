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
            raw = list(adapter.iter_examples(split))
        except FileNotFoundError:
            continue

        for raw_ex in raw:
            ie = adapter.to_internal_example(raw_ex, split)
            if not ie.schema_id and not ie.schema_text:
                continue

            benchmark_labeled = (
                ie.scalar_gold_params is not None
                or ie.formulation_text is not None
                or ie.solver_artifact_path is not None
            )
            entry = {
                "id": ie.id or f"{dataset_name}:{split}",
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

    return entries


def build_catalog(out_path: Path) -> int:
    all_entries: list[dict] = []
    seen_ids: set[str] = set()

    for dataset_name in list_datasets():
        entries = collect_schema_entries(dataset_name)
        if not entries:
            all_entries.append(
                {
                    "id": f"source_only_{dataset_name}",
                    "source_dataset": dataset_name,
                    "schema_id": None,
                    "schema_text": None,
                    "source_metadata": {
                        "notes": "Adapter registered but no local split data discovered.",
                    },
                    "benchmark_labeled": False,
                    "entry_status": "source-only",
                    "nl_query": None,
                    "metadata": {"adapter_registered": True, "no_local_data": True},
                }
            )
            continue

        for entry in entries:
            uid = entry["id"]
            if uid in seen_ids:
                uid = f"{uid}__{entry['source_dataset']}"
                entry["id"] = uid
            seen_ids.add(uid)
            all_entries.append(entry)

    all_entries.extend(SOURCE_ONLY_MANIFEST)

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
