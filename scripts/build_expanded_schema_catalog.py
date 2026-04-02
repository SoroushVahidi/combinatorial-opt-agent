#!/usr/bin/env python3
"""Build an expanded schema catalog by collecting normalized schema entries from all adapters.

Reads from all registered dataset adapters and collects schema entries.
Outputs a merged JSONL file at data/processed/expanded_schema_catalog.jsonl.

Each output row includes:
- id: unique identifier
- source_dataset: adapter name
- schema_id: schema/problem identifier
- schema_text: human-readable schema description
- source_url: original upstream URL (from metadata)
- catalog_only: whether the entry has no benchmark labels
- benchmark_labeled: whether the entry has gold labels
- nl_query: natural language query (if available)
- metadata: provenance metadata
"""

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


def _is_benchmark_labeled(entry_metadata: dict) -> bool:
    return not entry_metadata.get("catalog_only", False)


def collect_schema_entries(dataset_name: str) -> list[dict]:
    """Collect schema entries from a single adapter."""
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

            catalog_only = ie.metadata.get("catalog_only", False)
            entry = {
                "id": ie.id,
                "source_dataset": ie.source_dataset,
                "schema_id": ie.schema_id,
                "schema_text": ie.schema_text,
                "source_url": ie.metadata.get("source_url"),
                "catalog_only": catalog_only,
                "benchmark_labeled": not catalog_only and (
                    ie.scalar_gold_params is not None
                    or ie.formulation_text is not None
                    or ie.solver_artifact_path is not None
                ),
                "nl_query": ie.nl_query,
                "metadata": ie.metadata,
            }
            entries.append(entry)

    return entries


def build_catalog(out_path: Path) -> int:
    """Build the expanded schema catalog and write to out_path. Returns row count."""
    all_entries: list[dict] = []
    seen_ids: set[str] = set()

    for dataset_name in list_datasets():
        entries = collect_schema_entries(dataset_name)
        for entry in entries:
            uid = entry["id"]
            if uid in seen_ids:
                # Mark as duplicate but keep it with provenance
                entry["metadata"]["duplicate_of"] = uid
                uid = f"{uid}__{entry['source_dataset']}"
                entry["id"] = uid
            seen_ids.add(uid)
            all_entries.append(entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(all_entries)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build expanded schema catalog from all normalized adapters."
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed" / "expanded_schema_catalog.jsonl",
        help="Output JSONL path (default: data/processed/expanded_schema_catalog.jsonl)",
    )
    args = ap.parse_args()

    print(f"Building expanded schema catalog → {args.out}")
    n = build_catalog(args.out)
    print(f"Done. Wrote {n} schema entries.")


if __name__ == "__main__":
    main()
