#!/usr/bin/env python3
"""Export lightweight grounding training pairs from normalized adapters where feasible."""

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


def _extract_pairs(dataset_name: str) -> tuple[list[dict], list[str]]:
    adapter = create_adapter(dataset_name)
    pairs: list[dict] = []
    blockers: list[str] = []

    splits = adapter.list_splits()
    if not splits:
        blockers.append("no local splits available")
        return pairs, blockers

    for split in splits:
        for idx, raw in enumerate(adapter.iter_examples(split)):
            ie = adapter.to_internal_example(raw, split)
            example_id = ie.id or f"{split}-{idx}"
            if not ie.nl_query:
                blockers.append(f"{split}:{example_id} missing nl_query")
                continue
            if not ie.schema_id and not ie.schema_text:
                blockers.append(f"{split}:{example_id} missing schema signal")
                continue
            pairs.append(
                {
                    "pair_id": f"{dataset_name}:{split}:{example_id}",
                    "source_dataset": dataset_name,
                    "split": split,
                    "example_id": example_id,
                    "mention_text": ie.nl_query,
                    "slot_or_schema": ie.schema_id or ie.schema_text,
                    "label": None,
                    "feasible_for_supervised_labeling": ie.scalar_gold_params is not None,
                    "metadata": {
                        "has_scalar_gold": ie.scalar_gold_params is not None,
                        "has_structured_gold": ie.structured_gold_params is not None,
                        "has_formulation": ie.formulation_text is not None,
                    },
                }
            )

    if not pairs:
        blockers.append("no pairable examples after schema/nl_query checks")
    return pairs, sorted(set(blockers))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export grounding training-pair stubs from normalized datasets.")
    ap.add_argument("--dataset", choices=list_datasets(), default=None)
    ap.add_argument("--all-datasets", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    args = ap.parse_args()

    if not args.all_datasets and not args.dataset:
        raise SystemExit("Pass --dataset <name> or --all-datasets")

    datasets = list_datasets() if args.all_datasets else [args.dataset]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    for ds in datasets:
        pairs, blockers = _extract_pairs(ds)
        out = args.out_dir / f"grounding_training_pairs_{ds}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for row in pairs:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = {
            "dataset": ds,
            "n_pairs": len(pairs),
            "blockers": blockers,
            "output": str(out),
        }
        summary.append(report)
        print(json.dumps(report, indent=2))

    summary_path = args.out_dir / "grounding_training_pairs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
