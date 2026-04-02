#!/usr/bin/env python3
"""Capability-aware multi-dataset benchmark compatibility runner.

This runner is additive: it does not change existing NLP4LP evaluation logic.
It reports dataset readiness/coverage for available target fields and prints N/A
for unsupported metrics by capability.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_adapters.registry import create_adapter, list_datasets


def _metric_or_na(supported: bool, value: float | None) -> str:
    if not supported:
        return "N/A"
    if value is None:
        return "0.0000"
    return f"{value:.4f}"


def _safe_mean(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


def evaluate_dataset(dataset_name: str) -> list[dict[str, str]]:
    adapter = create_adapter(dataset_name)
    rows: list[dict[str, str]] = []
    for split in adapter.list_splits():
        raw = list(adapter.iter_examples(split))
        total = len(raw)
        if total == 0:
            continue
        internal = [adapter.to_internal_example(x, split) for x in raw]
        schema_present = sum(1 for x in internal if x.schema_id)
        scalar_present = sum(1 for x in internal if isinstance(x.scalar_gold_params, dict) and len(x.scalar_gold_params) > 0)
        formulation_present = sum(1 for x in internal if (x.formulation_text or x.solver_artifact_path))

        caps = adapter.capabilities
        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "n_examples": str(total),
                "supports_schema_retrieval": str(caps.supports_schema_retrieval),
                "supports_scalar_instantiation": str(caps.supports_scalar_instantiation),
                "supports_solver_eval": str(caps.supports_solver_eval),
                "supports_full_formulation": str(caps.supports_full_formulation),
                "schema_target_coverage": _metric_or_na(
                    caps.supports_schema_retrieval, _safe_mean(schema_present, total)
                ),
                "scalar_target_coverage": _metric_or_na(
                    caps.supports_scalar_instantiation, _safe_mean(scalar_present, total)
                ),
                "formulation_target_coverage": _metric_or_na(
                    caps.supports_full_formulation or caps.supports_solver_eval,
                    _safe_mean(formulation_present, total),
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset",
        "split",
        "n_examples",
        "supports_schema_retrieval",
        "supports_scalar_instantiation",
        "supports_solver_eval",
        "supports_full_formulation",
        "schema_target_coverage",
        "scalar_target_coverage",
        "formulation_target_coverage",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run capability-aware dataset benchmark compatibility checks.")
    ap.add_argument("--dataset", choices=list_datasets(), default=None)
    ap.add_argument("--all-datasets", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    args = ap.parse_args()

    if not args.all_datasets and not args.dataset:
        raise SystemExit("Pass --dataset <name> or --all-datasets.")

    selected = list_datasets() if args.all_datasets else [args.dataset]

    all_rows: list[dict[str, str]] = []
    for ds in selected:
        ds_rows = evaluate_dataset(ds)
        if not ds_rows:
            print(f"[warn] no rows produced for dataset={ds}")
        all_rows.extend(ds_rows)

    if not all_rows:
        raise SystemExit("No benchmark rows produced. Check data availability and run dataset get_* scripts.")

    merged_csv = args.out_dir / "dataset_benchmark_summary.csv"
    merged_json = args.out_dir / "dataset_benchmark_summary.json"
    write_csv(merged_csv, all_rows)
    write_json(merged_json, all_rows)
    print(f"Wrote {merged_csv}")
    print(f"Wrote {merged_json}")

    for ds in selected:
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        if not ds_rows:
            continue
        ds_csv = args.out_dir / f"dataset_benchmark_{ds}.csv"
        write_csv(ds_csv, ds_rows)
        print(f"Wrote {ds_csv}")


if __name__ == "__main__":
    main()

