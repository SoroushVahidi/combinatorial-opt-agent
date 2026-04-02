#!/usr/bin/env python3
"""Prepare OptMATH snapshots into data/external/optmath.

By default this script is conservative and does not force full 200K+ downloads.
Use --max-rows for small local snapshots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "optmath"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/prepare OptMATH split snapshots.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files.")
    ap.add_argument("--max-rows", type=int, default=1000, help="Rows per split to export (default 1000).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[manual] datasets package unavailable: {e}")
        print("Follow manual instructions in docs/dataset_integration_report.md.")
        return

    # Public training corpus on HF; benchmark variants may differ by release.
    candidates = ["shushulei/OptMATH-Train"]
    loaded = None
    for name in candidates:
        try:
            loaded = load_dataset(name)
            print(f"[ok] loaded dataset: {name}")
            break
        except Exception as e:
            print(f"[warn] could not load {name}: {e}")
    if loaded is None:
        print("[manual] Could not load OptMATH automatically.")
        print("Use scripts/get_optmath.py after setting HF auth, or place JSONL files manually in data/external/optmath.")
        return

    for split in loaded.keys():
        out = OUT_DIR / f"{split}.jsonl"
        if out.exists() and not args.force:
            print(f"[skip] {out} already exists")
            continue
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in loaded[split]:
                normalized = {
                    "id": row.get("id") or row.get("instance_id"),
                    "nl_query": row.get("nl_query") or row.get("problem") or row.get("text"),
                    "problem_type": row.get("problem_type"),
                    "scalar_gold_params": row.get("scalar_gold_params"),
                    "structured_gold_params": row.get("structured_gold_params"),
                    "target_model": row.get("target_model") or row.get("formulation_text"),
                    "metadata": {"source_row_keys": sorted(row.keys())},
                }
                f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                n += 1
                if args.max_rows and n >= args.max_rows:
                    break
        print(f"[ok] wrote {out} ({n} rows)")


if __name__ == "__main__":
    main()

