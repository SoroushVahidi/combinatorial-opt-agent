#!/usr/bin/env python3
"""Prepare Text2Zinc snapshots into data/external/text2zinc.

Text2Zinc is gated on Hugging Face. This script tries `datasets` API if access
is granted; otherwise it prints clear manual steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "text2zinc"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/prepare Text2Zinc.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split outputs.")
    ap.add_argument("--max-rows", type=int, default=0, help="Optional max rows per split (0 = all).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"[manual] datasets package unavailable: {e}")
        print("Install dependencies and follow manual setup in docs/dataset_integration_report.md.")
        return

    try:
        ds = load_dataset("skadio/text2zinc")
    except Exception as e:
        print(f"[manual] Could not access skadio/text2zinc: {e}")
        print("This dataset is gated. Request access on HF and retry.")
        return

    for split in ds.keys():
        out = OUT_DIR / f"{split}.jsonl"
        if out.exists() and not args.force:
            print(f"[skip] {out} already exists")
            continue
        n = 0
        with open(out, "w", encoding="utf-8") as f:
            for row in ds[split]:
                input_obj = row.get("input.json")
                output_obj = row.get("output.json")
                if isinstance(input_obj, str):
                    try:
                        input_obj = json.loads(input_obj)
                    except Exception:
                        input_obj = {"description": input_obj}
                if isinstance(output_obj, str):
                    try:
                        output_obj = json.loads(output_obj)
                    except Exception:
                        output_obj = {"raw_output": output_obj}
                normalized = {
                    "id": (
                        (input_obj or {}).get("metadata", {}).get("identifier")
                        if isinstance(input_obj, dict)
                        else None
                    ),
                    "input": input_obj,
                    "output": output_obj,
                    "model_mzn": row.get("model.mzn"),
                    "data_dzn": row.get("data.dzn"),
                    "has_mzn": row.get("has_mzn"),
                    "has_dzn": row.get("has_dzn"),
                    "is_verified": row.get("is_verified"),
                }
                f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                n += 1
                if args.max_rows and n >= args.max_rows:
                    break
        print(f"[ok] wrote {out} ({n} rows)")


if __name__ == "__main__":
    main()

