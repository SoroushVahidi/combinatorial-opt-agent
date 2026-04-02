#!/usr/bin/env python3
"""Prepare ComplexOR-like data from the closest clearly licensed public source.

Policy: use Text2Zinc rows where metadata.source == "complexor" (CC-BY-4.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "complexor"


def _source_name(row: dict) -> str | None:
    input_obj = row.get("input")
    if not isinstance(input_obj, dict):
        return None
    meta = input_obj.get("metadata")
    if not isinstance(meta, dict):
        return None
    src = meta.get("source")
    return src.lower() if isinstance(src, str) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ComplexOR snapshot from Text2Zinc-derived rows.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t2z_dir = ROOT / "data" / "external" / "text2zinc"
    if not t2z_dir.exists():
        print("[manual] Text2Zinc local data not found.")
        print("Run scripts/get_text2zinc.py first, then rerun this script.")
        return

    found_any = False
    for split_file in sorted(t2z_dir.glob("*.jsonl")):
        split = split_file.stem
        out = OUT_DIR / f"{split}.jsonl"
        if out.exists() and not args.force:
            print(f"[skip] {out} already exists")
            continue
        n_in = 0
        n_out = 0
        with open(split_file, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                row = json.loads(line)
                if _source_name(row) != "complexor":
                    continue
                input_obj = row.get("input") if isinstance(row.get("input"), dict) else {}
                meta = input_obj.get("metadata") if isinstance(input_obj.get("metadata"), dict) else {}
                normalized = {
                    "id": meta.get("identifier"),
                    "nl_query": input_obj.get("description"),
                    "problem_family": meta.get("domain") or "complexor",
                    "model_mzn": row.get("model_mzn"),
                    "structured_gold_params": input_obj,
                    "metadata": {"derived_from": "text2zinc", "source_split": split},
                }
                fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                n_out += 1
        print(f"[ok] {split}: kept {n_out}/{n_in} complexor rows -> {out}")
        found_any = found_any or (n_out > 0)

    if not found_any:
        print("No complexor rows found in local Text2Zinc snapshots.")
        print("Verify Text2Zinc data exists and includes metadata.source=complexor.")


if __name__ == "__main__":
    main()

