#!/usr/bin/env python3
"""Fetch and prepare StructuredOR dataset (LLM4OR/StructuredOR) from HuggingFace.

Source: https://huggingface.co/datasets/LLM4OR/StructuredOR
License: Unknown (no license in dataset card as of 2026-04-02)
Splits: train (~86 files), test (~38 files) – one JSON object per file

The dataset is stored as individual JSON files under train/ and test/.
This script enumerates file paths via the HuggingFace API siblings list,
downloads each file, merges them into per-split JSONL files, and writes
a provenance record.

Outputs are written to data/external/structuredor/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "structuredor"

HF_API = "https://huggingface.co/api/datasets/LLM4OR/StructuredOR"
HF_RESOLVE = "https://huggingface.co/datasets/LLM4OR/StructuredOR/resolve/main"
KNOWN_SPLITS = ("train", "test")


def _fetch_text(url: str, timeout: int = 30) -> str | None:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except URLError as exc:
        print(f"[error] fetch failed for {url}: {exc}", file=sys.stderr)
        return None


def _get_sibling_paths() -> list[str] | None:
    raw = _fetch_text(HF_API)
    if raw is None:
        return None
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[error] failed to parse API response: {exc}", file=sys.stderr)
        return None
    siblings = meta.get("siblings", [])
    return [s["rfilename"] for s in siblings if "rfilename" in s]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download StructuredOR dataset from HuggingFace.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files.")
    ap.add_argument(
        "--max-per-split",
        type=int,
        default=None,
        help="Limit number of files downloaded per split (for testing).",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict = {
        "dataset": "StructuredOR",
        "source": "https://huggingface.co/datasets/LLM4OR/StructuredOR",
        "license": "UNKNOWN – no license specified in dataset card",
        "retrieval_method": "urllib per-file download via HuggingFace resolve URL; file list from siblings API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "warnings": ["License is unknown. Verify before redistribution or use in training."],
    }

    # Enumerate siblings
    siblings = _get_sibling_paths()
    if siblings is None:
        msg = "Could not retrieve file list from HuggingFace API"
        print(f"[error] {msg}", file=sys.stderr)
        provenance["warnings"].append(msg)
        prov_path = OUT_DIR / "provenance.json"
        with open(prov_path, "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, indent=2, ensure_ascii=False)
        sys.exit(1)

    any_success = False
    for split in KNOWN_SPLITS:
        out_path = OUT_DIR / f"{split}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} already exists (use --force to overwrite)")
            rows_count = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
            provenance["splits"][split] = {"rows": rows_count, "status": "existing"}
            any_success = True
            continue

        split_files = sorted(f for f in siblings if f.startswith(f"{split}/") and f.endswith(".json"))
        if not split_files:
            msg = f"No files found for split '{split}'"
            print(f"[warn] {msg}", file=sys.stderr)
            provenance["warnings"].append(msg)
            provenance["splits"][split] = {"rows": 0, "status": "no_files_found"}
            continue

        if args.max_per_split is not None:
            split_files = split_files[: args.max_per_split]

        rows: list[dict] = []
        errors: list[str] = []
        for rfilename in split_files:
            stem = Path(rfilename).stem
            url = f"{HF_RESOLVE}/{rfilename}"
            raw = _fetch_text(url, timeout=20)
            if raw is None:
                errors.append(f"fetch_failed:{rfilename}")
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"parse_error:{rfilename}:{exc}")
                continue
            label = obj.get("label")
            rows.append(
                {
                    "id": f"structuredor_{split}_{stem}",
                    "nl_query": (obj.get("question") or "").strip(),
                    "label": label,
                    "objective_value": obj.get("objective_value"),
                    "split": split,
                    "source": "LLM4OR/StructuredOR",
                    "metadata": {"source_file": rfilename},
                }
            )

        if errors:
            msg = f"{len(errors)} file(s) failed in split '{split}'"
            print(f"[warn] {msg}: {errors[:5]}", file=sys.stderr)
            provenance["warnings"].append(f"{msg}: {errors}")

        if not rows:
            msg = f"No rows collected for split '{split}'"
            print(f"[warn] {msg}", file=sys.stderr)
            provenance["splits"][split] = {"rows": 0, "status": "empty_after_fetch"}
            continue

        with open(out_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[ok] {split}: wrote {len(rows)} rows -> {out_path}")
        provenance["splits"][split] = {
            "rows": len(rows),
            "status": "ok",
            "files_attempted": len(split_files),
            "files_failed": len(errors),
        }
        any_success = True

    prov_path = OUT_DIR / "provenance.json"
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)
    print(f"[ok] provenance written -> {prov_path}")

    if not any_success:
        print(
            "[warn] No data was downloaded. Check internet connectivity.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
