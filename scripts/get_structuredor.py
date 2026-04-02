#!/usr/bin/env python3
"""Fetch/prepare StructuredOR dataset snapshots into data/external/structuredor.

Source: https://github.com/CardinalOperations/StructuredOR
Splits: train, validation, dev, test

This script is offline-friendly:
- it first looks for pre-existing local files,
- then attempts git clone,
- always writes a provenance report,
- exits non-zero when retrieval is blocked.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "structuredor"
META_PATH = OUT_DIR / "provenance.json"
KNOWN_SPLITS = ("train", "validation", "dev", "test")
REPO_URL = "https://github.com/CardinalOperations/StructuredOR"


def _to_jsonl(src: Path, dst: Path) -> int:
    text = src.read_text(encoding="utf-8")
    try:
        obj = json.loads(text)
        with open(dst, "w", encoding="utf-8") as fh:
            if isinstance(obj, list):
                for row in obj:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    with open(dst, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/prepare StructuredOR dataset.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split JSONL files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    row_counts: dict[str, int] = {}
    found: list[str] = []
    retrieval_method = "none"

    for split in KNOWN_SPLITS:
        p = OUT_DIR / f"{split}.jsonl"
        if p.exists() and not args.force:
            found.append(split)
            with p.open(encoding="utf-8") as fh:
                row_counts[split] = sum(1 for _ in fh)

    if len(found) == len(KNOWN_SPLITS):
        retrieval_method = "preexisting_local_files"
    else:
        if shutil.which("git") is None:
            msg = "git binary unavailable for retrieval"
            print(f"[error] {msg}", file=sys.stderr)
            errors.append(msg)
        else:
            clone_dir = OUT_DIR / "downloads" / "structuredor_repo"
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                retrieval_method = "git_clone"
                for split in KNOWN_SPLITS:
                    if split in found:
                        continue
                    candidates = list(clone_dir.rglob(f"*{split}*.json")) + list(clone_dir.rglob(f"*{split}*.jsonl"))
                    if not candidates:
                        msg = f"missing split-like file for {split}"
                        print(f"[warn] {msg}", file=sys.stderr)
                        warnings.append(msg)
                        continue
                    src = sorted(candidates)[0]
                    out = OUT_DIR / f"{split}.jsonl"
                    row_counts[split] = _to_jsonl(src, out)
                    found.append(split)
                    print(f"[ok] {split}: {row_counts[split]} rows -> {out}")
            except subprocess.CalledProcessError as e:
                msg = f"git clone failed: {e.stderr.strip() or e.stdout.strip()}"
                print(f"[error] {msg}", file=sys.stderr)
                errors.append(msg)

    provenance = {
        "dataset": "StructuredOR",
        "source": "CardinalOperations/StructuredOR",
        "source_url": REPO_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_method": retrieval_method,
        "splits": sorted(set(found)),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    META_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[ok] provenance written -> {META_PATH}")

    missing = sorted(set(KNOWN_SPLITS) - set(found))
    if missing:
        print(json.dumps(provenance, indent=2))
        print(f"[warn] Could not prepare StructuredOR splits: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] StructuredOR ready: {sorted(set(found))}")


if __name__ == "__main__":
    main()
