#!/usr/bin/env python3
"""Fetch/prepare StructuredOR snapshots into data/external/structuredor."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "structuredor"
META_PATH = OUT_DIR / "provenance.json"
KNOWN_SPLITS = ("train", "validation", "test")
REPO_URL = "https://github.com/CardinalOperations/StructuredOR"


def _write_meta(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    return sum(1 for _ in open(dst, encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/prepare StructuredOR dataset.")
    ap.add_argument("--force", action="store_true")
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
            row_counts[split] = sum(1 for _ in open(p, encoding="utf-8"))

    if len(found) < len(KNOWN_SPLITS):
        if shutil.which("git") is None:
            errors.append("git binary unavailable for retrieval")
        else:
            clone_dir = OUT_DIR / "downloads" / "structuredor_repo"
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
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
                        warnings.append(f"missing split-like file for {split}")
                        continue
                    src = sorted(candidates)[0]
                    out = OUT_DIR / f"{split}.jsonl"
                    row_counts[split] = _to_jsonl(src, out)
                    found.append(split)
            except subprocess.CalledProcessError as e:
                errors.append(f"git clone failed: {e.stderr.strip() or e.stdout.strip()}")

    payload = {
        "source": "CardinalOperations/StructuredOR",
        "source_url": REPO_URL,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "retrieval_method": retrieval_method,
        "splits": sorted(set(found)),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    _write_meta(payload)

    missing = sorted(set(KNOWN_SPLITS) - set(found))
    if missing:
        print(json.dumps(payload, indent=2))
        raise SystemExit(f"Blocked: unable to prepare StructuredOR splits: {missing}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
