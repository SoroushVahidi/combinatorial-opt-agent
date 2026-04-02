#!/usr/bin/env python3
"""Fetch/prepare IndustryOR snapshots into data/external/industryor.

Source: https://github.com/CardinalOperations/IndustryOR
Splits: train, dev, validation, test

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
OUT_DIR = ROOT / "data" / "external" / "industryor"
META = OUT_DIR / "provenance.json"
REPO_URL = "https://github.com/CardinalOperations/IndustryOR"
KNOWN_SPLITS = ("train", "dev", "validation", "test")


def _collect_json_candidates(repo_dir: Path, split: str) -> list[Path]:
    return sorted(list(repo_dir.rglob(f"*{split}*.json")) + list(repo_dir.rglob(f"*{split}*.jsonl")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/prepare IndustryOR snapshots.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split JSONL files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    row_counts: dict[str, int] = {}
    found: list[str] = []
    method = "none"

    for split in KNOWN_SPLITS:
        p = OUT_DIR / f"{split}.jsonl"
        if p.exists() and not args.force:
            found.append(split)
            with p.open(encoding="utf-8") as fh:
                row_counts[split] = sum(1 for _ in fh)

    if len(found) == len(KNOWN_SPLITS):
        method = "preexisting_local_files"
    else:
        if shutil.which("git") is None:
            msg = "git binary unavailable for clone"
            print(f"[error] {msg}", file=sys.stderr)
            errors.append(msg)
        else:
            repo_dir = OUT_DIR / "downloads" / "industryor_repo"
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                method = "git_clone"
                for split in KNOWN_SPLITS:
                    if split in found:
                        continue
                    cands = _collect_json_candidates(repo_dir, split)
                    if not cands:
                        msg = f"{split}: no matching split file found in repo"
                        print(f"[warn] {msg}", file=sys.stderr)
                        warnings.append(msg)
                        continue
                    src = cands[0]
                    text = src.read_text(encoding="utf-8")
                    out = OUT_DIR / f"{split}.jsonl"
                    lines: list[str] = []
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, list):
                            lines = [json.dumps(r, ensure_ascii=False) for r in obj]
                        else:
                            lines = [json.dumps(obj, ensure_ascii=False)]
                    except Exception:
                        lines = [ln for ln in text.splitlines() if ln.strip()]
                    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    row_counts[split] = len(lines)
                    found.append(split)
                    print(f"[ok] {split}: {len(lines)} rows -> {out}")
            except subprocess.CalledProcessError as e:
                msg = f"git clone failed: {e.stderr.strip() or e.stdout.strip()}"
                print(f"[error] {msg}", file=sys.stderr)
                errors.append(msg)

    provenance = {
        "dataset": "IndustryOR",
        "source": "CardinalOperations/IndustryOR",
        "source_url": REPO_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_method": method,
        "splits": sorted(set(found)),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    META.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[ok] provenance written -> {META}")

    missing = sorted(set(KNOWN_SPLITS) - set(found))
    if missing:
        print(json.dumps(provenance, indent=2))
        print(f"[warn] Could not prepare IndustryOR splits: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] IndustryOR ready: {sorted(set(found))}")


if __name__ == "__main__":
    main()
