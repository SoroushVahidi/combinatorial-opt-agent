#!/usr/bin/env python3
"""Fetch/prepare IndustryOR snapshots into data/external/industryor."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "industryor"
META = OUT_DIR / "provenance.json"
REPO_URL = "https://github.com/CardinalOperations/IndustryOR"
KNOWN_SPLITS = ("train", "validation", "test")


def _collect_json_candidates(repo_dir: Path, split: str) -> list[Path]:
    return sorted(list(repo_dir.rglob(f"*{split}*.json")) + list(repo_dir.rglob(f"*{split}*.jsonl")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
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

    if len(found) < len(KNOWN_SPLITS):
        if shutil.which("git") is None:
            errors.append("git binary unavailable for clone")
        else:
            repo_dir = OUT_DIR / "downloads" / "industryor_repo"
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            if repo_dir.exists() and args.force:
                shutil.rmtree(repo_dir)
            try:
                if not repo_dir.exists():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    method = "git_clone"
                else:
                    method = "existing_repo"
                for split in KNOWN_SPLITS:
                    if split in found:
                        continue
                    cands = _collect_json_candidates(repo_dir, split)
                    if not cands:
                        warnings.append(f"{split}: no matching split file found in repo")
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
            except subprocess.CalledProcessError as e:
                errors.append(f"git clone failed: {e.stderr.strip() or e.stdout.strip()}")

    payload = {
        "source": "CardinalOperations/IndustryOR",
        "source_url": REPO_URL,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "retrieval_method": method,
        "splits": sorted(set(found)),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    META.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    missing = sorted(set(KNOWN_SPLITS) - set(found))
    if missing:
        print(json.dumps(payload, indent=2))
        raise SystemExit(f"Blocked: unable to prepare IndustryOR splits: {missing}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
