#!/usr/bin/env python3
"""Prepare NL4Opt data into data/external/nl4opt.

This script avoids vendoring raw archives into git. It supports:
1) optional automated fetch from the official repo URLs, or
2) manual placement and normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "nl4opt"

OFFICIAL_BASE = "https://raw.githubusercontent.com/nl4opt/nl4opt-competition/main"
CANDIDATE_FILES = {
    "train": [
        f"{OFFICIAL_BASE}/generation_data/train.json",
        f"{OFFICIAL_BASE}/generation_data/train.jsonl",
    ],
    "dev": [
        f"{OFFICIAL_BASE}/generation_data/dev.json",
        f"{OFFICIAL_BASE}/generation_data/dev.jsonl",
        f"{OFFICIAL_BASE}/generation_data/valid.json",
    ],
    "test": [
        f"{OFFICIAL_BASE}/generation_data/test.json",
        f"{OFFICIAL_BASE}/generation_data/test.jsonl",
    ],
}


def _try_fetch(url: str) -> str | None:
    try:
        with urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8")
    except URLError:
        return None


def _normalize_to_jsonl(raw: str) -> list[dict]:
    raw = raw.strip()
    if not raw:
        return []
    if raw[0] == "[":
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/prepare NL4Opt into local external data dir.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing prepared split files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    for split, urls in CANDIDATE_FILES.items():
        out = OUT_DIR / f"{split}.jsonl"
        if out.exists() and not args.force:
            print(f"[skip] {out} already exists")
            continue
        payload = None
        used_url = None
        for u in urls:
            payload = _try_fetch(u)
            if payload:
                used_url = u
                break
        if not payload:
            print(
                f"[manual] Could not fetch {split}. Add {split}.jsonl manually at {out} "
                f"(one JSON object per line, include id/query/schema fields when available)."
            )
            continue
        rows = _normalize_to_jsonl(payload)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok] wrote {out} ({len(rows)} rows) from {used_url}")
        wrote_any = True

    if not wrote_any:
        print("No files downloaded. See docs/dataset_integration_report.md manual steps.")


if __name__ == "__main__":
    main()

