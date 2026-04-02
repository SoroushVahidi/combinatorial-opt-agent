#!/usr/bin/env python3
"""Fetch/prepare CardinalOperations/NL4OPT snapshots into data/external/cardinal_nl4opt."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "cardinal_nl4opt"
META = OUT_DIR / "provenance.json"
BASE = "https://raw.githubusercontent.com/CardinalOperations/NL4OPT/main"
SPLIT_URLS = {
    "train": f"{BASE}/generation_data/train.jsonl",
    "dev": f"{BASE}/generation_data/dev.jsonl",
    "test": f"{BASE}/generation_data/test.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    row_counts: dict[str, int] = {}
    found: list[str] = []
    method = "direct_raw_http"

    for split, url in SPLIT_URLS.items():
        out = OUT_DIR / f"{split}.jsonl"
        if out.exists() and not args.force:
            found.append(split)
            row_counts[split] = sum(1 for _ in open(out, encoding="utf-8"))
            continue
        try:
            with urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            row_counts[split] = len(lines)
            found.append(split)
        except HTTPError as e:
            errors.append(f"{split}: HTTPError {e.code} from {url}")
        except URLError as e:
            errors.append(f"{split}: URLError {e.reason} from {url}")
        except Exception as e:
            errors.append(f"{split}: unexpected error: {e}")

    payload = {
        "source": "CardinalOperations/NL4OPT",
        "source_url": "https://github.com/CardinalOperations/NL4OPT",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "retrieval_method": method,
        "splits": sorted(found),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    META.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    missing = sorted(set(SPLIT_URLS) - set(found))
    if missing:
        print(json.dumps(payload, indent=2))
        raise SystemExit(f"Blocked: unable to fetch Cardinal NL4OPT splits: {missing}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
