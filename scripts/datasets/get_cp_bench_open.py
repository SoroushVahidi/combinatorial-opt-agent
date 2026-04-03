#!/usr/bin/env python3
"""Download DCP-Bench-Open sample JSONL into data/external/cp_bench.

Upstream: https://github.com/DCP-Bench/DCP-Bench-Open (Apache-2.0).

This script only stages **public** raw artifacts (default: ``sample_test.jsonl``).
It does **not** run DCP-Bench evaluation or claim benchmark scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = (
    "https://raw.githubusercontent.com/DCP-Bench/DCP-Bench-Open/main/sample_test.jsonl"
)
OUT_DIR = ROOT / "data" / "external" / "cp_bench"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage DCP-Bench-Open sample_test.jsonl locally.")
    ap.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Raw URL for JSONL (default: upstream sample_test.jsonl).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR})",
    )
    ap.add_argument(
        "--output-name",
        default="sample_test",
        help="Basename for <name>.jsonl (default: sample_test).",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing JSONL.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / f"{args.output_name}.jsonl"
    if target.exists() and not args.force:
        print(f"[skip] {target} already exists (use --force)")
        return 0

    try:
        with urllib.request.urlopen(args.url, timeout=120) as resp:
            body = resp.read()
    except Exception as e:
        print(f"[error] download failed: {e}", file=sys.stderr)
        print("Check network access or pass --url to a mirror.", file=sys.stderr)
        return 1

    target.write_bytes(body)
    try:
        out_rel = str(target.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        out_rel = str(target.resolve())
    manifest = {
        "dataset": "cp_bench",
        "upstream_name": "DCP-Bench-Open",
        "upstream_repo": "https://github.com/DCP-Bench/DCP-Bench-Open",
        "license": "Apache-2.0",
        "source_url": args.url,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": _sha256_bytes(body),
        "bytes": len(body),
        "output_file": out_rel,
    }
    mf = args.out_dir / "staging_manifest.json"
    mf.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[ok] wrote {target} ({len(body)} bytes)")
    print(f"[ok] wrote {mf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
