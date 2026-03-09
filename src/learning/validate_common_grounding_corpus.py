#!/usr/bin/env python3
"""Validate common learning corpus JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.common_corpus_schema import validate_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_dir", type=Path, default=ROOT / "artifacts" / "learning_corpus")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    corpus_dir = args.corpus_dir
    if not corpus_dir.exists():
        print(f"Corpus dir not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    all_ok = True
    for path in sorted(corpus_dir.glob("*.jsonl")):
        if path.name.startswith("."):
            continue
        n = 0
        err_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    if args.verbose:
                        print(f"{path.name} line {n}: JSON error {e}", file=sys.stderr)
                    err_count += 1
                    all_ok = False
                    continue
                errs = validate_record(rec)
                if errs:
                    err_count += 1
                    all_ok = False
                    if args.verbose:
                        print(f"{path.name} line {n} ({rec.get('instance_id')}): {errs}", file=sys.stderr)
        status = "OK" if err_count == 0 else f"FAIL ({err_count} invalid)"
        print(f"{path.name}: {n} records, {status}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
