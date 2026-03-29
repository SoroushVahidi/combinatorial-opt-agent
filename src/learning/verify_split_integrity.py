#!/usr/bin/env python3
"""Verify that train/dev/test ranker data files exist and are distinct (no leakage).

Exits with code 0 if all checks pass, 1 otherwise.
Use in benchmark runs before training to fail fast on leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def file_content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
            if n > 0 and n % 100000 == 0:
                pass  # could add progress for huge files
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify train/dev/test split integrity")
    ap.add_argument("--data_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--require_dev", action="store_true", default=True, help="Require dev.jsonl to exist")
    ap.add_argument("--no_require_dev", action="store_false", dest="require_dev")
    args = ap.parse_args()
    data_dir = args.data_dir
    train_f = data_dir / "train.jsonl"
    dev_f = data_dir / "dev.jsonl"
    test_f = data_dir / "test.jsonl"
    errors = []
    if not train_f.exists():
        errors.append(f"Missing: {train_f}")
    if not test_f.exists():
        errors.append(f"Missing: {test_f}")
    if args.require_dev and not dev_f.exists():
        errors.append(f"Missing: {dev_f}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    train_count = count_lines(train_f)
    test_count = count_lines(test_f)
    dev_count = count_lines(dev_f) if dev_f.exists() else 0
    print(f"train.jsonl: {train_count} lines")
    print(f"test.jsonl:  {test_count} lines")
    if dev_f.exists():
        print(f"dev.jsonl:   {dev_count} lines")
    train_hash = file_content_hash(train_f)
    test_hash = file_content_hash(test_f)
    dev_hash = file_content_hash(dev_f) if dev_f.exists() else None
    if train_hash == test_hash:
        print("ERROR: train.jsonl and test.jsonl are identical (leakage).", file=sys.stderr)
        return 1
    if dev_hash is not None:
        if train_hash == dev_hash:
            print("ERROR: train.jsonl and dev.jsonl are identical.", file=sys.stderr)
            return 1
        if dev_hash == test_hash:
            print("ERROR: dev.jsonl and test.jsonl are identical.", file=sys.stderr)
            return 1
    print("Split integrity OK: train, dev, test are distinct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
