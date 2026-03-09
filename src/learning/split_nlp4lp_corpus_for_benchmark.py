#!/usr/bin/env python3
"""Split a single NLP4LP corpus file into train/dev/test by instance_id for benchmark-valid runs.

Reads corpus_dir/nlp4lp_test.jsonl (or --input_file), groups by instance_id,
splits with a fixed seed into train 70% / dev 15% / test 15%, and writes
nlp4lp_train.jsonl, nlp4lp_dev.jsonl, nlp4lp_test.jsonl to corpus_dir.

This produces distinct splits from the same source so that no test-as-train
fallback is needed. Use for benchmark mode only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Split NLP4LP corpus into train/dev/test by instance_id")
    ap.add_argument("--corpus_dir", type=Path, default=ROOT / "artifacts" / "learning_corpus")
    ap.add_argument("--input_file", type=Path, default=None, help="If set, read this file; else corpus_dir/nlp4lp_test.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--dev_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    corpus_dir = args.corpus_dir
    corpus_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.input_file or (corpus_dir / "nlp4lp_test.jsonl")
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    # Load and group by instance_id
    by_id: dict[str, list[dict]] = defaultdict(list)
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("dataset") != "nlp4lp":
                continue
            iid = rec.get("instance_id", "")
            if not iid:
                continue
            by_id[iid].append(rec)
    instance_ids = sorted(by_id.keys())
    n = len(instance_ids)
    if n == 0:
        print("ERROR: No instances found.", file=sys.stderr)
        sys.exit(1)
    rng = __import__("random").Random(args.seed)
    rng.shuffle(instance_ids)
    t1 = int(n * args.train_ratio)
    t2 = int(n * (args.train_ratio + args.dev_ratio))
    train_ids = set(instance_ids[:t1])
    dev_ids = set(instance_ids[t1:t2])
    test_ids = set(instance_ids[t2:])
    assert train_ids & dev_ids == set(), "train/dev overlap"
    assert train_ids & test_ids == set(), "train/test overlap"
    assert dev_ids & test_ids == set(), "dev/test overlap"
    assert len(train_ids) + len(dev_ids) + len(test_ids) == n
    for split_name, id_set in [("train", train_ids), ("dev", dev_ids), ("test", test_ids)]:
        out_path = corpus_dir / f"nlp4lp_{split_name}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for iid in id_set:
                for rec in by_id[iid]:
                    rec["split"] = split_name
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
        print(f"{split_name}: {len(id_set)} instances, {count} records -> {out_path}")
    if args.verbose:
        print(f"Seed: {args.seed} | train: {len(train_ids)} | dev: {len(dev_ids)} | test: {len(test_ids)}")


if __name__ == "__main__":
    main()
