#!/usr/bin/env python3
"""Evaluate rule (handcrafted) baseline only — no transformer load. Use for same-test comparison without GPU/threads."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.models.decoding import argmax_per_slot, one_to_one_matching


def rule_score(row: dict) -> float:
    s = 0.0
    s += 2.0 * float(row.get("feat_type_match", 0))
    s += 1.5 * float(row.get("feat_operator_cue_match", 0))
    s += 0.5 * float(row.get("feat_slot_mention_overlap", 0))
    if row.get("feat_operator_cue_match") == -1:
        s -= 3.0
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--decoder", choices=["argmax", "one_to_one"], default="argmax")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()
    data_path = args.data_dir / f"{args.split}.jsonl"
    if not data_path.exists():
        print(f"Data not found: {data_path}", file=sys.stderr)
        return 1
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("No rows.", file=sys.stderr)
        return 1
    group_scores: list[tuple[str, str, float]] = []
    for row in rows:
        gid = row.get("group_id", "")
        mid = row.get("mention_id", "")
        sc = rule_score(row)
        group_scores.append((gid, mid, sc))
    if args.decoder == "one_to_one":
        pred_assignments = one_to_one_matching(group_scores)
    else:
        pred_assignments = argmax_per_slot(group_scores)
    gold_by_group: dict[str, str | None] = {}
    for row in rows:
        gid = row.get("group_id", "")
        gold = row.get("gold_mention_id_for_slot")
        if gid not in gold_by_group:
            gold_by_group[gid] = gold
    pair_correct = 0
    pair_total = 0
    by_group: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for gid, mid, sc in group_scores:
        by_group[gid].append((mid, sc))
    for gid, candidates in by_group.items():
        gold_mid = gold_by_group.get(gid)
        if gold_mid is None:
            continue
        candidates_sorted = sorted(candidates, key=lambda x: -x[1])
        best_mid = candidates_sorted[0][0] if candidates_sorted else None
        pair_total += 1
        if best_mid == gold_mid:
            pair_correct += 1
    pairwise_acc = pair_correct / pair_total if pair_total else 0.0
    slot_correct = sum(1 for gid, pred_mid in pred_assignments.items() if pred_mid == gold_by_group.get(gid))
    slot_total = len(gold_by_group)
    slot_acc = slot_correct / slot_total if slot_total else 0.0
    instance_slots: dict[str, dict[str, str]] = defaultdict(dict)
    for gid, pred_mid in pred_assignments.items():
        if "::" in gid:
            inst, slot = gid.split("::", 1)
            instance_slots[inst][slot] = pred_mid
    instance_gold: dict[str, dict[str, str]] = defaultdict(dict)
    for gid, gold_mid in gold_by_group.items():
        if gold_mid and "::" in gid:
            inst, slot = gid.split("::", 1)
            instance_gold[inst][slot] = gold_mid
    exact_instances = sum(
        1 for inst, pred_slots in instance_slots.items()
        if instance_gold.get(inst) and all(pred_slots.get(s) == instance_gold[inst].get(s) for s in instance_gold[inst])
    )
    total_instances = len(instance_gold)
    exact_acc = exact_instances / total_instances if total_instances else 0.0
    row_by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        row_by_key[(row.get("group_id", ""), row.get("mention_id", ""))] = row
    type_match_count = 0
    type_match_total = 0
    for gid, pred_mid in pred_assignments.items():
        row = row_by_key.get((gid, pred_mid))
        if not row:
            continue
        type_match_total += 1
        expected = row.get("expected_type")
        actual = row.get("mention_type_bucket")
        if expected and actual and expected == actual:
            type_match_count += 1
    type_match_acc = type_match_count / type_match_total if type_match_total else 0.0
    metrics = {
        "run_name": "rule_baseline_only",
        "split": args.split,
        "decoder": args.decoder,
        "model_source": "rule",
        "training_mode": "none",
        "pairwise_accuracy": pairwise_acc,
        "slot_selection_accuracy": slot_acc,
        "exact_slot_fill_accuracy": exact_acc,
        "type_match_after_decoding": type_match_acc,
        "pair_correct": pair_correct,
        "pair_total": pair_total,
        "slot_correct": slot_correct,
        "slot_total": slot_total,
        "exact_instances": exact_instances,
        "total_instances": total_instances,
        "type_match_count": type_match_count,
        "type_match_total": type_match_total,
    }
    print(json.dumps(metrics, indent=2))
    out_dir = args.out_dir or (ROOT / "artifacts" / "learning_runs" / "rule_baseline_same_test")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {out_dir / 'metrics.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
