#!/usr/bin/env python3
"""Evaluate NLP4LP pairwise ranker: pairwise acc, slot selection acc, exact slot-fill, type-match."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.learning.models.decoding import argmax_per_slot, one_to_one_matching
from src.learning.models.features import row_to_feature_vector
from src.learning.models.pairwise_ranker import PairwiseRanker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--run_dir", type=Path, default=None, help="Load checkpoint from this run dir")
    ap.add_argument("--decoder", choices=["argmax", "one_to_one"], default="argmax")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()
    data_path = args.data_dir / f"{args.split}.jsonl"
    if not data_path.exists():
        print(f"Data not found: {data_path}", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("No rows.", file=sys.stderr)
        sys.exit(1)
    ranker = PairwiseRanker(use_structured_features=False)
    if args.run_dir:
        # Prefer pairwise_only.pt from multitask runs (encoder + head only)
        if (args.run_dir / "pairwise_only.pt").exists():
            ranker.load(str(args.run_dir / "pairwise_only.pt"))
        elif (args.run_dir / "checkpoint.pt").exists():
            ranker.load(str(args.run_dir / "checkpoint.pt"))
    def _rule_score(row: dict) -> float:
        """Fallback when no model: weighted handcrafted features."""
        s = 0.0
        s += 2.0 * float(row.get("feat_type_match", 0))
        s += 1.5 * float(row.get("feat_operator_cue_match", 0))
        s += 0.5 * float(row.get("feat_slot_mention_overlap", 0))
        if row.get("feat_operator_cue_match") == -1:
            s -= 3.0
        return s

    group_scores: list[tuple[str, str, float]] = []
    for row in rows:
        gid = row.get("group_id", "")
        mid = row.get("mention_id", "")
        feats = row_to_feature_vector(row) if ranker.use_structured_features else None
        sc = ranker.score(
            slot_name=row.get("slot_name", ""),
            slot_role=row.get("slot_role"),
            mention_surface=row.get("mention_surface", ""),
            context=row.get("sentence_or_context"),
            feature_vector=feats,
        )
        if ranker.model is None:
            sc = _rule_score(row)
        group_scores.append((gid, mid, sc))
    if args.decoder == "one_to_one":
        pred_assignments = one_to_one_matching(group_scores)
    else:
        pred_assignments = argmax_per_slot(group_scores)
    # Build group_id -> gold mention_id from rows (same for all rows in group)
    gold_by_group: dict[str, str | None] = {}
    for row in rows:
        gid = row.get("group_id", "")
        gold = row.get("gold_mention_id_for_slot")
        if gid not in gold_by_group:
            gold_by_group[gid] = gold
    # Pairwise accuracy: over all pairs, did we rank the gold above others in the same group?
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
    # Exact instance: group_id contains instance_id::slot_id; count instances with all slots correct
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
    exact_instances = 0
    for inst, pred_slots in instance_slots.items():
        gold_slots = instance_gold.get(inst, {})
        if gold_slots and all(pred_slots.get(s) == gold_slots.get(s) for s in gold_slots):
            exact_instances += 1
    total_instances = len(instance_gold)
    exact_acc = exact_instances / total_instances if total_instances else 0.0
    # Type-match after decoding: of predicted (slot, mention), how many have matching type?
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
    run_name = args.run_dir.name if args.run_dir else "eval_out"
    model_source = "rule"
    if ranker.model is not None and args.run_dir:
        if (args.run_dir / "pairwise_only.pt").exists():
            model_source = "stage2"
        else:
            model_source = "stage1"
    config: dict = {}
    if args.run_dir and (args.run_dir / "config.json").exists():
        try:
            with open(args.run_dir / "config.json", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass
    training_mode = config.get("mode") or ("stage1_pairwise" if model_source == "stage1" else ("pretrain_then_finetune" if config.get("use_entity") else None))
    if model_source == "rule":
        training_mode = "none"
    use_structured_features = bool(ranker.use_structured_features or config.get("use_features"))
    use_nl4opt_entity = bool(config.get("use_entity", False))
    use_nl4opt_bound = bool(config.get("use_bound", False))
    use_nl4opt_role = bool(config.get("use_role", False))
    metrics = {
        "run_name": run_name,
        "split": args.split,
        "decoder": args.decoder,
        "model_source": model_source,
        "training_mode": training_mode or "",
        "use_structured_features": use_structured_features,
        "use_nl4opt_entity": use_nl4opt_entity,
        "use_nl4opt_bound": use_nl4opt_bound,
        "use_nl4opt_role": use_nl4opt_role,
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
    out_dir = args.out_dir or (args.run_dir if args.run_dir else ROOT / "artifacts" / "learning_runs" / "eval_out")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    md_lines = [
        f"# {run_name}",
        "",
        f"- **split:** {args.split} | **decoder:** {args.decoder} | **model_source:** {model_source}",
        f"- pairwise_accuracy: {pairwise_acc:.4f} | slot_selection_accuracy: {slot_acc:.4f}",
        f"- exact_slot_fill_accuracy: {exact_acc:.4f} | type_match_after_decoding: {type_match_acc:.4f}",
    ]
    with open(out_dir / "metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for gid, mid in pred_assignments.items():
            f.write(json.dumps({"group_id": gid, "mention_id": mid}) + "\n")
    print(f"Wrote {out_dir / 'metrics.json'}, {out_dir / 'metrics.md'}, {out_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
