#!/usr/bin/env python3
"""Evaluate on NLP4LP bottleneck slices: multiple floats, lower/upper cues, multi-entity."""

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


def _rule_score(row: dict) -> float:
    s = 0.0
    s += 2.0 * float(row.get("feat_type_match", 0))
    s += 1.5 * float(row.get("feat_operator_cue_match", 0))
    s += 0.5 * float(row.get("feat_slot_mention_overlap", 0))
    if row.get("feat_operator_cue_match") == -1:
        s -= 3.0
    return s


def _build_slices(rows: list[dict]) -> dict[str, set[str]]:
    """Return slice_name -> set of instance_id that belong to the slice."""
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        gid = row.get("group_id", "")
        if "::" in gid:
            inst, _ = gid.split("::", 1)
            by_inst[inst].append(row)
    multi_float: set[str] = set()
    bound_cues: set[str] = set()
    multi_entity: set[str] = set()
    for inst, inst_rows in by_inst.items():
        mention_ids = set()
        slot_ids = set()
        has_bound_cue = False
        for r in inst_rows:
            mention_ids.add(r.get("mention_id"))
            slot_ids.add(r.get("slot_id"))
            if r.get("feat_operator_cue_match") and r.get("feat_operator_cue_match") != 0:
                has_bound_cue = True
            if (r.get("slot_role") or "").lower() in ("lower_bound", "upper_bound"):
                has_bound_cue = True
        if len(mention_ids) >= 3:
            multi_float.add(inst)
        if has_bound_cue:
            bound_cues.add(inst)
        if len(slot_ids) >= 2:
            multi_entity.add(inst)
    return {
        "multiple_float_like": multi_float,
        "lower_upper_cues": bound_cues,
        "multi_entity": multi_entity,
    }


def _metrics_for_instances(
    rows: list[dict],
    group_scores: list[tuple[str, str, float]],
    instance_ids: set[str],
    decoder: str = "argmax",
) -> dict:
    if decoder == "one_to_one":
        pred_assignments = one_to_one_matching(group_scores)
    else:
        pred_assignments = argmax_per_slot(group_scores)
    gold_by_group: dict[str, str | None] = {}
    for row in rows:
        gid = row.get("group_id", "")
        if gid not in gold_by_group:
            gold_by_group[gid] = row.get("gold_mention_id_for_slot")
    by_group: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for gid, mid, sc in group_scores:
        by_group[gid].append((mid, sc))
    pair_correct = pair_total = slot_correct = slot_total = exact_instances = total_instances = 0
    instance_gold: dict[str, dict[str, str]] = defaultdict(dict)
    for gid, gold_mid in gold_by_group.items():
        if gold_mid and "::" in gid:
            inst, slot = gid.split("::", 1)
            if instance_ids and inst not in instance_ids:
                continue
            instance_gold[inst][slot] = gold_mid
    if instance_ids:
        instance_gold = {k: v for k, v in instance_gold.items() if k in instance_ids}
    total_instances = len(instance_gold)
    instance_slots: dict[str, dict[str, str]] = defaultdict(dict)
    for gid, pred_mid in pred_assignments.items():
        if "::" in gid:
            inst, slot = gid.split("::", 1)
            if instance_ids and inst not in instance_ids:
                continue
            instance_slots[inst][slot] = pred_mid
    for inst, gold_slots in instance_gold.items():
        pred_slots = instance_slots.get(inst, {})
        if all(pred_slots.get(s) == gold_slots.get(s) for s in gold_slots):
            exact_instances += 1
    for gid in list(by_group.keys()):
        inst = gid.split("::", 1)[0] if "::" in gid else ""
        if instance_ids and inst not in instance_ids:
            continue
        gold_mid = gold_by_group.get(gid)
        if gold_mid is None:
            continue
        pair_total += 1
        cands = sorted(by_group[gid], key=lambda x: -x[1])
        if cands and cands[0][0] == gold_mid:
            pair_correct += 1
        if pred_assignments.get(gid) == gold_mid:
            slot_correct += 1
        slot_total += 1
    return {
        "pairwise_accuracy": pair_correct / pair_total if pair_total else 0.0,
        "slot_selection_accuracy": slot_correct / slot_total if slot_total else 0.0,
        "exact_slot_fill_accuracy": exact_instances / total_instances if total_instances else 0.0,
        "pair_correct": pair_correct,
        "pair_total": pair_total,
        "slot_correct": slot_correct,
        "slot_total": slot_total,
        "exact_instances": exact_instances,
        "total_instances": total_instances,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--run_dirs", nargs="*", default=[], help="Run dirs to evaluate (e.g. rule_baseline run0 multitask_run0)")
    ap.add_argument("--out_dir", type=Path, default=ROOT / "artifacts" / "learning_runs")
    ap.add_argument("--decoder", choices=["argmax", "one_to_one"], default="argmax")
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
    slices = _build_slices(rows)
    all_instances = set()
    for s in slices.values():
        all_instances |= s
    slice_names = ["multiple_float_like", "lower_upper_cues", "multi_entity", "overall"]
    results: dict[str, dict[str, dict]] = defaultdict(dict)
    run_dirs = list(args.run_dirs)
    if not run_dirs:
        run_dirs = ["rule_baseline"]
    for run_name in run_dirs:
        run_dir = args.out_dir / run_name
        ranker = PairwiseRanker(use_structured_features=False)
        if run_name != "rule_baseline" and run_dir.exists():
            if (run_dir / "pairwise_only.pt").exists():
                ranker.load(str(run_dir / "pairwise_only.pt"))
            elif (run_dir / "checkpoint.pt").exists():
                ranker.load(str(run_dir / "checkpoint.pt"))
        group_scores = []
        for row in rows:
            gid = row.get("group_id", "")
            mid = row.get("mention_id", "")
            sc = ranker.score(
                slot_name=row.get("slot_name", ""),
                slot_role=row.get("slot_role"),
                mention_surface=row.get("mention_surface", ""),
                context=row.get("sentence_or_context"),
            )
            if ranker.model is None:
                sc = _rule_score(row)
            group_scores.append((gid, mid, sc))
        for slice_name in slice_names:
            if slice_name == "overall":
                inst_set = set()
            else:
                inst_set = slices.get(slice_name, set())
            metrics = _metrics_for_instances(rows, group_scores, inst_set, decoder=args.decoder)
            results[run_name][slice_name] = metrics
    out_base = args.out_dir / "bottleneck_slices"
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "slice_metrics.json", "w") as f:
        json.dump(dict(results), f, indent=2)
    for run_name in run_dirs:
        run_slice_dir = args.out_dir / run_name / "bottleneck_slices"
        run_slice_dir.mkdir(parents=True, exist_ok=True)
        with open(run_slice_dir / "slice_metrics.json", "w") as f:
            json.dump(results.get(run_name, {}), f, indent=2)
    lines = [
        "# Bottleneck slice report",
        "",
        f"**Split:** {args.split}",
        "",
        "## Slice definitions (heuristic)",
        "- **multiple_float_like:** instances with ≥3 distinct numeric mentions",
        "- **lower_upper_cues:** instances with operator_cue_match or slot_role lower_bound/upper_bound",
        "- **multi_entity:** instances with ≥2 slots",
        "- **overall:** all instances",
        "",
        "## Metrics by run and slice",
        "",
    ]
    for run_name in run_dirs:
        lines.append(f"### {run_name}")
        lines.append("| Slice | pairwise_acc | slot_acc | exact_acc | pair_total | slot_total | instances |")
        lines.append("|-------|--------------|----------|-----------|------------|------------|-----------|")
        for slice_name in slice_names:
            m = results[run_name].get(slice_name, {})
            lines.append(
                f"| {slice_name} | {m.get('pairwise_accuracy', 0):.3f} | {m.get('slot_selection_accuracy', 0):.3f} | "
                f"{m.get('exact_slot_fill_accuracy', 0):.3f} | {m.get('pair_total', 0)} | "
                f"{m.get('slot_total', 0)} | {m.get('total_instances', 0)} |"
            )
        lines.append("")
    report_path = out_base / "bottleneck_slice_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_base / 'slice_metrics.json'}, {report_path}")
    print(json.dumps(dict(results), indent=2))


if __name__ == "__main__":
    main()
