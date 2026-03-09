#!/usr/bin/env python3
"""Label disagreements between optimization_role_repair and optimization_role_relation_repair.

Reads the per-instance comparison CSV and assigns coarse heuristic categories to each
disagreement (where pred_opt_repair != pred_relation_repair).

Categories (heuristic):
  - objective_vs_bound: one method assigns a value to an objective-like slot, the other to a bound-like slot (by name/role)
  - lower_vs_upper_bound: min/max slot confusion (slot name or value swap)
  - total_vs_per_unit: total-like vs per-unit slot confusion
  - percent_ratio_confusion: percent/ratio slot vs non-ratio
  - wrong_variable_association: same value assigned to different slots (variable swap)
  - other

Output: results/paper/nlp4lp_focused_disagreement_labels.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _slot_looks_objective(name: str) -> bool:
    n = (name or "").lower()
    return any(x in n for x in ["profit", "revenue", "return", "cost", "objective", "per", "unit"])


def _slot_looks_bound(name: str) -> bool:
    n = (name or "").lower()
    return any(x in n for x in ["min", "max", "minimum", "maximum", "least", "most", "bound", "at least", "at most"])


def _slot_looks_lower(name: str) -> bool:
    n = (name or "").lower()
    return "min" in n or "minimum" in n or "least" in n or "lower" in n


def _slot_looks_upper(name: str) -> bool:
    n = (name or "").lower()
    return "max" in n or "maximum" in n or "most" in n or "upper" in n


def _slot_looks_total(name: str) -> bool:
    n = (name or "").lower()
    return any(x in n for x in ["total", "budget", "available", "capacity", "limit"])


def _slot_looks_per_unit(name: str) -> bool:
    n = (name or "").lower()
    return "per" in n or "each" in n or "unit" in n


def _slot_looks_ratio(name: str) -> bool:
    n = (name or "").lower()
    return "percent" in n or "ratio" in n or "fraction" in n or "percentage" in n


def _value_looks_percent(val) -> bool:
    if val is None:
        return False
    s = str(val).strip()
    return "%" in s or (s.replace(".", "").replace("-", "").isdigit() and "." in s and float(s) <= 1.0 and float(s) >= 0)


def label_disagreement(
    gold_assignments: dict,
    pred_opt: dict,
    pred_relation: dict,
    slot_names: list[str],
) -> list[str]:
    """Assign one or more heuristic labels for this instance's disagreement."""
    labels: set[str] = set()
    if pred_opt == pred_relation:
        return []

    slots_opt = set(pred_opt.keys())
    slots_rel = set(pred_relation.keys())
    common_slots = slots_opt & slots_rel
    for slot in common_slots:
        v_opt = pred_opt.get(slot)
        v_rel = pred_relation.get(slot)
        if v_opt == v_rel:
            continue
        if _slot_looks_lower(slot) and _slot_looks_upper(slot):
            pass
        elif _slot_looks_lower(slot) or _slot_looks_upper(slot):
            other_slots = [s for s in slot_names if s != slot]
            for o in other_slots:
                if _slot_looks_lower(o) or _slot_looks_upper(o):
                    if (pred_opt.get(slot) == pred_relation.get(o) and pred_relation.get(slot) == pred_opt.get(o)):
                        labels.add("lower_vs_upper_bound")
                        break
        if _slot_looks_total(slot) and _slot_looks_per_unit(slot):
            pass
        elif _slot_looks_total(slot):
            for o in slot_names:
                if o != slot and _slot_looks_per_unit(o):
                    if pred_opt.get(slot) == pred_relation.get(o) or pred_relation.get(slot) == pred_opt.get(o):
                        labels.add("total_vs_per_unit")
                        break
        if _slot_looks_ratio(slot):
            if _value_looks_percent(v_opt) != _value_looks_percent(v_rel):
                labels.add("percent_ratio_confusion")
        if _slot_looks_objective(slot) and _slot_looks_bound(slot):
            pass
        elif _slot_looks_objective(slot):
            for o in slot_names:
                if o != slot and _slot_looks_bound(o):
                    if pred_opt.get(slot) == pred_relation.get(o) or pred_relation.get(slot) == pred_opt.get(o):
                        labels.add("objective_vs_bound")
                        break
    if not labels:
        labels.add("wrong_variable_association")
    return sorted(labels)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-csv", type=Path, default=ROOT / "results" / "paper" / "nlp4lp_focused_per_instance_comparison.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    args = p.parse_args()

    if not args.comparison_csv.exists():
        print(f"Comparison file not found: {args.comparison_csv}", file=sys.stderr)
        print("Run build_nlp4lp_per_instance_comparison.py first.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with open(args.comparison_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_opt = json.loads(row.get("pred_opt_repair") or "{}")
            pred_relation = json.loads(row.get("pred_relation_repair") or "{}")
            if pred_opt == pred_relation:
                continue
            try:
                gold = json.loads(row.get("gold_assignments") or "{}")
            except Exception:
                gold = {}
            slot_names = list(gold.keys()) or list(pred_opt.keys()) or list(pred_relation.keys())
            labels = label_disagreement(gold, pred_opt, pred_relation, slot_names)
            rows.append({
                "query_id": row.get("query_id", ""),
                "gold_doc_id": row.get("gold_doc_id", ""),
                "pred_doc_id": row.get("pred_doc_id", ""),
                "schema_hit": row.get("schema_hit", ""),
                "disagreement_labels": "|".join(labels),
                "n_filled_opt": row.get("n_filled_opt", ""),
                "n_filled_relation": row.get("n_filled_relation", ""),
            })

    out_path = args.out_dir / "nlp4lp_focused_disagreement_labels.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["query_id", "gold_doc_id", "pred_doc_id", "schema_hit", "disagreement_labels", "n_filled_opt", "n_filled_relation"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    label_counts: dict[str, int] = {}
    for row in rows:
        for lab in (row.get("disagreement_labels") or "").split("|"):
            if lab:
                label_counts[lab] = label_counts.get(lab, 0) + 1
    print(f"Wrote {out_path} ({len(rows)} disagreement rows)")
    print("Label counts:", label_counts)


if __name__ == "__main__":
    main()
