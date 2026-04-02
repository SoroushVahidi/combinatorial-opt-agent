#!/usr/bin/env python3
"""
Build TARGETED synthetic pairwise ranker data for known bottlenecks.

High-precision templated examples only. NOT weak labels; each (slot, mention)
pair is designed so the correct fill is unambiguous from the template.
Categories: percent_vs_scalar, total_vs_per_unit, min_max_bounds,
capacity_demand_limit, objective_vs_bound, float_values, paraphrase.

Output: artifacts/learning_ranker_data/targeted_synth/train.jsonl, stats.json, README.md.
Schema matches current pairwise ranker (group_id, slot_*, mention_*, label, feat_*, etc.).
Extra keys: synthetic_category, template_id (for analysis only; trainer ignores them).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "learning_ranker_data" / "targeted_synth"


def _feat_type_match(expected: str | None, actual: str) -> int:
    if not expected or actual == "unknown":
        return 0
    if expected == actual:
        return 1
    if expected == "percent" and actual != "percent":
        return -1
    if expected in ("int", "float") and actual in ("int", "float"):
        return 1
    return 0


def _feat_operator_cue(slot_role: str | None, context: str) -> int:
    if not slot_role or not context:
        return 0
    ctx = context.lower()
    if "lower" in (slot_role or "").lower() or "min" in (slot_role or "").lower():
        if "at least" in ctx or "no less" in ctx or "minimum" in ctx:
            return 1
        if "at most" in ctx or "maximum" in ctx:
            return -1
    if "upper" in (slot_role or "").lower() or "max" in (slot_role or "").lower():
        if "at most" in ctx or "no more" in ctx or "maximum" in ctx:
            return 1
        if "at least" in ctx or "minimum" in ctx:
            return -1
    return 0


def _feat_total_per(slot_name: str, context: str) -> tuple[int, int]:
    s = (slot_name + " " + context).lower()
    total = 1 if "total" in s or "budget" in s or "capacity" in s or "demand" in s else 0
    per = 1 if "per unit" in s or "per item" in s or "per unit" in context.lower() else 0
    return total, per


def _feat_overlap(slot_name: str, context: str) -> int:
    a = set(w for w in slot_name.lower().split() if len(w) > 1)
    b = set(w for w in context.lower().split() if len(w) > 1)
    return min(5, len(a & b))


def row(
    instance_id: str,
    slot_id: str,
    slot_name: str,
    slot_role: str | None,
    expected_type: str | None,
    mention_id: str,
    mention_surface: str,
    mention_value: float,
    mention_type: str,
    context: str,
    label: int,
    category: str,
    template_id: str,
) -> dict:
    group_id = f"{instance_id}::{slot_id}"
    feat_type = _feat_type_match(expected_type, mention_type)
    feat_op = _feat_operator_cue(slot_role, context)
    feat_total, feat_per = _feat_total_per(slot_name, context)
    feat_overlap = _feat_overlap(slot_name, context)
    return {
        "group_id": group_id,
        "instance_id": instance_id,
        "schema_name": instance_id,
        "slot_id": slot_id,
        "slot_name": slot_name,
        "slot_role": slot_role,
        "expected_type": expected_type,
        "mention_id": mention_id,
        "mention_surface": mention_surface,
        "mention_value": mention_value,
        "mention_type_bucket": mention_type,
        "sentence_or_context": context,
        "label": label,
        "gold_mention_id_for_slot": mention_id if label == 1 else None,
        "feat_type_match": feat_type,
        "feat_operator_cue_match": feat_op,
        "feat_total_like": feat_total,
        "feat_per_unit_like": feat_per,
        "feat_slot_mention_overlap": feat_overlap,
        "synthetic_category": category,
        "template_id": template_id,
    }


# ----- Templates: (context, slots with correct mention, other mentions as distractors) -----
# Each slot: (slot_id, slot_name, slot_role, expected_type, correct_surface, correct_value, correct_type)
# Other mentions: (surface, value, type) for negatives
TEMPLATES = [
    # A. Percent vs scalar
    {
        "category": "percent_vs_scalar",
        "template_id": "A1",
        "context": "At least 20% of total budget must go to marketing. The fixed cost is 5000 units.",
        "slots": [
            ("min_pct_budget", "min_pct_budget", "lower_bound", "percent", "20%", 0.2, "percent"),
            ("fixed_cost", "fixed_cost", None, "int", "5000", 5000.0, "int"),
        ],
        "other_mentions": [("20", 20.0, "int")],
    },
    {
        "category": "percent_vs_scalar",
        "template_id": "A2",
        "context": "No more than 12.5% can be allocated to risk assets. We have 12.5 units in reserve.",
        "slots": [
            ("max_pct_risk", "max_pct_risk", "upper_bound", "percent", "12.5%", 0.125, "percent"),
            ("reserve_units", "reserve_units", None, "float", "12.5", 12.5, "float"),
        ],
        "other_mentions": [],
    },
    {
        "category": "percent_vs_scalar",
        "template_id": "A3",
        "context": "The margin must be at least 15%. Total output is 100 units.",
        "slots": [
            ("min_margin_pct", "min_margin_pct", "lower_bound", "percent", "15%", 0.15, "percent"),
            ("total_output", "total_output", None, "int", "100", 100.0, "int"),
        ],
        "other_mentions": [("15", 15.0, "int")],
    },
    # B. Total vs per-unit
    {
        "category": "total_vs_per_unit",
        "template_id": "B1",
        "context": "Total budget is 50000. Profit per unit is 7.5. We must stay within both limits.",
        "slots": [
            ("total_budget", "total_budget", "upper_bound", "int", "50000", 50000.0, "int"),
            ("profit_per_unit", "profit_per_unit", None, "float", "7.5", 7.5, "float"),
        ],
        "other_mentions": [],
    },
    {
        "category": "total_vs_per_unit",
        "template_id": "B2",
        "context": "Cost per item is 3.2. Total demand for the period is 180 units.",
        "slots": [
            ("cost_per_item", "cost_per_item", None, "float", "3.2", 3.2, "float"),
            ("total_demand", "total_demand", None, "int", "180", 180.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "total_vs_per_unit",
        "template_id": "B3",
        "context": "Labor cost per hour is 22.5. Total available hours is 400.",
        "slots": [
            ("labor_cost_per_hour", "labor_cost_per_hour", None, "float", "22.5", 22.5, "float"),
            ("total_hours_available", "total_hours_available", "upper_bound", "int", "400", 400.0, "int"),
        ],
        "other_mentions": [],
    },
    # C. Min/max and lower/upper bound
    {
        "category": "min_max_bounds",
        "template_id": "C1",
        "context": "Minimum investment is 10000. Maximum production must not exceed 200 units.",
        "slots": [
            ("min_investment", "min_investment", "lower_bound", "int", "10000", 10000.0, "int"),
            ("max_production", "max_production", "upper_bound", "int", "200", 200.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "min_max_bounds",
        "template_id": "C2",
        "context": "At least 50 units must be produced. No more than 500 units can be stored.",
        "slots": [
            ("lower_bound_production", "lower_bound_production", "lower_bound", "int", "50", 50.0, "int"),
            ("upper_bound_storage", "upper_bound_storage", "upper_bound", "int", "500", 500.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "min_max_bounds",
        "template_id": "C3",
        "context": "No less than 0.2 and no more than 0.8 of capacity can be used.",
        "slots": [
            ("min_capacity_use", "min_capacity_use", "lower_bound", "float", "0.2", 0.2, "float"),
            ("max_capacity_use", "max_capacity_use", "upper_bound", "float", "0.8", 0.8, "float"),
        ],
        "other_mentions": [],
    },
    # D. Capacity / demand / resource limit
    {
        "category": "capacity_demand_limit",
        "template_id": "D1",
        "context": "Warehouse capacity is 2500 units. Demand this month is 1200.",
        "slots": [
            ("warehouse_capacity", "warehouse_capacity", "upper_bound", "int", "2500", 2500.0, "int"),
            ("demand", "demand", "lower_bound", "int", "1200", 1200.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "capacity_demand_limit",
        "template_id": "D2",
        "context": "Labor hours available are 320. Machine time limit is 200 hours.",
        "slots": [
            ("labor_hours_available", "labor_hours_available", "upper_bound", "int", "320", 320.0, "int"),
            ("machine_time_limit", "machine_time_limit", "upper_bound", "int", "200", 200.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "capacity_demand_limit",
        "template_id": "D3",
        "context": "Resource budget is 100000. Demand requirement is at least 500 units.",
        "slots": [
            ("resource_budget", "resource_budget", "upper_bound", "int", "100000", 100000.0, "int"),
            ("demand_requirement", "demand_requirement", "lower_bound", "int", "500", 500.0, "int"),
        ],
        "other_mentions": [],
    },
    # E. Objective coefficient vs bound
    {
        "category": "objective_vs_bound",
        "template_id": "E1",
        "context": "Each unit yields 4.5 profit. Production cannot exceed 200 units.",
        "slots": [
            ("profit_per_unit", "profit_per_unit", None, "float", "4.5", 4.5, "float"),
            ("production_cap", "production_cap", "upper_bound", "int", "200", 200.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "objective_vs_bound",
        "template_id": "E2",
        "context": "Storage cost is 1.2 per unit. Budget is capped at 100000.",
        "slots": [
            ("storage_cost_per_unit", "storage_cost_per_unit", None, "float", "1.2", 1.2, "float"),
            ("budget_cap", "budget_cap", "upper_bound", "int", "100000", 100000.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "objective_vs_bound",
        "template_id": "E3",
        "context": "Revenue per unit is 25. The maximum order size is 150.",
        "slots": [
            ("revenue_per_unit", "revenue_per_unit", None, "int", "25", 25.0, "int"),
            ("max_order_size", "max_order_size", "upper_bound", "int", "150", 150.0, "int"),
        ],
        "other_mentions": [],
    },
    # F. Float-heavy values
    {
        "category": "float_values",
        "template_id": "F1",
        "context": "The discount rate is 0.05. Minimum fill rate is 0.75.",
        "slots": [
            ("discount_rate", "discount_rate", None, "float", "0.05", 0.05, "float"),
            ("min_fill_rate", "min_fill_rate", "lower_bound", "float", "0.75", 0.75, "float"),
        ],
        "other_mentions": [],
    },
    {
        "category": "float_values",
        "template_id": "F2",
        "context": "Unit weight is 1.5 kg. Price is 99.95 per unit.",
        "slots": [
            ("unit_weight", "unit_weight", None, "float", "1.5", 1.5, "float"),
            ("price_per_unit", "price_per_unit", None, "float", "99.95", 99.95, "float"),
        ],
        "other_mentions": [],
    },
    {
        "category": "float_values",
        "template_id": "F3",
        "context": "At least 0.2 of inventory must be reserved. Maximum utilization 0.9.",
        "slots": [
            ("min_reserve_ratio", "min_reserve_ratio", "lower_bound", "float", "0.2", 0.2, "float"),
            ("max_utilization", "max_utilization", "upper_bound", "float", "0.9", 0.9, "float"),
        ],
        "other_mentions": [],
    },
    # G. Paraphrase / short variants
    {
        "category": "paraphrase",
        "template_id": "G1",
        "context": "Cap 500. Need 200 min.",
        "slots": [
            ("capacity", "capacity", "upper_bound", "int", "500", 500.0, "int"),
            ("min_requirement", "min_requirement", "lower_bound", "int", "200", 200.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "paraphrase",
        "template_id": "G2",
        "context": "Budget: 50k. Profit/unit: 3.5. Limit 1000.",
        "slots": [
            ("budget", "budget", "upper_bound", "int", "50k", 50000.0, "int"),
            ("profit_per_unit", "profit_per_unit", None, "float", "3.5", 3.5, "float"),
            ("limit", "limit", "upper_bound", "int", "1000", 1000.0, "int"),
        ],
        "other_mentions": [],
    },
    {
        "category": "paraphrase",
        "template_id": "G3",
        "context": "Min investment 5k. Max 20% in any single asset.",
        "slots": [
            ("min_investment", "min_investment", "lower_bound", "int", "5k", 5000.0, "int"),
            ("max_pct_single_asset", "max_pct_single_asset", "upper_bound", "percent", "20%", 0.2, "percent"),
        ],
        "other_mentions": [],
    },
    # More percent vs scalar
    {"category": "percent_vs_scalar", "template_id": "A4", "context": "Allocate at least 30% to bonds. The floor is 30 units.", "slots": [("min_pct_bonds", "min_pct_bonds", "lower_bound", "percent", "30%", 0.3, "percent"), ("floor_units", "floor_units", "lower_bound", "int", "30", 30.0, "int")], "other_mentions": []},
    {"category": "percent_vs_scalar", "template_id": "A5", "context": "Maximum 8% fee. Cap at 8 units.", "slots": [("max_fee_pct", "max_fee_pct", "upper_bound", "percent", "8%", 0.08, "percent"), ("cap_units", "cap_units", "upper_bound", "int", "8", 8.0, "int")], "other_mentions": []},
    # More total vs per-unit
    {"category": "total_vs_per_unit", "template_id": "B4", "context": "Total cost 75000. Cost per unit 2.5.", "slots": [("total_cost", "total_cost", None, "int", "75000", 75000.0, "int"), ("cost_per_unit", "cost_per_unit", None, "float", "2.5", 2.5, "float")], "other_mentions": []},
    {"category": "total_vs_per_unit", "template_id": "B5", "context": "Revenue per unit 15. Total revenue target 45000.", "slots": [("revenue_per_unit", "revenue_per_unit", None, "int", "15", 15.0, "int"), ("total_revenue_target", "total_revenue_target", None, "int", "45000", 45000.0, "int")], "other_mentions": []},
    # More min/max
    {"category": "min_max_bounds", "template_id": "C4", "context": "Lower bound 10. Upper bound 100.", "slots": [("lower_bound", "lower_bound", "lower_bound", "int", "10", 10.0, "int"), ("upper_bound", "upper_bound", "upper_bound", "int", "100", 100.0, "int")], "other_mentions": []},
    {"category": "min_max_bounds", "template_id": "C5", "context": "Minimum order 25. Maximum order 500.", "slots": [("min_order", "min_order", "lower_bound", "int", "25", 25.0, "int"), ("max_order", "max_order", "upper_bound", "int", "500", 500.0, "int")], "other_mentions": []},
    # More capacity/demand
    {"category": "capacity_demand_limit", "template_id": "D4", "context": "Plant capacity 1000. Customer demand 600.", "slots": [("plant_capacity", "plant_capacity", "upper_bound", "int", "1000", 1000.0, "int"), ("customer_demand", "customer_demand", None, "int", "600", 600.0, "int")], "other_mentions": []},
    {"category": "capacity_demand_limit", "template_id": "D5", "context": "Storage limit 8000. Minimum order quantity 100.", "slots": [("storage_limit", "storage_limit", "upper_bound", "int", "8000", 8000.0, "int"), ("min_order_quantity", "min_order_quantity", "lower_bound", "int", "100", 100.0, "int")], "other_mentions": []},
    # More objective vs bound
    {"category": "objective_vs_bound", "template_id": "E4", "context": "Contribution margin 6.5 per unit. Capacity limit 350.", "slots": [("contribution_margin", "contribution_margin", None, "float", "6.5", 6.5, "float"), ("capacity_limit", "capacity_limit", "upper_bound", "int", "350", 350.0, "int")], "other_mentions": []},
    # More float
    {"category": "float_values", "template_id": "F4", "context": "Interest rate 0.04. Minimum ratio 0.25.", "slots": [("interest_rate", "interest_rate", None, "float", "0.04", 0.04, "float"), ("min_ratio", "min_ratio", "lower_bound", "float", "0.25", 0.25, "float")], "other_mentions": []},
    {"category": "float_values", "template_id": "F5", "context": "Efficiency 0.92. Max load 12.5.", "slots": [("efficiency", "efficiency", None, "float", "0.92", 0.92, "float"), ("max_load", "max_load", "upper_bound", "float", "12.5", 12.5, "float")], "other_mentions": []},
    # More paraphrase
    {"category": "paraphrase", "template_id": "G4", "context": "Budget 25k. Cost/unit 4. Max 600.", "slots": [("budget", "budget", "upper_bound", "int", "25k", 25000.0, "int"), ("cost_per_unit", "cost_per_unit", None, "int", "4", 4.0, "int"), ("max_quantity", "max_quantity", "upper_bound", "int", "600", 600.0, "int")], "other_mentions": []},
    {"category": "paraphrase", "template_id": "G5", "context": "Min 10. Max 100. Pct 25%.", "slots": [("min_val", "min_val", "lower_bound", "int", "10", 10.0, "int"), ("max_val", "max_val", "upper_bound", "int", "100", 100.0, "int"), ("pct", "pct", None, "percent", "25%", 0.25, "percent")], "other_mentions": []},
]


def run_template(t: dict, instance_prefix: str = "synth") -> list[dict]:
    rows_out = []
    instance_id = f"{instance_prefix}_{t['template_id']}"
    context = t["context"]
    category = t["category"]
    template_id = t["template_id"]
    slots = t["slots"]
    other = t.get("other_mentions") or []
    all_mentions = [(m[4], m[5], m[6]) for m in slots] + other  # (surface, value, type) per slot
    for si, (slot_id, slot_name, slot_role, expected_type, correct_surf, correct_val, correct_type) in enumerate(slots):
        # Positive
        rows_out.append(
            row(
                instance_id=instance_id,
                slot_id=slot_id,
                slot_name=slot_name,
                slot_role=slot_role,
                expected_type=expected_type,
                mention_id="m_correct",
                mention_surface=correct_surf,
                mention_value=correct_val,
                mention_type=correct_type,
                context=context,
                label=1,
                category=category,
                template_id=template_id,
            )
        )
        # Negatives: other slots' correct mentions + other_mentions
        for mi, (surf, val, typ) in enumerate(all_mentions):
            if (surf, val, typ) == (correct_surf, correct_val, correct_type):
                continue
            rows_out.append(
                row(
                    instance_id=instance_id,
                    slot_id=slot_id,
                    slot_name=slot_name,
                    slot_role=slot_role,
                    expected_type=expected_type,
                    mention_id=f"m_{mi}",
                    mention_surface=surf,
                    mention_value=val,
                    mention_type=typ,
                    context=context,
                    label=0,
                    category=category,
                    template_id=template_id,
                )
            )
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build targeted synthetic pairwise ranker data")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--instance_prefix", type=str, default="synth")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for t in TEMPLATES:
        all_rows.extend(run_template(t, args.instance_prefix))

    # Write JSONL (include extra keys; trainer ignores unknown keys)
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {train_path} ({len(all_rows)} rows)")

    # Stats
    pos = sum(1 for r in all_rows if r.get("label") == 1)
    neg = len(all_rows) - pos
    by_cat: dict[str, int] = {}
    for r in all_rows:
        c = r.get("synthetic_category", "unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
    slot_types: dict[str, int] = {}
    for r in all_rows:
        rt = r.get("slot_role") or "none"
        slot_types[rt] = slot_types.get(rt, 0) + 1
    num_types: dict[str, int] = {}
    for r in all_rows:
        nt = r.get("mention_type_bucket", "unknown")
        num_types[nt] = num_types.get(nt, 0) + 1
    stats = {
        "total_rows": len(all_rows),
        "positive_labels": pos,
        "negative_labels": neg,
        "num_templates": len(TEMPLATES),
        "by_synthetic_category": by_cat,
        "by_slot_role": slot_types,
        "by_mention_type_bucket": num_types,
        "note": "Targeted high-precision synthetic data. Labels are template-defined, not weak.",
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")

    readme = """# Targeted synthetic pairwise ranker data

Generated by `tools/build_targeted_synth_ranker_data.py`.

- **Purpose:** High-precision auxiliary training for known bottlenecks (percent vs scalar, total vs per-unit, min/max bounds, capacity/demand, objective vs bound, float values, paraphrase).
- **Labels:** Template-defined correct (slot, mention) pairs; not weak. Each row has label 0 or 1 from the template design.
- **Use:** Auxiliary pretraining only. Benchmark remains held-out NLP4LP. Do not treat as gold benchmark data.
- **Schema:** Same as NLP4LP pairwise ranker (group_id, slot_*, mention_*, label, feat_*). Extra keys: synthetic_category, template_id (for analysis).
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Wrote {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
