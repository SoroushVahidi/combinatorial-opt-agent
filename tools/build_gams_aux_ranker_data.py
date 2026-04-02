#!/usr/bin/env python3
"""
Convert selected GAMS structured examples into pairwise ranker auxiliary data.

Reads artifacts/gams_example_audit/gams_examples_structured.jsonl and
artifacts/gams_example_audit/selected_aux_models.json; outputs rows in the same
format as NLP4LP ranker data so the same trainer can consume them.

WEAK LABELS: We assign one positive (slot, mention) pair per parameter by
heuristic: for parameter at index i, the positive numeric is at index
i % len(numerics). All other (slot, mention) pairs for that slot are negative.
These labels are NOT gold. Do not use for benchmark evaluation.

Output: artifacts/learning_ranker_data/gams_aux/train.jsonl and stats.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_JSONL = REPO_ROOT / "artifacts" / "gams_example_audit" / "gams_examples_structured.jsonl"
DEFAULT_SELECTED = REPO_ROOT / "artifacts" / "gams_example_audit" / "selected_aux_models.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "learning_ranker_data" / "gams_aux"


def _infer_type_bucket(num: float) -> str:
    if isinstance(num, int) or (isinstance(num, float) and num == int(num)):
        return "int"
    return "float"


def _infer_slot_role(param_name: str, objective_direction: str | None) -> str | None:
    """Heuristic: min/max in name or objective sense."""
    lower = (param_name or "").lower()
    if "min" in lower or "lower" in lower or "demand" in lower:
        return "lower_bound"
    if "max" in lower or "upper" in lower or "capacity" in lower or "cap" in lower:
        return "upper_bound"
    if objective_direction == "MIN" and "cost" in lower:
        return "upper_bound"
    return None


def build_row(
    model_name: str,
    slot_id: str,
    slot_name: str,
    slot_role: str | None,
    mention_id: str,
    mention_surface: str,
    mention_value: float,
    label: int,
    context: str,
) -> dict:
    """One row in ranker format. feat_* set to 0 (no handcrafted features from GAMS)."""
    group_id = f"gams_{model_name}::{slot_id}"
    type_bucket = _infer_type_bucket(mention_value)
    return {
        "group_id": group_id,
        "instance_id": f"gams_{model_name}",
        "schema_name": model_name,
        "slot_id": slot_id,
        "slot_name": slot_name,
        "slot_role": slot_role,
        "expected_type": None,
        "mention_id": mention_id,
        "mention_surface": mention_surface,
        "mention_value": mention_value,
        "mention_type_bucket": type_bucket,
        "sentence_or_context": context,
        "label": label,
        "gold_mention_id_for_slot": mention_id if label == 1 else None,
        "feat_type_match": 0,
        "feat_operator_cue_match": 0,
        "feat_total_like": 0,
        "feat_per_unit_like": 0,
        "feat_slot_mention_overlap": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GAMS auxiliary pairwise ranker data (weak labels)")
    ap.add_argument("--audit_jsonl", type=Path, default=DEFAULT_AUDIT_JSONL)
    ap.add_argument("--selected_json", type=Path, default=DEFAULT_SELECTED)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.audit_jsonl.exists():
        print(f"ERROR: Audit JSONL not found: {args.audit_jsonl}")
        return
    selected_names = set()
    if args.selected_json.exists():
        sel = json.loads(args.selected_json.read_text(encoding="utf-8"))
        for m in sel.get("models", []):
            selected_names.add(m.get("model_name"))
    else:
        print("WARNING: selected_aux_models.json not found; using all models with params and numerics")

    # Load all structured records
    records_by_name = {}
    with open(args.audit_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            name = rec.get("model_name")
            if name:
                records_by_name[name] = rec

    if selected_names:
        records = [records_by_name[n] for n in selected_names if n in records_by_name]
    else:
        records = list(records_by_name.values())

    rows = []
    skipped_no_numerics = 0
    skipped_no_params = 0
    slot_counts: dict[str, int] = {}

    for rec in records:
        model_name = rec.get("model_name", "")
        params = rec.get("parameters") or []
        param_names = rec.get("parameter_names") or []
        numerics = rec.get("numeric_constants_sample") or []
        objective = rec.get("objective_direction")
        desc = (rec.get("description_snippet") or "")[:200]

        if not param_names:
            skipped_no_params += 1
            continue
        if not numerics:
            skipped_no_numerics += 1
            continue

        for pi, param in enumerate(params):
            slot_id = param.get("name", "")
            if not slot_id:
                continue
            slot_name = slot_id
            slot_role = _infer_slot_role(slot_name, objective)
            slot_counts[slot_name] = slot_counts.get(slot_name, 0) + 1
            # Heuristic positive: one numeric per slot at index pi % len(numerics)
            pos_idx = pi % len(numerics) if numerics else 0
            for ni, num in enumerate(numerics):
                mention_id = f"m{ni}"
                surface = str(int(num)) if isinstance(num, float) and num == int(num) else str(num)
                label = 1 if ni == pos_idx else 0
                row = build_row(
                    model_name=model_name,
                    slot_id=slot_id,
                    slot_name=slot_name,
                    slot_role=slot_role,
                    mention_id=mention_id,
                    mention_surface=surface,
                    mention_value=num,
                    label=label,
                    context=desc,
                )
                rows.append(row)

    train_path = out_dir / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {train_path} ({len(rows)} rows)")

    pos = sum(1 for r in rows if r.get("label") == 1)
    neg = len(rows) - pos
    top_slots = sorted(slot_counts.items(), key=lambda x: -x[1])[:20]
    stats = {
        "num_models_used": len([r for r in records if (r.get("parameter_names") and r.get("numeric_constants_sample"))]),
        "num_models_skipped_no_params": skipped_no_params,
        "num_models_skipped_no_numerics": skipped_no_numerics,
        "num_rows": len(rows),
        "num_positive_labels": pos,
        "num_negative_labels": neg,
        "label_note": "WEAK HEURISTIC: one positive per (model, slot) by index; not gold.",
        "top_slot_names": [s[0] for s in top_slots],
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")
    print(f"  models used: {stats['num_models_used']}, positive: {pos}, negative: {neg}")

if __name__ == "__main__":
    main()
