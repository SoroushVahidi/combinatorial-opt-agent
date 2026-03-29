#!/usr/bin/env python3
"""Build downstream failure audit from focused evaluation artifacts.

Reads:
  - results/paper/nlp4lp_focused_per_instance_comparison.csv
  - results/paper/nlp4lp_focused_disagreement_labels.csv (optional)

Produces:
  - results/paper/nlp4lp_downstream_failure_patterns.csv
  - results/paper/nlp4lp_downstream_hard_cases.csv
  - results/paper/nlp4lp_downstream_failure_audit.md

No modeling code modified; analysis only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAILURE_FAMILIES = [
    "objective_vs_bound",
    "lower_vs_upper_bound",
    "total_vs_per_unit",
    "percent_ratio_confusion",
    "wrong_variable_association",
    "multiple_float_like_values",
    "currency_vs_scalar_confusion",
    "missing_operator_grounding",
    "other",
]


def _slot_looks_min(s: str) -> bool:
    s = (s or "").lower()
    return "min" in s or "minimum" in s or "least" in s


def _slot_looks_max(s: str) -> bool:
    s = (s or "").lower()
    return "max" in s or "maximum" in s or "most" in s


def _slot_looks_total(s: str) -> bool:
    s = (s or "").lower()
    return any(x in s for x in ["total", "budget", "available", "capacity"]) and "per" not in s


def _slot_looks_per_unit(s: str) -> bool:
    s = (s or "").lower()
    return "per" in s or "each" in s


def _slot_looks_ratio(s: str) -> bool:
    s = (s or "").lower()
    return "percent" in s or "ratio" in s or "fraction" in s


def _slot_looks_objective(s: str) -> bool:
    s = (s or "").lower()
    return any(x in s for x in ["profit", "revenue", "cost", "return", "objective"])


def _slot_looks_bound(s: str) -> bool:
    return _slot_looks_min(s) or _slot_looks_max(s)


def _val_looks_percent(v) -> bool:
    if v is None:
        return False
    s = str(v).strip()
    return "%" in s or (s.replace(".", "").replace("-", "").isdigit() and 0 <= float(s) <= 1.0)


def _val_looks_currency(v) -> bool:
    if v is None:
        return False
    s = str(v)
    return "$" in s or "€" in s or (isinstance(v, (int, float)) and abs(v) > 100)


def infer_failure_family(
    gold: dict,
    pred_opt: dict,
    pred_relation: dict,
    schema_hit: bool,
    exact_opt: float,
    exact_relation: float,
    disagreement_labels: str,
) -> tuple[str, str]:
    """Return (failure_family, brief_reason). Uses heuristics from slot names and values."""
    slots = list(gold.keys()) or list(pred_opt.keys()) or list(pred_relation.keys())
    if not slots:
        return "other", "No slots"

    if disagreement_labels:
        labs = [x.strip() for x in disagreement_labels.split("|") if x.strip()]
        for fam in FAILURE_FAMILIES:
            if fam in labs:
                return fam, f"From disagreement label: {fam}"

    both_wrong = schema_hit and (exact_opt == 0 or exact_opt == "") and (exact_relation == 0 or exact_relation == "")
    opt_wrong_rel_right = schema_hit and (exact_opt == 0 or exact_opt == "") and exact_relation and float(exact_relation) > 0

    for slot in slots:
        g = gold.get(slot)
        po = pred_opt.get(slot)
        pr = pred_relation.get(slot)
        if _slot_looks_min(slot) or _slot_looks_max(slot):
            for s2 in slots:
                if s2 == slot:
                    continue
                if _slot_looks_min(s2) or _slot_looks_max(s2):
                    if (po == gold.get(s2) and pr == gold.get(slot)) or (pr == gold.get(s2) and po == gold.get(slot)):
                        return "lower_vs_upper_bound", "Min/max slot value swap"
        if _slot_looks_ratio(slot):
            if _val_looks_percent(po) != _val_looks_percent(g) or _val_looks_percent(pr) != _val_looks_percent(g):
                return "percent_ratio_confusion", "Percent/ratio slot value mismatch"
        if _slot_looks_total(slot):
            for s2 in slots:
                if s2 != slot and _slot_looks_per_unit(s2):
                    if po == gold.get(s2) or pr == gold.get(s2):
                        return "total_vs_per_unit", "Total vs per-unit slot confusion"
        if _slot_looks_objective(slot):
            for s2 in slots:
                if s2 != slot and _slot_looks_bound(s2):
                    if po == gold.get(s2) or pr == gold.get(s2):
                        return "objective_vs_bound", "Objective vs bound slot confusion"

    num_float_like = sum(1 for v in list(gold.values()) + list(pred_opt.values()) + list(pred_relation.values())
                        if v is not None and isinstance(v, (int, float)))
    if len(slots) >= 2 and num_float_like >= 4:
        return "multiple_float_like_values", "Many numeric values; likely wrong association"

    for slot in slots:
        g = gold.get(slot)
        if _val_looks_currency(g) and slot and "budget" not in (slot or "").lower() and "cost" not in (slot or "").lower():
            if po != g or pr != g:
                return "currency_vs_scalar_confusion", "Currency value in non-currency slot or swap"

    if both_wrong and slots:
        return "wrong_variable_association", "Same-schema wrong slot assignment"
    if both_wrong:
        return "missing_operator_grounding", "Possible operator/role grounding failure"
    return "other", "Unclassified"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--comparison-csv", type=Path, default=ROOT / "results" / "paper" / "nlp4lp_focused_per_instance_comparison.csv")
    p.add_argument("--disagreement-csv", type=Path, default=ROOT / "results" / "paper" / "nlp4lp_focused_disagreement_labels.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    args = p.parse_args()

    if not args.comparison_csv.exists():
        print(f"Missing: {args.comparison_csv}", file=__import__("sys").stderr)
        print("Run build_nlp4lp_per_instance_comparison.py and optionally run_nlp4lp_focused_eval.py first.", file=__import__("sys").stderr)
        raise SystemExit(1)

    rows: list[dict] = []
    with open(args.comparison_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    disagreement_by_qid: dict[str, str] = {}
    if args.disagreement_csv.exists():
        with open(args.disagreement_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                disagreement_by_qid[row.get("query_id", "")] = row.get("disagreement_labels", "")

    for row in rows:
        try:
            gold = json.loads(row.get("gold_assignments") or "{}")
        except Exception:
            gold = {}
        try:
            pred_opt = json.loads(row.get("pred_opt_repair") or "{}")
        except Exception:
            pred_opt = {}
        try:
            pred_relation = json.loads(row.get("pred_relation_repair") or "{}")
        except Exception:
            pred_relation = {}
        schema_hit = int(row.get("schema_hit") or 0)
        exact_opt = row.get("exact_opt_repair")
        exact_relation = row.get("exact_relation_repair")
        try:
            exact_opt_f = float(exact_opt) if exact_opt != "" else 0.0
        except (TypeError, ValueError):
            exact_opt_f = 0.0
        try:
            exact_relation_f = float(exact_relation) if exact_relation != "" else 0.0
        except (TypeError, ValueError):
            exact_relation_f = 0.0
        fam, reason = infer_failure_family(
            gold, pred_opt, pred_relation,
            schema_hit=schema_hit,
            exact_opt=exact_opt_f,
            exact_relation=exact_relation_f,
            disagreement_labels=disagreement_by_qid.get(row.get("query_id", ""), ""),
        )
        row["_failure_family"] = fam
        row["_brief_reason"] = reason
        row["_exact_opt_f"] = exact_opt_f
        row["_exact_relation_f"] = exact_relation_f

    # --- A. Failure patterns summary (only schema-hit assignment failures) ---
    fam_stats: dict[str, dict] = {f: {"count": 0, "count_schema_hit": 0, "count_opt_wrong_relation_right": 0, "count_both_wrong": 0, "query_ids": []} for f in FAILURE_FAMILIES}
    fam_stats["other"] = {"count": 0, "count_schema_hit": 0, "count_opt_wrong_relation_right": 0, "count_both_wrong": 0, "query_ids": []}

    for row in rows:
        schema_hit = int(row.get("schema_hit") or 0)
        exact_opt_f = row.get("_exact_opt_f", 0.0)
        exact_relation_f = row.get("_exact_relation_f", 0.0)
        if not schema_hit or (exact_opt_f >= 1.0 and exact_relation_f >= 1.0):
            continue
        fam = row.get("_failure_family", "other")
        if fam not in fam_stats:
            fam_stats[fam] = {"count": 0, "count_schema_hit": 0, "count_opt_wrong_relation_right": 0, "count_both_wrong": 0, "query_ids": []}
        fam_stats[fam]["count"] += 1
        fam_stats[fam]["count_schema_hit"] += 1
        if exact_opt_f < 0.01 and exact_relation_f > 0:
            fam_stats[fam]["count_opt_wrong_relation_right"] += 1
        if exact_opt_f < 0.01 and exact_relation_f < 0.01:
            fam_stats[fam]["count_both_wrong"] += 1
        qid = row.get("query_id", "")
        if qid and qid not in fam_stats[fam]["query_ids"]:
            fam_stats[fam]["query_ids"].append(qid)
            if len(fam_stats[fam]["query_ids"]) > 20:
                fam_stats[fam]["query_ids"] = fam_stats[fam]["query_ids"][:20]

    explanations = {
        "objective_vs_bound": "Objective (e.g. profit) slot filled with bound value or vice versa.",
        "lower_vs_upper_bound": "Min and max (or lower/upper) slot values swapped.",
        "total_vs_per_unit": "Total budget/capacity vs per-unit cost/profit confusion.",
        "percent_ratio_confusion": "Percent/ratio slot given non-percent value or wrong slot.",
        "wrong_variable_association": "Correct schema but number assigned to wrong slot.",
        "multiple_float_like_values": "Many similar numbers; wrong many-to-many association.",
        "currency_vs_scalar_confusion": "Currency vs plain scalar type or slot mix-up.",
        "missing_operator_grounding": "Operator (min/max/total/per) not grounded to correct slot.",
        "other": "Unclassified or mixed.",
    }

    patterns_path = args.out_dir / "nlp4lp_downstream_failure_patterns.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(patterns_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["failure_family", "count", "count_schema_hit", "count_opt_wrong_relation_right", "count_both_wrong", "representative_query_ids", "short_explanation"])
        for fam in FAILURE_FAMILIES:
            st = fam_stats.get(fam, fam_stats["other"])
            rep = ";".join((st["query_ids"] or [])[:10])
            w.writerow([
                fam,
                st["count"],
                st["count_schema_hit"],
                st["count_opt_wrong_relation_right"],
                st["count_both_wrong"],
                rep,
                explanations.get(fam, explanations["other"]),
            ])
    print(f"Wrote {patterns_path}")

    # --- Per-query failure family (for bottleneck analysis) ---
    family_path = args.out_dir / "nlp4lp_downstream_per_query_failure_family.csv"
    with open(family_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "schema_hit", "failure_family"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "query_id": r.get("query_id", ""),
                "schema_hit": r.get("schema_hit", ""),
                "failure_family": r.get("_failure_family", "other"),
            })
    print(f"Wrote {family_path}")

    # --- B. Hard cases (schema_hit=1, exact_opt=0, exact_relation=0) ---
    hard = [r for r in rows if int(r.get("schema_hit") or 0) == 1
            and (r.get("_exact_opt_f") or 0) < 0.01 and (r.get("_exact_relation_f") or 0) < 0.01]
    hard_path = args.out_dir / "nlp4lp_downstream_hard_cases.csv"
    hard_cols_base = [
        "query_id", "gold_doc_id", "pred_doc_id", "mentions_summary", "gold_assignments",
        "pred_opt_repair", "pred_relation_repair",
    ]
    hard_cols_optional = [
        "pred_anchor_linking", "pred_bottomup_beam", "pred_entity_semantic_beam",
        "exact_anchor", "exact_beam", "exact_entity_semantic_beam",
    ]
    hard_cols_end = ["likely_failure_family", "brief_reason"]
    sample = rows[0] if rows else {}
    hard_cols = hard_cols_base + [c for c in hard_cols_optional if c in sample] + hard_cols_end
    with open(hard_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hard_cols, extrasaction="ignore")
        w.writeheader()
        for r in hard:
            out = {col: r.get(col, "") for col in hard_cols_base}
            out.update({col: r.get(col, "") for col in hard_cols_optional if col in hard_cols})
            out["likely_failure_family"] = r.get("_failure_family", "other")
            out["brief_reason"] = r.get("_brief_reason", "")
            w.writerow(out)
    print(f"Wrote {hard_path} ({len(hard)} rows)")

    # --- C. Markdown report ---
    md_path = args.out_dir / "nlp4lp_downstream_failure_audit.md"
    sorted_fams = sorted(
        [f for f in FAILURE_FAMILIES if fam_stats.get(f, {}).get("count", 0) > 0],
        key=lambda f: fam_stats.get(f, {}).get("count", 0),
        reverse=True,
    )
    lines = [
        "# NLP4LP Downstream Failure Audit",
        "",
        "Based on focused per-instance comparison and disagreement labels (schema-hit emphasis).",
        "",
        "## 1. Top failure families (by count)",
        "",
    ]
    for i, fam in enumerate(sorted_fams, 1):
        st = fam_stats.get(fam, {})
        lines.append(f"{i}. **{fam}** — count={st.get('count', 0)}, schema_hit={st.get('count_schema_hit', 0)}, opt_wrong_relation_right={st.get('count_opt_wrong_relation_right', 0)}, both_wrong={st.get('count_both_wrong', 0)}")
        lines.append(f"   - {explanations.get(fam, explanations['other'])}")
        lines.append("")
    lines.extend([
        "## 2. Signals currently missing",
        "",
        "- **Explicit min/max pairing:** Slot-slot relation that two slots form a min-max pair; constraint that the smaller value goes to min slot.",
        "- **Sentence/clause scope:** Which numbers sit in the same sentence or same clause; reduces wrong variable association when many numbers.",
        "- **Operator anchoring:** Stronger binding of phrases like \"at least X\" to the slot that expects a lower bound.",
        "- **Percent vs scalar type lock:** Hard rule that values with % or in [0,1] when context says percent go only to ratio/percent slots.",
        "- **Total vs per-unit lock:** Once a slot is tagged total-like (budget, capacity), only total-like mentions allowed; idem for per-unit.",
        "",
        "## 3. Highest-value next deterministic change",
        "",
        "**Recommendation: Enforce min-max ordering in incremental admissibility.**",
        "",
        "When the schema has both a min and a max slot for the same quantity, the partial-state admissibility check should require that the value assigned to the min slot is less than or equal to the value assigned to the max slot (when both are numeric). Today we check operator tags (min/max mention) but not the numeric ordering of the assigned values. Adding a hard constraint `value(min_slot) <= value(max_slot)` in `_is_partial_admissible` would reject many lower_vs_upper_bound failures without changing retrieval or other assignment logic.",
        "",
    ])
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
