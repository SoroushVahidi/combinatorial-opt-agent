#!/usr/bin/env python3
"""Compare downstream methods on the top three failure families (schema_hit=1).

Reads:
  - results/paper/nlp4lp_focused_per_instance_comparison.csv
  - results/paper/nlp4lp_downstream_per_query_failure_family.csv (from build_nlp4lp_failure_audit.py)

Produces:
  - results/paper/nlp4lp_three_bottlenecks_comparison.md

Focus: wrong_variable_association, multiple_float_like_values, lower_vs_upper_bound.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

THREE_FAMILIES = ["wrong_variable_association", "multiple_float_like_values", "lower_vs_upper_bound"]

METHOD_COLS = [
    ("exact_opt_repair", "opt_repair"),
    ("exact_relation_repair", "relation_repair"),
    ("exact_anchor", "anchor_linking"),
    ("exact_beam", "bottomup_beam"),
    ("exact_entity_semantic_beam", "entity_semantic_beam"),
]


def _float(val, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Compare methods on three bottleneck failure families")
    p.add_argument("--comparison-csv", type=Path, default=ROOT / "results" / "paper" / "nlp4lp_focused_per_instance_comparison.csv")
    p.add_argument("--family-csv", type=Path, default=ROOT / "results" / "paper" / "nlp4lp_downstream_per_query_failure_family.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    args = p.parse_args()

    if not args.comparison_csv.exists():
        print(f"Missing: {args.comparison_csv}", file=__import__("sys").stderr)
        raise SystemExit(1)

    # Load comparison rows keyed by query_id
    comp_by_qid: dict[str, dict] = {}
    with open(args.comparison_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row.get("query_id", "")
            if qid:
                comp_by_qid[qid] = row

    # Load per-query failure family (if available)
    qid_to_family: dict[str, str] = {}
    if args.family_csv.exists():
        with open(args.family_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = row.get("query_id", "")
                fam = row.get("failure_family", "other")
                if qid:
                    qid_to_family[qid] = fam
    else:
        print("Warning: per-query failure family CSV not found; run build_nlp4lp_failure_audit.py first.", file=__import__("sys").stderr)

    # Restrict to schema_hit=1
    schema_hit_qids = {qid for qid, r in comp_by_qid.items() if int(r.get("schema_hit") or 0) == 1}

    lines = [
        "# NLP4LP Three-Bottleneck Comparison (schema_hit=1)",
        "",
        "Focus: wrong_variable_association, multiple_float_like_values, lower_vs_upper_bound.",
        "",
    ]

    for fam in THREE_FAMILIES:
        # Rows in this family with schema_hit=1
        qids_in_fam = [qid for qid in schema_hit_qids if qid_to_family.get(qid) == fam]
        if not qids_in_fam:
            lines.append(f"## {fam}")
            lines.append("")
            lines.append("No rows with this failure family in the per-query family file (or file missing).")
            lines.append("")
            continue

        rows = [comp_by_qid[qid] for qid in qids_in_fam]
        n = len(rows)
        lines.append(f"## {fam} (n={n})")
        lines.append("")

        # Mean exact* per method
        lines.append("### Mean exact20_on_hits (within family)")
        lines.append("")
        for col, label in METHOD_COLS:
            if col not in rows[0]:
                continue
            vals = [_float(r.get(col)) for r in rows]
            mean_v = sum(vals) / len(vals) if vals else 0.0
            lines.append(f"- **{label}**: {mean_v:.4f}")
        lines.append("")

        # Wins: entity_semantic_beam vs others (only if column present)
        if "exact_entity_semantic_beam" in rows[0]:
            lines.append("### Queries where entity_semantic_beam beats other methods")
            lines.append("")
            for col, label in METHOD_COLS:
                if col == "exact_entity_semantic_beam":
                    continue
                if col not in rows[0]:
                    continue
                wins = sum(1 for r in rows if _float(r.get("exact_entity_semantic_beam")) > _float(r.get(col)))
                lines.append(f"- entity_semantic_beam > {label}: **{wins}** / {n}")
            lines.append("")
        lines.append("")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "nlp4lp_three_bottlenecks_comparison.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
