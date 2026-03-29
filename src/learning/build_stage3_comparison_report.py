#!/usr/bin/env python3
"""Build Stage 3 paper comparison report: learned vs deterministic, bottleneck slices, interpretation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", type=Path, default=ROOT / "artifacts" / "learning_runs" / "stage3_results_summary.json")
    ap.add_argument("--out_path", type=Path, default=ROOT / "artifacts" / "learning_runs" / "stage3_paper_comparison.md")
    args = ap.parse_args()
    if not args.summary_json.exists():
        print(f"Run collect_stage3_results.py first. Missing: {args.summary_json}", file=sys.stderr)
        sys.exit(1)
    with open(args.summary_json, encoding="utf-8") as f:
        summary = json.load(f)
    learned = summary.get("learned_runs", [])
    slice_data = summary.get("bottleneck_slices", {})
    det_opt = summary.get("deterministic_optimization_role_repair")
    det_rel = summary.get("deterministic_relation_repair")
    lines: list[str] = [
        "# Stage 3 paper comparison report",
        "",
        "First experiment round: learned downstream vs rule baseline vs deterministic baselines.",
        "",
        "---",
        "",
        "## 1. Overall learned-run comparison",
        "",
        "Evaluation: ranker data (pairwise / slot / exact instance / type-match).",
        "",
        "| run_name | pairwise_acc | slot_acc | exact_slot_fill_acc | type_match | model_source |",
        "|----------|--------------|----------|---------------------|------------|--------------|",
    ]
    for m in learned:
        if m.get("missing"):
            lines.append(f"| {m['run_name']} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {m.get('run_name', '')} | {m.get('pairwise_accuracy', 0):.4f} | "
                f"{m.get('slot_selection_accuracy', 0):.4f} | {m.get('exact_slot_fill_accuracy', 0):.4f} | "
                f"{m.get('type_match_after_decoding', 0):.4f} | {m.get('model_source', '')} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Direct comparison to deterministic optimization_role_repair")
    lines.append("")
    lines.append("Deterministic baseline: full pipeline (TF-IDF retrieval + optimization_role_repair downstream) on query-level NLP4LP eval. Metrics are **not** directly comparable (different eval setup).")
    lines.append("")
    if det_opt:
        lines.append("| metric | deterministic (opt_repair) |")
        lines.append("|--------|----------------------------|")
        for k in ["param_coverage", "exact20_on_hits", "instantiation_ready", "schema_R1"]:
            v = det_opt.get(k, "")
            if isinstance(v, (int, float)):
                v = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("(Deterministic optimization_role_repair JSON not found. Run tools/run_nlp4lp_focused_eval.py --variant orig --safe to produce results/paper/ artifacts.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Direct comparison to deterministic relation_repair")
    lines.append("")
    if det_rel:
        lines.append("| metric | deterministic (relation_repair) |")
        lines.append("|--------|----------------------------------|")
        for k in ["param_coverage", "exact20_on_hits", "instantiation_ready", "schema_R1"]:
            v = det_rel.get(k, "")
            if isinstance(v, (int, float)):
                v = f"{v:.4f}" if isinstance(v, float) else str(v)
            lines.append(f"| {k} | {v} |")
    else:
        lines.append("(Deterministic relation_repair JSON not found.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Bottleneck-slice table")
    lines.append("")
    if slice_data:
        run_names = list(slice_data.keys())
        slice_names = ["overall", "multiple_float_like", "lower_upper_cues", "multi_entity"]
        lines.append("| run | slice | pairwise_acc | slot_acc | exact_acc |")
        lines.append("|-----|-------|--------------|----------|------------|")
        for r in run_names:
            for sl in slice_names:
                m = slice_data.get(r, {}).get(sl, {})
                lines.append(f"| {r} | {sl} | {m.get('pairwise_accuracy', 0):.3f} | {m.get('slot_selection_accuracy', 0):.3f} | {m.get('exact_slot_fill_accuracy', 0):.3f} |")
    else:
        lines.append("(No bottleneck slice metrics.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Evidence-based interpretation")
    lines.append("")
    valid_learned = [m for m in learned if not m.get("missing") and m.get("pairwise_accuracy") is not None]
    rule = next((m for m in valid_learned if m.get("model_source") == "rule"), None)
    nlp4lp_only = [m for m in valid_learned if m.get("model_source") == "stage1"]
    nl4opt_runs = [m for m in valid_learned if m.get("model_source") == "stage2"]
    lines.append("- **Learning helps at all?** Compare rule_baseline vs any learned run (pairwise/slot/exact). If at least one learned run strictly improves over rule on slot_acc or exact_slot_fill_accuracy, learning helps.")
    lines.append("- **NL4Opt helps over NLP4LP-only?** Compare best NLP4LP-only run vs best NL4Opt-augmented run (pretrain_then_finetune or joint).")
    lines.append("- **Which bottleneck(s) improve most?** Inspect bottleneck-slice table: multiple_float_like, lower_upper_cues, multi_entity. Improvement on a slice suggests the model addresses that failure mode.")
    lines.append("- **Exactness vs readiness?** exact_slot_fill_accuracy (all slots correct per instance) vs pairwise_accuracy (per-slot ranking). If exact improves more than pairwise, full-instance correctness is improving.")
    lines.append("- **Inconclusive:** If data is missing or only rule_baseline was run, the round is inconclusive; run full experiment matrix and collect_stage3_results again.")
    lines.append("")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
