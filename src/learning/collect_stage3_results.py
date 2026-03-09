#!/usr/bin/env python3
"""Gather Stage 3 results: learned run metrics, bottleneck slices, deterministic baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "learning" / "experiment_matrix_stage3.json")
    ap.add_argument("--runs_dir", type=Path, default=ROOT / "artifacts" / "learning_runs")
    ap.add_argument("--deterministic_dir", type=Path, default=ROOT / "results" / "paper")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.runs_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        matrix = json.load(f)
    run_names = [r["run_name"] for r in matrix.get("runs", [])]
    learned: list[dict] = []
    for name in run_names:
        metrics_path = args.runs_dir / name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                learned.append(json.load(f))
        else:
            learned.append({"run_name": name, "missing": True})
    slice_path = args.runs_dir / "bottleneck_slices" / "slice_metrics.json"
    slice_data: dict = {}
    if slice_path.exists():
        with open(slice_path, encoding="utf-8") as f:
            slice_data = json.load(f)
    det_opt_repair: dict | None = None
    det_relation_repair: dict | None = None
    for base, fname in [
        ("tfidf_optimization_role_repair", "nlp4lp_downstream_orig_tfidf_optimization_role_repair.json"),
        ("tfidf_optimization_role_relation_repair", "nlp4lp_downstream_orig_tfidf_optimization_role_relation_repair.json"),
    ]:
        p = args.deterministic_dir / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            agg = data.get("aggregate", {})
            if base == "tfidf_optimization_role_repair":
                det_opt_repair = agg
            else:
                det_relation_repair = agg
    summary = {
        "learned_runs": learned,
        "bottleneck_slices": slice_data,
        "deterministic_optimization_role_repair": det_opt_repair,
        "deterministic_relation_repair": det_relation_repair,
    }
    with open(out_dir / "stage3_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lines = [
        "# Stage 3 results summary",
        "",
        "## Learned runs (ranker eval)",
        "",
        "| run_name | pairwise_acc | slot_acc | exact_acc | type_match | model_source |",
        "|----------|--------------|----------|-----------|------------|--------------|",
    ]
    for m in learned:
        if m.get("missing"):
            lines.append(f"| {m['run_name']} | (missing) | | | | |")
        else:
            lines.append(
                f"| {m.get('run_name', '')} | {m.get('pairwise_accuracy', 0):.4f} | "
                f"{m.get('slot_selection_accuracy', 0):.4f} | {m.get('exact_slot_fill_accuracy', 0):.4f} | "
                f"{m.get('type_match_after_decoding', 0):.4f} | {m.get('model_source', '')} |"
            )
    lines.append("")
    lines.append("## Deterministic baselines (full pipeline, query-level)")
    lines.append("")
    if det_opt_repair:
        lines.append(f"- **tfidf_optimization_role_repair:** param_coverage={det_opt_repair.get('param_coverage', 0):.4f}, exact20_on_hits={det_opt_repair.get('exact20_on_hits', 0):.4f}, instantiation_ready={det_opt_repair.get('instantiation_ready', 0):.4f}")
    if det_relation_repair:
        lines.append(f"- **tfidf_optimization_role_relation_repair:** param_coverage={det_relation_repair.get('param_coverage', 0):.4f}, exact20_on_hits={det_relation_repair.get('exact20_on_hits', 0):.4f}, instantiation_ready={det_relation_repair.get('instantiation_ready', 0):.4f}")
    if not det_opt_repair and not det_relation_repair:
        lines.append("(No deterministic baseline JSONs found under results/paper/)")
    lines.append("")
    lines.append("## Bottleneck slices")
    if slice_data:
        for run_name, slices in slice_data.items():
            lines.append(f"### {run_name}")
            for slice_name, m in slices.items():
                lines.append(f"- **{slice_name}:** pairwise_acc={m.get('pairwise_accuracy', 0):.3f}, slot_acc={m.get('slot_selection_accuracy', 0):.3f}, exact_acc={m.get('exact_slot_fill_accuracy', 0):.3f}")
    else:
        lines.append("(No slice_metrics.json found)")
    with open(out_dir / "stage3_results_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = out_dir / "stage3_comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "pairwise_accuracy", "slot_selection_accuracy", "exact_slot_fill_accuracy", "type_match_after_decoding", "model_source", "training_mode"])
        for m in learned:
            if m.get("missing"):
                w.writerow([m["run_name"], "", "", "", "", "", ""])
            else:
                w.writerow([
                    m.get("run_name", ""),
                    m.get("pairwise_accuracy", ""),
                    m.get("slot_selection_accuracy", ""),
                    m.get("exact_slot_fill_accuracy", ""),
                    m.get("type_match_after_decoding", ""),
                    m.get("model_source", ""),
                    m.get("training_mode", ""),
                ])
    print(f"Wrote {out_dir / 'stage3_results_summary.json'}, {out_dir / 'stage3_results_summary.md'}, {csv_path}")


if __name__ == "__main__":
    main()
