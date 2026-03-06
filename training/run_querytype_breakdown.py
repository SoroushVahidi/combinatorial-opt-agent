"""Ablation B: evaluate baselines by query type (template, description, alias_only, other)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_catalog():
    from retrieval.search import _load_catalog as load_
    return load_()


def _classify_query_type(query: str, problem: dict) -> str:
    q = (query or "").strip()
    name = (problem.get("name") or "").strip()
    aliases = [a.strip() for a in (problem.get("aliases") or []) if a]
    desc = (problem.get("description") or "").strip()
    if q in aliases:
        return "alias_only"
    if q == name:
        return "name_only"
    if q == desc or q == (desc.split(".")[0].strip() + "."):
        return "description"
    TEMPLATE_PREFIXES = (
        "What is ", "Describe ", "Formulation for ", "ILP for ", "Integer program for ",
        "Linear program for ", "How do I formulate ", "Give me the formulation of ",
        "Optimization problem:", "Mathematical model for ", "Variables and constraints for ",
        "Find the integer program for ", "Write the ILP for ", "Combinatorial optimization:",
        "Problem:", "Need to ", "Looking for formulation of ", "Explain ", "Define ",
        "Can you give the IP for ", "MIP formulation for ", "Mixed integer program for ",
    )
    if any(q.startswith(pref) for pref in TEMPLATE_PREFIXES):
        return "template"
    return "other"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run query-type breakdown ablation")
    p.add_argument("--eval-file", type=Path, default=None,
                   help="Eval JSONL (default: data/processed/eval_test.jsonl)")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    eval_path = args.eval_file or ROOT / "data" / "processed" / "eval_test.jsonl"
    results_dir = Path(args.results_dir or ROOT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    catalog = _load_catalog()
    by_id = {p.get("id"): p for p in catalog if p.get("id")}

    # Load eval instances and classify
    buckets: dict[str, list[tuple[str, str]]] = {"template": [], "description": [], "alias_only": [], "name_only": [], "other": []}
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            pid = obj.get("problem_id")
            if not q or not pid:
                continue
            prob = by_id.get(pid)
            if not prob:
                continue
            qt = _classify_query_type(q, prob)
            buckets.setdefault(qt, []).append((q, pid))

    from retrieval.baselines import get_baseline
    from training.metrics import compute_metrics

    id_to_name = {p.get("id"): p.get("name", "") for p in catalog if p.get("id")}
    baselines = ["bm25", "tfidf", "sbert", "sbert_finetuned"]

    out_path = results_dir / "querytype_breakdown.csv"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("query_type,baseline,P@1,MRR@10\n")
        for qt, pairs in buckets.items():
            if not pairs:
                continue
            for bl_name in baselines:
                try:
                    baseline = get_baseline(bl_name)
                    baseline.fit(catalog)
                except Exception as e:
                    print(f"Skipping {bl_name!r} for {qt}: {e}")
                    continue
                results_for_metrics: list[tuple[list[str], str]] = []
                for q, pid in pairs:
                    ranked = baseline.rank(q, top_k=args.k)
                    ranked_names = [id_to_name.get(xpid, "") for xpid, _ in ranked]
                    expected_name = id_to_name.get(pid, "")
                    results_for_metrics.append((ranked_names, expected_name))
                metrics = compute_metrics(results_for_metrics, k=args.k)
                f.write(
                    f"{qt},{bl_name},{metrics['P@1']},{metrics[f'MRR@{args.k}']}\n"
                )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

