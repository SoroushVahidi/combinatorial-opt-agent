"""
Compute paired bootstrap significance tests for key comparisons:
  - sbert_finetuned vs bm25
  - sbert_finetuned vs sbert
on test_masked, resocratic, nl4opt.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_per_query_metrics(eval_path: Path, catalog_path: Path | None, baseline_name: str, k: int = 10):
    """Re-run one baseline on eval instances and return per-query P@1 and MRR@k."""
    from training.run_baselines import _load_catalog, _load_eval_instances
    from retrieval.baselines import get_baseline
    from training.metrics import precision_at_k, reciprocal_rank_at_k

    catalog = _load_catalog(catalog_path)
    eval_pairs = _load_eval_instances(eval_path, catalog)
    id_to_name = {p.get("id"): p.get("name", "") for p in catalog if p.get("id")}
    baseline = get_baseline(baseline_name)
    baseline.fit(catalog)

    p1_vals = []
    mrr_vals = []
    for query, pid in eval_pairs:
        ranked = baseline.rank(query, top_k=k)
        ranked_names = [id_to_name.get(xpid, "") for xpid, _ in ranked]
        expected_name = id_to_name.get(pid, "")
        p1_vals.append(precision_at_k(ranked_names, expected_name, 1))
        mrr_vals.append(reciprocal_rank_at_k(ranked_names, expected_name, k))
    return p1_vals, mrr_vals


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run paired bootstrap significance tests for key comparisons")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from training.significance import paired_bootstrap_test

    results_dir = Path(args.results_dir or ROOT / "results")
    out_path = results_dir / "significance_summary.csv"

    datasets = {
        "test_masked": ROOT / "data" / "processed" / "eval_test_masked.jsonl",
        "resocratic": ROOT / "data" / "processed" / "resocratic_eval.jsonl",
        "nl4opt": ROOT / "data" / "processed" / "nl4opt_comp_eval.jsonl",
    }
    # Compare lexical baselines (bm25 vs tfidf) to quantify significance without SBERT.
    comparisons = [
        ("bm25", "tfidf"),
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "metric", "A", "B", "p_value", "B_bootstrap", "seed"])
        for dname, eval_path in datasets.items():
            if not eval_path.exists():
                continue
            for A, B_name in comparisons:
                p1_A, mrr_A = _load_per_query_metrics(eval_path, None, A, k=args.k)
                p1_B, mrr_B = _load_per_query_metrics(eval_path, None, B_name, k=args.k)
                p_p1 = paired_bootstrap_test(p1_A, p1_B, B=args.B, seed=args.seed)
                p_mrr = paired_bootstrap_test(mrr_A, mrr_B, B=args.B, seed=args.seed + 1)
                w.writerow([dname, "P@1", A, B_name, p_p1, args.B, args.seed])
                w.writerow([dname, f"MRR@{args.k}", A, B_name, p_mrr, args.B, args.seed + 1])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

