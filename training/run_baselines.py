"""
Run all retrieval baselines on the same leak-free eval file and output
paper-ready results: JSON (full config + metrics per baseline) and CSV (one row per baseline).
Deterministic when --seed is set for eval regeneration.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_catalog(catalog_path: Path | None) -> list[dict]:
    if catalog_path is not None and catalog_path.exists():
        with open(catalog_path, encoding="utf-8") as f:
            # Support both JSON (list of problems) and JSONL (one object per line).
            if catalog_path.suffix == ".jsonl":
                items: list[dict] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    # Map doc-level JSONL used by custom benchmarks into the problem schema.
                    if "doc_id" in obj or "text" in obj:
                        doc_id = obj.get("doc_id") or obj.get("id")
                        text = obj.get("text") or obj.get("description") or ""
                        meta = obj.get("meta") or {}
                        name = meta.get("name") or doc_id
                        aliases = meta.get("aliases") or []
                        items.append(
                            {
                                "id": doc_id,
                                "name": name,
                                "description": text,
                                "aliases": aliases,
                                "meta": meta,
                            }
                        )
                    else:
                        items.append(obj)
                return items
            return json.load(f)
    from retrieval.search import _load_catalog as load_
    return load_()


def _generate_eval_instances(
    catalog: list[dict],
    seed: int,
    num_instances: int,
    problem_ids: list[str] | None,
) -> list[tuple[str, str]]:
    sys.path.insert(0, str(ROOT))
    from training.generate_samples import generate_queries_for_problem
    id_set = set(problem_ids) if problem_ids else None
    rng = random.Random(seed)
    all_pairs: list[tuple[str, str]] = []
    for problem in catalog:
        pid = problem.get("id")
        if not pid or (id_set is not None and pid not in id_set):
            continue
        passage = (
            (problem.get("name") or "")
            + " "
            + " ".join(problem.get("aliases") or [])
            + " "
            + (problem.get("description") or "")
        ).strip()
        if not passage:
            continue
        queries = generate_queries_for_problem(problem, rng, target_per_problem=50)
        for q in queries:
            all_pairs.append((q.strip(), pid))
    rng.shuffle(all_pairs)
    return all_pairs[:num_instances]


def _load_eval_instances(eval_path: Path, catalog: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            if "problem_id" in obj:
                pid = obj["problem_id"]
                if pid:
                    out.append((q, pid))
            elif "relevant_doc_id" in obj:
                pid = obj["relevant_doc_id"]
                if pid:
                    out.append((q, pid))
            elif "problem_index" in obj:
                idx = int(obj["problem_index"])
                if 0 <= idx < len(catalog) and catalog[idx].get("id"):
                    out.append((q, catalog[idx]["id"]))
    return out


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Evaluate all baselines on same leak-free eval set")
    p.add_argument("--catalog", type=Path, default=None, help="Catalog JSON (default: data/processed/all_problems.json or extended)")
    p.add_argument("--splits", type=Path, default=None, help="Splits JSON for split-aware eval")
    p.add_argument("--split", type=str, default=None, choices=("train", "dev", "test"))
    p.add_argument(
        "--eval-file",
        "--eval",
        dest="eval_file",
        type=Path,
        default=None,
        help="Eval JSONL (default: data/processed/eval_<split>.jsonl)",
    )
    p.add_argument("--regenerate", action="store_true", help="Regenerate eval instances")
    p.add_argument("--num", type=int, default=500, help="Num eval instances when regenerating")
    p.add_argument("--seed", type=int, default=999, help="RNG seed for eval generation (deterministic)")
    p.add_argument("--k", type=int, default=10, help="Top-k for MRR/nDCG/Coverage")
    p.add_argument("--baselines", type=str, nargs="+", default=["bm25", "tfidf", "sbert"],
                   help="Baseline names (default: bm25 tfidf sbert)")
    p.add_argument("--results-dir", type=Path, default=None, help="Output dir (default: results/)")
    p.add_argument("--dataset-name", type=str, default=None,
                   help="Short name for dataset (e.g. test_normal, test_masked, resocratic, nl4opt). "
                        "Used to name CI outputs.")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="Number of bootstrap resamples for CIs (0 to disable).")
    p.add_argument("--ci-metrics", type=str, default="P@1,MRR@10,nDCG@10",
                   help="Comma-separated list of metrics to compute CIs for (default: P@1,MRR@10,nDCG@10).")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for compact metrics JSON (e.g. for NLP4LP benchmark).",
    )
    args = p.parse_args()

    catalog = _load_catalog(args.catalog)
    id_to_name = {p.get("id"): (p.get("name") or p.get("id") or "") for p in catalog if p.get("id")}

    split_name: str | None = None
    split_problem_ids: list[str] | None = None
    if args.splits or args.split:
        if not args.splits or not args.split:
            p.error("Both --splits and --split are required for split-aware eval.")
        from training.splits import load_splits, get_problem_ids_for_split
        splits = load_splits(args.splits)
        split_name = args.split.strip().lower()
        split_problem_ids = get_problem_ids_for_split(splits, split_name)

    results_dir = Path(args.results_dir or ROOT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_path = args.eval_file
    if eval_path is None and split_name:
        eval_path = ROOT / "data" / "processed" / f"eval_{split_name}.jsonl"
    if eval_path is None:
        eval_path = ROOT / "data" / "processed" / "eval_500.jsonl"
    eval_path = Path(eval_path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    if args.regenerate or not eval_path.exists():
        print(f"Generating eval instances (seed={args.seed})...")
        pairs = _generate_eval_instances(
            catalog, seed=args.seed, num_instances=args.num, problem_ids=split_problem_ids
        )
        with open(eval_path, "w", encoding="utf-8") as f:
            for q, pid in pairs:
                f.write(json.dumps({"query": q, "problem_id": pid}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} instances to {eval_path}")
        eval_pairs = pairs
    else:
        eval_pairs = _load_eval_instances(eval_path, catalog)
        print(f"Loaded {len(eval_pairs)} instances from {eval_path}")

    if not eval_pairs:
        print("No eval instances. Exiting.")
        return

    k = max(args.k, 1)
    from retrieval.baselines import get_baseline
    from training.metrics import compute_metrics, diagnose_failures
    from training.bootstrap import bootstrap_ci

    config = {
        "catalog": str(args.catalog) if args.catalog else "default",
        "splits": str(args.splits) if args.splits else None,
        "split": split_name,
        "eval_file": str(eval_path),
        "num_instances": len(eval_pairs),
        "seed": args.seed,
        "k": k,
        "baselines": list(args.baselines),
    }
    all_metrics: dict[str, dict[str, float]] = {}
    per_baseline_items: dict[str, list[tuple[list[str], str]]] = {}
    per_baseline_instances: dict[str, list[tuple[str, list[str], str]]] = {}
    baseline_times: dict[str, float] = {}

    for bl_name in args.baselines:
        print(f"Running baseline: {bl_name}...", flush=True)
        start_time = time.time()
        try:
            baseline = get_baseline(bl_name)
            baseline.fit(catalog)
        except Exception as e:
            print(f"Skipping {bl_name!r}: {e}", flush=True)
            continue
        results_for_metrics: list[tuple[list[str], str]] = []
        instances_for_diagnosis: list[tuple[str, list[str], str]] = []
        for query, expected_id in eval_pairs:
            ranked = baseline.rank(query, top_k=k)
            ranked_names = [id_to_name.get(pid, "") for pid, _ in ranked]
            expected_name = id_to_name.get(expected_id, "")
            results_for_metrics.append((ranked_names, expected_name))
            instances_for_diagnosis.append((query, ranked_names, expected_name))
        metrics = compute_metrics(results_for_metrics, k=k)
        all_metrics[bl_name] = metrics
        per_baseline_items[bl_name] = results_for_metrics
        per_baseline_instances[bl_name] = instances_for_diagnosis
        baseline_times[bl_name] = time.time() - start_time

    # Decide dataset name for file prefixes
    if args.dataset_name:
        dataset_name = args.dataset_name
    elif split_name:
        dataset_name = f"{split_name}"
    else:
        # Derive from eval file basename as a fallback
        dataset_name = Path(eval_path).stem

    out_name = f"baselines_{dataset_name}"
    out_json = results_dir / f"{out_name}.json"
    out_csv = results_dir / f"{out_name}.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        failure_modes: dict[str, dict] = {
            bl_name: diagnose_failures(per_baseline_instances[bl_name], k=k)
            for bl_name in per_baseline_instances
        }
        json.dump(
            {"config": config, "baselines": all_metrics, "failure_modes": failure_modes},
            f,
            indent=2,
        )

    metric_keys = ["P@1", "P@5", f"MRR@{k}", f"nDCG@{k}", f"Coverage@{k}"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["baseline"] + metric_keys)
        for bl_name in all_metrics:
            row = [bl_name] + [all_metrics[bl_name].get(m, 0.0) for m in metric_keys]
            w.writerow(row)

    # Optional bootstrap CIs
    if args.bootstrap > 0 and all_metrics:
        B = args.bootstrap
        metrics_for_ci = [m.strip() for m in args.ci_metrics.split(",") if m.strip()]
        ci_rows = []
        for bl_name, items in per_baseline_items.items():
            def make_metric_fn(metric_name: str):
                from training.metrics import precision_at_k, reciprocal_rank_at_k, ndcg_at_k
                def _fn(subitems: list[tuple[list[str], str]]) -> float:
                    # subitems: list of (ranked_names, expected_name)
                    if not subitems:
                        return 0.0
                    if metric_name == "P@1":
                        return sum(precision_at_k(names, exp, 1) for names, exp in subitems) / len(subitems)
                    if metric_name == "P@5":
                        return sum(precision_at_k(names, exp, 5) for names, exp in subitems) / len(subitems)
                    if metric_name == f"MRR@{k}":
                        return sum(reciprocal_rank_at_k(names, exp, k) for names, exp in subitems) / len(subitems)
                    if metric_name == f"nDCG@{k}":
                        return sum(ndcg_at_k(names, exp, k) for names, exp in subitems) / len(subitems)
                    if metric_name == f"Coverage@{k}":
                        # coverage@k is identical to P@k on this dataset
                        return sum(precision_at_k(names, exp, k) for names, exp in subitems) / len(subitems)
                    # Fallback: return 0
                    return 0.0
                return _fn

            for mname in metrics_for_ci:
                # Normalize metric name to match keys
                if mname == "P@1":
                    metric_key = "P@1"
                elif mname == "P@5":
                    metric_key = "P@5"
                elif mname in (f"MRR@{k}", "MRR@10"):
                    metric_key = f"MRR@{k}"
                elif mname in (f"nDCG@{k}", "nDCG@10"):
                    metric_key = f"nDCG@{k}"
                elif mname in (f"Coverage@{k}", "Coverage@10"):
                    metric_key = f"Coverage@{k}"
                else:
                    continue
                fn = make_metric_fn(metric_key)
                mean, lo, hi = bootstrap_ci(fn, items, B=B, seed=args.seed)
                ci_rows.append({
                    "dataset": dataset_name,
                    "baseline": bl_name,
                    "metric": metric_key,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "B": B,
                    "seed": args.seed,
                })

        ci_csv = results_dir / f"{out_name}_ci.csv"
        ci_json = results_dir / f"{out_name}_ci.json"
        # Write CSV
        with open(ci_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "baseline", "metric", "mean", "ci_low", "ci_high", "B", "seed"])
            for row in ci_rows:
                w.writerow([
                    row["dataset"],
                    row["baseline"],
                    row["metric"],
                    row["mean"],
                    row["ci_low"],
                    row["ci_high"],
                    row["B"],
                    row["seed"],
                ])
        # Write JSON
        with open(ci_json, "w", encoding="utf-8") as f:
            json.dump(ci_rows, f, indent=2)

    # Optional compact metrics JSON (used for NLP4LP benchmark).
    if args.out is not None and all_metrics:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary: dict = {
            "config": config,
            "k": k,
            "baselines": {},
        }
        for bl_name, metrics in all_metrics.items():
            summary["baselines"][bl_name] = {
                "Recall@1": metrics.get("P@1", 0.0),
                "Recall@5": metrics.get("P@5", 0.0),
                "Recall@10": metrics.get(f"Coverage@{k}", 0.0),
                f"MRR@{k}": metrics.get(f"MRR@{k}", 0.0),
                f"nDCG@{k}": metrics.get(f"nDCG@{k}", 0.0),
                "runtime_sec": baseline_times.get(bl_name, 0.0),
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print()
    print("=" * 60)
    print("Baseline results (leak-free eval)")
    print("=" * 60)
    for bl_name, m in all_metrics.items():
        print(f"  {bl_name}: P@1={m['P@1']:.4f} P@5={m['P@5']:.4f} MRR@{k}={m.get(f'MRR@{k}', 0):.4f}")
    print("=" * 60)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
