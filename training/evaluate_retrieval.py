"""
Evaluate retrieval: generate split-aware eval instances (query, expected problem)
and compute P@1, P@5, MRR@10, nDCG@10, Coverage@10.
When --splits and --split are set, eval instances are only for problems in that
split (no overlap with training problems if you train on train and eval on test).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent


def _load_catalog():
    from retrieval.search import _load_catalog as load_
    return load_()


def _generate_eval_instances(
    catalog: list[dict],
    seed: int = 999,
    num_instances: int = 500,
    problem_ids: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Build (query, problem_id) pairs. If problem_ids is set, only generate for those problems
    (for leak-free eval: only test/dev problems). Otherwise all problems (legacy).
    """
    sys.path.insert(0, str(ROOT))
    from training.generate_samples import generate_queries_for_problem

    id_set = set(problem_ids) if problem_ids else None
    rng = random.Random(seed)
    all_pairs: list[tuple[str, str]] = []  # (query, problem_id)
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


def _load_eval_instances(
    eval_path: Path,
    catalog: list[dict],
) -> list[tuple[str, str]]:
    """
    Load eval instances from JSONL. Each line: {"query": str, "problem_id": str} or
    {"query": str, "problem_index": int}. Returns list of (query, problem_id).
    """
    id_to_idx = {p.get("id"): i for i, p in enumerate(catalog) if p.get("id")}
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
                if pid and pid in id_to_idx:
                    out.append((q, pid))
            elif "problem_index" in obj:
                idx = int(obj["problem_index"])
                if 0 <= idx < len(catalog) and catalog[idx].get("id"):
                    out.append((q, catalog[idx]["id"]))
            else:
                continue
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate retrieval with optional split-aware leak-free eval")
    p.add_argument("--num", type=int, default=500, help="Number of eval instances when regenerating (default 500)")
    p.add_argument("--seed", type=int, default=999, help="RNG seed for eval set (default 999)")
    p.add_argument("--eval-file", type=Path, default=None, help="Path to eval JSONL (default: derived from split or data/processed/eval_500.jsonl)")
    p.add_argument("--regenerate", action="store_true", help="Regenerate eval set")
    p.add_argument("--splits", type=Path, default=None, help="Path to splits JSON (train/dev/test problem IDs)")
    p.add_argument("--split", type=str, default=None, choices=("train", "dev", "test"),
                   help="Evaluate only on this split (requires --splits). Use test for final metrics.")
    p.add_argument("--results-dir", type=Path, default=None, help="Write metrics JSON here (default: results/)")
    p.add_argument("--top-k", type=int, default=10, help="Top-k for MRR/nDCG/Coverage (default 10)")
    p.add_argument(
        "--no-short-query-expansion",
        action="store_true",
        help="Disable short-query expansion heuristic when calling retrieval.search.search()",
    )
    args = p.parse_args()

    catalog = _load_catalog()
    split_problem_ids: list[str] | None = None
    split_name: str | None = None
    if args.splits is not None or args.split is not None:
        if args.splits is None or args.split is None:
            p.error("Both --splits and --split are required for split-aware evaluation.")
        from training.splits import load_splits, get_problem_ids_for_split
        splits = load_splits(args.splits)
        split_name = args.split.strip().lower()
        split_problem_ids = get_problem_ids_for_split(splits, split_name)

    results_dir = args.results_dir or ROOT / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_path = args.eval_file
    if eval_path is None and split_name:
        eval_path = ROOT / "data" / "processed" / f"eval_{split_name}.jsonl"
    if eval_path is None:
        eval_path = ROOT / "data" / "processed" / "eval_500.jsonl"
    eval_path = Path(eval_path)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    if args.regenerate or not eval_path.exists():
        print("Generating evaluation instances...")
        pairs = _generate_eval_instances(
            catalog,
            seed=args.seed,
            num_instances=args.num,
            problem_ids=split_problem_ids,
        )
        with open(eval_path, "w", encoding="utf-8") as f:
            for query, problem_id in pairs:
                f.write(json.dumps({"query": query, "problem_id": problem_id}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} instances to {eval_path}")
        eval_pairs = pairs
    else:
        eval_pairs = _load_eval_instances(eval_path, catalog)
        print(f"Loaded {len(eval_pairs)} instances from {eval_path}")

    if not eval_pairs:
        print("No eval instances. Exiting.")
        return

    id_to_name = {p.get("id"): p.get("name", "") for p in catalog if p.get("id")}

    print("Loading retrieval model and building index...")
    from sentence_transformers import SentenceTransformer
    from retrieval.search import _default_model_path, build_index, search

    model = SentenceTransformer(_default_model_path())
    embeddings = build_index(catalog, model)
    k = max(args.top_k, 1)

    results_for_metrics: list[tuple[list[str], str]] = []
    for i, (query, expected_id) in enumerate(eval_pairs):
        if (i + 1) % 100 == 0:
            print(f"  evaluated {i + 1}/{len(eval_pairs)}...", flush=True)
        search_results = search(
            query,
            catalog=catalog,
            model=model,
            embeddings=embeddings,
            top_k=k,
            expand_short_queries=not args.no_short_query_expansion,
        )
        top_names = [p.get("name", "") for p, _ in search_results]
        expected_name = id_to_name.get(expected_id, "")
        results_for_metrics.append((top_names, expected_name))

    from training.metrics import compute_metrics
    metrics = compute_metrics(results_for_metrics, k=k)

    print()
    print("=" * 60)
    print("Retrieval evaluation (split-aware, leak-free when using --splits --split test)")
    print("=" * 60)
    print(f"  Instances:     {len(eval_pairs)}")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}")
    print("=" * 60)

    out_json = results_dir / (f"eval_{split_name}.json" if split_name else "eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"split": split_name, "num_instances": len(eval_pairs), "metrics": metrics}, f, indent=2)
    print(f"Wrote metrics to {out_json}")


if __name__ == "__main__":
    main()
