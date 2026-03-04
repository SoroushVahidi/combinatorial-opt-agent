"""
Generate 500 evaluation instances (query, expected problem index) from the catalog,
then run the retrieval model and report Precision@1 and Precision@5.
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


def _generate_eval_instances(seed: int = 999, num_instances: int = 500):
    """Build (query, problem_index) pairs using same catalog and query generator as training."""
    catalog = _load_catalog()
    sys.path.insert(0, str(ROOT))
    from training.generate_samples import generate_queries_for_problem

    rng = random.Random(seed)
    all_pairs = []  # (query, problem_index)
    for idx, problem in enumerate(catalog):
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
            all_pairs.append((q.strip(), idx))
    rng.shuffle(all_pairs)
    return all_pairs[:num_instances]


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate retrieval precision on 500 instances")
    p.add_argument("--num", type=int, default=500, help="Number of eval instances (default 500)")
    p.add_argument("--seed", type=int, default=999, help="RNG seed for eval set (default 999)")
    p.add_argument("--eval-file", type=Path, default=None, help="Path to eval JSONL (default: data/processed/eval_500.jsonl)")
    p.add_argument("--regenerate", action="store_true", help="Regenerate eval set from catalog")
    args = p.parse_args()

    eval_path = args.eval_file or ROOT / "data" / "processed" / "eval_500.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    if args.regenerate or not eval_path.exists():
        print("Generating evaluation instances...")
        pairs = _generate_eval_instances(seed=args.seed, num_instances=args.num)
        with open(eval_path, "w", encoding="utf-8") as f:
            for query, idx in pairs:
                f.write(json.dumps({"query": query, "problem_index": idx}, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pairs)} instances to {eval_path}")
        eval_pairs = pairs
    else:
        eval_pairs = []
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                eval_pairs.append((obj["query"], obj["problem_index"]))
        print(f"Loaded {len(eval_pairs)} instances from {eval_path}")

    catalog = _load_catalog()

    print("Loading retrieval model and building index...")
    from sentence_transformers import SentenceTransformer
    from retrieval.search import _default_model_path, build_index, search

    model = SentenceTransformer(_default_model_path())
    embeddings = build_index(catalog, model)

    correct_at_1 = 0
    correct_at_5 = 0
    n = len(eval_pairs)
    for i, (query, expected_idx) in enumerate(eval_pairs):
        if (i + 1) % 100 == 0:
            print(f"  evaluated {i + 1}/{n}...", flush=True)
        results = search(
            query,
            catalog=catalog,
            model=model,
            embeddings=embeddings,
            top_k=5,
        )
        top_names = [p.get("name", "") for p, _ in results]
        expected_name = catalog[expected_idx].get("name", "")
        if top_names and top_names[0] == expected_name:
            correct_at_1 += 1
        if expected_name in top_names:
            correct_at_5 += 1

    p_at_1 = correct_at_1 / n if n else 0.0
    p_at_5 = correct_at_5 / n if n else 0.0
    print()
    print("=" * 60)
    print("Retrieval evaluation (classification by top result)")
    print("=" * 60)
    print(f"  Instances:     {n}")
    print(f"  Correct @1:   {correct_at_1}")
    print(f"  Correct @5:   {correct_at_5}")
    print(f"  Precision@1:  {p_at_1:.4f}  ({100 * p_at_1:.2f}%)")
    print(f"  Precision@5:  {p_at_5:.4f}  ({100 * p_at_5:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
