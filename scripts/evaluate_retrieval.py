"""
Evaluate retrieval (classification) precision over 500 generated instances.

Each instance is a (query, expected_problem_id) pair. Queries are derived from
the catalog (name, alias, or description) to simulate how users might describe
a problem. We run the retrieval model and measure accuracy@1, @3, @5 and MRR.

Requires: sentence-transformers and PyTorch (e.g. pip install torch sentence-transformers).
On some platforms PyTorch 2.4+ may be required by sentence-transformers.

Usage (from project root):
    python scripts/evaluate_retrieval.py [--num-instances 500] [--seed 42]
    python scripts/evaluate_retrieval.py --instances-file data/processed/eval_instances_500.json
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.search import (
    _load_catalog,
    build_index,
    search,
)


def _first_sentence(text: str, max_len: int = 200) -> str:
    """Take first sentence or first max_len chars."""
    text = (text or "").strip()
    if not text:
        return ""
    # First sentence: up to first . ! ?
    match = re.search(r"^.[^.!?]*[.!?]", text)
    if match:
        s = match.group(0).strip()
        return s[:max_len] + ("..." if len(s) > max_len else "")
    return text[:max_len].strip() + ("..." if len(text) > max_len else "")


def generate_instances(
    catalog: list[dict],
    num_instances: int = 500,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Generate (query, expected_id) pairs. Each query is derived from a random
    problem: either name, one alias, full description, or first sentence.
    """
    rng = random.Random(seed)
    instances: list[tuple[str, str]] = []
    n = len(catalog)

    query_types = ["name", "alias", "description", "first_sentence"]

    for _ in range(num_instances):
        idx = rng.randint(0, n - 1)
        problem = catalog[idx]
        expected_id = problem.get("id") or ""
        if not expected_id:
            continue

        qtype = rng.choice(query_types)
        name = (problem.get("name") or "").strip()
        aliases = problem.get("aliases") or []
        description = (problem.get("description") or "").strip()

        if qtype == "name" and name:
            query = name
        elif qtype == "alias" and aliases:
            query = rng.choice(aliases)
        elif qtype == "description" and description:
            query = description
        elif qtype == "first_sentence" and description:
            query = _first_sentence(description)
        else:
            # Fallback: name or description
            query = name or description

        if not query.strip():
            continue
        instances.append((query.strip(), expected_id))

    return instances


def _searchable_text(problem: dict) -> str:
    """Mirror retrieval.search._searchable_text for indexing."""
    parts = [problem.get("name", "")]
    parts.extend(problem.get("aliases") or [])
    parts.append(problem.get("description", ""))
    return " ".join(p for p in parts if p)


def run_evaluation_tfidf(
    catalog: list[dict],
    instances: list[tuple[str, str]],
    top_k_max: int = 5,
) -> dict:
    """Fallback: TF-IDF + cosine similarity when sentence-transformers unavailable."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        raise SystemExit("Install scikit-learn for TF-IDF fallback: pip install scikit-learn")

    corpus = [_searchable_text(p) for p in catalog]
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=1)
    X_cat = vectorizer.fit_transform(corpus)
    n = len(instances)
    correct_at_k = [0] * (top_k_max + 1)
    mrr_sum = 0.0

    for i, (query, expected_id) in enumerate(instances):
        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i + 1}/{n} instances (TF-IDF)...", flush=True)
        q_vec = vectorizer.transform([query])
        sim = cosine_similarity(q_vec, X_cat).flatten()
        idx = sim.argsort()[::-1][:top_k_max]
        retrieved_ids = [catalog[j].get("id") or "" for j in idx]
        for k in range(1, top_k_max + 1):
            if expected_id in retrieved_ids[:k]:
                correct_at_k[k] += 1
        for r, rid in enumerate(retrieved_ids, start=1):
            if rid == expected_id:
                mrr_sum += 1.0 / r
                break

    acc = [correct_at_k[k] / n if n else 0 for k in range(1, top_k_max + 1)]
    return {
        "n": n,
        "accuracy_at_1": acc[0] if len(acc) >= 1 else 0.0,
        "accuracy_at_3": acc[2] if len(acc) >= 3 else acc[-1] if acc else 0.0,
        "accuracy_at_5": acc[4] if len(acc) >= 5 else acc[-1] if acc else 0.0,
        "accuracy_at_k": acc,
        "mrr": mrr_sum / n if n else 0,
        "model": "TF-IDF (fallback)",
    }


def run_evaluation(
    catalog: list[dict],
    instances: list[tuple[str, str]],
    top_k_max: int = 5,
    use_tfidf_fallback: bool = False,
    expand_short_queries: bool = True,
    rerank: bool = False,
    rerank_weight: float = 0.3,
    grounding_rerank: bool = False,
    grounding_lambda: float = 0.15,
):
    """Load model, run search for each instance, return metrics.

    Args:
        expand_short_queries: toggle short-query expansion (ablation flag).
        rerank: toggle deterministic lexical reranking (ablation flag).
        rerank_weight: weight for reranker term when rerank=True.
        grounding_rerank: toggle grounding-consistency second-stage rerank (ablation flag).
        grounding_lambda: weight for grounding term when grounding_rerank=True.
    """
    if use_tfidf_fallback:
        return run_evaluation_tfidf(catalog, instances, top_k_max)

    try:
        from sentence_transformers import SentenceTransformer
    except (ImportError, NameError, OSError) as e:
        print(f"  sentence-transformers not available ({e}). Using TF-IDF fallback.", flush=True)
        return run_evaluation_tfidf(catalog, instances, top_k_max)

    from retrieval.search import _default_model_path

    model = SentenceTransformer(_default_model_path())
    embeddings = build_index(catalog, model)

    correct_at_k = [0] * (top_k_max + 1)
    mrr_sum = 0.0
    n = len(instances)

    for i, (query, expected_id) in enumerate(instances):
        if (i + 1) % 100 == 0:
            print(f"  Evaluated {i + 1}/{n} instances...", flush=True)

        results = search(
            query,
            catalog=catalog,
            embeddings=embeddings,
            model=model,
            top_k=top_k_max,
            expand_short_queries=expand_short_queries,
            rerank=rerank,
            rerank_weight=rerank_weight,
            grounding_rerank=grounding_rerank,
            grounding_lambda=grounding_lambda,
        )
        retrieved_ids = [p.get("id") or "" for p, _ in results]

        for k in range(1, top_k_max + 1):
            if expected_id in retrieved_ids[:k]:
                correct_at_k[k] += 1

        rank = 0
        for r, rid in enumerate(retrieved_ids, start=1):
            if rid == expected_id:
                rank = r
                break
        if rank > 0:
            mrr_sum += 1.0 / rank

    accuracy_at_k = [correct_at_k[k] / n if n else 0 for k in range(1, top_k_max + 1)]
    mrr = mrr_sum / n if n else 0.0

    return {
        "n": n,
        "accuracy_at_1": accuracy_at_k[0],
        "accuracy_at_3": accuracy_at_k[2],
        "accuracy_at_5": accuracy_at_k[4],
        "accuracy_at_k": accuracy_at_k,
        "mrr": mrr,
        "model": "sentence-transformers",
    }


def run_ablation(
    catalog: list[dict],
    instances: list[tuple[str, str]],
    top_k_max: int = 5,
    use_tfidf_fallback: bool = False,
) -> list[dict]:
    """Run all ablation variants and return a list of result dicts.

    Variants (in order of increasing features):
      1. baseline  — no expansion, no rerank, no grounding
      2. +expansion — short-query expansion only
      3. +rerank    — expansion + lexical reranking
      4. +grounding — expansion + lexical reranking + grounding-consistency rerank

    Each result dict has ``"variant"`` plus all keys from ``run_evaluation``.
    """
    variants = [
        {
            "variant": "baseline",
            "expand_short_queries": False,
            "rerank": False,
            "grounding_rerank": False,
        },
        {
            "variant": "+expansion",
            "expand_short_queries": True,
            "rerank": False,
            "grounding_rerank": False,
        },
        {
            "variant": "+rerank",
            "expand_short_queries": True,
            "rerank": True,
            "grounding_rerank": False,
        },
        {
            "variant": "+grounding",
            "expand_short_queries": True,
            "rerank": True,
            "grounding_rerank": True,
        },
    ]
    results = []
    for v in variants:
        print(f"\n  Running ablation variant: {v['variant']}", flush=True)
        m = run_evaluation(
            catalog,
            instances,
            top_k_max=top_k_max,
            use_tfidf_fallback=use_tfidf_fallback,
            expand_short_queries=v["expand_short_queries"],
            rerank=v["rerank"],
            grounding_rerank=v["grounding_rerank"],
        )
        m["variant"] = v["variant"]
        results.append(m)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval precision on 500 instances")
    parser.add_argument("--num-instances", type=int, default=500, help="Number of test instances")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for instance generation")
    parser.add_argument("--top-k", type=int, default=5, help="Max top-k for accuracy@k")
    parser.add_argument(
        "--save-instances",
        type=str,
        default="",
        metavar="PATH",
        help="Save (query, expected_id) instances to JSON for inspection or reuse",
    )
    parser.add_argument(
        "--instances-file",
        type=str,
        default="",
        metavar="PATH",
        help="Load instances from JSON (list of {query, expected_id}); overrides --num-instances",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Run ablation: evaluate all feature variants "
            "(baseline / +expansion / +rerank / +grounding) and print a comparison table."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable deterministic lexical reranking (default: off).",
    )
    parser.add_argument(
        "--grounding-rerank",
        action="store_true",
        help="Enable grounding-consistency second-stage rerank (default: off).",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable short-query expansion (for ablation).",
    )
    args = parser.parse_args()

    print("Loading catalog...")
    catalog = _load_catalog()
    print(f"  Catalog size: {len(catalog)} problems")

    if args.instances_file:
        import json
        path = Path(args.instances_file)
        if not path.exists():
            raise SystemExit(f"Instances file not found: {path}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        instances = [(d["query"], d["expected_id"]) for d in data]
        print(f"  Loaded {len(instances)} instances from {path}")
    else:
        print(f"Generating {args.num_instances} instances (seed={args.seed})...")
        instances = generate_instances(catalog, num_instances=args.num_instances, seed=args.seed)
        print(f"  Generated {len(instances)} (query, expected_id) pairs")

    if args.save_instances:
        import json
        out_path = Path(args.save_instances)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = [{"query": q, "expected_id": eid} for q, eid in instances]
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved instances to {out_path}")

    if args.ablation:
        print("\nRunning ablation comparison across all feature variants...")
        ablation_results = run_ablation(catalog, instances, top_k_max=args.top_k)
        print("\n" + "=" * 70)
        print("RETRIEVAL ABLATION RESULTS")
        print("=" * 70)
        header = f"{'Variant':<20} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>7} {'N':>5}"
        print(header)
        print("-" * 70)
        for m in ablation_results:
            row = (
                f"{m['variant']:<20} "
                f"{m['accuracy_at_1']:>6.3f} "
                f"{m['accuracy_at_3']:>6.3f} "
                f"{m['accuracy_at_5']:>6.3f} "
                f"{m['mrr']:>7.4f} "
                f"{m['n']:>5}"
            )
            print(row)
        print("=" * 70)
        return None

    print("Loading model and building index...")
    metrics = run_evaluation(
        catalog,
        instances,
        top_k_max=args.top_k,
        expand_short_queries=not args.no_expand,
        rerank=args.rerank,
        grounding_rerank=args.grounding_rerank,
    )

    model_label = metrics.get("model", "retrieval")
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Model:         {model_label}")
    print(f"  Instances:     {metrics['n']}")
    print(f"  Accuracy@1:    {metrics['accuracy_at_1']:.2%}  (top prediction correct)")
    print(f"  Accuracy@3:    {metrics['accuracy_at_3']:.2%}  (correct in top 3)")
    print(f"  Accuracy@5:    {metrics['accuracy_at_5']:.2%}  (correct in top 5)")
    print(f"  MRR:           {metrics['mrr']:.4f}  (mean reciprocal rank)")
    print("=" * 50)

    # Precision: when we take the top-1 as the "classification", precision = accuracy@1
    print(f"\nPrecision (top-1 classification): {metrics['accuracy_at_1']:.2%}")
    if model_label == "TF-IDF (fallback)":
        print("  (Run with sentence-transformers + PyTorch for neural retrieval metrics.)")
    return None


if __name__ == "__main__":
    main()
