"""
Generate synthetic (query, passage) training pairs from the problem catalog.
For each problem we create multiple query phrasings that should retrieve that problem's passage.
Passage = name + aliases + description (same as retrieval searchable text).
Target: 100 instances per problem so the model learns to recognize the problem and
retrieval can return it (then the app shows description, ILP, etc.).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def searchable_text(problem: dict) -> str:
    """Same as retrieval.search._searchable_text: one string for embedding."""
    parts = [problem.get("name", "")]
    parts.extend(problem.get("aliases") or [])
    parts.append(problem.get("description", ""))
    return " ".join(p for p in parts if p)


# Many templates to reach ~100 queries per problem. {name}, {alias}, {desc} are placeholders.
QUERY_TEMPLATES = [
    "{text}",
    "What is {text}?",
    "Describe {text}.",
    "Formulation for {text}.",
    "ILP for {text}.",
    "Integer program for {text}.",
    "Linear program for {text}.",
    "How do I formulate {text}?",
    "Give me the formulation of {text}.",
    "I need an LP or IP for {text}.",
    "Optimization problem: {text}.",
    "Mathematical model for {text}.",
    "Variables and constraints for {text}.",
    "What is the optimization problem: {text}?",
    "Find the integer program for {text}.",
    "Write the ILP for {text}.",
    "Model the following: {text}.",
    "Combinatorial optimization: {text}.",
    "Problem: {text}.",
    "I want to {text}.",
    "Need to {text}.",
    "Looking for formulation of {text}.",
    "Explain {text}.",
    "Define {text}.",
    "Can you give the IP for {text}?",
    "MIP formulation for {text}.",
    "Mixed integer program for {text}.",
    "Minimize cost subject to: {text}.",
    "Maximize profit with constraints: {text}.",
    "I have a problem: {text}.",
    "Formulate as IP: {text}.",
    "Need variables and constraints for {text}.",
    "Which optimization problem fits: {text}?",
    "Classic problem: {text}.",
]


def _sentences_from_description(desc: str, max_len: int = 200) -> list[str]:
    """Split description into sentence-level chunks for more query variety."""
    if not desc or len(desc) < 20:
        return []
    # Simple split on . ! ?
    parts = re.split(r"[.!?]+", desc)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 15 and len(p) <= max_len:
            out.append(p + ".")
        elif len(p) > max_len:
            # One chunk from start
            out.append(p[:max_len].rsplit(" ", 1)[0] + "...")
    return out[:10]  # at most 10 sentence-based queries


def generate_queries_for_problem(
    problem: dict, rng: random.Random, target_per_problem: int = 100
) -> list[str]:
    """Generate up to target_per_problem synthetic query strings that should match this problem."""
    name = problem.get("name", "")
    desc = problem.get("description", "")
    aliases = list(problem.get("aliases") or [])[:10]
    queries = []
    seen = set()

    def add(q: str) -> None:
        q = (q or "").strip()
        if q and len(q) >= 3 and q not in seen:
            seen.add(q)
            queries.append(q)

    # Full and shortened description
    if desc:
        add(desc)
        short = desc.split(".")[0].strip() + "." if "." in desc else desc[:120]
        if short != desc.strip():
            add(short)
        if len(desc) > 150:
            add(desc[:150] + "...")
        for sent in _sentences_from_description(desc):
            add(sent)

    # Name with many templates
    if name:
        add(name)
        for t in QUERY_TEMPLATES:
            if "{text}" in t and len(queries) < target_per_problem * 2:  # allow over then dedupe
                add(t.format(text=name))

    # Aliases with templates
    for a in aliases:
        if not a:
            continue
        add(a)
        for t in QUERY_TEMPLATES[:15]:  # subset to avoid explosion
            if "{text}" in t:
                add(t.format(text=a))

    # Description-based with templates (use short desc in template)
    if desc:
        snippet = desc.split(".")[0].strip()[:80] + ("..." if len(desc) > 80 else "")
        for t in QUERY_TEMPLATES[:12]:
            if "{text}" in t:
                add(t.format(text=snippet))

    # Shuffle and cap to target_per_problem
    rng.shuffle(queries)
    return list(dict.fromkeys(queries))[:target_per_problem]  # preserve order, dedupe, cap


def generate_all_samples(
    catalog_path: Path | None = None,
    seed: int = 42,
    instances_per_problem: int = 100,
) -> list[tuple[str, str]]:
    """Return list of (query, passage) pairs for training."""
    root = _project_root()
    data_dir = root / "data" / "processed"
    if catalog_path is None:
        extended = data_dir / "all_problems_extended.json"
        base = data_dir / "all_problems.json"
        catalog_path = extended if extended.exists() else base
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    rng = random.Random(seed)
    pairs = []
    for problem in catalog:
        passage = searchable_text(problem)
        if not passage:
            continue
        queries = generate_queries_for_problem(
            problem, rng, target_per_problem=instances_per_problem
        )
        for q in queries:
            pairs.append((q, passage))
    return pairs


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Generate synthetic (query, passage) samples from catalog"
    )
    p.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    p.add_argument("--catalog", type=Path, default=None, help="Catalog JSON (default: all_problems_extended.json if present else all_problems.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--instances-per-problem",
        type=int,
        default=100,
        help="Number of (query, passage) pairs to generate per problem (default 100)",
    )
    args = p.parse_args()
    pairs = generate_all_samples(
        catalog_path=args.catalog,
        seed=args.seed,
        instances_per_problem=args.instances_per_problem,
    )
    out = args.output or _project_root() / "data" / "processed" / "training_pairs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for q, passage in pairs:
            f.write(json.dumps({"query": q, "passage": passage}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {out} ({args.instances_per_problem} per problem target)")


if __name__ == "__main__":
    main()
