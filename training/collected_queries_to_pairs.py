"""
Convert logged user queries (data/collected_queries/user_queries.jsonl) into
(query, passage) training pairs using the top result's problem from the catalog.
Run on Wulver after collecting traffic; then merge with synthetic pairs or train on them.

Usage:
  python -m training.collected_queries_to_pairs [--output path] [--min-score 0.0]
"""
from __future__ import annotations

import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def searchable_text(problem: dict) -> str:
    """Same as retrieval.search._searchable_text."""
    parts = [problem.get("name", "")]
    parts.extend(problem.get("aliases") or [])
    parts.append(problem.get("description", ""))
    return " ".join(p for p in parts if p)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Convert collected user queries to (query, passage) pairs")
    p.add_argument("--input", type=Path, default=None, help="Input JSONL (default: data/collected_queries/user_queries.jsonl)")
    p.add_argument("--output", type=Path, default=None, help="Output JSONL (default: data/processed/collected_training_pairs.jsonl)")
    p.add_argument("--min-score", type=float, default=0.0, help="Min relevance score for top result to include (default 0)")
    args = p.parse_args()

    root = _project_root()
    input_path = args.input or root / "data" / "collected_queries" / "user_queries.jsonl"
    output_path = args.output or root / "data" / "processed" / "collected_training_pairs.jsonl"

    if not input_path.exists():
        print(f"No collected queries at {input_path}. Run the app and use it to collect some first.")
        return

    # Load catalog and index by name
    catalog_path = root / "data" / "processed" / "all_problems_extended.json"
    if not catalog_path.exists():
        catalog_path = root / "data" / "processed" / "all_problems.json"
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    by_name = {p.get("name", ""): p for p in catalog if p.get("name")}

    pairs = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            query = (rec.get("query") or "").strip()
            results = rec.get("results") or []
            if not query or not results:
                continue
            top = results[0]
            score = top.get("score", 0.0)
            if score < args.min_score:
                continue
            name = top.get("name", "")
            problem = by_name.get(name)
            if not problem:
                continue
            passage = searchable_text(problem)
            if not passage:
                continue
            pairs.append((query, passage))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for q, passage in pairs:
            f.write(json.dumps({"query": q, "passage": passage}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs from {input_path} to {output_path}")


if __name__ == "__main__":
    main()
