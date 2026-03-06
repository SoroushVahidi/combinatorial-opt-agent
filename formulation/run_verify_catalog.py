"""
Run schema and formulation verifiers on the full catalog; write results/validation_catalog.jsonl
and print a summary table: pass rate overall, by source, and top-10 most common errors.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent


def load_catalog(path: Path | None = None) -> list[dict]:
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from retrieval.search import _load_catalog as load_
    return load_()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Validate full catalog and write results")
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    args = p.parse_args()

    from formulation.verify import verify_problem_schema, verify_formulation_structure

    catalog = load_catalog(args.catalog)
    results_dir = Path(args.results_dir or ROOT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "validation_catalog.jsonl"

    by_source: dict[str, list[dict]] = defaultdict(list)
    error_counter: Counter = Counter()
    n_pass = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for problem in catalog:
            pid = problem.get("id") or ""
            source = problem.get("source") or "unknown"
            schema_errors = verify_problem_schema(problem)
            formulation_errors = verify_formulation_structure(problem)
            errors = schema_errors + formulation_errors
            if not errors:
                n_pass += 1
            for e in errors:
                error_counter[e] += 1
            rec = {
                "id": pid,
                "source": source,
                "num_errors": len(errors),
                "error_types": list(errors),
            }
            by_source[source].append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(catalog)
    pass_rate = n_pass / n if n else 0.0

    print()
    print("=" * 60)
    print("Validation summary (catalog)")
    print("=" * 60)
    print(f"  Total problems:  {n}")
    print(f"  Pass (no errors): {n_pass}")
    print(f"  Pass rate (overall): {pass_rate:.2%}")
    print()
    print("  Pass rate by source:")
    for src in sorted(by_source.keys()):
        recs = by_source[src]
        src_pass = sum(1 for r in recs if r["num_errors"] == 0)
        rate = src_pass / len(recs) if recs else 0.0
        print(f"    {src}: {rate:.2%}  ({src_pass}/{len(recs)})")
    print()
    print("  Top-10 most common errors:")
    for err, count in error_counter.most_common(10):
        print(f"    {count:5d}  {err}")
    print("=" * 60)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
