"""
Build NL4Opt-type evaluation sets from local NL4Opt test split with
canonical problem type labels where possible.

Writes:
  - data/processed/nl4opt_type_eval_test.jsonl
  - data/processed/nl4opt_type_eval_test_masked.jsonl
  - results/nl4opt_type_coverage.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_nl4opt_test() -> list[dict]:
    path = ROOT / "data" / "raw" / "nl4opt" / "test.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    items = _load_nl4opt_test()
    eval_path = ROOT / "data" / "processed" / "nl4opt_type_eval_test.jsonl"
    masked_path = ROOT / "data" / "processed" / "nl4opt_type_eval_test_masked.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    cov_path = ROOT / "results" / "nl4opt_type_coverage.json"
    cov_path.parent.mkdir(parents=True, exist_ok=True)

    from training.mask_eval_queries import mask_query

    total = 0
    labeled = 0

    def map_to_type(desc: str) -> str | None:
        """Heuristic mapping from NL4Opt description to canonical problem type id."""
        text = (desc or "").lower()
        # Knapsack
        if "knapsack" in text:
            return "knapsack_01"
        # Set cover / covering
        if "set cover" in text or "set covering" in text:
            return "set_cover"
        # Vertex cover
        if "vertex cover" in text:
            return "vertex_cover"
        # Facility location
        if "facility location" in text or "warehouse location" in text:
            return "facility_location_ufl"
        # Traveling salesman
        if "traveling salesman" in text or "travelling salesman" in text or "tsp" in text:
            return "tsp"
        # Shortest path
        if "shortest path" in text:
            return "shortest_path"
        # Assignment
        if "assignment problem" in text or "assign each" in text:
            return "assignment_problem"
        return None
    with open(eval_path, "w", encoding="utf-8") as fout_eval, open(
        masked_path, "w", encoding="utf-8"
    ) as fout_masked:
        for obj in items:
            desc = (obj.get("description") or "").strip()
            if not desc:
                continue
            total += 1
            type_id = map_to_type(desc)
            if not type_id:
                continue  # unlabeled, skip
            labeled += 1
            # canonical label for evaluation
            fout_eval.write(
                json.dumps({"query": desc, "problem_id": type_id}, ensure_ascii=False)
                + "\n"
            )
            # mask query text only
            problem = {"name": "", "aliases": []}
            masked = mask_query(desc, problem)
            fout_masked.write(
                json.dumps({"query": masked, "problem_id": type_id}, ensure_ascii=False)
                + "\n"
            )

    cov = {
        "total": total,
        "labeled": labeled,
        "coverage": (labeled / total) if total else 0.0,
    }
    with open(cov_path, "w", encoding="utf-8") as f_cov:
        json.dump(cov, f_cov, indent=2)

    print(
        f"NL4Opt-type eval: total={total}, labeled={labeled}, "
        f"coverage={cov['coverage']:.2%}, wrote {eval_path} and {masked_path}"
    )


if __name__ == "__main__":
    main()

