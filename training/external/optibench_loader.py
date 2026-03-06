"""Load OptiBench evaluation NL queries and map to catalog.
Requires a local clone of ReSocratic (OptiBench) repo.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_catalog() -> list[dict]:
    from retrieval.search import _load_catalog as load_
    return load_()


def _normalize_name(name: str) -> str:
    name = (name or "").strip().lower()
    if name.endswith(" problem"):
        name = name[:-8].strip()
    return name


def _build_name_index(catalog: list[dict]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for p in catalog:
        pid = p.get("id")
        if not pid:
            continue
        base = _normalize_name(p.get("name") or "")
        if base:
            idx.setdefault(base, pid)
    return idx


def load_optibench(root: Path) -> list[dict]:
    """Load OptiBench NL queries from a local ReSocratic clone.
    This is a heuristic loader; it expects a JSONL file containing fields like
    'problem_name' and 'nl_description' or 'question'.
    """
    if not root or not root.exists():
        print("OptiBench root not provided or does not exist; skipping.")
        return []
    # Heuristic: look for a file named optibench.jsonl under data/
    candidates = list(root.glob("**/optibench*.jsonl"))
    if not candidates:
        print(f"No optibench*.jsonl found under {root}; skipping.")
        return []
    path = candidates[0]
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = None
            for key in ("nl_description", "question", "text", "natural_language"):
                if key in obj and obj[key]:
                    text = obj[key]
                    break
            name = None
            for key in ("problem_name", "name", "title"):
                if key in obj and obj[key]:
                    name = obj[key]
                    break
            if not text or not name:
                continue
            items.append({"query": str(text), "problem_name": str(name)})
    return items


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build OptiBench eval JSONL")
    p.add_argument("--root", type=Path, default=None, help="Path to ReSocratic root (containing OptiBench JSONL)")
    p.add_argument("--eval-out", type=Path, default=None)
    p.add_argument("--coverage-out", type=Path, default=None)
    args = p.parse_args()

    catalog = _load_catalog()
    name_idx = _build_name_index(catalog)

    items = load_optibench(args.root) if args.root else []

    eval_out = args.eval_out or ROOT / "data" / "processed" / "optibench_eval.jsonl"
    cov_out = args.coverage_out or ROOT / "results" / "optibench_coverage.json"
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    cov_out.parent.mkdir(parents=True, exist_ok=True)

    total = len(items)
    mapped = 0
    with open(eval_out, "w", encoding="utf-8") as f:
        for ex in items:
            q = (ex.get("query") or "").strip()
            name = _normalize_name(ex.get("problem_name") or "")
            if not q or not name:
                continue
            pid = name_idx.get(name)
            if not pid:
                continue
            mapped += 1
            f.write(json.dumps({"query": q, "problem_id": pid}, ensure_ascii=False) + "\n")

    cov = {"total": total, "mapped": mapped, "coverage": (mapped / total) if total else 0.0}
    with open(cov_out, "w", encoding="utf-8") as f:
        json.dump(cov, f, indent=2)
    print(f"OptiBench: total={total}, mapped={mapped}, coverage={cov['coverage']:.2%}")
    print(f"Eval written to {eval_out}")
    print(f"Coverage written to {cov_out}")


if __name__ == "__main__":
    main()

