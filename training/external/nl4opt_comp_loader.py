"""
Load NL4Opt competition (already downloaded by pipeline) and build eval JSONL.
Uses simple name-based mapping to the catalog.
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


def _load_nl4opt_raw() -> list[dict]:
    raw_dir = ROOT / "data" / "raw" / "nl4opt"
    items: list[dict] = []
    for split in ("train", "dev", "test"):
        path = raw_dir / f"{split}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            items.extend(json.load(f))
    return items


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build NL4Opt competition eval JSONL")
    p.add_argument("--eval-out", type=Path, default=None)
    p.add_argument("--coverage-out", type=Path, default=None)
    args = p.parse_args()

    catalog = _load_catalog()
    name_idx = _build_name_index(catalog)
    items = _load_nl4opt_raw()

    eval_out = args.eval_out or ROOT / "data" / "processed" / "nl4opt_comp_eval.jsonl"
    cov_out = args.coverage_out or ROOT / "results" / "nl4opt_comp_coverage.json"
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    cov_out.parent.mkdir(parents=True, exist_ok=True)

    total = len(items)
    mapped = 0
    with open(eval_out, "w", encoding="utf-8") as f:
        for ex in items:
            q = (ex.get("natural_language") or ex.get("problem") or "").strip()
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
    print(f"NL4Opt_comp: total={total}, mapped={mapped}, coverage={cov['coverage']:.2%}")
    print(f"Eval written to {eval_out}")
    print(f"Coverage written to {cov_out}")


if __name__ == "__main__":
    main()

