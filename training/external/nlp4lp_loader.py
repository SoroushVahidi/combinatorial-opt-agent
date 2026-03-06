"""Load NLP4LP (udell-lab/NLP4LP) from HuggingFace and build eval JSONL.
Each line: {"query": <nl_text>, "problem_id": <catalog id>} for mapped subset.
Coverage summary is written to results/nlp4lp_coverage.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

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


def load_nlp4lp(split: str = "test") -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"datasets not available: {e}")
        return []
    try:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        ds = load_dataset("udell-lab/NLP4LP", split=split, token=token)
    except Exception as e:
        print("Could not load udell-lab/NLP4LP. If this dataset is gated,")
        print("request access on HuggingFace and set HF_TOKEN in your shell, then re-run:")
        print("  export HF_TOKEN=...  # do NOT hard-code it in code")
        print(f"Loader error was: {e}")
        return []
    items = []
    for ex in ds:
        # For this dataset, 'description' holds the NL text; there is no explicit problem_name.
        text = (ex.get("description") or "").strip()
        if not text:
            continue
        # There is no canonical problem_name field; use a dummy label so we treat everything as one pool.
        # Downstream mapping to our catalog may still be empty, but at least total reflects dataset size.
        items.append({"query": str(text), "problem_name": "nlp4lp_unknown"})
    return items


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build NLP4LP eval JSONL")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--eval-out", type=Path, default=None)
    p.add_argument("--coverage-out", type=Path, default=None)
    args = p.parse_args()

    catalog = _load_catalog()
    name_idx = _build_name_index(catalog)
    items = load_nlp4lp(split=args.split)

    eval_out = args.eval_out or ROOT / "data" / "processed" / "nlp4lp_eval.jsonl"
    cov_out = args.coverage_out or ROOT / "results" / "nlp4lp_coverage.json"
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
    print(f"NLP4LP: total={total}, mapped={mapped}, coverage={cov['coverage']:.2%}")
    print(f"Eval written to {eval_out}")
    print(f"Coverage written to {cov_out}")


if __name__ == "__main__":
    main()

