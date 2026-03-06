"""Create name-masked, number-stripped eval set for harder lexical evaluation."""
from __future__ import annotations

import json
import re
from pathlib import Path

from retrieval.search import _load_catalog

ROOT = Path(__file__).resolve().parent.parent


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _acronyms(name: str) -> list[str]:
    words = [w for w in re.split(r"\W+", name) if w]
    if not words:
        return []
    ac = "".join(w[0] for w in words if w[0].isalpha())
    # drop trailing 'p' if it stands for Problem
    if ac.lower().endswith("p") and len(ac) > 2:
        return [ac, ac[:-1]]
    return [ac]


def mask_query(query: str, problem: dict) -> str:
    text = query
    name = problem.get("name") or ""
    aliases = problem.get("aliases") or []
    tokens = [name] + aliases
    for t in list(tokens):
        if not t:
            continue
        # mask acronym variants
        for ac in _acronyms(t):
            if ac:
                tokens.append(ac)
    for t in tokens:
        if not t:
            continue
        pattern = r"\b" + re.escape(t) + r"\b"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # drop numbers
    text = re.sub(r"\d+(?:\.\d+)?", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text or query  # fallback to original if we deleted everything


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Create name-masked, number-stripped eval JSONL")
    p.add_argument("--eval-file", type=Path, default=None,
                   help="Input eval JSONL (default: data/processed/eval_test.jsonl)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSONL (default: data/processed/eval_test_masked.jsonl)")
    args = p.parse_args()

    in_path = args.eval_file or ROOT / "data" / "processed" / "eval_test.jsonl"
    out_path = args.output or ROOT / "data" / "processed" / "eval_test_masked.jsonl"

    catalog = _load_catalog()
    by_id = {p.get("id"): p for p in catalog if p.get("id")}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = (obj.get("query") or "").strip()
            pid = obj.get("problem_id")
            if not q or not pid:
                continue
            problem = by_id.get(pid)
            if not problem:
                continue
            n_in += 1
            masked = mask_query(q, problem)
            fout.write(json.dumps({"query": masked, "problem_id": pid}, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Masked {n_out}/{n_in} eval queries -> {out_path}")


if __name__ == "__main__":
    main()

