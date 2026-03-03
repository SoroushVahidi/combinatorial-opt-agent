"""
Collect OptMATH benchmark: natural language optimization problem descriptions.
Source: https://github.com/optsuite/OptMATH (benchmark/OptMATH_Bench.json)
Uses en_question as description; no structured LP/IP in benchmark (answer is numeric only).
Adds diversity: job shop, aircraft landing, TSP, facility location, etc.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import urlopen

OPTMATH_BENCH_URL = (
    "https://raw.githubusercontent.com/optsuite/OptMATH/main/benchmark/OptMATH_Bench.json"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _slug(s: str, max_len: int = 50) -> str:
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")[:max_len]
    return s.lower() or "opt"


def _first_sentence(text: str, max_chars: int = 120) -> str:
    """Use as problem name/summary."""
    text = (text or "").strip()
    for sep in (". ", "\n\n"):
        if sep in text:
            part = text.split(sep)[0].strip()
            if len(part) >= 10:
                return part[:max_chars] + ("..." if len(part) > max_chars else "")
    return text[:max_chars] + ("..." if len(text) > max_chars else "") if text else "OptMATH instance"


def collect_optmath(max_items: int | None = None) -> list[dict]:
    """Download OptMATH_Bench.json and convert to project problem schema."""
    root = _project_root()
    raw_dir = root / "data" / "raw" / "optmath"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with urlopen(OPTMATH_BENCH_URL, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    problems = []
    for i, item in enumerate(data):
        if max_items is not None and i >= max_items:
            break
        en_q = (item.get("en_question") or "").strip()
        if not en_q or len(en_q) < 50:
            continue
        pid = item.get("id", i)
        name = _first_sentence(en_q)
        slug = _slug(name) or f"optmath_{pid}"
        problem_id = f"optmath_bench_{pid}"
        problems.append({
            "id": problem_id,
            "name": name,
            "aliases": [f"OptMATH instance {pid}", slug],
            "description": en_q,
            "formulation": {
                "variables": [],
                "objective": {"sense": "", "expression": "See description (benchmark has no structured formulation)."},
                "constraints": [],
            },
            "formulation_latex": "",
            "complexity": "unknown",
            "source": "optmath_bench",
        })

    out_file = raw_dir / "bench.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(problems)} OptMATH benchmark problems to {out_file}")
    return problems


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Collect OptMATH benchmark into data/raw/optmath/")
    p.add_argument("--max-items", type=int, default=None, help="Cap number of problems (default: all)")
    args = p.parse_args()
    collect_optmath(max_items=args.max_items)


if __name__ == "__main__":
    main()
