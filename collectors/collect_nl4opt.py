"""
Collect NL4Opt dataset: natural language + LP formulations.
Source: https://github.com/nl4opt/nl4opt-competition (generation_data: train, dev, test .jsonl)
Public, free to use. Converts each record to the project's unified problem schema.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import urlopen

# Base URL for raw NL4Opt generation_data
NL4OPT_BASE = "https://raw.githubusercontent.com/nl4opt/nl4opt-competition/main/generation_data"
FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _slug(s: str, max_len: int = 40) -> str:
    """Short alphanumeric slug from text for use in id/name."""
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")[:max_len]
    return s.lower() or "lp"


def _obj_expression(obj: dict) -> str:
    """Build objective expression string from NL4Opt obj_declaration."""
    terms = obj.get("terms") or {}
    if not terms:
        return obj.get("name", "objective")
    parts = []
    for var, coef in terms.items():
        parts.append(f"{coef}*{var}")
    return " + ".join(parts)


def _constraint_expression(c: dict) -> tuple[str, str]:
    """Build (expression, description) from NL4Opt const_declarations entry."""
    ctype = c.get("type", "")
    direction = c.get("direction", "")
    limit = c.get("limit", "")
    op = c.get("operator", "")
    if ctype == "sum":
        return (
            f"sum of variables ≥ {limit}" if "GREATER" in str(op) else f"sum of variables ≤ {limit}",
            f"Total {direction} {limit}",
        )
    if ctype == "lowerbound":
        var = c.get("var", "x")
        return (
            f"{var} ≥ {limit}" if "GREATER" in str(op) else f"{var} ≤ {limit}",
            f"{var} {direction} {limit}",
        )
    if ctype == "linear":
        terms = c.get("terms") or {}
        expr = " + ".join(f"{coef}*{v}" for v, coef in terms.items())
        return (
            f"{expr} ≤ {limit}" if "LESS" in str(op) else f"{expr} ≥ {limit}",
            f"Linear constraint {direction} {limit}",
        )
    if ctype == "xby":
        x_var = c.get("x_var", "x")
        y_var = c.get("y_var", "y")
        param = c.get("param", "")
        return (
            f"{x_var} ≥ {param} * {y_var}",
            f"{x_var} {direction} {param} of {y_var}",
        )
    if ctype == "ratio":
        var = c.get("var", "x")
        return (
            f"{var} ≥ {limit}% of total",
            f"{var} {direction} {limit}%",
        )
    return ("", "Constraint")


def nl4opt_record_to_problem(record: dict, record_id: str, source_tag: str = "nl4opt") -> dict:
    """Convert one NL4Opt JSON record to unified problem schema."""
    doc = record.get("document", "").strip()
    if not doc:
        raise ValueError("Empty document")
    vars_list = record.get("vars") or []
    obj = record.get("obj_declaration") or {}
    consts = record.get("const_declarations") or []

    # Short name from first sentence or slug of start of document
    first_sentence = doc.split(".")[0].strip()[:60]
    name = first_sentence + ("..." if len(first_sentence) >= 60 else "")
    slug = _slug(doc[:80])
    uid = f"nl4opt_{source_tag}_{record_id}_{slug}"[:80]

    variables = []
    for v in vars_list:
        variables.append({
            "symbol": v,
            "description": "Decision variable",
            "domain": "continuous (LP)",
        })

    sense = (obj.get("direction") or "minimize").lower()
    if "max" in sense:
        sense = "maximize"
    else:
        sense = "minimize"
    objective = {
        "sense": sense,
        "expression": _obj_expression(obj),
    }

    constraints = []
    for c in consts:
        expr, desc = _constraint_expression(c)
        if expr:
            constraints.append({"expression": expr, "description": desc})

    return {
        "id": uid,
        "name": name,
        "aliases": [],
        "description": doc,
        "formulation": {
            "variables": variables,
            "objective": objective,
            "constraints": constraints,
        },
        "formulation_latex": "",
        "complexity": "P (LP)",
        "source": "NL4Opt",
    }


def fetch_nl4opt_jsonl(url: str) -> list[dict]:
    """Download a single .jsonl file and return list of records (one per line; each line is one JSON object)."""
    with urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    records = []
    for line in text.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        # Each line is a single object with one key (id) and value = record
        for key, value in data.items():
            if isinstance(value, dict) and value.get("document"):
                records.append((key, value))
    return records


def collect_nl4opt(out_dir: Path | None = None, max_per_file: int | None = None) -> list[dict]:
    """
    Download NL4Opt train/dev/test and convert to unified schema.
    out_dir: if set, write each file's parsed JSON here.
    max_per_file: if set, only take first N records per file (for quick tests).
    Returns list of problem dicts in project schema.
    """
    out_dir = out_dir or _project_root() / "data" / "raw" / "nl4opt"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_problems = []
    seen_docs = set()

    for filename in FILES:
        url = f"{NL4OPT_BASE}/{filename}"
        tag = filename.replace(".jsonl", "")
        try:
            records = fetch_nl4opt_jsonl(url)
        except Exception as e:
            print(f"Warning: could not fetch {url}: {e}")
            continue
        if max_per_file is not None:
            records = records[: max_per_file]
        file_problems = []
        for record_id, record in records:
            try:
                problem = nl4opt_record_to_problem(record, record_id, source_tag=tag)
                # Deduplicate by description (many NL4Opt problems are similar)
                doc_key = problem["description"][:200]
                if doc_key in seen_docs:
                    continue
                seen_docs.add(doc_key)
                file_problems.append(problem)
                all_problems.append(problem)
            except Exception as e:
                continue
        out_path = out_dir / filename.replace(".jsonl", ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(file_problems, f, indent=2, ensure_ascii=False)
        print(f"NL4Opt {filename}: {len(file_problems)} problems -> {out_path}")

    return all_problems


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Collect NL4Opt dataset into unified schema")
    p.add_argument("--max-per-file", type=int, default=None, help="Max records per file (default: all)")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for raw JSON")
    args = p.parse_args()
    problems = collect_nl4opt(out_dir=args.out_dir, max_per_file=args.max_per_file)
    print(f"Total NL4Opt problems: {len(problems)}")


if __name__ == "__main__":
    main()
