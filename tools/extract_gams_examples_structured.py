#!/usr/bin/env python3
"""
Extract structured metadata from GAMSPy example .py files for audit and
potential use in the learning pipeline (slot/parameter vocabulary, numeric constants).

Reads Python source only; does not execute models.
Output: artifacts/gams_example_audit/gams_examples_structured.jsonl and .csv.

Heuristic: docstring tags (MODELTYPE, KEYWORDS, GAMSSOURCE), Parameter/Set/Variable/Equation
names and descriptions, objective sense, and numeric literals from records= and arrays.
Labels are NOT gold; use for catalog and weak-supervision experiments only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data_private" / "gams_models" / "raw" / "gamspy-examples" / "models"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "gams_example_audit"


def _docstring_meta(text: str) -> dict:
    """Extract ## KEY: value from docstring."""
    out = {}
    for m in re.finditer(r"##\s*([A-Za-z_]+)\s*:\s*(.+)", text):
        out[m.group(1).strip().lower()] = m.group(2).strip()
    # First line of description (after optional ## lines)
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("##")]
    if lines:
        out["description_first_line"] = lines[0][:200]
    return out


def _symbols(text: str, kind: str) -> list[dict]:
    """Extract name and description for Parameter/Set/Variable/Equation declarations.
    kind in ('Parameter', 'Set', 'Variable', 'Equation'). Heuristic: name= and description=."""
    pattern = re.compile(
        r"\b" + kind + r"\s*\(\s*[^)]*?"
        r'name\s*=\s*["\']([^"\']+)["\']'
        r"(?:(?!\b" + kind + r"\s*\()[\s\S])*?"
        r'(?:description\s*=\s*["\']([^"\']*)["\'])?',
        re.IGNORECASE | re.DOTALL,
    )
    # Also short form: Parameter(m, "p", description="...", domain=i)
    short = re.compile(
        r"\b" + kind + r"\s*\(\s*[^,]+,\s*[\"']([^\"']+)[\"']"
        r"(?:(?!\b" + kind + r"\s*\()[\s\S])*?"
        r'(?:description\s*=\s*["\']([^"\']*)["\'])?',
        re.IGNORECASE | re.DOTALL,
    )
    found = []
    for regex in (pattern, short):
        for m in regex.finditer(text):
            desc = m.group(2) if m.lastindex >= 2 and m.group(2) else ""
            found.append({"name": m.group(1), "description": desc[:300]})
    # Dedupe by name
    by_name = {x["name"]: x for x in found}
    return list(by_name.values())


def _objective_sense(text: str) -> str | None:
    """Infer MIN/MAX from sense=Sense.MIN, sense='min', etc."""
    m = re.search(r"sense\s*=\s*(?:Sense\.)?(MIN|MAX|min|max)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'Sense\.(MIN|MAX)', text)
    if m:
        return m.group(1).upper()
    return None


def _numeric_literals(text: str, max_sample: int = 30) -> list[float]:
    """Heuristic: numbers in records=, np.array([...]), and [ [ n, n ], ... ]. No execution."""
    nums = []
    # records=[ [...], [...] ] or records=[[1,2],[3,4]]
    for m in re.finditer(r"records\s*=\s*(?:np\.array\s*)?(\[[\s\S]*?\](?:\s*,\s*\[[\s\S]*?\])*)", text):
        chunk = m.group(1)
        for n in re.findall(r"-?\d+\.?\d*", chunk):
            try:
                nums.append(float(n))
            except ValueError:
                pass
    # Single scalar in records= e.g. records=350 or records=[["seattle",350]]
    for m in re.finditer(r"records\s*=\s*(\d+\.?\d*)", text):
        try:
            nums.append(float(m.group(1)))
        except ValueError:
            pass
    # Literal lists of numbers
    for m in re.finditer(r"\[[\s\d.,\-]+\]", text):
        for n in re.findall(r"-?\d+\.?\d*", m.group(0)):
            try:
                nums.append(float(n))
            except ValueError:
                pass
    # Dedupe order-preserving, then cap
    seen = set()
    out = []
    for x in nums:
        if x not in seen and len(out) < max_sample:
            seen.add(x)
            out.append(x)
    return out


def _model_type_from_doc(meta: dict) -> str | None:
    """MODELTYPE from docstring meta."""
    return meta.get("modeltype") or meta.get("model_type")


def process_file(path: Path, rel_root: Path) -> dict | None:
    """Parse one .py file; return structured dict or None on read error."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    # Docstring: first triple-quoted block
    doc = ""
    for m in re.finditer(r'"""(.*?)"""', text, re.DOTALL):
        doc = m.group(1).strip()
        break
    meta = _docstring_meta(doc)
    model_type = _model_type_from_doc(meta)
    sense = _objective_sense(text)
    params = _symbols(text, "Parameter")
    sets = _symbols(text, "Set")
    variables = _symbols(text, "Variable")
    equations = _symbols(text, "Equation")
    numerics = _numeric_literals(text)
    name = path.stem
    rel_path = path.relative_to(rel_root) if rel_root in path.parents else path.name
    return {
        "model_name": name,
        "source_path": str(rel_path),
        "model_type": model_type or "unknown",
        "objective_direction": sense,
        "keywords": meta.get("keywords", ""),
        "gams_source_url": meta.get("gamssource", ""),
        "description_snippet": meta.get("description_first_line", ""),
        "parameter_names": [p["name"] for p in params],
        "parameters": params,
        "set_names": [s["name"] for s in sets],
        "sets": sets,
        "variable_names": [v["name"] for v in variables],
        "variables": variables,
        "equation_names": [e["name"] for e in equations],
        "equations": equations,
        "numeric_constants_sample": numerics,
        "num_parameters": len(params),
        "num_sets": len(sets),
        "num_variables": len(variables),
        "num_equations": len(equations),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured metadata from GAMSPy example .py files")
    ap.add_argument("--input_dir", type=Path, default=DEFAULT_RAW, help="Root of gamspy-examples/models")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max_examples", type=int, default=None, help="Cap number of files to process (default: all)")
    args = ap.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}", flush=True)
        print("No artifact written. Clone gamspy-examples to data_private or set --input_dir.", flush=True)
        return
    py_files = sorted(input_dir.rglob("*.py"))
    if args.max_examples:
        py_files = py_files[: args.max_examples]
    rows = []
    for path in py_files:
        if path.name.startswith("_"):
            continue
        rec = process_file(path, input_dir)
        if rec:
            rows.append(rec)
    # JSONL
    jsonl_path = out_dir / "gams_examples_structured.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {jsonl_path} ({len(rows)} examples)")
    # CSV summary
    csv_path = out_dir / "gams_examples_structured.csv"
    fieldnames = [
        "model_name", "source_path", "model_type", "objective_direction",
        "num_parameters", "num_sets", "num_variables", "num_equations",
        "parameter_names", "numeric_constants_sample", "description_snippet",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in fieldnames}
            row["parameter_names"] = ";".join(r.get("parameter_names", []))
            row["numeric_constants_sample"] = ";".join(str(x) for x in (r.get("numeric_constants_sample") or [])[:15])
            w.writerow(row)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
