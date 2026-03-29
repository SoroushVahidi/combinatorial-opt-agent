"""Collector for the NL4Opt dataset.

Supports two collection modes:
  1. HTTP download (default): fetches train/dev/test .jsonl files directly from
     https://github.com/nl4opt/nl4opt-competition via raw.githubusercontent.com.
     No local git checkout required; suitable for CI and lightweight runs.
  2. Git clone (fallback): clones the repository into data/raw/nl4opt/ and
     parses JSON split files from disk.  Activated automatically when the raw
     directory already contains checked-out files, or explicitly via
     clone_or_update().

All output conforms to the project's unified problem schema
(schema/problem_template.json) and is written to data/processed/nl4opt.json.

Usage (standalone):
    python collectors/collect_nl4opt.py
    python collectors/collect_nl4opt.py --max-per-file 100
    python collectors/collect_nl4opt.py --out-dir data/raw/nl4opt

Usage (callable API from pipeline):
    from collectors.collect_nl4opt import collect_nl4opt
    problems = collect_nl4opt(max_per_file=200)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    import git as _git  # optional; only needed for clone_or_update()
    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

# HTTP download (primary approach — no local git required)
NL4OPT_BASE = (
    "https://raw.githubusercontent.com/nl4opt/nl4opt-competition"
    "/main/generation_data"
)
FILES = ("train.jsonl", "dev.jsonl", "test.jsonl")

# Git clone (alternative approach)
REPO_URL = "https://github.com/nl4opt/nl4opt-competition"
RAW_DIR = Path("data/raw/nl4opt")
OUTPUT_FILE = Path("data/processed/nl4opt.json")


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _slug(s: str, max_len: int = 40) -> str:
    """Short alphanumeric slug from text for use in id/name fields."""
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "_", s).strip("_")[:max_len]
    return s.lower() or "lp"


def _entity_type_to_var_type(entity_type: str) -> str:
    """Map NL4Opt entity types to unified schema variable types."""
    mapping = {
        "VAR": "continuous",
        "CONST": "continuous",
        "PARAM": "continuous",
        "OBJ": "continuous",
    }
    return mapping.get(entity_type.upper(), "continuous")


def _obj_expression(obj: dict) -> str:
    """Build objective expression string from an NL4Opt obj_declaration dict."""
    terms = obj.get("terms") or {}
    if not terms:
        return obj.get("name", "objective")
    return " + ".join(f"{coef}*{var}" for var, coef in terms.items())


def _constraint_expression(c: dict) -> tuple[str, str]:
    """Return (expression, description) from an NL4Opt const_declarations entry."""
    ctype = c.get("type", "")
    direction = c.get("direction", "")
    limit = c.get("limit", "")
    op = str(c.get("operator", ""))

    if ctype == "sum":
        expr = (
            f"sum of variables >= {limit}"
            if "GREATER" in op
            else f"sum of variables <= {limit}"
        )
        return expr, f"Total {direction} {limit}"

    if ctype == "lowerbound":
        var = c.get("var", "x")
        expr = f"{var} >= {limit}" if "GREATER" in op else f"{var} <= {limit}"
        return expr, f"{var} {direction} {limit}"

    if ctype == "linear":
        terms = c.get("terms") or {}
        expr_str = " + ".join(f"{coef}*{v}" for v, coef in terms.items())
        expr = f"{expr_str} <= {limit}" if "LESS" in op else f"{expr_str} >= {limit}"
        return expr, f"Linear constraint {direction} {limit}"

    if ctype == "xby":
        x_var = c.get("x_var", "x")
        y_var = c.get("y_var", "y")
        param = c.get("param", "")
        return f"{x_var} >= {param} * {y_var}", f"{x_var} {direction} {param} of {y_var}"

    if ctype == "ratio":
        var = c.get("var", "x")
        return f"{var} >= {limit}% of total", f"{var} {direction} {limit}%"

    return ("", "Constraint")


# --------------------------------------------------------------------------- #
# Schema conversion                                                            #
# --------------------------------------------------------------------------- #

def nl4opt_record_to_problem(
    record: dict, record_id: str, source_tag: str = "nl4opt"
) -> dict[str, Any]:
    """Convert one NL4Opt JSONL record to the project's unified problem schema.

    Parameters
    ----------
    record:
        A single NL4Opt record dict (value from a JSONL line).
    record_id:
        The string key identifying this record within its file.
    source_tag:
        File-level tag (e.g. ``"train"``, ``"dev"``, ``"test"``).

    Returns
    -------
    dict
        Problem entry conforming to the unified schema.
    """
    doc = record.get("document", "").strip()
    if not doc:
        raise ValueError("Empty document")

    vars_list = record.get("vars") or []
    obj = record.get("obj_declaration") or {}
    consts = record.get("const_declarations") or []

    # Problem name from first sentence
    first_sentence = doc.split(".")[0].strip()[:60]
    name = first_sentence + ("..." if len(first_sentence) >= 60 else "")
    slug = _slug(doc[:80])
    uid = f"nl4opt_{source_tag}_{record_id}_{slug}"[:80]

    # Decision variables
    decision_variables = [
        {"symbol": v, "description": "Decision variable", "type": "continuous"}
        for v in vars_list
    ]

    # Objective
    sense = (obj.get("direction") or "minimize").lower()
    sense = "maximize" if "max" in sense else "minimize"
    objective = {
        "sense": sense,
        "expression_latex": _obj_expression(obj),
        "description": "",
    }

    # Constraints
    constraints_ilp = []
    for i, c in enumerate(consts):
        expr, desc = _constraint_expression(c)
        if expr:
            constraints_ilp.append(
                {"name": f"c{i + 1}", "expression_latex": expr, "description": desc}
            )

    return {
        "problem_id": uid,
        "problem_name": name,
        "aliases": [],
        "category": "lp",
        "tags": ["nl4opt", source_tag],
        "natural_language_descriptions": [doc],
        "complexity_class": "unknown",
        "formulation": {
            "sets": [],
            "parameters": [],
            "decision_variables": decision_variables,
            "objective": objective,
            "constraints_ilp": constraints_ilp,
            "constraints_lp_relaxation": "All variables are already continuous.",
            "alternative_formulations": [],
        },
        "solver_code": {"pyomo": None, "gurobi": None, "pulp": None},
        "benchmark_instances": [],
        "references": ["NL4Opt NeurIPS 2022 Competition"],
        "source_dataset": "nl4opt",
    }


def convert_to_schema(raw_problem: dict[str, Any], source_split: str) -> dict[str, Any]:
    """Convert a NL4Opt JSON record (file-based format) to the unified schema.

    This handles the JSON files produced by :func:`clone_or_update` / the
    locally-cloned repository, where records may use a slightly different
    field layout than the raw JSONL files.

    Parameters
    ----------
    raw_problem:
        A dictionary parsed from a local NL4Opt JSON file.
    source_split:
        One of ``'train'``, ``'dev'``, or ``'test'``.

    Returns
    -------
    dict
        Problem entry conforming to the unified schema.
    """
    problem_id = raw_problem.get("id", "")
    nl_desc = raw_problem.get("description", raw_problem.get("problem_text", ""))

    entities = raw_problem.get("entities", {})
    variables: list[dict] = []
    parameters: list[dict] = []
    for ent_name, info in entities.items():
        ent_type = info.get("type", "")
        if ent_type.upper() == "VAR":
            variables.append(
                {
                    "symbol": ent_name,
                    "description": info.get("definition", ""),
                    "type": "continuous",
                }
            )
        else:
            parameters.append(
                {
                    "symbol": ent_name,
                    "description": info.get("definition", ""),
                    "type": "scalar",
                }
            )

    obj_sense = "minimize"
    obj_expr = ""
    if "objective" in raw_problem:
        obj = raw_problem["objective"]
        obj_sense = obj.get("sense", "minimize").lower()
        obj_expr = obj.get("expression", "")

    constraints_ilp = []
    for i, c in enumerate(raw_problem.get("constraints", [])):
        constraints_ilp.append(
            {
                "name": f"c{i + 1}",
                "expression_latex": c if isinstance(c, str) else c.get("expression", ""),
                "description": "",
            }
        )

    return {
        "problem_id": f"nl4opt_{source_split}_{problem_id}",
        "problem_name": raw_problem.get("name", ""),
        "aliases": [],
        "category": raw_problem.get("category", "lp"),
        "tags": ["nl4opt", source_split],
        "natural_language_descriptions": [nl_desc] if nl_desc else [],
        "complexity_class": "unknown",
        "formulation": {
            "sets": [],
            "parameters": parameters,
            "decision_variables": variables,
            "objective": {
                "sense": obj_sense,
                "expression_latex": obj_expr,
                "description": "",
            },
            "constraints_ilp": constraints_ilp,
            "constraints_lp_relaxation": "All variables are already continuous.",
            "alternative_formulations": [],
        },
        "solver_code": {"pyomo": None, "gurobi": None, "pulp": None},
        "benchmark_instances": [],
        "references": ["NL4Opt NeurIPS 2022 Competition"],
        "source_dataset": "nl4opt",
    }


# --------------------------------------------------------------------------- #
# Data acquisition                                                             #
# --------------------------------------------------------------------------- #

def fetch_nl4opt_jsonl(url: str) -> list[tuple[str, dict]]:
    """Download a single .jsonl file and return a list of (record_id, record) pairs.

    Each line of a NL4Opt JSONL file is a JSON object with one key (the
    record id) whose value is the record dict.
    """
    with urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    records: list[tuple[str, dict]] = []
    for line in text.strip().split("\n"):
        if not line:
            continue
        data = json.loads(line)
        for key, value in data.items():
            if isinstance(value, dict) and value.get("document"):
                records.append((key, value))
    return records


def clone_or_update(repo_url: str = REPO_URL, dest: Path = RAW_DIR) -> None:
    """Clone the NL4Opt repository if not already present (requires gitpython).

    Parameters
    ----------
    repo_url:
        URL of the git repository to clone.
    dest:
        Local destination path.
    """
    if not _GIT_AVAILABLE:
        logger.warning("gitpython is not installed; cannot clone repository.")
        return
    if dest.exists():
        logger.info("Repository already cloned at %s; skipping clone.", dest)
    else:
        logger.info("Cloning %s into %s ...", repo_url, dest)
        _git.Repo.clone_from(repo_url, dest, depth=1)
        logger.info("Clone complete.")


def parse_split(split_path: Path, split_name: str) -> list[dict[str, Any]]:
    """Parse a locally-cloned NL4Opt JSON split file into unified schema entries.

    Parameters
    ----------
    split_path:
        Path to a ``.json`` split file on disk.
    split_name:
        One of ``'train'``, ``'dev'``, ``'test'``.

    Returns
    -------
    list[dict]
        Problem entries in the unified schema.
    """
    if not split_path.exists():
        logger.warning("Split file not found: %s", split_path)
        return []

    try:
        with open(split_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to parse %s: %s", split_path, exc)
        return []

    problems: list[dict[str, Any]] = []

    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        logger.warning("Unexpected data format in %s", split_path)
        return []

    for entry in items:
        try:
            problems.append(convert_to_schema(entry, split_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping entry in %s: %s", split_path, exc)

    logger.info("Parsed %d problems from %s split.", len(problems), split_name)
    return problems


# --------------------------------------------------------------------------- #
# Public collection API                                                        #
# --------------------------------------------------------------------------- #

def collect_nl4opt(
    out_dir: Path | None = None,
    max_per_file: int | None = None,
) -> list[dict[str, Any]]:
    """Download NL4Opt train/dev/test splits and convert to unified schema.

    Uses HTTP download (no git required).  Each split is cached as a ``.json``
    file under *out_dir* and the full merged list is written to
    ``data/processed/nl4opt.json``.

    Parameters
    ----------
    out_dir:
        Directory for per-split raw JSON cache.
        Defaults to ``<project_root>/data/raw/nl4opt``.
    max_per_file:
        If set, only the first *N* records per file are processed.
        Useful for quick smoke-tests.

    Returns
    -------
    list[dict]
        All collected problem entries in the unified schema.
    """
    out_dir = out_dir or (_project_root() / "data" / "raw" / "nl4opt")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_problems: list[dict[str, Any]] = []
    seen_docs: set[str] = set()

    for filename in FILES:
        url = f"{NL4OPT_BASE}/{filename}"
        tag = filename.replace(".jsonl", "")
        try:
            records = fetch_nl4opt_jsonl(url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch %s: %s", url, exc)
            continue

        if max_per_file is not None:
            records = records[:max_per_file]

        file_problems: list[dict[str, Any]] = []
        for record_id, record in records:
            try:
                problem = nl4opt_record_to_problem(record, record_id, source_tag=tag)
                doc_key = problem["natural_language_descriptions"][0][:200]
                if doc_key in seen_docs:
                    continue
                seen_docs.add(doc_key)
                file_problems.append(problem)
                all_problems.append(problem)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping record %s: %s", record_id, exc)

        cache_path = out_dir / filename.replace(".jsonl", ".json")
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(file_problems, fh, indent=2, ensure_ascii=False)
        logger.info("NL4Opt %s: %d problems -> %s", filename, len(file_problems), cache_path)

    # Write unified output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_problems, fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d NL4Opt problems to %s.", len(all_problems), OUTPUT_FILE)

    return all_problems


# --------------------------------------------------------------------------- #
# Standalone entry point                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    """CLI entry point: collect NL4Opt dataset and save to unified schema JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect NL4Opt dataset into the unified problem schema."
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=None,
        help="Max records per file (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for per-split raw JSON cache.",
    )
    parser.add_argument(
        "--clone",
        action="store_true",
        help=(
            "Clone the repository instead of using HTTP download. "
            "Requires gitpython."
        ),
    )
    args = parser.parse_args()

    if args.clone:
        clone_or_update()

    problems = collect_nl4opt(out_dir=args.out_dir, max_per_file=args.max_per_file)
    print(f"Total NL4Opt problems: {len(problems)}")


if __name__ == "__main__":
    main()
