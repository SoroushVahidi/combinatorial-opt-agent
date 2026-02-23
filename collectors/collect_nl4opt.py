"""Collector for the NL4Opt dataset.

Clones https://github.com/nl4opt/nl4opt-competition into data/raw/nl4opt/,
parses train/dev/test JSON splits, converts each problem to the unified schema,
and saves results to data/processed/nl4opt.json.

Usage:
    python collectors/collect_nl4opt.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import git

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/nl4opt/nl4opt-competition"
RAW_DIR = Path("data/raw/nl4opt")
OUTPUT_FILE = Path("data/processed/nl4opt.json")


def clone_or_update(repo_url: str, dest: Path) -> None:
    """Clone the repository if not present, otherwise pull the latest changes."""
    if dest.exists():
        logger.info("Repository already cloned at %s; skipping clone.", dest)
    else:
        logger.info("Cloning %s into %s ...", repo_url, dest)
        git.Repo.clone_from(repo_url, dest, depth=1)
        logger.info("Clone complete.")


def _entity_type_to_var_type(entity_type: str) -> str:
    """Map NL4Opt entity types to unified schema variable types."""
    mapping = {
        "VAR": "continuous",
        "CONST": "continuous",
        "PARAM": "continuous",
        "OBJ": "continuous",
    }
    return mapping.get(entity_type.upper(), "continuous")


def convert_to_schema(raw_problem: dict[str, Any], source_split: str) -> dict[str, Any]:
    """Convert a single NL4Opt problem entry to the unified schema.

    Parameters
    ----------
    raw_problem:
        A dictionary parsed from the NL4Opt JSON dataset.
    source_split:
        One of 'train', 'dev', or 'test'.

    Returns
    -------
    dict
        A problem entry conforming to the unified schema.
    """
    problem_id = raw_problem.get("id", "")
    nl_desc = raw_problem.get("description", raw_problem.get("problem_text", ""))

    entities = raw_problem.get("entities", {})
    variables = []
    parameters = []
    for name, info in entities.items():
        ent_type = info.get("type", "")
        if ent_type.upper() in ("VAR",):
            variables.append(
                {
                    "symbol": name,
                    "description": info.get("definition", ""),
                    "type": "continuous",
                }
            )
        else:
            parameters.append(
                {
                    "symbol": name,
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
        "solver_code": {
            "pyomo": None,
            "gurobi": None,
            "pulp": None,
        },
        "benchmark_instances": [],
        "references": ["NL4Opt NeurIPS 2022 Competition"],
        "source_dataset": "nl4opt",
    }


def parse_split(split_path: Path, split_name: str) -> list[dict[str, Any]]:
    """Parse a single NL4Opt JSON split file and return unified schema entries."""
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

    # NL4Opt stores problems as a dict keyed by problem id, or as a list
    if isinstance(data, dict):
        items = data.values()
    elif isinstance(data, list):
        items = data
    else:
        logger.warning("Unexpected data format in %s", split_path)
        return []

    for entry in items:
        try:
            problems.append(convert_to_schema(entry, split_name))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping entry in %s due to error: %s", split_path, exc)

    logger.info("Parsed %d problems from %s split.", len(problems), split_name)
    return problems


def main() -> None:
    """Main entry point: clone, parse, and save NL4Opt dataset."""
    clone_or_update(REPO_URL, RAW_DIR)

    all_problems: list[dict[str, Any]] = []

    # NL4Opt dataset stores splits at train/train.json, dev/dev.json, test/test.json
    # but also possible as generation_prefs/ sub-directory; search broadly.
    split_candidates = {
        "train": [
            RAW_DIR / "train" / "train.json",
            RAW_DIR / "train.json",
        ],
        "dev": [
            RAW_DIR / "dev" / "dev.json",
            RAW_DIR / "dev.json",
        ],
        "test": [
            RAW_DIR / "test" / "test.json",
            RAW_DIR / "test.json",
        ],
    }

    for split_name, candidates in split_candidates.items():
        for candidate in candidates:
            if candidate.exists():
                all_problems.extend(parse_split(candidate, split_name))
                break

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_problems, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d NL4Opt problems to %s.", len(all_problems), OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
