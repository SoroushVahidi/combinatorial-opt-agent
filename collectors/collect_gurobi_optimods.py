"""Collector for Gurobi OptiMods.

Clones https://github.com/Gurobi/gurobi-optimods into data/raw/gurobi-optimods/,
scans docs/source/mods/ for .rst files, extracts problem descriptions and LaTeX
math formulations, and saves results to data/processed/gurobi_optimods.json.

Usage:
    python collectors/collect_gurobi_optimods.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import git

from parsers.rst_parser import parse_rst_documentation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Gurobi/gurobi-optimods"
RAW_DIR = Path("data/raw/gurobi-optimods")
OUTPUT_FILE = Path("data/processed/gurobi_optimods.json")


def clone_or_update(repo_url: str, dest: Path) -> None:
    """Clone the repository if not present, otherwise skip."""
    if dest.exists():
        logger.info("Repository already cloned at %s; skipping clone.", dest)
    else:
        logger.info("Cloning %s into %s ...", repo_url, dest)
        git.Repo.clone_from(repo_url, dest, depth=1)
        logger.info("Clone complete.")


def find_python_implementation(mod_name: str) -> str | None:
    """Locate the Python source file for a given mod name."""
    src_path = RAW_DIR / "src" / "gurobi_optimods" / f"{mod_name}.py"
    if src_path.exists():
        try:
            return src_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read %s: %s", src_path, exc)
    return None


def convert_to_schema(parsed: dict[str, Any], rst_path: Path) -> dict[str, Any]:
    """Convert a parsed RST document to the unified schema.

    Parameters
    ----------
    parsed:
        Output from :func:`parsers.rst_parser.parse_rst_documentation`.
    rst_path:
        Path to the source .rst file (used to derive the mod name).

    Returns
    -------
    dict
        A problem entry conforming to the unified schema.
    """
    mod_name = rst_path.stem
    python_code = find_python_implementation(mod_name)

    constraints_ilp = []
    for i, math_block in enumerate(parsed.get("math_formulations", [])):
        constraints_ilp.append(
            {
                "name": f"formulation_block_{i + 1}",
                "expression_latex": math_block,
                "description": "",
            }
        )

    nl_descriptions = []
    if parsed.get("description"):
        nl_descriptions.append(parsed["description"])

    return {
        "problem_id": f"gurobi_optimods_{mod_name}",
        "problem_name": parsed.get("title", mod_name),
        "aliases": [],
        "category": "mixed_integer",
        "tags": ["gurobi", "optimods"],
        "natural_language_descriptions": nl_descriptions,
        "complexity_class": "NP-hard",
        "formulation": {
            "sets": [],
            "parameters": [],
            "decision_variables": [],
            "objective": {
                "sense": "minimize",
                "expression_latex": "",
                "description": "",
            },
            "constraints_ilp": constraints_ilp,
            "constraints_lp_relaxation": "Relax all integrality constraints.",
            "alternative_formulations": [],
        },
        "solver_code": {
            "pyomo": None,
            "gurobi": python_code,
            "pulp": None,
        },
        "benchmark_instances": [],
        "references": [
            "Gurobi OptiMods — https://github.com/Gurobi/gurobi-optimods"
        ],
        "source_dataset": "gurobi_optimods",
    }


def main() -> None:
    """Main entry point: clone, parse, and save Gurobi OptiMods dataset."""
    clone_or_update(REPO_URL, RAW_DIR)

    mods_dir = RAW_DIR / "docs" / "source" / "mods"
    if not mods_dir.exists():
        logger.error("Mods documentation directory not found: %s", mods_dir)
        return

    rst_files = list(mods_dir.glob("*.rst"))
    logger.info("Found %d .rst files in %s.", len(rst_files), mods_dir)

    all_problems: list[dict[str, Any]] = []
    for rst_path in sorted(rst_files):
        try:
            parsed = parse_rst_documentation(rst_path)
            all_problems.append(convert_to_schema(parsed, rst_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s due to error: %s", rst_path, exc)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_problems, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d Gurobi OptiMod problems to %s.", len(all_problems), OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
