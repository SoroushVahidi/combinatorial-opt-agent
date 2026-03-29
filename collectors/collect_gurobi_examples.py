"""Collector for Gurobi Modeling Examples.

Clones https://github.com/Gurobi/modeling-examples into
data/raw/gurobi-examples/, scans each subdirectory for .ipynb notebooks,
extracts markdown descriptions and LaTeX formulations as well as Gurobi
solver code, and saves results to data/processed/gurobi_examples.json.

Usage:
    python collectors/collect_gurobi_examples.py
"""

import json
import logging
from pathlib import Path
from typing import Any

import git

from parsers.notebook_parser import parse_jupyter_notebook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Gurobi/modeling-examples"
RAW_DIR = Path("data/raw/gurobi-examples")
OUTPUT_FILE = Path("data/processed/gurobi_examples.json")

# Well-known subdirectories; the collector also discovers others automatically.
KNOWN_SUBDIRS = [
    "traveling_salesman",
    "facility_location",
    "cell_tower_coverage",
    "supply_network_design",
    "workforce",
    "factory_planning",
    "food_manufacturing",
    "portfolio_selection_optimization",
    "marketing_campaign_optimization",
    "colgen-cutting_stock",
    "customer_assignment",
    "curve_fitting",
]


def clone_or_update(repo_url: str, dest: Path) -> None:
    """Clone the repository if not present, otherwise skip."""
    if dest.exists():
        logger.info("Repository already cloned at %s; skipping clone.", dest)
    else:
        logger.info("Cloning %s into %s ...", repo_url, dest)
        git.Repo.clone_from(repo_url, dest, depth=1)
        logger.info("Clone complete.")


def convert_to_schema(
    parsed: dict[str, Any], notebook_path: Path
) -> dict[str, Any]:
    """Convert a parsed Jupyter notebook to the unified schema.

    Parameters
    ----------
    parsed:
        Output from :func:`parsers.notebook_parser.parse_jupyter_notebook`.
    notebook_path:
        Path to the source .ipynb file.

    Returns
    -------
    dict
        A problem entry conforming to the unified schema.
    """
    example_name = notebook_path.parent.name
    problem_id = f"gurobi_examples_{example_name}"

    nl_descriptions = parsed.get("nl_descriptions", [])

    constraints_ilp = []
    for i, latex_block in enumerate(parsed.get("latex_formulations", [])):
        constraints_ilp.append(
            {
                "name": f"formulation_block_{i + 1}",
                "expression_latex": latex_block,
                "description": "",
            }
        )

    gurobi_code = "\n\n".join(parsed.get("code_cells", []))

    return {
        "problem_id": problem_id,
        "problem_name": example_name.replace("_", " ").title(),
        "aliases": [],
        "category": "mixed_integer",
        "tags": ["gurobi", "modeling_examples"],
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
            "gurobi": gurobi_code if gurobi_code else None,
            "pulp": None,
        },
        "benchmark_instances": [],
        "references": [
            "Gurobi Modeling Examples — https://github.com/Gurobi/modeling-examples"
        ],
        "source_dataset": "gurobi_examples",
    }


def main() -> None:
    """Main entry point: clone, parse, and save Gurobi Modeling Examples."""
    clone_or_update(REPO_URL, RAW_DIR)

    all_problems: list[dict[str, Any]] = []
    seen_dirs: set[str] = set()

    # Collect notebooks from known subdirectories first, then discover the rest.
    candidate_dirs = list(KNOWN_SUBDIRS)
    for subdir in sorted(RAW_DIR.iterdir()):
        if subdir.is_dir() and subdir.name not in seen_dirs and not subdir.name.startswith("."):
            if subdir.name not in candidate_dirs:
                candidate_dirs.append(subdir.name)

    for subdir_name in candidate_dirs:
        subdir_path = RAW_DIR / subdir_name
        if not subdir_path.is_dir():
            continue

        notebooks = list(subdir_path.glob("*.ipynb"))
        if not notebooks:
            logger.debug("No notebooks found in %s", subdir_path)
            continue

        seen_dirs.add(subdir_name)

        for nb_path in sorted(notebooks):
            try:
                parsed = parse_jupyter_notebook(nb_path)
                all_problems.append(convert_to_schema(parsed, nb_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s due to error: %s", nb_path, exc)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_problems, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d Gurobi example problems to %s.", len(all_problems), OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
