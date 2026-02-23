"""Collector for Pyomo examples.

Performs a sparse checkout of the examples/ directory from
https://github.com/Pyomo/pyomo into data/raw/pyomo/, parses Python model
files to extract sets, parameters, variables, objectives, and constraints,
and saves results to data/processed/pyomo_examples.json.

Usage:
    python collectors/collect_pyomo.py
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any

import git

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Pyomo/pyomo"
RAW_DIR = Path("data/raw/pyomo")
OUTPUT_FILE = Path("data/processed/pyomo_examples.json")

# Directories within the Pyomo repo that contain optimization examples
EXAMPLE_DIRS = ["examples/pyomo", "examples/gdp"]

# AST node names that indicate Pyomo model components
PYOMO_COMPONENT_NAMES = {
    "Set", "Param", "Var", "Objective", "Constraint",
    "ConcreteModel", "AbstractModel", "Block",
}


def clone_sparse(repo_url: str, dest: Path) -> None:
    """Clone only the examples/ directory via sparse checkout."""
    if dest.exists():
        logger.info("Repository already cloned at %s; skipping clone.", dest)
        return
    logger.info("Sparse-cloning %s into %s ...", repo_url, dest)
    try:
        repo = git.Repo.clone_from(
            repo_url,
            dest,
            depth=1,
            no_checkout=True,
        )
        # Configure sparse checkout
        repo.git.config("core.sparseCheckout", "true")
        sparse_file = dest / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text("examples/\n", encoding="utf-8")
        repo.git.checkout("HEAD")
        logger.info("Sparse checkout complete.")
    except git.GitCommandError as exc:
        logger.error("Git error during sparse clone: %s", exc)


def extract_pyomo_components(source: str) -> dict[str, list[str]]:
    """Parse a Python source file and extract Pyomo model component names.

    Parameters
    ----------
    source:
        Python source code as a string.

    Returns
    -------
    dict
        Mapping of component type (e.g. 'Var', 'Constraint') to list of names.
    """
    components: dict[str, list[str]] = {
        "sets": [],
        "parameters": [],
        "variables": [],
        "objectives": [],
        "constraints": [],
    }

    type_map = {
        "Set": "sets",
        "Param": "parameters",
        "Var": "variables",
        "Objective": "objectives",
        "Constraint": "constraints",
    }

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        logger.debug("Syntax error parsing file: %s", exc)
        return components

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                name = None
                if isinstance(target, ast.Name):
                    name = target.id
                elif isinstance(target, ast.Attribute):
                    name = target.attr

                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    func_name = ""
                    if isinstance(func, ast.Name):
                        func_name = func.id
                    elif isinstance(func, ast.Attribute):
                        func_name = func.attr

                    if func_name in type_map and name:
                        components[type_map[func_name]].append(name)

    return components


def convert_to_schema(py_path: Path) -> dict[str, Any] | None:
    """Convert a Pyomo Python example file to the unified schema.

    Parameters
    ----------
    py_path:
        Path to the Python source file.

    Returns
    -------
    dict or None
        Unified schema entry, or None if the file is not a Pyomo model.
    """
    try:
        source = py_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read %s: %s", py_path, exc)
        return None

    # Heuristic: only process files that reference Pyomo components
    if not any(name in source for name in PYOMO_COMPONENT_NAMES):
        return None

    components = extract_pyomo_components(source)

    variables = [
        {"symbol": v, "description": "", "type": "continuous"}
        for v in components["variables"]
    ]
    parameters = [
        {"symbol": p, "description": "", "type": "scalar"}
        for p in components["parameters"]
    ]
    sets = [{"symbol": s, "description": ""} for s in components["sets"]]

    constraints_ilp = [
        {"name": c, "expression_latex": "", "description": ""}
        for c in components["constraints"]
    ]

    obj_sense = "minimize"
    if "maximize" in source.lower():
        obj_sense = "maximize"

    # Classify LP / ILP / MILP based on detected variable types
    has_integer = bool(components["variables"]) and any(
        kw in source for kw in ("Binary", "Boolean", "Integer", "within=pyo.Binary")
    )
    category = "ilp" if has_integer else "lp"

    example_id = py_path.stem
    return {
        "problem_id": f"pyomo_{example_id}",
        "problem_name": example_id.replace("_", " ").title(),
        "aliases": [],
        "category": category,
        "tags": ["pyomo"],
        "natural_language_descriptions": [],
        "complexity_class": "unknown",
        "formulation": {
            "sets": sets,
            "parameters": parameters,
            "decision_variables": variables,
            "objective": {
                "sense": obj_sense,
                "expression_latex": "",
                "description": "",
            },
            "constraints_ilp": constraints_ilp,
            "constraints_lp_relaxation": "Relax all integrality constraints.",
            "alternative_formulations": [],
        },
        "solver_code": {
            "pyomo": source,
            "gurobi": None,
            "pulp": None,
        },
        "benchmark_instances": [],
        "references": ["Pyomo Project — https://github.com/Pyomo/pyomo"],
        "source_dataset": "pyomo",
    }


def main() -> None:
    """Main entry point: clone, parse, and save Pyomo examples."""
    clone_sparse(REPO_URL, RAW_DIR)

    all_problems: list[dict[str, Any]] = []

    for example_subdir in EXAMPLE_DIRS:
        example_path = RAW_DIR / example_subdir
        if not example_path.exists():
            logger.warning("Example directory not found: %s", example_path)
            continue

        py_files = list(example_path.rglob("*.py"))
        logger.info(
            "Found %d Python files in %s.", len(py_files), example_path
        )

        for py_path in sorted(py_files):
            try:
                entry = convert_to_schema(py_path)
                if entry is not None:
                    all_problems.append(entry)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s due to error: %s", py_path, exc)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_problems, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Saved %d Pyomo example problems to %s.", len(all_problems), OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
