"""Scaffold for validating optimization problem formulations.

Loads a problem from the unified schema, optionally builds and solves the
Pyomo model if solver_code.pyomo is present, and reports the result.

This script will be expanded in Phases 2–3 to support Gurobi and PuLP
solvers and to run full benchmark validation suites.

Usage:
    python validation/solve_and_verify.py <path_to_problem_json>
    python validation/solve_and_verify.py data/processed/nl4opt.json --index 0
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_problem(json_path: Path, index: int = 0) -> dict[str, Any] | None:
    """Load a single problem entry from a unified schema JSON file.

    Parameters
    ----------
    json_path:
        Path to the JSON file (list of problem entries).
    index:
        Index of the problem to load (default: 0).

    Returns
    -------
    dict or None
        The problem entry, or None if loading fails.
    """
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load %s: %s", json_path, exc)
        return None

    if not isinstance(data, list):
        logger.error("Expected a JSON array in %s.", json_path)
        return None

    if index >= len(data):
        logger.error("Index %d out of range (file has %d entries).", index, len(data))
        return None

    return data[index]


def solve_with_pyomo(problem: dict[str, Any]) -> bool:
    """Attempt to build and solve a Pyomo model from solver_code.pyomo.

    Parameters
    ----------
    problem:
        A unified schema problem entry.

    Returns
    -------
    bool
        True if the model was solved successfully, False otherwise.
    """
    pyomo_code = problem.get("solver_code", {}).get("pyomo")
    if not pyomo_code:
        logger.info("No Pyomo solver code found for problem '%s'.",
                    problem.get("problem_name", ""))
        return False

    # Security note: exec() is used here on code sourced from the project's own
    # curated dataset (data/processed/). This script is an internal validation tool
    # and should not be used with untrusted external code.
    try:
        import pyomo.environ as pyo  # noqa: F401

        namespace: dict[str, Any] = {}
        exec(pyomo_code, namespace)  # noqa: S102

        # Look for a model object in the namespace
        model = namespace.get("model") or namespace.get("instance")
        if model is None:
            logger.warning("No 'model' or 'instance' object found after exec.")
            return False

        solver = pyo.SolverFactory("glpk")
        if not solver.available():
            logger.warning("GLPK solver not available; skipping solve.")
            return False

        result = solver.solve(model)
        termination = result.solver.termination_condition
        logger.info("Pyomo solve result: %s", termination)
        return str(termination) == "optimal"

    except ImportError:
        logger.warning("Pyomo is not installed; skipping Pyomo validation.")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during Pyomo solve: %s", exc)
        return False


def solve_with_gurobi(problem: dict[str, Any]) -> bool:
    """Placeholder for Gurobi solver validation (Phase 2–3).

    Parameters
    ----------
    problem:
        A unified schema problem entry.

    Returns
    -------
    bool
        Always False until implemented.
    """
    logger.info("Gurobi validation is a placeholder for Phase 2–3.")
    return False


def solve_with_pulp(problem: dict[str, Any]) -> bool:
    """Placeholder for PuLP solver validation (Phase 2–3).

    Parameters
    ----------
    problem:
        A unified schema problem entry.

    Returns
    -------
    bool
        Always False until implemented.
    """
    logger.info("PuLP validation is a placeholder for Phase 2–3.")
    return False


def main() -> None:
    """Parse arguments, load a problem, and attempt validation."""
    parser = argparse.ArgumentParser(
        description="Validate an optimization problem formulation by solving it."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a unified schema JSON file (list of problem entries).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the problem entry to validate (default: 0).",
    )
    args = parser.parse_args()

    problem = load_problem(args.json_path, args.index)
    if problem is None:
        sys.exit(1)

    name = problem.get("problem_name", "Unknown")
    logger.info("Validating problem: %s", name)

    success = solve_with_pyomo(problem)

    # Placeholder calls for future solver support
    if not success:
        solve_with_gurobi(problem)
        solve_with_pulp(problem)

    if success:
        logger.info("Validation PASSED for '%s'.", name)
    else:
        logger.warning(
            "Validation could not confirm optimality for '%s'. "
            "This may be expected if no solver code is available.",
            name,
        )


if __name__ == "__main__":
    main()
