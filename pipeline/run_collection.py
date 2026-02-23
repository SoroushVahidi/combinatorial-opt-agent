"""Master pipeline script for orchestrating all data collectors.

Runs each Phase 1 collector in sequence, merges outputs into a single
all_problems.json file, and prints summary statistics.

Usage:
    python pipeline/run_collection.py
    python pipeline/run_collection.py --sources nl4opt gurobi_optimods
    python pipeline/run_collection.py --sources nl4opt --skip-clone
    python pipeline/run_collection.py --output-dir data/processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Mapping of source name → (collector module path, output JSON file)
PHASE1_SOURCES: dict[str, tuple[str, str]] = {
    "nl4opt": ("collectors.collect_nl4opt", "nl4opt.json"),
    "gurobi_optimods": ("collectors.collect_gurobi_optimods", "gurobi_optimods.json"),
    "gurobi_examples": ("collectors.collect_gurobi_examples", "gurobi_examples.json"),
    "pyomo": ("collectors.collect_pyomo", "pyomo_examples.json"),
}

# Phase 2 sources (scaffolds only — not run by default)
PHASE2_SOURCES: dict[str, tuple[str, str]] = {
    "gamslib": ("collectors.collect_gamslib", "gamslib.json"),
    "miplib": ("collectors.collect_miplib", "miplib.json"),
    "or_library": ("collectors.collect_or_library", "or_library.json"),
}

ALL_SOURCES = {**PHASE1_SOURCES, **PHASE2_SOURCES}


def _import_and_run(module_path: str, skip_clone: bool) -> bool:
    """Dynamically import a collector module and call its main() function.

    Parameters
    ----------
    module_path:
        Dotted module path (e.g. ``collectors.collect_nl4opt``).
    skip_clone:
        If True, set the ``SKIP_CLONE`` environment variable so collectors
        skip the git clone step.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    import importlib
    import os

    if skip_clone:
        os.environ["SKIP_CLONE"] = "1"

    try:
        module = importlib.import_module(module_path)
        module.main()  # type: ignore[attr-defined]
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Collector %s failed: %s", module_path, exc)
        return False
    finally:
        os.environ.pop("SKIP_CLONE", None)


def count_problems(json_path: Path) -> int:
    """Return the number of problems in a processed JSON file."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return len(data) if isinstance(data, list) else 0
    except (OSError, json.JSONDecodeError):
        return 0


def merge_outputs(output_dir: Path, source_files: list[str]) -> dict[str, Any]:
    """Merge individual source JSON files into a single all_problems.json.

    Parameters
    ----------
    output_dir:
        Directory containing per-source JSON files.
    source_files:
        List of filenames (relative to output_dir) to merge.

    Returns
    -------
    dict
        Summary statistics.
    """
    all_problems: list[dict[str, Any]] = []

    for filename in source_files:
        path = output_dir / filename
        if not path.exists():
            logger.warning("Output file not found, skipping: %s", path)
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                problems = json.load(fh)
            if isinstance(problems, list):
                all_problems.extend(problems)
                logger.info("Merged %d problems from %s.", len(problems), filename)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read %s: %s", path, exc)

    merged_path = output_dir / "all_problems.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w", encoding="utf-8") as fh:
        json.dump(all_problems, fh, indent=2, ensure_ascii=False)

    logger.info("Merged %d total problems → %s", len(all_problems), merged_path)

    # Compute statistics
    by_source: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_type: dict[str, int] = {}

    for p in all_problems:
        src = p.get("source_dataset", "unknown")
        cat = p.get("category", "unknown")
        tags = p.get("tags", [])

        by_source[src] = by_source.get(src, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

        # Classify as LP, ILP, or MILP based on decision variable types
        var_types = {
            v.get("type", "")
            for v in p.get("formulation", {}).get("decision_variables", [])
        }
        if "binary" in var_types or "integer" in var_types:
            if "continuous" in var_types:
                problem_type = "MILP"
            else:
                problem_type = "ILP"
        else:
            problem_type = "LP"
        by_type[problem_type] = by_type.get(problem_type, 0) + 1

    return {
        "total": len(all_problems),
        "by_source": by_source,
        "by_category": by_category,
        "by_type": by_type,
    }


def print_summary(stats: dict[str, Any]) -> None:
    """Print a human-readable summary of collection statistics."""
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    print(f"Total problems collected: {stats['total']}")

    print("\nBy source:")
    for src, count in sorted(stats["by_source"].items()):
        print(f"  {src:<30} {count:>5}")

    print("\nBy category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat:<30} {count:>5}")

    print("\nBy problem type (LP / ILP / MILP):")
    for ptype, count in sorted(stats["by_type"].items()):
        print(f"  {ptype:<30} {count:>5}")
    print("=" * 60 + "\n")


def main() -> None:
    """Parse arguments and orchestrate the collection pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the combinatorial optimization data collection pipeline."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(PHASE1_SOURCES.keys()),
        choices=list(ALL_SOURCES.keys()),
        help="Which sources to collect (default: all Phase 1 sources).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed JSON outputs (default: data/processed).",
    )
    parser.add_argument(
        "--skip-clone",
        action="store_true",
        help="Skip git clone steps if repositories are already downloaded.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    collected_files: list[str] = []

    for source_name in args.sources:
        if source_name not in ALL_SOURCES:
            logger.warning("Unknown source: %s — skipping.", source_name)
            continue

        module_path, output_file = ALL_SOURCES[source_name]
        logger.info("Running collector: %s ...", source_name)
        success = _import_and_run(module_path, args.skip_clone)
        results[source_name] = success

        if success:
            count = count_problems(output_dir / output_file)
            logger.info("[OK] %s: collected %d problems.", source_name, count)
            collected_files.append(output_file)
        else:
            logger.error("[FAIL] %s: collector failed.", source_name)

    # Merge all outputs
    stats = merge_outputs(output_dir, collected_files)
    print_summary(stats)

    # Report overall success/failure
    failed = [s for s, ok in results.items() if not ok]
    if failed:
        logger.warning("The following collectors failed: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
