"""Master pipeline script for orchestrating all data collectors.

Runs each configured collector in sequence, merges all outputs into a single
``data/processed/all_problems.json`` file, runs the catalog merge step, then
optionally rebuilds ``data/processed/all_problems_extended.json`` so the search
service always has access to the full catalog.

Pipeline steps
--------------
1. Collect each requested source (NL4Opt, OptMATH, Gurobi examples, Pyomo, ...).
2. Merge per-source JSON files into ``data/processed/all_problems.json``.
3. Run ``scripts/merge_catalog`` to merge base + custom + collected problems
   into the full catalog used by the app.
4. Rebuild ``data/processed/all_problems_extended.json`` so the search service
   always has access to the full catalog.  Without this step the extended
   catalog can become stale and silently restrict search to a tiny subset
   (a known production bug: only 36 of 1,597 problems were searchable when
   the catalog grew but the extended file was not rebuilt).

Usage examples::

    # Run all Phase 1 collectors:
    python pipeline/run_collection.py

    # Run only NL4Opt with a record limit (useful for quick smoke-tests):
    python pipeline/run_collection.py --sources nl4opt --max-per-file 100

    # Run specific sources, custom output dir, skip git clones:
    python pipeline/run_collection.py --sources nl4opt gurobi_optimods \\
        --output-dir data/processed --skip-clone

    # Rebuild extended catalog with web enrichment after collection:
    python pipeline/run_collection.py --enrich
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so submodules resolve correctly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Source registry                                                              #
# --------------------------------------------------------------------------- #

# Phase 1 sources — run by default
PHASE1_SOURCES: dict[str, tuple[str, str]] = {
    "nl4opt": ("collectors.collect_nl4opt", "nl4opt.json"),
    "optmath": ("collectors.collect_optmath", "optmath.json"),
    "gurobi_optimods": ("collectors.collect_gurobi_optimods", "gurobi_optimods.json"),
    "gurobi_examples": ("collectors.collect_gurobi_examples", "gurobi_examples.json"),
    "pyomo": ("collectors.collect_pyomo", "pyomo_examples.json"),
}

# Phase 2 sources — scaffolds only, not run by default
PHASE2_SOURCES: dict[str, tuple[str, str]] = {
    "gamslib": ("collectors.collect_gamslib", "gamslib.json"),
    "miplib": ("collectors.collect_miplib", "miplib.json"),
    "or_library": ("collectors.collect_or_library", "or_library.json"),
}

ALL_SOURCES: dict[str, tuple[str, str]] = {**PHASE1_SOURCES, **PHASE2_SOURCES}


# --------------------------------------------------------------------------- #
# Pipeline helpers                                                             #
# --------------------------------------------------------------------------- #

def _import_and_run(module_path: str, skip_clone: bool) -> bool:
    """Dynamically import a collector module and call its ``main()`` function.

    Parameters
    ----------
    module_path:
        Dotted module path, e.g. ``collectors.collect_nl4opt``.
    skip_clone:
        When *True*, sets the ``SKIP_CLONE`` environment variable before
        calling ``main()`` so collectors can honour it.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if an exception is raised.
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
    """Return the number of entries in a processed JSON list file."""
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return len(data) if isinstance(data, list) else 0
    except (OSError, json.JSONDecodeError):
        return 0


def merge_outputs(output_dir: Path, source_files: list[str]) -> dict[str, Any]:
    """Merge per-source JSON files into ``all_problems.json``.

    Parameters
    ----------
    output_dir:
        Directory that contains the per-source JSON files.
    source_files:
        Filenames (relative to *output_dir*) to merge.

    Returns
    -------
    dict
        Summary statistics: total count, breakdowns by source / category /
        problem type (LP, ILP, MILP).
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
    logger.info("Merged %d total problems -> %s", len(all_problems), merged_path)

    # Compute statistics
    by_source: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_type: dict[str, int] = {}

    for p in all_problems:
        src = p.get("source_dataset", "unknown")
        cat = p.get("category", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

        # Classify as LP / ILP / MILP from decision variable types
        var_types = {
            v.get("type", "")
            for v in p.get("formulation", {}).get("decision_variables", [])
        }
        if "binary" in var_types or "integer" in var_types:
            problem_type = "MILP" if "continuous" in var_types else "ILP"
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
    """Print a human-readable collection summary to stdout."""
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


def _run_merge_catalog() -> None:
    """Run ``scripts/merge_catalog`` to merge base + custom + collected problems.

    Skipped silently if the module is not present.
    """
    try:
        from scripts.merge_catalog import main as merge_catalog_main  # type: ignore[import]
        print("Merging into catalog (NL4Opt + OptMATH + classic_extra)...")
        merge_catalog_main()
    except ImportError:
        logger.debug("scripts.merge_catalog not available; skipping catalog merge.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Catalog merge failed: %s", exc)


def _rebuild_extended_catalog(enrich: bool) -> None:
    """Rebuild ``all_problems_extended.json`` via ``build_extended_catalog``.

    This step keeps the extended catalog in sync with the base catalog so the
    search service always has access to the full problem set.  Skipped
    silently when the module is not present.
    """
    try:
        from build_extended_catalog import build_extended_catalog  # type: ignore[import]
        print("Rebuilding all_problems_extended.json (merging base + custom + enrichment)...")
        out = build_extended_catalog(enrich=enrich, verbose=True)
        print(f"Extended catalog written to: {out}")
    except ImportError:
        logger.debug(
            "build_extended_catalog not available; skipping extended catalog rebuild."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Extended catalog rebuild failed: %s", exc)


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #

def main() -> None:
    """Parse CLI arguments and orchestrate the collection pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the combinatorial optimization data collection pipeline."
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=list(PHASE1_SOURCES.keys()),
        choices=list(ALL_SOURCES.keys()),
        help="Sources to collect (default: all Phase 1 sources).",
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
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=None,
        help="Limit NL4Opt records per file (default: all). Useful for quick tests.",
    )
    parser.add_argument(
        "--max-optmath",
        type=int,
        default=None,
        help="Limit OptMATH benchmark problems (default: all).",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help=(
            "Fetch missing formulations from the web when rebuilding the "
            "extended catalog (requires network access)."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool] = {}
    collected_files: list[str] = []

    for source_name in args.sources:
        if source_name not in ALL_SOURCES:
            logger.warning("Unknown source: %s -- skipping.", source_name)
            continue

        module_path, output_file = ALL_SOURCES[source_name]
        logger.info("Running collector: %s ...", source_name)

        # For NL4Opt, call the callable API directly to honour --max-per-file.
        if source_name == "nl4opt" and args.max_per_file is not None:
            try:
                from collectors.collect_nl4opt import collect_nl4opt
                collect_nl4opt(max_per_file=args.max_per_file)
                success = True
            except Exception as exc:  # noqa: BLE001
                logger.error("Collector nl4opt failed: %s", exc)
                success = False
        # For OptMATH, call the callable API directly to honour --max-optmath.
        elif source_name == "optmath" and args.max_optmath is not None:
            try:
                from collectors.collect_optmath import collect_optmath
                collect_optmath(max_items=args.max_optmath)
                success = True
            except Exception as exc:  # noqa: BLE001
                logger.error("Collector optmath failed: %s", exc)
                success = False
        else:
            success = _import_and_run(module_path, args.skip_clone)

        results[source_name] = success

        if success:
            n = count_problems(output_dir / output_file)
            logger.info("[OK] %s: collected %d problems.", source_name, n)
            collected_files.append(output_file)
        else:
            logger.error("[FAIL] %s: collector failed.", source_name)

    # Merge all per-source outputs into all_problems.json
    stats = merge_outputs(output_dir, collected_files)
    print_summary(stats)

    # Run the catalog merge step (NL4Opt + OptMATH + classic_extra)
    _run_merge_catalog()

    # Rebuild the extended catalog so the search service sees the full catalog
    _rebuild_extended_catalog(enrich=args.enrich)

    # Exit with non-zero status if any collector failed
    failed = [s for s, ok in results.items() if not ok]
    if failed:
        logger.warning("The following collectors failed: %s", failed)
        sys.exit(1)

    # Always rebuild the extended catalog so the search service sees the full
    # catalog.  Without this step, all_problems_extended.json can become stale
    # and silently restrict search to a tiny subset of the catalog.
    print("Rebuilding all_problems_extended.json (merging base + custom + enrichment)...")
    from build_extended_catalog import build_extended_catalog
    out = build_extended_catalog(enrich=args.enrich, verbose=True)
    print(f"Extended catalog written to: {out}")


if __name__ == "__main__":
    main()

