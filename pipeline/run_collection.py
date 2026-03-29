"""
Master pipeline: run all collectors and merge into data/processed/all_problems.json,
then rebuild data/processed/all_problems_extended.json so that the search service
always has access to the full catalog.

Usage: python pipeline/run_collection.py [--max-per-file N]
  --max-per-file N  Limit NL4Opt to N records per file (default: all).

The pipeline runs in order:
  1. Collect NL4Opt (natural-language + LP) problems.
  2. Collect OptMATH benchmark problems.
  3. Merge everything into data/processed/all_problems.json.
  4. Rebuild data/processed/all_problems_extended.json (merges base + custom
     + optional web-enriched formulations via --enrich).

Step 4 ensures the extended catalog stays in sync with the base catalog.
Without it the search service is restricted to whatever subset happened to be
in all_problems_extended.json at deploy time (a critical pipeline bug: only 36
of 1,597 problems were searchable when the catalog grew but the extended file
was not rebuilt).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collectors.collect_nl4opt import collect_nl4opt
from collectors.collect_optmath import collect_optmath
from scripts.merge_catalog import main as merge_catalog_main


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Collect public data and update problem catalog")
    p.add_argument("--max-per-file", type=int, default=None,
                   help="Max NL4Opt records per file (default: all)")
    p.add_argument("--max-optmath", type=int, default=None,
                   help="Max OptMATH benchmark problems (default: all)")
    p.add_argument("--enrich", action="store_true", default=False,
                   help=(
                       "Fetch missing formulations from the web when rebuilding "
                       "the extended catalog (requires network access)."
                   ))
    args = p.parse_args()

    print("Collecting NL4Opt (natural language + LP)...")
    collect_nl4opt(max_per_file=args.max_per_file)

    print("Collecting OptMATH benchmark (NL optimization descriptions)...")
    collect_optmath(max_items=args.max_optmath)

    print("Merging into catalog (NL4Opt + OptMATH + classic_extra)...")
    merge_catalog_main()

    # Always rebuild the extended catalog so the search service sees the full
    # catalog.  Without this step, all_problems_extended.json can become stale
    # and silently restrict search to a tiny subset of the catalog.
    print("Rebuilding all_problems_extended.json (merging base + custom + enrichment)...")
    from build_extended_catalog import build_extended_catalog
    out = build_extended_catalog(enrich=args.enrich, verbose=True)
    print(f"Extended catalog written to: {out}")


if __name__ == "__main__":
    main()

