"""
Master pipeline: run all collectors and merge into data/processed/all_problems.json.
Usage: python pipeline/run_collection.py [--max-per-file N]
  --max-per-file N  Limit NL4Opt to N records per file (default: all).
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
    args = p.parse_args()

    print("Collecting NL4Opt (natural language + LP)...")
    collect_nl4opt(max_per_file=args.max_per_file)

    print("Collecting OptMATH benchmark (NL optimization descriptions)...")
    collect_optmath(max_items=args.max_optmath)

    print("Merging into catalog (NL4Opt + OptMATH + classic_extra)...")
    merge_catalog_main()


if __name__ == "__main__":
    main()
