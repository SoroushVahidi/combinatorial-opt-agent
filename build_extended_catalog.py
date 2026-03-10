from __future__ import annotations

"""
Utility to extend the core problem catalog with additional problems.

Usage (from project root):

    python build_extended_catalog.py

It will:
- Read data/processed/all_problems.json  (existing core problems)
- Read data/processed/custom_problems.json  (optional, you create/fill this)
- Merge them by id (custom entries override on id collision)
- Write data/processed/all_problems_extended.json

Optional --enrich flag:
    python build_extended_catalog.py --enrich

When --enrich is given, the script additionally:
- Scans the base catalog for incomplete problems (missing variables / objective
  / constraints in their formulation).
- Attempts to fetch the missing formulation data from public web sources
  (currently: Gurobi modeling-examples Jupyter notebooks on GitHub).
- Treats successfully enriched entries exactly like custom entries (they
  override the corresponding base-catalog entry by id).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from retrieval.catalog_enrichment import enrich_catalog as _enrich_catalog


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_extended_catalog(enrich: bool = False, verbose: bool = False) -> Path:
    root = _project_root()
    data_dir = root / "data" / "processed"

    base_path = data_dir / "all_problems.json"
    if not base_path.exists():
        raise FileNotFoundError(f"Base catalog not found: {base_path}")

    base_catalog: List[Dict] = _load_json(base_path)

    custom_path = data_dir / "custom_problems.json"
    custom_catalog: List[Dict] = []
    if custom_path.exists():
        custom_catalog = _load_json(custom_path)

    # Optionally enrich incomplete entries from the web
    enriched_entries: List[Dict] = []
    if enrich:
        enriched_entries = _enrich_catalog(base_catalog, verbose=verbose)

    # Merge by id; priority (highest → lowest):
    #   custom_problems.json > web-enriched > base catalog
    by_id: Dict[str, Dict] = {}
    for entry in base_catalog:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        by_id[entry_id] = entry

    for entry in enriched_entries:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        by_id[entry_id] = entry

    for entry in custom_catalog:
        entry_id = entry.get("id")
        if not entry_id:
            continue
        by_id[entry_id] = entry

    merged: List[Dict] = list(by_id.values())

    out_path = data_dir / "all_problems_extended.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the extended problem catalog."
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help=(
            "Fetch missing formulation data from the web for incomplete "
            "catalog entries before merging."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress messages during enrichment.",
    )
    args = parser.parse_args()
    out_path = build_extended_catalog(enrich=args.enrich, verbose=args.verbose)
    print(f"Wrote extended catalog to: {out_path}")


if __name__ == "__main__":
    main()

