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
"""

import json
from pathlib import Path
from typing import Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_extended_catalog() -> Path:
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

    # Merge by id; later entries override earlier ones on id collision.
    by_id: Dict[str, Dict] = {}
    for entry in base_catalog:
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
    out_path = build_extended_catalog()
    print(f"Wrote extended catalog to: {out_path}")


if __name__ == "__main__":
    main()

