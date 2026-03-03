"""
Merge collected problem sources into data/processed/all_problems.json.
Keeps existing problems; adds new ones from data/raw/* and collectors.
Run after collect_nl4opt (or other collectors) to update the bot's catalog.
"""
from __future__ import annotations

import json
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_existing_catalog() -> list[dict]:
    path = _project_root() / "data" / "processed" / "all_problems.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw_nl4opt() -> list[dict]:
    raw_dir = _project_root() / "data" / "raw" / "nl4opt"
    out = []
    for name in ("train", "dev", "test"):
        p = raw_dir / f"{name}.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            out.extend(json.load(f))
    return out


def load_raw_optmath() -> list[dict]:
    path = _project_root() / "data" / "raw" / "optmath" / "bench.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_raw_classic_extra() -> list[dict]:
    path = _project_root() / "data" / "raw" / "classic_extra.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_catalog(existing: list[dict], new_problems: list[dict]) -> tuple[list[dict], int]:
    """Merge new problems into existing; skip if id already present."""
    ids = {p["id"] for p in existing}
    merged = list(existing)
    added = 0
    for p in new_problems:
        if p.get("id") and p["id"] not in ids:
            ids.add(p["id"])
            merged.append(p)
            added += 1
    return merged, added


def main() -> None:
    root = _project_root()
    catalog_path = root / "data" / "processed" / "all_problems.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_catalog()
    all_new = []
    all_new.extend(load_raw_nl4opt())
    all_new.extend(load_raw_optmath())
    all_new.extend(load_raw_classic_extra())
    merged, added = merge_catalog(existing, all_new)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Catalog: {len(existing)} existing, +{added} new -> {len(merged)} total at {catalog_path}")


if __name__ == "__main__":
    main()
