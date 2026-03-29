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


def _slug_to_name(slug: str) -> str:
    """Convert slug like 'min_cost_flow' to 'Min Cost Flow'."""
    return slug.replace("_", " ").strip().title()


def load_raw_gurobi_optimods() -> list[dict]:
    path = _project_root() / "data" / "sources" / "gurobi_optimods.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mods = data.get("mods") or []
    return [
        {
            "id": f"gurobi_optimods_{m}",
            "name": _slug_to_name(m),
            "description": f"Gurobi OptiMods: {_slug_to_name(m)}. See {data.get('url', '')}.",
            "source": "gurobi_optimods",
        }
        for m in mods
    ]


def load_raw_gurobi_modeling_examples() -> list[dict]:
    path = _project_root() / "data" / "sources" / "gurobi_modeling_examples.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    folders = data.get("example_folders") or []
    return [
        {
            "id": f"gurobi_ex_{f}",
            "name": _slug_to_name(f),
            "description": f"Gurobi modeling example: {_slug_to_name(f)}. Jupyter notebook and formulation from {data.get('url', '')}.",
            "source": "gurobi_modeling_examples",
        }
        for f in folders
    ]


def load_raw_or_library() -> list[dict]:
    path = _project_root() / "data" / "sources" / "or_library.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    families = data.get("problem_families") or []
    return [
        {
            "id": f"or_lib_{f}",
            "name": _slug_to_name(f),
            "description": f"OR-Library (J.E. Beasley): {_slug_to_name(f)}. Test instances and problem family. {data.get('url', '')}.",
            "source": "or_library",
        }
        for f in families
    ]


def load_raw_gams() -> list[dict]:
    path = _project_root() / "data" / "sources" / "gams_models.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    models = data.get("models") or []
    return [
        {
            "id": f"gams_{m}",
            "name": _slug_to_name(m),
            "description": f"GAMS Model Library: {_slug_to_name(m)}. {data.get('url', '')}.",
            "source": "gams",
        }
        for m in models
    ]


def load_raw_miplib() -> list[dict]:
    path = _project_root() / "data" / "sources" / "miplib.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries") or []
    for e in entries:
        e.setdefault("source", "miplib")
    return entries


def load_raw_pyomo() -> list[dict]:
    path = _project_root() / "data" / "sources" / "pyomo_examples.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = data.get("examples") or []
    return [
        {
            "id": f"pyomo_{ex}",
            "name": _slug_to_name(ex),
            "description": f"Pyomo example: {_slug_to_name(ex)}. Open-source optimization model from {data.get('url', '')}.",
            "source": "pyomo",
        }
        for ex in examples
    ]


def merge_catalog(existing: list[dict], new_problems: list[dict]) -> tuple[list[dict], int]:
    """Merge new problems into existing; skip if id already present."""
    ids = {p.get("id") for p in existing if p.get("id")}
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
    all_new.extend(load_raw_gurobi_optimods())
    all_new.extend(load_raw_gurobi_modeling_examples())
    all_new.extend(load_raw_or_library())
    all_new.extend(load_raw_gams())
    all_new.extend(load_raw_miplib())
    all_new.extend(load_raw_pyomo())
    merged, added = merge_catalog(existing, all_new)
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Catalog: {len(existing)} existing, +{added} new -> {len(merged)} total at {catalog_path}")


if __name__ == "__main__":
    main()
