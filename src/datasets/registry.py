"""Load ``data/dataset_registry.json`` (tracked metadata for external validation)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_PATH = _REPO_ROOT / "data" / "dataset_registry.json"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset registry: {_REGISTRY_PATH}")
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def get_dataset_entry(name: str) -> dict[str, Any]:
    reg = load_registry()
    key = name.strip().lower()
    ds = reg.get("datasets") or {}
    if key not in ds:
        raise KeyError(f"Unknown dataset in registry: {name}. Known: {sorted(ds)}")
    return ds[key]
