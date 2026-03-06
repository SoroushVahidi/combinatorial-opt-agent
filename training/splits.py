"""
Leak-free train/dev/test splits by problem ID.
Stratifies by source when possible so that each split has similar source distribution.
Writes a JSON file so generate_samples and evaluate_retrieval can filter by split.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_catalog(catalog_path: Path | None = None) -> list[dict]:
    """Load problem catalog; each entry must have 'id'. Uses extended if present."""
    root = _project_root()
    if catalog_path is None:
        extended = root / "data" / "processed" / "all_problems_extended.json"
        base = root / "data" / "processed" / "all_problems.json"
        catalog_path = extended if extended.exists() else base
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    with open(catalog_path, encoding="utf-8") as f:
        return json.load(f)


def build_splits(
    catalog: list[dict],
    seed: int = 42,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[str]]:
    """
    Return disjoint train/dev/test lists of problem IDs.
    Stratifies by source: within each source, problems are split by train_ratio/dev_ratio/test_ratio.
    Problems without 'id' are skipped. Problems without 'source' go to a single bucket then split.
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)

    by_source: dict[str, list[str]] = {}
    no_source: list[str] = []
    for p in catalog:
        pid = p.get("id")
        if not pid:
            continue
        src = p.get("source") or ""
        if not src:
            no_source.append(pid)
        else:
            by_source.setdefault(src, []).append(pid)

    train_ids: list[str] = []
    dev_ids: list[str] = []
    test_ids: list[str] = []

    def split_list(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        rng.shuffle(ids)
        n = len(ids)
        n_train = max(0, int(n * train_ratio))
        n_dev = max(0, int(n * dev_ratio))
        n_test = max(0, n - n_train - n_dev)
        t = ids[:n_train]
        d = ids[n_train : n_train + n_dev]
        s = ids[n_train + n_dev : n_train + n_dev + n_test]
        return t, d, s

    for src, ids in by_source.items():
        t, d, s = split_list(ids)
        train_ids.extend(t)
        dev_ids.extend(d)
        test_ids.extend(s)
    if no_source:
        t, d, s = split_list(no_source)
        train_ids.extend(t)
        dev_ids.extend(d)
        test_ids.extend(s)

    return {"train": train_ids, "dev": dev_ids, "test": test_ids}


def write_splits(
    splits: dict[str, list[str]],
    out_path: Path,
) -> None:
    """Write splits JSON: { "train": [...], "dev": [...], "test": [...] }."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Load splits from JSON. Keys: train, dev, test; values: list of problem IDs."""
    with open(splits_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "train": list(data.get("train", [])),
        "dev": list(data.get("dev", [])),
        "test": list(data.get("test", [])),
    }


def get_problem_ids_for_split(splits: dict[str, list[str]], split_name: str) -> list[str]:
    """Return list of problem IDs for the given split (train, dev, or test)."""
    name = (split_name or "").strip().lower()
    if name not in ("train", "dev", "test"):
        raise ValueError(f"split must be one of train, dev, test; got {split_name!r}")
    return list(splits.get(name, []))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Build train/dev/test splits by problem ID (stratified by source)")
    p.add_argument("--splits-out", type=Path, default=None,
                   help="Output JSON path (default: data/processed/splits.json)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--dev-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    args = p.parse_args()

    catalog = load_catalog()
    splits = build_splits(
        catalog,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )
    out = args.splits_out or _project_root() / "data" / "processed" / "splits.json"
    write_splits(splits, out)
    print(f"Wrote splits to {out}: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
