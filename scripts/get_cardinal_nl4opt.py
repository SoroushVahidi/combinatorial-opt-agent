#!/usr/bin/env python3
"""Fetch and prepare CardinalOperations/NL4OPT dataset from HuggingFace.

Source: https://huggingface.co/datasets/CardinalOperations/NL4OPT
License: CC-BY-NC-4.0 (non-commercial use only)
Splits: test (NL4OPT_with_optimal_solution.json, 245 rows)

Note: This is a distinct dataset from the existing 'nl4opt' adapter which uses
data from the nl4opt-competition GitHub repository. This version (cardinal_nl4opt)
is the CardinalOperations curation and is registered as a separate adapter.

Outputs are written to data/external/cardinal_nl4opt/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "cardinal_nl4opt"

HF_RESOLVE = "https://huggingface.co/datasets/CardinalOperations/NL4OPT/resolve/main"

SPLITS = {
    "test": f"{HF_RESOLVE}/NL4OPT_with_optimal_solution.json",
}


def _fetch(url: str, timeout: int = 30) -> str | None:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except URLError as exc:
        print(f"[error] fetch failed for {url}: {exc}", file=sys.stderr)
        return None


def _parse_jsonl(raw: str) -> list[dict]:
    rows: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download CardinalOperations/NL4OPT dataset from HuggingFace."
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict = {
        "dataset": "CardinalOperations/NL4OPT",
        "source": "https://huggingface.co/datasets/CardinalOperations/NL4OPT",
        "license": "CC-BY-NC-4.0",
        "retrieval_method": "urllib direct download via HuggingFace resolve URL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "warnings": [],
        "note": (
            "Distinct from the existing 'nl4opt' adapter (nl4opt-competition GitHub)."
            " This adapter is keyed as 'cardinal_nl4opt'."
        ),
    }

    any_success = False
    for split_name, url in SPLITS.items():
        out_path = OUT_DIR / f"{split_name}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path} already exists (use --force to overwrite)")
            rows_count = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
            provenance["splits"][split_name] = {"rows": rows_count, "status": "existing"}
            any_success = True
            continue

        raw = _fetch(url)
        if raw is None:
            msg = f"Could not fetch {split_name} from {url}"
            print(f"[error] {msg}", file=sys.stderr)
            provenance["warnings"].append(msg)
            provenance["splits"][split_name] = {"rows": 0, "status": "fetch_failed", "url": url}
            continue

        try:
            rows = _parse_jsonl(raw)
        except json.JSONDecodeError as exc:
            msg = f"JSON parse error for {split_name}: {exc}"
            print(f"[error] {msg}", file=sys.stderr)
            provenance["warnings"].append(msg)
            provenance["splits"][split_name] = {"rows": 0, "status": "parse_error"}
            continue

        if not rows:
            msg = f"No rows parsed for split {split_name}"
            print(f"[warn] {msg}", file=sys.stderr)
            provenance["warnings"].append(msg)
            provenance["splits"][split_name] = {"rows": 0, "status": "empty"}
            continue

        normalized: list[dict] = []
        for idx, row in enumerate(rows):
            normalized.append(
                {
                    "id": f"cardinal_nl4opt_{split_name}_{idx}",
                    "nl_query": (row.get("en_question") or "").strip(),
                    "en_answer": row.get("en_answer"),
                    "split": split_name,
                    "source": "CardinalOperations/NL4OPT",
                    "metadata": {"original_index": idx},
                }
            )

        with open(out_path, "w", encoding="utf-8") as fh:
            for r in normalized:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"[ok] {split_name}: wrote {len(normalized)} rows -> {out_path}")
        provenance["splits"][split_name] = {"rows": len(normalized), "status": "ok", "url": url}
        any_success = True

    prov_path = OUT_DIR / "provenance.json"
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)
    print(f"[ok] provenance written -> {prov_path}")

    if not any_success:
        print(
            "[warn] No data was downloaded. Check internet connectivity.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
