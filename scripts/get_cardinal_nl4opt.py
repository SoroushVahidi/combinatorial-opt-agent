#!/usr/bin/env python3
"""Fetch/prepare CardinalOperations/NL4OPT snapshots into data/external/cardinal_nl4opt.

Source: https://github.com/CardinalOperations/NL4OPT
Splits: train, dev, test

Note: This is a distinct dataset from the existing 'nl4opt' adapter which uses
data from the nl4opt-competition GitHub repository. This version (cardinal_nl4opt)
is the CardinalOperations curation and is registered as a separate adapter.

Exits non-zero with details when retrieval is blocked.
Outputs are written to data/external/cardinal_nl4opt/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "cardinal_nl4opt"
META = OUT_DIR / "provenance.json"
BASE = "https://raw.githubusercontent.com/CardinalOperations/NL4OPT/main"
SPLIT_URLS = {
    "train": f"{BASE}/generation_data/train.jsonl",
    "dev": f"{BASE}/generation_data/dev.jsonl",
    "test": f"{BASE}/generation_data/test.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/prepare Cardinal NL4OPT snapshots.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split JSONL files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    row_counts: dict[str, int] = {}
    found: list[str] = []
    method = "none"

    for split, url in SPLIT_URLS.items():
        out = OUT_DIR / f"{split}.jsonl"
        if out.exists() and not args.force:
            found.append(split)
            with out.open(encoding="utf-8") as fh:
                row_counts[split] = sum(1 for _ in fh)
            continue
        try:
            with urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            lines = [ln for ln in text.splitlines() if ln.strip()]
            # Validate JSONL payload to avoid treating HTML/error pages as successful splits.
            parsed = 0
            for ln in lines:
                try:
                    json.loads(ln)
                    parsed += 1
                except Exception:
                    parsed = -1
                    break
            if parsed <= 0:
                msg = f"{split}: fetched payload was not valid JSONL from {url}"
                print(f"[error] {msg}", file=sys.stderr)
                errors.append(msg)
                continue
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            row_counts[split] = parsed
            found.append(split)
            method = "direct_raw_http"
            print(f"[ok] {split}: {parsed} rows -> {out}")
        except HTTPError as e:
            msg = f"{split}: HTTPError {e.code} from {url}"
            print(f"[error] {msg}", file=sys.stderr)
            errors.append(msg)
        except URLError as e:
            msg = f"{split}: URLError {e.reason} from {url}"
            print(f"[error] {msg}", file=sys.stderr)
            errors.append(msg)
        except Exception as e:
            msg = f"{split}: unexpected error: {e}"
            print(f"[error] {msg}", file=sys.stderr)
            errors.append(msg)

    # If everything was preexisting and no network was used, reflect that in provenance.
    if method == "none" and found:
        method = "preexisting_local_files"

    provenance = {
        "dataset": "CardinalOperations/NL4OPT",
        "source": "CardinalOperations/NL4OPT",
        "source_url": "https://github.com/CardinalOperations/NL4OPT",
        "note": (
            "Distinct from the existing 'nl4opt' adapter (nl4opt-competition GitHub)."
            " This adapter is keyed as 'cardinal_nl4opt'."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_method": method,
        "splits": sorted(found),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    META.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[ok] provenance written -> {META}")

    missing = sorted(set(SPLIT_URLS) - set(found))
    if missing:
        print(json.dumps(provenance, indent=2))
        print(f"[warn] Could not fetch Cardinal NL4OPT splits: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] Cardinal NL4OPT ready: {sorted(found)}")
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
