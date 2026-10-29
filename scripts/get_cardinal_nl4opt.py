#!/usr/bin/env python3
"""Stage raw CardinalOperations/NL4OPT JSONL snapshots (not competition NL4Opt).

Only an explicit CLI invocation downloads data. Existing files are validated and
reused unless --force is given. Every payload must contain nonempty JSON objects;
failed downloads never replace an existing split. Upstream fields are preserved,
not converted into inferred schema, scalar-gold or formulation annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "cardinal_nl4opt"
BASE = "https://raw.githubusercontent.com/CardinalOperations/NL4OPT/main"
SPLIT_URLS = {split: f"{BASE}/generation_data/{split}.jsonl" for split in ("train", "dev", "test")}


def validate_jsonl(text: str) -> int:
    """Count nonblank object rows; reject empty, malformed or non-object data."""
    count = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        if not isinstance(row, dict) or not row:
            raise ValueError(f"line {line_no}: expected a nonempty JSON object")
        count += 1
    if not count:
        raise ValueError("empty JSONL payload")
    return count


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace only after the complete validated content has been written."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fh:
            temp_path = Path(fh.name)
            fh.write(data)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare(output_dir: Path, *, force: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, errors = {}, []
    for split, url in SPLIT_URLS.items():
        path = output_dir / f"{split}.jsonl"
        try:
            reused = path.exists() and not force
            if reused:
                payload = path.read_bytes()
            else:
                with urlopen(url, timeout=30) as response:
                    payload = response.read()
            count = validate_jsonl(payload.decode("utf-8-sig"))
            if not reused:
                _atomic_write(path, payload)
            records[split] = {"rows": count, "status": "existing" if reused else "downloaded",
                              "url": url, "sha256": hashlib.sha256(payload).hexdigest()}
        except (OSError, URLError, ValueError) as exc:
            errors.append(f"{split}: {type(exc).__name__}: {exc}")
    report = {
        "dataset": "CardinalOperations/NL4OPT",
        "source_url": "https://github.com/CardinalOperations/NL4OPT",
        "note": "Raw snapshots; distinct from the nl4opt-competition dataset. No inferred labels added.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": sorted(records), "row_counts": {s: r["rows"] for s, r in records.items()},
        "files": records, "errors": errors,
    }
    # Preserve prior provenance along with reused local files. An explicit force
    # request may refresh it; callers always receive the current validation report.
    provenance = output_dir / "provenance.json"
    if not provenance.exists() or force:
        _atomic_write(provenance, (json.dumps(report, indent=2) + "\n").encode())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Explicitly replace valid split files and provenance.")
    args = parser.parse_args(argv)
    try:
        report = prepare(args.output_dir, force=args.force)
    except OSError as exc:
        print(f"[error] Cannot write staging directory: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return int(bool(report["errors"]) or set(report["splits"]) != set(SPLIT_URLS))


if __name__ == "__main__":
    raise SystemExit(main())
