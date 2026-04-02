#!/usr/bin/env python3
"""Fetch/prepare MAMO dataset snapshots into data/external/mamo.

This script is offline-friendly:
- it first looks for pre-existing local files,
- then attempts network retrieval,
- always writes a provenance report,
- exits non-zero with exact blocker details when retrieval is blocked.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "mamo"
META_PATH = OUT_DIR / "provenance.json"

DATA_URLS = {
    "train": "https://raw.githubusercontent.com/FreedomIntelligence/Mamo/main/benchmark/MAMO_train.json",
    "validation": "https://raw.githubusercontent.com/FreedomIntelligence/Mamo/main/benchmark/MAMO_valid.json",
    "test": "https://raw.githubusercontent.com/FreedomIntelligence/Mamo/main/benchmark/MAMO_test.json",
}


def _write_metadata(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _count_json_rows(path: Path) -> int | None:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        if isinstance(obj, list):
            return len(obj)
    except Exception:
        return None
    return None


def _attempt_git_clone(tmp_dir: Path) -> tuple[bool, str]:
    if shutil.which("git") is None:
        return False, "git binary not available"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/FreedomIntelligence/Mamo.git", str(tmp_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True, "git clone"
    except subprocess.CalledProcessError as e:
        return False, f"git clone failed: {e.stderr.strip() or e.stdout.strip()}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/prepare MAMO snapshots.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split JSONL files.")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    warnings: list[str] = []
    errors: list[str] = []
    row_counts: dict[str, int | None] = {}
    splits_found: list[str] = []
    retrieval_method = "none"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in DATA_URLS:
        if (OUT_DIR / f"{split}.jsonl").exists() and not args.force:
            splits_found.append(split)
            with open(OUT_DIR / f"{split}.jsonl", encoding="utf-8") as fh:
                row_counts[split] = sum(1 for _ in fh)

    if splits_found and len(splits_found) == len(DATA_URLS):
        retrieval_method = "preexisting_local_files"
    else:
        # First attempt direct file retrieval from likely raw URLs.
        for split, url in DATA_URLS.items():
            out_path = OUT_DIR / f"{split}.jsonl"
            if out_path.exists() and not args.force:
                continue
            try:
                with urlopen(url, timeout=30) as resp:
                    payload = resp.read().decode("utf-8")
                src_path = OUT_DIR / f"{split}.source.json"
                src_path.write_text(payload, encoding="utf-8")
                src_rows = _count_json_rows(src_path)
                success = False
                with open(out_path, "w", encoding="utf-8") as out_fh:
                    if src_rows is not None:
                        # JSON list; normalize to JSONL
                        for row in json.loads(payload):
                            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        row_counts[split] = src_rows
                        success = True
                    else:
                        # Validate as JSONL: each non-empty line must be valid JSON
                        valid_rows = 0
                        for line in payload.splitlines():
                            stripped = line.strip()
                            if not stripped:
                                continue
                            try:
                                json.loads(stripped)
                            except json.JSONDecodeError:
                                valid_rows = 0
                                break
                            else:
                                out_fh.write(stripped + "\n")
                                valid_rows += 1
                        if valid_rows > 0:
                            row_counts[split] = valid_rows
                            success = True
                if success:
                    splits_found.append(split)
                    retrieval_method = "direct_raw_http"
                else:
                    snippet = payload[:100].replace("\n", " ")
                    errors.append(
                        f"{split}: non-JSON response from {url} (expected JSON list or JSONL); "
                        f"response starts with: {snippet!r}"
                    )
                    try:
                        out_path.unlink()
                    except FileNotFoundError:
                        pass
            except HTTPError as e:
                errors.append(f"{split}: HTTPError {e.code} from {url}")
            except URLError as e:
                errors.append(f"{split}: URLError {e.reason} from {url}")
            except Exception as e:
                errors.append(f"{split}: unexpected error from {url}: {e}")

        # Fallback: clone repo and search benchmark directory
        if len(splits_found) < len(DATA_URLS):
            tmp_dir = OUT_DIR / "downloads" / "mamo_repo"
            tmp_dir.parent.mkdir(parents=True, exist_ok=True)
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            ok, method = _attempt_git_clone(tmp_dir)
            if ok:
                retrieval_method = method
                benchmark_dir = tmp_dir / "benchmark"
                if benchmark_dir.exists():
                    for split in DATA_URLS:
                        if split in splits_found:
                            continue
                        candidates = list(benchmark_dir.glob(f"*{split}*.json")) + list(benchmark_dir.glob(f"*{split}*.jsonl"))
                        if not candidates:
                            warnings.append(f"{split}: no file matching *{split}*.json/.jsonl under benchmark/")
                            continue
                        src = sorted(candidates)[0]
                        out = OUT_DIR / f"{split}.jsonl"
                        text = src.read_text(encoding="utf-8")
                        try:
                            obj = json.loads(text)
                            with open(out, "w", encoding="utf-8") as fh:
                                if isinstance(obj, list):
                                    for row in obj:
                                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                                else:
                                    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            with open(out, encoding="utf-8") as fh:
                                row_counts[split] = sum(1 for _ in fh)
                            splits_found.append(split)
                        except Exception:
                            out.write_text(text, encoding="utf-8")
                            with open(out, encoding="utf-8") as fh:
                                row_counts[split] = sum(1 for _ in fh)
                            splits_found.append(split)
                else:
                    warnings.append("cloned Mamo repo but benchmark/ directory missing")
            else:
                errors.append(method)

    payload = {
        "source": "FreedomIntelligence/Mamo",
        "source_url": "https://github.com/FreedomIntelligence/Mamo",
        "retrieval_method": retrieval_method,
        "timestamp_utc": now,
        "splits": sorted(set(splits_found)),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    _write_metadata(payload)

    missing = sorted(set(DATA_URLS) - set(splits_found))
    if missing:
        print(json.dumps(payload, indent=2))
        raise SystemExit(f"Blocked: could not prepare MAMO splits: {missing}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
