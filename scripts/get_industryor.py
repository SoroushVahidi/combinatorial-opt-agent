#!/usr/bin/env python3
"""Fetch/prepare IndustryOR snapshots into data/external/industryor.

Source: https://github.com/CardinalOperations/IndustryOR
Splits: train, dev, validation, test

This script is offline-friendly:
- it first looks for pre-existing local files,
- then attempts git clone,
- always writes a provenance report,
- exits non-zero when retrieval is blocked.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "industryor"
META = OUT_DIR / "provenance.json"
REPO_URL = "https://github.com/CardinalOperations/IndustryOR"
KNOWN_SPLITS = ("train", "dev", "validation", "test")


def _collect_json_candidates(repo_dir: Path, split: str) -> list[Path]:
    return sorted(list(repo_dir.rglob(f"*{split}*.json")) + list(repo_dir.rglob(f"*{split}*.jsonl")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/prepare IndustryOR snapshots.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing split JSONL files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    errors: list[str] = []
    row_counts: dict[str, int] = {}
    found: list[str] = []
    method = "none"

    for split in KNOWN_SPLITS:
        p = OUT_DIR / f"{split}.jsonl"
        if p.exists() and not args.force:
            found.append(split)
            with p.open(encoding="utf-8") as fh:
                row_counts[split] = sum(1 for _ in fh)

    if len(found) == len(KNOWN_SPLITS):
        method = "preexisting_local_files"
    else:
        if shutil.which("git") is None:
            msg = "git binary unavailable for clone"
            print(f"[error] {msg}", file=sys.stderr)
            errors.append(msg)
        else:
            repo_dir = OUT_DIR / "downloads" / "industryor_repo"
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                method = "git_clone"
                for split in KNOWN_SPLITS:
                    if split in found:
                        continue
                    cands = _collect_json_candidates(repo_dir, split)
                    if not cands:
                        msg = f"{split}: no matching split file found in repo"
                        print(f"[warn] {msg}", file=sys.stderr)
                        warnings.append(msg)
                        continue
                    src = cands[0]
                    text = src.read_text(encoding="utf-8")
                    out = OUT_DIR / f"{split}.jsonl"
                    lines: list[str] = []
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, list):
                            lines = [json.dumps(r, ensure_ascii=False) for r in obj]
                        else:
                            lines = [json.dumps(obj, ensure_ascii=False)]
                    except Exception:
                        lines = [ln for ln in text.splitlines() if ln.strip()]
                    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    row_counts[split] = len(lines)
                    found.append(split)
                    print(f"[ok] {split}: {len(lines)} rows -> {out}")
            except subprocess.CalledProcessError as e:
                msg = f"git clone failed: {e.stderr.strip() or e.stdout.strip()}"
                print(f"[error] {msg}", file=sys.stderr)
                errors.append(msg)

    provenance = {
        "dataset": "IndustryOR",
        "source": "CardinalOperations/IndustryOR",
        "source_url": REPO_URL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "retrieval_method": method,
        "splits": sorted(set(found)),
        "row_counts": row_counts,
        "warnings": warnings,
        "errors": errors,
    }
    META.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[ok] provenance written -> {META}")

    missing = sorted(set(KNOWN_SPLITS) - set(found))
    if missing:
        print(json.dumps(provenance, indent=2))
        print(f"[warn] Could not prepare IndustryOR splits: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] IndustryOR ready: {sorted(set(found))}")

import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external" / "industryor"

HF_RESOLVE = "https://huggingface.co/datasets/CardinalOperations/IndustryOR/resolve/main"

SPLITS = {
    "test": f"{HF_RESOLVE}/IndustryOR.json",
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
    ap = argparse.ArgumentParser(description="Download IndustryOR dataset from HuggingFace.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provenance: dict = {
        "dataset": "IndustryOR",
        "source": "https://huggingface.co/datasets/CardinalOperations/IndustryOR",
        "license": "CC-BY-NC-4.0",
        "retrieval_method": "urllib direct download via HuggingFace resolve URL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "splits": {},
        "warnings": [],
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
            row_id = row.get("id")
            if not row_id:
                row_id = f"industryor_{split_name}_{idx}"
            normalized.append(
                {
                    "id": str(row_id),
                    "nl_query": (row.get("en_question") or "").strip(),
                    "en_answer": row.get("en_answer"),
                    "difficulty": row.get("difficulty"),
                    "split": split_name,
                    "source": "CardinalOperations/IndustryOR",
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
