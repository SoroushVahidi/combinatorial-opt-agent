#!/usr/bin/env python3
"""Run deterministic + LLM baselines on NLP4LP variants and print/export comparison table.

This script does not alter metric definitions. It uses existing summary rows from
tools/nlp4lp_downstream_utility.py and maps:
  Top-1 Schema Acc    <- schema_R1
  InstantiationReady  <- instantiation_ready
  TypeConsistency     <- type_match
  SlotCoverage        <- param_coverage
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import nlp4lp_downstream_utility as dutil


METHODS = ["tfidf", "bm25", "lsa", "openai", "gemini"]
VARIANTS = ["orig", "noisy", "short"]


def _fmt(x: str | float | int) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _load_summary(summary_path: Path) -> list[dict]:
    if not summary_path.exists():
        return []
    with open(summary_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run and compare deterministic + LLM baselines on NLP4LP variants")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "paper")
    ap.add_argument("--skip-run", action="store_true", help="Only read existing summary CSV and produce comparison table")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "nlp4lp_downstream_summary.csv"
    table_path = out_dir / "nlp4lp_llm_baseline_comparison.csv"

    if not args.skip_run:
        dutil._apply_low_resource_env()
        catalog_path = ROOT / "data" / "catalogs" / "nlp4lp_catalog.jsonl"
        catalog, _ = dutil._load_catalog_as_problems(catalog_path)
        doc_ids = [p["id"] for p in catalog if p.get("id")]
        gold_by_id = dutil._load_hf_gold(split="test")
        for variant in VARIANTS:
            eval_path = ROOT / "data" / "processed" / f"nlp4lp_eval_{variant}.jsonl"
            eval_items = dutil._load_eval(eval_path)
            for method in METHODS:
                # Optional key checks for LLM methods.
                if method == "openai" and not os.environ.get("OPENAI_API_KEY"):
                    print("Skipping openai: OPENAI_API_KEY not set", file=sys.stderr)
                    continue
                if method == "gemini" and not os.environ.get("GEMINI_API_KEY"):
                    print("Skipping gemini: GEMINI_API_KEY not set", file=sys.stderr)
                    continue
                print(f"Running variant={variant} method={method}")
                ok = dutil.run_single_setting(
                    variant=variant,
                    baseline_arg=method,
                    assignment_mode="typed",
                    out_dir=out_dir,
                    eval_items=eval_items,
                    gold_by_id=gold_by_id,
                    catalog=catalog,
                    doc_ids=doc_ids,
                )
                if not ok:
                    print(f"Warning: failed variant={variant} method={method}", file=sys.stderr)

    rows = _load_summary(summary_path)
    by_key = {(r.get("variant", ""), r.get("baseline", "")): r for r in rows}
    out_rows: list[dict] = []
    for v in VARIANTS:
        for m in METHODS:
            r = by_key.get((v, m), {})
            out_rows.append(
                {
                    "Variant": v,
                    "Method": m.upper() if m in ("bm25", "lsa", "tfidf") else m.capitalize(),
                    "Top-1 Schema Acc": r.get("schema_R1", ""),
                    "InstantiationReady": r.get("instantiation_ready", ""),
                    "TypeConsistency": r.get("type_match", ""),
                    "SlotCoverage": r.get("param_coverage", ""),
                }
            )

    cols = [
        "Variant",
        "Method",
        "Top-1 Schema Acc",
        "InstantiationReady",
        "TypeConsistency",
        "SlotCoverage",
    ]
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {table_path}")

    for v in VARIANTS:
        print(f"\nVariant: {v}")
        print("Method | Top-1 Schema Acc | InstantiationReady | TypeConsistency | SlotCoverage")
        print("------ | ---------------- | ------------------ | --------------- | ------------")
        for m in METHODS:
            r = next((x for x in out_rows if x["Variant"] == v and x["Method"].lower() == m), None)
            if not r:
                continue
            print(
                f"{r['Method']} | {_fmt(r['Top-1 Schema Acc'])} | {_fmt(r['InstantiationReady'])} | "
                f"{_fmt(r['TypeConsistency'])} | {_fmt(r['SlotCoverage'])}"
            )


if __name__ == "__main__":
    main()

