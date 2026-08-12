"""Regenerate results/eswa_revision/13_tables/robustness_by_variant.csv.

Phase 3 (2026-08-12) found this table's `orig` rows were stale (same
pre-correction snapshot fixed for table1_main_benchmark_summary.csv in
Phase 2/3 -- see docs/ALGORITHM_IMPROVEMENT_ROADMAP.md and
results/CANONICAL_RESULTS.md). No dedicated generator script for this table
was found in the repository; this script reconstructs it deterministically
from the canonical corrected source
(results/eswa_revision/13_tables/postfix_main_metrics.csv) for the same
four methods the original file covered, instead of hand-typing corrected
values.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = ROOT / "results" / "eswa_revision" / "13_tables" / "postfix_main_metrics.csv"
OUTPUT_CSV = ROOT / "results" / "eswa_revision" / "13_tables" / "robustness_by_variant.csv"

METHODS = [
    "tfidf_typed_greedy",
    "tfidf_optimization_role_repair",
    "tfidf_hierarchical_acceptance_rerank",
    "oracle_typed_greedy",
]
VARIANTS = ["orig", "noisy", "short"]


def _fmt(value: str) -> str:
    if value in ("", "None"):
        return "N/A"
    return f"{float(value):.4f}".rstrip("0").rstrip(".") if "." in value else value


def main() -> None:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as fh:
        source_rows = {(r["method"], r["variant"]): r for r in csv.DictReader(fh)}

    out_rows = []
    for method in METHODS:
        for variant in VARIANTS:
            row = source_rows.get((method, variant))
            if row is None:
                raise SystemExit(f"Missing source row for method={method} variant={variant} in {SOURCE_CSV}")
            out_rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "Coverage": _fmt(row["Coverage"]),
                    "TypeMatch": _fmt(row["TypeMatch"]),
                    "Exact20": _fmt(row["Exact20_on_hits"]),
                    "InstReady": _fmt(row["InstReady"]),
                    "source": "measured",
                }
            )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["method", "variant", "Coverage", "TypeMatch", "Exact20", "InstReady", "source"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {OUTPUT_CSV}")
    for row in out_rows:
        print(f"  {row['method']:38s} {row['variant']:6s} Coverage={row['Coverage']} TypeMatch={row['TypeMatch']} InstReady={row['InstReady']}")


if __name__ == "__main__":
    main()
