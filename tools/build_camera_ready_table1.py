"""Regenerate results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv.

Phase 1 of the 2026-08-11 repository-polish effort found that this table's
Coverage/TypeMatch/InstantiationReady columns were stale (populated from a
pre-correction intermediate snapshot); `manuscript/main.tex` documents that
the correct values were regenerated from live per-query artifacts during
final KAIS preparation. This script reconstructs the table from those two
already-verified, already-correct canonical sources instead of hand-typing
replacement numbers:

- Coverage / TypeMatch / Exact20_on_hits / InstantiationReady come from
  `results/eswa_revision/13_tables/postfix_main_metrics.csv` (`variant=orig`),
  which is itself produced by `training/external/run_full_downstream_benchmark.py`.
- Schema_R1 (unaffected by the staleness bug -- verified unchanged against
  `manuscript/main.tex`'s retrieval section) comes from
  `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`,
  as the original table already did.

Preserves the exact row/column shape of the original table (4 rows: TF-IDF,
BM25, Oracle in the "core" group, plus a duplicate TF-IDF "best_downstream"
row) so that `tools/build_eaai_camera_ready_figures.py`'s figure2 build,
which reads this exact schema, is unaffected in shape -- only the numeric
values and `source_file` provenance column change.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "paper" / "eaai_camera_ready_tables"
POSTFIX_CSV = ROOT / "results" / "eswa_revision" / "13_tables" / "postfix_main_metrics.csv"
RETRIEVAL_CSV = ROOT / "results" / "eswa_revision" / "13_tables" / "deterministic_method_comparison_orig.csv"
OUTPUT_CSV = TABLE_DIR / "table1_main_benchmark_summary.csv"

METHODS = [
    ("TF-IDF", "tfidf_typed_greedy", "core"),
    ("BM25", "bm25_typed_greedy", "core"),
    ("Oracle schema", "oracle_typed_greedy", "core"),
    ("Best downstream grounding used in final paper (TF-IDF + typed_greedy)", "tfidf_typed_greedy", "best_downstream"),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    postfix_rows = {r["method"]: r for r in _read_csv(POSTFIX_CSV) if r["variant"] == "orig"}
    retrieval_rows = {r["method"]: r for r in _read_csv(RETRIEVAL_CSV)}

    out_rows = []
    for label, method_key, group in METHODS:
        postfix = postfix_rows[method_key]
        retrieval = retrieval_rows[method_key]
        out_rows.append(
            {
                "method_label": label,
                "method_key": method_key,
                "group": group,
                "schema_retrieval_r1": retrieval["Schema_R1"],
                "coverage_metric": f"{float(postfix['Coverage']):.4f}",
                "type_match_metric": f"{float(postfix['TypeMatch']):.4f}",
                "instantiation_ready": f"{float(postfix['InstReady']):.4f}",
                "source_file": "results/eswa_revision/13_tables/postfix_main_metrics.csv (Coverage/TypeMatch/InstReady); "
                "results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv (Schema_R1, unaffected by the correction)",
            }
        )

    fieldnames = list(out_rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"wrote {OUTPUT_CSV}")
    for row in out_rows:
        print(
            f"  {row['method_label']}: R@1={row['schema_retrieval_r1']} "
            f"Coverage={row['coverage_metric']} TypeMatch={row['type_match_metric']} "
            f"InstReady={row['instantiation_ready']}"
        )


if __name__ == "__main__":
    main()
