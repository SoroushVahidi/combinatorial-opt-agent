"""Partial, deterministic recount of the manuscript's error-taxonomy row
"Wrong type assignment (mainly float-related) = 230" against the frozen,
post-ratio-patch per-query artifact.

Stage-1 (docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md, Section 8)
established that this row (and 6 of the other 8 rows in
manuscript/dke/main.tex Table tab:nlp4lp-error-taxonomy) is a verbatim,
unattributed copy of results/eswa_revision/13_tables/error_taxonomy_counts.csv
(dated 2026-03-10), which predates both the `_is_type_match` bugfix
(2026-03-14) and the August-2026 ratio-aware extraction patch. TypeMatch was
~0.227 when that table was generated; it is 0.8665 in the frozen pipeline.

This script recomputes only the coarse aggregate ("how many schema-hit
queries have a type mismatch") from the frozen per-query CSV. It does NOT
recompute the finer 6-way breakdown (wrong slot disambiguation, total-vs-unit
confusion, min/max inversion, percent-vs-absolute, float ambiguity), because
the current per-query artifact has no per-slot type column -- that would
require new instrumentation in the grounding pipeline and a fresh (though
still cheap, deterministic, CPU-only) rerun, which is out of scope for this
evidence-preservation stage. See the Stage-2 report for the exact Stage-3
recommendation.
"""
import csv
import json

SRC = "results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv"
OUT = "results/final_resubmission_method/type_mismatch_recount_2026-08-27.json"


def main():
    rows = list(csv.DictReader(open(SRC)))
    hits = [r for r in rows if int(float(r.get("schema_hit", 0) or 0)) == 1]

    def is_not_ready(r):
        return not (
            float(r.get("param_coverage", 0) or 0) >= 0.8
            and float(r.get("type_match", 0) or 0) >= 0.8
        )

    def has_type_mismatch(r):
        return float(r.get("type_match", 0) or 0) < 1.0

    schema_hit_not_ready = [r for r in hits if is_not_ready(r)]
    schema_hit_type_mismatch = [r for r in hits if has_type_mismatch(r)]
    schema_hit_not_ready_type_mismatch = [
        r for r in schema_hit_not_ready if has_type_mismatch(r)
    ]

    result = {
        "source_file": SRC,
        "manuscript_table_row": "Wrong type assignment (mainly float-related)",
        "manuscript_value": 230,
        "manuscript_value_source": (
            "results/eswa_revision/13_tables/error_taxonomy_counts.csv "
            "(dated 2026-03-10, pre-`_is_type_match`-fix, pre-ratio-patch; "
            "explicitly disowned as stale by docs/CURRENT_BOTTLENECK_ANALYSIS.md)"
        ),
        "n_schema_hit": len(hits),
        "n_schema_hit_with_any_type_mismatch": len(schema_hit_type_mismatch),
        "n_schema_hit_not_ready_with_type_mismatch": len(schema_hit_not_ready_type_mismatch),
        "recomputed_current_estimate": len(schema_hit_not_ready_type_mismatch),
        "independent_corroboration": (
            "docs/METHOD_NOVELTY_EFFICIENCY_AUDIT_2026-08-13.md Section 3 "
            "(fresh, same-week diagnostic on the prepatch rerun) reports 30-33 "
            "type/unit-related failures among schema-hit, not-ready queries -- "
            "consistent order of magnitude with this recount."
        ),
        "caveat": (
            "This is a coarse aggregate only (schema_hit=1 AND type_match<1, "
            "restricted to not-ready queries to match the manuscript row's "
            "intent). It does NOT establish that 'float' specifically dominates "
            "-- the current per-query artifact has no per-slot expected-type "
            "column, so the float/int/percent/currency attribution in the "
            "manuscript's parenthetical cannot be verified or falsified from "
            "existing frozen artifacts. A true per-type breakdown requires new "
            "per-slot instrumentation in the grounding pipeline (a small, "
            "deterministic, CPU-only code change and rerun -- not a new "
            "experiment requiring model/API inference -- but out of scope for "
            "this evidence-preservation stage)."
        ),
        "conclusion": (
            f"The manuscript's figure of 230 is off by roughly "
            f"{230 / max(len(schema_hit_not_ready_type_mismatch), 1):.1f}x "
            "relative to the frozen post-patch pipeline and must not be used "
            "as current evidence."
        ),
    }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
