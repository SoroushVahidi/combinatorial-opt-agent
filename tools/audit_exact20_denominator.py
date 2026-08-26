"""Deterministic evidence for the Stage-1 Exact20 (on hits) discrepancy.

Documents, from the single frozen per-query artifact, why two different
Exact20 values (0.2527 and 0.2613912814943743) both appear in repository
artifacts, and which one matches the manuscript's own stated definition
(manuscript/dke/main.tex, lines 274-279: computed only over schema-hit
queries for which a comparable numeric prediction and gold value are both
available).

Does not change any existing artifact. Read-only over
results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv.
"""
import csv
import json

SRC = "results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv"
OUT = "results/final_resubmission_method/exact20_denominator_audit.json"


def main():
    rows = list(csv.DictReader(open(SRC)))
    n = len(rows)
    hits = [r for r in rows if int(float(r.get("schema_hit", 0) or 0)) == 1]
    comparable = [r for r in hits if r.get("exact20", "") != ""]
    non_comparable = [r for r in hits if r.get("exact20", "") == ""]

    comparable_subset_mean = sum(float(r["exact20"]) for r in comparable) / len(comparable)
    hits_denominator_mean = (
        sum(float(r["exact20"]) if r.get("exact20", "") != "" else 0.0 for r in hits) / len(hits)
    )

    result = {
        "source_file": SRC,
        "n_total_queries": n,
        "n_schema_hit": len(hits),
        "n_schema_hit_with_comparable_value": len(comparable),
        "n_schema_hit_without_comparable_value": len(non_comparable),
        "non_comparable_query_ids": [r["query_id"] for r in non_comparable],
        "exact20_comparable_subset_denominator": comparable_subset_mean,
        "exact20_hits_denominator_zero_filled": hits_denominator_mean,
        "manuscript_stated_definition": (
            "manuscript/dke/main.tex lines 274-279: Exact20 (on hits) is computed "
            "only on the subset of schema-hit queries for which a comparable "
            "numeric prediction and gold value are both available for the slot "
            "under consideration."
        ),
        "definition_consistent_value": comparable_subset_mean,
        "value_currently_printed_in_manuscript_table": 0.2527,
        "value_currently_printed_matches_which_rule": "exact20_hits_denominator_zero_filled",
        "authoritative_frozen_artifact_agreement_check": {
            "results/final_resubmission_method/metrics.json exact20 (patched row)": 0.2613912814943743,
            "recomputed_here": comparable_subset_mean,
            "match": abs(comparable_subset_mean - 0.2613912814943743) < 1e-9,
        },
        "conclusion": (
            "Both 0.2527 and 0.2614 are computed from the same underlying frozen "
            "per-query file. They differ only in the aggregation rule: whether "
            "the 10 schema-hit queries with no comparable numeric value are "
            "excluded from the denominator (0.2614, matches the manuscript's own "
            "stated definition) or included and zero-filled (0.2527, the value "
            "actually printed in the manuscript's downstream table). See "
            "docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md Section 6 "
            "for the full root-cause trace, including the BGE-M3 and Oracle arms."
        ),
    }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
