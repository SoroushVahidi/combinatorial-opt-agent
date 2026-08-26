"""Stage-3 replacement for the stale, heuristic 9-row error-taxonomy table
(manuscript/dke/main.tex, tab:nlp4lp-error-taxonomy), which Stage 1/2 found
to be an unattributed, unreproducible carry-over from a pre-bugfix
(2026-03-10) artifact.

This script defines a small, mutually exclusive, fully reproducible
query-level partition of the 331-query denominator, computed only from
columns that exist in the frozen post-ratio-patch per-query artifact
(results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv):
schema_hit, param_coverage, type_match, exact20. No per-slot instrumentation
exists in the current artifact, so this deliberately does NOT attempt a
fine-grained float/int/percent/currency breakdown (see the Stage-2 report for
why that would require new code + a rerun, out of scope here).

Categories (mutually exclusive, in this evaluation order, over all 331 queries):
  1. wrong_schema:            schema_hit == 0
  2. incomplete_coverage:     schema_hit == 1 AND param_coverage < 1.0
  3. type_mismatch:           schema_hit == 1 AND param_coverage == 1.0 AND type_match < 1.0
  4. value_inaccurate:        schema_hit == 1 AND param_coverage == 1.0 AND type_match == 1.0
                               AND (exact20 is comparable AND exact20 < 1.0)
  5. fully_correct:           schema_hit == 1 AND param_coverage == 1.0 AND type_match == 1.0
                               AND (exact20 is NOT comparable OR exact20 == 1.0)

Every query falls into exactly one category; counts sum to 331 by construction.
"""
import csv
import json

SRC = "results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv"
OUT = "results/final_resubmission_method/residual_error_analysis_2026-08-27.json"


def classify(r):
    hit = int(float(r.get("schema_hit", 0) or 0))
    if hit == 0:
        return "wrong_schema"
    cov = float(r.get("param_coverage", 0) or 0)
    if cov < 1.0:
        return "incomplete_coverage"
    tm = float(r.get("type_match", 0) or 0)
    if tm < 1.0:
        return "type_mismatch"
    exact20_raw = r.get("exact20", "")
    if exact20_raw != "" and float(exact20_raw) < 1.0:
        return "value_inaccurate"
    return "fully_correct"


def main():
    rows = list(csv.DictReader(open(SRC)))
    n = len(rows)
    counts = {}
    ids = {}
    for r in rows:
        c = classify(r)
        counts[c] = counts.get(c, 0) + 1
        ids.setdefault(c, []).append(r["query_id"])

    order = ["wrong_schema", "incomplete_coverage", "type_mismatch", "value_inaccurate", "fully_correct"]
    assert sum(counts.get(c, 0) for c in order) == n

    result = {
        "source_file": SRC,
        "n_total": n,
        "definition": (
            "Mutually exclusive query-level partition, evaluated in order: "
            "(1) wrong_schema if schema_hit==0; (2) incomplete_coverage if "
            "schema_hit==1 and param_coverage<1.0; (3) type_mismatch if "
            "schema_hit==1, param_coverage==1.0, type_match<1.0; (4) "
            "value_inaccurate if schema_hit==1, param_coverage==1.0, "
            "type_match==1.0, and the query has a comparable exact20 value "
            "<1.0; (5) fully_correct otherwise."
        ),
        "counts": {c: counts.get(c, 0) for c in order},
        "fractions_of_331": {c: counts.get(c, 0) / n for c in order},
        "example_query_ids": {c: ids.get(c, [])[:5] for c in order},
        "note": (
            "Categories are mutually exclusive and sum to n_total by "
            "construction (no double-counting, unlike the superseded "
            "heuristic taxonomy). This does not decompose type_mismatch or "
            "value_inaccurate into finer sub-causes (e.g. float vs percent) "
            "because the current per-query artifact has no per-slot type "
            "column; see docs/SNCS_STAGE2_LOCAL_WULVER_GITHUB_AUDIT_2026-08-27.md "
            "Section 11 for why that is deferred rather than fabricated."
        ),
    }

    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
