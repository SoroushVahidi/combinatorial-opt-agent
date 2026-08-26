"""Stage-3 correction: recompute Exact20 (on hits) for all four downstream
arms (TFIDF baseline/prepatch, TFIDF ratio-aware/patched, BGE-M3 ratio-aware,
Oracle-TG) under a single, uniform, manuscript-definition-consistent rule:
mean of the per-query `exact20` value over schema-hit queries that have a
comparable numeric prediction/gold pair, EXCLUDING schema-hit queries with no
comparable value (rather than zero-filling them).

This replaces the previously inconsistent manuscript table, where TF-IDF and
BGE-M3 used a hits-denominator, zero-filled rule (0.2527, 0.2358) while
Oracle alone used the comparable-subset rule (0.2505). See
docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md Section 6/7 and
tools/audit_exact20_denominator.py for the root-cause trace.

Inputs (all frozen, already-committed artifacts; no new inference):
  - results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv (prepatch TF-IDF)
  - results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv (patched TF-IDF)
  - results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv (BGE-M3)
  - results/oracle_recomputation_2026-08-15/nlp4lp_downstream_per_query_orig_oracle.csv (Oracle)
"""
import csv
import json

ARMS = {
    "TFIDF-TG (baseline extraction)": "results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv",
    "TFIDF-TG (ratio-aware extraction)": "results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv",
    "BGE-M3 (dense) + ratio-aware grounding": "results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv",
    "Oracle-TG": "results/oracle_recomputation_2026-08-15/nlp4lp_downstream_per_query_orig_oracle.csv",
}

OUT = "results/final_resubmission_method/exact20_uniform_2026-08-27.json"


def comparable_subset_exact20(path):
    rows = list(csv.DictReader(open(path)))
    n = len(rows)
    hits = [r for r in rows if int(float(r.get("schema_hit", 0) or 0)) == 1]
    comparable = [r for r in hits if r.get("exact20", "") != ""]
    non_comparable = len(hits) - len(comparable)
    value = sum(float(r["exact20"]) for r in comparable) / len(comparable) if comparable else None
    return {
        "source_file": path,
        "n": n,
        "n_schema_hit": len(hits),
        "n_comparable": len(comparable),
        "n_hit_without_comparable_value": non_comparable,
        "exact20_comparable_subset": value,
    }


def main():
    result = {arm: comparable_subset_exact20(path) for arm, path in ARMS.items()}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
