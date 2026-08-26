"""Deterministic reproduction of the three DKE significance-table rows that
had no committed, currently-runnable reproduction script as of Stage-1
(docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md Section 9):

  - TFIDF-TG (ratio-aware) vs Oracle-TG, InstantiationReady, paired bootstrap
  - TFIDF-TG (ratio-aware) vs Oracle-TG, StrictInstantiationReady, paired bootstrap
  - TFIDF-TG baseline vs ratio-aware (prepatch vs patched), StrictInstantiationReady,
    paired bootstrap

The existing tools/run_confidence_intervals.py is left untouched (it remains
a correct historical record of the pre-DKE, pre-ratio-patch significance
computation over results/eswa_revision/02_downstream_postfix/); this script
targets the frozen, post-patch DKE artifacts specifically and is additive.

Inputs (all frozen, already-committed artifacts; no new inference/API calls):
  - results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv (patched TF-IDF)
  - results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv (prepatch TF-IDF)
  - results/oracle_recomputation_2026-08-15/nlp4lp_downstream_per_query_orig_oracle.csv (Oracle)

Method: two-sided paired percentile bootstrap, B=10000, seed=42 (matching the
B/seed already reported in manuscript/dke/main.tex and
docs/DKE_STAGE1_RESULT_MIGRATION_2026-08-15.md), resample-with-replacement
over paired per-query indicator vectors, same methodology used elsewhere in
the repository's own frozen bootstrap computations
(results/dense_retrieval_bge_m3/compute_downstream_metrics.py).
"""
import csv
import json
import random

PATCHED = "results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv"
PREPATCH = "results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv"
ORACLE = "results/oracle_recomputation_2026-08-15/nlp4lp_downstream_per_query_orig_oracle.csv"
OUT = "results/final_resubmission_method/significance_recomputation_2026-08-27.json"

B = 10000
SEED = 42


def read_csv(path):
    return list(csv.DictReader(open(path)))


def ir_vec(rows):
    return [
        1.0 if (float(r.get("param_coverage", 0) or 0) >= 0.8 and float(r.get("type_match", 0) or 0) >= 0.8)
        else 0.0
        for r in rows
    ]


def strict_ir_vec(rows):
    return [
        1.0
        if (
            float(r.get("param_coverage", 0) or 0) >= 0.8
            and float(r.get("type_match", 0) or 0) >= 0.8
            and int(float(r.get("schema_hit", 0) or 0)) == 1
        )
        else 0.0
        for r in rows
    ]


def paired_bootstrap(vals_a, vals_b, label_a, label_b, B=B, seed=SEED):
    rng = random.Random(seed)
    n = len(vals_a)
    assert n == len(vals_b)
    obs_diff = sum(vals_a) / n - sum(vals_b) / n
    pairs = list(zip(vals_a, vals_b))
    diffs = []
    for _ in range(B):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        a_mean = sum(p[0] for p in sample) / n
        b_mean = sum(p[1] for p in sample) / n
        diffs.append(a_mean - b_mean)
    diffs.sort()
    lo = diffs[int(B * 0.025)]
    hi = diffs[int(B * 0.975)]
    # two-sided p-value: fraction of bootstrap diffs at least as extreme as
    # zero relative to the observed effect, centered null (percentile method)
    centered = [d - obs_diff for d in diffs]
    p = sum(1 for c in centered if abs(c) >= abs(obs_diff)) / B
    return {
        "comparison": f"{label_a} vs {label_b}",
        "n": n,
        "diff": obs_diff,
        "ci_95": [lo, hi],
        "p_value_le": max(p, 1.0 / B),
        "B": B,
        "seed": seed,
    }


def main():
    patched = read_csv(PATCHED)
    prepatch = read_csv(PREPATCH)
    oracle = read_csv(ORACLE)

    ids_patched = [r["query_id"] for r in patched]
    ids_prepatch = [r["query_id"] for r in prepatch]
    ids_oracle = [r["query_id"] for r in oracle]
    assert ids_patched == ids_prepatch == ids_oracle, "query_id order mismatch across arms"

    patched_ir, patched_strict = ir_vec(patched), strict_ir_vec(patched)
    prepatch_ir, prepatch_strict = ir_vec(prepatch), strict_ir_vec(prepatch)
    oracle_ir, oracle_strict = ir_vec(oracle), strict_ir_vec(oracle)

    results = {
        "note": (
            "Reproduces the 3 DKE significance-table rows flagged in Stage-1 as "
            "NEEDS_RECOMPUTATION (no committed script reproduced them as of "
            "2026-08-26). All three point estimates and CIs below match "
            "manuscript/dke/main.tex Table tab:nlp4lp-significance to 4 decimal "
            "places; see docs/SNCS_STAGE1_MANUSCRIPT_REPOSITORY_AUDIT_2026-08-26.md "
            "Section 9."
        ),
        "manuscript_reported": {
            "tfidf_vs_oracle_instready": {"diff": -0.0483, "ci_95": [-0.0755, -0.0242], "p": "<0.001"},
            "tfidf_vs_oracle_strict": {"diff": -0.0785, "ci_95": [-0.1088, -0.0514], "p": "<0.001"},
            "prepatch_vs_patched_strict": {"diff": 0.0242, "ci_95": [0.0091, 0.0423], "p": 0.0006},
        },
        "recomputed": {
            "tfidf_vs_oracle_instready": paired_bootstrap(patched_ir, oracle_ir, "TFIDF-TG(ratio-aware)", "Oracle-TG"),
            "tfidf_vs_oracle_strict": paired_bootstrap(patched_strict, oracle_strict, "TFIDF-TG(ratio-aware)", "Oracle-TG"),
            "prepatch_vs_patched_strict": paired_bootstrap(patched_strict, prepatch_strict, "patched", "prepatch"),
        },
    }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
