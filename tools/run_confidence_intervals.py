"""
1B – Statistical significance / uncertainty reporting.

Computes non-parametric bootstrap confidence intervals for key retrieval and
downstream metrics, and paired bootstrap significance tests for important
method comparisons.

All inputs are per-query CSV files already present under
  results/eswa_revision/02_downstream_postfix/

Retrieval Schema_R@1 is derived from the schema_hit column of those files.

Usage (from repo root):
    python tools/run_confidence_intervals.py

Outputs (written to results/eswa_revision/15_significance/):
  confidence_intervals.csv
  paired_significance.csv
  SIGNIFICANCE_SUMMARY.md
"""
from __future__ import annotations

import csv
import math
import random
import sys
from pathlib import Path
from typing import Callable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOWNSTREAM_DIR = ROOT / "results" / "eswa_revision" / "02_downstream_postfix"
OUT_DIR = ROOT / "results" / "eswa_revision" / "15_significance"

# ── per-query CSV files of interest ─────────────────────────────────────────
# Naming: nlp4lp_downstream_per_query_{variant}_{method}.csv
MAIN_METHODS = {
    "tfidf": "tfidf",
    "bm25": "bm25",
    "lsa": "lsa",
    "oracle": "oracle",
    "tfidf_ar": "tfidf_acceptance_rerank",
    "tfidf_har": "tfidf_hierarchical_acceptance_rerank",
    "tfidf_const": "tfidf_constrained",
    "tfidf_sem": "tfidf_semantic_ir_repair",
    "tfidf_opt": "tfidf_optimization_role_repair",
    # Method Family 1: Global Compatibility Grounding
    "gcg_local": "tfidf_global_compat_local",
    "gcg_pairwise": "tfidf_global_compat_pairwise",
    "gcg_full": "tfidf_global_compat_full",
    # Method Family 2: Relation-Aware Linking
    "ral_basic": "tfidf_relation_aware_basic",
    "ral_ops": "tfidf_relation_aware_ops",
    "ral_semantic": "tfidf_relation_aware_semantic",
    "ral_full": "tfidf_relation_aware_full",
    # Method Family 3: Ambiguity-Aware Grounding
    "aag_greedy": "tfidf_ambiguity_candidate_greedy",
    "aag_beam": "tfidf_ambiguity_aware_beam",
    "aag_abstain": "tfidf_ambiguity_aware_abstain",
    "aag_full": "tfidf_ambiguity_aware_full",
}

VARIANTS = ["orig", "noisy", "short"]

# Paired comparisons to run (each as (label, method_A, method_B, metric))
# Using the short method keys above
PAIRED_COMPARISONS = [
    ("TFIDF vs BM25 (Schema_R@1)", "tfidf", "bm25", "schema_hit"),
    ("TFIDF-TG vs BM25-TG (InstReady)", "tfidf", "bm25", "inst_ready"),
    ("TFIDF-TG vs Oracle-TG (InstReady)", "tfidf", "oracle", "inst_ready"),
    ("TFIDF-TG vs TFIDF-AR (InstReady)", "tfidf", "tfidf_ar", "inst_ready"),
    ("TFIDF-TG vs TFIDF-HAR (InstReady)", "tfidf", "tfidf_har", "inst_ready"),
    # Method Family 1 vs TFIDF
    ("TFIDF-TG vs GCG-Full (InstReady)", "tfidf", "gcg_full", "inst_ready"),
    # Method Family 2 vs TFIDF
    ("TFIDF-TG vs RAL-Basic (InstReady)", "tfidf", "ral_basic", "inst_ready"),
    ("TFIDF-TG vs RAL-Full (InstReady)", "tfidf", "ral_full", "inst_ready"),
    # Method Family 3 vs TFIDF
    ("TFIDF-TG vs AAG-Beam (InstReady)", "tfidf", "aag_beam", "inst_ready"),
    ("TFIDF-TG vs AAG-Full (InstReady)", "tfidf", "aag_full", "inst_ready"),
    # Best new vs Oracle
    ("RAL-Basic vs Oracle-TG (InstReady)", "ral_basic", "oracle", "inst_ready"),
    # Coverage comparisons for abstain variant
    ("TFIDF-TG vs AAG-Abstain (Coverage)", "tfidf", "aag_abstain", "param_coverage"),
]


# ── utility ──────────────────────────────────────────────────────────────────

def _load_per_query(variant: str, method_key: str) -> list[dict]:
    """Load per-query CSV for a (variant, method) combination."""
    fname = f"nlp4lp_downstream_per_query_{variant}_{MAIN_METHODS[method_key]}.csv"
    path = DOWNSTREAM_DIR / fname
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col_floats(rows: list[dict], col: str) -> list[float]:
    """Extract a numeric column from CSV rows, skipping empties."""
    out = []
    for r in rows:
        v = r.get(col, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            pass
    return out


def _inst_ready_col(rows: list[dict]) -> list[float]:
    """Binary per-instance instantiation_ready signal (coverage>=0.8 & type_match>=0.8)."""
    out = []
    for r in rows:
        try:
            cov = float(r["param_coverage"])
            tm = float(r["type_match"])
        except (ValueError, KeyError):
            continue
        out.append(1.0 if cov >= 0.8 and tm >= 0.8 else 0.0)
    return out


def bootstrap_ci(
    vals: list[float],
    B: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (observed_mean, ci_lo, ci_hi) from percentile bootstrap."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    n = len(vals)
    observed = sum(vals) / n
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(B):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int((alpha / 2) * B)
    hi_idx = max(0, int((1 - alpha / 2) * B) - 1)
    return observed, boot_means[lo_idx], boot_means[hi_idx]


def paired_bootstrap_test(
    vals_a: list[float],
    vals_b: list[float],
    B: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Paired bootstrap test for mean(A) - mean(B).

    Returns (obs_diff, ci_lo_diff, ci_hi_diff, two_sided_p).
    """
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a = vals_a[:n]
    b = vals_b[:n]
    obs_diff = sum(a) / n - sum(b) / n
    rng = random.Random(seed)
    boot_diffs: list[float] = []
    count_opposite = 0
    for _ in range(B):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(a[i] - b[i] for i in idxs) / n
        boot_diffs.append(d)
        if obs_diff > 0 and d <= 0:
            count_opposite += 1
        elif obs_diff < 0 and d >= 0:
            count_opposite += 1
        elif obs_diff == 0 and d != 0:
            count_opposite += 1
    p = min(1.0, max(0.0, (count_opposite / B) * 2))
    boot_diffs.sort()
    lo = boot_diffs[int(0.025 * B)]
    hi = boot_diffs[max(0, int(0.975 * B) - 1)]
    return obs_diff, lo, hi, p


# ── main ─────────────────────────────────────────────────────────────────────

def main(B: int = 1000, seed: int = 42) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Confidence intervals ─────────────────────────────────────────────────
    ci_rows: list[dict] = []

    # Metrics to bootstrap: schema_hit (Schema_R@1), param_coverage, type_match, inst_ready
    CI_METRICS = [
        ("Schema_R@1", "schema_hit"),
        ("Coverage", "param_coverage"),
        ("TypeMatch", "type_match"),
        ("InstReady", "inst_ready"),
    ]

    for variant in VARIANTS:
        for method_key in MAIN_METHODS:
            rows = _load_per_query(variant, method_key)
            if not rows:
                continue
            for metric_name, col in CI_METRICS:
                if col == "inst_ready":
                    vals = _inst_ready_col(rows)
                elif col == "schema_hit":
                    vals = _col_floats(rows, "schema_hit")
                else:
                    vals = _col_floats(rows, col)
                if not vals:
                    continue
                obs, lo, hi = bootstrap_ci(vals, B=B, seed=seed)
                ci_rows.append({
                    "variant": variant,
                    "method": MAIN_METHODS[method_key],
                    "metric": metric_name,
                    "observed": f"{obs:.4f}",
                    "ci_lo_95": f"{lo:.4f}",
                    "ci_hi_95": f"{hi:.4f}",
                    "n": len(vals),
                    "B": B,
                    "seed": seed,
                })

    ci_path = OUT_DIR / "confidence_intervals.csv"
    with open(ci_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "method", "metric", "observed", "ci_lo_95", "ci_hi_95", "n", "B", "seed"])
        w.writeheader()
        w.writerows(ci_rows)
    print(f"Wrote {ci_path} ({len(ci_rows)} rows)")

    # ── Paired significance tests ─────────────────────────────────────────────
    sig_rows: list[dict] = []

    for variant in VARIANTS:
        for comp_label, key_a, key_b, col in PAIRED_COMPARISONS:
            rows_a = _load_per_query(variant, key_a)
            rows_b = _load_per_query(variant, key_b)
            if not rows_a or not rows_b:
                continue
            if col == "inst_ready":
                vals_a = _inst_ready_col(rows_a)
                vals_b = _inst_ready_col(rows_b)
            else:
                vals_a = _col_floats(rows_a, col)
                vals_b = _col_floats(rows_b, col)
            obs_diff, lo_diff, hi_diff, p = paired_bootstrap_test(vals_a, vals_b, B=B, seed=seed)
            mean_a = sum(vals_a) / len(vals_a) if vals_a else float("nan")
            mean_b = sum(vals_b) / len(vals_b) if vals_b else float("nan")
            metric_name = "Schema_R@1" if col == "schema_hit" else (
                "InstReady" if col == "inst_ready" else col
            )
            sig_rows.append({
                "variant": variant,
                "comparison": comp_label,
                "metric": metric_name,
                "method_A": MAIN_METHODS[key_a],
                "mean_A": f"{mean_a:.4f}",
                "method_B": MAIN_METHODS[key_b],
                "mean_B": f"{mean_b:.4f}",
                "obs_diff_A_minus_B": f"{obs_diff:.4f}",
                "ci_lo_diff_95": f"{lo_diff:.4f}",
                "ci_hi_diff_95": f"{hi_diff:.4f}",
                "p_two_sided": f"{p:.4f}",
                "n": min(len(vals_a), len(vals_b)),
                "B": B,
                "seed": seed,
            })

    sig_path = OUT_DIR / "paired_significance.csv"
    with open(sig_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "comparison", "metric", "method_A", "mean_A",
            "method_B", "mean_B", "obs_diff_A_minus_B",
            "ci_lo_diff_95", "ci_hi_diff_95", "p_two_sided", "n", "B", "seed",
        ])
        w.writeheader()
        w.writerows(sig_rows)
    print(f"Wrote {sig_path} ({len(sig_rows)} rows)")

    # ── Markdown summary ──────────────────────────────────────────────────────
    _write_summary(ci_rows, sig_rows, B, seed)
    print(f"Wrote {OUT_DIR / 'SIGNIFICANCE_SUMMARY.md'}")


def _write_summary(ci_rows: list[dict], sig_rows: list[dict], B: int, seed: int) -> None:
    lines = [
        "# Statistical Significance Summary",
        "",
        f"Bootstrap samples: B={B}, seed={seed}, 95% CIs (percentile method).",
        "All tests are two-sided paired bootstrap tests on per-instance binary outcomes.",
        "**Conservative interpretation**: p < 0.05 is noted but not over-interpreted;",
        "overlapping CIs or p ≥ 0.05 are explicitly called out as non-significant.",
        "",
        "---",
        "",
        "## Confidence Intervals (orig variant, key methods)",
        "",
        "| Method | Metric | Observed | 95% CI |",
        "|--------|--------|----------|--------|",
    ]
    shown_methods = {"tfidf", "bm25", "lsa", "oracle", "tfidf_acceptance_rerank", "tfidf_hierarchical_acceptance_rerank"}
    shown_metrics = {"Schema_R@1", "Coverage", "TypeMatch", "InstReady"}
    for row in ci_rows:
        if row["variant"] != "orig":
            continue
        if row["method"] not in shown_methods:
            continue
        if row["metric"] not in shown_metrics:
            continue
        lines.append(
            f"| {row['method']} | {row['metric']} | {row['observed']} | [{row['ci_lo_95']}, {row['ci_hi_95']}] |"
        )

    lines += [
        "",
        "---",
        "",
        "## Paired Significance Tests (orig variant)",
        "",
        "| Comparison | Metric | A | B | Diff | 95% CI (diff) | p-value | Interpretation |",
        "|------------|--------|---|---|------|---------------|---------|----------------|",
    ]

    for row in sig_rows:
        if row["variant"] != "orig":
            continue
        try:
            p = float(row["p_two_sided"])
            lo = float(row["ci_lo_diff_95"])
            hi = float(row["ci_hi_diff_95"])
        except ValueError:
            continue
        if p < 0.01:
            interp = "p<0.01 (robust)"
        elif p < 0.05:
            interp = "p<0.05 (significant)"
        elif lo <= 0 <= hi:
            interp = "CI contains 0 (not significant)"
        else:
            interp = "p≥0.05 (weak evidence)"
        lines.append(
            f"| {row['comparison']} | {row['metric']} | {row['mean_A']} | {row['mean_B']} "
            f"| {row['obs_diff_A_minus_B']} | [{row['ci_lo_diff_95']}, {row['ci_hi_diff_95']}] "
            f"| {row['p_two_sided']} | {interp} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes and Caveats",
        "",
        "- All bootstrap CIs are 95% percentile intervals from B bootstrap resamples.",
        "- Paired tests resample instance indices jointly so each resample preserves instance-level pairing.",
        "- Schema_R@1 and InstReady are binary 0/1 per-instance outcomes; Coverage and TypeMatch",
        "  are continuous per-instance values (mean over slots per query).",
        "- Exact20_on_hits has a conditional denominator (schema-hit queries only) and is not",
        "  included in paired tests to avoid denominator instability.",
        "- The 'orig' variant is the primary eval split; noisy/short results are in the full CSV.",
        "- For the full table including noisy and short variants, see confidence_intervals.csv",
        "  and paired_significance.csv.",
    ]

    md_path = OUT_DIR / "SIGNIFICANCE_SUMMARY.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Compute bootstrap CIs and paired significance tests")
    ap.add_argument("--B", type=int, default=1000, help="Bootstrap samples (default: 1000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = ap.parse_args()
    main(B=args.B, seed=args.seed)
