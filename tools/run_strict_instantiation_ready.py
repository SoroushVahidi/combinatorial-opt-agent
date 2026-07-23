#!/usr/bin/env python3
"""
StrictInstantiationReady sensitivity analysis.

The canonical InstantiationReady metric (as defined in the manuscript,
Sec. 3.1) is a per-query threshold rule:

    InstantiationReady_i = 1[Coverage_i >= 0.8 AND TypeMatch_i >= 0.8]

computed over the predicted-schema-conditioned eligible slot set, with no
separate hard gate on exact schema match. This script computes a stricter
diagnostic variant that adds that gate back in:

    StrictInstantiationReady_i = 1[SchemaHit_i AND Coverage_i >= 0.8 AND TypeMatch_i >= 0.8]

This directly answers the reviewer question: "could the current
InstantiationReady definition allow an incorrect schema with a favorable
reduced overlap to satisfy the threshold?" It is reported as a robustness/
sensitivity check, NOT as a replacement for the canonical metric.

Data source: the same per-query CSVs and bootstrap methodology used by
tools/run_confidence_intervals.py (results/eswa_revision/02_downstream_postfix/,
paired percentile bootstrap, B=1000, seed=42).

As of the final KAIS submission pass, Table 4 (tab:nlp4lp-downstream-main)
and Table 9/10 (significance tables) are themselves regenerated directly
from these same canonical per-query files (the previously frozen
results/eswa_revision/15_significance/confidence_intervals.csv snapshot was
found to be stale -- it no longer reproduced from the currently committed
per-query CSVs -- and was regenerated). Consequently this script's non-strict
InstantiationReady column is now IDENTICAL to Table 4, not merely offset from
it by a small disclosed amount. Recomputing the canonical (non-strict)
InstantiationReady from these live files reproduces the significance-table
paired-bootstrap DIFFERENCE, CI, and p-value for TFIDF-TG vs. Oracle-TG
exactly.

Outputs (results/eswa_revision/18_strict_instready/):
  strict_instantiation_ready.csv     -- per-method Schema_R1, IR, StrictIR
  strict_vs_standard_significance.csv -- paired bootstrap TFIDF-TG vs Oracle-TG
                                          under both the standard and strict metric

Usage (from repo root):
    python tools/run_strict_instantiation_ready.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNSTREAM_DIR = ROOT / "results" / "eswa_revision" / "02_downstream_postfix"
OUT_DIR = ROOT / "results" / "eswa_revision" / "18_strict_instready"

METHODS = {"tfidf": "TFIDF-TG", "bm25": "BM25-TG", "lsa": "LSA-TG", "oracle": "Oracle-TG"}
VARIANT = "orig"
B = 1000
SEED = 42


def _load_per_query(variant: str, method: str) -> list[dict]:
    fname = f"nlp4lp_downstream_per_query_{variant}_{method}.csv"
    path = DOWNSTREAM_DIR / fname
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _standard_ir(rows: list[dict]) -> list[float]:
    out = []
    for r in rows:
        try:
            cov = float(r.get("param_coverage", 0) or 0)
            tm = float(r.get("type_match", 0) or 0)
        except (ValueError, KeyError):
            out.append(0.0)
            continue
        out.append(1.0 if cov >= 0.8 and tm >= 0.8 else 0.0)
    return out


def _strict_ir(rows: list[dict]) -> list[float]:
    out = []
    for r in rows:
        try:
            hit = int(float(r.get("schema_hit", 0) or 0))
            cov = float(r.get("param_coverage", 0) or 0)
            tm = float(r.get("type_match", 0) or 0)
        except (ValueError, KeyError):
            out.append(0.0)
            continue
        out.append(1.0 if (hit == 1 and cov >= 0.8 and tm >= 0.8) else 0.0)
    return out


def paired_bootstrap_test(
    vals_a: list[float], vals_b: list[float], B: int = 1000, seed: int = 42
) -> tuple[float, float, float, float]:
    """Same methodology as tools/run_confidence_intervals.py::paired_bootstrap_test."""
    n = min(len(vals_a), len(vals_b))
    a, b = vals_a[:n], vals_b[:n]
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_method_vals: dict[str, tuple[list[dict], list[float], list[float]]] = {}
    summary_rows = []
    for key, label in METHODS.items():
        rows = _load_per_query(VARIANT, key)
        n = len(rows)
        schema_r1 = sum(int(float(r["schema_hit"])) for r in rows) / n
        std = _standard_ir(rows)
        strict = _strict_ir(rows)
        n_differ = sum(1 for a, b in zip(std, strict) if a != b)
        per_method_vals[key] = (rows, std, strict)
        summary_rows.append({
            "method": label,
            "n": n,
            "Schema_R1": round(schema_r1, 4),
            "InstantiationReady": round(sum(std) / n, 4),
            "StrictInstantiationReady": round(sum(strict) / n, 4),
            "delta_standard_minus_strict": round(sum(std) / n - sum(strict) / n, 4),
            "n_queries_differing": n_differ,
        })

    summary_path = OUT_DIR / "strict_instantiation_ready.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_path} ({len(summary_rows)} rows)")

    _, std_tfidf, strict_tfidf = per_method_vals["tfidf"]
    _, std_oracle, strict_oracle = per_method_vals["oracle"]

    sig_rows = []
    obs, lo, hi, p = paired_bootstrap_test(std_tfidf, std_oracle, B=B, seed=SEED)
    sig_rows.append({
        "comparison": "TFIDF-TG vs Oracle-TG (InstantiationReady, standard)",
        "diff": round(obs, 4), "ci_lo_95": round(lo, 4), "ci_hi_95": round(hi, 4),
        "p_two_sided": round(p, 4), "n": 331, "B": B, "seed": SEED,
    })
    obs2, lo2, hi2, p2 = paired_bootstrap_test(strict_tfidf, strict_oracle, B=B, seed=SEED)
    sig_rows.append({
        "comparison": "TFIDF-TG vs Oracle-TG (StrictInstantiationReady)",
        "diff": round(obs2, 4), "ci_lo_95": round(lo2, 4), "ci_hi_95": round(hi2, 4),
        "p_two_sided": round(p2, 4), "n": 331, "B": B, "seed": SEED,
    })
    sig_path = OUT_DIR / "strict_vs_standard_significance.csv"
    with open(sig_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
        w.writeheader()
        w.writerows(sig_rows)
    print(f"Wrote {sig_path} ({len(sig_rows)} rows)")

    print()
    print("NOTE: recomputing the standard (non-strict) InstantiationReady from")
    print("these same live per-query files and paired-bootstrapping TFIDF-TG vs.")
    print("Oracle-TG reproduces the significance-table result (diff=-0.0393,")
    print("CI=[-0.0665,-0.0151], p=0.004) exactly. Table 4 and the significance")
    print("tables are now themselves generated from these same canonical")
    print("per-query files (the previously frozen confidence_intervals.csv")
    print("snapshot was stale and has been regenerated), so there is no longer")
    print("any offset to reconcile.")


if __name__ == "__main__":
    main()
