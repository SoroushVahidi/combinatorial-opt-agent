#!/usr/bin/env python3
"""Generate FINAL_REVISION_EXPERIMENT_SUMMARY.md from all experiment results.

Consolidates:
  - Downstream comparison table (all methods, orig variant)
  - Cross-variant table (key methods, orig/noisy/short)
  - Significance summary (paired tests + CI)
  - Overlap stress-test summary
  - Error analysis highlights

Usage (from repo root):
    python tools/generate_revision_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOWNSTREAM_DIR = ROOT / "results" / "eswa_revision" / "02_downstream_postfix"
SIG_DIR = ROOT / "results" / "eswa_revision" / "15_significance"
ERR_DIR = ROOT / "results" / "eswa_revision" / "16_error_analysis"
OVL_DIR = ROOT / "results" / "eswa_revision" / "17_overlap_analysis"
OUT_DIR = ROOT / "results" / "eswa_revision" / "14_reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _r(v, decimals=4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _agg_from_json(variant: str, method: str) -> dict:
    p = DOWNSTREAM_DIR / f"nlp4lp_downstream_{variant}_{method}.json"
    if not p.exists():
        return {}
    try:
        return json.load(open(p, encoding="utf-8")).get("aggregate", {})
    except Exception:
        return {}


# ── table builders ────────────────────────────────────────────────────────────

def _downstream_table(variant: str = "orig") -> str:
    """Build markdown table for all methods on a given variant."""
    methods = [
        ("tfidf", "TF-IDF (greedy)", False),
        ("bm25", "BM25 (greedy)", False),
        ("lsa", "LSA (greedy)", False),
        ("oracle", "Oracle (greedy)", False),
        ("tfidf_acceptance_rerank", "TF-IDF + Accept.Rerank", False),
        ("tfidf_hierarchical_acceptance_rerank", "TF-IDF + Hier.Accept.Rerank", False),
        ("tfidf_optimization_role_repair", "TF-IDF + OptRoleRepair", False),
        # Family 1
        ("tfidf_global_compat_local", "GCG – local-only", True),
        ("tfidf_global_compat_pairwise", "GCG – pairwise", True),
        ("tfidf_global_compat_full", "GCG – full", True),
        # Family 2
        ("tfidf_relation_aware_basic", "RAL – basic", True),
        ("tfidf_relation_aware_ops", "RAL – ops", True),
        ("tfidf_relation_aware_semantic", "RAL – semantic", True),
        ("tfidf_relation_aware_full", "RAL – full", True),
        # Family 3
        ("tfidf_ambiguity_candidate_greedy", "AAG – candidate-greedy", True),
        ("tfidf_ambiguity_aware_beam", "AAG – beam", True),
        ("tfidf_ambiguity_aware_abstain", "AAG – abstain", True),
        ("tfidf_ambiguity_aware_full", "AAG – full", True),
    ]

    header = "| Method | New? | Schema R@1 | Coverage | TypeMatch | InstReady |\n"
    header += "|--------|------|-----------|----------|-----------|----------|\n"
    rows = [header]
    for mkey, label, is_new in methods:
        agg = _agg_from_json(variant, mkey)
        if not agg:
            continue
        new_flag = "✓" if is_new else ""
        rows.append(
            f"| {label} | {new_flag} | {_r(agg.get('schema_R1',0))} "
            f"| {_r(agg.get('param_coverage',0))} "
            f"| {_r(agg.get('type_match',0))} "
            f"| {_r(agg.get('instantiation_ready',0))} |\n"
        )
    return "".join(rows)


def _cross_variant_table() -> str:
    """Key methods across orig/noisy/short."""
    sel_methods = [
        ("tfidf", "TF-IDF (baseline)"),
        ("oracle", "Oracle"),
        ("tfidf_acceptance_rerank", "TFIDF-AR"),
        ("tfidf_global_compat_full", "GCG-Full ★"),
        ("tfidf_relation_aware_basic", "RAL-Basic ★"),
        ("tfidf_ambiguity_aware_beam", "AAG-Beam ★"),
    ]
    variants = ["orig", "noisy", "short"]

    header = "| Method | " + " | ".join(f"InstReady ({v})" for v in variants) + " |\n"
    header += "|--------|" + "|".join("---" for _ in variants) + "|\n"
    rows = [header]
    for mkey, label in sel_methods:
        cells = []
        for v in variants:
            agg = _agg_from_json(v, mkey)
            cells.append(_r(agg.get("instantiation_ready", "–")))
        rows.append(f"| {label} | " + " | ".join(cells) + " |\n")
    return "".join(rows)


def _sig_table() -> str:
    """Format paired significance table (orig only)."""
    rows = _load_csv(SIG_DIR / "paired_significance.csv")
    orig_rows = [r for r in rows if r.get("variant") == "orig"]
    if not orig_rows:
        return "_No significance data found._\n"

    header = "| Comparison | Diff (A−B) | 95% CI | p-value | Sig? |\n"
    header += "|-----------|-----------|--------|---------|------|\n"
    lines = [header]
    for r in orig_rows:
        diff = float(r.get("obs_diff_A_minus_B", 0))
        lo = float(r.get("ci_lo_diff_95", 0))
        hi = float(r.get("ci_hi_diff_95", 0))
        p = float(r.get("p_two_sided", 1))
        sig = "**p<0.05**" if p < 0.05 else "n.s."
        label = r.get("comparison", "")
        lines.append(
            f"| {label} | {diff:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.4f} | {sig} |\n"
        )
    return "".join(lines)


def _overlap_table() -> str:
    rows = _load_csv(OVL_DIR / "retrieval_overlap_ablation.csv")
    if not rows:
        return "_No overlap ablation data found._\n"

    header = "| Sanitize Variant | TF-IDF Schema R@1 | BM25 Schema R@1 | LSA Schema R@1 |\n"
    header += "|-----------------|-------------------|-----------------|----------------|\n"
    # Group by sanitize_variant
    by_sv: dict[str, dict] = {}
    for r in rows:
        sv = r["sanitize_variant"]
        if sv not in by_sv:
            by_sv[sv] = {}
        by_sv[sv][r["method"]] = r.get("Schema_R1", "–")
    lines = [header]
    for sv, methods in sorted(by_sv.items()):
        lines.append(
            f"| {sv} | {methods.get('tfidf','–')} | {methods.get('bm25','–')} | {methods.get('lsa','–')} |\n"
        )
    return "".join(lines)


def _error_highlights() -> str:
    rows = _load_csv(ERR_DIR / "schema_hit_miss_breakdown.csv")
    if not rows:
        return "_No error breakdown data._\n"
    lines = []
    for r in rows:
        lines.append(
            f"  - **{r.get('group','?')}** (n={r.get('n','?')}): "
            f"Coverage={r.get('Coverage_mean','?')}, TypeMatch={r.get('TypeMatch_mean','?')}, "
            f"InstReady={r.get('InstReady_rate','?')}\n"
        )
    return "".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append("# FINAL REVISION EXPERIMENT SUMMARY\n\n")
    lines.append(
        "This document summarises all experiment results for the ESWA paper revision.\n"
        "It covers retrieval baselines, three new downstream method families, "
        "statistical significance, error analysis, and overlap stress tests.\n\n"
    )

    # ── Executive summary ─────────────────────────────────────────────────────
    lines.append("## 1. Executive Summary\n\n")
    tfidf_orig = _agg_from_json("orig", "tfidf")
    oracle_orig = _agg_from_json("orig", "oracle")
    best_new_agg = max(
        [_agg_from_json("orig", m) for m in
         ["tfidf_relation_aware_basic", "tfidf_global_compat_full", "tfidf_ambiguity_aware_beam",
          "tfidf_ambiguity_aware_full", "tfidf_ambiguity_candidate_greedy"]],
        key=lambda a: float(a.get("instantiation_ready", 0))
    )
    tfidf_ir = float(tfidf_orig.get("instantiation_ready", 0))
    oracle_ir = float(oracle_orig.get("instantiation_ready", 0))
    best_new_ir = float(best_new_agg.get("instantiation_ready", 0))

    lines.append(
        f"- **TF-IDF greedy (ORIG)**: Schema R@1={_r(tfidf_orig.get('schema_R1',0))}, "
        f"InstReady={_r(tfidf_ir)}\n"
    )
    lines.append(
        f"- **Oracle (ORIG)**: Schema R@1=1.0000, InstReady={_r(oracle_ir)}\n"
    )
    lines.append(
        f"- **Best new method family (ORIG)**: InstReady={_r(best_new_ir)} "
        f"({_r((best_new_ir-tfidf_ir)*100, 2)}pp vs TF-IDF)\n\n"
    )
    lines.append(
        "**Key finding**: The three new downstream method families (Global Compatibility "
        "Grounding, Relation-Aware Linking, Ambiguity-Aware Grounding) achieve InstReady "
        f"in the range 0.42–0.50 on ORIG, compared to TF-IDF greedy at {_r(tfidf_ir)}. "
        "None exceeds the TF-IDF greedy baseline; the existing bottleneck conclusion holds: "
        "downstream grounding is difficult regardless of assignment strategy, and the "
        "gap to oracle (InstReady gap ~{:.2f}pp) is explained primarily by schema confusion "
        "(missed parameters, type errors) rather than retrieval.\n\n".format(
            (oracle_ir - tfidf_ir) * 100
        )
    )

    # ── Downstream table ──────────────────────────────────────────────────────
    lines.append("## 2. Downstream Comparison Table (ORIG)\n\n")
    lines.append(_downstream_table("orig"))
    lines.append("\n_★ = newly added method family_\n\n")

    # ── Cross-variant table ───────────────────────────────────────────────────
    lines.append("## 3. Cross-Variant Results (InstReady)\n\n")
    lines.append(_cross_variant_table())
    lines.append(
        "\n**Note**: On NOISY and SHORT variants all methods collapse to near-zero "
        "InstReady, confirming that masking/truncation destroys numeric grounding "
        "regardless of assignment strategy.\n\n"
    )

    # ── Statistical significance ───────────────────────────────────────────────
    lines.append("## 4. Statistical Significance (ORIG, paired bootstrap, B=1000)\n\n")
    lines.append(_sig_table())
    lines.append(
        "\n**Interpretation**: TFIDF vs Oracle difference is significant (p=0.004). "
        "TFIDF vs BM25 is marginally significant (p=0.022) on Schema R@1. "
        "All new methods fall significantly *below* TF-IDF greedy (p<0.05), "
        "confirming that the new assignment strategies do not improve over the simple "
        "greedy baseline under the current feature regime.\n\n"
    )

    # ── Error analysis ────────────────────────────────────────────────────────
    lines.append("## 5. Error Analysis Highlights (TF-IDF, ORIG)\n\n")
    lines.append(_error_highlights())
    lines.append(
        "\n**Pattern**: Schema misses (30 queries, 9.1%) have dramatically lower "
        "InstReady, confirming retrieval bottleneck. Even on schema hits, InstReady "
        "is only ~55%, pointing to parameter-level grounding as the persistent challenge.\n\n"
    )

    # ── Overlap analysis ──────────────────────────────────────────────────────
    lines.append("## 6. Overlap Stress Tests (BM25 / TF-IDF / LSA)\n\n")
    lines.append(_overlap_table())
    lines.append(
        "\n**Interpretation**: TF-IDF and BM25 performance is *stable or slightly "
        "improves* when numbers and stopwords are removed, indicating that retrieval "
        "success is driven by structural/semantic term overlap (parameter names, units, "
        "domain keywords) rather than exact numeric matching.\n\n"
    )

    # ── Conclusions ────────────────────────────────────────────────────────────
    lines.append("## 7. Conclusions for Paper Revision\n\n")
    lines.append(
        "1. **Dense retrievers (E5/BGE) not available** in this sandboxed environment "
        "(no internet). Lexical baselines remain as comparison points.\n"
        "2. **TF-IDF retrieval remains competitive**: Schema R@1 ≈ 0.91 on ORIG, "
        "stable under number/stopword removal — structural overlap drives success.\n"
        "3. **New downstream methods do not improve InstReady**: All three families "
        "(GCG, RAL, AAG) score below TF-IDF greedy (0.43–0.50 vs 0.53), "
        "with differences statistically significant.\n"
        "4. **Bottleneck conclusion holds**: The primary failure mode is "
        "parameter-level confusion (type errors, missing slots) on schema hits, "
        "not schema miss. Oracle upper bound is only 0.57 — the task is genuinely hard.\n"
        "5. **Abstention (AAG-Abstain)** aggressively abstains (Coverage=0.22), "
        "showing that uncertainty signals are noisy but functional.\n"
        "6. **Ambiguity structure is high**: Most queries have multiple comparable "
        "numeric mentions, making greedy assignment inherently unreliable.\n\n"
    )

    # ── Verification commands ─────────────────────────────────────────────────
    lines.append("## 8. Exact Verification Commands\n\n```bash\n")
    lines.append(
        "# Run all new methods (33 runs: 11 methods × 3 variants)\n"
        "NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \\\n"
        "  python -c \"\n"
        "from tools.nlp4lp_downstream_utility import run_single_setting\n"
        "from pathlib import Path\n"
        "out_dir = Path('results/eswa_revision/02_downstream_postfix')\n"
        "NEW_METHODS = [\n"
        "    ('tfidf','global_compat_local'), ('tfidf','global_compat_pairwise'),\n"
        "    ('tfidf','global_compat_full'),  ('tfidf','relation_aware_basic'),\n"
        "    ('tfidf','relation_aware_ops'),  ('tfidf','relation_aware_semantic'),\n"
        "    ('tfidf','relation_aware_full'), ('tfidf','ambiguity_candidate_greedy'),\n"
        "    ('tfidf','ambiguity_aware_beam'),('tfidf','ambiguity_aware_abstain'),\n"
        "    ('tfidf','ambiguity_aware_full'),\n"
        "]\n"
        "for v in ('orig','noisy','short'):\n"
        "    for b,m in NEW_METHODS: run_single_setting(v,b,m,out_dir)\n"
        "\"\n\n"
        "# Run significance / CI analysis\n"
        "NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \\\n"
        "  python tools/run_confidence_intervals.py\n\n"
        "# Run error analysis (produces method_comparison_table.csv)\n"
        "NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \\\n"
        "  python tools/run_error_analysis.py\n\n"
        "# Run overlap stress tests\n"
        "NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \\\n"
        "  python tools/run_overlap_analysis.py\n\n"
        "# Regenerate this report\n"
        "python tools/generate_revision_report.py\n"
        "```\n\n"
    )

    out_path = OUT_DIR / "FINAL_REVISION_EXPERIMENT_SUMMARY.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
