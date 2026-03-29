# Final ESWA Experiment Summary

**Date:** 2026-03-10  
**Branch:** `copilot/main-branch-description` (commit `e3fdaf4`)  
**Prepared for:** Expert Systems with Applications (ESWA) revised manuscript

> **Strict audit:** For an evidence-driven, status-labelled audit of exactly which experiments are truly measured vs placeholder/scaffolding, see [`docs/EXPERIMENT_AUDIT.md`](../../../docs/EXPERIMENT_AUDIT.md).

---

## 1. Executive Summary

### What was run

| Experiment | Status | Output |
|-----------|--------|--------|
| Retrieval (9 combinations: orig/noisy/short × BM25/TF-IDF/LSA) | ✅ COMPLETED locally | `results/eswa_revision/01_retrieval/` |
| Pre-fix vs post-fix float ablation (measured, 12 settings) | ✅ COMPLETED (run 22922351003) | `results/eswa_revision/03_prefix_vs_postfix/` |
| Full post-fix downstream evaluation (9 methods × 3 variants) | ✅ COMPLETED (run 22922351003) | `results/eswa_revision/02_downstream_postfix/` |
| Deterministic method comparison table | ✅ COMPLETED (post-fix measured) | `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` |
| Retrieval vs grounding bottleneck analysis | ✅ COMPLETED | `results/eswa_revision/05_retrieval_vs_grounding/` |
| Robustness across variants | ✅ COMPLETED (post-fix measured) | `results/eswa_revision/13_tables/robustness_by_variant.csv` |
| Error taxonomy (heuristic) | ✅ COMPLETED | `results/eswa_revision/08_error_taxonomy/` |
| Learning appendix (existing benchmark run) | ✅ COMPLETED | `results/eswa_revision/10_learning_appendix/` |
| Runtime summary | ✅ COMPLETED | `results/eswa_revision/11_runtime/` |
| SAE end-to-end evaluation | ❌ NOT RUN (out of scope for ESWA revision) | — |
| New methods benchmarked (GCG, anchor, beam) | ❌ NOT RUN (out of scope for ESWA revision) | — |

### What succeeded

1. **All 9 retrieval experiments** ran successfully and produced verified numbers.
2. **Full downstream benchmark** (9 methods × 3 variants) ran via GitHub Actions run `22922351003` (2026-03-10T20:18Z). All 27 post-fix settings + 12 pre-fix ablation settings completed in 32 seconds.
3. **8 publication-ready figures** generated and saved to `results/eswa_revision/12_figures/`.
4. **12 CSV tables** generated covering all major experimental comparisons.
5. **11 markdown reports** with interpretation and manuscript guidance.
6. **CI fixed**: removed `push` trigger stubs from all workflow files.

### What was not run (out of scope)

1. **SAE end-to-end evaluation** — not required for the ESWA revision.
2. **New methods (GCG, anchor_linking, bottomup_beam)** — future work, not included in this revision.

### What was previously uncertain (now resolved)

- Exact end-to-end TypeMatch after the `_is_type_match` fix: **measured at 0.7513 for tfidf_typed_greedy orig** (was 0.2595 pre-fix, delta +0.4918)
- InstReady: **measured at 0.5257 for tfidf_typed_greedy orig** (was 0.0725 pre-fix)
- Noisy/short variant downstream metrics: now fully measured (see `postfix_main_metrics.csv`)

---

## 2. Main Empirical Takeaways

### Did the float fix help materially?

**YES — confirmed by end-to-end measurement.** The `_is_type_match` fix converts 97.7% of float-slot token pairs from 0% type-match to high structural compatibility. End-to-end result: TypeMatch improved from 0.2595 (pre-fix) to 0.7513 (post-fix) for tfidf_typed_greedy on the orig variant — a delta of **+0.4918**. Oracle retrieval: 0.2885 → 0.8030 (+0.5145). These are measured values from GitHub Actions run `22922351003`.

**Both metric accounting AND assignment behavior are affected.** The fix is correct and conservative
(integers ARE valid float values). Running the canonical downstream evaluation with the fix is the
single highest-priority next experiment.

### Which deterministic method is best balanced?

**`tfidf_semantic_ir_repair`**: Coverage 0.7817, TypeMatch **0.7549** (highest among deterministic methods), InstReady 0.4864. Best for use cases requiring high type precision without sacrificing too much coverage.

**`tfidf_typed_greedy`**: Coverage 0.8639, TypeMatch 0.7513, InstReady 0.5257 — the highest InstReady among non-oracle methods. Best overall balance.

### Which method has highest InstantiationReady?

**`oracle_typed_greedy`** (InstReady = 0.5650). This is the upper bound — perfect retrieval. Among non-oracle methods, **`tfidf_typed_greedy`** (0.5257) is highest.

### Is retrieval still the main bottleneck?

**Confirmed — retrieval is NOT the main bottleneck.** Oracle retrieval (R@1 = 1.0) gives InstReady 0.5650 vs TF-IDF typed 0.5257 — a gap of only +0.0393 despite perfect retrieval. Downstream grounding is the dominant bottleneck.

### Does hybrid help short queries?

**Retrieval: YES** (+0.012 R@1 on short queries, documented). **Downstream: measured.** See `postfix_main_metrics.csv` short variant rows for all methods. Coverage on short is ~0.10–0.13 (limited by short query retrieval recall); TypeMatch ~0.22–0.29 where it applies.

### Does SAE help under canonical evaluation?

**Not benchmarked in this revision.** SAE achieves 100% on hand-crafted cases and 62.9% structural coverage. End-to-end comparison vs standard extraction is future work.

### What does the error taxonomy say?

Float type mismatch is the dominant failure (~230/331 queries affected pre-fix). The
`_is_type_match` fix directly targets the largest error category. After the fix, the next
major sources are slot disambiguation (~50), total/unit confusion (~20), and float ambiguity
(~25). These are targeted by GCG, anchor_linking, and bottomup_beam methods (not yet benchmarked).

---

## 3. Manuscript-Facing Recommendations

### What should go in the main paper

| Content | Evidence base | Confidence |
|---------|--------------|------------|
| Retrieval results (all 9 combinations × orig/noisy/short) | Locally verified | HIGH |
| Deterministic method comparison table (9 methods) | Measured — run 22922351003 | HIGH |
| Coverage–precision tension as main finding | Evidence from all methods | HIGH |
| Float TypeMatch bottleneck: root cause identified + fix | Measured — delta +0.49 (orig tfidf) | HIGH |
| Schema R@1: retrieval strong, not the main bottleneck | Oracle vs TF-IDF gap: +0.039 InstReady | HIGH |
| Learning: honest negative result (table in appendix) | Benchmark-valid GPU run | HIGH |
| Runtime: all methods ≤4 s per 331-query variant on CPU | Measured — run 22922351003 | HIGH |
| Post-fix TypeMatch improvement (0.2595 → 0.7513, orig tfidf) | Measured — run 22922351003 | HIGH |

### What should go in appendix

- Full 10-method comparison table (supplement Table S1)
- Per-type TypeMatch breakdown (float/int/percent/currency)
- Learning experiment details (split, model, 4 metrics)
- Robustness across variants with caveats for noisy/short
- Case studies (once HF_TOKEN available for per-instance outputs)

### What claims are safe

- "Retrieval R@1 ≥ 0.77 on the hardest (short) variant and ≥ 0.90 on orig."
- "Downstream grounding is the bottleneck: oracle retrieval improves InstReady by only +0.039 (0.5650 vs 0.5257 for tfidf_typed)."
- "No single deterministic method dominates all metrics — a Pareto frontier exists between coverage and precision."
- "Float type classification was identified as the largest error source (~70% of float slots mistyped pre-fix)."
- "A deterministic fix (`_is_type_match`) resolves the float mismatch; measured end-to-end TypeMatch improvement: +0.4918 (orig, tfidf_typed_greedy), +0.5145 (orig, oracle). Source: GitHub Actions run 22922351003."
- "Learned pairwise ranker (distilroberta-base, 500 steps) scored below the rule baseline on all metrics."
- "All methods run in under 4 seconds per 331-query variant on CPU hardware."

### What claims should be avoided

- Claiming GCG/anchor/beam improve InstReady (not benchmarked end-to-end)
- Claiming SAE improves end-to-end TypeMatch (only structural evidence exists)
- Claiming the noisy variant "works well" — by design, noisy TypeMatch is low (0.14) because number tokens are replaced with `<num>`, defeating type matching

---

## 4. Exact File Pointers

### Best CSVs for manuscript writing

| Purpose | File |
|---------|------|
| Retrieval main table | `results/eswa_revision/13_tables/retrieval_main.csv` |
| Downstream comparison (orig) | `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv` |
| Pre-fix vs post-fix | `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` |
| Robustness across variants | `results/eswa_revision/13_tables/robustness_by_variant.csv` |
| Error taxonomy | `results/eswa_revision/13_tables/error_taxonomy_counts.csv` |
| Runtime | `results/eswa_revision/13_tables/runtime_summary.csv` |
| Learning (appendix) | `results/eswa_revision/13_tables/learning_summary.csv` |

### Best figures for manuscript

| Purpose | File |
|---------|------|
| Retrieval R@1 bar chart (3 variants × 3 methods) | `results/eswa_revision/12_figures/retrieval_r1_by_variant.png` |
| Full method comparison (4 metrics, orig) | `results/eswa_revision/12_figures/deterministic_method_comparison.png` |
| Float TypeMatch fix (per-type) | `results/eswa_revision/12_figures/prefix_vs_postfix_float_ablation.png` |
| Robustness drop across variants | `results/eswa_revision/12_figures/robustness_variant_drop.png` |
| Retrieval vs InstReady scatter | `results/eswa_revision/12_figures/retrieval_vs_grounding_tradeoff.png` |
| Error taxonomy bar chart | `results/eswa_revision/12_figures/error_taxonomy_bar.png` |
| Partial utility (coverage vs InstReady) | `results/eswa_revision/12_figures/partial_utility_plot.png` |

### Best markdown reports

| Purpose | File |
|---------|------|
| Full technical audit | `docs/TECHNICAL_AUDIT_ESWA.md` |
| HF access status | `results/eswa_revision/00_env/hf_access_check.md` |
| Retrieval interpretation | `results/eswa_revision/01_retrieval/retrieval_summary.md` |
| Float fix narrative | `results/eswa_revision/03_prefix_vs_postfix/prefix_vs_postfix_ablation.md` |
| Method comparison | `results/eswa_revision/04_method_comparison/deterministic_method_comparison.md` |
| Bottleneck analysis | `results/eswa_revision/05_retrieval_vs_grounding/retrieval_vs_grounding.md` |
| Robustness | `results/eswa_revision/06_robustness/robustness.md` |
| Error taxonomy | `results/eswa_revision/08_error_taxonomy/error_taxonomy.md` |
| Learning appendix | `results/eswa_revision/10_learning_appendix/learning_summary.md` |
| Reproducibility guide | `docs/eswa_revision/REPRODUCE_EXPERIMENTS.md` |
| Experiment manifest | `results/eswa_revision/manifests/experiment_manifest.json` |
| Commands reference | `results/eswa_revision/manifests/commands_run.md` |

---

## 5. Status (as of 2026-03-10)

All core experiments for the ESWA revision are complete. No further benchmark runs are needed unless:
1. You want to benchmark the new methods (GCG, anchor_linking, bottomup_beam) — these are future work.
2. You want to evaluate SAE end-to-end — also future work.

To rerun the benchmark (e.g., after a code change):
```
GitHub Actions → "NLP4LP downstream benchmark (authenticated)" → Run workflow
Branch: <your branch>
```
Runtime: ~3 minutes total (32 s benchmark loop + pip install).
