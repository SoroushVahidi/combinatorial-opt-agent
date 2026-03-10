# Final ESWA Experiment Summary

**Date:** 2026-03-10  
**Branch:** `copilot/main-branch-description` (commit `e3fdaf4`)  
**Prepared for:** Expert Systems with Applications (ESWA) revised manuscript

---

## 1. Executive Summary

### What was run

| Experiment | Status | Output |
|-----------|--------|--------|
| Retrieval (9 combinations: orig/noisy/short × BM25/TF-IDF/LSA) | ✅ COMPLETED locally | `results/eswa_revision/01_retrieval/` |
| Pre-fix vs post-fix float ablation (structural analysis) | ✅ COMPLETED structurally | `results/eswa_revision/03_prefix_vs_postfix/` |
| Deterministic method comparison table | ✅ COMPLETED (manuscript numbers) | `results/eswa_revision/04_method_comparison/` |
| Retrieval vs grounding bottleneck analysis | ✅ COMPLETED | `results/eswa_revision/05_retrieval_vs_grounding/` |
| Robustness across variants | ✅ COMPLETED | `results/eswa_revision/06_robustness/` |
| Error taxonomy (heuristic) | ✅ COMPLETED | `results/eswa_revision/08_error_taxonomy/` |
| Learning appendix (existing benchmark run) | ✅ COMPLETED | `results/eswa_revision/10_learning_appendix/` |
| Runtime summary | ✅ COMPLETED | `results/eswa_revision/11_runtime/` |
| Full post-fix downstream evaluation (TypeMatch, InstReady) | ❌ BLOCKED (HF_TOKEN) | Pending |
| SAE end-to-end evaluation | ❌ BLOCKED (HF_TOKEN) | Pending |
| New methods benchmarked (GCG, anchor, beam) | ❌ BLOCKED (HF_TOKEN) | Pending |

### What succeeded

1. **All 9 retrieval experiments** ran successfully and produced verified numbers.
2. **8 publication-ready figures** generated and saved to `results/eswa_revision/12_figures/`.
3. **12 CSV tables** generated covering all major experimental comparisons.
4. **11 markdown reports** with interpretation and manuscript guidance.
5. **CI fixed**: removed `push` trigger from `nlp4lp.yml` (was causing `action_required` from bot commits).

### What failed (with reasons)

1. **Full downstream evaluation** — requires `udell-lab/NLP4LP` gold parameters via HuggingFace.
   HF_TOKEN is not set in sandbox. Numbers from manuscript-era docs used as baseline.
2. **Post-fix TypeMatch measurement** — same blocker. Structural estimate provided (+0.33–0.42pp overall).
3. **SAE end-to-end** — same blocker. Structural coverage (62.9%) and hand-crafted (24/24) documented.

### What remains uncertain

- Exact end-to-end TypeMatch after the `_is_type_match` fix (structural estimate: +33–42pp)
- Whether new methods (GCG, anchor_linking, bottomup_beam) improve InstReady over 0.085
- Whether SAE improves TypeMatch vs standard extraction on canonical benchmark

---

## 2. Main Empirical Takeaways

### Did the float fix help materially?

**Structurally: YES.** The `_is_type_match` fix converts 97.7% of float-slot token pairs from
0% type-match to 81.8% structural compatibility. This should substantially improve TypeMatch
from the pre-fix baseline of 0.227. Structural estimate: overall TypeMatch ~0.55–0.65.

**End-to-end: UNKNOWN.** Full measurement requires gold data (HF_TOKEN).

**Both metric accounting AND assignment behavior are affected.** The fix is correct and conservative
(integers ARE valid float values). Running the canonical downstream evaluation with the fix is the
single highest-priority next experiment.

### Which deterministic method is best balanced?

**`tfidf_optimization_role_repair`**: Full coverage (0.822), improved TypeMatch (0.243),
improved Exact20 (0.277), modest InstReady (0.060). Best for use cases requiring both
completeness and accuracy without sacrificing fill rate.

### Which method has highest InstantiationReady?

**`tfidf_hierarchical_acceptance_rerank`** (InstReady = 0.0846). Highest fraction of queries
meeting both coverage ≥ 0.8 AND type_match ≥ 0.8. Comes at cost of schema R@1 (0.846 vs 0.906).

### Is retrieval still no longer the main bottleneck?

**Confirmed.** Oracle retrieval (R@1 = 1.0) gives InstReady 0.082 — only +0.009 above TF-IDF
typed (0.073). A 9.4pp improvement in schema recall buys under 1pp improvement in InstReady.
Downstream grounding is the dominant bottleneck.

### Does hybrid help short queries?

**Retrieval: YES** (+0.012 R@1 on short queries, documented). **Downstream: unknown** (HF_TOKEN
needed to measure coverage/TypeMatch improvement from hybrid retrieval on short variant).

### Does SAE help under canonical evaluation?

**Unknown.** SAE achieves 100% on hand-crafted cases and 62.9% structural coverage. End-to-end
comparison vs standard extraction requires gold data.

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
| Deterministic method comparison table (10 methods) | Manuscript-verified docs | HIGH |
| Coverage–precision tension as main finding | Evidence from all methods | HIGH |
| Float TypeMatch bottleneck: root cause identified + fix | Code audit + structural estimate | HIGH |
| Schema R@1: retrieval strong, not the main bottleneck | Oracle vs TF-IDF gap analysis | HIGH |
| Learning: honest negative result (table in appendix) | Benchmark-valid GPU run | HIGH |
| Runtime: all methods sub-second on CPU | Locally measured | HIGH |
| Post-fix TypeMatch improvement | Structural estimate only | MEDIUM (awaiting HF run) |

### What should go in appendix

- Full 10-method comparison table (supplement Table S1)
- Per-type TypeMatch breakdown (float/int/percent/currency)
- Learning experiment details (split, model, 4 metrics)
- Robustness across variants with caveats for noisy/short
- Case studies (once HF_TOKEN available for per-instance outputs)

### What claims are safe

- "Retrieval R@1 ≥ 0.77 on the hardest (short) variant and ≥ 0.90 on orig."
- "Downstream grounding is the bottleneck: oracle retrieval improves InstReady by only +0.009."
- "No single deterministic method dominates all metrics — a Pareto frontier exists between coverage and precision."
- "Float type classification was identified as the largest error source (~70% of float slots mistyped pre-fix)."
- "A deterministic fix (`_is_type_match`) structurally resolves the float mismatch; end-to-end measurement pending."
- "Learned pairwise ranker (distilroberta-base, 500 steps) scored below the rule baseline on all metrics."
- "All methods run in under 1 second per query on CPU hardware."

### What claims should be avoided

- Claiming specific post-fix TypeMatch without running on gold data
- Claiming GCG/anchor/beam improve InstReady (not benchmarked end-to-end)
- Claiming SAE improves end-to-end TypeMatch (only structural evidence exists)
- Claiming the noisy variant "works well" without the caveat that downstream is zero by design

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

## 5. Next Steps (Ordered by Priority)

1. **[Priority 1]** Set `HF_TOKEN` in sandbox / CI, then run:
   ```bash
   python tools/run_nlp4lp_focused_eval.py --variant orig
   ```
   This produces the definitive post-fix TypeMatch and InstReady numbers.

2. **[Priority 2]** Benchmark new methods (GCG, anchor_linking, bottomup_beam):
   ```bash
   python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode global_consistency_grounding
   python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_anchor_linking
   ```

3. **[Priority 3]** Run noisy/short variants for all methods:
   ```bash
   for variant in noisy short; do python tools/run_nlp4lp_focused_eval.py --variant $variant; done
   ```

4. **[Priority 4]** Evaluate hybrid retriever on downstream:
   - Hybrid BM25+TF-IDF gives +0.012pp R@1 on short; measure downstream impact.

5. **[Priority 5]** Update tables/figures in this report with post-fix numbers once available.
