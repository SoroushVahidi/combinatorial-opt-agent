# Full Repo Summary — Current GitHub State

**Generated:** 2026-03-09  
**Based on:** exact files present in the GitHub-accessible clone.  
**No Wulver-only data is assumed.**

---

## 1. Manuscript sources found

There is no LaTeX/PDF manuscript in the repo. The manuscript-facing material is spread across these files (all under `docs/`):

| File | Role |
|------|------|
| `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` | Canonical source of truth: all table numbers, denominators, rounding rules, and compact summary for every reported result. |
| `docs/NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md` | Clarification checklist and polished table captions/footnotes ready for insertion into a paper. |
| `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md` | Full per-result-group verification table: formula, source file, reproducibility status. |
| `docs/NLP4LP_CHATGPT_CLARIFICATION_PACKAGE.md` | Bullet-by-bullet clarifications (random, oracle, Exact5/20, hit/miss, per-type) with exact code-line citations. |
| `docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md` | Acceptance-rerank and hierarchical-rerank result tables. |
| `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md` | Constrained assignment result tables + qualitative examples. |
| `docs/NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md` | Semantic IR repair result tables. |
| `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md` | Optimization-role repair result tables. |
| `docs/BOTTLENECK_ANALYSIS.md` | Summary of all four identified bottlenecks and their fixes. |
| `docs/Q1_JOURNAL_AUDIT.md` | Milestone roadmap and gap analysis relative to the envisioned system. |

---

## 2. Result artifacts found

### 2.1 Result CSV/JSON files

**`results/` directory does NOT exist** in the current GitHub clone. All downstream result numbers are therefore only present as manually written tables embedded in the `docs/NLP4LP_*.md` files above.  
The scripts that would generate the canonical CSV outputs are:
- `tools/nlp4lp_downstream_utility.py` → `results/paper/nlp4lp_downstream_summary.csv`
- `tools/summarize_nlp4lp_results.py` → `results/nlp4lp_retrieval_summary.csv`
- `tools/make_nlp4lp_paper_artifacts.py` → `results/paper/nlp4lp_*.csv` and `*.tex`

None of those output files are committed. The numbers in the docs files are stated to be read from those CSVs, so the docs files are the only evidence of the actual computed numbers in the GitHub repo.

### 2.2 Learning audit artifacts (present)

All CPU-only bottleneck audit outputs are committed under `artifacts/learning_audit/`:

| File | Content |
|------|---------|
| `artifacts/learning_audit/bottleneck_audit_summary.json` | 331-query slice counts (see §4 below) |
| `artifacts/learning_audit/bottleneck_audit_summary.md` | Human-readable version of the same |
| `artifacts/learning_audit/entity_association_risk_examples.jsonl` | 14 flagged queries |
| `artifacts/learning_audit/lower_upper_risk_examples.jsonl` | 116 flagged queries |
| `artifacts/learning_audit/multi_numeric_confusion_examples.jsonl` | 288 flagged queries |
| `artifacts/learning_audit/total_vs_per_unit_risk_examples.jsonl` | 191 flagged queries |
| `artifacts/learning_audit/percent_vs_absolute_risk_examples.jsonl` | 43 flagged queries |
| `artifacts/learning_audit/pairwise_data_quality.json` | Quality report (empty: pairwise ranker data absent) |
| `artifacts/learning_audit/pairwise_data_quality.md` | Human-readable version |
| `artifacts/learning_audit/pairwise_feature_analysis.json` | Feature stats (corpus-proxy mode only) |
| `artifacts/learning_audit/pairwise_feature_analysis.md` | Human-readable version |
| `artifacts/learning_audit/manual_inspection_cases.jsonl` | 89 hard cases across 4 categories |
| `artifacts/learning_audit/manual_inspection_cases.md` | Human-readable version |

### 2.3 Eval data (present)

| File | Content |
|------|---------|
| `data/processed/nlp4lp_eval_orig.jsonl` | 331 original NLP4LP queries |
| `data/processed/nlp4lp_eval_noisy.jsonl` | 331 `<num>`-masked variants |
| `data/processed/nlp4lp_eval_short.jsonl` | 331 short variants |
| `data/processed/nlp4lp_eval_noentity.jsonl` | 331 entity-stripped variants |
| `data/processed/nlp4lp_eval_nonum.jsonl` | 331 number-stripped variants |
| `data/catalogs/nlp4lp_catalog.jsonl` | NLP4LP schema catalog |

---

## 3. Methods definitely implemented in the GitHub version

All methods below are present in `tools/nlp4lp_downstream_utility.py` as callable assignment modes.

### 3.1 Retrieval baselines

| Method | Implementation file | Status |
|--------|---------------------|--------|
| `bm25` | `retrieval/baselines.py` — `BM25Baseline` | ✅ Implemented |
| `tfidf` | `retrieval/baselines.py` — `TfidfBaseline` | ✅ Implemented |
| `lsa` | `retrieval/baselines.py` — `LSABaseline` | ✅ Implemented |
| Oracle (gold schema) | `tools/nlp4lp_downstream_utility.py` line ~351 | ✅ Implemented |
| Random (seeded) | `tools/nlp4lp_downstream_utility.py` line ~352 | ✅ Implemented |
| `tfidf_acceptance_rerank` | `tools/nlp4lp_downstream_utility.py` `make_rerank_rank_fn()` | ✅ Implemented |
| `tfidf_hierarchical_acceptance_rerank` | Same, `use_hierarchy=True` | ✅ Implemented |
| `bm25_acceptance_rerank` | Same pattern with BM25 base | ✅ Implemented |
| `bm25_hierarchical_acceptance_rerank` | Same pattern with BM25 base | ✅ Implemented |

### 3.2 Downstream assignment modes

| Mode (assignment_mode flag) | Baseline label produced | Status |
|-----------------------------|-------------------------|--------|
| `typed` (default greedy) | `tfidf`, `bm25`, `lsa`, `oracle` | ✅ Implemented |
| `untyped` | `*_untyped` | ✅ Implemented |
| `constrained` | `*_constrained` | ✅ Implemented |
| `semantic_ir_repair` | `*_semantic_ir_repair` | ✅ Implemented |
| `optimization_role_repair` | `*_optimization_role_repair` | ✅ Implemented |

**Methods NOT yet implemented** (referenced in problem statement but absent from code):
- `optimization_role_relation_repair` — not present in `nlp4lp_downstream_utility.py` or anywhere in the repo
- `optimization_role_anchor_linking` — not present
- `optimization_role_bottomup_beam_repair` — not present
- `optimization_role_entity_semantic_beam_repair` — not present

These four names appear only in the problem statement; they have no corresponding code, documentation, or results in the GitHub repo.

### 3.3 Learning infrastructure (present but not executed)

| Component | File(s) | Status |
|-----------|---------|--------|
| Bottleneck audit (CPU) | `src/learning/audit_nlp4lp_bottlenecks.py` | ✅ Implemented, output committed |
| Pairwise data quality check | `src/learning/check_nlp4lp_pairwise_data_quality.py` | ✅ Implemented |
| Feature analysis | `src/learning/analyze_pairwise_features.py` | ✅ Implemented |
| Manual case export | `src/learning/export_manual_inspection_cases.py` | ✅ Implemented |
| Training env check | `src/learning/check_training_env.py` | ✅ Implemented |
| Mention-slot pair generation | `training/generate_mention_slot_pairs.py` | ✅ Implemented (no output yet) |
| Pairwise ranker training | `src/learning/train_nlp4lp_pairwise_ranker.py` | ❌ NOT YET CREATED |
| Multitask grounder training | `src/learning/train_multitask_grounder.py` | ❌ NOT YET CREATED |
| Pairwise ranker eval | `src/learning/eval_nlp4lp_pairwise_ranker.py` | ❌ NOT YET CREATED |
| Bottleneck-slice eval | `src/learning/eval_bottleneck_slices.py` | ❌ NOT YET CREATED |
| Stage-3 experiment runner | `src/learning/run_stage3_experiments.py` | ❌ NOT YET CREATED |
| Slurm batch scripts | `batch/learning/*.sbatch` | ✅ Present (infrastructure only) |

**Confirmed blocker** (`docs/LEARNING_FIRST_REAL_TRAINING_BLOCKER.md`): `torch`, `transformers`, `sentence_transformers`, `datasets`, and `accelerate` are not installed in the current Python environment; `artifacts/learning_ranker_data/nlp4lp/` does not exist; no training run has been attempted.

---

## 4. Best current repo results

All numbers below are taken from the documentation files. The underlying CSVs are not committed; these are the only evidence available in the GitHub repo.

### 4.1 Retrieval (Schema R@1 = Recall@1, N = 331)

Source: `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` §1

| Baseline | orig | noisy | short |
|----------|------|-------|-------|
| TF-IDF | **0.9063** | 0.9033 | 0.7855 |
| BM25 | 0.8852 | 0.8943 | 0.7734 |
| LSA | 0.8550 | 0.8912 | 0.7704 |
| Random (theoretical) | 0.0030 | 0.0030 | 0.0030 |

### 4.2 Downstream — typed greedy baseline (orig, N = 331)

Source: `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` §2.1

| Baseline | Schema_R@1 | param_coverage | type_match | key_overlap | Exact20_on_hits | InstantiationReady |
|----------|-----------|----------------|------------|-------------|-----------------|-------------------|
| TF-IDF | 0.9063 | 0.8222 | 0.2267 | 0.9188 | 0.2140 | **0.0725** |
| BM25 | 0.8852 | 0.8133 | 0.2251 | 0.8936 | 0.2175 | 0.0755 |
| LSA | 0.8550 | 0.7976 | 0.2063 | 0.8657 | 0.1965 | 0.0604 |
| Oracle | 1.0000 | 0.8695 | 0.2475 | 0.9953 | 0.1871 | 0.0816 |
| Random | 0.0060 | 0.0101 | 0.0060 | 0.0082 | 0.1250 | 0.0060 |

### 4.3 Downstream — all assignment modes compared (orig, TF-IDF retrieval)

Source: `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md`, `docs/NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md`, `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md`, `docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md`

| Method | param_coverage | type_match | Exact20_on_hits | InstantiationReady |
|--------|----------------|------------|-----------------|-------------------|
| `tfidf` (typed greedy) | 0.8222 | 0.2267 | 0.2140 | **0.0725** |
| `tfidf_untyped` (ablation) | 0.8222 | 0.1677 | 0.1539 | 0.0453 |
| `tfidf_constrained` | 0.7720 | 0.1980 | **0.3279** | 0.0272 |
| `tfidf_semantic_ir_repair` | 0.7780 | **0.2540** | 0.2610 | 0.0630 |
| `tfidf_optimization_role_repair` | 0.8220 | 0.2430 | 0.2770 | 0.0600 |
| `tfidf_acceptance_rerank` (Schema_R@1=0.8761) | 0.7974 | 0.2275 | — | 0.0816 |
| `tfidf_hierarchical_acceptance_rerank` (Schema_R@1=0.8459) | 0.7771 | 0.2303 | — | **0.0846** |

Key observations:
- **Best InstantiationReady:** `tfidf_hierarchical_acceptance_rerank` at **0.0846** (Schema_R@1 = 0.8459) — trades retrieval accuracy for slightly higher downstream readiness.
- **Best Exact20_on_hits (numeric precision):** `tfidf_constrained` at **0.3279** — but at the cost of low coverage (0.7720) and low InstantiationReady (0.0272).
- **Best balance of coverage + type + Exact20 without sacrificing InstantiationReady:** `tfidf_optimization_role_repair` — preserves coverage (0.8222), improves type_match and Exact20 over typed greedy, with modest InstantiationReady trade-off.
- **Typed greedy** (`tfidf`) remains the best single method by InstantiationReady among the deterministic non-rerank methods (0.0725).

### 4.4 Learning results

**None.** No trained model exists in the repo. No pairwise ranker or multitask grounder has been trained. The only learning-related evidence is the CPU-only heuristic audit in `artifacts/learning_audit/`.

---

## 5. Manuscript vs repo comparison

### 5.1 What the manuscript docs report

The manuscript-facing docs (`docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md`, `docs/NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md`) report the following as the main result set:

- Retrieval: TF-IDF Schema R@1 = **0.9063** (orig), 0.9033 (noisy), 0.7855 (short)
- Downstream typed greedy: TF-IDF InstantiationReady = **0.0725**, Oracle InstantiationReady = **0.0816**
- Typed vs untyped ablation: typed improves InstantiationReady from 0.0453 to 0.0725 (TF-IDF)
- Per-type: integer has highest type_match (~0.99), float the lowest (~0.03)
- Oracle only modestly above TF-IDF on InstantiationReady (0.0816 vs 0.0725), showing retrieval is not the only bottleneck

The newer structured methods (constrained, semantic_ir_repair, optimization_role_repair, acceptance rerank) are documented in their own `docs/NLP4LP_*_RESULTS.md` files but are **not yet in the main reporting package** — suggesting they have not been incorporated into the manuscript tables.

### 5.2 What the repo evidences

The repo evidences exactly the numbers in the docs files. No independent CSV or JSON result file is committed to contradict or confirm those numbers. All result numbers are therefore:
- Reproducible (scripts and data exist to regenerate them)
- But not independently verified by committed output artifacts (the `results/` directory is absent)

### 5.3 Mismatch assessment

| Dimension | Manuscript docs say | Repo evidence | Assessment |
|-----------|--------------------|--------------------|------------|
| Retrieval R@1 | TF-IDF 0.9063 orig | Numbers in docs only (no CSV committed) | Consistent; reproducible from code + eval data |
| Best downstream method (InstantiationReady) | `tfidf` typed greedy = 0.0725 | Docs show hierarchical rerank = 0.0846 and acceptance rerank = 0.0816 | **Manuscript understates current best** — hierarchical rerank (0.0846) is not in the main reporting package |
| Best downstream method (Exact20) | Not highlighted | Docs show `tfidf_constrained` = 0.3279 | **Manuscript does not yet report the constrained Exact20 result** |
| optimization_role_repair | Not in main reporting package | Documented in `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md` | **Present in repo docs but not in the main manuscript result set** |
| Learning results | Not claimed | None exist | Consistent — no gap |
| Newer structured methods (relation_repair, anchor_linking, bottomup_beam, entity_semantic_beam) | Not claimed | Not implemented | Consistent — no gap |

---

## 6. Current honest conclusion for the GitHub version

### What is definitely implemented

1. **Three retrieval baselines:** BM25, TF-IDF, LSA — fully implemented, evaluated, numbers documented.
2. **Five downstream assignment modes:** typed greedy, untyped, constrained, semantic_ir_repair, optimization_role_repair — all implemented in `tools/nlp4lp_downstream_utility.py`.
3. **Acceptance reranking variants:** tfidf/bm25 × flat/hierarchical — implemented; results documented.
4. **Full evaluation pipeline:** `training/run_baselines.py`, `tools/nlp4lp_downstream_utility.py`, `tools/make_nlp4lp_paper_artifacts.py` — all present and runnable.
5. **CPU-only bottleneck audit infrastructure** — scripts + output artifacts committed.

### What is definitely evaluated

- All retrieval baselines on orig/noisy/short (331 queries each) — numbers in docs.
- All five assignment modes with TF-IDF and Oracle on orig — numbers in docs.
- Typed/untyped ablation — numbers in docs.
- Per-type (currency/float/integer/percent) breakdown — numbers in docs.
- Acceptance rerank variants — numbers in docs.

### What is only infrastructure / partial / not supported by result evidence

- **Learning (trained models):** Only infrastructure. No trained pairwise ranker or multitask grounder. The 5 key training scripts do not yet exist (`train_nlp4lp_pairwise_ranker.py`, etc.).
- **Stage-3 experiment matrix:** Config exists (`configs/learning/experiment_matrix_stage3.json`), sbatch scripts exist, but no runs have been submitted.
- **Pairwise ranker data:** `artifacts/learning_ranker_data/nlp4lp/` does not exist.
- **Newer structured methods** (relation_repair, anchor_linking, bottomup_beam, entity_semantic_beam): Not implemented anywhere in the repo.
- **Committed result CSVs:** The `results/` directory does not exist in the GitHub clone.

### Whether the paper's current honest conclusion should be

Based on the evidence in the GitHub repo:

1. **"Retrieval is no longer the main bottleneck"** — ✅ Supported. TF-IDF Schema R@1 is 0.9063; Oracle InstantiationReady (0.0816) is barely above TF-IDF (0.0725), confirming the bottleneck is downstream, not retrieval.

2. **"Downstream number-to-slot grounding remains the main bottleneck"** — ✅ Supported. Even with perfect retrieval (oracle), InstantiationReady is only 0.0816. TypeMatch is 0.2267–0.2475 across methods; float type_match is ~0.03. The bottleneck audit shows 87% of queries have ≥3 confusable numeric values.

3. **"optimization_role_repair is still the main winning downstream method"** — ⚠️ Partially supported. Among deterministic non-rerank methods, `optimization_role_repair` gives the best balance of coverage + type_match + Exact20. But for InstantiationReady specifically, `tfidf` typed greedy (0.0725) edges it (0.0600), and `tfidf_hierarchical_acceptance_rerank` (0.0846) outperforms both. The "winning" method depends on which metric is prioritized.

4. **"Newer structured methods are useful explorations but not overall winners"** — ✅ Supported. No newer method dominates on all metrics. `tfidf_constrained` wins on Exact20 but collapses on InstantiationReady. `tfidf_acceptance_rerank` and hierarchical variant improve InstantiationReady but hurt Schema R@1.

---

## 7. Missing evidence / caveats

1. **No committed result CSVs.** The `results/` directory is absent. All result numbers come from prose tables in docs files. These are stated to be reproducible from `tools/nlp4lp_downstream_utility.py` + eval data, but the final outputs are not independently committed. Any reviewer wanting to verify a number must rerun the pipeline.

2. **No trained learning models.** The entire `src/learning/` training stack is unexecuted. The blocking issues are: (a) torch/transformers not installed, (b) pairwise ranker data not generated, (c) training scripts not written. The GitHub repo therefore has **zero learning results** beyond the heuristic audit.

3. **Newer structured methods absent.** `optimization_role_relation_repair`, `optimization_role_anchor_linking`, `optimization_role_bottomup_beam_repair`, and `optimization_role_entity_semantic_beam_repair` — none of these are implemented or referenced anywhere in the code.

4. **Acceptance rerank results not in main reporting package.** The best InstantiationReady result in the repo (0.0846 for hierarchical acceptance rerank) is documented only in `docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md` and not incorporated into the main manuscript table docs.

5. **Random baseline inconsistency (known).** Retrieval table uses theoretical random 1/331 = 0.0030; downstream uses empirical 2/331 = 0.0060. Documented and justified in `docs/NLP4LP_MANUSCRIPT_CONSISTENCY_PLAN.md` §3; requires explicit statement in paper captions.

6. **Noisy variant instantiation_ready = 0** for all baselines (type_match = 0 because `<num>` placeholders are not recovered). Short variant is near-zero. These are correct and documented but require explanation in the paper.

---

## 8. Ready-to-paste summary for ChatGPT

```
PROJECT: Natural-language optimization problem instantiation (NLP4LP pipeline).
Three stages: (1) retrieve the right optimization schema, (2) extract numeric mentions
from the query, (3) assign numbers to scalar slots in the schema.
Test set: 331 queries per variant (orig, noisy, short). No LaTeX manuscript or PDF
committed; all numbers come from docs/ files.

--- RETRIEVAL ---
Schema R@1 (Recall@1) on orig: TF-IDF 0.9063, BM25 0.8852, LSA 0.8550.
Retrieval is strong. TF-IDF is best.

--- DOWNSTREAM (orig, N=331) ---
Main metric: InstantiationReady = fraction of queries with param_coverage ≥ 0.8
AND type_match ≥ 0.8.

Method                                     | Coverage | TypeMatch | Exact20 | InstReady
tfidf (typed greedy, baseline)             | 0.8222   | 0.2267    | 0.2140  | 0.0725
tfidf_constrained                          | 0.7720   | 0.1980    | 0.3279  | 0.0272
tfidf_semantic_ir_repair                   | 0.7780   | 0.2540    | 0.2610  | 0.0630
tfidf_optimization_role_repair             | 0.8220   | 0.2430    | 0.2770  | 0.0600
tfidf_acceptance_rerank                    | 0.7974   | 0.2275    |   —     | 0.0816
tfidf_hierarchical_acceptance_rerank       | 0.7771   | 0.2303    |   —     | 0.0846 ← best
oracle (typed greedy, perfect retrieval)   | 0.8695   | 0.2475    | 0.1871  | 0.0816

KEY FINDING 1: Even with perfect retrieval (oracle), InstantiationReady = 0.082.
The bottleneck is NOT retrieval; it is downstream number-to-slot grounding.

KEY FINDING 2: TypeMatch on hits is only 0.23–0.25 across all methods; float
type_match ≈ 0.03 (hardest type). 87% of queries have ≥3 confusable numeric values.

KEY FINDING 3: No single method dominates all metrics. tfidf_constrained is best on
Exact20 (0.328) but worst on InstantiationReady (0.027). tfidf_hierarchical_acceptance_rerank
is best on InstantiationReady (0.085) but lowers Schema R@1 to 0.846. The typed-greedy
tfidf baseline is the most balanced deterministic method.

KEY FINDING 4: optimization_role_repair gives the best balance (preserves coverage,
improves TypeMatch and Exact20 over typed greedy) among the non-rerank structured methods.

--- WHAT IS NOT YET IN THE REPO ---
- No trained learning models (torch not installed; training scripts not written).
- No committed result CSVs (results/ directory absent; numbers only in docs files).
- optimization_role_relation_repair, anchor_linking, bottomup_beam_repair,
  entity_semantic_beam_repair: NOT implemented anywhere.
- Stage-3 experiment matrix exists as config/scripts but has never been run.

--- HONEST PAPER CONCLUSION ---
1. Retrieval is no longer the main bottleneck (evidence: oracle barely moves InstReady).
2. Downstream number-to-slot grounding is the main bottleneck.
3. optimization_role_repair is the best-balanced structured deterministic method.
4. Hierarchical acceptance rerank gives highest InstantiationReady (0.085) but
   at the cost of retrieval accuracy.
5. No learning results exist yet; all results are deterministic heuristic methods.
```
