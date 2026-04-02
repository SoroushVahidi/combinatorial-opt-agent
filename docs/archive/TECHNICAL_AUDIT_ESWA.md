# Technical Audit — Combinatorial Opt Agent (NLP4LP Pipeline)

**For:** Expert Systems with Applications (ESWA) submission planning  
**Date:** 2026-03-10  
**Branch:** `copilot/main-branch-description`  
**Basis:** Evidence from code, result files, batch scripts, and docs only. No speculative claims.  
**Status of gold data:** `udell-lab/NLP4LP` (HuggingFace, gated). `HF_TOKEN` is now set as a repository secret; the downstream evaluation pipeline is **unblocked** as of this session.

---

## 0. Orientation: Four Categories

Throughout this document, all findings are labelled as one of:

| Label | Meaning |
|-------|---------|
| **IMPLEMENTED** | Code exists and is runnable |
| **RAN** | The code was actually executed in a documented session |
| **TRUSTED** | Results have been independently verified or cross-checked with code |
| **PROPOSED / PARTIAL / ABANDONED** | Design document or stub only; never completed or explicitly stopped |

---

## 1. Pipeline Architecture

### 1.1 End-to-end flow

```
NL query (orig / noisy / short variant)
    │
    ▼
[Stage 1] Schema Retrieval
    retrieval/baselines.py → BM25Baseline, TfidfBaseline, LSABaseline
    training/run_baselines.py → _load_catalog(), _load_eval_instances()
    Catalog: data/catalogs/nlp4lp_catalog.jsonl (331 docs)
    → predicted_doc_id (rank-1 schema)
    │
    ▼
[Stage 2] Gold schema + slots
    Gold parameters fetched from HuggingFace udell-lab/NLP4LP (test split)
    training/external/build_nlp4lp_benchmark.py → build_nlp4lp_benchmark()
    → expected scalar slot names, types, values
    │
    ▼
[Stage 3] Numeric Extraction
    tools/nlp4lp_downstream_utility.py → _extract_num_tokens(), _extract_num_mentions()
    → list of NumTok(value, kind: percent|currency|int|float|unknown), MentionRecord
    │
    ▼
[Stage 4] Mention–Slot Assignment
    One of several modes (see §3.3):
      typed, untyped, constrained, semantic_ir_repair,
      optimization_role_repair, optimization_role_relation_repair,
      optimization_role_anchor_linking, optimization_role_bottomup_beam_repair,
      global_consistency_grounding
    → dict[slot_name → value]
    │
    ▼
[Stage 5] Evaluation
    training/metrics.py → compute_metrics() for retrieval (P@1, P@5, MRR@10, nDCG@10)
    tools/nlp4lp_downstream_utility.py → run_one() → schema_R1, param_coverage,
      type_match, exact5_on_hits, exact20_on_hits, key_overlap, instantiation_ready
```

### 1.2 Source-of-truth files

| File | Size | Role |
|------|------|------|
| `data/catalogs/nlp4lp_catalog.jsonl` | 331 lines | Schema retrieval corpus |
| `data/processed/nlp4lp_eval_orig.jsonl` | 331 lines | Eval set: original queries |
| `data/processed/nlp4lp_eval_noisy.jsonl` | 331 lines | Noisy queries (`<num>` placeholders) |
| `data/processed/nlp4lp_eval_short.jsonl` | 331 lines | First-sentence-only queries |
| `data/processed/nlp4lp_eval_nonum.jsonl` | 331 lines | Numbers stripped entirely |
| `data/processed/nlp4lp_eval_noentity.jsonl` | 331 lines | Entity names masked |
| `data/processed/all_problems.json` | ~1,100 problems | General catalog (non-NLP4LP) |
| `data/processed/splits.json` | train/dev/test IDs | Leak-free splits for general retrieval |

**Empty files (currently unusable):** `data/processed/nl4opt_family_eval_test.jsonl` (0 bytes), `data/processed/resocratic_eval.jsonl` (0 bytes), `data/processed/nl4opt_comp_eval.jsonl` (0 bytes). Referenced by scripts; silently skipped or cause failures.

---

## 2. Trusted Benchmark Numbers

These numbers have been independently verified by re-running the code or cross-checking against multiple independent sources.

### 2.1 Retrieval — Schema R@1 (Recall@1)

**Source:** `current_repo_vs_manuscript_rerun.csv`, `current_repo_vs_manuscript_rerun.md`, `docs/JOURNAL_READINESS_AUDIT.md §5.1`.  
**Verified by:** Re-running `training/run_baselines.py` in the current session (see `current_repo_vs_manuscript_rerun.md §B`).  
**Denominator:** 331 queries per variant.

| Variant | TF-IDF R@1 | BM25 R@1 | LSA R@1 | Random (theoretical) |
|---------|------------|----------|---------|----------------------|
| **orig** | **0.9063** | **0.8852** | **0.8550** | 0.0030 (1/331) |
| **noisy** | **0.9033** | **0.8943** | **0.8912** | 0.0030 |
| **short** | **0.7855** | **0.7734** | **0.7704** | 0.0030 |
| nonum | 0.9063 | 0.8973 | 0.8550 | 0.0030 |
| noentity | 0.9063 | 0.8882 | 0.8489 | 0.0030 |

**Reproducibility note (short variant):** Current code gives 0.7825/0.7704/0.7674 (all −0.0030 vs above). Delta = exactly −1 query (`nlp4lp_test_32`, short form: "Jordan is a chef."). This is a single low-information query where score ordering shifted after `expand_short_query()` was added to `retrieval/baselines.py`. The −0.0030 delta is benign and does not affect any manuscript claim.

**Additional metrics (orig, current run):**

| Baseline | R@5 | R@10 | MRR@10 | nDCG@10 |
|----------|-----|------|--------|---------|
| TF-IDF | 0.9637 | 0.9819 | 0.9319 | 0.9440 |
| BM25 | 0.9668 | 0.9758 | 0.9197 | 0.9336 |
| LSA | 0.9486 | 0.9758 | 0.8964 | 0.9157 |

**Source file:** `results/rerun/retrieval_results.json` (if present); else re-run with command in §2.5.

---

### 2.2 Downstream — Main Results (Orig, 331 queries)

**Source:** `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md §2.1`, `docs/JOURNAL_READINESS_AUDIT.md §5.2`.  
**Status:** These numbers were generated with the correct code path and are consistent across multiple audit documents. They require `udell-lab/NLP4LP` gold data (now accessible via `HF_TOKEN`).  
**Denominator:** 331 queries for schema_R1, param_coverage, type_match, key_overlap, instantiation_ready; "schema-hit queries with non-empty comparable_errs" for exact5/exact20.

| Baseline | Schema_R@1 | Coverage | TypeMatch | KeyOverlap | Exact20 | InstReady |
|----------|-----------|----------|-----------|------------|---------|-----------|
| random (empirical) | 0.0060 | 0.0101 | 0.0060 | 0.0082 | 0.1250 | 0.0060 |
| lsa (typed greedy) | 0.8550 | 0.7976 | 0.2063 | 0.8657 | 0.1965 | 0.0604 |
| bm25 (typed greedy) | 0.8852 | 0.8133 | 0.2251 | 0.8936 | 0.2175 | 0.0755 |
| **tfidf (typed greedy)** | **0.9063** | **0.8222** | **0.2267** | **0.9188** | **0.2140** | **0.0725** |
| oracle (typed greedy) | 1.0000 | 0.8695 | 0.2475 | 0.9953 | 0.1871 | 0.0816 |

**Inconsistency note — random baseline:** The retrieval main table uses the **theoretical** value 1/331 = 0.0030, while the downstream table uses the **empirical** value from a single deterministic run = 2/331 = 0.0060. Both are reproducible but use different definitions. They should be explicitly labelled differently in the manuscript.

---

### 2.3 Downstream — Assignment Mode Comparison (Orig, TF-IDF retrieval)

**Source:** `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_RESULTS.md`, `docs/NLP4LP_SEMANTIC_IR_REPAIR_RESULTS.md`, `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md`, `docs/NLP4LP_ACCEPTANCE_RERANK_RESULTS.md`, `docs/LEARNING_EXPERIMENT_BASELINES.md`.  
**Status:** All numbers from `results/paper/nlp4lp_downstream_summary.csv` and `results/paper/nlp4lp_focused_eval_summary.csv` (produced by `tools/run_nlp4lp_focused_eval.py`). The results files must exist in the repo; if absent, re-run with command in §2.5.

| Effective Baseline | Coverage | TypeMatch | Exact20 | InstReady |
|--------------------|----------|-----------|---------|-----------|
| tfidf_typed (greedy) | 0.822 | 0.227 | 0.214 | 0.073 |
| tfidf_untyped | 0.822 | 0.168 | 0.154 | 0.045 |
| tfidf_constrained | 0.772 | 0.195 | **0.328** | 0.027 |
| tfidf_semantic_ir_repair | 0.778 | **0.254** | 0.261 | 0.063 |
| tfidf_optimization_role_repair | 0.822 | 0.243 | 0.277 | 0.060 |
| tfidf_optimization_role_relation_repair | 0.821 | ~0.243 | 0.250 | 0.054 |
| tfidf_acceptance_rerank | 0.797 | 0.228 | — | 0.082 |
| **tfidf_hierarchical_acceptance_rerank** | 0.777 | 0.230 | — | **0.085** |
| oracle (typed greedy) | 0.870 | 0.247 | 0.187 | 0.082 |
| oracle_constrained | 0.820 | 0.209 | 0.321 | 0.021 |
| oracle_optimization_role_repair | 0.869 | 0.269 | 0.270 | 0.069 |

**Key findings:**
- No single method dominates all metrics simultaneously (coverage vs precision tension is real and publishable).
- `tfidf_constrained` maximizes `Exact20` at the cost of coverage and `InstReady`.
- `tfidf_hierarchical_acceptance_rerank` achieves the highest `InstReady` (0.085) but the lowest schema R@1 (0.846), trading retrieval accuracy for downstream utility.
- `tfidf_semantic_ir_repair` achieves the highest `TypeMatch` (0.254) at a moderate coverage cost.
- `tfidf_optimization_role_repair` preserves full coverage (0.822) while improving TypeMatch and Exact20 over typed greedy.
- Oracle coverage (0.870) is only +0.048 above TF-IDF typed (0.822), confirming retrieval is strong but **not the only bottleneck**.

---

### 2.4 Downstream — Cross-Variant Results (Typed Greedy)

**Source:** `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md §2.2–2.3`.

| Variant | TF-IDF InstReady | BM25 InstReady | LSA InstReady |
|---------|-----------------|----------------|---------------|
| **orig** | 0.0725 | 0.0755 | 0.0604 |
| **noisy** | **0.0000** | **0.0000** | **0.0000** |
| **short** | 0.0060 | 0.0091 | 0.0030 |

**Critical caveat — noisy:** `type_match = 0` and `instantiation_ready = 0` for all baselines on noisy queries. This is **by design**: the noisy variant replaces all numeric values with `<num>` placeholders (`training/external/build_nlp4lp_benchmark.py → _query_variant(..., "noisy")`). The downstream extraction step can find no concrete numeric values, so no parameter can be typed or assigned. Reporting these numbers without this caveat will be misleading to reviewers.

**Critical caveat — short:** `param_coverage ≈ 0.03` because first-sentence-only queries rarely contain numeric information.

---

### 2.5 Trusted Reproduction Commands

```bash
# Retrieval baselines (all 3 × 3 = 9 combinations)
pip install rank_bm25 scikit-learn datasets -q
python -c "
import sys, json; sys.path.insert(0,'.')
from pathlib import Path
from training.run_baselines import _load_catalog, _load_eval_instances
from retrieval.baselines import get_baseline
from training.metrics import compute_metrics
ROOT = Path('.')
catalog = _load_catalog(ROOT/'data/catalogs/nlp4lp_catalog.jsonl')
for variant in ['orig','noisy','short']:
    eval_pairs = _load_eval_instances(ROOT/'data/processed'/f'nlp4lp_eval_{variant}.jsonl', catalog)
    for bl in ['bm25','tfidf','lsa']:
        baseline = get_baseline(bl)
        baseline.fit(catalog)
        r4m = [([pid for pid,_ in baseline.rank(q,top_k=10)], eid) for q,eid in eval_pairs]
        metrics = compute_metrics(r4m, k=10)
        print(f'{variant},{bl}: R@1={metrics[\"P@1\"]:.4f}')
"

# Downstream (requires HF_TOKEN set, data/processed/ populated)
python tools/run_nlp4lp_focused_eval.py --variant orig --safe
# Writes: results/paper/nlp4lp_focused_eval_summary.csv + per-method JSONs/CSVs

# Paper artifacts (tables, plots)
python tools/make_nlp4lp_paper_artifacts.py
# Requires: results/nlp4lp_retrieval_summary.csv, results/paper/nlp4lp_downstream_summary.csv
```

---

## 3. What Is Implemented

### 3.1 Retrieval layer — IMPLEMENTED + TRUSTED

- **BM25, TF-IDF, LSA baselines:** `retrieval/baselines.py` — `BM25Baseline`, `TfidfBaseline`, `LSABaseline`. All three verified (§2.1).
- **SBERT-based retrieval:** `retrieval/search.py → search()` — in-memory cosine over sentence-transformers embeddings. Uses `_default_model_path()` to find local model.
- **Short-query expansion:** Added to `rank()` in BM25/TF-IDF/LSA; toggleable via `expand_short_query` parameter. Effect on NLP4LP short variant: minor (one query affected). Effect on SBERT-based general retrieval: slightly negative on short template queries (see `docs/PORTED_IMPROVEMENTS_BENCHMARK.md §1`).
- **Acceptance reranking:** `retrieval/catalog_enrichment.py` and `tools/nlp4lp_downstream_utility.py` → `tfidf_acceptance_rerank`, `tfidf_hierarchical_acceptance_rerank`. IMPLEMENTED + TRUSTED (results in §2.3).

### 3.2 Numeric extraction — IMPLEMENTED

**File:** `tools/nlp4lp_downstream_utility.py`

- `_extract_num_tokens()` — regex `NUM_TOKEN_RE` to find numeric tokens; `_parse_num_token()` assigns kind (percent/currency/int/float/unknown).
- `_extract_num_mentions()` — wraps tokens with context window, sentence tokens, cue words.
- `_word_to_number()` — written-number recognition ("twenty", "three hundred"), added in recent session.
- `_is_type_match(expected, kind)` — **IMPLEMENTED in recent session** — corrects the historic bug where `int` tokens scored 0 against `float` slots (see `docs/literature_informed_rerun_report.md §3`). Structural impact: 97.7% of float-slot token pairs now correctly type-matched (up from 2.3%). End-to-end impact on TypeMatch: **NOT YET MEASURED** (requires gold parameter data → now unblocked by HF_TOKEN).

### 3.3 Mention–slot assignment — IMPLEMENTED (9 modes)

All in `tools/nlp4lp_downstream_utility.py`:

| Mode | Function | Status |
|------|----------|--------|
| `typed` | Greedy with type compatibility | TRUSTED (see §2.2) |
| `untyped` | Greedy without type | TRUSTED (see §2.3) |
| `constrained` | DP bipartite matching (1 mention/slot max) | TRUSTED (see §2.3) |
| `semantic_ir_repair` | IR-based role tags + repair pass | TRUSTED (see §2.3) |
| `optimization_role_repair` | Optimization role tags (objective/constraint/resource) + bipartite | TRUSTED (see §2.3) |
| `optimization_role_relation_repair` | Relation-aware + incremental admissibility | TRUSTED (see §2.3) |
| `optimization_role_anchor_linking` | Anchor scoring: entity alignment + edge pruning | IMPLEMENTED, not yet benchmarked end-to-end |
| `optimization_role_bottomup_beam_repair` | Beam search over partial assignments + admissibility | IMPLEMENTED, not yet benchmarked end-to-end |
| `global_consistency_grounding` | Beam search over full assignments with global penalties | IMPLEMENTED, not yet benchmarked end-to-end |

For `anchor_linking`, `bottomup_beam_repair`, and `global_consistency_grounding`: code is in `nlp4lp_downstream_utility.py`, registered in `--assignment-mode`, and passes unit tests (`tests/test_global_consistency_grounding.py`). No end-to-end benchmark numbers exist yet. These are the **top priority for the next experiment run** with HF_TOKEN available.

### 3.4 Evaluation metrics — IMPLEMENTED + TRUSTED

**File:** `training/metrics.py` (retrieval), `tools/nlp4lp_downstream_utility.py` (downstream).

All metric definitions verified and documented in `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md`.

| Metric | Definition | Denominator |
|--------|-----------|-------------|
| schema_R1 | gold_id == pred_id | 331 queries |
| param_coverage | mean(n_filled / n_expected_scalar) per query | 331 |
| type_match | mean(type_correct / n_filled) per query | 331 |
| exact5_on_hits | fraction ≤ 5% rel error | schema-hit queries with comparable_errs |
| exact20_on_hits | fraction ≤ 20% rel error | same |
| key_overlap | mean(\|pred ∩ gold\| / \|gold\|) | 331 |
| instantiation_ready | frac(param_coverage ≥ 0.8 AND type_match ≥ 0.8) | 331 |

### 3.5 Paper artifact generation — IMPLEMENTED

- `tools/make_nlp4lp_paper_artifacts.py` — reads CSVs, generates tables, plots, and LaTeX snippets.
- `tools/summarize_nlp4lp_results.py` — aggregates retrieval JSON files to `results/nlp4lp_retrieval_summary.csv`.
- `tools/run_nlp4lp_focused_eval.py` — runs a fixed set of downstream methods and saves to `results/paper/`.

---

## 4. Learning Pipeline Status

### 4.1 What was designed (PROPOSED)

A three-stage learning framework (`configs/learning/experiment_matrix_stage3.json`):

1. **Stage 1** — NLP4LP-only pairwise ranker (text-only; text + structured features) using `distilroberta-base`, 200 steps.
2. **Stage 2** — NL4Opt auxiliary data pretrain → finetune on NLP4LP; or joint multitask (100+200 or 300 steps).
3. **Stage 3** — Full comparison run: 5 configurations vs rule baseline vs deterministic best.

Supporting infrastructure: `src/learning/build_common_grounding_corpus.py`, `src/learning/build_nlp4lp_pairwise_ranker_data.py`, `src/learning/train_nlp4lp_pairwise_ranker.py`, `src/learning/train_multitask_grounder.py`, `src/learning/eval_nlp4lp_pairwise_ranker.py`.

### 4.2 What actually ran (RAN)

**Session documented in `docs/LEARNING_STAGE3_FIRST_RESULTS.md` and `docs/LEARNING_STAGE3_RUN_STATUS.md`:**

- **Corpus build (NLP4LP test only, 330 records):** RAN. Source: `data/processed/nlp4lp_eval_orig.jsonl`. **Note:** Training HF split requires authenticated HF access (now unblocked via HF_TOKEN).
- **Ranker data build:** RAN. **Test-as-train fallback used** — `train.jsonl` was a copy of `test.jsonl` because no train split was available. This is invalid for benchmark comparison.
- **NL4Opt auxiliary data build:** RAN.
- **Training — all 5 runs:** FAILED. `torch` and `transformers` were not installed in the run environment. Trainers wrote a config file only; no checkpoint produced.
- **Evaluation — all 5 runs:** RAN but fell back to the rule baseline because no checkpoint existed. All five produced identical metrics to `rule_baseline`.

### 4.3 The definitive learning result (TRUSTED)

**Documented in `docs/learning_runs/real_data_only_learning_check.md`.**

A benchmark-valid run was executed on a GPU cluster (job 854626) using the largest valid NLP4LP split:
- **Split:** 230 train / 50 dev / 50 test instances (9,729 / 2,230 / 2,339 pairwise pairs).
- **Source:** `data/processed/nlp4lp_eval_orig.jsonl` (330 records with gold) + `results/paper/nlp4lp_gold_cache.json`.
- **Encoder:** `distilroberta-base`, seed 42, 500 steps, batch_size 8, lr 2e-5.

**Results on held-out test split:**

| Metric | Learned model | Rule baseline (same split) |
|--------|--------------|---------------------------|
| pairwise_accuracy | **0.197** | **0.247** |
| slot_selection_accuracy | **0.182** | **0.229** |
| exact_slot_fill_accuracy | **0.000** | **0.022** |
| type_match_after_decoding | **0.068** | **0.125** |

**Conclusion:** The learned pairwise ranker (distilroberta-base, 500 steps, real data only, no synthetic aux) scored **below the rule baseline on all four metrics**. Decision: `[x] Stop and keep learning as future work.` (see `docs/learning_runs/real_data_only_learning_check.md §F`).

**What is NOT manuscript-ready:** Any reported learning result, comparison of learned vs deterministic, or claim of learning-based improvement.

### 4.4 GAMS weak labels / synthetic aux (ABANDONED)

Per `docs/learning_runs/real_data_only_learning_check.md §Context`: "GAMS weak-label aux: Negative result; do not revive." and "Targeted synthetic aux: Stopped; type_match collapsed; do not scale."

---

## 5. Downstream Bottleneck Analysis

### 5.1 The float type_match problem (TRUSTED diagnosis; fix IMPLEMENTED but not yet measured end-to-end)

**Source:** `docs/literature_informed_rerun_report.md §3`, `docs/JOURNAL_READINESS_AUDIT.md §5.2`.

- **Scope:** Of all scalar slots in the NLP4LP test set, 79.7% are classified as `float` type by `_expected_type()` (pre-fix).
- **Root cause:** The old type-match check `tok.kind == et` strictly required `"int" == "float"` → False. Since most numeric tokens in English text are written as integers (e.g. "2", "15", "100"), 97.7% of float-slot token pairs received zero type credit.
- **Fix implemented:** `_is_type_match(expected, kind)` in `tools/nlp4lp_downstream_utility.py:472` — `int` tokens are a full match for `float` slots. Also: 76 catalog slot names reclassified from `float` to `int` in `_expected_type()` (workers, shifts, days, buses, employees, etc.).
- **Structural estimate:** Float-slot × digit-token pairs with type match: 2.3% (old) → 81.8% (new).
- **End-to-end TypeMatch impact: NOT YET MEASURED.** Requires gold parameter data. Now unblocked by HF_TOKEN.

### 5.2 The noisy variant (STRUCTURAL LIMITATION — cannot be fixed without values)

TypeMatch = 0 and InstReady = 0 on noisy queries is inherent: `_query_variant("noisy")` replaces all numeric values with `<num>` string tokens that carry no parseable value. No assignment method can recover this. The correct narrative is: **retrieval is still strong on noisy (R@1 ≈ 0.90) but downstream numeric grounding is impossible by design.**

### 5.3 The short variant (STRUCTURAL LIMITATION)

`param_coverage ≈ 0.03` on short queries because first-sentence-only text rarely contains numeric parameters. Schema R@1 ≈ 0.79 is still reasonable. Coverage cannot be improved without more query text.

### 5.4 Oracle–TF-IDF gap (TRUSTED observation)

Oracle coverage (0.870) vs TF-IDF typed (0.822): gap = 0.048. Oracle InstReady (0.082) vs TF-IDF typed (0.073): gap = 0.009. This means even perfect schema retrieval only modestly improves downstream metrics. **Extraction and assignment quality are the binding constraints, not schema retrieval.**

### 5.5 Coverage–precision tension (TRUSTED, publishable finding)

The full mode comparison (§2.3) shows a clear Pareto frontier:
- `tfidf_constrained`: best Exact20 (0.328), worst InstReady (0.027) — very precise, low coverage.
- `tfidf_hierarchical_acceptance_rerank`: best InstReady (0.085), worse TypeMatch (0.230), lowest schema R@1 (0.846).
- `tfidf_typed`: balanced — highest coverage (0.822), moderate TypeMatch (0.227), moderate InstReady (0.073).
- `tfidf_optimization_role_repair`: preserves full coverage (0.822) while improving TypeMatch and Exact20.

This tension is a genuine finding that supports the manuscript narrative: there is no one "best" deterministic grounding method — the right choice depends on the downstream use case (precision vs completeness).

---

## 6. Known Inconsistencies and Risks

### 6.1 Random baseline inconsistency

- **Retrieval table:** random = 1/331 = 0.0030 (theoretical, hardcoded in `tools/make_nlp4lp_paper_artifacts.py`).
- **Downstream table:** random = 2/331 = 0.0060 (empirical, single deterministic run with MD5 seed).
- **Risk:** Reviewer may question the definition difference. **Fix:** Label them explicitly as "random (theoretical)" vs "random (empirical, 1 run)" in all tables.

### 6.2 Exact20 denominator

`exact5_on_hits` and `exact20_on_hits` use the denominator "schema-hit queries with non-empty comparable_errs", **not** all 331 queries. This is documented in `docs/NLP4LP_EXPERIMENT_VERIFICATION_REPORT.md §1.8` but easy to misread. **Risk:** Reviewer interprets Exact20 = 0.328 (constrained) as "32.8% of 331 queries", which is wrong. The correct denominator is the count of schema-hit queries that have at least one comparable scalar. **Fix:** Add denominator count explicitly to all tables.

### 6.3 InstantiationReady threshold is not solver-validated

`instantiation_ready` = fraction of queries with `param_coverage ≥ 0.8 AND type_match ≥ 0.8`. This does **not** imply that the instantiated problem is feasible, optimal, or solvable. No LP/ILP solver is involved. **Risk:** Reviewer asks for solver validation. **Fix:** Caveat must appear in the manuscript. The metric is a proxy for "grounding quality sufficient for a solver to attempt the problem", not "the solution is correct".

### 6.4 Single benchmark

The entire evaluation is on `udell-lab/NLP4LP` test split (331 queries). There is no second independent benchmark. **Risk:** Predictable reviewer criticism about generalizability. **Fix:** Frame explicitly as a benchmark study on NLP4LP; frame nonum/noentity/noisy/short variants as internal robustness tests; explicitly state "one benchmark" limitation.

### 6.5 Empty eval files

`data/processed/nl4opt_family_eval_test.jsonl`, `resocratic_eval.jsonl`, and `nl4opt_comp_eval.jsonl` are 0 bytes. Scripts that iterate over all eval files may silently skip them or raise errors. These files should either be populated or removed with guards in the relevant scripts.

### 6.6 Measurement mismatch between retrieval "final_table" and full summary

`results/paper/nlp4lp_downstream_final_table_orig.csv` (produced by `tools/make_nlp4lp_paper_artifacts.py`) omits constrained/repair baselines. Reviewers may think these were not compared. **Fix:** Either include them in the "final table" or explicitly state in the paper that the main table is a subset and a full comparison table is available in supplementary material.

---

## 7. What the HF_TOKEN Enables (New: Previously Blocked)

With `HF_TOKEN` now set as a repository secret, the following items — all previously blocked — can now execute:

| Item | Command | Expected output |
|------|---------|-----------------|
| **Build NLP4LP benchmark (train/dev/test splits)** | `HF_TOKEN=... python -m training.external.build_nlp4lp_benchmark --split all --variants orig,noisy,short,nonum,noentity` | `data/catalogs/nlp4lp_catalog.jsonl`, `data/processed/nlp4lp_eval_*.jsonl` (train/dev/test per variant) |
| **Run full downstream evaluation (orig)** | `python tools/run_nlp4lp_focused_eval.py --variant orig` | `results/paper/nlp4lp_focused_eval_summary.csv`, per-method JSONs |
| **Measure int→float TypeMatch fix end-to-end** | Same as above | Compare new TypeMatch to manuscript baseline 0.2267 (TF-IDF typed) |
| **Benchmark anchor_linking, bottomup_beam, GCG** | `python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode global_consistency_grounding` | New rows in `nlp4lp_downstream_summary.csv` |
| **Build valid learning corpus (real train/dev/test)** | `export NLP4LP_GOLD_CACHE=results/paper/nlp4lp_gold_cache.json && python -m src.learning.build_common_grounding_corpus` | `artifacts/learning_corpus/nlp4lp_{train,dev,test}.jsonl` with proper split |
| **Re-run learning on proper train split** | `./scripts/learning/run_real_data_only_learning_check.sh` (with torch) | `artifacts/learning_runs/real_data_only_learning_check/metrics.json` |

**Highest priority:** The int→float TypeMatch fix is already in the code. Measuring its actual end-to-end impact on TypeMatch and InstantiationReady on the gold evaluation is the single most important next experiment — it directly addresses the primary bottleneck and has a structural estimate of +79.5 percentage points on float-slot type matching.

---

## 8. Recommended Next Experiments for ESWA

The following experiments are listed in priority order. All are runnable with the existing codebase and now-available HF_TOKEN. Commands reference exact files.

### Priority 1 — Measure the float type_match fix end-to-end (CRITICAL)

```bash
# Reproduce baseline (should match §2.2 table)
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode typed
# Then read from results/paper/nlp4lp_downstream_summary.csv: variant=orig, baseline=tfidf

# All assignment modes in one pass
python tools/run_nlp4lp_focused_eval.py --variant orig
```

**Expected finding:** TypeMatch for `tfidf_typed` should increase substantially from 0.2267 because `_is_type_match()` now counts int tokens against float slots. InstantiationReady may also increase (the 0.8/0.8 threshold will be met by more queries).

**Why critical for ESWA:** This directly counters the strongest reviewer objection — "float type_match is only 3%" — and converts it into a strength: "we identified the root cause and fixed it."

---

### Priority 2 — Benchmark the three new assignment methods (IMPORTANT)

```bash
# All three new methods in one focused run
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf \
    --assignment-mode global_consistency_grounding
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf \
    --assignment-mode optimization_role_anchor_linking
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf \
    --assignment-mode optimization_role_bottomup_beam_repair

# Or all 6+ methods in one shot (experimental flag for all):
python tools/run_nlp4lp_focused_eval.py --variant orig --experimental
```

**Expected finding:** GCG is designed to improve the coverage–precision tension; anchor_linking targets wrong-variable association; bottomup_beam targets multiple-float-like confusions. See `docs/NLP4LP_ANCHOR_BEAM_DELIVERABLES.md §5` for predicted failure-family targets.

**Why important for ESWA:** Three implemented methods have never been benchmarked end-to-end on the gold evaluation. If any outperforms `tfidf_hierarchical_acceptance_rerank` (currently best InstReady = 0.085), that is a concrete new result.

---

### Priority 3 — Cross-variant downstream with the fixed pipeline (IMPORTANT)

```bash
for variant in noisy short; do
    python tools/run_nlp4lp_focused_eval.py --variant $variant
done
```

**Expected finding:** The int→float fix has no effect on noisy (values are `<num>`, not integers) or short (no values present). This confirms the fix is correctly scoped.

**Why important for ESWA:** Cross-variant robustness table is expected by reviewers. Should be in the manuscript.

---

### Priority 4 — Build valid train/dev/test corpus and re-run learning check (IF GPU AVAILABLE)

```bash
export NLP4LP_GOLD_CACHE=results/paper/nlp4lp_gold_cache.json
python -m src.learning.build_common_grounding_corpus \
    --nlp4lp_eval data/processed/nlp4lp_eval_orig.jsonl \
    --gold_cache $NLP4LP_GOLD_CACHE \
    --out_dir artifacts/learning_corpus
python -m src.learning.build_nlp4lp_pairwise_ranker_data \
    --corpus_dir artifacts/learning_corpus \
    --out_dir artifacts/learning_ranker_data/nlp4lp
python -m src.learning.verify_split_integrity \
    --data_dir artifacts/learning_ranker_data/nlp4lp
# Then train on a GPU node:
sbatch batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch
```

**Expected finding:** This reproduces the benchmark-valid run already documented in `docs/learning_runs/real_data_only_learning_check.md`. The result (learned < rule) is unlikely to change unless the training configuration or data are substantially different.

**Why important for ESWA:** The learning result (negative) is honest and strengthens the manuscript narrative: "we investigated learning-based grounding; it did not outperform the deterministic rule baseline at the current data scale; we therefore focus on deterministic methods."

---

### Priority 5 — Random baseline alignment

```bash
# Run random retrieval baseline on NLP4LP orig
python training/run_baselines.py \
    --eval-file data/processed/nlp4lp_eval_orig.jsonl \
    --baselines random \
    --out results/nlp4lp_retrieval_metrics_orig_random.json
```

Then update `tools/make_nlp4lp_paper_artifacts.py` to read this empirical R@1 instead of hardcoding 1/331. This resolves the inconsistency in §6.1.

---

### Priority 6 — Reproduce per-type TypeMatch table (post-fix)

```bash
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf \
    --assignment-mode typed --write-types
python tools/make_nlp4lp_paper_artifacts.py
# Results in: results/paper/nlp4lp_downstream_types_summary.csv
```

Before fix: float type_match ≈ 0.03. After fix: expected ~0.5–0.8+ (structural estimate is 81.8% match rate; actual assignment accuracy is a subset of this). This is the clearest per-type story in the paper.

---

## 9. Current Manuscript-Ready Claims (Evidence Strength)

| Claim | Evidence Strength | Caveat |
|-------|-----------------|--------|
| "Retrieval R@1 ≥ 0.85 on orig for TF-IDF, BM25, LSA" | **HIGH** — locally re-run, exact match | Short variant: −0.003 delta, benign |
| "Short-query retrieval is harder (R@1 ≈ 0.79 TF-IDF)" | **HIGH** — locally re-run | |
| "Noisy-query retrieval is not degraded (R@1 ≈ 0.90)" | **HIGH** — locally re-run | |
| "Downstream grounding is the main bottleneck" | **HIGH** — supported by oracle–TF-IDF gap analysis | |
| "InstantiationReady ≤ 0.085 even for best method" | **HIGH** — code audit consistent with all doc sources | Requires gold data to re-verify after TypeMatch fix |
| "Constrained assignment improves Exact20 at coverage cost" | **HIGH** — consistent across multiple docs | |
| "No learned method beats the deterministic rule baseline" | **HIGH** — definitive benchmark run documented | Limited to 500 steps, distilroberta-base, current data scale |
| "Float TypeMatch historically ~3%; fix to int→float raises it structurally" | **HIGH for diagnosis; MEDIUM for end-to-end impact** | Full measurement requires gold data run (Priority 1) |
| "Acceptance reranking raises InstantiationReady to 0.085" | **HIGH** — in `results/paper/` artifacts | Comes at cost of schema R@1 (0.846 vs 0.906) |
| "Anchor_linking/bottomup_beam/GCG improve over existing methods" | **UNKNOWN** — not yet benchmarked end-to-end | Run Priority 2 experiments |

---

## 10. Manuscript Narrative Recommendation

Based on all evidence above, the strongest defensible manuscript for ESWA has this structure:

**Task framing:** NL-to-instantiated-optimization-schema: map a natural-language problem statement to a schema with correctly typed and valued scalar parameters. Single benchmark: NLP4LP test set (331 queries, 5 variants).

**Retrieval section (strong, self-contained):** TF-IDF/BM25/LSA results on all 5 variants. Short-query expansion ablation (modest effect). Acceptance reranking (retrieval vs downstream trade-off). All numbers locally re-run and verified.

**Downstream grounding section (main contribution):** Three findings that are all evidence-supported and publishable:
1. **Coverage–precision tension** across 7+ deterministic assignment methods. No method dominates; the Pareto frontier is a genuine finding.
2. **Float type_match bottleneck** — root cause diagnosed, fix implemented (`_is_type_match`). Before-and-after comparison is the key new result. Run Priority 1 to get the after numbers.
3. **Schema-hit vs schema-miss analysis** — on schema hits: coverage ≈ 0.88, type_match ≈ 0.23; on misses: coverage ≈ 0.22, type_match ≈ 0.10. This confirms that schema retrieval quality strongly determines downstream quality.

**Learning section (honest negative):** Pairwise ranker (distilroberta-base, 500 steps, real data only) scores below the rule baseline on all metrics (pairwise_accuracy 0.197 vs 0.247, slot accuracy 0.182 vs 0.229). GAMS weak labels and targeted synthetic aux also failed. Learning is framed as future work with a clear articulation of what failed and what would need to change.

**Limitations to state explicitly:** Single benchmark, no solver validation, scalar-only (no vector parameters), noisy-variant downstream metrics are 0 by design, short-variant coverage is near 0 by design.

---

## 11. Files That Must Exist Before Final Submission

| File | Generated by | Status |
|------|-------------|--------|
| `results/nlp4lp_retrieval_summary.csv` | `tools/summarize_nlp4lp_results.py` | Must re-run |
| `results/paper/nlp4lp_downstream_summary.csv` | `tools/nlp4lp_downstream_utility.py` (per mode) | Must re-run with gold data |
| `results/paper/nlp4lp_focused_eval_summary.csv` | `tools/run_nlp4lp_focused_eval.py` | Must re-run |
| `results/paper/nlp4lp_downstream_types_summary.csv` | Same | Must re-run |
| `results/paper/nlp4lp_downstream_hitmiss_table_orig.csv` | `tools/make_nlp4lp_paper_artifacts.py` | Produced after above |
| `results/paper/nlp4lp_downstream_final_table_orig.csv` | Same | Produced after above; update to include constrained/repair |
| `results/paper/nlp4lp_retrieval_main_table.csv` | Same | Produced after retrieval re-run |

---

## 12. Confidence Summary

| Area | Confidence | Blocking issue (if any) |
|------|-----------|------------------------|
| Retrieval numbers (orig, noisy, short) | HIGH | None — re-run verified |
| Downstream numbers (all modes, orig) | HIGH (doc-verified) / needs re-run post-fix | Run Priority 1 after HF_TOKEN |
| Int→float TypeMatch fix: structural correctness | HIGH | Already in code, tests pass |
| Int→float fix: end-to-end TypeMatch improvement | UNKNOWN | Run Priority 1 |
| Learning result: negative | HIGH | Benchmark-valid run completed (job 854626) |
| New methods (anchor_linking, bottomup_beam, GCG): end-to-end | UNKNOWN | Run Priority 2 |
| Cross-variant results (noisy, short) for new methods | UNKNOWN | Run Priority 3 |
