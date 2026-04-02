# Current Project State

**Date:** 2026-03-09  
**Branch:** `copilot/analyze-combinatorial-optimization-bot` (merged with `main` — contains both)

---

## 1. What Is Strong: Retrieval

The retrieval pipeline is **production-ready** for the NLP4LP benchmark.

| Baseline | Exact5 (real HF) | Exact20 (real HF) |
|----------|-----------------|-------------------|
| tfidf (orig split) | 0.2053 | 0.2330 |

- `retrieval/baselines.py` has TF-IDF, BM25, LSA.
- Embedding cache is fixed (built once at startup, not per query).
- Short-query expansion is live (inference-time, no retrain needed).
- PDF upload works via `retrieval/pdf_utils.py` + `pypdf`.
- Retrieval test suite: `tests/test_baselines.py` passes.

**Bottleneck status for retrieval:** None remaining. All three major bottlenecks
(embedding cache, short-query expansion, type-pattern / catalog-formulation display)
are resolved and tested.

---

## 2. Main Bottleneck: Downstream Number-to-Slot Grounding

The downstream instantiation task — assigning extracted numeric values to the
correct named slots in the retrieved formulation — is the **primary unsolved problem**.

### What is actually implemented (deterministic methods)

All methods live in `tools/nlp4lp_downstream_utility.py`.

| Method | Status | Notes |
|--------|--------|-------|
| `typed` | ✅ Active | Type-bucket baseline |
| `untyped` | ✅ Active | No type information used |
| `constrained` | ✅ Active | Hard type constraints |
| `semantic_ir_repair` | ✅ Active | Semantic role tags |
| `optimization_role_repair` | ✅ Active | Best prior deterministic method |
| `global_consistency_grounding` (GCG) | ✅ Active | New — 6 consistency signals |
| `optimization_role_relation_repair` | ✅ Active | Relation-aware variant (from main) |
| `optimization_role_anchor_linking` | ✅ Active | Context-aware anchor (from main) |
| `optimization_role_bottomup_beam_repair` | ✅ Active | Bottom-up beam search (from main) |
| `optimization_role_entity_semantic_beam_repair` | ✅ Active | Entity+semantic beam (from main) |

### Known real-benchmark results (last HF-accessible run)

| Method | Coverage | TypeMatch | Exact20 | InstReady |
|--------|----------|-----------|---------|-----------|
| tfidf / typed | 0.822 | 0.231 | 0.205 | 0.073 |
| tfidf / optimization_role_repair | 0.822 | 0.243 | **0.277** | 0.060 |
| tfidf / GCG | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> **GCG real numbers are pending.** Directional improvement confirmed by synthetic eval
> (TypeMatch +0.0056, InstReady +0.0060 vs `optimization_role_repair` on orig variant).
> Full real HF evaluation requires network access to `huggingface.co`.

### Why Exact20 is the key metric

`exact20_on_hits` = fraction of retrieved problems where **all** grounded values
match gold within ±20%. It is the paper's primary downstream metric.
Current best deterministic: **0.277** (`optimization_role_repair`).
Target to beat: 0.30+.

---

## 3. Learning Infrastructure: Exists, But No Trained Models Yet

### What exists

| Category | Files | Status |
|----------|-------|--------|
| Pairwise ranker training | `src/learning/train_nlp4lp_pairwise_ranker.py` | Code ready |
| Multitask grounder | `src/learning/train_multitask_grounder.py` | Code ready |
| Data builders | `src/learning/build_nlp4lp_pairwise_ranker_data.py`, `build_nl4opt_aux_data.py`, etc. | Code ready |
| Eval scripts | `src/learning/eval_nlp4lp_pairwise_ranker.py`, `eval_bottleneck_slices.py` | Code ready |
| Models | `src/learning/models/{pairwise_ranker,multitask_grounder,features,decoding}.py` | Code ready |
| Experiment matrix | `configs/learning/experiment_matrix_stage3.json` | 5 runs defined |
| SLURM batch scripts | `batch/learning/*.sbatch` (12 files) | Ready to submit |
| Shell wrappers | `scripts/learning/*.sh` (12 files) | Ready |
| Audit infrastructure | `src/learning/audit_nlp4lp_bottlenecks.py`, `check_nlp4lp_pairwise_data_quality.py` | Code ready |

### What does NOT yet exist

- ❌ **No trained model checkpoints.** Zero training has run successfully.
- ❌ **No real learning-based Exact20 numbers.** All Stage-3 result numbers are absent.
- ❌ **No HF-benchmark GCG real results.** Network was blocked in sandbox.

### Why training has not run

`torch` and `transformers` are not available in the current environment.
This is a login-node constraint; training must be submitted via SLURM:

```bash
sbatch batch/learning/run_stage3_experiments.sbatch
```

See `docs/LEARNING_FIRST_REAL_TRAINING_BLOCKER.md` for full diagnosis.

---

## 4. Test Suite

| File | Tests | Status |
|------|-------|--------|
| `tests/test_baselines.py` | Retrieval baselines | ✅ passes |
| `tests/test_bottlenecks_3_4.py` | Type-pattern, catalog display | ✅ passes |
| `tests/test_global_consistency_grounding.py` | GCG unit tests (30 tests) | ✅ passes |
| `tests/test_metrics.py` | Metric helpers | ✅ passes |
| `tests/test_pdf_upload.py` | PDF upload | ✅ passes |
| `tests/test_short_query.py` | Short-query expansion | ✅ passes |

---

## 5. Summary

| Area | Status |
|------|--------|
| Retrieval | ✅ Strong, no open issues |
| Deterministic downstream (GCG + 9 methods) | ✅ Implemented, tested |
| Real GCG benchmark numbers | ⏳ Pending network access |
| Learning infrastructure | ✅ Code complete |
| Trained learning models | ❌ None yet |
| Learning-based Exact20 improvement | ❌ Not yet measured |
