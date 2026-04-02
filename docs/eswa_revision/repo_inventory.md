# Repository Inventory — ESWA Revision

**Date:** 2026-03-10  
**Branch:** `copilot/main-branch-description`

---

## Dataset loading

| File | Purpose | Status |
|------|---------|--------|
| `training/external/build_nlp4lp_benchmark.py` | Fetches gold data from HuggingFace, builds eval JSONL files | Requires HF_TOKEN |
| `training/external/verify_hf_access.py` | Smoke-tests HF access | Runnable; exit 0 = OK |
| `training/run_baselines.py` | `_load_catalog()`, `_load_eval_instances()` | Runnable without HF_TOKEN |

## Retrieval benchmark scripts

| File | Purpose | Status |
|------|---------|--------|
| `retrieval/baselines.py` | BM25Baseline, TfidfBaseline, LSABaseline | Runnable |
| `retrieval/search.py` | SBERT-based semantic search | Requires local model |
| `retrieval/catalog_enrichment.py` | Acceptance reranker | Runnable |
| `training/metrics.py` | `compute_metrics()` — P@1, P@5, MRR@10, nDCG@10 | Runnable |
| `training/evaluate_retrieval.py` | SBERT eval with CLI flags | Runnable (needs SBERT model) |

## Downstream evaluation scripts

| File | Purpose | Status |
|------|---------|--------|
| `tools/nlp4lp_downstream_utility.py` | Core pipeline: extraction, assignment, eval | Requires HF_TOKEN for full run |
| `tools/run_nlp4lp_focused_eval.py` | Runs 7–10 methods in one pass | Requires HF_TOKEN |
| `tools/make_nlp4lp_paper_artifacts.py` | Generates paper tables/figures from CSVs | Requires result CSVs |
| `tools/summarize_nlp4lp_results.py` | Aggregates per-run JSONs to retrieval_summary.csv | Runnable |
| `tools/build_nlp4lp_per_instance_comparison.py` | Per-instance CSV for case analysis | Requires HF_TOKEN |
| `tools/build_nlp4lp_failure_audit.py` | Failure patterns report | Requires result CSVs |
| `tools/analyze_nlp4lp_downstream_disagreements.py` | Labels method disagreements | Requires result CSVs |
| `tools/run_nlp4lp_focused_eval.py` | Side-by-side focused run | Requires HF_TOKEN |

## Assignment mode implementations

All in `tools/nlp4lp_downstream_utility.py`:

| Mode | Function | Status |
|------|----------|--------|
| `typed` | Greedy with type compatibility | TRUSTED |
| `untyped` | Greedy without type | TRUSTED |
| `constrained` | DP bipartite 1-to-1 | TRUSTED |
| `semantic_ir_repair` | IR role tags + repair | TRUSTED |
| `optimization_role_repair` | Opt role tags + bipartite | TRUSTED |
| `optimization_role_relation_repair` | Relation-aware incremental | TRUSTED |
| `optimization_role_anchor_linking` | Anchor scoring + entity alignment | IMPLEMENTED (not benchmarked) |
| `optimization_role_bottomup_beam_repair` | Beam search over partial assignments | IMPLEMENTED (not benchmarked) |
| `global_consistency_grounding` | Global beam + penalties | IMPLEMENTED (not benchmarked) |

## SAE scripts

| File | Purpose | Status |
|------|---------|--------|
| `tools/nlp4lp_downstream_utility.py` → `slot_aware_extraction()` | SAE method | IMPLEMENTED |
| Hand-crafted benchmark | 24 cases, 24/24 exact | Verified locally |

## Hybrid retrieval

| File | Purpose | Status |
|------|---------|--------|
| `retrieval/baselines.py` → `HybridRetriever` | BM25+TF-IDF RRF | IMPLEMENTED |
| Short query R@1 gain: +0.012pp | Documented in `docs/PORTED_IMPROVEMENTS_BENCHMARK.md` | Verified |

## Plotting / report builders

| File | Purpose | Status |
|------|---------|--------|
| `tools/make_nlp4lp_paper_artifacts.py` | Main paper artifact generator | Requires result CSVs |
| `tools/nlp4lp_dataset_characterization.py` | Dataset stats | Runnable |
| ESWA revision plots | Generated in `results/eswa_revision/12_figures/` | COMPLETED |

## Existing results

| File | Contents |
|------|---------|
| `results/nlp4lp_retrieval_summary.csv` | Retrieval R@1/R@5/MRR by variant/baseline |
| `results/paper/nlp4lp_downstream_summary.csv` | Downstream metrics by method (pre-fix) |
| `results/paper/nlp4lp_focused_eval_summary.csv` | 7-method side-by-side |
| `docs/audits/current_repo_vs_manuscript_rerun.csv` | Manuscript vs rerun comparison |
| `results/eswa_revision/` | All ESWA revision outputs |

## Tests

| File | Coverage |
|------|---------|
| `tests/test_float_type_match.py` | 46 cases for _is_type_match fix |
| `tests/test_global_consistency_grounding.py` | 7+ GCG unit tests |
| `tests/test_baselines.py` | Retrieval baselines |
| `tests/` (full suite) | 220+ tests, 1 pre-existing skip |
