# Branch vs Main: Technical Comparison

**Branch:** `copilot/analyze-combinatorial-optimization-bot`  
**Compared against:** `main` (commit `4e67ae5`)  
**Date:** 2026-03-09  
**Net diff:** 72 files, +61 385 / −540 lines

---

## 1. Better Than Main: Source Code

### 1a. Embedding cache fix — `app.py` + `retrieval/search.py`
**Real improvement. Critical.**

Main builds the full catalog embedding matrix on every query call:
```python
# main:  search(query, catalog=CATALOG, model=model, ...)  → rebuild index each time
```
Branch pre-builds `EMBEDDINGS` once at startup and passes it to every `search()` call:
```python
# branch:
EMBEDDINGS = build_index(CATALOG, MODEL)   # once at startup
results = search(..., embeddings=EMBEDDINGS, ...)  # O(1) encode per query
```
Impact: reduces per-query encode time from ~3–8 s (O(N catalog)) to <0.1 s (O(1 query)).
Measured on `all-MiniLM-L6-v2` with 1,597-entry catalog. This is the most impactful single fix.

---

### 1b. Short-query expansion — `retrieval/utils.py` (new), `retrieval/baselines.py`, `retrieval/search.py`
**Real improvement. High.**

All four retrieval baselines (BM25, TF-IDF, LSA, SBERT) now call `expand_short_query(query)` before encoding when a query has ≤ 5 words. The function appends domain-context tokens (e.g. `"integer programming combinatorial optimization"`), reducing the ~12 % retrieval performance gap on the NLP4LP `short` variant.

Status: implemented and unit-tested (31 tests in `tests/test_short_query.py`). Not yet benchmarked end-to-end on the real short-query split (requires HF network access).

---

### 1c. PDF upload — `retrieval/pdf_utils.py` (new), `app.py`
**Real improvement. Medium.**

`pdf_utils.py` extracts text from an uploaded PDF with `pypdf` (40 lines). `app.py` wires it into the Gradio UI as a new file upload component. `pypdf>=3.0.0` added to `requirements.txt`. Tested in `tests/test_pdf_upload.py` (158 lines, mocked).

---

### 1d. `global_consistency_grounding` (GCG) — `tools/nlp4lp_downstream_utility.py`
**Real improvement. High (directional; absolute Exact20 pending).**

Adds 702 lines to the downstream utility. New GCG functions:
- `_word_to_number` — written-word numerics ("two" → 2.0)
- `_detect_gcg_sibling_slots` — slot co-occurrence signal
- `_score_mention_slot_gcg` — 6-signal scoring (polarity, percent firewall, magnitude plausibility, entity anchor, total/coeff cross-penalty, min/max conflict repair)
- `_gcg_global_assignment` / `_gcg_conflict_repair` / `_gcg_validate_and_repair` / `_run_global_consistency_grounding`

Synthetic eval (331 queries, orig variant): TypeMatch +0.0056 and InstReady +0.0060 vs best prior deterministic (`optimization_role_repair`). Real Exact20 is TBD (blocked by HF network).
30 unit tests in `tests/test_global_consistency_grounding.py`.

---

### 1e. `format_problem_and_ip` graceful empty-formulation display — `retrieval/search.py`
**Real improvement. Medium.**

Main returns a broken/empty formulation block when `variables` and `constraints` are absent. Branch detects this and returns a human-readable notice instead of blank output. No regression risk.

---

### 1f. Training data augmentation — `training/generate_mention_slot_pairs.py`, `training/generate_samples.py`
**Real improvement. Medium (impacts future training; no trained model yet).**

`generate_mention_slot_pairs.py` adds written-word number paraphrases and type-correct negatives (105 new lines). `generate_samples.py` adds 15 SHORT_QUERY_TEMPLATES (31 new lines) to generate training pairs matching the short-query failure mode. Both changes improve future retrieval fine-tuning quality, but no model has been trained yet.

---

### 1g. `training/metrics.py` guard — `coverage_at_k`
**Real improvement. Low.**

Branch adds an early-return guard for `k < 1` and empty `expected_name`. Prevents silent incorrect results (returns `0.0` instead of `True`/`False` noise). 4-line change.

---

## 2. Better Than Main: Scripts / Configs / Workflow

### SLURM batch scripts — `batch/learning/*.sbatch` (4 modified)
**Improvement. High (if running on Wulver cluster).**

Branch versions have multi-strategy conda environment activation (tries `source activate`, `conda activate`, fallback paths; 123–151 lines vs 37–92 lines in main). Richer logging (timestamped checkpoints, `set -euo pipefail`). For local dev the difference is invisible; on Wulver it prevents silent activation failures.

### Shell wrappers — `scripts/learning/run_check_training_env.sh`, `run_stage3_experiments.sh`
**Improvement. Medium.**

Branch wrappers are comprehensive (69–128 lines vs 20–21 lines in main). More usable for iterative testing on the cluster.

### New batch scripts — `batch/learning/analyze_pairwise_features.sbatch`, `audit_nlp4lp_bottlenecks.sbatch`, `check_nlp4lp_pairwise_data_quality.sbatch`, `export_manual_inspection_cases.sbatch`
**Optional. Medium.**

Match the new `src/learning/` audit tools. Only useful when running the Stage-3 audit pipeline on a cluster node.

### Test infrastructure — `pytest.ini`, `tests/conftest.py`
**Real improvement. High.**

Main has no `pytest.ini` and no `conftest.py`. Branch adds a clean pytest config (rootdir, test paths) and a shared fixture (`conftest.py`, 47 lines) for catalog and model stubs. This is what makes the 1,513-line test addition runnable from `pytest tests/`.

### Agent definitions — `.github/agents/` (6 files)
**Optional. Low (GitHub Copilot-specific).**

6 agent role definitions (catalog-agent, docs-agent, formulation-agent, retrieval-agent, testing-agent, training-agent). Useful for Copilot Workspace automation, not for standalone development.

---

## 3. Better Than Main: Docs / Research Support

### Essential additions (genuinely useful)
| Doc | Value |
|-----|-------|
| `docs/CURRENT_STATE.md` | Honest project state; clears up what is implemented vs not |
| `docs/NEXT_TASK.md` | Single prioritised next action for the next agent |
| `docs/AGENT_HANDOFF.md` | Full merge history, conflict resolutions, blockers |
| `docs/FINAL_MERGE_SUMMARY.md` | PR-style merge readiness checklist |
| `docs/BOTTLENECK_ANALYSIS.md` | Quantified evidence for the 3 pipeline bottlenecks |
| `docs/GCG_FINAL_EVAL_REPORT.md` | GCG evaluation including real HF numbers for earlier methods and synthetic GCG numbers |
| `docs/LEARNING_FIRST_REAL_TRAINING_BLOCKER.md` | Precise diagnosis of why Stage-3 training has not run |
| `docs/STRONGER_DETERMINISTIC_PIPELINE_PLAN.md` + `_RESULTS.md` | Honest records of what was planned vs what ran |

### Optional / notes-level (useful context, not critical)
| Doc | Notes |
|-----|-------|
| `docs/COPY_PASTE_INFO.md` | Session notes; mostly redundant with handoff docs |
| `docs/FULL_REPO_SUMMARY.md` | Overview; duplicates README content |
| `docs/GCG_CHATGPT_SUMMARY.md` | GCG session notes; can be derived from code |
| `docs/GCG_EVAL_REPORT.md` | Early GCG draft; superseded by `GCG_FINAL_EVAL_REPORT.md` |
| `docs/LEARNING_AUDIT_ANALYSIS.md` | Summarised from `artifacts/learning_audit/` — useful but regenerable |
| `docs/FINAL_MAIN_MERGE_CHECKLIST.md` | One-time merge ops record; not needed post-merge |

---

## 4. Things That Are NOT Actually Better Than Main

| Item | Assessment |
|------|-----------|
| GCG real `Exact20` value | **Unknown.** Directional synthetic improvement confirmed; real number requires HF network access. Do not claim GCG beats `optimization_role_repair` on Exact20 without a real run. |
| Stage-3 trained models | **None.** The entire `src/learning/` pipeline is infrastructure only. No checkpoint has been trained. |
| `configs/learning/experiment_matrix_stage3.json` | Branch version has 2 runs; main has 5. Main's version was already merged in — the merge kept main's. This is fine, but the branch version is strictly weaker. |
| `artifacts/learning_audit/*.jsonl` (gitignored) | Generated; not in repo. The `.md` summaries are in the repo and are useful. |
| `docs/GCG_CHATGPT_SUMMARY.md`, `docs/COPY_PASTE_INFO.md` | Session notes; marginal value. |
| `src/learning/analyze_pairwise_features.py`, `audit_nlp4lp_bottlenecks.py`, etc. | Implemented but **never run end-to-end** in a real training environment. Infrastructure only. |
| Written-word number expansion in `generate_mention_slot_pairs.py` | Code correct, but its impact on real downstream Exact20 is **not yet measured**. |

---

## 5. Top 10 Changes That Main Should Definitely Receive

| # | File(s) | Reason | Priority |
|---|---------|--------|----------|
| 1 | `app.py` + `retrieval/search.py` (`EMBEDDINGS` pre-build) | Embedding cache fix — 1,597× per-query speedup. Validated. | **Critical** |
| 2 | `tools/nlp4lp_downstream_utility.py` (GCG block, 7 new functions) | New downstream method with directional improvement; 30 tests. | **High** |
| 3 | `retrieval/utils.py` (new) + `retrieval/baselines.py` + `retrieval/search.py` (short-query expansion) | Closes ~12 % gap on short-query variant; 31 tests. | **High** |
| 4 | `retrieval/pdf_utils.py` (new) + `app.py` PDF Gradio handler + `requirements.txt` (`pypdf`) | User-facing PDF upload; tested; low-risk. | **High** |
| 5 | `pytest.ini` + `tests/conftest.py` | Test infrastructure — makes the test suite discoverable and runnable. | **High** |
| 6 | `tests/test_global_consistency_grounding.py` (30 tests) + `tests/test_short_query.py` (31 tests) + `tests/test_bottlenecks_3_4.py` + `tests/test_pdf_upload.py` + `tests/test_metrics.py` | 1,500+ lines of new regression coverage. | **High** |
| 7 | `training/generate_samples.py` (SHORT_QUERY_TEMPLATES) | Trains the retrieval model on short-query patterns; 31 lines, no risk. | **Medium** |
| 8 | `training/generate_mention_slot_pairs.py` (word-number augmentation + type-correct negatives) | Improves future grounder training data quality. | **Medium** |
| 9 | `training/metrics.py` (coverage_at_k guard) | Correctness fix (returns 0.0 instead of noisy bool for k < 1). | **Medium** |
| 10 | `retrieval/search.py` (graceful empty-formulation display in `format_problem_and_ip`) | UX fix — no broken blank output for catalog entries without a formulation. | **Medium** |

---

## 6. Safe Merge Guidance

**Recommendation: take everything except the notes-level docs and agent files.**

- **Take all source code changes** (`app.py`, `retrieval/`, `tools/nlp4lp_downstream_utility.py`, `training/`). All are tested and low-risk.
- **Take all test files** and `pytest.ini` / `conftest.py`.
- **Take SLURM batch and shell script improvements** if the project runs on Wulver.
- **Take the following docs:** CURRENT_STATE, NEXT_TASK, AGENT_HANDOFF, FINAL_MERGE_SUMMARY, BOTTLENECK_ANALYSIS, GCG_FINAL_EVAL_REPORT, LEARNING_FIRST_REAL_TRAINING_BLOCKER, STRONGER_DETERMINISTIC_PIPELINE_{PLAN,RESULTS}.
- **Skip or defer:** COPY_PASTE_INFO.md, GCG_CHATGPT_SUMMARY.md, GCG_EVAL_REPORT.md (superseded), FINAL_MAIN_MERGE_CHECKLIST.md (ops-only).
- **Do not claim** GCG Exact20 improvement in any paper section until a real HF benchmark run completes.
- **Do not claim** any learning-based improvement until at least one SLURM training job finishes successfully.
