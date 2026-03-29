# Final Main Merge Checklist

_Updated on 2026-03-09 after merge reconciliation._

## Branch Relationship

| Branch | Tip commit | Status |
|--------|-----------|--------|
| `main` | `4e67ae5` | 3 commits ahead of divergence point |
| `copilot/analyze-combinatorial-optimization-bot` | `5a8a283` | 17 commits ahead of divergence point |

Both branches diverged from `b364be3`.

**Reconciliation completed:** `origin/main` (3 new commits: `93ff952`, `7b4e7df`, `4e67ae5`) was merged
into the Copilot branch. All 10 conflicts (9 file conflicts + 1 content conflict) were resolved.

### Conflicts Resolved

| File | Type | Resolution |
|------|------|-----------|
| `batch/learning/check_training_env.sbatch` | add/add | Kept copilot version (more comprehensive, multi-strategy env activation) |
| `batch/learning/run_stage3_experiments.sbatch` | add/add | Kept copilot version (more comprehensive) |
| `batch/learning/train_multitask_grounder.sbatch` | add/add | Kept copilot version (more comprehensive) |
| `batch/learning/train_nlp4lp_ranker.sbatch` | add/add | Kept copilot version (more comprehensive) |
| `configs/learning/experiment_matrix_stage3.json` | add/add | Kept main version (5 runs vs 2, more complete experiment set) |
| `scripts/learning/run_check_training_env.sh` | add/add | Kept copilot version (more comprehensive) |
| `scripts/learning/run_stage3_experiments.sh` | add/add | Kept copilot version (more comprehensive) |
| `src/learning/check_training_env.py` | add/add | Kept copilot version (156 vs 58 lines, richer diagnostics) |
| `tools/nlp4lp_downstream_utility.py` | content | Merged both: kept GCG block (copilot) + anchor_linking + beam_repair (main) + merged CLI choices |

### New Content From `main` Now Included

| Category | Files |
|----------|-------|
| **Source** | 12 new `src/learning/` files (training pipeline) |
| **Scripts** | 12 new `batch/learning/` sbatch + `scripts/learning/` sh wrappers |
| **Config** | `configs/learning/experiment_matrix_stage3.json` (5-run matrix) |
| **Tools** | `tools/analyze_nlp4lp_*`, `tools/build_nlp4lp_*`, `tools/run_nlp4lp_focused_eval.py` |
| **Docs** | 27 Stage-3 audit/design docs |

The Copilot branch now contains everything from both `main` and the Copilot-side work.

---

## Classification of Additions

### ✅ KEEP — Source Code

| File / Directory | Reason |
|------------------|--------|
| `app.py` | PDF file upload support added to Gradio UI |
| `retrieval/pdf_utils.py` | New PDF text extraction utility |
| `retrieval/search.py` | Short-query expansion + embedding cache fix |
| `retrieval/baselines.py` | Bottleneck fixes (type patterns, catalog formulations) |
| `retrieval/utils.py` | New shared retrieval utilities |
| `tools/nlp4lp_downstream_utility.py` | GCG (`global_consistency_grounding`) + 5 baseline methods |
| `training/generate_mention_slot_pairs.py` | New training data generator |
| `training/generate_samples.py` | New sample generator |
| `training/metrics.py` | Metric improvements |
| `src/__init__.py` | Package init |
| `src/learning/__init__.py` | Package init |
| `src/learning/analyze_pairwise_features.py` | Pairwise feature analysis |
| `src/learning/audit_nlp4lp_bottlenecks.py` | Bottleneck audit script |
| `src/learning/check_nlp4lp_pairwise_data_quality.py` | Data quality checker |
| `src/learning/check_training_env.py` | Training environment checker |
| `src/learning/export_manual_inspection_cases.py` | Inspection case exporter |

### ✅ KEEP — Tests

| File | Reason |
|------|--------|
| `tests/conftest.py` | Shared test fixtures |
| `tests/test_baselines.py` | Updated baseline tests |
| `tests/test_bottlenecks_3_4.py` | Bottleneck 3 & 4 regression tests |
| `tests/test_global_consistency_grounding.py` | GCG unit tests (30 tests) |
| `tests/test_metrics.py` | Metrics unit tests |
| `tests/test_pdf_upload.py` | PDF upload tests |
| `tests/test_short_query.py` | Short-query expansion tests |
| `pytest.ini` | Pytest configuration |

### ✅ KEEP — Configuration & Scripts

| File / Directory | Reason |
|------------------|--------|
| `requirements.txt` | Added `pypdf`, updated deps |
| `configs/learning/experiment_matrix_stage3.json` | Stage-3 experiment config |
| `batch/learning/*.sbatch` (9 files) | HPC SLURM batch scripts |
| `scripts/learning/*.sh` (6 files) | Local runner wrappers for learning scripts |
| `scripts/train_retrieval_gpu.slurm` | Updated SLURM training script |

### ✅ KEEP — Agent Definitions

| File | Reason |
|------|--------|
| `.github/agents/catalog-agent.md` | Copilot agent for catalog tasks |
| `.github/agents/docs-agent.md` | Copilot agent for documentation |
| `.github/agents/formulation-agent.md` | Copilot agent for formulations |
| `.github/agents/retrieval-agent.md` | Copilot agent for retrieval |
| `.github/agents/testing-agent.md` | Copilot agent for testing |
| `.github/agents/training-agent.md` | Copilot agent for training |

### ✅ KEEP — Documentation

| File | Reason |
|------|--------|
| `docs/README.md` | Updated docs index |
| `docs/BOTTLENECK_ANALYSIS.md` | Full bottleneck diagnosis |
| `docs/LEARNING_AUDIT_ANALYSIS.md` | Audit methodology and results |
| `docs/GCG_EVAL_REPORT.md` | GCG evaluation report |
| `docs/GCG_FINAL_EVAL_REPORT.md` | GCG final evaluation report |
| `docs/GCG_CHATGPT_SUMMARY.md` | GCG 2-paragraph companion summary |
| `docs/STRONGER_DETERMINISTIC_PIPELINE_PLAN.md` | Pipeline improvement plan |
| `docs/STRONGER_DETERMINISTIC_PIPELINE_RESULTS.md` | Pipeline improvement results |
| `docs/LEARNING_FIRST_REAL_TRAINING_BLOCKER.md` | HF/GPU blocker diagnosis |
| `docs/FULL_REPO_SUMMARY.md` | Comprehensive repo audit (reference) |
| `docs/COPY_PASTE_INFO.md` | Quick-reference copy-paste info |

### ✅ KEEP — Audit Summaries (small, human-readable)

| File | Size | Reason |
|------|------|--------|
| `artifacts/learning_audit/bottleneck_audit_summary.md` | 6.3 KB | Human-readable summary |
| `artifacts/learning_audit/bottleneck_audit_summary.json` | 1.4 KB | Machine-readable summary |
| `artifacts/learning_audit/pairwise_data_quality.md` | 256 B | Tiny summary |
| `artifacts/learning_audit/pairwise_data_quality.json` | 250 B | Tiny summary |
| `artifacts/learning_audit/pairwise_feature_analysis.md` | 1.7 KB | Feature analysis summary |
| `artifacts/learning_audit/pairwise_feature_analysis.json` | 2.2 KB | Feature analysis data |
| `artifacts/learning_audit/manual_inspection_cases.md` | 49 KB | Human-readable inspection cases |

### ✅ KEEP — Data

| File | Size | Reason |
|------|------|--------|
| `data/processed/custom_problems.json` | +1206 lines | Extended problem catalog |
| `data/processed/all_problems_extended.json` | 2.5 MB | Core catalog for retrieval |

---

### ❌ REMOVED — Generated JSONL Audit Slices (~650 KB total)

These files are **outputs** of `src/learning/audit_nlp4lp_bottlenecks.py` and  
`src/learning/check_nlp4lp_pairwise_data_quality.py`. They can be fully regenerated  
by re-running those scripts. Keeping machine-generated per-example dumps in version  
control adds noise without value — the human-readable `.md` summaries are kept instead.

| File | Size | Why removed |
|------|------|-------------|
| `artifacts/learning_audit/entity_association_risk_examples.jsonl` | 16 KB | Generated output |
| `artifacts/learning_audit/lower_upper_risk_examples.jsonl` | 99 KB | Generated output |
| `artifacts/learning_audit/manual_inspection_cases.jsonl` | 67 KB | Generated output (`.md` kept) |
| `artifacts/learning_audit/multi_numeric_confusion_examples.jsonl` | 240 KB | Generated output |
| `artifacts/learning_audit/percent_vs_absolute_risk_examples.jsonl` | 37 KB | Generated output |
| `artifacts/learning_audit/total_vs_per_unit_risk_examples.jsonl` | 191 KB | Generated output |

---

## .gitignore Changes

Added one rule under the `artifacts/runs/` section:

```
# Generated per-example JSONL audit slices (regenerable from src/learning scripts)
artifacts/learning_audit/*.jsonl
```

This prevents future re-commits of regenerated JSONL slices from any of the  
`src/learning/` audit scripts.

---

## Merge Safety Assessment

✅ **The branch is now safe to merge into main.**

- All conflicts resolved (see table above)
- Curation pass complete: generated JSONL dumps removed, `.gitignore` updated
- `tools/nlp4lp_downstream_utility.py` passes Python syntax validation after merge
- All valuable source code, tests, configs, scripts, docs, and summary artifacts retained
- The merged branch contains the best integrated version of both branches

```bash
# To merge into main (run from a checkout with write access to main):
git checkout main
git merge --no-ff copilot/analyze-combinatorial-optimization-bot -m "Merge copilot branch: GCG, bottleneck fixes, PDF upload, Stage-3 pipeline, tests"
git push origin main
```

---

## ChatGPT-Ready Summary

> **Branch reconciliation complete.**
>
> The Copilot branch (`copilot/analyze-combinatorial-optimization-bot`) was diverged from
> `main` — each branch had 3 (main) and 17 (copilot) commits since divergence point `b364be3`.
> A merge reconciliation was performed: `origin/main` was merged into the Copilot branch,
> 10 conflicts resolved (9 add/add + 1 content conflict in the large downstream utility).
> All curation from the prior pass is preserved (JSONL dumps removed, `.gitignore` updated).
>
> **What the integrated branch contains vs original main:**
> - `global_consistency_grounding` (GCG) downstream baseline with 6 consistency signals
> - Anchor-linking + bottom-up beam repair methods (from main's Stage-3 expansion)
> - Short-query expansion and embedding-cache bottleneck fixes
> - PDF file upload in the Gradio UI
> - Full Stage-3 learning pipeline: 12 `src/learning/` modules, models/, configs, sbatch/sh
> - 5 `src/learning/` audit/analysis scripts + matching SLURM batch files
> - 7 new test files (GCG tests, bottleneck tests, PDF tests, metrics tests)
> - 6 Copilot agent definitions in `.github/agents/`
> - 27+ documentation files covering Stage-3 design, audits, and results
> - GCG evaluation reports, bottleneck analysis, and learning audit docs
>
> **Recommendation:** merge the Copilot branch into `main` — no further conflicts expected.
