# Final Main Merge Checklist

_Curated on 2026-03-09 before fast-forward merging `copilot/analyze-combinatorial-optimization-bot` into `main`._

## Branch Relationship

| Branch | Tip commit | Status |
|--------|-----------|--------|
| `main` | `b364be3` | fully contained in Copilot branch |
| `copilot/analyze-combinatorial-optimization-bot` | `eb83455` | +16 commits, 0 conflicts |

The Copilot branch is a **strict linear extension** of main.  
`git log main..copilot/...` → 16 commits ahead, 0 behind.

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

## Fast-Forward Merge Safety

✅ **The branch is safe to fast-forward merge into main.**

- No conflicts (Copilot branch is a linear extension of main)
- Curation pass complete: generated JSONL dumps removed, `.gitignore` updated
- All valuable source code, tests, configs, scripts, docs, and summary artifacts retained
- 0 regressions: existing test suite passes on the Copilot branch

```bash
# To fast-forward merge (run from a checkout with write access to main):
git checkout main
git merge --ff-only copilot/analyze-combinatorial-optimization-bot
git push origin main
```

---

## ChatGPT-Ready Summary

> **Branch reconciliation complete.**
>
> The Copilot branch (`copilot/analyze-combinatorial-optimization-bot`) is a strict
> linear extension of `main` — zero conflicts, 16 new commits, 74 files added/modified.
> A curation pass removed 6 large generated JSONL audit dumps (~650 KB) and added a
> `.gitignore` rule to prevent future re-commits.  All source code, tests, batch/SLURM
> scripts, learning configs, Copilot agent definitions, and documentation were kept.
>
> **What's new vs main:**
> - `global_consistency_grounding` downstream baseline (GCG) with 6 consistency signals
> - Short-query expansion and embedding-cache bottleneck fixes
> - PDF file upload in the Gradio UI
> - 5 `src/learning/` audit/analysis scripts + matching SLURM batch files
> - 7 new test files (30 GCG tests, bottleneck tests, PDF tests, metrics tests)
> - 6 Copilot agent definitions in `.github/agents/`
> - GCG evaluation reports, bottleneck analysis, and learning audit docs
>
> **Recommendation:** fast-forward `main` to this branch tip.
