# Agent Handoff Document

**Date:** 2026-03-09  
**Prepared by:** GitHub Copilot coding agent  
**Branch:** `copilot/analyze-combinatorial-optimization-bot`

---

## 1. What Was Merged and Why

### Situation at start of session
The copilot branch (17 commits) and `main` (3 new commits) had diverged from
common ancestor `b364be3`. Fast-forward was no longer possible.

### What main had that copilot did not
Three commits (`93ff952`, `7b4e7df`, `4e67ae5`) adding:
- Full Stage-3 learning pipeline: 12 `src/learning/` training modules, `src/learning/models/`
- 12 SLURM/shell wrappers for Stage-3 jobs
- 6 analysis tools under `tools/` (downstream disagreements, bottleneck audit, etc.)
- 27 Stage-3 documentation files
- Updated `configs/learning/experiment_matrix_stage3.json` (5-run matrix)
- Updated `.gitignore` with Stage-3 artifact rules

### What copilot had that main did not
Seventeen commits adding:
- `global_consistency_grounding` (GCG) — new deterministic downstream method with 6 signals
- Short-query expansion fix (inference-time, no retrain)
- PDF file upload in Gradio UI (`retrieval/pdf_utils.py`, `pypdf`)
- Bottleneck fixes 2, 3, 4 (embedding cache, word-number recognition, catalog display)
- 6 GitHub Copilot agent definitions in `.github/agents/`
- 7 new test files (30 GCG unit tests, bottleneck tests, PDF tests, metrics tests)
- `pytest.ini`, `src/__init__.py`, `src/learning/__init__.py`
- Audit/analysis scripts (`src/learning/analyze_pairwise_features.py`, etc.)
- Curation pass: removed 6 JSONL audit dumps (~650 KB), updated `.gitignore`
- Evaluation reports and documentation

### How it was reconciled
`origin/main` was merged into the copilot branch (`git merge --no-ff origin/main`).

---

## 2. Conflict Resolutions (10 files)

| File | Type | Resolution | Rationale |
|------|------|-----------|-----------|
| `batch/learning/check_training_env.sbatch` | add/add | Kept copilot | Multi-strategy env activation (123 vs 37 lines) |
| `batch/learning/run_stage3_experiments.sbatch` | add/add | Kept copilot | More robust activation logic (142 vs 92 lines) |
| `batch/learning/train_multitask_grounder.sbatch` | add/add | Kept copilot | More comprehensive (151 vs 77 lines) |
| `batch/learning/train_nlp4lp_ranker.sbatch` | add/add | Kept copilot | More comprehensive (142 vs 68 lines) |
| `configs/learning/experiment_matrix_stage3.json` | add/add | Kept **main** | Main's version has 5 runs vs copilot's 2 |
| `scripts/learning/run_check_training_env.sh` | add/add | Kept copilot | More comprehensive (69 vs 21 lines) |
| `scripts/learning/run_stage3_experiments.sh` | add/add | Kept copilot | More comprehensive (128 vs 20 lines) |
| `src/learning/check_training_env.py` | add/add | Kept copilot | 156 lines with full package diagnostics vs 58 |
| `tools/nlp4lp_downstream_utility.py` | content | **Merged both** | GCG block (copilot) + anchor_linking + beam_repair (main) + unified CLI choices |
| `.gitignore` | auto | Deduplicated | Both added similar rules; removed duplicate `logs/`, unified `artifacts/` rules |

### Downstream utility merge detail
The most complex conflict. Both branches independently added ~500-line function blocks
at the same insertion point. Resolution:
- Kept the full GCG implementation (copilot's `_detect_gcg_sibling_slots`, `_score_mention_slot_gcg`, `_gcg_global_assignment`, `_gcg_validate_and_repair`, `_run_global_consistency_grounding`) 
- Kept main's `_run_optimization_role_anchor_linking` and `_run_optimization_role_bottomup_beam_repair`
- Merged the CLI `--assignment-mode choices` to include all 9 modes
- Merged the `elif args.assignment_mode` dispatch block
- All Python syntax verified after merge (`ast.parse()` → no errors)

---

## 3. Important Files Changed / Added

### Source code (merged result)
```
tools/nlp4lp_downstream_utility.py    — 4244 lines; all 9 assignment modes
retrieval/baselines.py                — bottleneck fixes
retrieval/search.py                   — short-query expansion + cache fix
retrieval/pdf_utils.py                — PDF upload support
retrieval/utils.py                    — shared utilities
app.py                                — PDF upload in Gradio UI
src/learning/                         — full training pipeline (15 modules)
src/learning/models/                  — 4 model files
training/generate_mention_slot_pairs.py, generate_samples.py, metrics.py
```

### Tests
```
tests/test_baselines.py
tests/test_bottlenecks_3_4.py
tests/test_global_consistency_grounding.py  (30 tests)
tests/test_metrics.py
tests/test_pdf_upload.py
tests/test_short_query.py
tests/conftest.py
pytest.ini
```

### Configuration and batch scripts
```
configs/learning/experiment_matrix_stage3.json  (5 experiments)
batch/learning/*.sbatch                          (12 SLURM batch files)
scripts/learning/*.sh                            (12 shell wrappers)
scripts/train_retrieval_gpu.slurm               (updated)
```

### Analysis tools (from main)
```
tools/analyze_nlp4lp_downstream_disagreements.py
tools/analyze_nlp4lp_three_bottlenecks.py
tools/build_nlp4lp_failure_audit.py
tools/build_nlp4lp_per_instance_comparison.py
tools/build_nlp4lp_situation_reports.py
tools/run_nlp4lp_focused_eval.py
```

### Curation (excluded from repo)
```
artifacts/learning_audit/*.jsonl  (6 JSONL files, ~650 KB, gitignored)
```

---

## 4. Remaining Blockers

| Blocker | Impact | Mitigation |
|---------|--------|-----------|
| No network access to `huggingface.co` | GCG real Exact20 is unknown | Run on any machine with internet |
| No `torch`/`transformers` in current env | Zero trained models | Submit via SLURM on Wulver GPU partition |
| `data/processed/nlp4lp_eval_orig.jsonl` may not be in repo | Stage-3 pipeline needs it | Run `build_nlp4lp_benchmark.py` first |

---

## 5. What the Next Agent Should Do First

**Priority 1 (any machine with internet):**
```bash
pip install datasets sentence-transformers
python -m tools.nlp4lp_downstream_utility \
  --variant orig --baseline tfidf \
  --assignment-mode global_consistency_grounding --split test
```
Fill in `_TBD_` values in `docs/GCG_FINAL_EVAL_REPORT.md`.

**Priority 2 (Wulver GPU node):**
```bash
sbatch batch/learning/run_stage3_experiments.sbatch
```
Then eval and record results in `docs/LEARNING_STAGE3_FIRST_RESULTS.md`.

**Do NOT:**
- Change retrieval — it is already strong.
- Commit `artifacts/learning_runs/`, `logs/`, or any `*.jsonl` under `artifacts/learning_audit/`.
- Add new methods to `tools/nlp4lp_downstream_utility.py` without tests.
- Claim trained-model results that haven't been run.

---

## 6. Branch State

```
Branch:  copilot/analyze-combinatorial-optimization-bot
Tip:     3307229
Main:    4e67ae5

Commits in copilot not in main:  18 (includes 1 merge commit)
Commits in main not in copilot:   0  ← main is fully contained

Untracked files:  none
Unstaged changes: none
Working tree:     clean
```

The branch is **ready to merge into main** with `--no-ff`.
