## Scope and goal

Reconcile the Wulver working tree with GitHub `main` so that:

- All **good Stage-3 source code, configs, scripts, and docs** are versioned.
- The **enhanced NLP4LP downstream utility** on Wulver is preserved.
- No logs, run artifacts, or machine-specific clutter are committed.

Both local `main` and `origin/main` point to the same commit:

- `HEAD = origin/main = b364be3 docs: update README and add documentation index`
- All differences are from the **working tree**, not the commit history.

---

## A. Tracked files to update

### 1. `tools/nlp4lp_downstream_utility.py` (MUST COMMIT)

- Local Wulver version differs from GitHub `main`:
  - `git diff --stat HEAD -- tools/nlp4lp_downstream_utility.py` → `1 file changed, 1034 insertions(+), 18 deletions(-)`.
  - Local file ~3618 lines vs ~2602 lines on GitHub.
- Local enhancements:
  - `_apply_low_resource_env` and cached HF gold loader (`_load_hf_gold` with `NLP4LP_GOLD_CACHE`).
  - New anchor-linking / entity-semantic scoring for optimization-role alignment:
    - `_score_mention_slot_anchor`
    - `_score_mention_slot_entity_semantic`
    - `_run_optimization_role_anchor_linking`
    - `_run_optimization_role_bottomup_beam_repair`
    - `_run_optimization_role_entity_semantic_beam_repair`
  - Relation-aware / PICARD-style incremental admissible assignment:
    - `_slot_slot_relation_tags`, `_mention_mention_relation_tags`
    - `_is_partial_admissible`, `_relation_bonus`
    - `_opt_role_incremental_admissible_assignment`
    - `_run_optimization_role_relation_repair`
  - `run_setting` extended with new `assignment_mode` (`optimization_role_relation_repair`).
- Plan:
  - **Keep the Wulver version** and commit it verbatim.
  - No evidence that existing GitHub behavior is removed; local version is a strict extension with guarded new modes.

---

## B. Untracked files to add (MUST COMMIT)

These are new, coherent pieces of the Stage-3 pipeline or related tooling and should be versioned.

### 1. `src/learning/` (Stage-3 learning stack)

Untracked:

- `src/learning/build_common_grounding_corpus.py`
- `src/learning/build_nl4opt_aux_data.py`
- `src/learning/build_nlp4lp_pairwise_ranker_data.py`
- `src/learning/build_stage3_comparison_report.py`
- `src/learning/check_training_env.py`
- `src/learning/collect_stage3_results.py`
- `src/learning/common_corpus_schema.py`
- `src/learning/eval_bottleneck_slices.py`
- `src/learning/eval_nlp4lp_pairwise_ranker.py`
- `src/learning/models/__init__.py`
- `src/learning/models/decoding.py`
- `src/learning/models/features.py`
- `src/learning/models/multitask_grounder.py`
- `src/learning/models/pairwise_ranker.py`
- `src/learning/run_stage3_experiments.py`
- `src/learning/summarize_common_grounding_corpus.py`
- `src/learning/train_multitask_grounder.py`
- `src/learning/train_nlp4lp_pairwise_ranker.py`
- `src/learning/validate_common_grounding_corpus.py`

Classification: **MUST COMMIT**.

Rationale:

- This is the full Stage-3 learning pipeline:
  - Common grounding corpus builders and validators.
  - Pairwise ranker and multitask grounder training entrypoints.
  - Stage-3 experiment orchestrator, evaluation, and bottleneck slicing.
  - Encoder/decoder models and feature logic.
- Coherent, self-contained, and directly referenced by Stage-3 docs and batch scripts.

### 2. `batch/learning/` (Stage-3 SLURM jobs)

Untracked:

- `batch/learning/build_common_corpus.sbatch`
- `batch/learning/build_nl4opt_aux_data.sbatch`
- `batch/learning/build_nlp4lp_ranker_data.sbatch`
- `batch/learning/build_stage3_comparison_report.sbatch`
- `batch/learning/check_training_env.sbatch`
- `batch/learning/collect_stage3_results.sbatch`
- `batch/learning/eval_bottleneck_slices.sbatch`
- `batch/learning/eval_nlp4lp_ranker.sbatch`
- `batch/learning/run_stage3_experiments.sbatch`
- `batch/learning/train_multitask_grounder.sbatch`
- `batch/learning/train_nlp4lp_ranker.sbatch`
- `batch/learning/validate_common_corpus.sbatch`

Classification: **MUST COMMIT**.

Rationale:

- These are the primary Wulver batch entrypoints for Stage-3 experiment orchestration, training, evaluation, and comparison reporting.
- They are part of the reproducible experiment pipeline and should live in version control.

### 3. `scripts/learning/` (local + batch orchestration)

Untracked:

- `scripts/learning/run_build_common_corpus.sh`
- `scripts/learning/run_build_nl4opt_aux_data.sh`
- `scripts/learning/run_build_nlp4lp_ranker_data.sh`
- `scripts/learning/run_build_stage3_comparison_report.sh`
- `scripts/learning/run_check_training_env.sh`
- `scripts/learning/run_collect_stage3_results.sh`
- `scripts/learning/run_eval_bottleneck_slices.sh`
- `scripts/learning/run_eval_nlp4lp_ranker.sh`
- `scripts/learning/run_stage3_experiments.sh`
- `scripts/learning/run_train_multitask_grounder.sh`
- `scripts/learning/run_train_nlp4lp_ranker.sh`
- `scripts/learning/run_validate_corpus.sh`

Classification: **MUST COMMIT**.

Rationale:

- Thin wrappers around the Stage-3 Python entrypoints and sbatch files.
- Give a consistent CLI for local dry-runs and batch submissions.

### 4. `configs/learning/` (Stage-3 experiment matrix)

Untracked:

- `configs/learning/experiment_matrix_stage3.json`

Classification: **MUST COMMIT**.

Rationale:

- Canonical configuration for Stage-3 runs:
  - `rule_baseline`, `nlp4lp_pairwise_text_only`, `nlp4lp_pairwise_text_plus_features`
  - `nl4opt_pretrain_then_finetune`, `nl4opt_joint_multitask`
- Tightly coupled to `run_stage3_experiments.py` and batch/learning jobs.

### 5. Stage-3 and audit docs under `docs/` (curated subset)

Untracked (all currently on Wulver only):

- `docs/ACCEPTANCE_AND_HIERARCHY_AUDIT.md`
- `docs/CURRENT_STATE_AUDIT.md`
- `docs/ENVIRONMENT_RESOURCES_AUDIT.md`
- `docs/JOURNAL_READINESS_AUDIT.md`
- `docs/LEARNING_CORPUS_FORMAT.md`
- `docs/LEARNING_DATASET_STRATEGY.md`
- `docs/LEARNING_ENVIRONMENT_DEBUG.md`
- `docs/LEARNING_EXPERIMENT_BASELINES.md`
- `docs/LEARNING_STAGE1_IMPLEMENTATION.md`
- `docs/LEARNING_STAGE2_IMPLEMENTATION.md`
- `docs/LEARNING_STAGE3_EXPERIMENTS.md`
- `docs/LEARNING_STAGE3_FIRST_RESULTS.md`
- `docs/LEARNING_STAGE3_RUN_STATUS.md`
- `docs/NL4OPT_AUX_SUPERVISION.md`
- `docs/NLP4LP_ANCHOR_AND_BEAM_IMPLEMENTATION_PLAN.md`
- `docs/NLP4LP_ANCHOR_BEAM_DELIVERABLES.md`
- `docs/NLP4LP_FOCUSED_EVAL.md`
- `docs/NLP4LP_LOW_RESOURCE_RUN_GUIDE.md`
- `docs/NLP4LP_RATSQL_PICARD_AUDIT.md`
- `docs/NLP4LP_RELATION_AWARE_METHOD_EXAMPLES.md`
- `docs/NLP4LP_RELATION_AWARE_METHOD_IMPLEMENTATION.md`
- `docs/NLP4LP_RELATION_AWARE_METHOD_RESULTS.md`
- `docs/OPTIMIZATION_ROLE_METHOD_AUDIT.md`
- `docs/OPTIMIZATION_ROLE_METRICS_COMPARISON.md`
- `docs/RESULTS_VS_CODE_VERIFICATION.md`
- `docs/TATQA_FINQA_DATASET_ACQUISITION_REPORT.md`
- `docs/GITHUB_RECONCILIATION_PLAN.md` (this file)

Classification: **MUST COMMIT** (for this pass).

Rationale:

- These documents:
  - Describe the Stage-3 pipeline design, data format, and baseline choices.
  - Capture environment audits and current state audits.
  - Provide method-specific implementation and results notes for NLP4LP and optimization-role methods.
- They are high-signal for future maintenance and reproducibility.
- Some results docs may need future edits to align with finalized experiments, but they should live in version control rather than remain ephemeral.

### 6. New NLP4LP utility tools under `tools/`

Untracked:

- `tools/analyze_nlp4lp_downstream_disagreements.py`
- `tools/analyze_nlp4lp_three_bottlenecks.py`
- `tools/build_nlp4lp_failure_audit.py`
- `tools/build_nlp4lp_per_instance_comparison.py`
- `tools/build_nlp4lp_situation_reports.py`
- `tools/run_nlp4lp_focused_eval.py`

Classification: **MUST COMMIT**.

Rationale:

- These are clearly structured analysis/diagnostic utilities for NLP4LP downstream behavior.
- They are reusable for troubleshooting and paper replication.

---

## C. Files that MAYBE SHOULD be committed later

These are not included in this reconciliation but are candidates for future versioning after explicit review.

### 1. `data_external/` and similar resources

- Currently untracked directories like `data_external/` (and possibly others under `data/`) may contain:
  - Public but large resources.
  - Semi-processed external corpora.
- Classification: **MAYBE COMMIT**.

Plan:

- Leave untracked for now.
- Revisit after checking:
  - Size and licensing constraints.
  - Whether they are true inputs vs cached downloads.

---

## D. Files that should NOT be committed

These should remain local-only and (where practical) be ignored via `.gitignore`.

### 1. Logs and artifacts

- `logs/` (job logs, training logs, status files).
- `artifacts/` (Stage-3 run outputs, checkpoints, metrics, comparison reports).
- `comparison_reports/` (e.g., `comparison_reports/wulver_vs_github_main.md`).

Classification: **DO NOT COMMIT**.

Plan:

- Update `.gitignore` to explicitly ignore:
  - `logs/`
  - `artifacts/`
  - `comparison_reports/`

### 2. Generated processed data outputs

- Known examples:
  - `data/processed/mention_slot_pairs.jsonl`
  - `data/processed/mention_slot_pairs_dev.jsonl`
- Already-ignored patterns (from existing `.gitignore`):
  - `data/processed/training_pairs*.jsonl`
  - `data/processed/collected_training_pairs.jsonl`
  - `data/processed/eval_*.jsonl`

Classification: **DO NOT COMMIT**.

Plan:

- Extend `.gitignore` to add:
  - `data/processed/mention_slot_pairs*.jsonl`

### 3. Machine-specific and environment outputs

- Any SLURM output/error logs beyond the generic patterns already ignored.
- Local venv files are already covered (`venv/`, `venv_sbert/`, etc.).

Classification: **DO NOT COMMIT**.

Plan:

- Rely on existing `.gitignore` plus the additional entries above; avoid `git add .`.

---

## E. .gitignore updates needed

Planned additions to `.gitignore` (project-specific section):

- Ignore Stage-3 run artifacts and logs:

  - `artifacts/`
  - `logs/`
  - `comparison_reports/`

- Ignore Stage-3 processed mention-slot pairs:

  - `data/processed/mention_slot_pairs*.jsonl`

Rationale:

- These are reproducible or ephemeral outputs and should not clutter the repo or GitHub history.

---

## F. Commit structure and risk areas

### Planned commits

1. **Commit 1: NLP4LP downstream utility + tools**
   - Update `tools/nlp4lp_downstream_utility.py` to Wulver version.
   - Add the new analysis/utilities in `tools/` listed above.

2. **Commit 2: Stage-3 learning pipeline code and configs**
   - Add `src/learning/` (all Stage-3 modules).
   - Add `batch/learning/` sbatch files.
   - Add `scripts/learning/` orchestration scripts.
   - Add `configs/learning/experiment_matrix_stage3.json`.

3. **Commit 3: Documentation + .gitignore**
   - Add curated Stage-3 and audit docs under `docs/`.
   - Add `docs/GITHUB_RECONCILIATION_PLAN.md`.
   - Update `.gitignore` with the new ignore rules.

### Risk / conflict areas

- **Downstream utility behavior**:
  - Local Wulver version is more complex; care must be taken not to regress any existing modes.
  - New functionality is largely additive and guarded by explicit `assignment_mode` branches, minimizing risk.

- **Docs claiming results**:
  - Some Stage-3 docs (e.g., `LEARNING_STAGE3_FIRST_RESULTS.md`, `NLP4LP_RELATION_AWARE_METHOD_RESULTS.md`) may describe preliminary or partial results.
  - They should be interpreted as in-progress notes; future edits may be needed to align with final experiments.

- **Large data / licensing**:
  - `data_external/` and any other large external datasets are intentionally left untracked for now.
  - Pushing them would require explicit license and size review.

---

## G. Summary

- **Tracked update:** Keep and commit the enhanced `tools/nlp4lp_downstream_utility.py`.
- **New code:** Commit the entire Stage-3 learning stack (`src/learning/`, `batch/learning/`, `scripts/learning/`, `configs/learning/`).
- **Docs:** Commit the curated Stage-3 and audit documents plus this reconciliation plan.
- **Ignore:** Add explicit `.gitignore` rules for logs, artifacts, comparison reports, and mention-slot processed data.
- **Artifacts:** Do **not** commit run artifacts, logs, or large external data; keep the repo focused on reproducible source and config.

