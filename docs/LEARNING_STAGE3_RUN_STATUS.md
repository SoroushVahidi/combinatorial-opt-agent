# Stage 3 run status

Quick status of the first experiment round. Full details: **docs/LEARNING_STAGE3_FIRST_RESULTS.md**.

## Attempted

- Full Stage 3 launcher in **local** mode (corpus → ranker data → NL4Opt aux → train all runs → eval all → bottleneck slices → collect → report).

## Succeeded

- Corpus build (NLP4LP test only; 330 records).
- Ranker data build (test.jsonl; train.jsonl created via **test-as-train fallback**).
- NL4Opt auxiliary data build.
- Eval for all five run names (all used rule baseline; no checkpoints).
- Bottleneck slice evaluation.
- Collect and paper comparison report.
- All result artifacts written to `artifacts/learning_runs/`.

## Failed

- **NLP4LP pairwise training (both runs):** Environment had no torch; trainer wrote config only, no checkpoint.
- **NL4Opt pretrain_then_finetune:** `train_multitask_grounder` exited with "torch/transformers required".
- **NL4Opt joint_multitask:** Same.

## Blockers

- **torch/transformers** not available in the run environment. Training must be run on a node with GPU and PyTorch/transformers (e.g. via `sbatch batch/learning/run_stage3_experiments.sbatch`).

## Artifacts present

- **rule_baseline:** metrics.json, metrics.md (rule scoring on full test set).
- **Learned run dirs:** metrics.json for all five runs; each shows rule baseline numbers (no learned model).
- **bottleneck_slices:** slice_metrics.json, bottleneck_slice_report.md (rule only).
- **stage3_manifest.json**, **stage3_results_summary.json/.md**, **stage3_comparison_table.csv**, **stage3_paper_comparison.md**.

## Exploratory first learning run (2025-03-09) — not benchmark-valid

A focused run was submitted via `batch/learning/train_nlp4lp_first_run.sbatch`:

- **Job ID:** 854608
- **Task:** NLP4LP pairwise mention-slot ranker (text-only, 200 steps)
- **Output:** `artifacts/learning_runs/first_learning_run/checkpoint.pt`
- **Record:** `docs/learning_runs/first_learning_run_record.md`

**Caveat:** That run used train data created by test-as-train fallback (train and test were identical). It is a systems success only, not scientifically valid for benchmark comparison.

Submit again (exploratory only): `sbatch batch/learning/train_nlp4lp_first_run.sbatch` or `./scripts/learning/submit_train_first_run.sh`

## Valid first learning run (benchmark-safe)

A **scientifically valid** first run uses distinct train/dev/test (no fallback):

- **Script:** `batch/learning/train_nlp4lp_valid_first_run.sbatch`
- **Flow:** Build test corpus → split by instance_id (70/15/15) → build ranker data → verify split integrity → train (seed 42) → evaluate on held-out test.
- **Output:** `artifacts/learning_runs/valid_first_run/` (checkpoint, config, metrics, predictions).
- **Record:** `docs/learning_runs/valid_first_learning_run_record.md`

Submit: `sbatch batch/learning/train_nlp4lp_valid_first_run.sbatch` or `./scripts/learning/submit_valid_first_run.sh`

## Real-data-only learning check (benchmark-safe)

The definitive real-data-only benchmark run and decision are documented in **docs/learning_runs/real_data_only_learning_check.md**. That run used the largest valid NLP4LP split (230/50/50 instances), no synthetic or GAMS aux, and compared the learned ranker to the deterministic rule baseline on the same held-out test. **Result:** learned model did not outperform the rule baseline; learning is documented as future work. Deterministic methods remain the main trusted story.

- **Script:** `batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch` or `scripts/learning/run_real_data_only_learning_check.sh`
- **Requires:** `NLP4LP_GOLD_CACHE` set (e.g. to `results/paper/nlp4lp_gold_cache.json`) so the full corpus is built.

## Minimum next action (benchmark-valid runs only)

For a **benchmark-valid** learning run (distinct train/dev/test, no fallback):

```bash
sbatch batch/learning/train_nlp4lp_valid_first_run.sbatch
# or the full real-data-only check (train + eval + rule comparison):
sbatch batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch
```

For the full Stage 3 experiment round (multiple run types):

```bash
sbatch batch/learning/run_stage3_experiments.sbatch
```

Then inspect `artifacts/learning_runs/stage3_manifest.json` and `stage3_paper_comparison.md` as applicable.
