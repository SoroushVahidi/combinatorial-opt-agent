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

## Minimum next action

Run the experiment round on a **GPU node with torch/transformers**:

```bash
sbatch batch/learning/run_stage3_experiments.sbatch
```

Then inspect `artifacts/learning_runs/stage3_manifest.json` and `stage3_paper_comparison.md` to confirm at least one NLP4LP-only and (if desired) one NL4Opt-augmented learned run completed.
