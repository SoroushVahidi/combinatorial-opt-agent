# GAMS aux vs NLP4LP-only comparison

**Purpose:** One controlled experiment to test whether GAMS-derived weak supervision (auxiliary pretraining) improves the NLP4LP pairwise ranker on held-out NLP4LP test.

**Job ID:** 854616 (submitted). Logs: `logs/learning/gams_aux_comparison_854616.out`

**Benchmark:** Evaluation is on the same held-out NLP4LP test split. Benchmark validity applies only to this evaluation. GAMS-derived labels are **weak/heuristic** and are NOT gold; do not claim benchmark validity for GAMS data.

## Setup

- **Baseline:** Train pairwise ranker on NLP4LP train only, 200 steps, seed 42.
- **Aux run:** Train 50 steps on GAMS auxiliary data (17 models, 1135 rows, weak labels), then 200 steps on NLP4LP from that checkpoint.
- **Evaluation:** Same held-out NLP4LP test for both; same eval script.

## Results (NEGATIVE — do not scale this approach)

### NLP4LP-only baseline

- pairwise_accuracy: 0.182 (2/11)
- slot_selection_accuracy: 0.182
- exact_slot_fill_accuracy: 0.0
- type_match_after_decoding: 0.727 (8/11)

### GAMS-aux then NLP4LP

- pairwise_accuracy: 0.091 (1/11)
- slot_selection_accuracy: 0.091
- exact_slot_fill_accuracy: 0.0
- type_match_after_decoding: 0.0 (0/11)

**Conclusion:** GAMS weak supervision **hurt** performance. We do not continue scaling the broad GAMS weak-label path. See `docs/learning_runs/targeted_synth_vs_nlp4lp_only.md` for the follow-up experiment using targeted high-precision synthetic data instead.

## Artifacts

- **Selected GAMS models:** `artifacts/gams_example_audit/selected_aux_models.json`
- **Converter:** `tools/build_gams_aux_ranker_data.py`
- **Aux data:** `artifacts/learning_ranker_data/gams_aux/train.jsonl`, `stats.json`
- **Batch script:** `batch/learning/train_nlp4lp_gams_aux_comparison.sbatch`
- **Baseline run dir:** `artifacts/learning_runs/nlp4lp_only_baseline/`
- **Aux pretrain dir:** `artifacts/learning_runs/gams_aux_pretrain/`
- **Aux then NLP4LP run dir:** `artifacts/learning_runs/gams_aux_then_nlp4lp/`

## Caveats

- Single seed (42); no variance reported.
- GAMS labels: one positive per (model, slot) by index heuristic; not gold.
- Small aux dataset (17 models, 1135 rows); scaling may behave differently.
