# Learning experiment records

This folder holds records of learning experiments for the NLP4LP mention–slot grounding pipeline.

## Key document

- **real_data_only_learning_check.md** — Real-data-only benchmark (no synthetic/GAMS aux) on the largest valid NLP4LP split. Documents split integrity, training config, metrics, and comparison to the deterministic rule baseline. **Conclusion:** learning did not outperform the rule baseline on this setup; learning is documented as future work. Use this as the reference for reproducible, benchmark-safe evaluation.

## Other records

Other files in this folder (e.g. first_learning_run_record.md, gams_aux_vs_nlp4lp_only.md, targeted_synth_vs_nlp4lp_only.md, valid_first_learning_run_record.md) are historical or exploratory. GAMS weak-label aux and targeted synthetic aux were tried; results were negative or stopped. Do not treat them as recommended paths.

## Benchmark-safe scripts

- **Valid train/dev/test (no fallback):** `batch/learning/train_nlp4lp_valid_first_run.sbatch`, `scripts/learning/submit_valid_first_run.sh`
- **Real-data-only check (train + eval + rule comparison):** `batch/learning/train_nlp4lp_real_data_only_learning_check.sbatch`, `scripts/learning/run_real_data_only_learning_check.sh`
