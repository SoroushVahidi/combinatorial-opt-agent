# Direction change: GAMS weak labels → targeted synthetic data

## What happened

1. **GAMS weak-label experiment (job 854616):** We used 17 GAMSPy models to build auxiliary (slot, mention) pairs with **heuristic labels**: one positive per (model, slot) by index (`pi % len(numerics)`). No natural language; only parameter names and numeric literals from code.
2. **Result:** GAMS-aux then NLP4LP **underperformed** NLP4LP-only on held-out NLP4LP test (e.g. pairwise 9.1% vs 18.2%, type match 0% vs 72.7%). So the broad GAMS weak-label approach **hurt**.
3. **Decision:** Do **not** continue scaling the GAMS weak-label path. Treat it as a **negative result** and document it.

## Why GAMS weak labels were likely too noisy

- **No NL alignment:** GAMS data is code/symbolic; slot names and numbers have no natural-language context. The model learned code-like patterns that did not transfer to NL problem text.
- **Arbitrary positive assignment:** The “correct” (slot, mention) pair was chosen by index, not by true semantics. Many positives were wrong; negatives were mixed.
- **Distribution shift:** Aux data (parameter names, numeric arrays) differs from NLP4LP (problem text, “at least X”, “capacity”, etc.). Pretraining on the former can interfere with the latter.

## What we preserve from the GAMS work

- **Two-stage training pipeline:** `--init_checkpoint` in the pairwise ranker; batch scripts that do aux pretrain then NLP4LP finetune.
- **GAMS audit and extraction:** `tools/extract_gams_examples_structured.py`, `artifacts/gams_example_audit/`, `docs/gams_and_additional_datasets_audit.md` remain useful for catalog/vocabulary, not for training labels.
- **Documented negative result:** `docs/learning_runs/gams_aux_vs_nlp4lp_only.md` states clearly that GAMS aux hurt and we do not scale it.

## What we do not reuse for training

- **GAMS-derived training rows** (`artifacts/learning_ranker_data/gams_aux/train.jsonl`) as auxiliary data for the ranker. We replaced that with **targeted synthetic** data.

## New direction: targeted high-precision synthetic data

- **Generator:** `tools/build_targeted_synth_ranker_data.py`.
- **Design:** Small, templated (slot, mention) pairs for **known bottlenecks**: percent vs scalar, total vs per-unit, min/max bounds, capacity/demand, objective vs bound, float values, paraphrase. Each template defines the **correct** fill; no heuristic assignment.
- **Scale:** ~155 rows, 34 templates. Quality over quantity.
- **Experiment:** Job 854618 — baseline NLP4LP-only vs targeted-synth pretrain then NLP4LP finetune; same held-out NLP4LP test.

See `docs/learning_runs/targeted_synth_vs_nlp4lp_only.md` for the comparison and conclusion once the job completes.
