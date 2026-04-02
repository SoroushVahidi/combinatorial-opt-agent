# Learning Stage 1 Implementation

First learning-stage infrastructure for the natural-language optimization instantiation project: common corpus, validation, NLP4LP pairwise ranker data, and a minimal train/eval scaffold.

## What was implemented

### A) Common intermediate corpus

- **docs/LEARNING_CORPUS_FORMAT.md** — Canonical JSONL schema and field descriptions.
- **src/learning/common_corpus_schema.py** — Schema constants, `validate_record()`, helpers `slot_to_dict` / `mention_to_dict`.
- **src/learning/build_common_grounding_corpus.py** — Builds JSONL from:
  - **NLP4LP:** `data/processed/nlp4lp_eval_orig.jsonl`, `data/catalogs/nlp4lp_catalog.jsonl`, gold from HF or `NLP4LP_GOLD_CACHE`; train/dev from HF `udell-lab/NLP4LP`.
  - **NL4Opt:** `data_external/raw/nl4opt_competition/generation_data/{train,dev,test}.jsonl`.
  - **TAT-QA:** `data_external/raw/tatqa/dataset_raw/tatqa_dataset_{train,dev,test}.json`.
  - **FinQA:** `data_external/raw/finqa/dataset/{train,dev,test}.json`.
- **artifacts/learning_corpus/README.md** — Short description of produced files.

### B) Validation and audit

- **src/learning/validate_common_grounding_corpus.py** — Validates required fields and schema per record; exits non-zero if any invalid.
- **src/learning/summarize_common_grounding_corpus.py** — Counts by dataset/split, slot/entity/bound/role supervision, missing fields; writes **artifacts/learning_corpus/corpus_summary.md** and **corpus_summary.json**; optional spot-checks (random examples, multi-float, bound cues, multi-entity).

### C) NLP4LP pairwise ranker data

- **src/learning/build_nlp4lp_pairwise_ranker_data.py** — Reads NLP4LP corpus JSONL, emits one row per (slot, mention) with binary label, `group_id`, slot/mention fields, handcrafted features (`feat_type_match`, `feat_operator_cue_match`, `feat_total_like`, `feat_per_unit_like`, `feat_slot_mention_overlap`), and `gold_mention_id_for_slot`. Output: **artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl**.

### D) Training / eval scaffold

- **src/learning/models/features.py** — `row_to_feature_vector()` for handcrafted features.
- **src/learning/models/decoding.py** — `argmax_per_slot()` and `one_to_one_matching()`.
- **src/learning/models/pairwise_ranker.py** — `PairwiseRanker`: encoder (default `distilroberta-base`), optional structured features, `score(slot, mention)`; saves/loads checkpoint.
- **src/learning/train_nlp4lp_pairwise_ranker.py** — Loads ranker data, trains with BCE loss (optional `--use_features`), writes **artifacts/learning_runs/<run_name>/checkpoint.pt** and **config.json**.
- **src/learning/eval_nlp4lp_pairwise_ranker.py** — Loads test (or dev) ranker data, runs ranker (or rule baseline if no checkpoint), decodes with argmax or one-to-one; reports **pairwise_accuracy**, **slot_selection_accuracy**, **exact_slot_fill_accuracy**, **type_match_after_decoding**; writes **metrics.json** and **predictions.jsonl** under an output dir.

### E) Batch and scripts

- **batch/learning/build_common_corpus.sbatch** — Build corpus; env: `DATASET`, `SPLIT`, `OUTPUT_DIR`, `MAX_EXAMPLES`, `SEED`, `VERBOSE`.
- **batch/learning/validate_common_corpus.sbatch** — Validate and summarize corpus.
- **batch/learning/build_nlp4lp_ranker_data.sbatch** — Build NLP4LP pairwise data.
- **batch/learning/train_nlp4lp_ranker.sbatch** — Train ranker; env: `RUN_NAME`, `ENCODER`, `USE_FEATURES`, `EPOCHS`, `MAX_STEPS`, etc.
- **batch/learning/eval_nlp4lp_ranker.sbatch** — Eval ranker; env: `RUN_DIR`, `SPLIT`, `DECODER`.
- **scripts/learning/run_build_common_corpus.sh** — Wrapper (optional `--sbatch`, `--dataset`, `--split`, `--max_examples`).
- **scripts/learning/run_validate_corpus.sh**, **run_build_nlp4lp_ranker_data.sh**, **run_train_nlp4lp_ranker.sh**, **run_eval_nlp4lp_ranker.sh** — Local or sbatch wrappers.

Logs: **logs/learning/** (sbatch stdout/stderr).

## Corpus format

See **docs/LEARNING_CORPUS_FORMAT.md**. Each record has: `dataset`, `split`, `instance_id`, `source_path`, `problem_text`, `schema_name`, `schema_description`, `slots`, `numeric_mentions`, `gold_slot_assignments`, `role_labels`, `entity_labels`, `bound_labels`, `metadata`. Slots and mentions use the documented sub-schemas.

## How to build corpus

From repo root:

```bash
python -m src.learning.build_common_grounding_corpus --dataset all --split all --output_dir artifacts/learning_corpus
```

With limits and verbosity:

```bash
python -m src.learning.build_common_grounding_corpus --dataset nlp4lp --split test --max_examples 100 --verbose
```

**NLP4LP train/dev:** Requires HuggingFace `udell-lab/NLP4LP` (and token in `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`) or a pre-built **NLP4LP_GOLD_CACHE** JSON file for that split.

Submit via batch:

```bash
sbatch batch/learning/build_common_corpus.sbatch
# Or: DATASET=nlp4lp SPLIT=test MAX_EXAMPLES=500 sbatch batch/learning/build_common_corpus.sbatch
```

## How to validate corpus

```bash
python -m src.learning.validate_common_grounding_corpus --corpus_dir artifacts/learning_corpus --verbose
python -m src.learning.summarize_common_grounding_corpus --corpus_dir artifacts/learning_corpus
```

Or: `./scripts/learning/run_validate_corpus.sh` or `sbatch batch/learning/validate_common_corpus.sbatch`.

## How to build NLP4LP pairwise ranker data

After the corpus is built:

```bash
python -m src.learning.build_nlp4lp_pairwise_ranker_data \
  --corpus_dir artifacts/learning_corpus \
  --output_dir artifacts/learning_ranker_data/nlp4lp
```

Or: `./scripts/learning/run_build_nlp4lp_ranker_data.sh` or `sbatch batch/learning/build_nlp4lp_ranker_data.sbatch`.

## How to run training and evaluation

**Training** (requires `torch` and `transformers`):

```bash
python -m src.learning.train_nlp4lp_pairwise_ranker \
  --run_name run0 --encoder distilroberta-base --max_steps 100
```

With structured features: `--use_features`. Submit: `RUN_NAME=run0 sbatch batch/learning/train_nlp4lp_ranker.sbatch`.

**Evaluation** (works without checkpoint; uses rule baseline):

```bash
python -m src.learning.eval_nlp4lp_pairwise_ranker --split test
```

With checkpoint: `--run_dir artifacts/learning_runs/run0`. Submit: `RUN_DIR=artifacts/learning_runs/run0 sbatch batch/learning/eval_nlp4lp_ranker.sbatch`.

## Where outputs and logs go

| Output | Path |
|--------|------|
| Corpus JSONL | `artifacts/learning_corpus/{dataset}_{split}.jsonl` |
| Corpus summary | `artifacts/learning_corpus/corpus_summary.md`, `corpus_summary.json` |
| Ranker data | `artifacts/learning_ranker_data/nlp4lp/{train,dev,test}.jsonl` |
| Run checkpoint/config | `artifacts/learning_runs/<run_name>/checkpoint.pt`, `config.json` |
| Eval metrics/predictions | `artifacts/learning_runs/eval_out/` or `--out_dir` |
| Sbatch logs | `logs/learning/*.out`, `logs/learning/*.err` |

## Intentionally not implemented yet

- NL4Opt / TAT-QA / FinQA **multitask training** — only the common corpus is produced so they can be added later.
- Full-scale training (long runs, large models) — Stage 1 is a minimal baseline (e.g. `max_steps=100`).
- Integration of the learned ranker into the existing NLP4LP **downstream pipeline** (e.g. `run_setting` with a learned assignment mode) — to be added when the ranker is stable.

## Recommended next implementation step

1. **Build and validate corpus:** Run the corpus builder for at least NLP4LP test (and optionally train/dev if HF is available), then validate and summarize.
2. **Build ranker data:** Run `build_nlp4lp_pairwise_ranker_data.py`.
3. **Run eval with rule baseline:** Run `eval_nlp4lp_pairwise_ranker.py` with no `--run_dir` to get baseline metrics.
4. **Optional:** Install `torch` and `transformers`, run a short training (`train_nlp4lp_pairwise_ranker.py --max_steps 100`), then eval with `--run_dir`.

After that, the next logical step is to wire the learned ranker into the NLP4LP evaluation pipeline and/or add NL4Opt auxiliary data to the ranker dataset.
