# TAT-QA and FinQA Dataset Acquisition Report

**Date:** 2025-03-08  
**Method:** Git clone from official GitHub repositories into `data_external/raw/`.

---

## Part 1: Status Before Acquisition (Verified)

- **TAT-QA:** Not present anywhere in the repo. No matches for `TAT-QA`, `tatqa`, `tat_qa` in code, docs, or configs. No files under `data/` or `data_external/` with those names.
- **FinQA:** Not present anywhere in the repo. No matches for `FinQA`, `finqa` in code, docs, or configs. No such files under `data/` or `data_external/`.

---

## Part 2 & 3: Acquisition and Verification

### TAT-QA

| Field | Value |
|-------|--------|
| **Status** | **success** |
| **Official source used** | https://github.com/NExTplusplus/TAT-QA |
| **Method** | `git clone --depth 1 https://github.com/NExTplusplus/TAT-QA.git tatqa` |
| **Exact local path** | `data_external/raw/tatqa/` |

**Files/folders found:**

- `dataset_raw/` — **actual dataset (canonical):**
  - `tatqa_dataset_train.json` (~12.3 MB)
  - `tatqa_dataset_dev.json` (~1.6 MB)
  - `tatqa_dataset_test.json` (~1.1 MB)
  - `tatqa_dataset_test_gold.json` (~2.2 MB)
- `dataset_tagop/` — alternate format: `tatqa_dataset_train.json`, `tatqa_dataset_dev.json`
- `tatqa_eval.py`, `tatqa_metric.py`, `tatqa_utils.py` — evaluation and utilities
- `sample_prediction.json`, `docs/`, `tag_op/` (code)
- `README.md`, `LICENSE`, `requirement.txt`, `.gitignore`

**Actual dataset present:** Yes. Train/dev/test (and test gold) JSON files are on disk under `dataset_raw/`.

**License:** MIT (Copyright (c) 2021 Fengbin Zhu). See `data_external/raw/tatqa/LICENSE`.

**Structure (brief):** Hybrid tabular + textual QA over financial reports; questions with contexts (tables + paragraphs), answers, and derivations. 16,552 questions, 2,757 contexts (per official description).

**Exact next step for integration:** Add a loader that reads `dataset_raw/tatqa_dataset_*.json`, maps to the repo’s query/schema format (e.g. query text, context, expected answer/derivation), and optionally writes to `data/processed/` (e.g. `tatqa_eval_train.jsonl`, `tatqa_eval_dev.jsonl`, `tatqa_eval_test.jsonl`) for evaluation scripts. Align field names and answer types with existing NLP4LP/NL4Opt pipelines if they will share evaluation code.

---

### FinQA

| Field | Value |
|-------|--------|
| **Status** | **success** |
| **Official source used** | https://github.com/czyssrs/FinQA |
| **Method** | `git clone --depth 1 https://github.com/czyssrs/FinQA.git finqa` |
| **Exact local path** | `data_external/raw/finqa/` |

**Files/folders found:**

- `dataset/` — **actual dataset:**
  - `train.json` (~1.9M lines; ~73 MB)
  - `dev.json` (~268k lines)
  - `test.json` (~356k lines)
  - `private_test.json` (~101k lines)
- `code/` — evaluation and example code (e.g. `evaluate/test.json`, `example_predictions.json`)
- `README.md`, `LICENSE`, `eg-intro.png`
- `.git/`

**Actual dataset present:** Yes. Train, dev, test, and private_test JSON files are on disk under `dataset/`.

**License:** MIT (Copyright (c) 2021 Zhiyu Chen). See `data_external/raw/finqa/LICENSE`.

**Structure (brief):** Numerical reasoning over financial data; each item has `pre_text` (context), question, and reasoning/answer. 8,281 QA pairs, 2.7k reports (per official description). JSON is list-of-objects with pre_text, question, and answer-related fields.

**Exact next step for integration:** Add a loader that reads `dataset/train.json`, `dataset/dev.json`, `dataset/test.json`, maps to the repo’s format (query, context, gold answer/reasoning), and optionally writes `data/processed/finqa_eval_*.jsonl` for shared evaluation. Handle program/reasoning fields if the pipeline will use them.

---

## Part 4: Failure / Partial Summary

- **TAT-QA:** First clone attempt failed with `fatal: unable to create thread: Resource temporarily unavailable` (environment resource limit). **Second attempt succeeded**; no manual action required.
- **FinQA:** Clone succeeded on first attempt.

No authentication, broken links, or missing-data issues. Both repos include the actual data files in the clone.

---

## Part 5: Optional Note on Loader/Preprocessing

- **TAT-QA:** Loader should parse `dataset_raw/tatqa_dataset_*.json` (and optionally `dataset_tagop/` if using TagOp format). Map: context (table + passages) → context field; question → query; answer/derivation → gold. Decide how to represent table structure (e.g. linearized or separate table field) to match downstream models. Evaluation can reuse or wrap `tatqa_metric.py` (EM/F1) if desired.
- **FinQA:** Loader should parse `dataset/*.json` (list of examples). Map: `pre_text` (and any table/text) → context; question → query; answer and reasoning program → gold. If the project uses a single “query + context → scalar/answer” format (e.g. like NLP4LP), flatten FinQA’s structure into that schema; keep program/reasoning as optional metadata for analysis.

No major integration (training scripts, model code, or configs) was started; only acquisition and this prep note.

---

## Part 6: Direct Summary

1. **TAT-QA acquired:** Yes. Data files are present locally.
2. **FinQA acquired:** Yes. Data files are present locally.
3. **Exact local paths:**
   - TAT-QA: `data_external/raw/tatqa/` (data in `data_external/raw/tatqa/dataset_raw/`).
   - FinQA: `data_external/raw/finqa/` (data in `data_external/raw/finqa/dataset/`).
4. **If anything failed:** TAT-QA’s first clone failed due to a transient “unable to create thread” error in the environment; a second clone succeeded. Nothing else failed.
5. **What you need to do manually:** Nothing for acquisition. For integration: add loaders and optional `data/processed/` outputs as in “Exact next step” above.
