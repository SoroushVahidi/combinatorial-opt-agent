# Dataset expansion status (Text2Zinc & CP-Bench)

Concise record of **what exists in the repo today** versus **what is still future work**.
For motivation and scope, see [`DATASET_EXPANSION_PLAN.md`](DATASET_EXPANSION_PLAN.md).

## Summary

| Dataset | Registry key | Adapter | Staging script | Local data in repo | Paper / camera-ready eval |
|---------|--------------|---------|----------------|-------------------|---------------------------|
| **NLP4LP** | `nlp4lp` | Yes | HF + existing pipeline | Processed eval files when built | **Yes** (Tables 1–5) |
| **Text2Zinc** | `text2zinc` | Yes (existing) | `scripts/get_text2zinc.py` | **No** (HF gated; `data/external/text2zinc/` gitignored) | **No** |
| **CP-Bench** | `cp_bench` | **Yes** (new) | `scripts/datasets/get_cp_bench_open.py` | **No** by default (gitignored); **tests** ship `tests/fixtures/datasets/cp_bench/sample_test.jsonl` | **No** |

## Files and scripts added (this integration)

- `docs/DATASET_EXPANSION_PLAN.md`, `docs/DATASET_EXPANSION_STATUS.md` (this file)
- `data/dataset_registry.json` — metadata for `nlp4lp`, `text2zinc`, `cp_bench`
- `src/datasets/registry.py` — `load_registry()`, `get_dataset_entry()`
- `src/data_adapters/cp_bench.py` — DCP-Bench-Open JSONL → `InternalExample`
- `scripts/datasets/get_cp_bench_open.py`, `scripts/datasets/README.md`
- `data/external/cp_bench/README.md` — staging directory documentation (tracked)
- `tests/fixtures/datasets/cp_bench/sample_test.jsonl` + `SOURCE.txt` (Apache-2.0 upstream sample)
- `src/data_adapters/registry.py` — register `cp_bench`
- Tests: `tests/test_dataset_adapters.py` (registry + CP-Bench fixture coverage)

## What was validated (lightweight)

- **Unit tests:** CP-Bench adapter parses the **fixture** `sample_test.jsonl`; `dataset_registry.json` loads; `cp_bench` appears in `list_datasets()`.
- **Optional:** Run `python scripts/datasets/get_cp_bench_open.py` on a networked machine to populate `data/external/cp_bench/` (gitignored bytes + `staging_manifest.json`).

## What remains before full external benchmarking

1. **Text2Zinc:** Obtain Hugging Face access; run `scripts/get_text2zinc.py`; define evaluation metrics and (if needed) MiniZinc/CP tooling.
2. **CP-Bench:** Decide whether to stage the **full** `dataset/` tree (large) or stay on JSONL exports; define how (if at all) NL queries are obtained or paired; CPMPy execution is **not** wired into this repo’s NLP4LP evaluator.
3. **Common:** Any new tables or claims must cite **new artifacts** under `results/` with provenance updates in `docs/RESULTS_PROVENANCE.md`.

## Dataset sources

- **Text2Zinc:** [Hugging Face `skadio/text2zinc`](https://huggingface.co/datasets/skadio/text2zinc) (gated; CC-BY-4.0 per dataset card — confirm on Hub).
- **CP-Bench (DCP-Bench-Open):** [github.com/DCP-Bench/DCP-Bench-Open](https://github.com/DCP-Bench/DCP-Bench-Open) (Apache-2.0).

## Terminology

- **CP-Bench** in docs/registry refers to the **public DCP-Bench-Open** corpus unless stated otherwise. The short registry key is `cp_bench` for stable Python imports.
