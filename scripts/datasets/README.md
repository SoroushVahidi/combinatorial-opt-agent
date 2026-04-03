# Dataset staging scripts

Small, non-destructive helpers to **download or stage** external corpora under `data/external/`.
Staged bytes are **gitignored**; only docs and this README are tracked.

| Script | Dataset | Notes |
|--------|---------|--------|
| [`get_cp_bench_open.py`](get_cp_bench_open.py) | **CP-Bench** (DCP-Bench-Open) | Public `sample_test.jsonl` → `data/external/cp_bench/` |
| [`../get_text2zinc.py`](../get_text2zinc.py) | **Text2Zinc** | Hugging Face `skadio/text2zinc` (**gated**; requires `HF_TOKEN`) |

See also:

- [`docs/DATASET_EXPANSION_PLAN.md`](../../docs/DATASET_EXPANSION_PLAN.md)
- [`docs/DATASET_EXPANSION_STATUS.md`](../../docs/DATASET_EXPANSION_STATUS.md)
- [`data/dataset_registry.json`](../../data/dataset_registry.json)
