# CP-Bench (DCP-Bench-Open) — local staging directory

This folder holds **optional** downloaded artifacts for the public **DCP-Bench-Open**
repository ([github.com/DCP-Bench/DCP-Bench-Open](https://github.com/DCP-Bench/DCP-Bench-Open)),
licensed under **Apache-2.0**.

## Populate

From the repository root:

```bash
python scripts/datasets/get_cp_bench_open.py
```

This writes `sample_test.jsonl` and `staging_manifest.json` here. Contents are
**gitignored** except this README.

## Adapter

The `cp_bench` dataset adapter (`src/data_adapters/cp_bench.py`) reads `*.jsonl`
splits from this directory. **No paper-table metrics** are produced automatically;
see `docs/DATASET_EXPANSION_STATUS.md`.
