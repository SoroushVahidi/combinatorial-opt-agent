# Dataset Integration Report

## Parallel dataset-layer expansion (2026-04-02)

This branch adds **dataset normalization + schema catalog support** only, with no grounding-core edits.

### Added retrieval scripts
- `scripts/get_mamo.py`
- `scripts/get_structuredor.py`
- `scripts/get_cardinal_nl4opt.py`
- `scripts/get_industryor.py`

All four scripts:
- write outputs under `data/external/<dataset>/`
- write `provenance.json` with source/splits/row-counts/method/timestamp/warnings/errors
- fail loudly (non-zero exit) when required splits are unavailable

### Added adapters
- `src/data_adapters/mamo.py`
- `src/data_adapters/structuredor.py`
- `src/data_adapters/cardinal_nl4opt.py`
- `src/data_adapters/industryor.py`

Each adapter maps to `InternalExample` conservatively:
- unknown fields become `None`
- no fabricated labels
- provenance retained in `metadata`

### Registry and benchmark compatibility
- `src/data_adapters/registry.py` updated with new adapter registrations.
- Existing `tools/run_dataset_benchmarks.py` remains unchanged and safely reports `N/A` where metrics are unsupported.

### Expanded schema catalog
- `scripts/build_expanded_schema_catalog.py` updated to emit:
  - benchmark-ready rows
  - catalog-only rows
  - source-only placeholders for registered adapters with no local data
  - additional source-only entries for large/non-vendored sources

### Grounding-pair export stub
- `scripts/export_grounding_training_pairs.py` added.
- Exports dataset-specific lightweight mention/schema pair stubs plus exact blocker summaries when export is not feasible.

### Reporting docs
- `analysis/dataset_parallel_work_audit.md`
- `docs/NORMALIZED_SOURCE_MATRIX.md`
- `docs/DATASET_PARALLEL_INTEGRATION_REPORT.md`

### Git hygiene
- Raw external data remains outside tracked files under ignored `data/external/*/`.
- Only code, docs, and tiny fixtures/tests are committed.
