# Dataset Integration Report

## What Was Added

### Round 1 (initial integration layer)
- A new generic dataset integration layer in `src/data_adapters/`:
  - `base.py` (normalized schema + capabilities)
  - `registry.py`
  - `nlp4lp.py`, `nl4opt.py`, `text2zinc.py`, `optmath.py`, `complexor.py`
- A new capability-aware runner:
  - `tools/run_dataset_benchmarks.py`
- Download/prepare scripts:
  - `scripts/get_nl4opt.py`
  - `scripts/get_text2zinc.py`
  - `scripts/get_optmath.py`
  - `scripts/get_complexor.py`
- Lightweight fixtures/tests:
  - `tests/fixtures/datasets/*`
  - `tests/test_dataset_adapters.py`
- GitHub size protection updates:
  - `.gitignore` updated for `data/external/{nl4opt,text2zinc,optmath,complexor}`

This work is additive and keeps `tools/nlp4lp_downstream_utility.py` as the NLP4LP-specific path.

### Round 2 (catalog-only and OptiMUS normalization)
- 7 new adapters normalizing previously raw-manifest-only sources:
  - `src/data_adapters/gurobi_modeling_examples.py` (catalog-only, 55 entries)
  - `src/data_adapters/gurobi_optimods.py` (catalog-only, 14 entries)
  - `src/data_adapters/gams_models.py` (catalog-only, 143 entries)
  - `src/data_adapters/miplib.py` (catalog-only, 1 entry)
  - `src/data_adapters/or_library.py` (catalog-only, 63 entries)
  - `src/data_adapters/pyomo_examples.py` (catalog-only, 19 entries)
  - `src/data_adapters/optimus.py` (benchmark-ready when JSONL data is downloaded)
- Expanded schema catalog build script:
  - `scripts/build_expanded_schema_catalog.py`
- Audit report:
  - `analysis/missing_normalized_sources_audit.md`
- New documentation:
  - `docs/NORMALIZED_SOURCE_MATRIX.md`
- Fixture for OptiMUS adapter tests:
  - `tests/fixtures/datasets/optimus/test.jsonl`
- `.gitignore` updated for `data/external/optimus/` and `data/processed/expanded_schema_catalog.jsonl`

### Round 3 (parallel dataset normalization for MAMO / StructuredOR / Cardinal NL4OPT / IndustryOR)
- New retrieval scripts:
  - `scripts/get_mamo.py`
  - `scripts/get_structuredor.py`
  - `scripts/get_cardinal_nl4opt.py`
  - `scripts/get_industryor.py`
- New adapters:
  - `src/data_adapters/mamo.py`
  - `src/data_adapters/structuredor.py`
  - `src/data_adapters/cardinal_nl4opt.py`
  - `src/data_adapters/industryor.py`
- Registry updates in `src/data_adapters/registry.py`.
- Expanded schema catalog script now emits explicit `entry_status` and `source_metadata` fields.
- Added grounding-pair export stub:
  - `scripts/export_grounding_training_pairs.py`
- Added tiny fixtures and smoke tests for the new adapters and export flow.

Note on retrieval realism:
- retrieval scripts intentionally fail loudly when blocked and always write provenance metadata.
- in restricted environments, upstream connectivity can fail (e.g., tunnel/network `403`), which is reported in `docs/DATASET_PARALLEL_INTEGRATION_REPORT.md`.

## Internal Normalized Example Schema

All adapters map to `InternalExample` with the following fields:

- `id`
- `source_dataset`
- `split`
- `nl_query`
- `schema_id`
- `schema_text`
- `candidate_schemas`
- `scalar_gold_params`
- `structured_gold_params`
- `formulation_text`
- `solver_artifact_path`
- `metadata`

Unavailable fields are set to `None`.

## Adapter Interface

Each adapter exposes:

- `list_splits()`
- `load_split(split_name)`
- `iter_examples(split_name)`
- `to_internal_example(example, split_name)`
- `get_schema_candidates()`
- `get_gold_targets(split_name)`

Capability flags per dataset:

- `supports_schema_retrieval`
- `supports_scalar_instantiation`
- `supports_solver_eval`
- `supports_full_formulation`

## What Is Fully Integrated vs Partial

- Fully integrated (adapter + script + runner compatibility), data-access dependent:
  - NL4Opt
  - Text2Zinc
  - OptMATH
  - ComplexOR (closest public variant path)
  - MAMO
  - StructuredOR
  - CardinalOperations/NL4OPT
  - IndustryOR
  - Gurobi Modeling Examples (catalog-only)
  - Gurobi OptiMods (catalog-only)
  - GAMS Model Library (catalog-only)
  - MIPLIB 2017 (catalog-only)
  - OR-Library (catalog-only)
  - Pyomo Examples (catalog-only)
- Benchmark-ready when data is downloaded:
  - OptiMUS

## Known Limitations

- Existing NLP4LP downstream scoring metrics are not rewritten for all new datasets.
- New runner is capability/readiness-oriented and avoids forcing invalid metric semantics on generation-first datasets.
