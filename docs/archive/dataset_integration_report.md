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
## Source, License, and Redistribution Notes

| Dataset | Public Source | License Clear | GitHub-vendorable raw data | Schema Retrieval | Scalar Instantiation | Full Formulation | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| NLP4LP | HF `udell-lab/NLP4LP` (existing repo usage) | Access-controlled HF dataset | No (download/script only) | Yes | Yes | Partial | Existing pipeline remains unchanged. |
| NL4Opt | [nl4opt/nl4opt-competition](https://github.com/nl4opt/nl4opt-competition) | Yes (MIT in repo) | Prefer script/manual, no vendored raw archives | Yes | Partial/adapter-dependent | Partial | Script supports official raw paths when accessible; manual fallback documented. |
| Text2Zinc | [skadio/text2zinc](https://huggingface.co/datasets/skadio/text2zinc) | Yes (`cc-by-4.0` in dataset card README), but gated access | No (gated; scripted local export) | No (treated as generation/modeling benchmark) | Yes | Yes | Requires HF access acceptance before download. |
| OptMATH | [optsuite/OptMATH](https://github.com/optsuite/OptMATH), HF `shushulei/OptMATH-Train` | Yes (Apache-2.0 in repo) | No (large: ~1.77GB HF listing) | No (modeled as generation/modeling benchmark) | Yes | Yes | Script exports local JSONL snapshots; default caps rows. |
| ComplexOR (closest public variant) | Text2Zinc subset where `metadata.source == complexor` | Yes for chosen path (inherits Text2Zinc licensing context) | No raw vendoring | No | Yes | Yes | Direct Chain-of-Experts repo did not expose a clear top-level LICENSE in audit; safest path uses clearly licensed public variant. |
| Gurobi Modeling Examples | [Gurobi/modeling-examples](https://github.com/Gurobi/modeling-examples) | See repo LICENSE | Manifest only (no notebooks vendored) | No | No | No | Catalog-only adapter. 55 entries from `data/sources/gurobi_modeling_examples.json`. |
| Gurobi OptiMods | [Gurobi/gurobi-optimods](https://github.com/Gurobi/gurobi-optimods) | See repo LICENSE | Manifest only | No | No | No | Catalog-only adapter. 14 entries from `data/sources/gurobi_optimods.json`. |
| GAMS Model Library | [gams.com](https://www.gams.com/latest/gamslib_ml/libhtml/) | Requires GAMS license | Manifest only | No | No | No | Catalog-only adapter. 143 entries from `data/sources/gams_models.json`. |
| MIPLIB 2017 | [miplib.zib.de](https://miplib.zib.de/) | Open benchmark | Manifest only (no MPS files) | No | No | No | Catalog-only adapter. 1 metadata entry from `data/sources/miplib.json`. |
| OR-Library | [brunel.ac.uk](http://people.brunel.ac.uk/~mastjjb/jeb/info.html) | Public | Manifest only | No | No | No | Catalog-only adapter. 63 problem families from `data/sources/or_library.json`. |
| Pyomo Examples | [Pyomo/pyomo](https://github.com/Pyomo/pyomo) | BSD-3 | Manifest only | No | No | No | Catalog-only adapter. 19 entries from `data/sources/pyomo_examples.json`. |
| OptiMUS | [teshnizi/OptiMUS](https://github.com/teshnizi/OptiMUS) | See repo LICENSE | No (JSONL download only) | Yes | Yes | Yes | Benchmark-ready adapter; reads `data/external/optimus/*.jsonl`; returns empty until downloaded. |

## ComplexOR / ComplexLP Ambiguity Handling

- The requested `ComplexOR / ComplexLP` naming is ambiguous across sources.
- The integration uses the closest clearly licensed public variant:
  - Text2Zinc rows tagged as `complexor` source.
- `scripts/get_complexor.py` derives local `complexor` JSONL files from local Text2Zinc snapshots.

## GitHub Size Decisions

- No large raw dataset archives are committed.
- No parquet/model binaries are committed.
- Only code, tiny fixtures, and docs were added.
- Downloaded artifacts are ignored via `.gitignore`.

## Download and Prepare Commands

```bash
# NL4Opt (automated when official files are reachable; otherwise manual fallback)
python scripts/get_nl4opt.py

# Text2Zinc (requires gated HF access)
python scripts/get_text2zinc.py --max-rows 1000

# OptMATH (large corpus; default exports capped snapshot)
python scripts/get_optmath.py --max-rows 1000

# ComplexOR closest public variant (derived from local Text2Zinc snapshot)
python scripts/get_complexor.py
```

## Evaluation Commands

```bash
# Single dataset
python tools/run_dataset_benchmarks.py --dataset nl4opt

# All integrated datasets
python tools/run_dataset_benchmarks.py --all-datasets
```

Outputs:

- `results/paper/dataset_benchmark_<dataset>.csv`
- `results/paper/dataset_benchmark_summary.csv`
- `results/paper/dataset_benchmark_summary.json`

Unsupported metrics are reported as `N/A` by capability instead of failing.

## What Is Fully Integrated vs Partial

- Fully integrated (adapter + script + runner compatibility):
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
  - OptiMUS (adapter present; reads `data/external/optimus/`)
- Partial/manual aspects:
  - Gated/permissioned datasets (Text2Zinc, potentially some upstream variants) require user auth.
  - NL4Opt and OptMATH can require manual local file placement if upstream structure differs from script assumptions.
  - Catalog-only sources have no benchmark labels; all metrics report N/A.

## Known Limitations

- Existing NLP4LP downstream scoring metrics are not rewritten for all new datasets.
- New runner is capability/readiness-oriented and avoids forcing invalid metric semantics on generation-first datasets.

