# Dataset Parallel Work Audit (April 2, 2026)

## Scope checked
- `src/data_adapters/`
- `src/data_adapters/registry.py`
- `tools/run_dataset_benchmarks.py`
- `docs/dataset_integration_report.md`
- `docs/data_sources.md`
- existing `scripts/get_*.py`

## Current normalization state before this pass

### Already adapter-normalized benchmark datasets
- NLP4LP
- NL4Opt
- Text2Zinc
- OptMATH
- ComplexOR (derived path)
- OptiMUS (adapter present; data-dependent)

### Already adapter-normalized catalog-only sources
- Gurobi Modeling Examples
- Gurobi OptiMods
- GAMS Model Library
- MIPLIB 2017 (summary-level catalog entry)
- OR-Library
- Pyomo Examples

### Missing / partial for target parallel scope
- **MAMO**: no adapter or retrieval script in `scripts/get_*.py`; referenced in docs only.
- **StructuredOR**: no adapter; no retrieval script.
- **CardinalOperations/NL4OPT**: existing `nl4opt` adapter is wired to competition files, not separate CardinalOperations namespace.
- **IndustryOR**: no adapter; no retrieval script.

## Risk and conflict notes
- High-conflict grounding-core files listed in task were not required for dataset-layer integration.
- Existing benchmark runner is capability-based and already tolerant of unsupported metrics via `N/A`; can be reused with additive adapters.

## Planned additive work
1. Add retrieval scripts with explicit provenance/error output.
2. Add four new adapters (`mamo`, `structuredor`, `cardinal_nl4opt`, `industryor`).
3. Register adapters in registry only (no grounding-core edits).
4. Expand schema catalog builder to include benchmark/catalog/source-only statuses.
5. Add training-pair export stub and smoke tests/fixtures for new adapters.
