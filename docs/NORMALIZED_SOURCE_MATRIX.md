# Normalized Source Matrix

This document describes the normalization status of every external data source referenced in the
repository, relative to the `src/data_adapters/` layer.

---

## 1. Fully Benchmark-Ready Normalized Datasets

Sources with labeled NL queries, gold parameters, and/or formulations that support meaningful
evaluation metrics.

| Source | Adapter | Capabilities | Splits | Notes |
|---|---|---|---|---|
| NLP4LP | `nlp4lp.py` | Schema retrieval ✓, Scalar inst. ✓ | orig, noisy, short | Backed by `data/processed/`; existing pipeline unchanged |
| NL4Opt | `nl4opt.py` | Schema retrieval ✓, Scalar inst. ✓, Formulation ✓ | train, dev, test | `data/external/nl4opt/` |
| Text2Zinc | `text2zinc.py` | Scalar inst. ✓, Solver eval ✓, Formulation ✓ | train, validation, test | `data/external/text2zinc/` |
| OptMATH | `optmath.py` | Scalar inst. ✓, Solver eval ✓, Formulation ✓ | train, validation, test, bench | `data/external/optmath/` |
| ComplexOR | `complexor.py` | Scalar inst. ✓, Solver eval ✓, Formulation ✓ | train, validation, test | Derived from Text2Zinc |

---

## 2. Benchmark-Ready When Data Is Present (Dynamic)

Sources that are fully normalized as benchmark datasets but require external data to be downloaded.

| Source | Adapter | Capabilities | Notes |
|---|---|---|---|
| OptiMUS | `optimus.py` | Schema retrieval ✓, Scalar inst. ✓, Formulation ✓ | Reads `data/external/optimus/*.jsonl`; returns 0 splits until downloaded from https://github.com/teshnizi/OptiMUS |

---

## 3. Catalog-Only Normalized Sources

Sources normalized as schema/problem catalogs. They provide problem families or schema entries
but do **not** provide labeled NL-query → formulation pairs. All benchmark metrics are `N/A`.

| Source | Adapter | Entry count | Entry type | Source URL |
|---|---|---|---|---|
| Gurobi Modeling Examples | `gurobi_modeling_examples.py` | 55 | Example folders | https://github.com/Gurobi/modeling-examples |
| Gurobi OptiMods | `gurobi_optimods.py` | 14 | Module names | https://github.com/Gurobi/gurobi-optimods |
| GAMS Model Library | `gams_models.py` | 143 | Model names | https://www.gams.com/latest/gamslib_ml/libhtml/ |
| MIPLIB 2017 | `miplib.py` | 1 | Benchmark collection entry | https://miplib.zib.de/ |
| OR-Library | `or_library.py` | 63 | Problem families | http://people.brunel.ac.uk/~mastjjb/jeb/info.html |
| Pyomo Examples | `pyomo_examples.py` | 19 | Example names | https://github.com/Pyomo/pyomo |

All catalog-only adapters have:
- `supports_schema_retrieval = False`
- `supports_scalar_instantiation = False`
- `supports_solver_eval = False`
- `supports_full_formulation = False`
- `metadata.catalog_only = True`

---

## 4. Raw Source Manifests (Not Yet Fully Normalized)

Sources present as JSON manifests in `data/sources/` but with limited normalization coverage
beyond the catalog-only entries described above.

| Source | Location | Reason not fully normalized |
|---|---|---|
| MIPLIB 2017 instance files | `data/sources/miplib.json` (metadata only) | Raw `.mps` instance files are large binaries; not vendored. Catalog entry covers metadata only. |
| Gurobi notebook content | `data/sources/gurobi_modeling_examples.json` (folder names only) | Jupyter notebooks not vendored; no stable lightweight download path. |
| GAMS `.gms` source files | `data/sources/gams_models.json` (model names only) | Require GAMS license to access and run. Name-level normalization only. |
| Real-world queries | `data/sources/real_world_queries.json` | Internal synthetic queries used ad-hoc; no external normalization target. |

---

## Expanded Schema Catalog

The script `scripts/build_expanded_schema_catalog.py` collects all schema entries from both
benchmark and catalog-only adapters and writes them to:

```
data/processed/expanded_schema_catalog.jsonl
```

Each row includes: `id`, `source_dataset`, `schema_id`, `schema_text`, `source_url`,
`catalog_only`, `benchmark_labeled`, `nl_query`, `metadata`.

Run with:
```bash
python scripts/build_expanded_schema_catalog.py
```
