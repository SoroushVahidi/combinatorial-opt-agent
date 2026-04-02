# Normalized Source Matrix

## 1) Fully normalized benchmark datasets (adapter + benchmark semantics)

| Source | Adapter | Status | Notes |
|---|---|---|---|
| NLP4LP | `nlp4lp.py` | Full | Existing repo-native benchmark path |
| NL4Opt (competition) | `nl4opt.py` | Full | Existing adapter over local split JSONL |
| Text2Zinc | `text2zinc.py` | Full | Data download/auth dependent |
| OptMATH | `optmath.py` | Full | Snapshot export script available |
| ComplexOR | `complexor.py` | Full (derived) | Derived from Text2Zinc subset |
| MAMO | `mamo.py` | Full when data present | New adapter with permissive schema mapping |
| StructuredOR | `structuredor.py` | Full when data present | New adapter |
| CardinalOperations/NL4OPT | `cardinal_nl4opt.py` | Full when data present | New adapter, separate from `nl4opt.py` |
| IndustryOR | `industryor.py` | Full when data present | New adapter |

## 2) Partially normalized / data-dependent benchmark datasets

| Source | Adapter | Partial reason |
|---|---|---|
| OptiMUS | `optimus.py` | Adapter is ready; requires external JSONL download |

## 3) Catalog-only normalized sources

| Source | Adapter | Classification |
|---|---|---|
| Gurobi Modeling Examples | `gurobi_modeling_examples.py` | Catalog-only |
| Gurobi OptiMods | `gurobi_optimods.py` | Catalog-only |
| GAMS Models | `gams_models.py` | Catalog-only |
| MIPLIB | `miplib.py` | Catalog-only summary entry |
| OR-Library | `or_library.py` | Catalog-only |
| Pyomo Examples | `pyomo_examples.py` | Catalog-only |

## 4) Source-only / not yet benchmark-normalized collections

| Source | Status |
|---|---|
| Full GAMS source corpus | Source-only (metadata/catalog level in-repo) |
| Full MIPLIB instance corpus | Source-only (large external binaries) |

## Expanded schema catalog
Run:

```bash
python scripts/build_expanded_schema_catalog.py
```

Output:
- `data/processed/expanded_schema_catalog.jsonl`

Each row includes:
- `id`
- `source_dataset`
- `schema_text`
- `source_metadata`
- `entry_status` (`benchmark-ready` / `catalog-only` / `source-only`)
