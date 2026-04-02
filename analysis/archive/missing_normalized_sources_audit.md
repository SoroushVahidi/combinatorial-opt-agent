# Missing Normalized Sources Audit

**Date:** 2026-04-02  
**Scope:** `src/data_adapters/` normalization layer

---

## 1. Already Normalized (pre-existing adapters)

| Source | Adapter | Split(s) | Benchmark-ready? | Notes |
|---|---|---|---|---|
| NLP4LP | `nlp4lp.py` | orig, noisy, short | Yes (schema retrieval + scalar) | Backed by `data/processed/nlp4lp_eval_*.jsonl` |
| NL4Opt | `nl4opt.py` | train, dev, test | Yes (scalar instantiation + schema) | Backed by `data/external/nl4opt/` |
| Text2Zinc | `text2zinc.py` | train, validation, test | Yes (formulation + solver eval) | Backed by `data/external/text2zinc/` |
| OptMATH | `optmath.py` | train, validation, test, bench | Yes (formulation + solver eval) | Backed by `data/external/optmath/` |
| ComplexOR | `complexor.py` | train, validation, test | Yes (formulation + solver eval) | Derived from Text2Zinc |

---

## 2. Newly Normalized (this audit)

All sources below were previously present only as raw JSON manifests in `data/sources/`. They are now
normalized through the adapter layer as **catalog-only** sources. Each adapter reads the manifest
and emits one `InternalExample` per catalog entry under a `"catalog"` split.

| Source | Adapter | Entry count | Capability | Notes |
|---|---|---|---|---|
| Gurobi Modeling Examples | `gurobi_modeling_examples.py` | 55 | catalog-only | Folder names from `data/sources/gurobi_modeling_examples.json` |
| Gurobi OptiMods | `gurobi_optimods.py` | 14 | catalog-only | Module names from `data/sources/gurobi_optimods.json` |
| GAMS Model Library | `gams_models.py` | 143 | catalog-only | Model names from `data/sources/gams_models.json` |
| MIPLIB 2017 | `miplib.py` | 1 | catalog-only | Single entry from `data/sources/miplib.json` |
| OR-Library | `or_library.py` | 63 | catalog-only | Problem families from `data/sources/or_library.json` |
| Pyomo Examples | `pyomo_examples.py` | 19 | catalog-only | Example names from `data/sources/pyomo_examples.json` |
| OptiMUS | `optimus.py` | 0 (dynamic) | benchmark-ready when data present | Reads `data/external/optimus/*.jsonl`; currently empty until download script is run |

### Catalog-only normalization rationale

Sources such as Gurobi Modeling Examples, GAMS Model Library, OR-Library, etc. provide **problem
families / schema entries** rather than labeled NL-query → formulation pairs. Forcing them into
a benchmark evaluation format would require fabricating labels (which is explicitly prohibited).

Instead, they are normalized as schema catalog entries:
- `schema_id` = original identifier (folder name, model name, problem family key)
- `schema_text` = human-readable label derived from the identifier
- `nl_query` = human-readable label (no invented NL query)
- All benchmark fields (`scalar_gold_params`, `formulation_text`, etc.) = `None`
- `metadata.catalog_only = True`

This makes them usable for catalog expansion and schema retrieval pool expansion without misrepresenting their data quality.

---

## 3. Sources that Remain Source-Only / Unnormalized

| Source | Reason |
|---|---|
| Real-world queries (`data/sources/real_world_queries.json`) | Internal synthetic queries; no stable external source to normalize against. Already used ad-hoc in the pipeline. |
| MIPLIB instance files (`.mps`) | Large binary files; not vendored. The manifest entry is normalized; raw instances would require a separate download + parser pipeline. |
| Gurobi notebook raw content | Jupyter notebooks not vendored; no stable download path in repo. Folder names are normalized as catalog entries. |
| GAMS `.gms` source files | Require GAMS license to execute; only model names are available without licensing. Name-level normalization is included. |

---

## 4. OptiMUS / NLP4LP Overlap Note

The `docs/data_sources.md` document lists "NLP4LP / OptiMUS" as a single entry, and the existing
`NLP4LPAdapter` covers the processed NLP4LP evaluation sets. The new `OptiMUSAdapter` provides a
separate adapter for raw OptiMUS JSONL files (from https://github.com/teshnizi/OptiMUS), if
downloaded to `data/external/optimus/`. These are complementary paths, not duplicates.

---

## 5. Summary Counts

| State | Count |
|---|---|
| Already normalized (pre-existing) | 5 |
| Newly normalized as catalog-only | 6 |
| Newly normalized as benchmark-ready (data-dependent) | 1 (OptiMUS) |
| Remaining source-only / not normalizable without raw data | 4 (MIPLIB instances, Gurobi notebooks, GAMS files, real-world queries) |
