# Public Optimization Data Collection

This document describes the public datasets and source repositories collected for training, development, and external evaluation of the NLP4LP / optimization-problem understanding pipeline. **No training has been performed yet;** this is a data-collection and inventory step.

## Layout

- **`data_external/raw/`** — Raw cloned or downloaded sources (one subdirectory per source).
- **`data_external/manifests/`** — Machine-readable manifest and collection logs.
- **`data_external/scripts/`** — Script used to collect data: `collect_public_optimization_data.sh`.

## What Was Collected

### Successfully collected (automatic)

| Source | Local path | Size (approx) | Main contents |
|--------|------------|---------------|---------------|
| **NL4Opt competition** | `data_external/raw/nl4opt_competition` | ~19 MB | `generation_data/` (train/dev/test .jsonl), `ner_data/`, README, LICENSE (MIT) |
| **ORQA** | `data_external/raw/orqa` | ~13 MB | `dataset/ORQA_test.jsonl`, `ORQA_validation.jsonl`, `src/` (eval scripts) |
| **Mamo** | `data_external/raw/mamo` | ~5 MB | JSON/JSONL data, Python and shell scripts |
| **OptMATH** | `data_external/raw/optmath` | ~16 MB | `benchmark/OptMATH_Bench.json`, MAMO_*.json, NL4OPT.json, assets |
| **DCP-Bench-Open** | `data_external/raw/dcp_bench_open` | ~3.4 MB | `sample_test.jsonl`, problem dirs (JSON, Python), eval framework |
| **Gurobi modeling examples** | `data_external/raw/gurobi_examples` | ~254 MB | 50+ example folders, each with README + Jupyter notebooks (.ipynb) |
| **OR-Tools** | `data_external/raw/ortools_examples` | ~419 MB | Full repo; examples under `examples/` (Python, C++, Java, data) |
| **CSPLib** | `data_external/raw/csplib` | ~353 MB | Problem descriptions, MiniZinc models, packages (CC BY 4.0) |
| **MiniZinc benchmarks** | `data_external/raw/minizinc_benchmarks` | ~326 MB | Many .mzn/.dzn files (MIT) |

### Not collected automatically

| Source | Reason | Manual steps |
|--------|--------|----------------|
| **GAMS Model Library** | No single public git or archive; web catalog only at https://www.gams.com/latest/gamslib_ml/libhtml/ | Browse catalog; download selected .gms or use existing `data/sources/gams_models.json` if populated. |

## Exact Source Addresses

- **NL4Opt competition:** https://github.com/nl4opt/nl4opt-competition  
- **ORQA:** https://github.com/nl4opt/ORQA  
- **Mamo:** https://github.com/FreedomIntelligence/Mamo  
- **OptMATH:** https://github.com/optsuite/OptMATH  
- **DCP-Bench-Open:** https://github.com/DCP-Bench/DCP-Bench-Open  
- **Gurobi modeling examples:** https://github.com/Gurobi/modeling-examples  
- **OR-Tools:** https://github.com/google/or-tools  
- **CSPLib:** https://github.com/csplib/csplib  
- **MiniZinc benchmarks:** https://github.com/MiniZinc/minizinc-benchmarks  
- **GAMS Model Library (manual):** https://www.gams.com/latest/gamslib_ml/libhtml/  

## Licenses (when visible)

- **nl4opt_competition:** MIT (Copyright 2022 Huawei Technologies)  
- **dcp_bench_open:** Apache-2.0  
- **gurobi_examples:** Apache-2.0 (notebooks); running code requires Gurobi license  
- **ortools_examples:** Apache-2.0 (OR-Tools); SCIP has separate license  
- **csplib:** CC BY 4.0  
- **minizinc_benchmarks:** MIT (public domain)  
- **orqa / mamo / optmath:** See each repository’s LICENSE or README  

## What Each Source Is Useful For

- **NL4Opt, OptMATH, Mamo:** Direct NL–formulation pairs or long NL descriptions; **training**, **dev**, **test** or **external eval**.
- **ORQA, DCP-Bench-Open:** QA/constraint-modeling benchmarks; **external evaluation** and **dev**.
- **Gurobi examples, OR-Tools examples:** **Mining** for (NL in notebooks/docs + code/formulation) pairs; optional **training** or **weak supervision**.
- **CSPLib, MiniZinc benchmarks:** **Mining** for constraint models and problem structure; **external eval** for CP-style tasks.
- **GAMS Model Library:** **Mining** if collected (formulations, limited NL in catalog).

## What Failed and Why

- **GAMS Model Library:** Not failed, but **not downloaded** — no public git or single archive; catalog is web-only. Documented as `manual_needed` in the manifest.

## What Requires Manual Action

- **GAMS:** Use the web catalog to download selected models or run a compliant crawl; or rely on existing project metadata in `data/sources/gams_models.json` if present.

## How to Re-run Collection

From the repository root:

```bash
bash data_external/scripts/collect_public_optimization_data.sh
```

- The script creates `data_external/raw/` and `data_external/manifests/` if missing.
- It **skips** any source directory that already exists and contains content (e.g. `.git` or files).
- It does **not** overwrite existing data.
- Log is written under `data_external/manifests/collection_YYYYMMDD_HHMMSS.log`.

## Basic Inventory (at collection time)

- **nl4opt_competition:** ~39 files; key: `generation_data/*.jsonl`, `ner_data`, LICENSE, README.  
- **orqa:** ~55 files; key: `dataset/*.jsonl`, `src/*.py`, `src/*.sh`.  
- **mamo:** ~98 files; JSON, JSONL, Python, shell.  
- **optmath:** ~218 files; key: `benchmark/*.json`, many .py/.md.  
- **dcp_bench_open:** ~369 files; key: `sample_test.jsonl`, many .py/.json.  
- **gurobi_examples:** ~430 files; many .ipynb, READMEs per example.  
- **ortools_examples:** ~4870 files; full repo, `examples/` holds example code.  
- **csplib:** ~2028 files; problem descriptions, MiniZinc, packages.  
- **minizinc_benchmarks:** ~18804 files; many .mzn, .dzn.  

## Manifest Path

- **Machine-readable manifest:** `data_external/manifests/public_data_manifest.json`  

Fields per source: `source_name`, `category`, `public_url`, `local_path`, `retrieval_method`, `status`, `short_description`, `likely_usage`, `license`, `notes`.

## Recommended Next Steps (preprocessing into train/dev/external-test)

1. **Unify schemas:** For each benchmark (NL4Opt, ORQA, Mamo, OptMATH, DCP-Bench-Open), map fields to your pipeline’s problem/query schema (e.g. `description`, `parameters`, `formulation`). Reuse or extend existing collectors (e.g. `collectors/collect_nl4opt.py`, `collectors/collect_optmath.py`) to read from `data_external/raw/...` and write to a shared `data/processed/` or `data_external/processed/` layout.
2. **Splits:** Define train/dev/external-test splits: e.g. NL4Opt train/dev for training, NL4Opt test + ORQA + DCP-Bench-Open (and optionally Mamo/OptMATH holdouts) for external eval. Avoid leakage (same problem ID never in train and test).
3. **Mining from examples:** For Gurobi and OR-Tools, add a small pipeline to extract (NL snippet, formulation/code) from notebooks and READMEs; optionally produce weak labels or additional training pairs.
4. **CSPLib / MiniZinc:** Parse .mzn/.dzn to get problem structure; use for constraint-modeling eval or as auxiliary mining.
5. **Versioning:** Keep raw data read-only; write all derived assets (JSONL, splits, mined pairs) under versioned paths or with clear provenance in the manifest so results are reproducible.
