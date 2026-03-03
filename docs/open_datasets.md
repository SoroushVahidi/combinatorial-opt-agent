# Open Datasets Used for Training

The retrieval model is trained on problems from the following **open datasets**. Each has natural-language descriptions; many include formulations (LP/IP), complexity, and aliases.

## 1. NL4Opt (already in use)

- **Source:** [nl4opt/nl4opt-competition](https://github.com/nl4opt/nl4opt-competition) — generation_data (train/dev/test JSONL).
- **Content:** Natural language descriptions of LP problems with structured formulations (variables, objective, constraints).
- **Collected by:** `collectors/collect_nl4opt.py` → `data/raw/nl4opt/*.json`.
- **Size:** ~1,101 problems (train + dev + test).

## 2. OptMATH Benchmark (added)

- **Source:** [optsuite/OptMATH](https://github.com/optsuite/OptMATH) — `benchmark/OptMATH_Bench.json`.
- **Content:** Long natural-language descriptions of optimization problems (job shop, aircraft landing, TSP, facility location, supply chain, etc.). No structured formulation in the benchmark file; ideal for teaching the model to recognize problem *types* from description.
- **Collected by:** `collectors/collect_optmath.py` → `data/raw/optmath/bench.json`.
- **Size:** 166 problems (after filtering very short items).

## 3. Classic + Classic Extra (added)

- **Classic (existing):** 32 well-known problems (knapsack, set cover, vertex cover, TSP, facility location, bin packing, max cut, etc.) with full formulations and complexity (e.g. NP-hard). Stored in the initial catalog.
- **Classic Extra:** `data/raw/classic_extra.json` — 7 additional problems with descriptions, LP/IP formulations, and complexity:
  - Job Shop Scheduling
  - Aircraft Landing
  - Maximum Independent Set
  - Set Packing
  - Graph Coloring
  - Shortest Path (LP)
  - Capacitated Facility Location

## 4. Other Public Sources (for reference)

- **MIPLIB** (miplib.zib.de): Benchmark *instances* (MPS/LP files) for solver comparison; no natural language. Not used for retrieval training.
- **OR-Library** (people.brunel.ac.uk/~mastjjb/jeb/info.html): Test instances for many OR problems; data files rather than NL descriptions. Not ingested here.
- **ML4MILP** (GitHub): MILP instances for ML solvers; problem *types* (MIS, MVC, Set Cover, etc.) align with our classic set.

## Pipeline

1. **Collect:**  
   `python -m pipeline.run_collection`  
   Runs NL4Opt + OptMATH collectors and merges everything (including classic_extra) into `data/processed/all_problems.json`.

2. **Merge only:**  
   `python -m scripts.merge_catalog`  
   Re-merges existing raw files without re-downloading.

3. **Train:**  
   `sbatch scripts/train_retrieval_gpu.slurm`  
   Generates up to 100 (query, passage) pairs per problem and fine-tunes the retrieval model so it learns to recognize all problems (descriptions, formulations, complexities) from the combined catalog.

## Current catalog size

After the latest collection: **1,302 problems** (1,133 existing + 169 new from OptMATH and classic_extra).
