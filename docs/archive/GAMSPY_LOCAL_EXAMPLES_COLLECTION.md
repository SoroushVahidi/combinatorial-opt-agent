# GAMSPy Local Examples Collection

This document describes where local GAMSPy/GAMS example and model data were found, what was copied vs only cataloged, counts by extension and family, and what is most useful for the optimization-problem understanding project.

## Where examples were found

- **GAMSPy package** (`~/.local/lib/python3.11/site-packages/gamspy/`): No bundled examples, `.gms` files, or data; only library code (formulations, backend, etc.).
- **GAMSPy share dir** (`~/.local/share/GAMSPy/`): License file only (`gamspy_license.txt`); no examples or model library.
- **Project raw clone**: All discoverable examples come from the cloned repository:
  - **`data_private/gams_models/raw/gamspy-examples`**  
    Source: https://github.com/GAMS-dev/gamspy-examples  
  - **`models/`**: 124 GAMSPy Python scripts (`.py`), one per model (e.g. trnsport, blend, nurses, clsp, tsp, knapsack, and many LP/MIP/NLP models).
  - **`notebooks/`**: 14 Jupyter notebooks (`.ipynb`) in 13 directories: transport, blend, nurses, clsp, millco, mpc, ms_cflp, pickstock, radar_placement, recipes, rowing_optimization, disneyland_itinerary, paintshop (plus one data-prep notebook under disneyland_itinerary/Data).

No separate GAMS model library (e.g. `.gms` files) or HTML docs were found in the inspected locations.

## What was copied vs only cataloged

- **Copied** into `data_private/gams_models/examples/`: Five small, representative model scripts (each &lt; 3.5 KB):
  - `trnsport.py` — transportation LP
  - `blend.py` — blending LP
  - `SimpleLP.py` — generic LP
  - `SimpleMIP.py` — generic MIP
  - `knapsack.py` — binary knapsack MIP  

  These are directly useful for schema acceptance, family classification, and a quick smoke test.

- **Cataloged only** (original path recorded in manifest; not copied):
  - Remaining 119 model `.py` files in `raw/gamspy-examples/models/`
  - All 14 notebooks in `raw/gamspy-examples/notebooks/`  
  Reasons: avoid duplication, keep repo light; many models are larger or depend on data files (e.g. nurses, clsp). The manifest points to the single source of truth under `raw/`.

## Counts by extension

| Extension | Count | Location |
|-----------|-------|----------|
| `.py`     | 124   | `raw/gamspy-examples/models/<name>/<name>.py` |
| `.ipynb`  | 14    | `raw/gamspy-examples/notebooks/<dir>/` |
| **Total** | **138** | |

(No `.gms` files were found in the local installation or the cloned examples repo.)

## Counts by likely family

Family is assigned heuristically from folder/base name (see `tools/collect_gams_examples_manifest.py` and `GAMSPY_LOCAL_EXAMPLES_NEXT_STEPS.md`). Summary:

| Family | Count |
|--------|-------|
| unknown | 65 |
| allocation_budgeting_investment | 14 |
| network_energy | 14 |
| scheduling_assignment | 9 |
| covering_facility_location | 7 |
| transportation_flow | 6 |
| production_blending | 6 |
| inventory_supply_capacity | 4 |
| generic_lp | 4 |
| packing_knapsack | 3 |
| network_graph_path | 3 |
| generic_mip | 1 |
| generic_qp | 1 |
| generic_control | 1 |

“Unknown” is used when the folder name is not in the heuristic map; those can be refined later from docstrings or keywords.

## What seems most useful for our project

- **Schema acceptance / reranking**: Small, self-contained models with clear parameter names and docstrings (e.g. trnsport, blend, SimpleLP, SimpleMIP, knapsack). The five copied examples are a good seed set.
- **Problem family grouping**: Models with standard names (transportation, blend, nurses, tsp, knapsack, clsp, etc.) already mapped in the manifest; the 65 “unknown” entries are candidates for docstring/keyword-based refinement.
- **Optimization-role extraction**: All 124 `.py` models expose sets, parameters, variables, and equations; header comments often include MODELTYPE (LP/MIP/NLP) and KEYWORDS. Useful for building schema/parameter catalogs.
- **External evaluation**: Classic, well-defined problems (transportation, blending, knapsack, nurses, TSP, facility location) are good for held-out or benchmark evaluation.
- **Weak supervision / training data**: Docstrings often contain short natural-language problem descriptions (e.g. trnsport, blend, knapsack, nurses). Notebooks add narrative and can be mined for NL→schema pairs; smaller notebooks (blend, transport) are more manageable.

## Manifests and inventory

- **`data_private/gams_models/manifests/gams_examples_manifest.csv`**  
  Per-file rows with: name, original_path, copied_path, extension, size_bytes, source_group, likely_family, likely_usage, notes.

- **`data_private/gams_models/manifests/gams_examples_inventory.json`**  
  Machine-readable inventory: same entries plus summary counts by extension and by family.

Regenerate with:

```bash
python tools/collect_gams_examples_manifest.py
```

## Project directories

- `data_private/gams_models/raw/` — clone of gamspy-examples (do not modify originals).
- `data_private/gams_models/examples/` — copied small examples (five scripts).
- `data_private/gams_models/catalog/` — reserved for future extracted metadata (e.g. schema/parameter catalogs).
- `data_private/gams_models/manifests/` — manifest CSV and inventory JSON.
