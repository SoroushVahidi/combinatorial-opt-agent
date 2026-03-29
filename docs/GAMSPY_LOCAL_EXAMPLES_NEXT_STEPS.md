# GAMSPy Local Examples — Next Steps

This document outlines which subset of the collected examples to process first and how they can support acceptance-aware reranking, hierarchy/family classification, optimization-role extraction, and train/dev/external-test creation.

## Subset to process first

1. **Copied examples** (`data_private/gams_models/examples/`): trnsport, blend, SimpleLP, SimpleMIP, knapsack.  
   - Use for: schema extraction, parameter/slot naming, and as a minimal dev set for acceptance/reranking.

2. **Small models with clear docstrings** (from manifest: size &lt; ~5 KB and known family):  
   BoundaryLP, hansmge, surface, reshop, sat, thai, prodmix, EmergencyCentreAllocation, PMU, control3, etc.  
   - Use for: expanding the schema/parameter catalog and refining family labels.

3. **Classic named models** (already mapped to families):  
   nurses, clsp, tsp, tsp4, flowshop, rcpsp, whouse, cutstock, cpack, ms_cflp, food, millco.  
   - Use for: family classification evaluation and external test set design.

4. **Notebooks** (cataloged only): Start with small ones (blend, transport) for NL/schema pairs; then nurses, clsp, pickstock, paintshop for diversity.

## How these examples support project goals

### Acceptance-aware reranking

- **Input**: Natural-language query + candidate schemas (e.g. from retrieval).
- **Use**: Run instantiation/assignment on small models (trnsport, blend, knapsack) with synthetic or real queries; measure “instantiation_ready” or coverage. Use the five copied examples as a quick benchmark; add nurses/clsp when data files are available.
- **Data**: Parameter names and types from model headers + docstrings; NL descriptions from docstrings and notebooks.

### Hierarchy / family classification

- **Input**: Model identifier or code snippet.
- **Use**: Current manifest has a heuristic `likely_family` for 73/138 entries; 65 are “unknown”. Next steps:
  - Parse `## MODELTYPE` and `## KEYWORDS` from docstrings to refine or auto-assign family.
  - Use known-family examples as seed labels; train or rule-based classifier for the rest.
- **Splits**: Use “unknown” and a held-out set of known families for evaluation.

### Optimization-role extraction

- **Input**: GAMSPy script or notebook.
- **Use**: Extract sets, parameters, variables, equations (names, domains, descriptions). Build a schema/parameter catalog in `data_private/gams_models/catalog/`. Prioritize models with rich docstrings (e.g. trnsport, blend, nurses, knapsack).
- **Output**: Structured records (e.g. JSON) per model: slots, types, and short descriptions for NL→slot alignment and weak supervision.

### Train / dev / external-test creation

- **Train**: Models with NL descriptions in docstrings or notebooks; use for NL→schema or query→parameter alignment (weak supervision or synthetic pairs).
- **Dev**: The five copied examples + a few more small models (e.g. BoundaryLP, sat) for tuning and ablation.
- **External test**: Hold out classic problems (e.g. one per family: tsp, nurses, clsp, ms_cflp, cutstock) or use notebooks not used in training for a blind evaluation set.

## Exact next step after this inventory

1. **Run a single smoke test** (if not already done): Execute one small example from `data_private/gams_models/examples/` (e.g. `trnsport.py`) and record command, success/failure, and log path.
2. **Extract metadata from the five copied examples**: Parse docstring (GAMSSOURCE, MODELTYPE, KEYWORDS) and first-level GAMSPy objects (Set, Parameter, Variable, Equation) into a minimal catalog (e.g. one JSON per model in `data_private/gams_models/catalog/`).
3. **Refine family labels**: For a sample of “unknown” models, inspect docstrings and add mappings in `tools/collect_gams_examples_manifest.py`; re-run the script to refresh the manifest and inventory.
4. **Add one notebook-based NL/schema pair**: Pick the transport or blend notebook; extract one or more “problem description → schema” pairs and store in a structured format for downstream use.

After that, iterate on acceptance reranking and role extraction using the catalog and manifest.
