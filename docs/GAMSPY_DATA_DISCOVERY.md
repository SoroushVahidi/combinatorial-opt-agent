# GAMSPy Data Discovery

## What example/model data was found

1. **GAMSPy package itself**  
   - No `.gms` or sample data files are shipped inside the `gamspy` or `gamspy_base` pip packages.  
   - The package provides the API (Container, Set, Parameter, Variable, Equation, Model, etc.) and the runtime (via `gamspy_base`).

2. **GAMSPy examples repository (cloned into the project)**  
   - **Source:** https://github.com/GAMS-dev/gamspy-examples  
   - **Local path:** `data_private/gams_models/raw/gamspy-examples/`  
   - **Contents:**  
     - **models/** — 124 GAMSPy models as Python scripts (e.g. `trnsport/trnsport.py`, `blend/blend.py`, `nurses/nurses.py`, `clsp/clsp.py`, `tsp/tsp.py`, plus many LP/MIP/NLP and application-specific models).  
     - **notebooks/** — 13 Jupyter notebooks (transport, blend, nurses, clsp, millco, mpc, ms_cflp, pickstock, radar_placement, recipes, rowing_optimization, disneyland_itinerary, paintshop).  
   - **Manifest:** `data_private/gams_models/manifests/gams_examples_manifest.csv`

3. **GAMS Model Library (external)**  
   - The official GAMS Model Library (400+ models in `.gms`) is **not** bundled with GAMSPy.  
   - It is available from: https://www.gams.com/latest/gamslib_ml/libhtml/  
   - Access is typically via the GAMS IDE or manual download; GAMSPy can work with `.gms` by translating or running via the engine, but the library itself was not collected here.

## Where it lives

| Asset | Path |
|-------|------|
| GAMSPy examples (full repo) | `data_private/gams_models/raw/gamspy-examples/` |
| Model scripts | `data_private/gams_models/raw/gamspy-examples/models/<name>/` |
| Notebooks | `data_private/gams_models/raw/gamspy-examples/notebooks/<name>/` |
| Manifest | `data_private/gams_models/manifests/gams_examples_manifest.csv` |

## What is likely useful for our project

- **For NLP4LP / optimization-problem understanding:**  
  The cloned GAMSPy examples are Python-based optimization models. They can be used to:  
  - Extract problem structure (sets, parameters, variables, equations) for cataloging or matching to natural language.  
  - Run small models (with a valid license) for testing or for generating (NL, formulation) pairs if we add descriptions.  
- **Notebooks** are especially useful: they often contain short textual descriptions plus code (e.g. transport, blend, nurses).  
- **Classic models** (trnsport, blend, nurses, clsp, tsp, knapsack, etc.) align with problem types we already use elsewhere in the repo.

## What still needs manual access or download

- **GAMS Model Library** (`.gms` models): Not included. To use it you would:  
  - Download or browse from https://www.gams.com/latest/gamslib_ml/libhtml/  
  - Place selected `.gms` files in something like `data_private/gams_models/raw/gamslib/` and extend the manifest if desired.  
- **License:** You must run `gamspy install license <access_code_or_path>` once so that larger or restricted models can be solved; see `docs/GAMSPY_SETUP_AND_LICENSE.md`.

## Next recommended step for turning examples into usable data

1. **Parse and catalog:** For each script in `data_private/gams_models/raw/gamspy-examples/models/`, optionally run a small script that imports the module, reads the Container’s sets/parameters/variables/equations (if exposed), or parses the `.py` file to extract symbol names and problem type (LP/MIP/NLP). Write results to something like `data_private/gams_models/manifests/models_catalog.json` for use in retrieval or schema matching.  
2. **Add NL descriptions:** Where possible, attach short natural-language descriptions (e.g. from notebook markdown or from the model’s docstring) to each model entry so that we can build (NL, formulation) pairs for training or evaluation.  
3. **Optional:** Download a subset of the GAMS Model Library (`.gms`) into `data_private/gams_models/raw/gamslib/` and add a similar catalog or converter (e.g. `.gms` → GAMSPy Container or to a unified schema) if you need more problem diversity.
