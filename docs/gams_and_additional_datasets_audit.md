# GAMS and Additional Datasets Audit

**Date:** 2026-03-09  
**Purpose:** Audit additional dataset opportunities for natural-language optimization problem instantiation, with focus on GAMS example libraries as a new data source, and build the first practical bridge from GAMS examples into the learning pipeline.  
**Bottleneck in scope:** Downstream number-to-slot grounding (especially float/role semantics). Retrieval is already strong; this work does not center on schema retrieval.

---

## 1. GAMS environment findings

### 1.1 Installation and license

| Item | Finding |
|------|--------|
| **GAMS binary** | Not found in PATH or under `/usr`, `/opt`, or user home. The system uses **GAMSPy** (Python API) rather than the standalone GAMS engine. |
| **GAMSPy** | Installed and importable: `~/.local/lib/python3.11/site-packages/gamspy`. |
| **License** | Active. Path: `~/.local/share/GAMSPy/gamspy_license.txt`. Type: **GAMSPy license**, **Academic**. Expiration: **2027-03-02**. |
| **License check** | `python -m gamspy show license` succeeds; lists solvers (CBC, CONOPT, CPLEX, GUROBI, etc.). |

### 1.2 Example and model library paths

| Location | Path | Contents |
|----------|------|----------|
| **GAMSPy examples (local)** | `data_private/gams_models/raw/gamspy-examples/` | Clone of https://github.com/GAMS-dev/gamspy-examples |
| **Model scripts** | `data_private/gams_models/raw/gamspy-examples/models/` | **124** Python (`.py`) files, one per model (e.g. trnsport, blend, knapsack, nurses, DED, CVaR). |
| **Notebooks** | `data_private/gams_models/raw/gamspy-examples/notebooks/` | 14 Jupyter notebooks (not processed in this prototype). |
| **Copied subset** | `data_private/gams_models/examples/` | Five small scripts: trnsport, blend, SimpleLP, SimpleMIP, knapsack. |
| **Manifests** | `data_private/gams_models/manifests/` | `gams_examples_manifest.csv`, `gams_examples_inventory.json` (from `tools/collect_gams_examples_manifest.py`). |
| **GAMS Model Library (.gms)** | Not local | Official 400+ model library is at https://www.gams.com/latest/gamslib_ml/libhtml/; not bundled with GAMSPy. Catalog names in `data/sources/gams_models.json`. |

### 1.3 Direct readability

- All 124 `.py` model files are **directly readable** from the filesystem under `data_private/gams_models/raw/gamspy-examples/models/`.
- No `.gms` files were found in the inspected GAMSPy install or the cloned examples repo.
- Helper scripts in repo: `tools/collect_gams_examples_manifest.py` (manifest + inventory); **new:** `tools/extract_gams_examples_structured.py` (structured extraction for audit and pipeline bridge).

---

## 2. GAMS examples as data — audit

### 2.1 Counts

- **124** GAMSPy example models (`.py`) processed in this audit.
- **14** notebooks not processed in the prototype (optional next step).

### 2.2 Optimization families (from extracted MODELTYPE)

- **LP:** 24  
- **NLP:** 35  
- **MIP:** 23  
- **QCP:** 13  
- **MINLP:** 6  
- **MCP:** 3  
- **Other/combined:** (LP,MIP), (LP,NLP), (MIP,NLP), (NLP,MIP,MPEC), DNLP, CNS, EMP, etc.  
- **unknown:** 1  

So we have a mix of LP, MIP, NLP, QCP, MINLP, and a few MCP/other.

### 2.3 Extractable structured information (per example)

From each `.py` file we can extract (without executing code):

| Information | How extracted | Use for bottleneck |
|-------------|----------------|-------------------|
| **Parameter names** | Regex on `Parameter(m, name="...", ...)` and short form | Slot vocabulary; slot–mention pairing |
| **Parameter descriptions** | `description="..."` in Parameter() | Slot semantics; alignment with NL |
| **Set names/descriptions** | Same for `Set(...)` | Schema context |
| **Variable/Equation names** | Same for `Variable(...)`, `Equation(...)` | Role/type hints |
| **Objective direction** | `sense=Sense.MIN` / `Sense.MAX` | Objective-slot role |
| **Scalar constants** | Heuristic: numbers in `records=`, `np.array([...])`, literal lists | Candidate numeric mentions |
| **Model type** | Docstring `## MODELTYPE: LP|MIP|NLP|...` | Family / schema classification |
| **Keywords / GAMSSOURCE** | Docstring `## KEYWORDS`, `## GAMSSOURCE` | Retrieval and grouping |

### 2.4 Alignment with NLP4LP-style slot filling

- **Aligned:** Models with **named scalar or indexed parameters** that correspond to “slots” (capacity, demand, price, cost, bound, etc.) and **numeric literals** in the same file that could fill those slots. Examples: trnsport (a, b, d, c; capacities 350, 600; demands 325, 300, 275), blend (price, rb, compdat; percentages and prices), knapsack (p, w, c; capacity and item data).
- **Partially aligned:** Models with many parameters and arrays; slot–value pairing is heuristic (same-file constants vs parameter names).
- **Less aligned:** Abstract or equation-heavy models with few scalar parameters, or models where “mentions” are only in external data files (e.g. knapsack loading from file). We can still use them for **slot vocabulary and type/role** (e.g. “capacity”, “demand”, “min”, “max”).

### 2.5 Realistic assessment for the current bottleneck

- **GAMS examples can help** the number-to-slot grounding bottleneck in **limited but concrete** ways:
  1. **Slot/parameter vocabulary and role:** Parameter names and descriptions (e.g. “capacity of plant i”, “demand at market j”) are direct slot semantics; objective sense gives min/max role.
  2. **Weakly supervised mention–slot pairs:** Within a file, parameter names + numeric literals can form (slot, value) pairs. These are **not** gold NL→slot labels; they are “this number appears in the context of this parameter” and can be used for pretraining or auxiliary data.
  3. **Schema diversity:** 124 models across LP/MIP/NLP/QCP increase variety of parameter names and numeric types (capacities, costs, bounds, percentages).
- **Limitations:** No natural-language problem text in the GAMSPy scripts; only code and docstrings. So GAMS does **not** replace NLP4LP/NL4Opt for NL→slot supervision. It can **augment** slot semantics and provide synthetic (code-context) mention–slot pairs.

---

## 3. Additional datasets beyond the four we already know

Known and already integrated or accessible: **NLP4LP**, **NL4Opt**, **Text2Zinc**, **ORQA**.

From repo manifest (`data_external/manifests/public_data_manifest.json`) and docs:

| Dataset/source | Category | Use for bottleneck | Direct usability | Notes |
|----------------|----------|--------------------|------------------|--------|
| **Mamo** | benchmark | NL + formulations; possible mining for constraints/objectives | B – light integration | Downloaded; JSON/JSONL; mathematical modeling with solvers. |
| **OptMATH** | benchmark | Long NL descriptions (job shop, TSP, facility location); OptMATH_Bench.json | B – light integration | In repo; benchmark/OptMATH_Bench.json; already used via collectors. |
| **DCP-Bench-Open** | benchmark | Constraint modeling (CP/IP/SAT); discrete combinatorial | B – light integration | sample_test.jsonl; more about constraint structure than numeric slot filling. |
| **Gurobi modeling-examples** | mining | Notebooks with NL + math + code; 50+ examples | B – light integration | One folder per example; extract NL + formulation. |
| **OR-Tools examples** | mining | LP, CP, routing, scheduling | B – light integration | Large repo; examples under examples/. |
| **CSPLib** | mining | 97 problems; MiniZinc + descriptions | B – light integration | Problem descriptions + models; CP-style. |
| **MiniZinc benchmarks** | mining | .mzn/.dzn; many instances | B – light integration | Structure and instances; no NL. |
| **TAT-QA** | benchmark | Numerical reasoning over tables/text | A – already in corpus | In build_common_grounding_corpus (tatqa). |
| **FinQA** | benchmark | Numerical reasoning over financial reports | A – already in corpus | In build_common_grounding_corpus (finqa). |
| **GAMS Model Library** | mining | 400+ .gms models | C – not directly accessible | Web catalog only; manual/crawl; no local .gms in this audit. |

**Ranked by usefulness for number-to-slot grounding (bottleneck):**

1. **NLP4LP** (already integrated) — direct NL problem + gold slot assignments.  
2. **NL4Opt** (already integrated) — NL→LP; parameters/limits/objectives.  
3. **GAMSPy examples** (now with extraction) — slot vocabulary + heuristic (slot, number) pairs from code.  
4. **OptMATH / Mamo** — long NL descriptions; can be mined for parameter/constraint mentions.  
5. **Gurobi/OR-Tools examples** — NL in notebooks + code; similar to GAMS use case.  
6. **TAT-QA / FinQA** — numerical QA; already in corpus; useful for numeric reasoning, less for optimization-slot semantics.  
7. **DCP-Bench / CSPLib / MiniZinc** — more constraint/structure than scalar slot filling.

---

## 4. Best use of GAMS examples

Evaluated options:

| Use case | Feasibility | Engineering | Attacks bottleneck? | Fits current pipeline? |
|---------|-------------|------------|--------------------|------------------------|
| Schema classification | High | Low | Indirect (family/type only) | Yes, as auxiliary signal |
| Component/role tagging | High | Medium | Yes (parameter = slot, role from sense/name) | Yes, if we emit slot-like records |
| Synthetic mention–slot pairs | High | Medium | **Yes** — pair parameter names with numeric literals in same file | Yes, can feed pairwise ranker or auxiliary data |
| Synthetic NL descriptions | Medium | High | Docstrings are short; not full “problem text” | Partial; docstring ≈ short description |
| Solver-grounded consistency | Low (needs run) | High | Indirect | Not in current pipeline |

**Best first use of GAMS examples for this repo:**  
**Synthetic mention–slot pair data (weak supervision)** from the same file: treat each (parameter_name, numeric_literal) co-occurrence as a candidate (slot, mention) pair, with slot role inferred from parameter name/description and objective sense (min/max). This directly supports number-to-slot grounding vocabulary and auxiliary training, with minimal engineering (we already have parameter names and numeric constants in the structured extract). It fits the existing pairwise ranker pipeline if we add a converter from GAMS-structured JSONL to ranker `(slot, mention, label)` rows (e.g. label = 1 if the number is the parameter’s value in the model, else 0 — heuristic from same-file co-occurrence or from running the model later).

---

## 5. Prototype artifact created

- **Script:** `tools/extract_gams_examples_structured.py`
- **Output dir:** `artifacts/gams_example_audit/`
- **Files:**
  - `artifacts/gams_example_audit/gams_examples_structured.jsonl` — one JSON object per model (124 records).
  - `artifacts/gams_example_audit/gams_examples_structured.csv` — same info in tabular form.

**Fields per example (summary):**

- `model_name`, `source_path`  
- `model_type` (LP, MIP, NLP, QCP, etc.)  
- `objective_direction` (MIN/MAX when inferable)  
- `keywords`, `gams_source_url`, `description_snippet`  
- `parameter_names`, `parameters` (name + description)  
- `set_names`, `sets`  
- `variable_names`, `variables`  
- `equation_names`, `equations`  
- `numeric_constants_sample` (heuristic list of numbers in the file)  
- `num_parameters`, `num_sets`, `num_variables`, `num_equations`  

**Caveats:** Extraction is heuristic (regex-based; no execution). Descriptions and MODELTYPE come from docstrings. Numeric constants are not gold “slot fill” values; they are candidates. Labels are **not** gold.

---

## 6. Map to current learning pipeline

1. **Can GAMS-derived artifacts be used directly?**  
   Not as-is. The common learning corpus expects `dataset`, `split`, `instance_id`, `problem_text`, `slots`, `numeric_mentions`, `gold_slot_assignments`. GAMS extract has no `problem_text` (only code + docstring) and no gold assignments.

2. **Minimal transformation needed**  
   - Add a dataset name (e.g. `"gams"`) to the corpus schema if we want GAMS in the same corpus, or keep a separate GAMS-derived corpus.  
   - Build “synthetic” corpus records: e.g. one record per model with `problem_text` = description_snippet (or concatenation of parameter descriptions), `slots` = list of slot-like dicts from parameters (slot_id = param name, slot_name/description, role from min/max if inferable), `numeric_mentions` = list of dicts from `numeric_constants_sample` (surface = str(number), normalized_value = number, type_bucket = int/float/percent/capacity etc. heuristic).  
   - **Gold:** We do not have true gold slot assignments. Options: (a) treat all (slot, number) pairs in the same file as positive (weak), (b) leave gold empty and use only for slot vocabulary/pretraining, (c) run the model and compare solution values to constants (solver-grounded; heavier).

3. **Mention–slot pair training rows**  
   Yes. From each GAMS record we can emit rows: for each parameter (slot) and each numeric constant (mention) in that file, one row with slot_name, mention_surface, and a heuristic label (e.g. 1 if this number is used in the parameter’s records/definition, else 0). That yields auxiliary pairwise ranker data.

4. **Auxiliary pretraining before NLP4LP fine-tuning**  
   Yes. Train the pairwise ranker on GAMS-derived pairs (optionally with lower weight or fewer steps), then fine-tune on NLP4LP. Risk: distribution shift (code/symbolic vs NL). Start with a small GAMS subset and one seed to avoid destabilizing.

5. **Minimum viable experiment**  
   - **Option A:** Add a converter `gams_structured_to_ranker_data.py` that reads `gams_examples_structured.jsonl`, produces (slot, mention, label) rows for a subset of models (e.g. trnsport, blend, knapsack, 10 others with scalar params), write to `artifacts/learning_ranker_data/gams_aux/` as `train.jsonl` (no dev/test split needed for aux). Train pairwise ranker on NLP4LP only (baseline) vs NLP4LP + GAMS aux (e.g. 50 steps GAMS then 200 steps NLP4LP) and compare held-out test on NLP4LP.  
   - **Option B:** Use GAMS extraction only for **vocabulary and slot-type lists** (e.g. list of parameter names and descriptions) to improve slot embedding or filtering, without adding GAMS training rows yet.

---

## 7. Recommended next experiment

**Recommended next action:**  
Implement **Option A** in minimal form: add a script that converts `artifacts/gams_example_audit/gams_examples_structured.jsonl` into pairwise ranker format (one row per (slot, mention) with heuristic label), using a **small subset** (e.g. 15–20 models with clear scalar parameters: trnsport, blend, knapsack, SimpleLP, nurses, DED, CVaR, etc.). Write to `artifacts/learning_ranker_data/gams_aux/train.jsonl`. Run **one** experiment: train NLP4LP ranker for 200 steps with seed 42 (baseline) vs. 50 steps on GAMS aux then 200 steps on NLP4LP (same seed). Evaluate on the existing NLP4LP held-out test. Compare test metrics (pairwise accuracy, exact slot-fill). If GAMS aux helps or is neutral, document and consider expanding the GAMS subset or refining labels.

Do **not** claim benchmark validity for GAMS-derived labels; treat as weak auxiliary data only.

---

## 8. References

- GAMSPy: https://github.com/GAMS-dev/gamspy  
- GAMSPy examples: https://github.com/GAMS-dev/gamspy-examples  
- GAMS Model Library: https://www.gams.com/latest/gamslib_ml/libhtml/  
- Repo docs: `docs/GAMSPY_LOCAL_EXAMPLES_COLLECTION.md`, `docs/GAMSPY_DATA_DISCOVERY.md`, `docs/GAMSPY_SETUP_AND_LICENSE.md`  
- Public data manifest: `data_external/manifests/public_data_manifest.json`
