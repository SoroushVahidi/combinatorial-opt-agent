# Data sources: full list of libraries and problems

This is the **canonical list** of external libraries and datasets used (or planned) to build the combinatorial optimization problem catalog. Each source can provide problem names, natural language descriptions, and/or integer/linear program formulations.

---

## 1. NL4Opt Competition

| Field | Value |
|--------|--------|
| **URL** | https://github.com/nl4opt/nl4opt-competition |
| **Type** | NL → Formulation |
| **Size** | 1,101 problems (generation_data) |
| **What we extract** | Natural language problem statements + LP formulations |
| **Format** | generation_data/ (CSV or similar); ner_data/ for NER |
| **Status** | ✅ Implemented — `collectors/collect_nl4opt.py`; run `python pipeline/run_collection.py` |

- **Subtask 1**: NER (entity extraction).  
- **Subtask 2**: Generation (NL → formulation).  
- Dataset is the main source of NL–LP pairs for training/retrieval.

---

## 2. NLP4LP / OptiMUS

| Field | Value |
|--------|--------|
| **URL** | https://github.com/teshnizi/OptiMUS |
| **Type** | NL → Formulation |
| **Size** | 269 problems |
| **What we extract** | Natural language + LP/MILP formulations |
| **Status** | Planned |

---

## 3. Gurobi Modeling Examples

| Field | Value |
|--------|--------|
| **URL** | https://github.com/Gurobi/modeling-examples |
| **Type** | Jupyter notebooks |
| **Size** | 50+ example folders |
| **What we extract** | NL description (in notebook), LaTeX/math, Gurobi Python code |

**Full list of example folders (problem names):**

- 3d_tic_tac_toe, agricultural_pricing, aviation_planning, battery_scheduling, burrito_optimization_game, car_rental, cell_tower_coverage, colgen-cutting_stock, constraint_optimization, covid19_facility_location, curve_fitting, customer_assignment, decentralization_planning, drone_network, economic_planning, efficiency_analysis, electrical_power_generation, facility_location, factory_planning, fantasy_basketball, farm_planning, food_manufacturing, food_program, index_tracking, intro_to_gurobipy, linear_regression, logical_design, lost_luggage_distribution, manpower_planning, market_sharing, marketing_campaign_optimization, milk_collection, mining, music_recommendation, offshore_wind_farming, opencast_mining, optimization101, optimization201, optimization202, optimization301, pooling, portfolio_selection_optimization, power_generation, price_optimization, pricing_competing_products, protein_comparison, protein_folding, railway_dispatching, refinery, supply_network_design, technician_routing_scheduling, text_dissimilarity, traveling_salesman, workforce, yield_management

**Status:** ✅ In catalog — problem names from `data/sources/gurobi_modeling_examples.json`; merged via `scripts/merge_catalog.py`.

---

## 4. Gurobi OptiMods

| Field | Value |
|--------|--------|
| **URL** | https://github.com/Gurobi/gurobi-optimods |
| **Type** | Documented Python mods (APIs + docs) |
| **Size** | ~15 mods |
| **What we extract** | Math formulation from docs + Python implementation |

**Full list of mods (module names):**

- bipartite_matching, datasets, line_optimization, max_flow, metromap, min_cost_flow, min_cut, mwis (maximum weight independent set), opf (optimal power flow), portfolio, qubo, regression, sharpe_ratio, workforce

**Status:** ✅ In catalog — from `data/sources/gurobi_optimods.json`; merged via `scripts/merge_catalog.py`.

---

## 5. GAMS Model Library

| Field | Value |
|--------|--------|
| **URL** | https://www.gams.com/latest/gamslib_ml/libhtml/ |
| **Type** | Model catalog |
| **Size** | 400+ models |
| **What we extract** | Categorized GAMS formulations (may need GAMS license to run) |
| **Status** | ✅ In catalog — subset in `data/sources/gams_models.json`; merged via `scripts/merge_catalog.py`. |

---

## 6. MIPLIB 2017

| Field | Value |
|--------|--------|
| **URL** | https://miplib.zib.de/ |
| **Type** | Benchmark instances |
| **Size** | 1,000+ instances |
| **What we extract** | Real-world MILP in MPS format (instance data; problem type often inferred) |
| **Status** | ✅ In catalog — `data/sources/miplib.json` (MIPLIB 2017 entry); merged via `scripts/merge_catalog.py`. |

---

## 7. OR-Library (J.E. Beasley)

| Field | Value |
|--------|--------|
| **URL** | http://people.brunel.ac.uk/~mastjjb/jeb/info.html |
| **Type** | Test data sets for OR problems (problem families + instances) |
| **Size** | 90+ problem types (many with multiple links) |
| **What we extract** | Problem family name, description, instance files (data); formulations are standard from literature. |

**Full list of OR-Library problem families (from main page):**

- Airport capacity  
- Assignment problem  
- Bin packing (two-dimensional, one-dimensional)  
- Capacitated minimal spanning tree  
- Crew scheduling  
- Corporate structuring  
- Data envelopment analysis  
- Edge-weighted k-cardinality tree  
- Equitable partitioning problem  
- Generalised assignment problem  
- Graph colouring  
- Index tracking  
- Knapsack (multidimensional, multi-demand multidimensional)  
- Linear programming  
- Location: uncapacitated warehouse location, p-median (capacitated, uncapacitated), p-hub, capacitated warehouse location  
- Lot sizing  
- Network flow (single commodity, concave costs, single source, uncapacitated)  
- Ore selection  
- Portfolio optimisation (multiple periods, single period)  
- Scheduling: weighted tardiness, shift minimization personnel task, open shop, lot streaming, job shop, hybrid reentrant shop, flow shop, common due date, aircraft landing  
- Set covering  
- Set partitioning  
- Shortest path (resource constrained)  
- Steiner: Steiner problem in graphs, rectilinear Steiner, Euclidean Steiner  
- Three-dimensional cutting/packing: container loading (with weight restrictions), boxes on shelves  
- Time series forecasting  
- Timetabling  
- Travelling salesman problem (period salesman)  
- Two-dimensional cutting/packing: unequal rectangles, unequal circles, unconstrained guillotine, strip packing, constrained non-guillotine, constrained guillotine, assortment problem  
- Unconstrained binary quadratic programming  
- Unit commitment  
- Vehicle routing: two-echelon, sparse feasibility graph, site-dependent multi-trip period routing, single period, period routing, fixed routes, fixed areas  

**Status:** ✅ In catalog — from `data/sources/or_library.json`; merged via `scripts/merge_catalog.py`. Formulations can be added from textbooks/literature.

---

## 8. Pyomo Examples

| Field | Value |
|--------|--------|
| **URL** | https://github.com/Pyomo/pyomo (examples in repo) |
| **Type** | Python optimization models |
| **Size** | 15+ models |
| **What we extract** | Formulation from code + comments |
| **Status** | ✅ In catalog — from `data/sources/pyomo_examples.json`; merged via `scripts/merge_catalog.py`. |

---

## Summary table (same as README, for single reference)

| Source | Type | Size | What we extract |
|--------|------|------|-----------------|
| [NL4Opt](https://github.com/nl4opt/nl4opt-competition) | NL → Formulation | 1,101 | NL + LP formulations |
| [OptiMUS](https://github.com/teshnizi/OptiMUS) | NL → Formulation | 269 | NL + LP/MILP formulations |
| [Gurobi OptiMods](https://github.com/Gurobi/gurobi-optimods) | Documented mods | ~15 | Math + Python code |
| [Gurobi Modeling Examples](https://github.com/Gurobi/modeling-examples) | Notebooks | 50+ | NL + LaTeX + Gurobi code |
| [GAMS Model Library](https://www.gams.com/latest/gamslib_ml/libhtml/) | Catalog | 400+ | GAMS formulations |
| [MIPLIB 2017](https://miplib.zib.de/) | Benchmarks | 1,000+ | MILP instances (MPS) |
| [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/info.html) | Problem families | 90+ | Families + test instances |
| [Pyomo](https://github.com/Pyomo/pyomo) | Examples | 15+ | Python models |

---

*This file is the single place in the project where the full list of libraries and their problem names are written. Manifests in `data/sources/` provide machine-readable lists for collectors.*
