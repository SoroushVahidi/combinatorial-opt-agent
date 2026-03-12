# Real/Public Optimization Examples — Collection Report

*Generated: 2026-03-12 20:40 UTC*

## Overview

- **Total examples collected:** 410
- **Sources considered:** 16
- **Sources with data collected:** 4

## Examples by Source

| Source | Count |
|---|---|
| OptMATH Benchmark | 166 |
| NL4Opt Competition | 120 |
| ORQA (OR Question Answering) | 100 |
| Gurobi Modeling Examples | 24 |

## Examples by Optimization Family

| Family | Count |
|---|---|
| general_or | 100 |
| general_lp_milp | 62 |
| general_lp | 41 |
| job_shop_scheduling | 38 |
| production_planning | 30 |
| transportation | 26 |
| retail_sales | 19 |
| scheduling | 18 |
| portfolio_finance | 14 |
| assignment | 10 |
| vehicle_routing | 9 |
| set_cover | 9 |
| traveling_salesman | 7 |
| facility_location | 3 |
| workforce_scheduling | 3 |
| diet_nutrition | 3 |
| agricultural | 3 |
| energy_power | 2 |
| supply_chain | 2 |
| workforce | 2 |
| travel_tourism | 2 |
| portfolio | 2 |
| resource_extraction | 1 |
| blending_diet | 1 |
| economic_planning | 1 |
| revenue_management | 1 |
| fleet_management | 1 |

## Sources Considered

| Source | Outcome | Notes |
|---|---|---|
| OptMATH Benchmark | `collected` | 166 NL optimization problem descriptions with numeric parameters; diverse famili |
| NL4Opt Competition | `collected` | NL + structured LP formulations. train/dev/test splits. Representative sample (4 |
| ORQA (OR Question Answering) | `collected` | OR question-answering dataset with scenario context + typed questions. Sampled 5 |
| Gurobi Modeling Examples | `collected` | README NL descriptions only; no numeric parameters. Full notebooks require Gurob |
| DCP-Bench-Open (sample_test.jsonl) | `accessible_not_useful` | sample_test.jsonl only contains 5 items that are solver code (CPMPy models), not |
| MAMO (FreedomIntelligence/Mamo) | `inaccessible` | HTTP 404 for expected data path benchmark/MathBench.json. Repository structure m |
| NEOS Guide Case Studies | `inaccessible` | DNS resolution failed for neos-guide.org in this environment (no address associa |
| MIT OpenCourseWare (15.053 Optimization Methods) | `inaccessible` | DNS resolution failed for ocw.mit.edu in sandboxed environment. Problem sets and |
| GAMS Model Library (web catalog) | `inaccessible` | DNS resolution failed for www.gams.com in sandboxed environment. Web catalog is  |
| OR-Library (J.E. Beasley) | `inaccessible` | DNS resolution failed in sandboxed environment. Provides test instance data file |
| LibreTexts Mathematics (LP/OR chapters) | `inaccessible` | DNS resolution failed in sandboxed environment. Open textbook platform with LP/O |
| AMPL Book Examples | `inaccessible` | DNS resolution failed in sandboxed environment. Book is freely viewable online ( |
| Pyomo Documentation Examples | `accessible_not_useful` | Pyomo docs provide solver code, not NL problem statements. Metadata already in d |
| CSPLib (Constraint Satisfaction Problem Library) | `inaccessible` | DNS resolution failed in sandboxed environment. Provides CP problem descriptions |
| MiniZinc Challenge Benchmark | `accessible_not_useful` | Contains .mzn/.dzn constraint models without natural-language problem descriptio |
| MIPLIB 2017 | `inaccessible` | DNS resolution failed in sandboxed environment. Provides MPS/LP instance files,  |

## Data File Locations

- `data/external_real_examples/collected_examples.jsonl` — main dataset (one JSON object per line)
- `data/external_real_examples/source_manifest.csv` — all sources with outcome
- `data/external_real_examples/failed_sources_report.md` — inaccessible/skipped source details
- `data/external_real_examples/collection_report.md` — this file

## How to Re-run

```bash
python tools/collect_real_optimization_examples.py
```

The script fetches from public GitHub raw URLs and Gurobi public READMEs.
No API keys or credentials required.

## License Notes

| Source | License |
|---|---|
| OptMATH Benchmark | public (research use; see repo LICENSE) |
| NL4Opt Competition | MIT |
| ORQA (OR Question Answering) | public (research use; see repo) |
| Gurobi Modeling Examples | Apache-2.0 |
