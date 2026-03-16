# Real/Public Optimization Examples — Collection Report

*Generated: 2026-03-16 13:24 UTC*

## Overview

- **Total examples collected:** 1460
- **Sources considered:** 16
- **Sources with data collected:** 4

## Examples by Source

| Source | Count |
|---|---|
| NL4Opt Competition | 1101 |
| OptMATH Benchmark | 166 |
| ORQA (OR Question Answering) | 143 |
| Gurobi Modeling Examples | 50 |

## Examples by Optimization Family

| Family | Count |
|---|---|
| general_lp | 406 |
| production_planning | 232 |
| retail_sales | 165 |
| general_or | 143 |
| transportation | 95 |
| diet_nutrition | 66 |
| portfolio_finance | 64 |
| general_lp_milp | 62 |
| workforce | 40 |
| job_shop_scheduling | 38 |
| agricultural | 32 |
| scheduling | 20 |
| assignment | 12 |
| healthcare | 12 |
| vehicle_routing | 9 |
| set_cover | 9 |
| travel_tourism | 9 |
| education | 8 |
| traveling_salesman | 7 |
| facility_location | 6 |
| general_milp | 6 |
| energy_power | 4 |
| workforce_scheduling | 3 |
| supply_chain | 2 |
| portfolio | 2 |
| resource_extraction | 2 |
| blending_diet | 2 |
| economic_planning | 1 |
| revenue_management | 1 |
| fleet_management | 1 |
| cutting_stock | 1 |

## Sources Considered

| Source | Outcome | Notes |
|---|---|---|
| OptMATH Benchmark | `collected` | 166 NL optimization problem descriptions with numeric parameters; diverse famili |
| NL4Opt Competition | `collected` | NL + structured LP formulations. train/dev/test splits. Full dataset collected ( |
| ORQA (OR Question Answering) | `collected` | OR question-answering dataset. All unique problem contexts collected (one entry  |
| Gurobi Modeling Examples | `collected` | README NL descriptions from all available examples. No numeric parameters. Descr |
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
