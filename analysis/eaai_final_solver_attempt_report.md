# EAAI final solver-backed executable attempt (last-attempt run)

## A) Solver-backed path found

**Found and executed.** We bypassed the `gurobipy` import blocker by running `optimus_code` through a lightweight in-repo compatibility shim that maps the used Gurobi modeling subset (`Model/addVar/addVars/addConstr/addConstrs/quicksum/setObjective`) to SciPy HiGHS MILP (`scipy.optimize.milp`).

This produced **real nonzero executable/solver outcomes** on a deterministic subset.

## B) Exact commands run

```bash
# 1) repository scan for solver paths and blockers
rg -n "gurobipy|pyomo|pulp|cbc|glpk|gamspy|optimus_code|executable" tools results analysis formulation -S

# 2) dependency availability check
python - <<'PY'
import importlib.util as u
mods=['pyomo','pulp','scipy','cvxpy','ortools','gurobipy']
for m in mods:
    print(m, bool(u.find_spec(m)))
PY

# 3) scipy MILP capability check
python - <<'PY'
import scipy
from scipy import optimize
print('scipy',scipy.__version__)
print('has milp',hasattr(optimize,'milp'))
PY

# 4) final attempt run (TF-IDF + Oracle, same grounding method)
python tools/run_eaai_final_solver_attempt.py \
  --variant orig \
  --limit 20 \
  --out-dir results/paper/eaai_final_solver_attempt

# 5) inspect output summary
cat results/paper/eaai_final_solver_attempt/final_solver_attempt_summary.csv
```

## C) Exact subset rule

Subset rule used in metadata:

- Start from `data/processed/nlp4lp_eval_orig.jsonl`
- Keep rows whose **gold** schema has non-empty `optimus_code`
- Keep only cases where `optimus_code` passes a static compatibility filter for the shim (no `addGenConstr`, `setPWLObj`, `max_(`, `min_(`, `abs_(`, `gp.`, `**`)
- Apply deterministic cap `--limit 20`

## D) Results table

From `results/paper/eaai_final_solver_attempt/final_solver_attempt_summary.csv`:

| Baseline | Subset size | Schema hit | Executable model | Solver-run success | Feasible solution | Objective produced |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 20 | 0.90 | 0.95 | 0.80 | 0.80 | 0.80 |
| Oracle schema | 20 | 1.00 | 0.95 | 0.75 | 0.75 | 0.75 |

Interpretation:
- **Success condition met** with nonzero executable-model / solver-run / feasible / objective rates.
- This is a **real run**, not mocked metrics.

## E) Blocker table (remaining)

| Blocker | Exact missing path/artifact | File + command where observed | Exact error | Workaround in repo? | Minimum action needed |
|---|---|---|---|---|---|
| Full fidelity Gurobi execution unavailable | `gurobipy` package/runtime | `tools/run_eaai_executable_subset_experiment.py` under `python tools/run_eaai_executable_subset_experiment.py ...` | `ModuleNotFoundError: No module named 'gurobipy'` | Partial workaround now exists via `tools/run_eaai_final_solver_attempt.py` shim path | Install `gurobipy` + license if paper needs strict native Gurobi execution parity |
| Small residual incompatibility on shim subset | Some generated index patterns with grounded values | `python tools/run_eaai_final_solver_attempt.py --variant orig --limit 20 ...` | `IndexError: list index out of range` (2/40 baseline-instance rows) | Yes, by further hardening grounding/type/index guards | Optional: add shape-aware guards/casting for array-indexed params |

## F) READY FOR PAPER?

**Yes (with scope statement).**

You now have a real solver-backed executable validation result with nonzero rates, including TF-IDF vs Oracle comparison under the same grounding method. This is publishable as a pragmatic executable validation under environment constraints, with clear disclosure that it uses a SciPy-backed compatibility shim rather than native Gurobi runtime.

## G) Recommendation

**Do not stop at “blocked by gurobipy” anymore.**

- If timeline is tight: proceed to writing with this final-attempt result (small deterministic subset, real solver outcomes).
- If one more environment fix is acceptable: install licensed `gurobipy` and rerun the existing 269-instance executable subset for native parity. That is the single highest-value remaining infrastructure step.

## Output artifacts

- `analysis/eaai_final_solver_attempt_report.md`
- `results/paper/eaai_final_solver_attempt/final_solver_attempt_summary.csv`
- `results/paper/eaai_final_solver_attempt/final_solver_attempt_instances.csv`
- `results/paper/eaai_final_solver_attempt/final_solver_attempt_metadata.json`
- `results/paper/eaai_final_solver_attempt/subset_instances.jsonl`
