# Restricted executable-solver subset experiment

## A) What exact experiment was run and why this is highest-value follow-up

We ran an end-to-end follow-up on the **largest deterministic executable-eligible NLP4LP subset available in local gold cache**: all `orig` eval instances whose gold entry has non-empty `optimus_code` (269 instances). For each instance we executed:

`NL query -> schema retrieval (tfidf / bm25 / oracle) -> scalar grounding (optimization_role_relation_repair) -> template instantiation checks -> executable-code attempt (OptiMUS/Gurobi code exec with grounded parameters) -> solver outcome extraction when possible`.

This is the highest-value immediate follow-up because the prior study already established structural validity trends; this run directly tests executable-model and solver-run outcomes on every instance where executable artifacts exist.

## B) Exact commands run

```bash
# Inspect executable artifact availability in local NLP4LP cache
python - <<'PY'
import json
from pathlib import Path
p=Path('results/eswa_revision/00_env/nlp4lp_gold_cache.json')
d=json.loads(p.read_text())['gold_by_id']
vals=list(d.values())
print('total',len(vals))
print('optimus_code',sum(1 for v in vals if v.get('optimus_code')))
print('solver_code.pyomo',sum(1 for v in vals if isinstance(v.get('solver_code'),dict) and v['solver_code'].get('pyomo')))
PY

# Run restricted executable subset experiment
python tools/run_eaai_executable_subset_experiment.py \
  --variant orig \
  --assignment-mode optimization_role_relation_repair

# Read generated summary and case studies
python - <<'PY'
from pathlib import Path
print(Path('results/paper/eaai_executable_subset/executable_subset_summary.csv').read_text())
print(Path('results/paper/eaai_executable_subset/executable_case_studies.csv').read_text())
PY
```

## C) Exact deterministic subset rule

Subset = all rows in `data/processed/nlp4lp_eval_orig.jsonl` where the corresponding `gold_by_id[relevant_doc_id].optimus_code` is non-empty in `results/eswa_revision/00_env/nlp4lp_gold_cache.json`.

- Total eval rows: 331
- Restricted executable-eligible subset size: 269
- Ordering: preserved eval file order (first-to-last)

## D) Results table (key metrics)

| Baseline | Subset | Schema hit rate | Structural valid rate | Instantiation complete rate | Executable model rate | Solver-run success rate | Feasible solution rate | Objective produced rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 269 | 0.9368 | 0.8141 | 0.6654 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bm25 | 269 | 0.9257 | 0.8104 | 0.6580 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle | 269 | 1.0000 | 0.8253 | 0.6840 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Failure category highlights from `failure_counts`:

- Dominant executable blocker: `ModuleNotFoundError` for `gurobipy` on all tfidf/oracle rows and 267/269 bm25 rows.
- BM25 had 2 additional rows with `missing_optimus_code` due schema miss to non-executable candidates.
- Structural/grounding failures still visible independent of solver dependency:
  - tfidf: `schema_miss=17`, `incomplete_instantiation=90`, `missing_scalar_slots=24`, `type_mismatch=27`
  - bm25: `schema_miss=20`, `incomplete_instantiation=92`, `missing_scalar_slots=24`, `type_mismatch=28`
  - oracle: `incomplete_instantiation=85`, `missing_scalar_slots=21`, `type_mismatch=27`

## E) Case studies (5 examples)

| Case | Query ID | Baseline | Outcome summary |
|---|---|---|---|
| Retrieval/schema failure | `nlp4lp_test_4` | tfidf | Predicted `nlp4lp_test_41` (schema miss), but structural/instantiation checks happened on wrong schema; executable step failed on missing `gurobipy`. |
| Grounding/parameter failure | `nlp4lp_test_3` | tfidf | Correct schema hit; instantiation incomplete despite structural validity; executable step failed on missing `gurobipy`. |
| Structural+instantiation success (pre-exec) | `nlp4lp_test_0` | tfidf | Schema hit + structural valid + instantiation complete; executable step still failed on missing `gurobipy`. |
| Structural+instantiation success (pre-exec) | `nlp4lp_test_1` | tfidf | Same pattern as above: usable artifact pre-exec, blocked at solver dependency. |
| Structural+instantiation success (pre-exec) | `nlp4lp_test_2` | tfidf | Same pattern as above: strong upstream pipeline, executable blocker downstream. |

## F) BLOCKERS

### Blocker 1: Missing `gurobipy` runtime dependency for all OptiMUS executable paths
- Missing dependency/code path: `gurobipy` import required by every `optimus_code` artifact.
- Exact failing path: `tools/run_eaai_executable_subset_experiment.py` during per-instance `exec(optimus_code)`.
- Exact error message (captured per-instance):
  - `ModuleNotFoundError: No module named 'gurobipy'`
- Existing workaround present in repo: **none** for executing OptiMUS code without Gurobi runtime.
- Minimum action needed from authors:
  1. Provide a working Gurobi Python runtime (`gurobipy`) in this environment.
  2. Provide a valid Gurobi license (or an alternative executable backend path for OptiMUS code).

### Blocker 2: Network-restricted package install prevented dependency acquisition
- Missing external access/dependency: pip access to retrieve `gurobipy` wheel.
- Exact failed command:
  - `python -m pip install gurobipy -q`
- Exact error message:
  - `ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))`
  - Final pip failure: `ERROR: No matching distribution found for gurobipy`
- Existing workaround present in repo: local preinstalled `gurobipy` not available; no vendored wheel.
- Minimum action needed from authors:
  1. Enable package/network access for `gurobipy`, or
  2. Preinstall `gurobipy` in the runtime image.

## G) READY FOR PAPER?

**Partially.** This follow-up is strong enough to add as evidence that upstream retrieval+grounding remains stable on the executable-eligible subset (269 instances), but it is **not yet a completed executable-solver validation** because solver execution is hard-blocked by missing `gurobipy` runtime/license. The manuscript can include this as a transparent executable-attempt study with explicit infrastructure blockers; final solver-feasibility claims should wait until the dependency/licensing blocker is resolved.

## Output artifacts

- `results/paper/eaai_executable_subset/executable_subset_summary.csv`
- `results/paper/eaai_executable_subset/executable_subset_instances.csv`
- `results/paper/eaai_executable_subset/executable_case_studies.csv`
- `results/paper/eaai_executable_subset/subset_instances.jsonl`
- `results/paper/eaai_executable_subset/executable_subset_metadata.json`
