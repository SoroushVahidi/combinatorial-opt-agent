# EAAI Engineering Validation Subset Experiment (NLP4LP)

## A) Experiment run and why this is highest-priority

We ran a direct end-to-end engineering validation on a manageable subset (60 instances) that explicitly measures whether retrieval-assisted schema grounding and scalar instantiation produce **usable model artifacts**, not just retrieval/slot-filling scores.

Pipeline per instance:

1. NL description
2. Schema retrieval (`tfidf`, `bm25`, or `oracle`)
3. Scalar grounding (`optimization_role_relation_repair`, shared across all baselines)
4. Structured artifact instantiation via `problem_info` (`objective`, `constraints`, `parametrized_description` with filled scalar values)
5. Structural validation (`coverage >= 0.8`, `type_match >= 0.8`), parseability checks, and solver-feasibility checks.

This is high-priority for EAAI because it directly tests practical utility under realistic retrieval errors while preserving an oracle upper reference.

## B) Exact commands run

```bash
rg --files | head -n 200
sed -n '1,260p' tools/nlp4lp_downstream_utility.py
sed -n '1,260p' retrieval/baselines.py
sed -n '1,260p' validation/solve_and_verify.py
python tools/run_eaai_engineering_subset_experiment.py --variant orig --subset-size 60 --assignment-mode optimization_role_relation_repair
python -m py_compile tools/run_eaai_engineering_subset_experiment.py
python - <<'PY'
import csv
p='results/paper/eaai_engineering_subset/engineering_subset_summary.csv'
print(list(csv.DictReader(open(p))))
PY
python - <<'PY'
import os
os.environ['NLP4LP_GOLD_CACHE']='results/eswa_revision/00_env/nlp4lp_gold_cache.json'
from tools.nlp4lp_downstream_utility import _load_hf_gold
G=_load_hf_gold('test')
print('pyomo_solver_code_count',sum(1 for v in G.values() if isinstance(v,dict) and (v.get('solver_code') or {}).get('pyomo')))
print('optimus_code_count',sum(1 for v in G.values() if v.get('optimus_code')))
PY
```

## C) Subset definition and selection

- Source split: `data/processed/nlp4lp_eval_orig.jsonl`
- Selection: first 60 instances where the gold problem has 2–8 scalar parameters.
- Deterministic: ordered scan from the eval file with a fixed predicate.
- Subset artifact saved to: `results/paper/eaai_engineering_subset/subset_instances.jsonl`.

## D) Results table (engineering metrics)

| baseline | subset_size | schema_hit_rate | mean_param_coverage | mean_type_match | structural_valid_rate | parseable_rate | instantiation_complete_rate | solver_feasible_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 60 | 0.9333 | 0.9464 | 0.9183 | 0.7500 | 0.9333 | 0.7500 | 0.0000 |
| bm25 | 60 | 0.9000 | 0.9446 | 0.9120 | 0.7333 | 0.9333 | 0.7333 | 0.0000 |
| oracle | 60 | 1.0000 | 0.9562 | 0.9244 | 0.7667 | 0.9500 | 0.7833 | 0.0000 |

Key takeaways:

- TF-IDF beats BM25 on schema-hit and downstream readiness in this subset.
- Oracle schema gives a measurable headroom (~3.3 points on structural validity, ~3.3 points on complete instantiation).
- Grounding quality, not only retrieval, remains a bottleneck even under oracle schema.

## E) Case-study table (5 examples)

| baseline | query_id | schema_hit | n_expected_scalar | n_filled | param_coverage | type_match | structural_valid | instantiation_complete | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| tfidf | nlp4lp_test_0 | 1 | 5 | 5 | 1.0000 | 1.0000 | 1 | 1 | Clean success end-to-end. |
| tfidf | nlp4lp_test_1 | 1 | 2 | 2 | 1.0000 | 1.0000 | 1 | 1 | Clean success with low slot count. |
| tfidf | nlp4lp_test_3 | 1 | 7 | 6 | 0.8571 | 0.8333 | 1 | 0 | Structurally valid but incomplete instantiation (missing slot fill). |
| tfidf | nlp4lp_test_10 | 1 | 8 | 7 | 0.8750 | 0.7143 | 0 | 0 | Type mismatch causes structural invalidity. |
| tfidf | nlp4lp_test_12 | 0 | 9 | 5 | 0.5556 | 0.8000 | 0 | 0 | Schema miss cascades into poor coverage. |

## F) BLOCKERS

### Blocker 1: Hugging Face online dataset access unavailable from this environment

- Failed command: internal call during `_load_hf_gold()` without cache.
- File/path: `tools/nlp4lp_downstream_utility.py` (`load_dataset("udell-lab/NLP4LP", split=...)`).
- Exact error: `httpx.ProxyError: 403 Forbidden`.
- Workaround used: set `NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json` and load cached gold data.
- Minimum action needed from maintainers: provide network access to HF or maintain the local gold cache artifact.

### Blocker 2: No executable Pyomo solver path in available gold instances

- Evidence command showed `pyomo_solver_code_count 0` for test split cache.
- Existing solver validator requires `solver_code.pyomo`; otherwise it immediately returns false.
- File/path: `validation/solve_and_verify.py` checks `problem.get("solver_code", {}).get("pyomo")`.
- Minimum action needed from maintainers: provide generated/curated Pyomo models (or a supported solver code path) per instance to enable solver-feasibility evaluation.

## G) READY FOR PAPER?

**Yes, with caveat.**

This subset experiment is strong enough to add now as an engineering validation section because it:

- compares TF-IDF, BM25, and Oracle schema under one grounding method,
- reports practical end-to-end usability metrics,
- includes explicit failure modes and case studies,
- quantifies retrieval-vs-grounding gap using oracle reference.

However, solver-feasibility remains blocked by missing solver-code artifacts; manuscript text should clearly state that this run reports structural/instantiation readiness rather than executable solve rates.
