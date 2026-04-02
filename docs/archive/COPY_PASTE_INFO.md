# Repository Information — Copy-Paste Reference

**Project:** NLP4LP — Natural-language optimization problem instantiation pipeline  
**Repo:** `SoroushVahidi/combinatorial-opt-agent`  
**Full audit:** see `docs/FULL_REPO_SUMMARY.md`

---

## What this repo does

Three-stage pipeline for natural-language → optimization:

1. **Retrieve** the right optimization schema from a catalog (331-query NLP4LP benchmark)
2. **Extract** numeric mentions from the natural-language query
3. **Assign** numbers to scalar slots in the retrieved schema

---

## Key numbers (copy-paste ready)

```
PROJECT: Natural-language optimization problem instantiation (NLP4LP pipeline).
Three stages: (1) retrieve the right optimization schema, (2) extract numeric
mentions from the query, (3) assign numbers to scalar slots in the schema.
Test set: 331 queries per variant (orig, noisy, short).
No LaTeX manuscript or PDF committed; all numbers come from docs/ files.

--- RETRIEVAL ---
Schema R@1 (Recall@1) on orig: TF-IDF 0.9063, BM25 0.8852, LSA 0.8550.
Retrieval is strong. TF-IDF is best.

--- DOWNSTREAM (orig, N=331) ---
Main metric: InstantiationReady = fraction of queries with param_coverage >= 0.8
AND type_match >= 0.8.

Method                                   | Coverage | TypeMatch | Exact20 | InstReady
tfidf (typed greedy, baseline)           | 0.8222   | 0.2267    | 0.2140  | 0.0725
tfidf_constrained                        | 0.7720   | 0.1980    | 0.3279  | 0.0272
tfidf_semantic_ir_repair                 | 0.7780   | 0.2540    | 0.2610  | 0.0630
tfidf_optimization_role_repair           | 0.8220   | 0.2430    | 0.2770  | 0.0600
tfidf_acceptance_rerank                  | 0.7974   | 0.2275    |   —     | 0.0816
tfidf_hierarchical_acceptance_rerank     | 0.7771   | 0.2303    |   —     | 0.0846  <- best
oracle (typed greedy, perfect retrieval) | 0.8695   | 0.2475    | 0.1871  | 0.0816
random (seeded)                          | 0.0101   | 0.0060    | 0.1250  | 0.0060

KEY FINDING 1: Even with perfect retrieval (oracle), InstantiationReady = 0.082.
The bottleneck is NOT retrieval; it is downstream number-to-slot grounding.

KEY FINDING 2: TypeMatch on hits is only 0.23-0.25 across all methods;
float type_match ~= 0.03 (hardest type). 87% of queries have >= 3 confusable
numeric values.

KEY FINDING 3: No single method dominates all metrics. tfidf_constrained is
best on Exact20 (0.328) but worst on InstantiationReady (0.027).
tfidf_hierarchical_acceptance_rerank is best on InstantiationReady (0.085)
but lowers Schema R@1 to 0.846. The typed-greedy tfidf baseline is the most
balanced deterministic method.

KEY FINDING 4: optimization_role_repair gives the best balance (preserves
coverage, improves TypeMatch and Exact20 over typed greedy) among the
non-rerank structured methods.

--- WHAT IS NOT YET IN THE REPO ---
- No trained learning models (torch not installed; training scripts not written).
- No committed result CSVs (results/ directory absent; numbers only in docs files).
- optimization_role_relation_repair, anchor_linking, bottomup_beam_repair,
  entity_semantic_beam_repair: NOT implemented anywhere.
- Stage-3 experiment matrix exists as config/scripts but has never been run.

--- HONEST PAPER CONCLUSION ---
1. Retrieval is no longer the main bottleneck (evidence: oracle barely moves
   InstReady from 0.0725 to 0.0816).
2. Downstream number-to-slot grounding is the main bottleneck.
3. optimization_role_repair is the best-balanced structured deterministic method.
4. Hierarchical acceptance rerank gives highest InstantiationReady (0.085) but
   at the cost of retrieval accuracy (Schema R@1 drops to 0.846).
5. No learning results exist yet; all results are deterministic heuristic methods.
```

---

## Implementation status

| Component | Status |
|-----------|--------|
| BM25 retrieval baseline | ✅ Implemented |
| TF-IDF retrieval baseline | ✅ Implemented |
| LSA retrieval baseline | ✅ Implemented |
| Oracle / Random baselines | ✅ Implemented |
| `typed` greedy assignment | ✅ Implemented |
| `untyped` assignment (ablation) | ✅ Implemented |
| `constrained` assignment | ✅ Implemented |
| `semantic_ir_repair` | ✅ Implemented |
| `optimization_role_repair` | ✅ Implemented |
| `acceptance_rerank` variants (flat + hierarchical) | ✅ Implemented |
| `optimization_role_relation_repair` | ❌ Not implemented |
| `optimization_role_anchor_linking` | ❌ Not implemented |
| `optimization_role_bottomup_beam_repair` | ❌ Not implemented |
| `optimization_role_entity_semantic_beam_repair` | ❌ Not implemented |
| Pairwise ranker training | ❌ Not yet created |
| Multitask grounder training | ❌ Not yet created |
| Committed result CSVs (`results/`) | ❌ Directory absent |

---

## Eval data (present)

| File | Content |
|------|---------|
| `data/processed/nlp4lp_eval_orig.jsonl` | 331 original NLP4LP queries |
| `data/processed/nlp4lp_eval_noisy.jsonl` | 331 `<num>`-masked variants |
| `data/processed/nlp4lp_eval_short.jsonl` | 331 short variants |
| `data/processed/nlp4lp_eval_noentity.jsonl` | 331 entity-stripped variants |
| `data/processed/nlp4lp_eval_nonum.jsonl` | 331 number-stripped variants |
| `data/catalogs/nlp4lp_catalog.jsonl` | NLP4LP schema catalog |

---

## Key source files

| File | Purpose |
|------|---------|
| `tools/nlp4lp_downstream_utility.py` | All downstream assignment modes |
| `retrieval/baselines.py` | BM25 / TF-IDF / LSA baselines |
| `training/run_baselines.py` | Main evaluation runner |
| `tools/make_nlp4lp_paper_artifacts.py` | Generates result CSVs/LaTeX (not yet run) |
| `docs/NLP4LP_MANUSCRIPT_REPORTING_PACKAGE.md` | Canonical result tables |
| `docs/FULL_REPO_SUMMARY.md` | Complete 8-section audit |
