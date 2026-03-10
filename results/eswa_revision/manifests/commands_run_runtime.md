# Commands Prepared for Runtime — Authenticated Downstream Benchmark

**Date:** 2026-03-10  
**Status:** READY TO RUN (awaiting GitHub Actions trigger by repo owner)  
**Branch:** copilot/main-branch-description (commit 8aa9507)

## Summary of what was done in this Copilot session

1. Created `training/external/run_full_downstream_benchmark.py` — full pipeline script
2. Created `.github/workflows/downstream_benchmark.yml` — standalone workflow
3. Updated `.github/workflows/nlp4lp.yml` — now includes full downstream benchmark

## Why the sandbox couldn't execute the experiments

```
huggingface.co: DNS lookup blocked by sandbox DNS monitoring proxy
api.github.com/actions: dispatches endpoint blocked by proxy (HTTP 403)
HF_TOKEN: not set in Copilot sandbox environment
```

The Copilot sandbox environment has network restrictions that block:
- huggingface.co (DNS)
- GitHub Actions API dispatch endpoints

## Commands the GitHub Actions workflow will run

```bash
# Phase 0: verify
python training/external/verify_hf_access.py

# Phase 1: build eval sets  
python training/external/build_nlp4lp_benchmark.py

# Phase 2: full downstream benchmark (all 30 settings)
NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \
  python training/external/run_full_downstream_benchmark.py

# Phase 3: git commit + push results (automated in workflow)
git add results/eswa_revision/ results/paper/
git commit -m "data: add measured downstream benchmark results"
git push origin HEAD:copilot/main-branch-description
```

## Methods that will be benchmarked (3 variants each)

| Method | baseline_arg | assignment_mode |
|--------|-------------|-----------------|
| random_seeded | tfidf | typed (random_control=True) |
| tfidf_typed_greedy | tfidf | typed |
| bm25_typed_greedy | bm25 | typed |
| lsa_typed_greedy | lsa | typed |
| oracle_typed_greedy | oracle | typed |
| tfidf_constrained | tfidf | constrained |
| tfidf_semantic_ir_repair | tfidf | semantic_ir_repair |
| tfidf_optimization_role_repair | tfidf | optimization_role_repair |
| tfidf_acceptance_rerank | tfidf_acceptance_rerank | typed |
| tfidf_hierarchical_acceptance_rerank | tfidf_hierarchical_acceptance_rerank | typed |

Total: 10 methods × 3 variants = 30 settings + 4 methods × 3 variants pre-fix ablation

## Trigger instructions

```
GitHub.com → SoroushVahidi/combinatorial-opt-agent
→ Actions → "NLP4LP benchmark"
→ Run workflow → branch: copilot/main-branch-description
```
