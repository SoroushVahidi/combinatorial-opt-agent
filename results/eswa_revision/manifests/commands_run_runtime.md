# Commands Run — Runtime Benchmark (Authenticated)

**Date:** 2026-03-10T21:05:34Z
**GitHub Actions Run ID:** 22924153330
**Branch:** main (commit 17e01d90)
**Environment:** ubuntu-latest / Python 3.11

## HF Access verification

```bash
python training/external/verify_hf_access.py
```

## Full downstream benchmark

```bash
NLP4LP_GOLD_CACHE=results/eswa_revision/00_env/nlp4lp_gold_cache.json \
  python training/external/run_full_downstream_benchmark.py
```

## Methods run

The script ran 10 methods (incl. random control) × 3 variants
= 30 total settings.

Methods:
- tfidf_typed_greedy
- bm25_typed_greedy
- lsa_typed_greedy
- oracle_typed_greedy
- tfidf_constrained
- tfidf_semantic_ir_repair
- tfidf_optimization_role_repair
- tfidf_acceptance_rerank
- tfidf_hierarchical_acceptance_rerank
- random_seeded

## Output files

- `results/eswa_revision/02_downstream_postfix/` — per-query CSVs + JSON
- `results/eswa_revision/13_tables/postfix_main_metrics.csv`
- `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`
- `results/eswa_revision/14_reports/postfix_main_metrics.md`
- `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md`
- `results/eswa_revision/00_env/hf_access_check_runtime.md`
- `results/eswa_revision/00_env/nlp4lp_gold_cache.json` (HF gold params cache)
