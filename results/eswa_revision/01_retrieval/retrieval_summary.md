# Retrieval Experiment Summary

**Date:** 2026-03-10  
**Script:** `training/run_baselines.py` (called via inline Python)  
**Catalog:** `data/catalogs/nlp4lp_catalog.jsonl` (335 docs)  
**Eval files:** `data/processed/nlp4lp_eval_{orig,noisy,short}.jsonl` (331 queries each)  
**Git commit:** e3fdaf4

## Results

| Variant | BM25 R@1 | TF-IDF R@1 | LSA R@1 | BM25 MRR | TF-IDF MRR | LSA MRR |
|---------|----------|------------|---------|----------|------------|---------|
| orig  | 0.8822 | 0.9094 | 0.8459 | 0.9182 | 0.9334 | 0.8904 |
| noisy | 0.8912 | 0.9033 | 0.8882 | 0.9245 | 0.9315 | 0.9240 |
| short | 0.7674 | 0.7795 | 0.7644 | 0.8178 | 0.8202 | 0.8096 |

**Note:** Catalog has 335 entries vs manuscript's 331 (4 extra entries added since last manuscript run).
This causes a small delta on some variants:
- orig TF-IDF: 0.9094 vs manuscript 0.9063 (+0.0031)
- short TF-IDF: 0.7795 vs manuscript 0.7855 (−0.006, one query boundary)

## Interpretation

- Retrieval is strong: orig TF-IDF R@1 ≈ 0.91, well above random (0.003).
- Noisy queries: retrieval is barely affected (0.90+) because the schema text still matches.
- Short queries: noticeable degradation (0.77–0.79) but still substantially above random.
- **Retrieval is not the main bottleneck**; downstream grounding is.

## Reproducibility

```bash
cd <repo_root>
python3 -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from training.run_baselines import _load_catalog, _load_eval_instances
from retrieval.baselines import get_baseline
from training.metrics import compute_metrics
ROOT = Path('.')
catalog = _load_catalog(ROOT/'data/catalogs/nlp4lp_catalog.jsonl')
for variant in ['orig','noisy','short']:
    eval_pairs = _load_eval_instances(ROOT/'data/processed'/f'nlp4lp_eval_{variant}.jsonl', catalog)
    for bl in ['bm25','tfidf','lsa']:
        baseline = get_baseline(bl)
        baseline.fit(catalog)
        r4m = [([pid for pid,_ in baseline.rank(q,top_k=10)], eid) for q,eid in eval_pairs]
        metrics = compute_metrics(r4m, k=10)
        print(f'{variant}/{bl}: R@1={metrics["P@1"]:.4f}')
"
```
