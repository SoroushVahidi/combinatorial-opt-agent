# Reproduce Experiments — ESWA Revision

**Repo:** `SoroushVahidi/combinatorial-opt-agent`  
**Branch:** `copilot/main-branch-description`  
**Date:** 2026-03-10

---

## Environment assumptions

- Python 3.10+ (tested with 3.11 and 3.12)
- All dependencies: `pip install -r requirements.txt`
  (includes `rank_bm25`, `scikit-learn`, `datasets`, `sentence-transformers`, `matplotlib`, `pandas`)
- CPU-only: all retrieval and deterministic downstream methods run without GPU
- HF_TOKEN: required only for downstream evaluation (gold parameters from `udell-lab/NLP4LP`)

---

## HF_TOKEN usage

The gated dataset `udell-lab/NLP4LP` requires a read access token:

```bash
# Option 1: export in shell session
export HF_TOKEN=hf_...

# Option 2: use huggingface_hub CLI
pip install huggingface_hub
huggingface-cli login --token hf_...

# Option 3: set as GitHub Actions secret
# Settings → Secrets and variables → Actions → New repository secret
# Name: HF_TOKEN, Value: hf_...
```

Verify access:
```bash
python training/external/verify_hf_access.py
# Should print: HuggingFace access: OK
```

---

## Mandatory experiments (for the manuscript)

### M1: Retrieval benchmark (NO HF_TOKEN needed)

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
        print(f'{variant}/{bl}: R@1={metrics[\"P@1\"]:.4f}  R@5={metrics.get(\"P@5\",0):.4f}  MRR={metrics.get(\"MRR@10\",0):.4f}')
"
```

**Expected outputs:**
- orig/tfidf: R@1 ≈ 0.9094 (catalog has 4 extra entries vs manuscript; expect ±0.006)
- noisy/tfidf: R@1 ≈ 0.9033
- short/tfidf: R@1 ≈ 0.7795

### M2: Full downstream evaluation (REQUIRES HF_TOKEN)

```bash
export HF_TOKEN=hf_...

# All 10 deterministic methods, orig variant
python tools/run_nlp4lp_focused_eval.py --variant orig

# Individual methods (for fine-grained control)
for mode in typed constrained semantic_ir_repair optimization_role_repair optimization_role_relation_repair; do
    python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode $mode
done

# Acceptance rerank variants
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf_acceptance_rerank --assignment-mode typed --acceptance-k 10
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf_hierarchical_acceptance_rerank --assignment-mode typed --acceptance-k 10

# Output: results/paper/nlp4lp_downstream_summary.csv
# Also: results/paper/nlp4lp_focused_eval_summary.csv
```

### M3: Cross-variant robustness (REQUIRES HF_TOKEN)

```bash
export HF_TOKEN=hf_...
for variant in noisy short; do
    python tools/run_nlp4lp_focused_eval.py --variant $variant
done
```

### M4: Paper artifact generation

```bash
python tools/make_nlp4lp_paper_artifacts.py
# Requires: results/nlp4lp_retrieval_summary.csv + results/paper/nlp4lp_downstream_summary.csv
# Outputs: LaTeX tables, CSV summaries, figures
```

---

## Optional experiments (new methods — REQUIRES HF_TOKEN)

```bash
export HF_TOKEN=hf_...

# New assignment modes (not yet benchmarked end-to-end)
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode global_consistency_grounding
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_anchor_linking
python tools/nlp4lp_downstream_utility.py --variant orig --baseline tfidf --assignment-mode optimization_role_bottomup_beam_repair
```

---

## ESWA revision experiment outputs (pre-generated)

All outputs from this revision round are in `results/eswa_revision/`:

| Directory | Contents |
|-----------|---------|
| `00_env/` | HF access check report |
| `01_retrieval/` | Retrieval JSON + markdown summary |
| `03_prefix_vs_postfix/` | Float fix ablation report |
| `04_method_comparison/` | Deterministic comparison report |
| `05_retrieval_vs_grounding/` | Bottleneck analysis |
| `06_robustness/` | Cross-variant robustness |
| `07_sae/` | SAE evaluation status |
| `08_error_taxonomy/` | Error analysis |
| `09_case_studies/` | Illustrative cases |
| `10_learning_appendix/` | Negative learning result |
| `11_runtime/` | Runtime measurements |
| `12_figures/` | 8 PNG figures |
| `13_tables/` | 12 CSV tables |
| `14_reports/` | Markdown reports + FINAL_ESWA_EXPERIMENT_SUMMARY.md |
| `manifests/` | Experiment manifest JSON + commands |

---

## Key files by experiment

| Experiment | Result file |
|-----------|-------------|
| Retrieval (all variants) | `results/eswa_revision/01_retrieval/retrieval_results.json` |
| Downstream (pre-existing) | `results/paper/nlp4lp_downstream_summary.csv` |
| Downstream (post-fix, pending) | `results/paper/nlp4lp_focused_eval_summary.csv` |
| Type distribution analysis | `results/paper/nlp4lp_downstream_types_summary.csv` |
| Learning run artifacts | `artifacts/learning_runs/real_data_only_learning_check/metrics.json` |
