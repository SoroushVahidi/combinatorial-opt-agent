> **Provenance notice:** This file is retained for audit history and is **not** the canonical public summary. See **[`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md)**.

# Repo Cleanup Report

**Date:** 2026-03-09  
**Archive folder:** `archive/cleanup_20260309/`

---

## 1. Summary of disk usage before cleanup

| Location | Size |
|----------|------|
| **Repo total** | ~15 GB |
| venv/ | 9.7 GB |
| venv_sbert/ | 2.8 GB |
| data_external/ | 1.6 GB |
| artifacts/ | 347 MB |
| data_private/ | 191 MB |
| data/ | 141 MB |
| results/ | 99 MB |
| logs/ | 518 KB |
| __pycache__ (repo source) | ~129 KB |
| .pytest_cache | ~6 KB |

---

## 2. Biggest directories/files found

| Path | Size | Action |
|------|------|--------|
| venv/ | 9.7 GB | **Kept** (project Python environment) |
| venv_sbert/ | 2.8 GB | **Kept** (alternate env; not deleted—uncertain if still used) |
| data_external/ | 1.6 GB | **Kept** (external datasets) |
| artifacts/learning_runs/first_learning_run/ | 314 MB | **Kept** (newest learning checkpoint) |
| artifacts/learning_ranker_data/ | 18 MB | **Kept** (training data) |
| artifacts/learning_aux_data/ | 14 MB | **Kept** (NL4Opt aux data) |
| results/paper/ | ~99 MB | **Kept** (manuscript artifacts) |

---

## 3. What was deleted (Group A)

| Item | Est. size |
|------|-----------|
| .pytest_cache | ~6 KB |
| collectors/__pycache__ | ~258 KB |
| tests/__pycache__ | ~643 KB |
| retrieval/__pycache__ | ~259 KB |
| tools/__pycache__ | ~385 KB |
| formulation/__pycache__ | ~258 KB |
| pipeline/__pycache__ | small |
| scripts/__pycache__ | small |
| training/__pycache__ | small |
| training/external/__pycache__ | small |
| src/learning/__pycache__ | small |
| src/learning/models/__pycache__ | small |
| __pycache__ (root) | small |
| data_external/raw/optmath/.../__pycache__ | small |
| data_external/raw/finqa/.../__pycache__ | small |

**Total deleted:** ~2–2.5 MB

---

## 4. What was archived (Group B)

| Item | Destination |
|------|-------------|
| logs/mention_slot_%j.out | archive/cleanup_20260309/logs/ |
| logs/mention_slot_%j.err | archive/cleanup_20260309/logs/ |
| logs/nlp4lp_focused_eval_853539.out | archive/cleanup_20260309/logs/ |
| logs/nlp4lp_focused_eval_853539.err | archive/cleanup_20260309/logs/ |
| logs/nlp4lp_focused_eval_853548.out | archive/cleanup_20260309/logs/ |
| logs/nlp4lp_focused_eval_853548.err | archive/cleanup_20260309/logs/ |
| data_external/manifests/collection_20260307_151515.log | archive/cleanup_20260309/data_external_manifests/ |

**Total archived:** ~262 KB

---

## 5. What was intentionally kept

- **Source code:** All .py, configs, scripts
- **Manuscript/paper artifacts:** results/paper/*, results/*.csv, *.tex, *.png
- **Current learning run:** artifacts/learning_runs/first_learning_run/ (checkpoint.pt, config.json)
- **Training data:** artifacts/learning_ranker_data/, artifacts/learning_aux_data/, artifacts/learning_corpus/
- **Stage 3 results:** artifacts/learning_runs/rule_baseline/, nlp4lp_pairwise_*, nl4opt_*, stage3_*.json/csv/md
- **Current logs:** logs/learning/ (includes train_nlp4lp_first_run_854608.out)
- **Datasets:** data/, data_external/, data_private/
- **Benchmark artifacts:** results/, comparison_reports/
- **Audit docs:** docs/learning_runs/, nlp4lp_manuscript_vs_current_repo_audit.*, learning_dataset_access_audit.*
- **Python environments:** venv/, venv_sbert/

---

## 6. Estimated space freed

| Category | Size |
|----------|------|
| Deleted (caches) | ~2–2.5 MB |
| Archived (logs) | ~262 KB |
| **Total** | ~2.5 MB |

---

## 7. Risky items left untouched

- **venv_sbert/ (2.8 GB):** Alternate Python environment. Not deleted; unclear if still needed. Consider removing if fully superseded by venv/.
- **data_private/ (191 MB):** Private data; not inspected or modified.
- **data_external/ (1.6 GB):** External datasets; kept for experiments.
- **artifacts/learning_runs/rule_baseline, nlp4lp_pairwise_*, nl4opt_*:** Old Stage 3 run dirs (no checkpoints, rule-only metrics). Kept; small (~650 KB total).

---

## 8. Recommendations for future .gitignore / cleanup hygiene

1. **Add to .gitignore** (if not already): `__pycache__/`, `.pytest_cache/`, `*.pyc`, `archive/`
2. **Log rotation:** Periodically archive or remove old SLURM logs (e.g. logs/*.out older than 30 days) to avoid unbounded growth.
3. **venv_sbert:** If no longer used, remove it to reclaim ~2.8 GB.
4. **Archive convention:** Use `archive/cleanup_YYYYMMDD/` for future cleanups; document contents in a README inside the archive.
5. **Large artifact review:** Before adding large outputs to artifacts/, consider whether they can be regenerated or stored elsewhere (e.g. SCRATCH).
