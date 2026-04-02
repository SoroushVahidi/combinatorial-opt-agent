> **Provenance notice:** This file is retained for audit history and is **not** the canonical public summary. For reviewer-facing status and headline metrics, see **[`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md)**.

# Learning Dataset Access Audit

**Date:** 2025-03-09  
**Scope:** NLP4LP, NL4OPT, Text2Zinc, ORQA  
**Method:** Repo inspection, file existence checks, lightweight HuggingFace verification

---

## 1. NLP4LP

### Where found

| Location | Path | Status |
|----------|------|--------|
| Eval JSONL | `data/processed/nlp4lp_eval_orig.jsonl` | 331 lines |
| Eval variants | `data/processed/nlp4lp_eval_{short,noisy,nonum,noentity}.jsonl` | Present |
| Catalog | `data/catalogs/nlp4lp_catalog.jsonl` | 205KB |
| Gold cache | `results/paper/nlp4lp_gold_cache.json` | ~2MB, valid JSON |
| Ranker data | `artifacts/learning_ranker_data/nlp4lp/train.jsonl`, `test.jsonl` | ~9MB each |
| HF cache | `~/.cache/huggingface/datasets/udell-lab___nlp4_lp` | Present |
| HF hub | `~/.cache/huggingface/hub/datasets--udell-lab--NLP4LP` | Present |

### Access status

- **Local:** All eval files, catalog, and gold cache are present and usable.
- **HF:** `load_dataset("udell-lab/NLP4LP", split="test")` was run on the login node and failed with `RuntimeError: can't start new thread` (resource limits). HF cache exists; dataset was previously downloaded.
- **Bypass:** `NLP4LP_GOLD_CACHE` can be set to `results/paper/nlp4lp_gold_cache.json` to avoid HF calls entirely.

### Evidence

```bash
# Local file count
wc -l data/processed/nlp4lp_eval_orig.jsonl
# 331 data/processed/nlp4lp_eval_orig.jsonl

# Gold cache
ls -la results/paper/nlp4lp_gold_cache.json
# -rw-r--r-- 1 sv96 sv96 2095397 Mar  8 14:34 results/paper/nlp4lp_gold_cache.json

# HF verification (run on login node; failed due to thread limits)
python3 -c "
from datasets import load_dataset
ds = load_dataset('udell-lab/NLP4LP', split='test', token=os.environ.get('HF_TOKEN'))
# NLP4LP_HF: FAIL - can't start new thread
"
```

### Ready for immediate use?

**Yes.** Use local data + gold cache:

- Eval: `data/processed/nlp4lp_eval_orig.jsonl`
- Catalog: `data/catalogs/nlp4lp_catalog.jsonl`
- Gold: `export NLP4LP_GOLD_CACHE="$(pwd)/results/paper/nlp4lp_gold_cache.json"`
- Ranker data: `artifacts/learning_ranker_data/nlp4lp/train.jsonl`, `test.jsonl`

### Next step if not ready

- If HF access is needed (e.g. train/dev splits): run `load_dataset` on a compute node with GPU/SLURM (e.g. `sbatch batch/learning/...`).
- If gold cache is missing: run `load_dataset` once on a machine with sufficient resources and write to `NLP4LP_GOLD_CACHE`.

---

## 2. NL4OPT

### Where found

| Location | Path | Status |
|----------|------|--------|
| Raw generation_data | `data_external/raw/nl4opt_competition/generation_data/train.jsonl` | ~9MB |
| | `data_external/raw/nl4opt_competition/generation_data/dev.jsonl` | ~1.4MB |
| | `data_external/raw/nl4opt_competition/generation_data/test.jsonl` | ~3.9MB |
| Aux data | `artifacts/learning_aux_data/nl4opt/entity_{train,dev,test}.jsonl` | Present |
| | `artifacts/learning_aux_data/nl4opt/bound_{train,dev,test}.jsonl` | Present |
| | `artifacts/learning_aux_data/nl4opt/role_{train,dev,test}.jsonl` | Present |

### Access status

- **Local:** All raw and auxiliary data are present.

### Evidence

```bash
ls -la data_external/raw/nl4opt_competition/generation_data/
# train.jsonl dev.jsonl test.jsonl

ls -la artifacts/learning_aux_data/nl4opt/
# entity_*.jsonl bound_*.jsonl role_*.jsonl summary.json summary.md
```

### Ready for immediate use?

**Yes.** No remote access needed.

### Next step if not ready

- If raw data is missing: `git clone https://github.com/nl4opt/nl4opt-competition` into `data_external/raw/nl4opt_competition`.
- If aux data is missing: `./scripts/learning/run_build_nl4opt_aux_data.sh` (or `--sbatch`).

---

## 3. Text2Zinc

### Where found

| Location | Path | Status |
|----------|------|--------|
| Repo | — | Not found |
| Cache | — | Not found |
| Configs/scripts | — | No references |

### Access status

- **HuggingFace:** `skadio/text2zinc` exists but is gated.
- **Verification:** `load_dataset("skadio/text2zinc", split="train")` returns:  
  `Dataset 'skadio/text2zinc' is a gated dataset on the Hub. Visit the dataset page at https://huggingface.co/datasets/skadio/text2zinc to ask for access.`

### Evidence

```bash
# Repo search
grep -ri "text2zinc" . 2>/dev/null | head -5
# No matches

# HF verification
python3 -c "
from datasets import load_dataset
ds = load_dataset('skadio/text2zinc', split='train', token=os.environ.get('HF_TOKEN'))
# FAIL - Dataset is gated; request access
"
```

### Ready for immediate use?

**No.** Access must be requested first.

### Next step if not ready

1. Visit https://huggingface.co/datasets/skadio/text2zinc to request access.
2. After approval, set `HF_TOKEN` and run:
   ```bash
   python3 -c "
   from datasets import load_dataset
   ds = load_dataset('skadio/text2zinc', split='train', token=os.environ.get('HF_TOKEN'))
   print(len(ds), ds[0].keys())
   "
   ```
3. Add a loader for the Text2Zinc schema (input.json, data.dzn, model.mzn, output.json) and integrate into the learning pipeline.

---

## 4. ORQA

### Where found

| Location | Path | Status |
|----------|------|--------|
| Dataset | `data_external/raw/orqa/dataset/ORQA_test.jsonl` | 1468 lines |
| | `data_external/raw/orqa/dataset/ORQA_validation.jsonl` | 45 lines |
| Manifest | `data_external/manifests/public_data_manifest.json` | Listed as downloaded |

### Access status

- **Local:** All dataset files are present.
- **Source:** GitHub `nl4opt/ORQA` (cloned via `git clone`).
- **Integration:** No training scripts or configs reference ORQA yet.

### Evidence

```bash
wc -l data_external/raw/orqa/dataset/*.jsonl
# 1468 ORQA_test.jsonl
# 45 ORQA_validation.jsonl
```

### Ready for immediate use?

**Yes for data access.** ORQA is QA (multiple choice), not slot filling. Training pipeline integration is not yet implemented.

### Next step if not ready

- If data is missing: `git clone https://github.com/nl4opt/ORQA` into `data_external/raw/orqa`.
- For learning experiments: add a loader that reads `ORQA_test.jsonl` / `ORQA_validation.jsonl` and maps CONTEXT, QUESTION, OPTIONS, TARGET_ANSWER to the format used by the learning pipeline.

---

## Summary: Usable datasets (ranked)

1. **NLP4LP** — Usable now. Local eval + catalog + gold cache; ranker data present. Primary learning target.
2. **NL4OPT** — Usable now. Raw and auxiliary data present. Auxiliary training.
3. **ORQA** — Usable now for data. No integration yet; QA format differs from slot filling.

---

## Summary: Datasets requiring extra action (ranked)

1. **Text2Zinc** — Request access at https://huggingface.co/datasets/skadio/text2zinc, then add loader and integration.

---

## Recommendation

**Start with NLP4LP.** It is:

- Present in the repo
- Has a gold cache that avoids HF on login nodes
- Has pre-built ranker data (`artifacts/learning_ranker_data/nlp4lp`)
- Is the main evaluation target for slot filling
- Is fully supported by training scripts (`scripts/learning/run_train_nlp4lp_ranker.sh`, etc.)

**Use NL4OPT as auxiliary** for entity/bound/role tasks; data is already present.

**Defer Text2Zinc** until access is granted and a loader is added.

**ORQA** can be used for evaluation or QA-style experiments once a loader is integrated.
