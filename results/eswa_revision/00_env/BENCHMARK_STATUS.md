# Benchmark Execution Status — Evidence-Based Verification

**Date of inspection:** 2026-03-10  
**Branch inspected:** `copilot/main-branch-description`  
**Short answer:** `NLP4LP downstream benchmark has NOT run. All post-fix downstream result files are placeholders.`

---

## Run workflow button — Root cause and fix

### Root cause

GitHub only shows the **"Run workflow"** button in the Actions UI for `workflow_dispatch` workflows
that exist on the repository's **default branch** (`main`).

Confirmed state (2026-03-10):
- `main` has **NO `.github/` directory** — verified by GitHub API (`GET /repos/.../contents/.github?ref=main` → 404)
- All 3 workflow files exist **only** on `copilot/main-branch-description`
- Result: workflows appear in the Actions sidebar (they have run history from push triggers), but
  the **"Run workflow" button is absent** on all 3 workflow pages

### Minimal fix applied in this PR

All 3 workflow files now have `push: branches: [main, copilot/main-branch-description]` triggers
with path filters. After this PR is merged:

1. The merge commit pushes the workflow files to `main`
2. The path-filtered push triggers fire on `main` → fast registration runs (~10s each, no benchmark)
3. All expensive steps are gated with `if: github.event_name == 'workflow_dispatch'`
4. GitHub detects `workflow_dispatch` on the default branch → **"Run workflow" button appears**

### Exact UI steps after PR merge

```
1. Merge this PR (or repo owner merges via GitHub PR page)
2. Go to: https://github.com/SoroushVahidi/combinatorial-opt-agent/actions
3. In the left sidebar, click "NLP4LP benchmark"
4. Click the "Run workflow" button (top-right of the run list)
5. In the dropdown, select branch: copilot/main-branch-description
   (or main, after results are merged back)
6. Click the green "Run workflow" button
7. Runtime: ~2–3 hours
8. Results committed automatically to the selected branch with source: measured
```

### If you cannot merge the PR yet (trigger right now via CLI)

```bash
# Trigger the full benchmark on the PR branch without merging:
gh workflow run nlp4lp.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref copilot/main-branch-description

# Or use the workflow ID:
gh workflow run 244220556 \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref copilot/main-branch-description
```

This uses the GitHub CLI which can trigger `workflow_dispatch` on any branch,
bypassing the UI's default-branch requirement.

---

## Section 1: Workflow Verification

### `.github/workflows/nlp4lp.yml` — Name: `NLP4LP benchmark`

- **Triggers:** `workflow_dispatch` + `push` (path-filtered to `[main, copilot/main-branch-description]`)
- **Jobs:** `build-benchmark` (single job, 5h timeout)
- **Exact run commands (workflow_dispatch only — gated):**
  ```bash
  python training/external/verify_hf_access.py
  python training/external/build_nlp4lp_benchmark.py
  python training/external/run_full_downstream_benchmark.py
  ```
- **Classification:** REAL FULL DOWNSTREAM BENCHMARK RUNNER (all 3 phases)
- **2-minute run plausible?** NO — 30 settings × ~4 min each ≈ 2 hours minimum
- **Uses HF_TOKEN:** YES (Phase 0, 1, 2)
- **Writes/commits results:** YES (`contents: write` permission, `git push`)
- **Workflow ID:** 244220556

### `.github/workflows/downstream_benchmark.yml` — Name: `NLP4LP downstream benchmark (authenticated)`

- **Triggers:** `workflow_dispatch` + `push` (path-filtered to `[main, copilot/main-branch-description]`)
- **Jobs:** `run-downstream-benchmark` (single job, 4h timeout)
- **Exact run commands (workflow_dispatch only — gated):**
  ```bash
  python training/external/verify_hf_access.py
  python training/external/run_full_downstream_benchmark.py
  ```
- **Classification:** FULL DOWNSTREAM BENCHMARK (skips eval-set build; assumes data/processed/ already populated)
- **2-minute run plausible?** NO — same 30-setting loop, ~2h runtime
- **Uses HF_TOKEN:** YES
- **Writes/commits results:** YES
- **Workflow ID:** 244290125

### `.github/workflows/check-hf-access.yml` — Name: `Check HF access`

- **Triggers:** `workflow_dispatch` + `push` (path-filtered to `[main, copilot/main-branch-description]`)
- **Jobs:** `verify` (single job, no timeout)
- **Exact run commands (workflow_dispatch only — gated):**
  ```bash
  python training/external/verify_hf_access.py
  ```
- **Classification:** QUICK CI CHECK ONLY — NOT the benchmark runner
- **2-minute run plausible?** YES — only verifies HF token; typically ~60s
- **Uses HF_TOKEN:** YES (verification only)
- **Writes/commits results:** NO

---

## Section 2: Benchmark Runner Script Verification

**File:** `training/external/run_full_downstream_benchmark.py`

- **Truly runs full benchmark:** YES — iterates over all methods × variants
- **Loops:**
  - Outer loop: `for variant in ["orig", "noisy", "short"]` (line 249)
  - Inner loop 1: `run_single_setting(..., random_control=True)` — 1 random control per variant
  - Inner loop 2: `for baseline_arg, assignment_mode, display_name in METHODS:` — 9 methods per variant (line 290)
  - Ablation loop: `for variant in VARIANTS:` × `for ... in ABLATION_METHODS:` (lines 333–384)
- **Total settings:** `(9 + 1) × 3 = 30` post-fix settings + `4 × 3 = 12` pre-fix ablation settings
- **Output files/directories it writes:**
  - `results/eswa_revision/02_downstream_postfix/` — per-query CSVs + per-variant JSON (30+ files)
  - `results/eswa_revision/03_prefix_vs_postfix/paper_prefix/` — pre-fix simulation outputs
  - `results/eswa_revision/13_tables/postfix_main_metrics.csv` — aggregate table (overwrites placeholder)
  - `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` — ablation table (overwrites)
  - `results/eswa_revision/14_reports/postfix_main_metrics.md` — markdown report (new file)
  - `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md` — ablation markdown (new file)
  - `results/eswa_revision/00_env/hf_access_check_runtime.md` — overwrites placeholder
  - `results/eswa_revision/manifests/commands_run_runtime.md` — overwrites placeholder
  - `results/paper/nlp4lp_downstream_summary.csv`
  - `results/paper/nlp4lp_downstream_types_summary.csv`
- **Includes pre-fix vs post-fix ablation:** YES — via `_run_setting_prefix()` which monkey-patches `_is_type_match` (line 120–152)
- **Real full run duration:** ~2–3 hours (30+ settings × individual evaluations per query)

---

## Section 3: Result File Verification

### `results/eswa_revision/13_tables/postfix_main_metrics.csv`

| Row | source column (exact cell value) |
|-----|----------------------------------|
| orig,random_seeded | `placeholder-pre-fix-manuscript-era (run NLP4LP benchmark workflow to replace)` |
| orig,tfidf_typed_greedy | `placeholder-pre-fix-manuscript-era (run NLP4LP benchmark workflow to replace)` |
| ... (all 10 rows) | same placeholder label |

**Conclusion:** NOT measured. These are pre-fix manuscript-era numbers used as placeholders.  
`run_full_downstream_benchmark.py` writes `source: measured` — that column value has NEVER appeared here.

### `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`

- Post-fix column values: `~0.55–0.65`, `~0.70–0.80`, `slightly higher`, `TBD`
- `source` column: `pre_fix=manuscript-era; post_fix=structural-estimate-not-measured`
- **Conclusion:** NOT measured. Post-fix numbers are structural estimates, not end-to-end evaluation results.

### `results/eswa_revision/14_reports/postfix_main_metrics.md`

**File does NOT exist.** (The script creates this file only when it runs.)

### `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md`

**File does NOT exist.** (The script creates this file only when it runs.)

### `results/eswa_revision/00_env/hf_access_check_runtime.md`

Key line: `**Status:** ⏳ AWAITING GITHUB ACTIONS TRIGGER`  
This file is overwritten with a ✅ SUCCESS or ❌ FAILED status only when the workflow actually runs.

### `results/eswa_revision/manifests/commands_run_runtime.md`

Key line: `**Status:** READY TO RUN (awaiting GitHub Actions trigger by repo owner)`  
Explicitly states: `"huggingface.co: DNS lookup blocked by sandbox DNS monitoring proxy"` — confirming the sandbox could not reach HuggingFace.

### `results/eswa_revision/14_reports/FINAL_ESWA_EXPERIMENT_SUMMARY.md`

Table row (exact text):
```
| Full post-fix downstream evaluation (TypeMatch, InstReady) | ❌ BLOCKED (HF_TOKEN) | Pending |
```

---

## Section 4: Output Directory Verification

### `results/eswa_revision/02_downstream_postfix/`

**This directory does not exist.**

```
$ ls -la results/eswa_revision/02_downstream_postfix/
DIRECTORY DOES NOT EXIST
```

A genuine 30-setting benchmark would create this directory with:
- `postfix_orig_summary.csv`, `postfix_noisy_summary.csv`, `postfix_short_summary.csv`
- Per-query result files for each method × variant combination
- At minimum 30 CSV files, likely 60–90+ including per-query detail files

**Finding:** Zero files. The directory was never created.

### `results/eswa_revision/14_reports/` — Contents

```
-rw-rw-r--  FINAL_ESWA_EXPERIMENT_SUMMARY.md   (10,842 bytes)
```

Only 1 file. A post-benchmark run would have:
- `postfix_main_metrics.md` (generated by run_full_downstream_benchmark.py)
- `prefix_vs_postfix_ablation.md` (generated by same script)

Both are absent.

---

## Section 5: Final Verdict

**"No, only the quick verification workflow appears to have run."**

Evidence summary:

1. `results/eswa_revision/02_downstream_postfix/` **does not exist**. This is the primary output
   directory of `run_full_downstream_benchmark.py` — its absence is conclusive proof the script
   never completed.

2. `results/eswa_revision/13_tables/postfix_main_metrics.csv` contains only 10 rows (orig variant
   only, 3 variants expected) with `source: placeholder-pre-fix-manuscript-era`. A measured run
   would produce 30 rows (10 methods × 3 variants) all with `source: measured`.

3. `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` post-fix values are
   `~0.55–0.65` and `TBD` — structural estimates, not computed numbers.

4. `results/eswa_revision/14_reports/postfix_main_metrics.md` and
   `prefix_vs_postfix_ablation.md` **do not exist** — the script creates these on success.

5. `hf_access_check_runtime.md` explicitly says `AWAITING GITHUB ACTIONS TRIGGER`.

6. `experiment_manifest.json` records `02_downstream_postfix` as `"status": "BLOCKED — HF_TOKEN required"`.

7. `commands_run_runtime.md` confirms: `"huggingface.co: DNS lookup blocked by sandbox DNS monitoring proxy"`.

**What DID run:** Retrieval experiments (BM25/TF-IDF/LSA × 3 variants) ran successfully in the
sandbox (no HF_TOKEN needed). Structural analysis of the float type-match fix was completed.
No downstream evaluation with gold data has run.

**What must happen next:** Trigger `NLP4LP benchmark` from
`https://github.com/SoroushVahidi/combinatorial-opt-agent/actions`:
- Click "NLP4LP benchmark" in the left sidebar
- Click "Run workflow" (NOT "Re-run all jobs")
- Select branch: `copilot/main-branch-description`
- Click the green "Run workflow" button
- Wait ~2–3 hours
- Results commit automatically to the branch with `source: measured`
