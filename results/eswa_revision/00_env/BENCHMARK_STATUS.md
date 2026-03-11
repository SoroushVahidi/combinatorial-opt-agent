# Benchmark Execution Status — Evidence-Based Verification

**Date of inspection:** 2026-03-10  
**Branch inspected:** `copilot/experiment-overview`  
**Short answer:** `NLP4LP downstream benchmark HAS run. Run 22922351003 completed at 2026-03-10T20:18:27Z. All result files have source: measured.`

> **UPDATE (2026-03-10T20:40Z):** The benchmark ran successfully via `downstream_benchmark.yml` (`workflow_dispatch`) as run `22922351003`. All 27 post-fix settings and 12 pre-fix ablation settings completed in 32 seconds. Results were committed to `copilot/main-branch-description` and imported here. See `docs/CI_ROOT_CAUSE_AUDIT.md` for the full investigation.

---

## Run workflow button — Root cause and fix

### What happened: fast (< 20 second) workflow completions

All runs you saw in the Actions tab were triggered by `push` events (git pushes to the branch),
**not** by the "Run workflow" button. Those push-triggered runs were deliberately designed to
exit immediately in ~20 seconds — they just printed a message and did nothing.

This was a mistake: the push triggers were added to "register" the workflows with GitHub so the
"Run workflow" button would appear. But the fast exit runs caused confusion because they showed
as green ✅ in ~20 seconds, making it look like the benchmark had completed.

### Fix applied

All three workflow files have been simplified:
- **Removed** the `push` triggers
- **Removed** the "Skip on push (registration-only run)" early-exit steps
- **Removed** all `if: github.event_name == 'workflow_dispatch'` guards

Now every run of each workflow is a real run — no more fast-exit placeholder runs.

### How to trigger the actual benchmark

The "Run workflow" button works for any branch that contains the workflow file with a
`workflow_dispatch` trigger. Since the workflows are on `copilot/main-branch-description`,
the button works right now without merging:

```
1. Go to: https://github.com/SoroushVahidi/combinatorial-opt-agent/actions
2. In the left sidebar, click "Check HF access" FIRST (quick test, ~60s)
   → Verifies your HF_TOKEN is configured correctly
   → If this fails: Settings → Secrets → Actions → add HF_TOKEN = hf_...
3. Once HF access is confirmed, click "NLP4LP downstream benchmark (authenticated)"
4. Click the "Run workflow" button (top-right of the run list)
5. In the dropdown, select branch: your branch
6. Click the green "Run workflow" button
7. Runtime: ~3 minutes (32 s benchmark loop + pip install)
8. Results committed automatically to the branch with source: measured
```

### If the "Run workflow" button is not visible

Use the GitHub CLI to trigger directly:

```bash
gh workflow run nlp4lp.yml \
  --repo SoroushVahidi/combinatorial-opt-agent \
  --ref copilot/main-branch-description
```

---

## Section 1: Workflow Verification

### `.github/workflows/nlp4lp.yml` — Name: `NLP4LP benchmark`

- **Triggers:** `workflow_dispatch` only (push triggers removed)
- **Jobs:** `build-benchmark` (single job, 5h timeout)
- **Exact run commands:**
  ```bash
  python training/external/verify_hf_access.py
  python training/external/build_nlp4lp_benchmark.py
  python training/external/run_full_downstream_benchmark.py
  ```
- **Classification:** REAL FULL DOWNSTREAM BENCHMARK RUNNER (all 3 phases)
- **2-minute run plausible?** YES — measured runtime is ~3 min total (32 s benchmark + pip install)
- **Note:** `--variants orig,noisy,short` was added to the `build_nlp4lp_benchmark.py` call.
- **Uses HF_TOKEN:** YES (Phase 0, 1, 2)
- **Writes/commits results:** YES (`contents: write` permission, `git push`)
- **Workflow ID:** 244220556

### `.github/workflows/downstream_benchmark.yml` — Name: `NLP4LP downstream benchmark (authenticated)`

- **Triggers:** `workflow_dispatch` only (push triggers removed)
- **Jobs:** `run-downstream-benchmark` (single job, 4h timeout)
- **Exact run commands:**
  ```bash
  python training/external/verify_hf_access.py
  python training/external/run_full_downstream_benchmark.py
  ```
- **Classification:** FULL DOWNSTREAM BENCHMARK (skips eval-set build; assumes data/processed/ already populated)
- **✅ CANONICAL WORKFLOW** — use this for future reruns. Measured runtime: ~3 min total.
- **2-minute run plausible?** YES — the benchmark loop itself takes 32 s
- **Uses HF_TOKEN:** YES
- **Writes/commits results:** YES
- **Workflow ID:** 244290125

### `.github/workflows/check-hf-access.yml` — Name: `Check HF access`

- **Triggers:** `workflow_dispatch` only (push triggers removed)
- **Jobs:** `verify` (single job, no timeout)
- **Exact run commands:**
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
- **Real full run duration:** ~32 seconds for the benchmark loop; ~3 min total job (including pip install)

---

## Section 3–5: Historical Pre-Run Record (preserved for audit trail)

> **These sections were written before the benchmark ran. They are preserved as an audit trail only.**
> **Current state: all downstream result files are real measured outputs. See header above.**

### Historical: `results/eswa_revision/13_tables/postfix_main_metrics.csv`

*Before run 22922351003:* all 10 rows had `source: placeholder-pre-fix-manuscript-era`.  
*After run 22922351003:* 27 rows (9 methods × 3 variants), all `source: measured`.

### Historical: `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv`

*Before run 22922351003:* post-fix values were structural estimates (`post_fix=structural-estimate-not-measured`).  
*After run 22922351003:* 12 rows (4 methods × 3 variants), all `source: measured`.

### Historical: Output directory state

*Before run 22922351003:* `results/eswa_revision/02_downstream_postfix/` did not exist.  
*After run 22922351003:* 59 files in that directory (per-query CSVs + per-variant JSONs).

### Historical: Final verdict (pre-run)

*Before run 22922351003:* Only the quick verification workflow had run. The downstream benchmark was blocked by HF_TOKEN in the sandbox. The fix was to trigger `downstream_benchmark.yml` via `workflow_dispatch` (not push). That was done on 2026-03-10T20:15Z and succeeded in 32 s.

