# GitHub Actions Workflow Audit

**Date:** 2026-03-10  
**Branch audited:** `copilot/main-branch-description`  
**Default branch:** `main` (commit `7ec73743`) — has NO `.github/workflows/` directory

---

## Short Answer

**Run this workflow: `NLP4LP benchmark`**  
**Branch to select: `copilot/main-branch-description`**  
**Button to click: "Run workflow" (not "Re-run all jobs")**

---

## All Workflow Files

### 1. `.github/workflows/nlp4lp.yml` — **THE ONE TO RUN**

| Property | Value |
|----------|-------|
| Workflow name | `NLP4LP benchmark` |
| Triggers | `workflow_dispatch` only |
| Job(s) | `build-benchmark` |
| Purpose | **Full pipeline**: verify HF → build eval sets → run all 30 downstream experiments → commit results |
| Expected runtime | ~2–3 hours |
| Uses HF_TOKEN | YES |
| Writes/commits results | YES (`contents: write`, `git push`) |
| Visible in Actions sidebar | **YES** (registered ID `244220556`) |

**Phases run by this workflow:**
1. **Phase 0:** `python training/external/verify_hf_access.py` — fast-fail if HF_TOKEN is invalid
2. **Phase 1:** `python training/external/build_nlp4lp_benchmark.py` — build eval sets from HF
3. **Phase 2:** `python training/external/run_full_downstream_benchmark.py` — all 30 experiment settings:
   - 9 methods × 3 variants (orig/noisy/short) + 3 random controls
   - Pre-fix vs post-fix ablation (4 methods × 3 variants, in-memory patch)
4. **Commit step:** commits measured results back to the branch
5. **Print step:** prints `postfix_main_metrics.csv` and `prefix_vs_postfix_ablation.csv`

---

### 2. `.github/workflows/downstream_benchmark.yml` — Standalone (secondary)

| Property | Value |
|----------|-------|
| Workflow name | `NLP4LP downstream benchmark (authenticated)` |
| Triggers | `workflow_dispatch` + `push` on path `.github/workflows/downstream_benchmark.yml` |
| Job(s) | `run-downstream-benchmark` |
| Purpose | Downstream benchmark only (skips the eval-set build phase) |
| Expected runtime | ~2–3 hours (when triggered via `workflow_dispatch`) |
| Uses HF_TOKEN | YES |
| Writes/commits results | YES |
| Visible in Actions sidebar | **After this PR push:** YES (push trigger on path) |

**Why it was previously invisible:**  
Only had `workflow_dispatch` trigger. GitHub only registers `workflow_dispatch` workflows
from the **default branch** for sidebar display. This file was only on the PR branch and had
never been triggered via `push`/`pull_request`, so GitHub had never registered it.

**Fix applied:** Added a `push` trigger restricted to `paths: [.github/workflows/downstream_benchmark.yml]`
on the PR branch. The benchmark steps are gated with `if: github.event_name == 'workflow_dispatch'`
so they are completely skipped when triggered by a push (only a `echo` step runs, costing <10 seconds).
Once this PR commit is pushed, GitHub registers the workflow and it appears in the sidebar.

---

### 3. `.github/workflows/check-hf-access.yml` — Quick smoke test

| Property | Value |
|----------|-------|
| Workflow name | `Check HF access` |
| Triggers | `workflow_dispatch` only |
| Job(s) | `verify` |
| Purpose | Smoke test: confirm HF_TOKEN is valid and `udell-lab/NLP4LP` is reachable |
| Expected runtime | ~60 seconds |
| Uses HF_TOKEN | YES |
| Writes/commits results | NO |
| Visible in Actions sidebar | NO (same reason as `downstream_benchmark.yml`) |

**Note:** Not critical — the same check is run as Phase 0 of `nlp4lp.yml` before the benchmark.

---

## Why `NLP4LP benchmark` Is Visible But Others Are Not

`nlp4lp.yml` was first pushed to this PR branch in commit `abd163c` with **both**
`workflow_dispatch` AND `push` triggers. GitHub registered it when the push event fired.
Later, the `push` trigger was removed, but GitHub keeps registered workflows in its
database permanently.

`downstream_benchmark.yml` and `check-hf-access.yml` were added with only `workflow_dispatch`
triggers and were never triggered via a `push`/`pull_request` event, so GitHub never
registered them.

**Root cause summary:**
> GitHub only discovers and registers `workflow_dispatch` workflows for sidebar display if they
> exist on the **default branch**. The default branch (`main`) has NO `.github/workflows/`
> directory — all workflow files are exclusively on the PR branch.
> `NLP4LP benchmark` is visible only because it was previously triggered by a push event.

---

## Exact Steps to Run the Full Benchmark

```
1. Open: https://github.com/SoroushVahidi/combinatorial-opt-agent/actions

2. In the left sidebar, click:  "NLP4LP benchmark"

3. On the right side of the run list, click the grey "Run workflow" button.
   (NOT "Re-run all jobs" — that would re-run the last run's job)

4. In the dropdown dialog that appears:
   - "Use workflow from" → select branch: copilot/main-branch-description
   - No other inputs to fill (this workflow has no manual inputs)

5. Click the green "Run workflow" button.

6. The workflow will check out the PR branch version of nlp4lp.yml
   (which has all 3 phases including the full downstream benchmark).

7. When complete (~2-3 hours):
   - Results are committed to the branch automatically
   - Artifacts are uploaded with 90-day retention
   - Run summary shows the CSV tables in the job log
```

**⚠️ IMPORTANT — Do NOT use "Re-run all jobs":**  
"Re-run all jobs" re-runs the last completed run on the same SHA/branch it used before.  
If the last run was triggered by a push event (old behavior) on an older commit,  
it will re-run that old job rather than the new full pipeline.  
Always use the "Run workflow" button.

---

## Expected Output Files (after successful run)

All result rows have `source: measured` — no manuscript-era estimates:

| File | Description |
|------|-------------|
| `results/eswa_revision/13_tables/postfix_main_metrics.csv` | Aggregate metrics, all 30 settings |
| `results/eswa_revision/13_tables/prefix_vs_postfix_ablation.csv` | Pre/post TypeMatch delta |
| `results/eswa_revision/14_reports/postfix_main_metrics.md` | Markdown report |
| `results/eswa_revision/14_reports/prefix_vs_postfix_ablation.md` | Ablation markdown |
| `results/eswa_revision/02_downstream_postfix/` | Per-query CSVs + JSON (30+ files) |
| `results/eswa_revision/03_prefix_vs_postfix/` | Pre-fix simulation outputs |
| `results/paper/nlp4lp_downstream_summary.csv` | Master summary |
| `results/paper/nlp4lp_downstream_types_summary.csv` | Per-type breakdown |
| `results/eswa_revision/00_env/nlp4lp_gold_cache.json` | HF gold params (cached) |
