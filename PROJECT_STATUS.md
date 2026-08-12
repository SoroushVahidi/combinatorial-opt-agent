# Project Status

**Last verified:** 2026-08-11 (Phase 1 repository-polish audit). This file is the
primary entry point for a new agent or contributor. It supersedes older
"start here" documents for orientation purposes — see [`docs/KAIS_SOURCE_OF_TRUTH.md`](docs/KAIS_SOURCE_OF_TRUTH.md)
for manuscript-specific authority and [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) /
[`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) for the full directory map.

---

## 1. Scientific Goal

This repository studies **retrieval-assisted instantiation of natural-language
optimization problems**: given a natural-language description of an optimization
problem, (a) retrieve the most compatible schema from a fixed catalog of
optimization problem templates, then (b) deterministically ground the schema's
scalar parameters from numeric evidence in the text. The manuscript frames this
as a knowledge-processing problem, not a full NL-to-solver compiler.

## 2. Current Pipeline

```
natural-language query
  → schema retrieval (TF-IDF / BM25 / LSA / Oracle control)
  → numeric mention extraction
  → schema-conditioned scalar grounding (typed greedy + extensions, see §5)
  → structural verification (formulation/verify.py, no live solver)
  → restricted solver-backed validation (SciPy HiGHS shim, 20-instance subset)
```

Core implementation: [`tools/nlp4lp_downstream_utility.py`](tools/nlp4lp_downstream_utility.py)
(retrieval-to-grounding pipeline); retrieval baselines in [`retrieval/`](retrieval/);
structural checks in [`formulation/verify.py`](formulation/verify.py).

## 3. Current Authoritative Results

**Primary benchmark:** NLP4LP (`udell-lab/NLP4LP`, gated HuggingFace dataset),
`orig` variant, 331 test queries.

**IMPORTANT — a real staleness issue was found and is corrected here.** Four
documents (`docs/EAAI_SOURCE_OF_TRUTH.md`, `docs/CURRENT_STATUS.md`,
`docs/RESULTS_PROVENANCE.md`, `README.md`) and one result file
(`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`)
cite numbers from a **stale intermediate significance snapshot**. The
manuscript itself (`manuscript/main.tex`, §"Strict InstantiationReady..."
discussion) explicitly documents this: the downstream table was found during
final KAIS preparation to have been populated from a stale snapshot and was
regenerated from live per-query artifacts. **The numbers below are the
corrected, currently-in-manuscript values** — use these, not the CSV listed
above, until that CSV is regenerated.

| Method | Schema R@1 | Coverage | TypeMatch | Exact20_on_hits | InstantiationReady |
|---|---|---|---|---|---|
| TF-IDF (typed greedy) | 0.9094 | 0.8609 | 0.7453 | 0.1834 | 0.5287 |
| BM25 (typed greedy) | 0.8822 | 0.8509 | 0.7336 | 0.1884 | 0.5196 |
| LSA (typed greedy) | 0.8459 | 0.8267 | 0.7054 | 0.1822 | 0.5076 |
| Oracle (typed greedy) | 1.0000 | 0.9151 | 0.7998 | 0.1745 | 0.5680 |

Source of these corrected numbers: `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv`
and `results/eswa_revision/13_tables/postfix_main_metrics.csv` (both currently
"live"/non-`.stale` files), cross-checked directly against `manuscript/main.tex`.

**StrictInstantiationReady** (adds a schema-match gate; from
`results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv`,
also matches the manuscript):

| Method | InstantiationReady | StrictInstantiationReady |
|---|---|---|
| TFIDF-TG | 0.5287 | 0.5045 |
| BM25-TG | 0.5196 | 0.4924 |
| LSA-TG | 0.5076 | 0.4864 |
| Oracle-TG | 0.5680 | 0.5680 |

**A disclosed, unresolved, small Schema R@1 offset exists** and should not be
"fixed" without more investigation: some diagnostic tables/scripts report
TF-IDF Schema R@1 = 0.9063 (not 0.9094) and BM25 = 0.8852 (not 0.8822). The
manuscript explicitly discloses this (§ "overlap analysis") and attributes it
to a 335- vs original-catalog-vintage document-count difference between the
diagnostic script and the canonical Table run. **Do not silently pick one
value** — report which script/table produced the number you cite.

**Other result families** (not found stale in this pass, but not independently
re-verified line-by-line either): engineering structural subset (60
instances), executable-attempt subset (269 instances), final solver-backed
subset (20 instances) — see `results/paper/eaai_camera_ready_tables/table2-5*.csv`
and `docs/EAAI_SOURCE_OF_TRUTH.md`'s "Key Metrics" section for those tables.

## 4. Main Scientific Finding

Retrieval is strong (TF-IDF Schema R@1 ≈ 0.91); the oracle-vs-TF-IDF gap on
InstantiationReady is modest (0.5680 vs 0.5287). **The primary bottleneck is
downstream semantic number-to-schema-slot grounding, not schema retrieval.**

## 5. Current Grounding Methods Already Implemented

Do **not** reinvent these — they exist and have been benchmarked (see
`results/eswa_revision/02_downstream_postfix/` for per-method result files):

- typed greedy (baseline)
- constrained matching (`tfidf_constrained`)
- semantic IR repair (`tfidf_semantic_ir_repair`)
- optimization-role repair (`tfidf_optimization_role_repair`)
- acceptance reranking (`tfidf_acceptance_rerank`)
- hierarchical acceptance reranking (`tfidf_hierarchical_acceptance_rerank`)
- global compatibility grounding / GCG (`tfidf_global_compat_{local,pairwise,full}`)
- relation-aware linking / RAL (`tfidf_relation_aware_{basic,ops,semantic,full}`)
- ambiguity-aware grounding / AAG (`tfidf_ambiguity_{candidate_greedy,aware_abstain,aware_beam,aware_full}`)
- maximum-weight bipartite matching (used inside global-compatibility grounding)
- structural validation/repair (`formulation/verify.py`)

## 6. What Did NOT Work / Negative Results

**Important, do not re-attempt these expecting a different outcome without new
evidence:** none of the richer deterministic grounding families (GCG,
relation-aware, ambiguity-aware) beat plain `tfidf_typed_greedy` on `orig`
InstantiationReady. Best new representative (`tfidf_relation_aware_basic`)
reaches 0.4985, still below TFIDF-TG's 0.5287; GCG's best variant falls to
0.4230-0.4320; the abstention-heavy ambiguity variant becomes overly
conservative (0.0272). This is a statistically supported negative result
(paired bootstrap significance tests; TF-IDF significantly beats GCG and AAG
representatives at p<0.001) documented in the manuscript and in
`results/eswa_revision/15_significance/SIGNIFICANCE_SUMMARY.md`.

Also: learned retrieval fine-tuning does not beat the rule-based retrieval
baseline on held-out eval (`docs/KNOWN_ISSUES.md`).

## 7. Current Weaknesses

- Scalar-only grounding (no vector/matrix-valued parameters)
- Coarse numeric typing
- Hand-engineered, context-window-based semantics (no learned contextual scorer)
- Fixed schema catalog (no open-vocabulary schema discovery)
- Benchmark dependence (NLP4LP-specific tuning risk)
- Limited semantic correctness even when structurally valid (see PaMOP
  forensics below for a stark example: 6/6 executable, but only 1/6 semantically
  correct on a separate baseline-reproduction pilot)
- Limited solver-backed scale (20/331 instances)
- No strong learned contextual grounding baseline yet (see §8)
- No broad end-to-end comparison yet against newest LLM-based methods
  (ORLM, OptMATH, DeepOR, OR-R1 — see §9)

## 8. Most Promising Improvement Direction

**Established technique to test before inventing new algorithms:** a learned
**local** contextual mention-slot scorer or cross-encoder, layered on top of
the *existing* global assignment (bipartite matching / GCG) and structural
verification — not a replacement for them. This can remain a local model and
does **not** require an external generative-LLM API at inference time,
preserving the pipeline's current no-external-LLM-at-inference property.

## 9. External Baseline Roadmap

| Baseline | Status |
|---|---|
| **PaMOP** (IJCAI 2025) | **IN PROGRESS** — independent reproduction, no official code available. See §10. |
| **ORLM** | **NOT STARTED** — no `baselines/orlm/` directory exists |
| **OptMATH** | **NOT STARTED** — no `baselines/optmath/` directory exists |
| **DeepOR** | **NOT STARTED** — no `baselines/deepor/` directory exists |
| **OR-R1** | **NOT STARTED** — no `baselines/or_r1/` directory exists |

Verified by directory listing: `baselines/` currently contains only `baselines/pamop/`.

## 10. PaMOP Reproduction Status

**Implemented stages** (verified present in `baselines/pamop/`): LLM extraction
(`G_extr`), partition tree (`partition.py`), self-augmented modeling merge
(`G_mod`, bottom-up), AMPL renderer/executor (`ampl_interface.py`), and the
`G_exe`/`G_rev`/`G_comp`/`G_remod` correction loop (`correction.py`).

**Known reproduction choices / gaps:**
- No official PaMOP code was found; this is an independent reproduction.
- Current LLM: `gpt-4.1-mini-2025-04-14` via Azure OpenAI (exact original
  PaMOP GPT-4 model/prompts are unknown/unavailable).
- The exact PaMOP 67-problem subset identity is **unresolved** — see
  `docs/PAMOP_REPRODUCTION_PLAN.md` §13.4-13.7 for the detailed investigation;
  a deterministic stratified 6-problem pilot subset was used instead for the
  forensics pass (problem IDs 14, 23, 34, 72, 84, 88).
- A major input-source/systematic bug was found and fixed prior to the
  current results (see `docs/PAMOP_PILOT_FAILURE_FORENSICS.md`).

**Current results** (verified against `results/pamop/forensics_targeted/summary.json`):
initial execution success 2/6 (0.333), final execution success 6/6 (1.0) after
the correction loop, semantic correctness 1/6 (0.167), mean correction
iterations = 1.0 among the 4 corrected problems (0.67 averaged across all 6),
total tokens = 24,194, decision gate = **"A. PROCEED TO LARGER RUN"**.

**Current next action:** proceed to a larger PaMOP reproduction run per the
decision gate above, while continuing to treat semantic-correctness fidelity
(1/6) as the dominant open concern, not execution success (6/6).

## 11. Data / Solver / API Environment

High-level only, no secrets:

- **NLP4LP**: gated HuggingFace dataset (`udell-lab/NLP4LP`); a local gold
  cache exists (`results/eswa_revision/00_env/nlp4lp_gold_cache.json`); no
  `HF_TOKEN` needed to read that cache, but a token is needed to re-pull fresh.
- **Gurobi**: a license is present on this workstation
  (`~/gurobi.lic`, outside the repo) and `gurobipy` is importable, but **only
  inside a dedicated virtualenv** (`~/.venvs/gurobi`, referenced via the
  `PAMOP_AMPLPY_PYTHON` env var in `baselines/pamop/README.md`) — **not** in
  the default `python3` / repo `venv`. The main paper's solver-backed subset
  intentionally uses a SciPy HiGHS shim instead and does not require Gurobi.
- **AMPL / amplpy**: same dedicated virtualenv as Gurobi above; verified
  importable there, not importable in the default environment.
- **Azure OpenAI**: working per PaMOP pilot/forensics runs (see §10); no
  further detail here (no secrets).
- **Other optional LLM providers**: OpenAI, Gemini, Mistral wiring exists for
  auxiliary (non-paper-core) reruns — see `docs/GEMINI_RERUN_REPORT.md`,
  `docs/MISTRAL_RERUN_REPORT.md`. Mistral Wulver jobs 902367/902368 did not
  produce output (missing `MISTRAL_API_KEY` in job env, per
  `docs/provenance/mistral_wulver_submission_2026-04-03.md`) — treat as
  unresolved infra, not a completed rerun.

## 12. Canonical Files

- **Main downstream tool:** `tools/nlp4lp_downstream_utility.py`
- **Retrieval:** `retrieval/search.py`, `retrieval/baselines.py`
- **Structural verification:** `formulation/verify.py`
- **Manuscript source (authoritative):** `manuscript/main.tex` (compiles to
  `manuscript/main.pdf`, 36 pages) — see `docs/KAIS_SOURCE_OF_TRUTH.md`
- **Corrected downstream results (use these, see §3):**
  `results/eswa_revision/14_reports/downstream_comparison_all_methods.csv`,
  `results/eswa_revision/13_tables/postfix_main_metrics.csv`,
  `results/eswa_revision/18_strict_instready/strict_instantiation_ready.csv`
- **Stale downstream results (do not cite without the §3 caveat):**
  `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
- **PaMOP implementation:** `baselines/pamop/` (see `baselines/pamop/README.md`)
- **PaMOP results:** `results/pamop/pilot/`, `results/pamop/forensics_targeted/`
- **Key docs:** `docs/KAIS_SOURCE_OF_TRUTH.md` (manuscript authority),
  `docs/REVIEWER_GUIDE.md` (reviewer orientation), `docs/KNOWN_ISSUES.md`,
  `docs/HOW_TO_REPRODUCE.md`, `docs/PAMOP_REPRODUCTION_PLAN.md`

## 13. Immediate Next Steps

**P0 (repository hygiene, this pass):**
- Regenerate `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
  from the corrected canonical per-query artifacts (or clearly mark it superseded)
  so a fifth stale copy doesn't get created by a future agent trusting the
  filename over the content.

**P1 (PaMOP, per its own decision gate):**
- Proceed to a larger PaMOP reproduction run (decision gate: "PROCEED TO
  LARGER RUN"), while explicitly tracking semantic-correctness rate, not just
  execution success rate.
- Continue investigating the exact PaMOP 67-problem subset identity if it
  becomes load-bearing for a fidelity claim.

**P2 (algorithm improvement):**
- Prototype the local learned contextual mention-slot scorer described in §8,
  evaluated against the existing global-matching + verification pipeline,
  before attempting further deterministic grounding variants (§6 already
  shows those plateau).

**P3 (baseline coverage):**
- Begin ORLM / OptMATH / DeepOR / OR-R1 baseline scaffolding, following the
  `baselines/pamop/` structure as a template.

## 14. Things a New Agent Must NOT Do

- Do not treat archived metrics under `docs/archive/`, `docs/archive_internal_status/`,
  `docs/provenance/`, or `results/eswa_revision/` (except the specific files
  named in §3/§12 as corrected sources) as current authoritative numbers.
- Do not reinvent global bipartite matching, GCG, relation-aware linking, or
  ambiguity-aware grounding — they exist and have been benchmarked (§5), with
  documented negative results (§6).
- Do not claim exact PaMOP reproduction — this is an independent, admittedly
  imperfect reproduction with named unresolved details (model, prompts,
  67-problem subset identity).
- Do not expose gated NLP4LP data or redistribute it outside HuggingFace's
  own access-control terms.
- Do not expose API keys, tokens, or the Gurobi/AMPL license contents in
  commits, logs, or chat.
- Do not run large/expensive experiments (full PaMOP larger run, full LLM
  provider reruns, full NLP4LP re-benchmark) before validating on a small
  pilot first, matching the existing pattern (`results/pamop/pilot/` before
  a larger run).
- Do not modify canonical result artifacts without recording provenance
  (generating script/commit, why the change was made) — follow the existing
  `.stale`-suffix convention (e.g. `downstream_comparison_all_methods.csv.stale`)
  when superseding a file rather than silently overwriting it.
- Do not modify the manuscript's scientific claims without separately
  verifying the underlying data supports the change (Phase 1 of this
  repository-polish effort explicitly did not touch manuscript claims).
