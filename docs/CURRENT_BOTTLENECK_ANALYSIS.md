# Current Bottleneck Analysis

**Scope:** `tfidf_typed_greedy`, `orig` variant, 331 queries — the canonical
method (see `PROJECT_STATUS.md` §3). All counts below are **derived directly**
from `results/eswa_revision/16_error_analysis/per_instance_diagnostics.csv`
(331 real per-query rows) computed in this pass; none are invented or copied
from the older, now-outdated `error_taxonomy_counts.csv`, which documents the
**pre-fix** state (before the `_is_type_match` correction) and should not be
used for current bottleneck claims — its dominant "float type mismatch (~230
cases)" finding is largely resolved (TypeMatch rose from ~0.23 to 0.7453
after the fix; see `docs/METHOD_INVENTORY.md` for the fix itself).

**2026-08-12 (Phase 3) update, RETRACTED same day (Phase 4):** this note
originally claimed `max_weight_matching` "closes most of the gap this
table documents" and superseded `typed_greedy` as the canonical method.
That claim compared `max_weight_matching`'s fresh number against a
typed-greedy baseline (0.5287) that turned out to be **stale relative to
current code** — a fresh rerun of plain typed greedy gives 0.7764, which
significantly *beats* `max_weight_matching` (0.7432, p=0.042). Full
correction: `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. Typed greedy
is **not** superseded; it remains the strongest known non-oracle method.

**Open question this raises, not yet resolved (2026-08-12, Phase 4):**
the `per_instance_diagnostics.csv` this document's counts are derived from
was itself generated against `typed_greedy` at some earlier codebase
state, and has **not** been re-verified to reproduce from current code the
way `postfix_main_metrics.csv` was found not to (see the staleness audit).
It is plausible (though not confirmed) that the specific bottleneck
*counts* below (82/331 type mismatch, etc.) are also somewhat stale, even
though the file's own retrieval numbers (Schema R@1 ≈0.91) are consistent
with the fresh, current-code family. Treat the *ranking* of bottleneck
categories below as directionally informative but the exact counts as
unverified against current code until someone re-derives them fresh (a
concrete next step — see `NEXT_STEPS.md`).

---

## Ranked bottleneck table (current, post-fix)

| Rank | Failure mode | Evidence | Count / 331 | Impact on InstantiationReady | Current mitigation | Remaining gap |
|---|---|---|---|---|---|---|
| 1 | **Type mismatch on an otherwise-fully-covered decision** (schema hit, all slots filled, ≥1 wrong type) | `per_instance_diagnostics.csv`: `schema_hit=1 & inst_ready=0 & any_type_mismatch=1 & full_coverage=1` | 82 (24.8% of all queries; 60.7% of hit-but-not-ready cases) | Directly zeroes `inst_ready` for these queries despite full coverage | The `_is_type_match` fix already resolved the *dominant pre-fix* form of this (int/float pairs); this is the *residual* post-fix type-mismatch population | **Largest open bottleneck.** None of the evaluated richer deterministic methods (see `docs/NEGATIVE_RESULTS.md`) reduce this; motivates the learned local scorer (H1/H2 in `RESEARCH_HYPOTHESES.md`) |
| 2 | **Combined type mismatch + incomplete coverage** | same file: `schema_hit=1 & inst_ready=0 & any_type_mismatch=1 & full_coverage=0` | 30 (9.1% of all queries; 22.2% of hit-but-not-ready cases) | Compounds two failure sources on the same query | None specific — same repair machinery as ranks 1 and 3 | Likely the hardest subset (two independent things must both be fixed) |
| 3 | **Schema retrieval miss** | `schema_hit=0` | 30 (9.1% of all queries) | InstReady forced to 0 regardless of downstream quality (retrieval-dependent eligible-slot design) | TF-IDF R@1 already ≈0.91; Oracle control isolates this as a *small* contributor to the overall gap (Oracle InstReady 0.5680 vs TF-IDF 0.5287, a 0.039 gap — see `docs/CANONICAL_RESULTS.md` §E) | Diminishing returns — retrieval is already comparatively strong; this is not the dominant lever |
| 4 | **Coverage gap only (missing slots, correct types on filled ones)** | `schema_hit=1 & inst_ready=0 & any_type_mismatch=0 & full_coverage=0` | 22 (6.6% of all queries; 16.3% of hit-but-not-ready cases) | Directly reduces Coverage and zeroes `inst_ready` | Numeric/word-number/enumeration extraction stages (`docs/METHOD_INVENTORY.md` Part 1, stages 3-5) | Likely extraction misses (mention never found) rather than mis-assignment — a *different* fix target than ranks 1-2 |
| 5 | **Zero-expected-scalar-slot queries** | `n_expected_scalar=0` | 20 (6.0% of all queries) | These queries structurally cannot be "ready" in the usual sense (edge case in the eligible-slot computation) | None identified | **Root-caused 2026-08-12 (Phase 3): NOT an independent failure mode.** 19/20 (95%) have `schema_hit=0` -- `n_expected_scalar` is computed from the query's *predicted* (not gold) schema's own gold-parameter structure (`_run_setting`'s `pred_scalar_keys` logic), so a wrong retrieval can point to a schema whose own template/gold-value structure happens to yield zero fillable scalar slots. This is a **downstream consequence of rank-3 (schema miss)**, not a fifth independent bottleneck. Only 1/20 is genuine: `nlp4lp_test_293`, a multi-industry manpower-allocation problem whose parameters are almost entirely vector/matrix-valued (`ManpowerOne`, `Stock`, `Capacity`, `InputOne`, etc.) -- a real instance of the "scalar-only grounding" architectural weakness (§A above), not a metric bug. No canonical numbers were regenerated; this is a correctly-computed, now-explained interaction, not an evaluation bug. |

**Sanity check:** ranks 1+2+4 (135) + rank 3 (30) accounts for 165/331
(49.9%) of all queries as some form of non-readiness; the remaining ~50%
(166/331) are `inst_ready=1`, consistent with InstantiationReady ≈ 0.5287
after accounting for the `orig` vs. Strict distinction (see
`results/CANONICAL_RESULTS.md` §D).

**Not independently re-derived in this pass** (would require re-reading
per-slot type-mismatch categories, not just the aggregate flag): the finer
breakdown by numeric type (percent/float/currency/count) that the pre-fix
`error_taxonomy.md` attempted. If a future agent needs that granularity,
compute it fresh from `per_instance_diagnostics.csv` or a per-slot log
rather than reusing the pre-fix percentages, which are not comparable
post-fix.

---

## Three types of weakness

### A. Architectural (would persist even with a perfect benchmark/perfect evaluation)

| Weakness | Where it shows up |
|---|---|
| Hand-engineered pairwise mention-slot scoring (`_score_mention_slot` and its IR/opt-role variants) | Root cause of rank-1/2 bottleneck above; every richer *deterministic* variant of this scoring also failed (`docs/NEGATIVE_RESULTS.md`) |
| Coarse four-way-ish numeric type system (`_expected_type`, `_is_type_match`) | Cannot distinguish semantically different quantities that share a coarse type (e.g., two floats with different units/roles) |
| Context-window-dependent, lexicon-based semantics (role/unit/operator tag detection) | Brittle to unseen phrasing outside the implemented cue vocabulary |
| Scalar-only grounding | Cannot represent vector/matrix-valued parameters at all |
| Fixed top-1 retrieval before grounding | Retrieval errors (rank 3 above) are unrecoverable downstream by construction; joint top-k reranking is untested (H5) |

### B. Benchmark / data

| Weakness | Where it shows up |
|---|---|
| Fixed schema catalog (331/335 entries) | Cannot generalize to problem types outside the catalog; not an algorithmic limitation |
| Gated dataset (`udell-lab/NLP4LP`) | Reproduction requires HF approval; local gold cache mitigates but doesn't eliminate this |
| Single primary benchmark (NLP4LP) | No cross-benchmark generalization evidence yet (Text2Zinc/CP-Bench are adapter-only, not camera-ready) |
| 331-vs-335 catalog-vintage offset in Schema R@1 | A measurement artifact, not a model weakness — see `results/CANONICAL_RESULTS.md` §A |
| Restricted structural/solver-backed subsets (60, 269, 20 instances) | Not benchmark-wide solver-readiness evidence, by the manuscript's own explicit framing |

### C. Evaluation / evidence

| Weakness | Where it shows up |
|---|---|
| No strong learned contextual grounding baseline exists yet | This is the literal gap `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` P0 addresses |
| Limited solver-backed scale (20/331) | Compatibility-filtered, not randomly sampled — explicit manuscript caveat |
| No broad comparison against newest LLM-based auto-formulation methods | ORLM/OptMATH/DeepOR/OR-R1 are all implemented/reconstructed for lightweight inference preparation but none has a runnable empirical result (checkpoint or environment blocked in every case) (`PROJECT_STATUS.md` §9) |
| PaMOP reproduction fidelity uncertain | 6/6 execution success but only 1/6 semantic correctness on the pilot — see `PROJECT_STATUS.md` §10 |
| Three grounding-method families implemented but never evaluated | `max_weight_matching`, `search_structured_grounding*`, `hierarchical_structured_grounding*` — see `docs/METHOD_INVENTORY.md` |

---

## Positive properties (balanced strengths — verified, not oversold)

- **No external generative-LLM API required at inference** for the core
  canonical pipeline (`typed_greedy` and all evaluated grounding variants
  in `docs/METHOD_INVENTORY.md` Part 2 are classical IR + deterministic
  rule-based grounding; OpenAI/Gemini/Mistral paths are optional auxiliary
  baselines, not part of the evaluated method). This should be preserved by
  any future learned-scorer addition (§P0 in the roadmap explicitly targets
  a **local** model, not an API call).
- **Deterministic and reproducible inference** — every evaluated method is
  seed-free/deterministic by construction (no sampling in the core
  pipeline); results are exactly reproducible given the same catalog and
  gold cache.
- **CPU-friendly core pipeline** — TF-IDF/BM25/LSA retrieval and rule-based
  grounding do not require a GPU; verified no GPU dependency in the
  evaluated method set (dense retrieval baselines SBERT/E5/BGE are
  supplementary only, not part of the headline pipeline).
- **Interpretable schema retrieval and assignment** — every intermediate
  decision (mention extraction, slot scoring, assignment, repair) is
  inspectable; this is an explicit manuscript contribution claim, and the
  per-query diagnostic CSVs used in this analysis are a direct consequence
  of that interpretability.
- **Explicit structural verification stage** (`formulation/verify.py`)
  independent of the grounding method, catching a distinct error class.
- **Modularity** — retrieval, extraction, scoring, and assignment are
  separable stages (`docs/METHOD_INVENTORY.md` Part 1), which is exactly
  what makes a targeted local-scorer replacement (touching only stage 8-12)
  feasible without rearchitecting the rest of the pipeline.
- **Strong fixed-catalog retrieval** — TF-IDF R@1 ≈ 0.91, not the
  bottleneck (rank 3 above is comparatively minor next to ranks 1-2).
- **Can run in restricted/offline environments** once the catalog and a
  local NLP4LP gold cache exist (verified: `results/eswa_revision/00_env/nlp4lp_gold_cache.json`
  is present locally) — no network calls needed for the core benchmark.

**Explicit non-overclaims:** this is not "no AI" (TF-IDF/BM25/LSA and the
optional dense baselines are themselves NLP techniques); it does not imply
superiority over full-model-generation systems on tasks this pipeline does
not attempt (e.g., open-ended constraint synthesis, which PaMOP-style
methods target and this pipeline explicitly does not).
