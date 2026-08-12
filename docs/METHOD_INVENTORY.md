# Method Inventory: Algorithm Decomposition and Grounding Methods

**Purpose:** (1) a technically accurate decomposition of the current pipeline
stage-by-stage, and (2) a definitive list of every grounding method already
implemented, so no future agent reinvents one. Verified 2026-08-11 by
reading `tools/nlp4lp_downstream_utility.py` (6,977 lines) and its satellite
modules directly, and cross-checking evaluated-method claims against actual
result files in `results/eswa_revision/02_downstream_postfix/`.

**2026-08-12 (Phase 4) correction — READ THIS FIRST if you are relying on
any InstReady number below.** Phase 3 (2026-08-12, earlier the same day)
declared `max_weight_matching`/`search_structured_grounding`/
`hierarchical_structured_grounding` the strongest methods ever found in
this repository by comparing their freshly-measured numbers against the
committed `tfidf_typed_greedy` = 0.5287. That baseline number turned out to
be **stale relative to the current codebase** — 49 commits of grounding
fixes landed after it was generated and it was never regenerated. A fresh,
same-day, same-code rerun of plain typed greedy gives **0.7764**, which
*beats* all three "strong positive result" methods, significantly on
`orig` (p<0.05 paired bootstrap). Full audit:
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. The three methods are
**demoted back to NEGATIVE_RESULT** below; treat any "CANONICAL — STRONG
POSITIVE RESULT" language you see elsewhere in this repository (dated
2026-08-12 but written *before* this correction) as superseded by this
note and the audit document.

---

## Part 1: Pipeline stage decomposition

| # | Stage | Input | Output | Technique | Learned/Deterministic | Known limitations |
|---|---|---|---|---|---|---|
| 1 | Schema catalog construction | Raw NLP4LP problem templates | Fixed catalog of schema entries (id, text, expected scalar slots) | Static JSONL catalog (`data/catalogs/nlp4lp_catalog.jsonl`) | Deterministic, fixed | Fixed catalog — cannot generalize to unseen problem types without catalog expansion |
| 2 | Schema retrieval | NL query text | Top-1 (or top-k) schema id | TF-IDF / BM25 / LSA cosine similarity; Oracle control (gold schema, upper bound); dense baselines (SBERT/E5/BGE) supplementary only | Deterministic (classical IR); dense variants use pretrained (not fine-tuned for this task) embeddings | Fixed top-1 before grounding (§9 lists top-k joint reranking as a candidate improvement, not yet default) |
| 3 | Numeric mention extraction | Query text, variant (`orig`/`noisy`/`short`) | List of `NumTok`/`MentionRecord` (value, kind, span, context) | Regex-based digit extraction (`_parse_num_token`, `_extract_num_tokens`) | Deterministic | Coarse; misses non-standard numeric phrasing outside the implemented patterns |
| 4 | Word-number extraction | Tokenized query | Parsed numeric value from words ("twenty" → 20) | `_word_to_number`, `_parse_word_num_span`, `_classify_word_num_tok` | Deterministic (lookup + span parsing) | Limited to implemented word-number vocabulary |
| 5 | Enumeration-derived counts | Query text | Implicit counts from enumerated lists | `_extract_enum_derived_counts` | Deterministic (pattern-based) | Pattern-based; brittle to unseen enumeration styles |
| 6 | Candidate slot construction | Expected scalar parameter names | `SlotRecord` / `SlotIR` / `SlotOptIR` objects | Camel-case splitting, alias generation, type inference (`_build_slot_records`, `_slot_aliases`, `_expected_type`) | Deterministic (rule-based) | Coarse four-way type system (see below) |
| 7 | Coarse type system | Slot name | One of a small set of type categories | `_expected_type`, `_is_type_match`, `_is_type_incompatible` | Deterministic rules | **Architectural weakness** — no fine-grained numeric semantics (e.g. rate vs. absolute count both map to a shared coarse type in some cases) |
| 8 | Lexical/context pair scoring (baseline) | One mention, one slot | Compatibility score | `_score_mention_slot` — hand-engineered feature combination | Deterministic, hand-engineered | This is exactly the stage §8 of `PROJECT_STATUS.md` proposes replacing/augmenting with a learned local scorer |
| 9 | Semantic IR scoring | Enriched mention (`MentionIR`), enriched slot (`SlotIR`) | Compatibility score + semantic tags | `_extract_enriched_mentions`, `_score_mention_slot_ir`, `_context_to_semantic_tags` | Deterministic, tag-based | Hand-engineered tag vocabulary |
| 10 | Optimization-role scoring | `MentionOptIR`, `SlotOptIR` | Compatibility score with role/polarity/bound cues | `_score_mention_slot_opt`, `_compute_primary_role`, `_compute_bound_role`, `_detect_opt_unit_tags` | Deterministic, lexicon-based | Lexicon-based role cues; does not generalize beyond implemented cue vocabulary |
| 11 | Relation-aware scoring | `MentionOptIR`/`SlotOptIR` pairs | Local + relation features (4 ablation modes: basic/ops/semantic/full) | `tools/relation_aware_linking.py` — mention-mention and slot-slot relation tables | Deterministic; **module docstring explicitly notes "a learned scorer can be plugged in"** | See docstring note — this is the most natural integration point for §8's proposed learned scorer |
| 12 | Ambiguity-aware scoring | Candidate sets per slot | Top-K candidates, ambiguity signals (margin/entropy), abstention | `tools/ambiguity_aware_grounding.py` | Deterministic | Abstention threshold is hand-tuned |
| 13 | Global assignment (greedy) | Scored (mention, slot) pairs | Final one-to-one assignment | Typed greedy (default `assignment_mode="typed"`) | Deterministic | Baseline; no global optimality guarantee |
| 14 | Global assignment (bipartite) | Scored pairs | Optimal one-to-one assignment under scores | `_run_max_weight_matching_grounding` (maximum-weight bipartite matching) | Deterministic (exact combinatorial optimization over the scored graph) | Evaluated 2026-08-12: InstantiationReady 0.7432 (reproducible, no leakage). **Loses to a fresh typed-greedy rerun (0.7764, p=0.042)** — the earlier same-day "strongest method found so far" claim compared it against a stale baseline; see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. Negative result. |
| 15 | Global assignment (search) | Scored pairs, partial assignment states | Assignment via deterministic beam search | `tools/search_structured_grounding.py` — beam search with global-consistency penalties and NULL abstention | Deterministic | Evaluated 2026-08-12: InstantiationReady 0.7039. **Loses to fresh typed greedy (0.7764, p<0.001).** Negative result; see staleness audit above. |
| 16 | Global assignment (hierarchical/regions) | Query-region decomposition, relation-aware + search-structured scoring | Assignment with region-role compatibility | `tools/hierarchical_structured_grounding.py` | Deterministic | Evaluated 2026-08-12: InstantiationReady 0.7039, identical to stage 15 on this benchmark. **Loses to fresh typed greedy (0.7764, p<0.001).** Negative result; see staleness audit above. |
| 17 | Validation / repair | Candidate assignment | Repaired assignment or rejection | `_validation_and_repair`, `_opt_role_validate_and_repair`, `_bound_swap_repair`, `_total_perunit_swap_repair` | Deterministic, rule-based | Repair rules are hand-written per failure family |
| 18 | Acceptance reranking | Top-k retrieved schemas | Reranked/accepted schema | `_acceptance_score`, `make_rerank_rank_fn`, optional hierarchy (`use_hierarchy`) | Deterministic scoring over retrieval candidates | Operates on retrieval output, not on grounding itself |
| 19 | Structural verification | Instantiated LP | Pass/fail structural checks (no live solver) | `formulation/verify.py` | Deterministic | Catches structural errors only, not semantic correctness |
| 20 | Solver-backed validation (restricted) | Structurally valid LP, 20-instance compatibility-filtered subset | Solve outcome | SciPy HiGHS shim | Deterministic (classical solver) | Restricted subset (20/331), compatibility-filtered not random |

**Main/canonical vs. experimental:** stages 1-10, 13, 17-20 are part of the
canonical `typed_greedy` pipeline used for the manuscript's headline numbers
(0.5287 InstantiationReady, as submitted — see staleness note below).
Stages 11-12 and their downstream evaluated variants (relation-aware,
ambiguity-aware) are canonical *as separately benchmarked methods*
(Part 2), and are negative results. **Stages 14-16 (max-weight matching,
search-structured, hierarchical-structured) were evaluated for the first
time on 2026-08-12 and were briefly (same day) believed to be the
highest-InstantiationReady methods in this repository (0.70-0.74 vs.
0.5287) — this was found, later the same day, to be an artifact of
comparing against a stale typed-greedy baseline number (0.5287, which
predates 49 commits of grounding fixes). A fresh, same-code typed-greedy
rerun gives 0.7764, which beats all three, significantly on `orig`. See
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` for the full correction and
`results/unevaluated_methods_evaluation/` for the (now superseded)
original claim.**

---

## Part 2: Grounding method inventory

Columns: **CLI dispatch** = `--assignment-mode` value (or `--baseline` value
for acceptance-rerank variants) in `tools/nlp4lp_downstream_utility.py`.
**Evaluated** = a result file exists either under
`results/eswa_revision/02_downstream_postfix/` (Phase-1/2-era methods,
**stale**, see below) or `results/unevaluated_methods_evaluation/`
(three methods first evaluated in Phase 3, 2026-08-12) for this exact
method. **Beats typed greedy?** = InstantiationReady vs. `tfidf_typed_greedy`,
`orig` variant.

**IMPORTANT (2026-08-12, Phase 4):** every InstReady number in the table
below that traces back to `results/eswa_revision/02_downstream_postfix/`
(i.e. everything except the three "first evaluated 2026-08-12" rows) was
measured against a codebase state that predates 49 subsequent commits of
grounding fixes and **does not reproduce from the current code**. The
"Fresh (2026-08-12)" column gives same-day, same-code numbers for every
method, measured via the identical harness function
(`run_single_setting()`), so the whole row set is now internally
comparable. Full audit: `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`;
raw artifacts: `results/baseline_staleness_audit_2026-08-12/`.

| Method | CLI dispatch | Impl. path | Mechanism | InstReady (stale, committed) | **InstReady (fresh, 2026-08-12)** | Beats fresh typed greedy (0.7764)? | Status |
|---|---|---|---|---|---|---|---|
| Typed greedy (baseline) | `typed` (default) | `tools/nlp4lp_downstream_utility.py` | Hand-engineered pairwise score + greedy fill | 0.5287 | **0.7764** | — (reference) | **CANONICAL** |
| Oracle typed greedy | `typed` + `--baseline oracle` | same file | Typed greedy with gold schema | 0.5680 | **0.8248** | Yes, +0.0483, p<0.001 (expected — retrieval upper bound) | **CANONICAL** (control, not a candidate method) |
| Constrained matching | `constrained` | same file, `_constrained_assignment` | 1-to-1 constraint enforcement | 0.4230 | **0.7492** | No, −0.0272, p=0.050 (borderline) | **NEGATIVE_RESULT** |
| Semantic IR repair | `semantic_ir_repair` | same file, `_run_semantic_ir_repair` | Semantic tag-based repair | 0.4864 | **0.7160** | No, −0.0604, p<0.001 | **NEGATIVE_RESULT** |
| Optimization-role repair | `optimization_role_repair` | same file, `_run_optimization_role_repair` | Lexicon-based role/bound repair | 0.4411 | **0.7372** | No, −0.0393, p=0.020 | **NEGATIVE_RESULT** |
| Acceptance reranking | `--baseline tfidf_acceptance_rerank` | same file, `_acceptance_score` | Rerank top-k retrieval by acceptance score | 0.5257 | **0.7644** | No, −0.0121, p=0.328 (n.s., tied) | **DIAGNOSTIC** (statistically indistinguishable from fresh typed greedy) |
| Hierarchical acceptance reranking | `--baseline tfidf_hierarchical_acceptance_rerank` | same file | Acceptance reranking + hierarchy | 0.5196 | **0.7190** | No, −0.0574, p<0.001 | **NEGATIVE_RESULT** (was DIAGNOSTIC under stale numbers; significantly worse under fresh) |
| Global compatibility grounding (local/pairwise/full) | `global_compat_local`/`global_compat_pairwise`/`global_compat_full` | same file, `_run_global_compatibility_grounding`, `_gcgp_beam_search` | Beam search with pairwise global-consistency penalties | full: 0.4320 | not yet regenerated fresh | No (stale-vs-stale comparison; likely still worse but not re-verified — see staleness audit) | **NEGATIVE_RESULT** (fresh number pending) |
| Relation-aware linking (basic/ops/semantic/full) | `relation_aware_basic/ops/semantic/full` | `tools/relation_aware_linking.py` | Mention-mention + slot-slot relation features, 4 ablation levels | basic: 0.4985; full: 0.4169 | not yet regenerated fresh | No (stale-vs-stale comparison; not re-verified) | **NEGATIVE_RESULT** (fresh number pending) |
| Ambiguity-aware grounding (candidate-greedy/beam/abstain/full) | `ambiguity_candidate_greedy`/`ambiguity_aware_beam`/`ambiguity_aware_abstain`/`ambiguity_aware_full` | `tools/ambiguity_aware_grounding.py` | Top-K candidates + competition-aware beam + abstention | beam: 0.4230; full: 0.4199; abstain: much lower (Coverage 0.2207) | not yet regenerated fresh | No (stale-vs-stale comparison; not re-verified) | **NEGATIVE_RESULT** (fresh number pending) |
| Maximum-weight bipartite matching | `max_weight_matching` | same file, `_run_max_weight_matching_grounding` | Exact optimal bipartite matching (Hungarian, `scipy.optimize.linear_sum_assignment`) over the same opt-role pairwise scores | n/a (first evaluated 2026-08-12) | **0.7432** | **No, −0.0332, p=0.042 (significantly worse)** | **NEGATIVE_RESULT.** Briefly (same day) misclassified CANONICAL by comparing against the stale 0.5287 baseline instead of fresh typed greedy; corrected same day. Reproducible, deterministic, no leakage (see staleness audit §2) — the method measurement itself is sound, only the original comparison was wrong. |
| Global consistency grounding (original/superseded GCG) | `global_consistency_grounding` | same file, `_run_global_consistency_grounding` | 6 hand-engineered consistency signals | Only a **synthetic** evaluation exists (`docs/archive/GCG_FINAL_EVAL_REPORT.md`) — real HF-gated benchmark was blocked at the time | n/a | n/a | **SUPERSEDED** by `global_compat_*` (which was properly evaluated on real gold data); do not conflate the two "GCG"-named methods |
| Search-structured grounding | `search_structured_grounding`/`_no_global`/`_counterfactual` | `tools/search_structured_grounding.py` | Deterministic beam search over partial assignments with abstention | n/a (first evaluated 2026-08-12) | **0.7039** | **No, −0.0725, p<0.001 (significantly worse)** | **NEGATIVE_RESULT.** Same correction as `max_weight_matching` above. |
| Hierarchical structured grounding | `hierarchical_structured_grounding`/`_no_regions`/`_no_search` | `tools/hierarchical_structured_grounding.py` | Query-region decomposition + search-structured assignment | n/a (first evaluated 2026-08-12) | **0.7039** (identical to search-structured on this benchmark) | **No, −0.0725, p<0.001 (significantly worse)** | **NEGATIVE_RESULT.** Same correction as `max_weight_matching` above. |
| Optimization-role relation repair / anchor linking / bottom-up beam repair / entity-semantic beam repair | `optimization_role_relation_repair`, `optimization_role_anchor_linking`, `optimization_role_bottomup_beam_repair`, `optimization_role_entity_semantic_beam_repair` | same file | Various repair/beam extensions of the optimization-role family | **No** (last three explicitly marked "Experimental/archived (not in default focused eval)" in the CLI help text itself) | n/a | n/a | **EXPERIMENTAL/SUPERSEDED** |
| Untyped assignment | `untyped` | same file | Assignment without type gating (diagnostic lower bound) | Not found in the official comparison set | n/a | n/a | **DIAGNOSTIC** (likely a deliberate ablation/lower-bound tool, not a candidate method) |

**BM25/LSA typed greedy, fresh (2026-08-12), for reference:** `bm25_typed_greedy`
0.7644 (n.s. vs. fresh tfidf typed greedy, p=0.322); `lsa_typed_greedy`
0.7341 (significantly worse, p<0.001). Both were also measured under the
same stale codebase state previously (not tabulated above since they were
never part of the richer-method comparison set).

**Bottom line for future agents (updated 2026-08-12, Phase 4 — supersedes
the Phase 3 version of this paragraph):** the methods evaluated in Phase 2
confirmed that richer *greedy or repair-based* deterministic grounding
underperforms or ties `typed_greedy` on `orig` InstantiationReady (see
`docs/NEGATIVE_RESULTS.md`) — but those Phase-2 numbers, like the
manuscript's own headline 0.5287, are measured against a codebase state
49 commits stale relative to current code. Phase 3 (same day, 2026-08-12)
found that three method families flagged as "implemented and tested,
never run against the real benchmark" — `max_weight_matching`,
`search_structured_grounding`, `hierarchical_structured_grounding` —
scored 0.70-0.74, dramatically above the *stale* 0.5287 typed-greedy
number, and concluded they were the strongest methods ever found here.
**Later the same day this was found to be an invalid comparison**: a
fresh, same-code typed-greedy rerun scores **0.7764**, significantly
beating all three (see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`).
**The correct, current bottom line is the original Phase 2 one, now on
firmer footing: no evaluated richer scoring, repair, or global-assignment
strategy beats plain typed greedy on `orig` InstantiationReady.** Plain
typed greedy is the strongest known non-oracle method in this repository
as of 2026-08-12. Any future comparison (including a future
learned-scorer attempt, see `docs/LEARNED_GROUNDING_P0.md`) must be run
fresh, same-day, same-code against `tfidf_typed_greedy` — **never against
a committed number without first confirming it still reproduces** (this
is now the single most important reproducibility lesson in this
repository's history; see `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`
§8 for the exact commands).
