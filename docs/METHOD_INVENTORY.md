# Method Inventory: Algorithm Decomposition and Grounding Methods

**Purpose:** (1) a technically accurate decomposition of the current pipeline
stage-by-stage, and (2) a definitive list of every grounding method already
implemented, so no future agent reinvents one. Verified 2026-08-11 by
reading `tools/nlp4lp_downstream_utility.py` (6,977 lines) and its satellite
modules directly, and cross-checking evaluated-method claims against actual
result files in `results/eswa_revision/02_downstream_postfix/`.

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
| 14 | Global assignment (bipartite) | Scored pairs | Optimal one-to-one assignment under scores | `_run_max_weight_matching_grounding` (maximum-weight bipartite matching) | Deterministic (exact combinatorial optimization over the scored graph) | **Evaluated 2026-08-12: InstantiationReady 0.7432, the strongest method found so far** (see Part 2) — optimal only w.r.t. the (still hand-engineered) pairwise scores, which is apparently sufficient |
| 15 | Global assignment (search) | Scored pairs, partial assignment states | Assignment via deterministic beam search | `tools/search_structured_grounding.py` — beam search with global-consistency penalties and NULL abstention | Deterministic | **Evaluated 2026-08-12: InstantiationReady 0.7039**, strong positive result |
| 16 | Global assignment (hierarchical/regions) | Query-region decomposition, relation-aware + search-structured scoring | Assignment with region-role compatibility | `tools/hierarchical_structured_grounding.py` | Deterministic | **Evaluated 2026-08-12: InstantiationReady 0.7039**, strong positive result, identical to stage 15 on this benchmark |
| 17 | Validation / repair | Candidate assignment | Repaired assignment or rejection | `_validation_and_repair`, `_opt_role_validate_and_repair`, `_bound_swap_repair`, `_total_perunit_swap_repair` | Deterministic, rule-based | Repair rules are hand-written per failure family |
| 18 | Acceptance reranking | Top-k retrieved schemas | Reranked/accepted schema | `_acceptance_score`, `make_rerank_rank_fn`, optional hierarchy (`use_hierarchy`) | Deterministic scoring over retrieval candidates | Operates on retrieval output, not on grounding itself |
| 19 | Structural verification | Instantiated LP | Pass/fail structural checks (no live solver) | `formulation/verify.py` | Deterministic | Catches structural errors only, not semantic correctness |
| 20 | Solver-backed validation (restricted) | Structurally valid LP, 20-instance compatibility-filtered subset | Solve outcome | SciPy HiGHS shim | Deterministic (classical solver) | Restricted subset (20/331), compatibility-filtered not random |

**Main/canonical vs. experimental:** stages 1-10, 13, 17-20 are part of the
canonical `typed_greedy` pipeline used for the manuscript's headline numbers
(0.5287 InstantiationReady). Stages 11-12 and their downstream evaluated
variants (relation-aware, ambiguity-aware) are canonical *as separately
benchmarked methods* (Part 2), and are negative results. **Stages 14-16
(max-weight matching, search-structured, hierarchical-structured) were
evaluated for the first time on 2026-08-12 and are now the
highest-InstantiationReady methods in this repository** (0.70-0.74,
vs. 0.5287 for the manuscript's own headline method) — not yet integrated
into the manuscript's Table 1, but canonical evidence within this
repository's internal evidence base. See
`results/unevaluated_methods_evaluation/`.

---

## Part 2: Grounding method inventory

Columns: **CLI dispatch** = `--assignment-mode` value (or `--baseline` value
for acceptance-rerank variants) in `tools/nlp4lp_downstream_utility.py`.
**Evaluated** = a result file exists either under
`results/eswa_revision/02_downstream_postfix/` (Phase-1/2-era methods) or
`results/unevaluated_methods_evaluation/` (three methods newly evaluated in
Phase 3, 2026-08-12) for this exact method. **Beats typed greedy?** =
InstantiationReady vs. `tfidf_typed_greedy` (0.5287), `orig` variant, per
`results/eswa_revision/14_reports/downstream_comparison_all_methods.csv`
(Phase-1/2 methods) or `results/unevaluated_methods_evaluation/significance.json`
(Phase-3 methods).

| Method | CLI dispatch | Impl. path | Mechanism | Evaluated? | InstReady (orig) | Beats typed greedy? | Status |
|---|---|---|---|---|---|---|---|
| Typed greedy (baseline) | `typed` (default) | `tools/nlp4lp_downstream_utility.py` | Hand-engineered pairwise score + greedy fill | Yes | 0.5287 | — (baseline) | **CANONICAL** |
| Constrained matching | `constrained` | same file, `_constrained_assignment` | 1-to-1 constraint enforcement | Yes | 0.4230 | No | **NEGATIVE_RESULT** |
| Semantic IR repair | `semantic_ir_repair` | same file, `_run_semantic_ir_repair` | Semantic tag-based repair | Yes | 0.4864 | No | **NEGATIVE_RESULT** |
| Optimization-role repair | `optimization_role_repair` | same file, `_run_optimization_role_repair` | Lexicon-based role/bound repair | Yes | 0.4411 | No | **NEGATIVE_RESULT** |
| Acceptance reranking | `--baseline tfidf_acceptance_rerank` | same file, `_acceptance_score` | Rerank top-k retrieval by acceptance score | Yes | 0.5257 | No (near-tie, not significant, p=0.89) | **DIAGNOSTIC** (statistically indistinguishable from baseline) |
| Hierarchical acceptance reranking | `--baseline tfidf_hierarchical_acceptance_rerank` | same file | Acceptance reranking + hierarchy | Yes | 0.5196 | No (not significant, p=0.58) | **DIAGNOSTIC** |
| Global compatibility grounding (local/pairwise/full) | `global_compat_local`/`global_compat_pairwise`/`global_compat_full` | same file, `_run_global_compatibility_grounding`, `_gcgp_beam_search` | Beam search with pairwise global-consistency penalties | Yes | full: 0.4320 | No (full is significantly worse, p<0.001) | **NEGATIVE_RESULT** |
| Relation-aware linking (basic/ops/semantic/full) | `relation_aware_basic/ops/semantic/full` | `tools/relation_aware_linking.py` | Mention-mention + slot-slot relation features, 4 ablation levels | Yes | basic: 0.4985; full: 0.4169 | No (basic not significant, p=0.15; full significantly worse, p<0.001) | **NEGATIVE_RESULT** (basic is the closest competitor, still not significant) |
| Ambiguity-aware grounding (candidate-greedy/beam/abstain/full) | `ambiguity_candidate_greedy`/`ambiguity_aware_beam`/`ambiguity_aware_abstain`/`ambiguity_aware_full` | `tools/ambiguity_aware_grounding.py` | Top-K candidates + competition-aware beam + abstention | Yes | beam: 0.4230; full: 0.4199; abstain: much lower (Coverage 0.2207, over-abstains) | No (beam/full significantly worse, p<0.001; abstain catastrophically over-conservative) | **NEGATIVE_RESULT** |
| Maximum-weight bipartite matching | `max_weight_matching` | same file, `_run_max_weight_matching_grounding` | Exact optimal bipartite matching (Hungarian, `scipy.optimize.linear_sum_assignment`) over the same opt-role pairwise scores | **Yes (2026-08-12)** | **0.7432** | **YES — by +0.2145, p<0.001 (robust)**, and exceeds Oracle-TG (0.5680) too | **CANONICAL — STRONG POSITIVE RESULT.** The single best-performing method in this repository's evaluated history. See `results/unevaluated_methods_evaluation/` |
| Global consistency grounding (original/superseded GCG) | `global_consistency_grounding` | same file, `_run_global_consistency_grounding` | 6 hand-engineered consistency signals | Only a **synthetic** evaluation exists (`docs/archive/GCG_FINAL_EVAL_REPORT.md`) — real HF-gated benchmark was blocked at the time | n/a | n/a | **SUPERSEDED** by `global_compat_*` (which was properly evaluated on real gold data); do not conflate the two "GCG"-named methods |
| Search-structured grounding | `search_structured_grounding`/`_no_global`/`_counterfactual` | `tools/search_structured_grounding.py` | Deterministic beam search over partial assignments with abstention | **Yes (2026-08-12)** | **0.7039** | **YES — by +0.1752, p<0.001 (robust)** | **CANONICAL — STRONG POSITIVE RESULT.** See `results/unevaluated_methods_evaluation/` |
| Hierarchical structured grounding | `hierarchical_structured_grounding`/`_no_regions`/`_no_search` | `tools/hierarchical_structured_grounding.py` | Query-region decomposition + search-structured assignment | **Yes (2026-08-12)** | **0.7039** (identical to search-structured on this benchmark — the region-decomposition layer did not change decisions on these 331 queries) | **YES — by +0.1752, p<0.001 (robust)** | **CANONICAL — STRONG POSITIVE RESULT.** See `results/unevaluated_methods_evaluation/` |
| Optimization-role relation repair / anchor linking / bottom-up beam repair / entity-semantic beam repair | `optimization_role_relation_repair`, `optimization_role_anchor_linking`, `optimization_role_bottomup_beam_repair`, `optimization_role_entity_semantic_beam_repair` | same file | Various repair/beam extensions of the optimization-role family | **No** (last three explicitly marked "Experimental/archived (not in default focused eval)" in the CLI help text itself) | n/a | n/a | **EXPERIMENTAL/SUPERSEDED** |
| Untyped assignment | `untyped` | same file | Assignment without type gating (diagnostic lower bound) | Not found in the official comparison set | n/a | n/a | **DIAGNOSTIC** (likely a deliberate ablation/lower-bound tool, not a candidate method) |

**Bottom line for future agents (updated 2026-08-12, Phase 3):** the
methods evaluated in Phase 2 confirmed that richer *greedy or repair-based*
deterministic grounding underperforms or ties `typed_greedy` on `orig`
InstantiationReady (see `docs/NEGATIVE_RESULTS.md`). But three method
families flagged in Phase 2 as "implemented and tested, never run against
the real benchmark" turned out, once actually run, to be the **strongest
methods in this repository's evaluated history** —
`max_weight_matching` reaches InstantiationReady **0.7432** (vs. typed
greedy's 0.5287 and even Oracle-TG's 0.5680), by decoding the same
opt-role pairwise scores with an **exact global assignment** instead of a
greedy or repair-based decode. This flips the prior "richer scoring
doesn't help" narrative on its head: the scoring function was never the
sole problem -- greedy/repair decoding strategies were leaving large,
recoverable gains on the table. See
`results/unevaluated_methods_evaluation/README.md` for the full record and
`PROJECT_STATUS.md` §3-4 for how this reframes the project's headline
narrative going forward. **`max_weight_matching` is now the strongest known
method and the correct baseline for any future comparison** (including any
future learned-scorer attempt, see `docs/LEARNED_GROUNDING_P0.md`).
