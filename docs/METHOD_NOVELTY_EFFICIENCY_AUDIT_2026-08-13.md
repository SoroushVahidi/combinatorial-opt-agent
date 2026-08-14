# Method Novelty + Efficiency Audit - 2026-08-13

**Scope.** Manuscript-resubmission method audit only. No manuscript files were
modified. No heavy training, GPU inference, or external API calls were run.

**Starting state verified.** Branch `main`, HEAD `28630bc`, upstream
`origin/main`, working tree clean at start, ahead/behind `0/0`.

## 1. Current Method

The current strongest non-oracle method is `tfidf_typed_greedy`, freshly rerun
from current code. It is not the richer optimization-role scorer and not any
structured assignment variant.

Algorithm:

```text
Input: query q, fixed NLP4LP schema catalog C, gold-cache-derived slot metadata

1. Build catalog:
   load 335 schema records from data/catalogs/nlp4lp_catalog.jsonl.

2. Retrieve:
   fit TF-IDF over schema descriptions; rank q by cosine similarity; keep top-1
   schema s_hat.

3. Select schema slots:
   from s_hat's problem_info.parameters, keep scalar keys that overlap scalar
   gold keys for the query under the current evaluation harness.

4. Extract numeric mentions:
   scan q for digit tokens, currency/percent contexts, word numbers, fractions.

5. For each scalar slot p in expected_scalar order:
   infer expected coarse type T(p) from slot name.
   choose one remaining numeric mention by _choose_token(T, candidates):
      rank by type preference, absolute numeric magnitude, raw string tie-break.
   remove the chosen mention from the candidate list.

6. Score readiness:
   coverage = filled_slots / expected_scalar_slots.
   type_match = type-compatible fills / filled_slots.
   InstantiationReady = 1 iff coverage >= 0.8 and type_match >= 0.8.

7. Optional controls:
   exact5/exact20 on schema hits; structural verification and solver-backed
   subsets are separate validation layers, not part of InstantiationReady.
```

Component details:

| Component | Input -> output | Algorithm | Complexity | Nature | Assumptions / failure modes |
|---|---|---:|---|---|---|
| Catalog construction | JSONL schemas -> fixed catalog | load + normalize fields | O(N) | deterministic | fixed 335-schema world |
| Retrieval | query, catalog -> top-1 schema | TF-IDF vectorize + cosine | fit O(NV), query O(nnz(q)+N_sparse_dot) | deterministic | top-1 errors unrecoverable in default path |
| Numeric extraction | query -> `NumTok` list | regex, word-number parsing, context typing | O(L*w) | deterministic heuristic | misses nonstandard phrases, imperfect percent/currency context |
| Mention typing | token/context -> int/float/currency/percent/unknown | context rules | O(1) per mention | deterministic heuristic | coarse semantic buckets |
| Slot typing | slot name -> expected type | name substring rules | O(|name|) | deterministic heuristic | name-dependent, not schema semantic model |
| Mention-slot scoring | slot type + candidate list -> one mention | `_choose_token`: type preference, magnitude, raw tie-break | O(M) per slot | deterministic heuristic | no lexical slot/mention context in winning branch |
| Greedy assignment | slots, mentions -> fills | one pass, remove used mention | O(SM) | deterministic | order-sensitive, no global optimality |
| Repair/postprocess | current typed branch | none beyond candidate removal | O(1) | deterministic | richer repair branches are separate negative-result methods |
| Structural verification | instantiated problem/code -> errors | schema/formulation/LP syntax checks | O(size(formulation)) | deterministic | catches structure, not semantic numeric correctness |
| InstantiationReady | per-query fills -> binary | coverage/type thresholds | O(S) | deterministic | does not require schema_hit; StrictInstantiationReady adds that gate |
| Solver subset | compatible subset -> solve status | SciPy HiGHS shim | solver-dependent | deterministic | restricted 20-instance subset |

## 2. Fresh Performance Reference

Fresh current-code rerun on 331 `orig` examples:

- `tfidf_typed_greedy`: InstantiationReady `0.776435` (257/331), Schema R@1
  `0.909366`, Coverage `0.879430`, TypeMatch `0.851545`,
  Exact20_on_hits `0.244888`.
- `oracle_typed_greedy`: InstantiationReady `0.824773`; the retrieval upper
  bound over TF-IDF is +0.0483 absolute.
- Current common-18 external-baseline manifest result: 10/18 =
  `0.556` InstantiationReady.
- Lightweight rerun in this audit: 1.08 s wall-clock, max RSS 189,356 KB.

The submitted manuscript still reports the historical `0.5287` value. That
submitted number is stale relative to current code and must not be silently
rewritten in this task.

## 3. Current Failure Taxonomy

Fresh typed-greedy coarse query-level taxonomy, derived in this audit from the
331-query rerun:

| Category | Count |
|---|---:|
| Ready under standard InstantiationReady | 257 |
| Not ready | 74 |
| Schema miss | 30 |
| Schema-hit but not ready | 54 |
| Schema-hit, coverage >=0.8, type_match <0.8 | 30 |
| Schema-hit, coverage <0.8, type_match >=0.8 | 11 |
| Schema-hit, coverage <0.8, type_match <0.8 | 13 |
| Zero expected scalar slots | 20 total; 1 genuine schema-hit case (`nlp4lp_test_293`) |

Additional lightweight diagnostic signals on the 54 schema-hit/not-ready cases:

| Signal | Query count |
|---|---:|
| type-compatible candidate ambiguity | 39 |
| same-type slot ambiguity | 38 |
| type mismatch / unit-type issue | 33 |
| wrong assignment while gold numeric value was extracted | 33 |
| missing or insufficient numeric mentions | 31 |
| gold value not extracted or scale mismatch | 22 |
| total/per-unit confusion | 11 |
| exact tie or low assignment margin | 10 |
| bound-role confusion | 4 |
| objective/constraint confusion | 1 |

Representative examples:

- `nlp4lp_test_14`: real-estate investment; percentages, dollars, max/min
  investment roles compete. Typed greedy assigns `$200,000` to a return-rate
  slot and `half` to an investment-bound slot.
- `nlp4lp_test_39`: staff/substitute shifts; per-shift hours/pay and total
  teaching-hours/budget quantities collide, producing total/per-unit confusion.
- `nlp4lp_test_47`: honey jars; "twice as many" is not extracted as a usable
  scalar, causing a ratio slot failure and downstream shifts.
- `nlp4lp_test_66`: stamping machines; rate, glue usage, and minimum/maximum
  slots show repeated same-type and bound-role ambiguity.
- `nlp4lp_test_293`: genuine scalar-only architecture limit; vector/matrix
  parameters dominate.

Important caveat: the diagnostic labels are heuristic. They are appropriate
for research planning, not for manuscript tables without a checked diagnostic
script and tests.

## 4. Reviewer-Gap Matrix

| Criticism | Status | Repository evidence |
|---|---|---|
| KAIS: insufficient comparative studies | PARTIALLY_RESOLVED | `baselines/comparison/` harness and `docs/EXTERNAL_BASELINE_COMPARISON_PROTOCOL.md` are complete; empirical ORLM/OptMATH/DeepOR/OR-R1 rows are pending/unavailable. |
| KAIS: insufficient recent references | PARTIALLY_RESOLVED | `docs/ALGORITHM_IMPROVEMENT_ROADMAP.md` surveys NL4Opt, Ner4Opt, MeasEval, schema-guided slot filling; manuscript not yet updated. |
| KAIS: missing baselines from strong venues | PARTIALLY_RESOLVED | PaMOP pilot validated; ORLM/OptMATH implemented; DeepOR/OR-R1 blocked by checkpoints/resources. |
| ESWA R1: weak theoretical grounding of retrieval | STILL_OPEN | TF-IDF/BM25/LSA are empirical baselines; no formal retrieval model beyond vector-space IR. |
| ESWA R1: TF-IDF ignores semantics | PARTIALLY_RESOLVED | Dense/semantic alternatives and acceptance reranking exist but do not beat fresh TF-IDF; semantics remains weak in top-1 retrieval. |
| ESWA R1: ambiguous typed-greedy assignments | STILL_OPEN | Fresh taxonomy shows 38 schema-hit/not-ready same-type ambiguity cases. |
| ESWA R1: unclear extraction/type rules | PARTIALLY_RESOLVED | Code has extensive deterministic rules in `tools/nlp4lp_downstream_utility.py`; method writeup still needs a concise formal decomposition. |
| ESWA R1: cascading schema errors | PARTIALLY_RESOLVED | Oracle control quantifies retrieval upper bound; default pipeline remains top-1 and unrecoverable. |
| ESWA R1: informal pipeline interactions | PARTIALLY_RESOLVED | `docs/METHOD_INVENTORY.md` decomposes stages; this audit further formalizes components. |
| ESWA R2: contribution/scope mismatch | STILL_OPEN | Current winning method is an efficient fixed-catalog scalar grounder, not general NL-to-optimization modeling. |
| ESWA R2: weak expert-system positioning | PARTIALLY_RESOLVED | CPU-only deterministic, inspectable, rule-based nature is strong; scientific framing needs sharper limits. |
| ESWA R2: lack of real reasoning mechanism | STILL_OPEN | Current typed branch is type/magnitude greedy; structured variants are negative results. |
| ESWA R2: benchmark may be lexically easy | PARTIALLY_RESOLVED | Overlap analysis shows number stripping does not explain retrieval, but fixed-catalog NLP4LP remains narrow. |
| ESWA R2: generalization beyond 331-schema setting | STILL_OPEN | External datasets are adapter-level; no held-out-family or cross-dataset scalar-grounding validation yet. |
| ESWA R2: grounding limited to scalar parameters | STILL_OPEN | `nlp4lp_test_293` confirms vector/matrix parameters are out of scope. |
| ESWA R2: type compatibility too weak | STILL_OPEN | Fresh failures include 33 schema-hit/not-ready type/unit issues. |
| ESWA R2: ablation/fix looks like repair, not science | STILL_OPEN | 49-commit improvement is real but reads as engineering unless reframed around a principled role/quantity model. |

## 5. Negative-Result Exclusions

Do not recommend any of these as a renamed main method without a new mechanism:

| Idea | Implementation / result | Failure interpretation |
|---|---|---|
| Constrained one-to-one assignment | `_constrained_assignment`; fresh `0.7492`, borderline worse than `0.7764` | Hard one-to-one constraints hurt more than they help with current local choices. |
| Semantic IR repair | `_run_semantic_ir_repair`; fresh `0.7160` | Hand semantic tags add noisy repair decisions. |
| Optimization-role repair | `_run_optimization_role_repair`; fresh `0.7372` | Lexicon role repair is brittle. |
| Acceptance reranking | `tfidf_acceptance_rerank`; fresh `0.7644`, n.s. | Schema-only acceptance does not recover much over strong top-1 retrieval. |
| Hierarchical acceptance reranking | fresh `0.7190` | Extra hierarchy penalty hurts. |
| Max-weight matching | `_run_max_weight_matching_grounding`; fresh `0.7432`, p=0.042 worse | Exact assignment cannot rescue wrong local scores. |
| Search structured grounding | `tools/search_structured_grounding.py`; fresh `0.7039` | Beam/global penalties add noise/cost. |
| Hierarchical structured grounding | `tools/hierarchical_structured_grounding.py`; fresh `0.7039` | Region decomposition did not improve current scores. |
| Global compatibility variants | `global_compat_*`; stale negative, fresh pending | Current hand penalties should not be repeated before fresh recheck. |
| Relation-aware variants | `relation_aware_*`; stale negative, fresh pending | More relation features historically hurt; feature extraction may still be useful. |
| Ambiguity-aware variants | `ambiguity_*`; stale negative, fresh pending | Candidate/beam/abstain machinery miscalibrated with existing scores. |
| Learned text-only scorer NR10 | `distilroberta-base`; loses on every diagnostic metric | Undertrained and lacked structured features, but exact setup is answered. |
| P0 feature-augmented learned scorer NR11 | `tools/learned_local_scorer.py`; best 0.80 vs M0 0.86 on 50 oracle-schema cases | Proxy slot-selection gains did not transfer downstream; no repeat without new supervision/target. |
| LSA/BM25 as replacement retrieval | BM25 0.7644 n.s.; LSA 0.7341 worse | Retrieval replacement is not primary lever. |

## 6. Efficiency Audit

Measured full 331-query typed-greedy rerun:

- wall-clock: 1.08 s
- max RSS: 189 MB
- per query: about 3.3 ms end-to-end, including TF-IDF fit/load path

Complexity:

- TF-IDF fit: O(NV) once per run; query ranking O(nnz(q) + sparse dot against
  N schemas).
- Numeric extraction: O(L*w) for token scan and local context windows.
- Typed greedy: O(SM), where S is scalar slots and M is extracted mentions.
- Evaluation and readiness: O(S).
- Whole current pipeline: dominated by retrieval vector operations and Python
  per-query extraction; small constants on NLP4LP.

Implementation bottlenecks in the 6,977-line utility:

- repeated local-context scans across `_extract_num_tokens`,
  `_extract_num_mentions`, `_extract_opt_role_mentions`, and acceptance features;
- repeated slot metadata construction and repeated `_expected_type` calls;
- duplicated type/scoring/evaluation loops across assignment-mode branches;
- several separate, similar score functions with drift risk;
- schema-side features are recomputed per query/method and could be cached;
- per-slot diagnostics are not emitted for the winning branch, forcing ad hoc
  audits.

Efficiency opportunity: precompute schema-side slot metadata and factor shared
evaluation/assignment instrumentation. This is an engineering improvement, not
by itself a scientific contribution.

## 7. Novelty Assessment

Existing/common ingredients:

- TF-IDF/BM25/LSA retrieval.
- Regex/word-number extraction.
- Coarse type constraints.
- Greedy matching.
- Hand-authored deterministic rules.
- Standard coverage/type-match/readiness metrics.

Repository-specific engineering:

- NLP4LP schema catalog, gold-cache integration, evaluation harness.
- Extensive type/quantity/cue fixes in `tools/nlp4lp_downstream_utility.py`.
- Multiple deterministic negative-result branches.
- External-baseline comparison harness with explicit availability/proxy states.

Potentially publishable contribution today:

- A deterministic, CPU-only, fixed-catalog scalar-grounding expert system with
  inspectable intermediate decisions and a strong empirical finding that top-1
  schema retrieval is not the main bottleneck.
- A useful negative-result study: richer deterministic assignment, repair,
  relation, ambiguity, and learned-local scorers do not beat a well-tuned typed
  greedy baseline.

Weak novelty today:

- The winning algorithm is too close to type-ranked greedy slot filling.
- The strongest accuracy gain came from accumulated deterministic fixes, not a
  clean mathematical mechanism.
- Current slot typing and role semantics are encoded as many local rules.

Strongest defensible novelty claim if submitted today:

> A reproducible, CPU-only, retrieval-assisted scalar-instantiation benchmark
> and expert-system pipeline showing that fixed-catalog schema retrieval is
> strong, while deterministic number-to-slot grounding remains the dominant
> bottleneck; extensive same-code comparisons show that several richer
> assignment and learned-local variants do not outperform a carefully engineered
> typed-greedy baseline.

That is a defensible systems/evaluation claim, not a strong new-algorithm claim.

## 8. Candidate Methods

No more than five serious candidates:

1. **Role-Quantity Factorized Grounding (top candidate).** Define a compact,
   mathematical compatibility model over type, unit, quantity role, clause
   locality, bound polarity, and entity anchor features; solve with typed greedy
   by default and structured assignment only when ambiguity triggers fire.
2. **Selective top-k schema + grounding reranking.** For low retrieval-margin
   cases only, run grounding over top-k schemas and rerank by fillability,
   type/role consistency, and verification. Low-margin probe: margin <=0.05
   covers 18/30 schema misses among 27 cases.
3. **Quantity semantics extractor.** Add a declarative unit/quantity-role layer
   for total, per-unit, rate, percent, currency, bound, capacity, demand, and
   objective coefficient; feed it into the existing typed branch rather than
   separate failed repair branches.
4. **Held-out schema-family generalization protocol.** Not an accuracy method,
   but directly answers generalization concerns via family splits and
   benchmark-independent metadata.
5. **LLM/API role-label oracle diagnostic.** Small control only: ask whether
   high-quality semantic role labels would fix remaining failures. Do not make
   it the main method unless the oracle gain is large and reproducible.

Rejected or deprioritized:

- another generic learned pair scorer;
- another matching/beam algorithm with unchanged local scores;
- larger hand-rule bundles without a factorized formulation;
- full external LLM fallback as the main method, because CPU-only/no-external-LLM
  inference is a real advantage.

## 9. Top-3 Ranking

| Rank | Candidate | Expected gain | Novelty | Efficiency | Reviewer relevance | Risk |
|---:|---|---|---|---|---|---|
| 1 | Role-Quantity Factorized Grounding with ambiguity cascade | +2 to +5 pp if it fixes same-type/type-role failures without broad regressions | Moderate: a clean structured compatibility objective over interpretable factors | Preserves fast path; structured branch only on hard cases | Directly addresses weak reasoning, ambiguity, type semantics, coherence | Medium; can collapse into rules if not kept declarative |
| 2 | Selective top-k schema+grounding reranking | up to about +2 to +4 pp, bounded by retrieval misses | Moderate, distinct from prior acceptance rerank because it uses grounding outcome | k-fold cost only on low-margin cases | Addresses cascading schema errors and top-1 brittleness | Medium-high; retrieval gap is modest |
| 3 | Declarative quantity semantics layer | +1 to +3 pp, mainly total/per-unit and unit errors | Low-moderate | Very cheap | Addresses type compatibility and scalar semantics | High risk of looking like more rules unless paired with TOP-1 formulation |

## 10. TOP-1 Mathematical Formulation

Let mentions be \(M\), scalar slots \(S\), and \(x_{ms} \in \{0,1\}\) indicate
mention \(m\) fills slot \(s\). Each mention has observed features:
coarse type \(t_m\), unit \(u_m\), role \(r_m\), clause/region \(c_m\), entity
anchor \(e_m\), bound polarity \(b_m\), and quantity form \(q_m\)
(total/per-unit/rate/percent/count/currency). Each slot has schema-side
features \(t_s,u_s,r_s,c_s,e_s,b_s,q_s\) from slot metadata.

Objective:

```text
maximize_x
  sum_{m,s} [
      alpha_type * phi_type(m,s)
    + alpha_unit * phi_unit(m,s)
    + alpha_role * phi_role(m,s)
    + alpha_quantity * phi_quantity(m,s)
    + alpha_clause * phi_clause(m,s)
    + alpha_entity * phi_entity(m,s)
    + alpha_bound * phi_bound(m,s)
    + alpha_lex * phi_lex(m,s)
  ] x_ms
  + beta_pair * sum_{(s1,s2),(m1,m2)} Phi_pair(s1,s2,m1,m2) x_m1s1 x_m2s2
  - beta_null * sum_s z_s
```

Constraints:

- each slot receives at most one mention: \(\sum_m x_{ms} + z_s = 1\);
- each non-repeatable mention fills at most one slot: \(\sum_s x_{ms} \le 1\);
- hard type exclusions for percent/currency/count incompatibilities;
- min/max slots sharing a stem must respect bound polarity when both filled;
- count slots cannot receive non-integer decimal/rate mentions;
- optional threshold: leave slot null when all compatible scores are below a
  confidence floor.

Complexity:

- If pair terms are disabled, this is a maximum-weight bipartite assignment:
  Hungarian/min-cost flow in O(max(|M|,|S|)^3), or greedy O(SM) fallback.
- With pairwise structural terms it becomes a small ILP/QAP-like problem; use
  it only in the cascade hard path or approximate with beam search.
- Fast path: current typed greedy remains O(SM) for unambiguous cases.

The scientific mechanism is not "use matching"; matching already failed. The
mechanism is the factorization of numeric grounding into quantity role, unit,
clause, bound, and entity factors with ablations that can show which factor
changes outcomes.

## 11. Why TOP-1 Differs From Failed Prior Methods

- It does not reuse `_score_mention_slot_opt` unchanged.
- It does not propose global assignment as the improvement by itself; MWM showed
  that unchanged local scores lose.
- It does not add ad hoc repair after a wrong assignment; the role/quantity
  semantics enter the objective before decoding.
- It is not another learned scorer; weights can be fixed/preregistered first.
- It can reuse feature extractors from relation/ambiguity modules, but with a
  cleaner declarative feature table and a predeclared ablation plan.
- It can be run only on ambiguity-triggered cases to avoid the known failure
  mode where richer machinery hurts easy cases.

## 12. Adaptive / Cascade Opportunity

Promising design:

- Easy path: current `tfidf_typed_greedy`.
- Trigger structured resolver when any cheap signal fires:
  - TF-IDF top1-top2 margin <=0.05;
  - schema hit unknown but retrieval low margin;
  - more than one type-compatible mention for a slot;
  - same-type multiplicity among scalar slots;
  - total/per-unit conflict features;
  - coverage or type-match verification below threshold after greedy;
  - exact-tie/low-margin assignment.
- Hard path: run TOP-1 factorized assignment, optionally over top-k schemas.
- Selection rule: accept structured output only if it improves predeclared
  verification/compatibility score without lowering coverage/type-match gates.

Expected cost: current 1.08 s full benchmark plus structured overhead on a
minority of cases. The retrieval-margin probe suggests margin <=0.05 selects
27/331 queries and catches 18/30 schema misses.

## 13. Generalization Plan

Most useful generalization improvement without a new project:

- create declarative schema-side slot metadata independent of NLP4LP ids:
  type, unit, role, quantity form, bound polarity, objective/constraint role,
  entity anchor slots;
- evaluate by held-out schema family or problem-family split, not only random
  instance split;
- report scalar-only scope explicitly and separate vector/matrix cases;
- test at least one external or synthetic family where schema names differ from
  NLP4LP naming conventions;
- compare against Ner4Opt/NL4Opt-style entity extraction as an extraction-stage
  reference if resources allow.

## 14. API / LLM Diagnostic

Do not use Azure OpenAI, Vertex, Cohere, or CloudRift as the main method now.
Use an API only as a small diagnostic control if Stage A needs an oracle:

- sample 20-30 schema-hit/not-ready cases enriched for same-type and
  total/per-unit ambiguity;
- ask for structured labels only: total/per-unit/rate/bound/objective/constraint
  for each numeric mention;
- feed labels into the deterministic TOP-1 objective;
- stop if the oracle-label gain is <+2 pp projected or fixes fewer than half of
  the targeted failures.

This would estimate whether role labels are the missing information. It must
remain separate from CPU-only main-method claims.

## 15. Success Gate

Reference baseline: fresh current-code `tfidf_typed_greedy`,
InstantiationReady `0.7764` on the same 331-query `orig` protocol.

Practical GO threshold for manuscript main-method integration:

- at least +2.0 percentage points absolute InstantiationReady on 331 queries
  (>=0.7964, i.e. at least 264/331 ready), and
- paired test not worse than p<0.10 with directionally positive CI; preferably
  p<0.05, and
- no material Schema R@1/Coverage/TypeMatch regression unless justified, and
- at least one predeclared major error class improves by >=25% relative within
  triggered cases, and
- runtime remains under 2x current typed greedy for full benchmark or the
  cascade overhead is restricted to <=25% of queries, and
- ablations show the gain comes from the new factor(s), not eligibility/sample
  changes, leakage, or unreported tuning.

NO-GO if the method only improves Exact20 while reducing InstantiationReady, or
if it wins only on a cherry-picked subset.

## 16. Staged Experiment Plan

Stage A - cheap diagnostic:

- build a read-only diagnostic over current typed-greedy failures that emits
  per-slot role/quantity labels and candidate margins;
- manually inspect 20 targeted failures;
- optionally run a tiny LLM role-label oracle only if deterministic labels are
  inconclusive.
- Stop if role/quantity labels would not change the chosen mention in at least
  10 targeted failures.

Stage B - minimal implementation:

- add a small module for declarative quantity/role features and scoring;
- no broad refactor of the 6,977-line utility;
- route only selected hard cases through the new resolver.
- Stop if unit tests reveal feature extraction is too brittle to define
  reproducibly.

Stage C - 331-query benchmark:

- rerun fresh typed greedy and candidate in the same process/date/code state;
- preserve per-query outputs.

Stage D - paired statistical test:

- exact paired outcomes; McNemar or paired bootstrap with fixed seed.
- Stop if gain is <+2 pp and p is clearly non-significant.

Stage E - error-class ablation:

- report targeted class counts before/after on the same failure taxonomy.

Stage F - runtime:

- `/usr/bin/time` full 331; report wall-clock and RSS.

Stage G - generalization:

- held-out family or schema-name-masked diagnostic.

Stage H - manuscript decision:

- integrate only if success gate passes; otherwise add to `docs/NEGATIVE_RESULTS.md`.

## 17. Required Ablations

For TOP-1:

- current typed greedy baseline;
- factorized local score with greedy decode;
- factorized local score with exact assignment;
- cascade vs always-structured;
- no role factor;
- no unit/quantity factor;
- no clause/locality factor;
- no entity-anchor factor;
- no bound-polarity factor;
- no pairwise structural term;
- top-1 schema only vs low-margin top-k schema cascade.

Each ablation must answer which new information caused any gain.

## 18. Manuscript Impact If Successful

Do not edit the manuscript until a candidate passes the gate.

Sections that would change:

- title/abstract only if the new method becomes the main contribution;
- contributions: from "retrieval-assisted instantiation pipeline" toward
  "factorized role-quantity grounding";
- method: add mathematical objective, pseudocode, cascade trigger;
- complexity: fast path and hard path;
- experiments: fresh 331 results, paired significance, ablations, runtime;
- baselines: preserve stale-submission caveat and current-code rerun separation;
- limitations: scalar-only, fixed catalog, role-feature brittleness;
- conclusion: stronger claim only if generalization/ablation holds.

Claims to weaken regardless:

- any implication that the winning method is a strong general reasoning engine;
- any overclaim that NLP4LP scalar grounding is full NL-to-optimization
  formulation;
- any unsupported superiority over external LLM formulation systems.

## 19. Refactoring Plan

Do not mix scientific changes and large refactoring.

Suggested modules:

- `mention_extraction.py`: digit, word-number, fraction, span/context extraction.
- `quantity_semantics.py`: total/per-unit/rate/percent/currency/bound labels.
- `slot_metadata.py`: expected type, role, unit, bound, entity metadata.
- `local_scoring.py`: current `_choose_token`, opt-role score, factorized score.
- `structured_assignment.py`: Hungarian/min-cost flow/beam/cascade dispatch.
- `repairs.py`: conservative post-assignment repairs.
- `verification.py`: readiness and structural checks.
- `evaluation.py`: shared row/type aggregation and diagnostics.

Migration order:

1. add characterization tests around current outputs for a 20-query fixture;
2. extract pure functions with no behavior change;
3. move schema-side metadata first, because it is easiest to cache;
4. move evaluation aggregation next to remove duplicated assignment-mode loops;
5. only then add TOP-1 scientific changes in a separate patch.

Risk: high, because current performance depends on many small heuristic
interactions. Require byte-for-byte aggregate preservation before any method
change.

## 20. Strict-Metric Addendum (2026-08-13)

The Stage-B follow-up to the TOP-2 candidate showed that ordinary
InstantiationReady can reward incorrect schemas. The strict-readiness
diagnostic in `docs/STRICT_INSTANTIATION_READY_DIAGNOSTIC_2026-08-13.md`
therefore supersedes the original +2 pp ordinary-InstantiationReady success
gate for future main-method claims.

Fresh reference values:

- `tfidf_typed_greedy`: ordinary readiness 257/331, strict readiness 247/331.
- `tfidf_selective_grounding_rerank`: ordinary readiness 265/331, strict
  readiness 249/331.
- `oracle_typed_greedy`: strict readiness 273/331.

Future candidate gates should use StrictInstantiationReady as the primary
native end-to-end proxy and report ordinary InstantiationReady only as a
predicted-schema diagnostic. The selective reranker remains a secondary
retrieval diagnostic, not a main-method improvement.

## 21. Documentation State

This audit should be treated as the current research plan. Older files still
contain some superseded positive wording in `results/unevaluated_methods_evaluation/README.md`
and early `docs/LEARNED_GROUNDING_P0.md` paragraphs; later status documents
correct the interpretation. Do not cite those older positive paragraphs without
the staleness correction.

## 22. Classification

**METHOD_IMPROVEMENT_PLAN_READY**

Single next action:

Run Stage A for TOP-1: build a lightweight per-slot diagnostic over the 54
schema-hit/not-ready typed-greedy cases to verify that role/quantity factors
would change the selected mention in at least 10 targeted failures before any
implementation.
