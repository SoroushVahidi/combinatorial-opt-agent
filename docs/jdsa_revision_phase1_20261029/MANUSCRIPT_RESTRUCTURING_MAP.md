# Manuscript restructuring map — JDSA Phase 1

## Objective and current scope

One coherent diagnostic journal paper, supported by a comprehensive supplement. The argument is the separation of (1) known-catalog schema identification, (2) scalar coverage/type readiness and (3) complete stored-reference scalar agreement. The positive matched-grounding observations and negative controlled-search findings answer different questions and must coexist. No separate publications, new experiments, performance tuning or autonomous-performance claim are proposed here.

The authoritative starting state is `main@a3f0dde397341a41842a30ddb946e7966706351b`. The Stage 1 Abstract and Introduction are byte-preserved. Phase 1 edits Section 5 and its transitions, preserves all original table data, moves archival detail to linked supplementary sections, and adds evidence-preservation coverage for previously omitted investigations. Sections 6–8, declarations, bibliography, figure and original supplement inputs are unchanged. Historical template comments remain. The original `manuscript/jdsa-stage1/` directory is untouched; the candidate is `manuscript/jdsa-phase1/`.

## Section-by-section final-paper proposal

| Section | Material that must stay in the main paper | Supplement destination / consolidation | Phase 1 disposition |
|---|---|---|---|
| Title, Abstract, Introduction | Diagnostic contribution, three research questions, 301/331 and 313/331 retrieval, 265/331 versus 7/331, qualified relationship evidence. Preserve the existing revised opening. | Avoid restating every limitation in each later result. Retain scope where it changes interpretation. | Unchanged verbatim. |
| 2 Related Work | Position against prior modeling/binding distinction, quantity/role grounding, classical assignment and separate execution/formulation validation. No novelty claim for retrieval or matching. | Keep the task-level comparison landscape in S1, not a performance leaderboard. Later shorten repeated task exclusions. | Unchanged; existing cited sources retained. |
| 3 Task and Evaluation Assumptions | Formal inventory construction, reference assistance, catalog exposure, metric denominators, absolute-small-value tolerance, all-key requirement. | S3 input-dependency table, reference-overlay witnesses and reproduction boundaries. A future pass can put Full_G's complete conjunction beside the existing metric definitions. | Unchanged; Section 5 refers to this account instead of repeating it. |
| 4 Methods and Experimental Protocol | Inspectable deterministic pipeline, diagram/algorithm, retrieval conditions, split exposure, saved-record provenance. | Version-specific slot ordering, historical missing records, frozen scorer/candidate details, factorial cell definitions. | Unchanged; no algorithm or implementation edits. |
| 5.1 Schema Retrieval Across Query Variants | Table 1 and the fixed-catalog retrieval finding, including BGE's paired contrast with its adjustment. | S8 retains higher-rank aggregates and their weaker reconstruction grade. | Revised and shortened. |
| 5.2 Readiness Does Not Establish Complete Scalar Agreement | Table 2; same-frame 265/255/7 contrast, BGE and Oracle counts; conditional Exact20 distinction. | Detailed metric definitions stay in Section 3; reconstruction caveats stay in S3. | Revised; central finding moved to Section 5 opening. |
| 5.3 Matched Grounding | **Table 3 remains prominent**: 7/40/37/35/46/9 all-key successes alongside readiness and coverage. Constrained gains/losses and 41 eligible-only versus 40 all-key exception. | Candidate/scoring details and historical comparator versions belong in S3/S8. | Revised; distinguishes useful numerical improvements from non-isolated search effects. |
| 5.4 Residual Boundary | **Table 4 remains prominent**: exhaustive ordered partition and 214/221 conditional numerical failure. | Reproduction history and historical source/reference defects in S3. Occurrence-level illustrative causes are not frequency estimates. | Moved before controlled-search discussion and tightened. |
| 5.5 Fixed-Score Controls | Slot improvements versus unchanged 5/49 full count; joint 3/49 and exact-retention negative result; no equivalence claim. | Full24-cell factorial, pruning partition, tied optima, enumeration certificates, reference sensitivity in S3; all original six pilot methods and candidate ceilings in S8. | Revised; duplicated 331-query method comparison removed from this subsection. |
| 5.6 Extraction and Robustness | Ratio ablation, corrected five-row statistical table, schema-gated sensitivity and qualified sanitization result. | S7 preserves the displaced statistical-method correction history; S2 retains sanitization rates; S8 preserves exploratory reranking history. | Consolidated; no new p-values/CIs generated. |
| 5.7 Restricted Execution and Limits | Short scope-qualified contextual result and transition to separate LLM evidence. | S2 full60/20 tables; S7 exact proxy predicates, full269 rates, runtime context and occurrence-level cautions. | Revised; all displaced details preserved. |
| 6 LLM-Based Comparative Evidence | Future main text should retain a compact common18 contextual table and a substantive relationship-validity/agreement finding, explicitly separate from scalar performance. | S4 should carry all generation configurations, case outcomes, terminal-objective/precision witnesses. S6 should carry request/recovery ledger, denominators and disagreement examples. D1 construction details belong in S8, with a concise later methods paragraph. | **No large rewrite this phase.** Section6 and relationship input unchanged. |
| 7 External Validation, Discussion, Limitations | Future synthesis: what metric disagreement establishes, which evidence is conditional, reference/exposure limits, external extraction as component-only transfer. | Keep OptMATH aggregate definitions and unavailable corrected-CI inputs in an expanded supplement; G1–G3 details stay S5; neural/NL4Opt audits S3/S8. | **Unchanged this phase.** Avoid converting an audit into a new performance experiment in the later pass. |
| 8 Conclusion | Future concise answer to RQ1–RQ3: numerical agreement adds information beyond retrieval/readiness; matched improvements do not prove a search-only cause; output agreement is not semantic truth. | Do not repeat every common18 protocol or all provider counts here; preserve them in their results/supplement locations. | **Unchanged this phase.** Future conclusion can explicitly reflect the 46/331 result without implying universal superiority. |
| Declarations / availability | Accurate author facts, AI assistance, public-versus-local evidence and upstream access conditions. | File-level release inventory should distinguish published sources from local-only verification material. No raw restricted corpus or private author log in this revision. | Existing wording retained; M5-specific publication/author-review statements need later reconciliation with actual releases. |
| S1–S6 | Existing provenance, all outcomes, negative/inconclusive findings and uncertainty. | Preserve all original files and table values. | Existing substantive content unchanged; cross-document PDF filenames updated only. |
| S7 Preserved Results Details | Exact moved definitions and archival evidence. | Restricted60/269/20 protocol, timing, occurrence boundary, statistical history. | Added with linked main references. |
| S8 Additional Evidence | Historical grounding variants, candidate/pilot detail, weak-label/auxiliary training, archived two-stage LLM observations, D1 construction and explicit non-results. | Distinct historical cohorts; safe summaries only. Detailed metric JSON and source hashes accompany the register. | Added to prevent omission during compression. |

## Claims that require precise wording

1. **Numerical agreement, not universal correctness:** Full_G is a conjunctive stored-reference scalar criterion with a type gate and complete key coverage. It excludes vectors, constraints, full formulation equivalence and independently established source semantics. The tolerance is absolute0.20 below unit magnitude, not always relative20%.
2. **Positive matched results:** 46/331 is the largest observed all-key count among these saved configurations. It is not an isolated effect of search, a prespecified multiplicity-adjusted numerical-superiority result, or an autonomous deployment score.
3. **Negative search findings:** unchanged aggregate5/49 includes one gain and one loss; exact retention9/49 and selected3/49 apply to the frozen objectives. A null observed difference is not proof of equivalence or that defensible constraints cannot help.
4. **Readiness and Exact20:** all331 queries enter ordinary/Strict readiness; Exact20 excludes noncomparable cases and has differing denominators. Lower coverage can accompany higher conditional Exact20. No mixed-denominator ranking.
5. **Statistics:** group-percentile intervals are marginal, exact swap tests have their exchangeability assumption, and Holm16/240 families are separate and retrospective. Earlier uncentered-bootstrap significance and shifted-null confidence interpretations are not reused. No global confirmatory familywise guarantee is claimed.
6. **Historical variants:** older .5287/.7764/.8006 readiness values identify different code/configuration cohorts. Do not collapse them into a single baseline or interpret the old negative-results ledger as disproving later all-key improvements.
7. **Learning and supervision:** actual training is retained as completed; missing checkpoints, weak labels, tiny auxiliary tests, evaluation-origin splits and conflicting rule comparators limit conclusions. Proposed MiniLM/NL4Opt runs have no fabricated outcomes.
8. **LLM evidence:** execution, terminal-objective agreement, structural JSON validity, model agreement and human semantic correctness remain distinct. Recovery changes output contracts and complete-case membership. PaMOP precision remains unresolved; ORLM execution remains unavailable.
9. **D1:** author-validated, AI-assisted construction is legitimate descriptive evidence. It is not independently blinded gold or a completed provider benchmark; it does not supply semantic truth for the separate217 relationships.
10. **Robustness and autonomy:** number/stopword stripping does not remove template exposure; G1/G2/G3 are information-boundary audits. No correction factor from assisted to autonomous performance is identifiable.

## Unresolved source evidence

- Original headline scalar mappings were not authenticated. Query metrics support reaggregation, not corrected-reference reevaluation; later cached choices cannot replace them.
- Original sanitized-condition predictions, complete60/269/20 membership/outcome files and some execution logs are absent. Their aggregate tables remain qualified rather than removed.
- BGE runtime hash seed/environment and some historical LLM prompts/model-serving defaults are incompletely retained.
- PaMOP displayed precision cannot establish its original absolute1e-6 agreement predicate. No substitute aggregate is inferred.
- OptMATH's corrected per-instance counts are missing, preventing an independent reproduction of its historical corrected confidence interval; set data do not identify occurrence recall.
- Early report/table conflicts and learned-rule comparator conflicts remain recorded. This phase prefers saved rows without inventing absent original measurements.
- Local-only audit/checkpoint/response bundles are not silently called publicly reproducible. The public register supplies content hashes and evidence grades; restricted source data and private author logs stay excluded.

## Next phase recommendation — not performed

After author review of this PDF and register, revise Sections6–8 around the shared diagnostic question, integrate the already completed D1 construction at its disclosed evidence level, and reconcile availability/declaration wording with the actual public artifacts. Preserve common18 and relationship studies as distinct evidence blocks; summarize auxiliary material without losing its supplement destination. Comprehensive JDSA formatting remains a later bounded pass. No new experiment or second publication is needed to execute that editorial plan.

## Exhaustive investigation-to-destination crosswalk

Each registered investigation appears below. Blocked/proposed work is preserved as status or limitation, not presented as a completed scientific performance finding.

| ID | Investigation | Destination / exclusion from performance claims |
|---|---|---|
| E01 | Lexical schema retrieval and ranked-list diagnostics | Main §5.1/Table 1; additional ranked-list cells in S8. |
| E02 | Pretrained dense retrieval | Main §5.1; provenance in S3. |
| E03 | Four-arm scalar-grounding benchmark | Main §5.2/Table 2; evidence boundary in S3. |
| E04 | Ratio-word extraction ablation | Main §5.6/Table 5; precursor diagnosis in S8. |
| E05 | Matched grounding alternatives | Main §5.3/Table 3, with numerical endpoint prominent. |
| E06 | Residual-error and full-scalar analyses | Main §5.4/Table 4. |
| E07 | Frozen-score factorial assignment | Main §5.5 summary; full table/certificates in S3.1. |
| E08 | Cached joint assignment pilot | Main §5.5; all six methods and no-op finding in S8/S3.1. |
| E09 | Exact objective and beam-pruning diagnosis | Main §5.5; detailed partition/certificates in S3.1. |
| E10 | Value/type-distinct exact enumeration | Main §5.5 negative result; detail S3.1. |
| E11 | Cached candidate availability and Stage-A error accounting | S8 candidate table and S3 context; main only search-failure connection. |
| E12 | Reference conventions and frozen-prediction sensitivity | Keep S3.2; concise main qualification only. |
| E13 | Corrected paired significance and factorial multiplicity | Main §5.6; procedural history S7; factorial details S3.1. |
| E14 | Sanitization and overlap robustness | Main concise qualification; complete historical cells S2. |
| E15 | Historical 60-case coverage/type proxy | Main §5.7 summary; S2 full table and S7 definitions. |
| E16 | Historical extended269 proxy | Main §5.7 summary; complete rates preserved verbatim in S7. |
| E17 | Historical20-case reference-program execution | Main §5.7; S2 table and S7 execution definitions. |
| E18 | Common18 formulation and terminal-objective audit | Retain §6 unchanged this phase; future concise main context, full S4 case ledger. |
| E19 | 217-item three-service relationship judgments and recovery | Keep unchanged this phase; later concise main semantic-evidence result, detailed S6. |
| E20 | D1 semantic contrast construction and author audit | New S8 descriptive construction/provenance summary; later integrate §6/RQ3. Private logs excluded. |
| E21 | NL4Opt proposal, annotation and template audit | Keep S3.6 and limitations; register preserves finite QC packet. |
| E22 | Historical DistilRoBERTa mention-slot learning | Keep negative absolute observations/ambiguity in S3.5; detailed counts S8/register. |
| E23 | P0 learned feature scorer and assignment | Keep S3.5; dev ablation detail S8. |
| E24 | GAMS auxiliary weak-label training | Preserve exact small-sample results in S8; caveat S3.5. |
| E25 | Targeted synthetic auxiliary training | Preserve in S8 and supervision limitations. |
| E26 | Historical fine-tuned schema retrieval | S8 historical context; no scalar performance table. |
| E27 | OptMATH external numeric extraction and audit | Retain unchanged now; future main short transfer result, detailed supplement audit. |
| E28 | G1 autonomous inventory feasibility | Retain concise boundary in §7; S5 full context. |
| E29 | G2 fixed-slot gold-filter intervention | Retain as methodological limitation, not ablation score. |
| E30 | G3 schema-only participation sets | Retain S5 and short limitations summary. |
| E31 | Historical global compatibility family | S8 full legacy variant table; no main headline substitution. |
| E32 | Historical relation-aware linking ablations | S8 legacy table with all three variants. |
| E33 | Historical ambiguity-aware grounding | S8 legacy table and abstention interpretation. |
| E34 | Baseline staleness, hierarchical search and classical alternatives | S8 twelve-method table and correction history. |
| E35 | Top-k schema and reranking feasibility | S8 ceilings and false-positive boundary. |
| E36 | Selective schema-grounding reranking | Main §5.6; supplemental context S7/S8. |
| E37 | Role/quantity Stage-A separability and cascade | S8 diagnostic negative result. |
| E38 | Strict-failure diagnosis and quick-fix precursor | S8 distinguish prototype and unexecuted alternatives. |
| E39 | Archived OpenAI two-stage schema/slot baseline | S8 archival table; no main leaderboard or conflation with §6. |
| E40 | Archived Mistral two-stage orig baseline | S8 archival table with provenance limitations. |
| E41 | Optional Gemini scalar-benchmark rerun | S8 status-only; no accuracy cell. |
| E42 | CloudRift Qwen relationship extension | S8 status-only; no scientific performance finding. |
| E43 | Independent relationship reference-label protocol | Keep future prerequisite; S8 status ledger, no new reviewer task. |
| E44 | Earlier synthetic global-consistency evaluation | S8 historical status and explicit exclusion from scientific benchmark comparison. |
| E45 | Neural/scorer scaffolds and proposed external supervision | Keep S3.5 and S8 status-only. |
| E46 | Historical prediction recovery and source integrity | Keep reproduction boundary and unresolved-issues register. |
| E47 | Public metrics and release provenance check | Keep data-access boundaries; source inventory records local-only evidence. |
