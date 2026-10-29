# Experimental evidence register — JDSA Phase 1

Authoritative manuscript: `main` at `a3f0dde397341a41842a30ddb946e7966706351b`, published from Stage 1 source commit `67fb2b51700429e0338d2ac63eeebd321095846e`. The 32-page PDF SHA-256 is `c58a8cacac78e1d969dc11e5fd19581f49bc1905cf8cceb91d486aaebca5b0ee`; `main.tex` is `6260a13991dd6c7b79632b8c2c2e692735edb6756ed5240598694177f8c9af85`. Phase 1 starts from these bytes, not an older SNCS source. Abstract and Introduction are preserved verbatim.

This register contains **47 investigation groups**, not 47 independent datasets or completed performance experiments. Status is assigned per entry, including partial, blocked and proposed work. No new experiment, prediction, model call, training, solver benchmark, Slurm job or dataset collection was performed. Offline aggregation reads already saved observations only.

## Scope, precedence and reconstruction grades

The full Stage 1 main manuscript and every supplementary input were read. Discovery covered current `results/`, research worktree evidence manifests, prior metric/correction/archival audits, M1–M5 handoffs and D1/G1–G3 records. Existing comprehensive method/history audits were reused rather than rerunning every historical source scan. The register covers relevant located campaigns, not an assertion that inaccessible private archives contain no other work. Source files are not moved or deleted.

Evidence precedence: authenticated saved observations and corrected predicates; then verified correction ledgers; then historical summaries; then narrative claims. A later prose summary does not override saved measurements. In particular, some ESWA summary cells differ from the preserved CSVs (typed orig 174 in one summary, 175 in saved rows). The additional supplement uses the saved rows, labels their historical vintage, and preserves the conflicting summary as a limitation. Earlier neural baseline comparisons conflict with stdout. Invalid old bootstrap p-values and stale-baseline superiority claims are not promoted into new findings.

- **R1:** retained predictions/choices or source records support independent outcome checks within the stated task; does not promise regeneration of missing environments.
- **R2:** saved per-query metrics support reaggregation/paired checks, but original predictions cannot be reevaluated.
- **R3:** aggregate, report or audit-supported; independent reconstruction of the underlying observation is incomplete.
- **R0:** no completed outcome exists or required evidence is absent. No fabricated numerical finding.

All-key scalar success additionally requires schema match, every reference scalar key, compatible types and the stored inclusive tolerance `abs(pred-gold)/max(1,abs(gold)) <= .20`. It is not exact equality or source-semantic truth. Exact20 averages only comparable schema-hit query fractions. Reference-assisted and exposed-catalog conditions remain material throughout.

## Source identities and public-release boundary

Every source locator below resolves through [EVIDENCE_SOURCE_MANIFEST.csv](EVIDENCE_SOURCE_MANIFEST.csv), which supplies the exact repository-relative file, SHA-256, source path's last commit and worktree base. A path commit is an archival identity, **not automatically the execution commit**. Execution commits are named separately when known. An uncommitted local report has a content hash and base only; its base is never claimed to contain it. Local-only evidence has not been bulk-published. This branch publishes the register and safe numeric summaries, not raw restricted corpora, private author-review logs, provider credentials, account identifiers or full response bundles.

`current` means the verified main snapshot inherited by this revision. Other locator prefixes identify the specific research worktree in the manifest. Existing source citations are preserved; this phase does not invent or newly validate bibliographic claims. Prior M3 privacy/publication assumptions superseded by M5 are not reinstated as scientific blockers. Missing public evidence remains an access limitation rather than a reason to erase supported findings.

The reproducible [offline metric check](verify_saved_metrics.py) and [its output](SAVED_METRIC_CHECKS.json) cover ten core arms, twelve retrieval cells, 72 historical campaign CSVs and four optional LLM CSVs. They generate no replacement observations. The [restructuring map](MANUSCRIPT_RESTRUCTURING_MAP.md) assigns every ID below a publication destination.

## E01 — Lexical schema retrieval and ranked-list diagnostics

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | BM25, TF–IDF, LSA; fixed benchmark-derived catalog; target ID scoring |
| Dataset / exact population | 331 NLP4LP queries in each of orig/noisy/short; 335-document main catalog |
| Available results | Orig hits BM25 292, TF–IDF 301, LSA 280; noisy 295/299/294; short 254/258/253. Saved P@5, MRR@10, nDCG@10 and Coverage@10 remain separate aggregate evidence. |
| Evidence files / commits | `current:results/eswa_revision/01_retrieval/retrieval_results.json`; `current:results/paper/nlp4lp_downstream_per_query_orig_tfidf.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 for top-1 IDs; R3 for higher-rank aggregates without full ranked lists. Twelve main retrieval cells checked offline. |
| Scientific limits | Known-catalog identification; evaluation-source templates and repeated development exposure. Sanitization uses another index. |
| Current location | §5.1, Table 1; S1 |
| Revised destination | Main §5.1/Table 1; additional ranked-list cells in S8. |

## E02 — Pretrained dense retrieval

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Off-the-shelf BGE-M3, no benchmark fine-tuning |
| Dataset / exact population | Same 331 queries × three variants; 335 templates |
| Available results | 313/331 orig, 312/331 noisy, 270/331 short; mean .9013. 15 BGE-only and three TF–IDF-only orig hits. |
| Evidence files / commits | `current:results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv`; `current:results/dense_retrieval_bge_m3/model_metadata.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 top-1 outcomes; generation replay incomplete. |
| Scientific limits | Metadata does not authenticate runtime device, serving weights, hash seed or full environment; no unseen-template claim. |
| Current location | §5.1, Table 1 |
| Revised destination | Main §5.1; provenance in S3. |

## E03 — Four-arm scalar-grounding benchmark

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Typed greedy with baseline/ratio extraction; TF–IDF, BGE-M3 or reference-schema control |
| Dataset / exact population | 331 orig queries per arm; eligible scalar inventories depend on selected schema and query reference |
| Available results | Baseline/ratio/BGE/Oracle coarse counts 257/265/273/281; Strict 247/255/265/281; full-all-key 6/7/3/8. Exact20 comparable denominators 291/291/303/320. |
| Evidence files / commits | `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/core331.json`; `current:results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2: all aggregates independently checked from ten saved metric CSVs; no original scalar-map reevaluation. |
| Scientific limits | Gold-assisted participation; conditional Exact20 differs from all-query success; Oracle is not an upper bound on other grounders. |
| Current location | §5.2, Table 2; S3 |
| Revised destination | Main §5.2/Table 2; evidence boundary in S3. |

## E04 — Ratio-word extraction ablation

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Baseline versus extraction of multiplicative ratio words; retrieval unchanged |
| Dataset / exact population | Same 331-query TF–IDF frame |
| Available results | Strict 247→255 (eight gains, zero losses); coarse 257→265; full scalar tolerance 6→7; Exact20 .2449→.2614. |
| Evidence files / commits | `current:results/final_resubmission_method/metrics.json`; `current:results/final_resubmission_method/transitions.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved metric rows and transitions. |
| Scientific limits | Improves coarse diagnostics; does not imply eight additional numerically correct problems. |
| Current location | §5.5/Table 5 |
| Revised destination | Main §5.6/Table 5; precursor diagnosis in S8. |

## E05 — Matched grounding alternatives

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Typed, constrained, max-weight, role repair, search-structured, semantic IR; fixed assistance rule but several mechanism changes |
| Dataset / exact population | 331 identical orig queries, TF–IDF retrieval held fixed |
| Available results | Full-all-key 7/40/37/35/46/9. Constrained eligible-only 41 excludes query 102 (3/7 keys). Constrained gains/losses 36/3. Coarse 265/252/246/244/233/237. |
| Evidence files / commits | `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/core331.json`; `stage1:results/joint_grounding_2026-10-27/completed_run_reanalysis.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2; exact all-key counts, means and conditional denominators checked from saved CSVs. |
| Scientific limits | Candidates, contexts, scores, abstention and decoding differ; no isolated search effect. Descriptive full-success ordering; no new significance test. |
| Current location | §5.3/Table 3; grounding_summary.tex |
| Revised destination | Main §5.3/Table 3, with numerical endpoint prominent. |

## E06 — Residual-error and full-scalar analyses

| Field | Evidence-qualified record |
|---|---|
| Status | Completed offline investigation |
| Purpose / conditions | Ordered partition: schema, coverage, type, value, success |
| Dataset / exact population | 331 cases per headline arm; ratio gate-passing subset 221 |
| Available results | Ratio 30/31/49/214/7; baseline 30/48/61/186/6; BGE 18/33/49/228/3; Oracle 0/35/49/239/8. Ratio conditional failure 214/221; BGE 228/231; Oracle 239/247. |
| Evidence files / commits | `current:results/final_resubmission_method/residual_error_analysis_2026-08-27.json`; `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/core331.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2: partitions reconstructed from query metrics. |
| Scientific limits | Stage-of-failure accounting, not adjudicated semantic cause frequencies; the 0.20 rule is absolute for magnitudes below one. |
| Current location | §5.5/Table 4 |
| Revised destination | Main §5.4/Table 4. |

## E07 — Frozen-score factorial assignment

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | 2 candidate pools ×3 scorers ×2 decoders ×2 reuse policies; identical edge matrices within decoder pairs |
| Dataset / exact population | 49 primary queries/277 gold slots; 64 collected cases =49 primary+12 development+1 overlap+2 reference-only |
| Available results | Context exact slots typed 106→136, rich 100→133; full tolerance 5→5/49, one gain/one loss. Context decoder adjusted p=1 in 240-test family. All 24 cells retained. |
| Evidence files / commits | `stage1:docs/wulver_grounding_attribution_2026-10-27/FACTORIAL_TABLE.md`; `stage1:docs/wulver_grounding_attribution_2026-10-27/RESULTS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 for saved choices/matrices; independent previous checks retained. No decoder/solver rerun in this phase. Execution job 1462366; implementation 4db7265, archival result 45dd183. |
| Scientific limits | Partially assisted template inventory; ordinal score cardinalization; exploratory, exposed small pilot. No whole-instance equivalence. |
| Current location | §5.4; S3.1 |
| Revised destination | Main §5.5 summary; full table/certificates in S3.1. |

## E08 — Cached joint assignment pilot

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Fixed top-8 pools, beam width32/top10; independent type/text/unit and joint controls |
| Dataset / exact population | Original 50 queries/280 gold scalars; primary slice excludes query 291:49/277 |
| Available results | Original six-method full-tolerance counts 1/2/2/3/3/3; local/joint units 3/49 on primary slice. Zero explicit parameter-only relations activated; constraint variant is a no-op. |
| Evidence files / commits | `stage1:results/joint_grounding_2026-10-27/pilot_metrics.json`; `stage1:results/joint_grounding_2026-10-27/pilot_summary.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 retained mappings/traces; protocol/scorer committed before execution (0288c52); archive df408d3. |
| Scientific limits | No-op arithmetic branch is not a test of meaningful constraints; source metadata remains assisted; differences are conditional. |
| Current location | §5.4; S3.1 |
| Revised destination | Main §5.5; all six methods and no-op finding in S8/S3.1. |

## E09 — Exact objective and beam-pruning diagnosis

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Frozen cached local/pairwise objective; exact MILP and correctness-restricted oracle diagnoses |
| Dataset / exact population | 50 pilot cases; 32 pool-compatible cases |
| Available results | 18 pool/inventory failures;14 prefix-pruning;6 complete-beam-pruning;4 top10-cut;5 retained-not-selected;3 selected. Among32:23 strict objective failures,6 ties,3 correct optima. Exact retention9/50 versus beam8/50. |
| Evidence files / commits | `stage1:docs/wulver_grounding_attribution_2026-10-27/RESULTS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 saved traces, frontiers and certificates at result commit45dd183; no new solves. |
| Scientific limits | Correctness-restricted solves are oracle diagnostics. Ties and retained reference criterion do not establish source semantics. |
| Current location | S3.1; §5.4 |
| Revised destination | Main §5.5; detailed partition/certificates in S3.1. |

## E10 — Value/type-distinct exact enumeration

| Field | Evidence-qualified record |
|---|---|
| Status | Completed negative follow-up |
| Purpose / conditions | Frozen scores/pools/top1; distinct numerical/type rather than mention-index mappings |
| Dataset / exact population | 49 primary queries; nine lists required additional historical enumeration |
| Available results | All49 lists certified;40 reused;23 historical extra solves for9 lists. Retention9/49 and selected3/49; zero gains/losses; empirical paired interval[0,0], p=1. |
| Evidence files / commits | `stage1:docs/grounding_followup_2026-10-27/RESULTS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 retained checkpoints; execution4d867f3/job1462557, result47c4733. |
| Scientific limits | Degenerate empirical interval is not population equivalence. No new run or full-dataset expansion. |
| Current location | §5.4; S3.1 |
| Revised destination | Main §5.5 negative result; detail S3.1. |

## E11 — Cached candidate availability and Stage-A error accounting

| Field | Evidence-qualified record |
|---|---|
| Status | Completed diagnosis |
| Purpose / conditions | Saved candidates; exact/tolerance pool ceilings with and without type gates |
| Dataset / exact population | 331 queries/1847 gold scalars; cache covers311 queries/1705 rows |
| Available results | All-pool value recall1467 exact/1473 tolerance; whole numerical ceilings203/206 of331. Top8 typed whole ceiling139/331. Pilot type-aware ceiling32/50; initial-pool error accounting preserved. |
| Evidence files / commits | `stage1:results/joint_grounding_2026-10-27/candidate_recall_summary.json`; `stage1:results/joint_grounding_2026-10-27/error_category_summary.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 retained candidate sets/selected values, conditioned on available cache. |
| Scientific limits | Product-of-pools oracle ceiling allows reuse, not selected performance. Missing20 queries stay in denominator; equal values do not identify occurrences. |
| Current location | S3.1/S3.3, partly summarized |
| Revised destination | S8 candidate table and S3 context; main only search-failure connection. |

## E12 — Reference conventions and frozen-prediction sensitivity

| Field | Evidence-qualified record |
|---|---|
| Status | Completed correction-only audit |
| Purpose / conditions | Source/equation witnesses; separate numerical_reference_v2 overlay; no original reference replacement |
| Dataset / exact population | 12 flagged slots;10 corrected fractional references; query206 keeps two percentage-point values |
| Available results | Stage-A exact slots348→356, tolerance401→409 of1847; full6/331 unchanged. No affected primary49 case; development177 adds no full success. 394 versus401 reconciled by seven metadata-excluded selections. |
| Evidence files / commits | `stage1:results/sncs_corrections_2026-10-27/cached_stage_a_reference_summary.json`; `stage1:results/sncs_corrections_2026-10-27/cached_inventory_scope_reconciliation.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 saved Stage-A values; R0 unavailable for missing headline scalar maps. |
| Scientific limits | Overlay not maintainer-certified; query102 alias and query177 coefficient issues remain; historical7/331 is not rescored. |
| Current location | S3.2 |
| Revised destination | Keep S3.2; concise main qualification only. |

## E13 — Corrected paired significance and factorial multiplicity

| Field | Evidence-qualified record |
|---|---|
| Status | Completed statistical correction |
| Purpose / conditions | Existing group bootstrap B10000/seed42; exact count-swap tests; Holm16; separate factorial Holm240 |
| Dataset / exact population | 331 paired queries/327 heuristic groups; separate49-query factorial |
| Available results | Ratio-baseline Strict p=.0078125, Holm .0301514; BGE retrieval p=.00753784, Holm .0301514; readiness alternative maximum adjusted .005859375. Main five CIs preserved. |
| Evidence files / commits | `stage1:results/sncs_corrections_2026-10-27/corrected_paired_statistics.csv`; `stage1:docs/sncs_corrections_2026-10-27/STATISTICAL_CORRECTIONS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved gate labels/group definitions support checks; retained corrected statistics checked, no new intervals generated. |
| Scientific limits | Post-outcome family, marginal CIs; exchangeability assumption; old bootstrap sign-tail and shifted-null intervals are not valid replacements. |
| Current location | §5.7/Table 6; S3.1 |
| Revised destination | Main §5.6; procedural history S7; factorial details S3.1. |

## E14 — Sanitization and overlap robustness

| Field | Evidence-qualified record |
|---|---|
| Status | Completed, aggregate-supported in part |
| Purpose / conditions | Remove numbers/stopwords;256-dimensional LSA; lexical Jaccard stratification |
| Dataset / exact population | 331 queries/cell;331-document diagnostic index;327 high-overlap/4 medium |
| Available results | TF–IDF baseline .9063→.9124 with stopwords removed; BM25 .8852→.9063; LSA .8459→.9184. High-overlap TF–IDF .9174. |
| Evidence files / commits | `current:results/eswa_revision/14_reports/FINAL_REVISION_EXPERIMENT_SUMMARY.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 full sanitization cells; no authenticated per-condition predictions. |
| Scientific limits | Different index from main335. Does not rule out shortcuts, template dependence or exposure. Old report causal interpretation rejected. |
| Current location | §5.7; S2/Table S4 |
| Revised destination | Main concise qualification; complete historical cells S2. |

## E15 — Historical 60-case coverage/type proxy

| Field | Evidence-qualified record |
|---|---|
| Status | Completed, aggregate-supported |
| Purpose / conditions | Optimization-role relation repair; metadata presence/template predicates |
| Dataset / exact population | First60 orig queries with2–8 gold scalar parameters |
| Available results | TF–IDF/BM25/Oracle schema .9333/.9000/1; proxy .7500/.7333/.7667; completion .7500/.7333/.7833. |
| Evidence files / commits | `current:manuscript/jdsa-stage1/supplement.tex`; `phase3:docs/manuscript_scientific_revision_phase3_2026-10-28/FINAL_EXPERIMENTAL_TABLE_AUDIT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 reports only; original membership and per-case outcomes unavailable. |
| Scientific limits | Proxy passes empty inventory; neither numerical correctness nor formulation validation. Some overlap with269; not random. |
| Current location | §5.9; S2/Table S5 |
| Revised destination | Main §5.7 summary; S2 full table and S7 definitions. |

## E16 — Historical extended269 proxy

| Field | Evidence-qualified record |
|---|---|
| Status | Completed, aggregate-supported |
| Purpose / conditions | Same relation-repair proxy, no solver execution |
| Dataset / exact population | 269 code-available queries;807 method-query evaluations |
| Available results | TF–IDF/BM25/Oracle schema .9368/.9257/1; proxy .8141/.8104/.8253; completion .6654/.6580/.6840. Events69 missing-slot,82 type,267 template-incompletion. |
| Evidence files / commits | `current:manuscript/jdsa-stage1/main.tex`; `phase3:docs/manuscript_scientific_revision_phase3_2026-10-28/FINAL_EXPERIMENTAL_TABLE_AUDIT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3; manifest/outcome rows absent; do not deduce counts from rounded rates. |
| Scientific limits | Events overlap; not partition of269; code availability does not imply execution; some shared60-case examples. |
| Current location | §5.9 |
| Revised destination | Main §5.7 summary; complete rates preserved verbatim in S7. |

## E17 — Historical20-case reference-program execution

| Field | Evidence-qualified record |
|---|---|
| Status | Completed, aggregate-supported |
| Purpose / conditions | SciPy HiGHS shim; selected reference code/default values; filled integers rounded |
| Dataset / exact population | First20 programs passing literal compatibility filter;40 method-case evaluations |
| Available results | TF–IDF/Oracle executable .95/.95; success/feasibility/objective .80/.75. Two reported shim IndexErrors across40; raw logs unavailable. |
| Evidence files / commits | `current:manuscript/jdsa-stage1/supplement.tex`; `phase2:docs/historical_scalar_assignment_recovery_2026-10-28/THREE_TABLE_REASSESSMENT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 only; original membership, outcomes and scalar maps not recovered. |
| Scientific limits | Gold-assisted execution; success and feasibility share status predicate; one-case gap not tested. |
| Current location | §5.9; S2/Table S6 |
| Revised destination | Main §5.7; S2 table and S7 execution definitions. |

## E18 — Common18 formulation and terminal-objective audit

| Field | Evidence-qualified record |
|---|---|
| Status | Completed generation; ORLM execution blocked; PaMOP agreement inconclusive |
| Purpose / conditions | PaMOP reconstruction/GPT5.4; OptMATH and ORLM released-model adaptations; GPT5.4 zero-shot; distinct prompts/solvers/budgets |
| Dataset / exact population | Same18 source problems;31 successful Python runs |
| Available results | Parse18/18 for Python; OptMATH15 runs/11 agreements, GPT16/12. PaMOP13 runs,11 references:2 robust discrepancies/9 unresolved,2 no-reference. ORLM no execution. Nine objective fields/seven labels corrected. |
| Evidence files / commits | `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/common18.json`; `m3:docs/manuscript_m3_2026-10-28/COMMON18_CORRECTED_CASES.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 archived code/stdout support terminal-proxy reaggregation; missing prompts/environment prevent full replay. |
| Scientific limits | Round-first5% predicate for Python; PaMOP1e-6 cannot be certified from displayed precision. No selective0/2 or inferred9/11, no ranking. |
| Current location | §6.1–6.2/Table 8; S4 |
| Revised destination | Retain §6 unchanged this phase; future concise main context, full S4 case ledger. |

## E19 — 217-item three-service relationship judgments and recovery

| Field | Evidence-qualified record |
|---|---|
| Status | Completed; semantic accuracy unmeasured |
| Purpose / conditions | Gemini/DeepSeek through io.net/Cohere; original, offline repair, targeted historical replacement; changed output contracts |
| Dataset / exact population | 217 relationships clustered in16 NL4Opt train problems;651 selected judgments |
| Available results | Original valid215/180/105; final217/217/197=631/651.59 offline repairs+72 replacements. Complete197, kappa .3708071786; pairs213/217,123/197,120/197;119 determinate unanimous+77 disagreement+1 unresolved. |
| Evidence files / commits | `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/llm217/llm217_validation.json`; `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/LLM_217_INDEPENDENT_VALIDATION.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 locally retained frozen responses/packets; M5 independent verifier5120 checks recorded. Public candidate not automatically released here. |
| Scientific limits | Convenience sample, clustered items, unequal recovery; model assertions not human truth. Serving weight hashes unknown. |
| Current location | §6.3; S6 |
| Revised destination | Keep unchanged this phase; later concise main semantic-evidence result, detailed S6. |

## E20 — D1 semantic contrast construction and author audit

| Field | Evidence-qualified record |
|---|---|
| Status | Completed construction/audit; provider pilot not executed |
| Purpose / conditions | Fixed candidate/source contrast; author validation with AI assistance; shared fixture lineage |
| Dataset / exact population | 12 accepted pairs/24 samples;8 categories;10 meaning changes/2 paraphrases |
| Available results | 12 correct/12 incorrect labels; no excluded/unresolved pairs.20 final labels directly AI-exposed; archived10 template groups are not certified independent families. |
| Evidence files / commits | `d1:docs/d1_final_pair_audit_2026-10-29/evidence/audit_verification.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 public summary; detailed private author logs remain local and are not published. |
| Scientific limits | Not independently blinded gold, no provider accuracy or causal explanation of7/331. Prior zero-gold gate does not erase later disclosed author approval. |
| Current location | Abstract/Introduction; full details absent |
| Revised destination | New S8 descriptive construction/provenance summary; later integrate §6/RQ3. Private logs excluded. |

## E21 — NL4Opt proposal, annotation and template audit

| Field | Evidence-qualified record |
|---|---|
| Status | Completed audit; training blocked |
| Purpose / conditions | Occurrence/declaration witnesses; conservative template quarantine; proposals not accepted labels |
| Dataset / exact population | 713 pinned NL4Opt train problems;100-problem/954-row review packet |
| Available results | 3114 positive proposals/3246 ambiguous links;68 problems with proposals for every declared parameter;399 quarantined/314 unflagged but uncertified;0 approved training relations. |
| Evidence files / commits | `labels:docs/nl4opt_label_audit_2026-10-27/NL4OPT_LABEL_AUDIT.md`; `labels:docs/nl4opt_label_audit_2026-10-27/TEMPLATE_INDEPENDENCE_AUDIT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 local source/audit outputs; hashes and pinned revision49f1e0d; not independent semantic annotation. |
| Scientific limits | First-match equality cannot resolve occurrence roles; no negatives/NULL gold; text-screen negatives do not certify template independence. |
| Current location | S3.6; §7.2 |
| Revised destination | Keep S3.6 and limitations; register preserves finite QC packet. |

## E22 — Historical DistilRoBERTa mention-slot learning

| Field | Evidence-qualified record |
|---|---|
| Status | Completed training; comparative interpretation inconclusive |
| Purpose / conditions | Weak first-equal-value labels, encoder+linear head; largest500 steps, batch8,lr2e-5,seed42 |
| Dataset / exact population | Jobs854608,854613,854626; largest230/50/50 internal eval-origin split,9729/2230/2339 pairs |
| Available results | 854613:5/59 selections,0/8 legacy exact.854626:51/280 overall,51/259 aligned,0/45 legacy exact,19/280 type.854608 used test-as-train fallback. |
| Evidence files / commits | `supervised:docs/supervised_grounding_feasibility_2026-10-27/EXISTING_METHODS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 retained stdout proves fitting/checkpoint writes; original checkpoints, exact run SHAs and predictions missing. |
| Scientific limits | Conflicting baseline summary; possible untrained-head nominal rule path; five test cases omitted from exact denominator; split leakage. No claim learning generally fails. |
| Current location | S3.5 |
| Revised destination | Keep negative absolute observations/ambiguity in S3.5; detailed counts S8/register. |

## E23 — P0 learned feature scorer and assignment

| Field | Evidence-qualified record |
|---|---|
| Status | Completed |
| Purpose / conditions | Logistic regression26 features incl frozen MiniLM; independent/exact matching/repair; no encoder fine-tune |
| Dataset / exact population | Oracle-schema50 test cases;245 development slot groups |
| Available results | Test coarse .80/.78/.78 versus retained typed comparator .86. Dev combined LR115/245,trees114/245,engineered109/245,embedding58/245. |
| Evidence files / commits | `current:results/learned_grounding_p0/test_results.csv`; `current:results/learned_grounding_p0/ablation_results.csv`; `supervised:docs/supervised_grounding_feasibility_2026-10-27/EXISTING_METHODS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved test metrics; dev counts retained; no new fitting. |
| Scientific limits | First-value labels, evaluation-origin splits and group crossings; no calibrated confidence or general learning-versus-rule conclusion. |
| Current location | S3.5 |
| Revised destination | Keep S3.5; dev ablation detail S8. |

## E24 — GAMS auxiliary weak-label training

| Field | Evidence-qualified record |
|---|---|
| Status | Completed, underpowered |
| Purpose / conditions | Job854616;50 auxsteps then44 realsteps |
| Dataset / exact population | 17 models/1135 pairs/85 heuristic positives;actual5/1/2 problem split |
| Available results | Baseline2/11 slot decisions versus auxiliary1/11;0/2 legacy exact for both. |
| Evidence files / commits | `supervised:docs/supervised_grounding_feasibility_2026-10-27/EXISTING_METHODS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 original training stdout audited; historical checkpoints absent. |
| Scientific limits | Index-derived weak labels; two effective test problems; cannot estimate broad transfer. |
| Current location | S3.5 brief |
| Revised destination | Preserve exact small-sample results in S8; caveat S3.5. |

## E25 — Targeted synthetic auxiliary training

| Field | Evidence-qualified record |
|---|---|
| Status | Completed, underpowered |
| Purpose / conditions | Job854618;20 auxiliary then6 real steps |
| Dataset / exact population | 155 template pairs;actual1 train/0 dev/1 test problem |
| Available results | Slot decisions0/8→1/8;full0/1 both;type7/8→0/8. |
| Evidence files / commits | `supervised:docs/supervised_grounding_feasibility_2026-10-27/EXISTING_METHODS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 retained stdout/audit. |
| Scientific limits | One test problem, synthetic labels and conflicting indicators; not a general positive or negative transfer result. |
| Current location | S3.5 brief |
| Revised destination | Preserve in S8 and supervision limitations. |

## E26 — Historical fine-tuned schema retrieval

| Field | Evidence-qualified record |
|---|---|
| Status | Completed fitting; separate task |
| Purpose / conditions | MiniLM sentence-transformer MultipleNegativesRankingLoss; synthetic/collected query-schema pairs |
| Dataset / exact population | Jobs843334/850246; latter80297 training pairs |
| Available results | Retained training-runtime/loss records establish fitting; no new trustworthy scalar-grounding endpoint inferred. |
| Evidence files / commits | `supervised:docs/supervised_grounding_feasibility_2026-10-27/EXISTING_METHODS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 retained training reports; source introductions d0cd0e9/94e11e4 are not exact run SHAs. |
| Scientific limits | Task is schema retrieval, not mention binding; benchmark-derived weights are not independent initialization. |
| Current location | S3.5 one sentence |
| Revised destination | S8 historical context; no scalar performance table. |

## E27 — OptMATH external numeric extraction and audit

| Field | Evidence-qualified record |
|---|---|
| Status | Completed; corrected CI only summary-supported |
| Purpose / conditions | Extraction/code-AST/text-support sets; two automated classifiers; no schema retrieval |
| Dataset / exact population | Fixed1000 OptMATH-Train instances;150-case automated cross-validation/4801 literals |
| Available results | Raw pooled .7501; auto text-support .8118, complete .5910. Corrected pooled23241/28799=.8070; macro .8135; archived instance-bootstrap CI[.7842,.8283]. Agreement98.04%;7.28% implementation constants. |
| Evidence files / commits | `current:results/external_validation/optmath/aggregate_metrics.json`; `current:results/external_validation/optmath/final_verification/verified_metrics.json`; `current:results/external_validation/optmath/final_verification/classifier_validation.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 original sets/counts; R3 corrected per-instance counts/CI unavailable. |
| Scientific limits | JSON micro/macro names reversed; occurrences not recoverable from sets; classifiers not human gold; no full-framework transfer. |
| Current location | §7.1 |
| Revised destination | Retain unchanged now; future main short transfer result, detailed supplement audit. |

## E28 — G1 autonomous inventory feasibility

| Field | Evidence-qualified record |
|---|---|
| Status | Completed feasibility audit; experiment blocked |
| Purpose / conditions | Trace whether slots/eligibility can be inferred without reference-derived structure |
| Dataset / exact population | Current validated pipeline/components; no evaluation sample launched |
| Available results | No validated autonomous slot-discovery path; no measured autonomous success rate. |
| Evidence files / commits | `phase3:docs/g1_gold_free_autonomous_scalar_grounding_2026-10-28/REPORT.txt`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 source/record findings; no predictions. |
| Scientific limits | Infeasibility in inspected components is not theoretical impossibility or zero accuracy. |
| Current location | §7.4; S5 |
| Revised destination | Retain concise boundary in §7; S5 full context. |

## E29 — G2 fixed-slot gold-filter intervention

| Field | Evidence-qualified record |
|---|---|
| Status | Completed identifiability audit; experiment blocked |
| Purpose / conditions | Separate query-gold slot participation from mention-candidate construction |
| Dataset / exact population | Native typed path with frozen required slots |
| Available results | No native gold-value mention filter exists. At fixed gold-required slots the proposed filter removal is vacuous; broader inventory changes task. |
| Evidence files / commits | `g2:docs/g2_gold_eligibility_ablation_2026-10-28/CONDITION_COMPARISON.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 source audit; no A/B measurements. |
| Scientific limits | No causal estimate or equivalence test; not a performance result. |
| Current location | §7.4; S5 |
| Revised destination | Retain as methodological limitation, not ablation score. |

## E30 — G3 schema-only participation sets

| Field | Evidence-qualified record |
|---|---|
| Status | Completed static audit; performance experiment not executed |
| Purpose / conditions | Compare strict scalar metadata against its gold-key intersection without decoding |
| Dataset / exact population | 331 saved TF–IDF IDs/1847 reference scalar occurrences; all49 primary cases equal-set |
| Available results | 301 equal/30 different;134 additions;172/1847 missing gold occurrences.29 changed cases miss required keys both ways;1 lacks numeric reference.7 cases differ from original fallback definition. |
| Evidence files / commits | `phase3:docs/g3_schema_only_slot_participation_feasibility_2026-10-28/inventory_counts.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 retained inventories support set accounting; no model/decoder outcomes. |
| Scientific limits | Structural non-identifiability, not measured zero effect. Added132 unknown keys+2 symbolic-value keys not automatically wrong. |
| Current location | §7.4; S5 |
| Revised destination | Retain S5 and short limitations summary. |

## E31 — Historical global compatibility family

| Field | Evidence-qualified record |
|---|---|
| Status | Completed legacy campaign |
| Purpose / conditions | Frozen historical scoring/beam families, predating later baseline changes |
| Dataset / exact population | 331 queries ×orig/noisy/short for each local/pairwise/full method |
| Available results | Orig coarse138/140/143 of331; same retained legacy typed175/331. All72 historical CSV aggregates separately preserved in SAVED_METRIC_CHECKS.json. |
| Evidence files / commits | `current:results/eswa_revision/02_downstream_postfix/nlp4lp_downstream_per_query_orig_tfidf_global_compat_full.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved CSVs; current offline aggregation. |
| Scientific limits | Legacy source vintage, not Stage6 controls; old report typed174 differs from retained corrected CSV175. No reused bootstrap significance or scoring-cause inference. |
| Current location | Omitted as family; only general prior-method discussion |
| Revised destination | S8 full legacy variant table; no main headline substitution. |

## E32 — Historical relation-aware linking ablations

| Field | Evidence-qualified record |
|---|---|
| Status | Completed legacy campaign |
| Purpose / conditions | Basic/operators/semantic/full hand-designed relation scoring |
| Dataset / exact population | 331 queries ×3 variants ×4 relation-feature modes |
| Available results | Orig coarse165/155/144/138 of331; saved variant metrics retained. |
| Evidence files / commits | `current:results/eswa_revision/02_downstream_postfix/nlp4lp_downstream_per_query_orig_tfidf_relation_aware_basic.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved metrics, not accepted semantic labels. |
| Scientific limits | Feature-package differences; old statistics and causal explanations not endorsed; not trained relation extraction. |
| Current location | Not separately tabulated |
| Revised destination | S8 legacy table with all three variants. |

## E33 — Historical ambiguity-aware grounding

| Field | Evidence-qualified record |
|---|---|
| Status | Completed legacy campaign |
| Purpose / conditions | Candidate competition, finite beam, confidence-triggered abstention |
| Dataset / exact population | 331 queries ×3 variants ×candidate-greedy/beam/abstain/full |
| Available results | Orig coarse140/140/9/139 of331; abstain coverage .2207. |
| Evidence files / commits | `current:results/eswa_revision/02_downstream_postfix/nlp4lp_downstream_per_query_orig_tfidf_ambiguity_aware_abstain.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved metrics. |
| Scientific limits | Low readiness is not calibrated uncertainty; neither universal negative nor numerical-correctness ranking. |
| Current location | Not separately tabulated |
| Revised destination | S8 legacy table and abstention interpretation. |

## E34 — Baseline staleness, hierarchical search and classical alternatives

| Field | Evidence-qualified record |
|---|---|
| Status | Completed comparison audit |
| Purpose / conditions | August12 synchronized records; includes BM25/LSA, acceptance/hierarchical rerank, hierarchical grounding |
| Dataset / exact population | 331 orig queries per12 retained configurations; earlier3-variant legacy cohort separate |
| Available results | Coarse typed257,Oracle273,BM25253,LSA243,constrained248,MWM246,role244,search233,hierarchical233,semantic237,acceptance253,hierarchical-rerank238. MWM versus typed:7 rescues/18 regressions. |
| Evidence files / commits | `current:results/baseline_staleness_audit_2026-08-12/regen_results_all_variants.json`; `current:results/max_weight_matching_validation/mechanism_and_error_analysis_summary.json`; `current:docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 archived CSVs and transitions. |
| Scientific limits | Stale .5287 comparator cannot establish a .7432 gain; no wholesale claim richer methods never improve numerical correctness. Historical ratio/Oracle versions remain separate. |
| Current location | Implicit, not fully represented |
| Revised destination | S8 twelve-method table and correction history. |

## E35 — Top-k schema and reranking feasibility

| Field | Evidence-qualified record |
|---|---|
| Status | Completed retrospective diagnostic |
| Purpose / conditions | Gold-schema/readiness oracle ceilings and multiple heuristic rerank rules |
| Dataset / exact population | 331 queries;30 top1 misses; saved top10 candidates |
| Available results | Miss ranks2/3/4–5/6–10/>10:9/5/4/6/6. Gold in top10:325. Gold-schema top10 ready270; readiness oracle286 includes32 wrong-schema-ready cases. Best observed readiness rule282 with20 schema regressions. |
| Evidence files / commits | `current:results/topk_schema_rerank_stage_a/summary.json`; `current:results/topk_schema_rerank_stage_a/reranker_results.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 retained candidate groundings; R2 saved summaries/transitions. |
| Scientific limits | Oracle/projection not deployable score; selecting best among rules is exposed exploratory analysis. TOP2_GO was a historical decision, not authorization. |
| Current location | Only selective follow-up mentioned |
| Revised destination | S8 ceilings and false-positive boundary. |

## E36 — Selective schema-grounding reranking

| Field | Evidence-qualified record |
|---|---|
| Status | Completed diagnostic |
| Purpose / conditions | Selective top2 reranking changes retrieval and grounding |
| Dataset / exact population | 331 queries;baseline257 ordinary-ready |
| Available results | Ordinary-ready265;8 newly ready of which6 wrong schema; motivates Strict gate. |
| Evidence files / commits | `current:docs/SELECTIVE_GROUNDING_RERANK_STAGE_B_2026-08-13.md`; `current:results/selective_grounding_rerank/nlp4lp_downstream_per_query_orig_tfidf.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved transitions and metric records. |
| Scientific limits | Not eight additional semantic successes, not fixed-retrieval grounding comparison. |
| Current location | §5.7/Table 7 |
| Revised destination | Main §5.6; supplemental context S7/S8. |

## E37 — Role/quantity Stage-A separability and cascade

| Field | Evidence-qualified record |
|---|---|
| Status | Completed negative feasibility diagnosis |
| Purpose / conditions | Saved roles/quantity conflict flags, separability and prospective triggers |
| Dataset / exact population | 331-query frame;54 schema-hit/non-ready queries;49 targeted wrong assignments |
| Available results | 28 separable/10 ambiguous/11 not-separable assignments;0 projected ready rescues;baseline/projected257. Recorded all-query cascade trigger331/331, false-trigger1.0. |
| Evidence files / commits | `current:results/role_quantity_stage_a/summary.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R1 cached per-slot records; summary projections are not executed new-policy outcomes. |
| Scientific limits | Readiness-neutral value corrections may still matter; no semantic generalization; no API escalation executed. |
| Current location | Not explicitly tabulated |
| Revised destination | S8 diagnostic negative result. |

## E38 — Strict-failure diagnosis and quick-fix precursor

| Field | Evidence-qualified record |
|---|---|
| Status | Completed diagnosis/prototype; other interventions only oracle projections |
| Purpose / conditions | Missing extraction/type/participation/reuse analysis; ratio-word prototype |
| Dataset / exact population | 331-query frame;54 schema-hit/non-ready and58 Oracle non-ready cases |
| Available results | Ratio prototype247→255 Strict (8 gains/0 losses); oracle extraction projected254,types271,reuse272,filter247. Overlapping cause counts remain in source. |
| Evidence files / commits | `current:results/strict_failure_quick_fix/summary.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved diagnosis/prototype records. |
| Scientific limits | Do not present oracle projections as deployed fixes. Only completed ratio patch supplies main ablation. |
| Current location | Ratio patch main; precursor not explicit |
| Revised destination | S8 distinguish prototype and unexecuted alternatives. |

## E39 — Archived OpenAI two-stage schema/slot baseline

| Field | Evidence-qualified record |
|---|---|
| Status | Completed saved-record campaign |
| Purpose / conditions | LLM schema selection then slot filling; separate from common18 GPT formulation |
| Dataset / exact population | 331 queries in each orig/noisy/short |
| Available results | Saved schema hits307/309/243;coarse280/116/4;conditional Exact20 .829103/.146067/.873016 over294/171/63 comparable hits. |
| Evidence files / commits | `current:results/paper/nlp4lp_downstream_orig_openai.json`; `current:results/paper/nlp4lp_downstream_per_query_orig_openai.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 all three metric files reaggregated; no original assignment replay asserted. |
| Scientific limits | Provider tag does not authenticate model snapshot or current serving settings; supplied catalog/slots and old protocol prevent autonomy/provider ranking. |
| Current location | Omitted |
| Revised destination | S8 archival table; no main leaderboard or conflation with §6. |

## E40 — Archived Mistral two-stage orig baseline

| Field | Evidence-qualified record |
|---|---|
| Status | Completed orig metric artifact; other variants unverified |
| Purpose / conditions | Separate historical schema/slot API path |
| Dataset / exact population | 331 orig queries;job903088 |
| Available results | 171 schema hits;154 coarse-ready;Exact20 .910143 over157 comparable hits;coverage .477090/type .483384. |
| Evidence files / commits | `current:results/rerun/mistral/run_903088/paper/nlp4lp_downstream_orig_mistral.json`; `current:results/rerun/mistral/run_903088/paper/nlp4lp_downstream_per_query_orig_mistral.csv`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 saved rows; incomplete request/model-version lineage. |
| Scientific limits | High conditional Exact20 is not unconditional success; no verified noisy/short completion or provider ranking. |
| Current location | Omitted |
| Revised destination | S8 archival table with provenance limitations. |

## E41 — Optional Gemini scalar-benchmark rerun

| Field | Evidence-qualified record |
|---|---|
| Status | Partially prepared/blocked; no full completed benchmark established |
| Purpose / conditions | Historical Gemini schema/slot workflow, not October217 relationship study |
| Dataset / exact population | Planned331 queries ×3 variants;retained job902357 preflight only |
| Available results | No authenticated full per-query benchmark artifact located in retained rerun directory; no performance value assigned. |
| Evidence files / commits | `current:docs/GEMINI_RERUN_REPORT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R0 performance; infrastructure/preflight only. |
| Scientific limits | Cannot infer failed scientific predictions from availability/setup failure; do not conflate separate successful relationship campaign. |
| Current location | Absent |
| Revised destination | S8 status-only; no accuracy cell. |

## E42 — CloudRift Qwen relationship extension

| Field | Evidence-qualified record |
|---|---|
| Status | Blocked before inference |
| Purpose / conditions | Historical requested model absent at availability gate; no substitute selected |
| Dataset / exact population | Planned217 relationships;0 attempted |
| Available results | No new judgments, no pilot, no four-model comparison. |
| Evidence files / commits | `labels:docs/nl4opt_cloudrift_qwen_2026-10-27/HANDOFF.txt`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R0 performance; historical availability receipt only. |
| Scientific limits | Account/budget identifiers not republished; unattempted is neither invalid judgment nor zero accuracy. |
| Current location | Absent |
| Revised destination | S8 status-only; no scientific performance finding. |

## E43 — Independent relationship reference-label protocol

| Field | Evidence-qualified record |
|---|---|
| Status | Proposed/prepared, not completed |
| Purpose / conditions | Finite blinded review design for selected217 frame |
| Dataset / exact population | 36 probability+12 targeted relationships;2 overlaps=46 distinct across16 problems |
| Available results | No accepted formal independent reference labels; semantic accuracy/precision remains unavailable. |
| Evidence files / commits | `labels:docs/nl4opt_scientific_evaluation_phase1_2026-10-27/METRIC_READINESS.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R0 performance; prepared sampling records only. |
| Scientific limits | Different from author-approved D1 generated contrasts; cannot transfer24 labels to217 historical judgments. |
| Current location | §7.3 |
| Revised destination | Keep future prerequisite; S8 status ledger, no new reviewer task. |

## E44 — Earlier synthetic global-consistency evaluation

| Field | Evidence-qualified record |
|---|---|
| Status | Completed synthetic exercise; real benchmark blocked |
| Purpose / conditions | Type-assigned synthetic references, distinct global_consistency mode |
| Dataset / exact population | 331 catalog-derived synthetic scalar assignments ×3 variants |
| Available results | Report orig readiness typed16/331 versus global18/331;real Exact20 unavailable. |
| Evidence files / commits | `current:docs/archive/GCG_FINAL_EVAL_REPORT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R3 historical report; original temporary harness/per-query files not authenticated. |
| Scientific limits | Artificial values cannot support real-data performance or report claim that relative differences transfer. Distinct from global_compat real campaign. |
| Current location | Omitted |
| Revised destination | S8 historical status and explicit exclusion from scientific benchmark comparison. |

## E45 — Neural/scorer scaffolds and proposed external supervision

| Field | Evidence-qualified record |
|---|---|
| Status | Implemented/proposed; completed non-smoke runs not verified |
| Purpose / conditions | Code/protocol only; future clean external-supervision design |
| Dataset / exact population | MiniLM cross-encoder, NL4Opt multitask auxiliary, export stubs;no approved clean training set |
| Available results | No authenticated completed MiniLM non-smoke or NL4Opt auxiliary checkpoint;0 approved semantic training links. Other auxiliary runs are E24/E25. |
| Evidence files / commits | `supervised:docs/supervised_grounding_feasibility_2026-10-27/EXISTING_METHODS.md`; `supervised:docs/supervised_grounding_feasibility_2026-10-27/GO_NO_GO.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R0 for unexecuted performance; source evidence only. |
| Scientific limits | File existence, tests or job submission are not model outcomes; no broad negative learning claim. |
| Current location | S3.5/§7.3 |
| Revised destination | Keep S3.5 and S8 status-only. |

## E46 — Historical prediction recovery and source integrity

| Field | Evidence-qualified record |
|---|---|
| Status | Completed archival investigation; recovery inconclusive/negative |
| Purpose / conditions | Inspect original writers, hashes, run identities; distinguish Stage-A cache from missing headline maps |
| Dataset / exact population | Four headline metric CSVs;sanitization/60/269/20 target artifacts |
| Available results | Four CSVs match first committed versions;13 metric columns;no authenticated original headline scalar export or full subset outcome manifests recovered. |
| Evidence files / commits | `phase2:docs/historical_scalar_assignment_recovery_2026-10-28/VERIFIED_ARTIFACTS.md`; `phase2:docs/historical_scalar_assignment_recovery_2026-10-28/THREE_TABLE_REASSESSMENT.md`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 integrity/aggregation; R0 absent predictions; no assumptions substituted. |
| Scientific limits | Absence does not prove deletion; newer cache/loose blobs are not originals. |
| Current location | S3.3 and §7.2 |
| Revised destination | Keep reproduction boundary and unresolved-issues register. |

## E47 — Public metrics and release provenance check

| Field | Evidence-qualified record |
|---|---|
| Status | Completed provenance audit; no performance experiment |
| Purpose / conditions | Offline public-mirror verification in M5; dataset unchanged this phase |
| Dataset / exact population | Pinned public revision24a047fa;331 canonical rows/4303 diagnostics/13 methods;1000 external rows |
| Available results | Published metrics reproduce301 schema/265 coarse/255 Strict/7 full;release is metrics-only, not independent supervision. |
| Evidence files / commits | `m5:docs/manuscript_m5_2026-10-28/verified_author_review_20261029T002829Z/verification/public_metrics_audit.json`. Exact hashes/commits: source manifest. |
| Independent reconstruction | R2 retained public-mirror audit; no fresh Hugging Face publication or modification. |
| Scientific limits | License/access differences; raw gated NLP4LP and unresolved-license OptMATH corpus not copied. M5 candidate not claimed fully public. |
| Current location | Declarations/S3.3 |
| Revised destination | Keep data-access boundaries; source inventory records local-only evidence. |
