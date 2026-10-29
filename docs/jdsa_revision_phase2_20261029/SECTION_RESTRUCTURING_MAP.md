# Section restructuring map — JDSA Phase 2

Source: Phase 1 branch `manuscript/jdsa-phase1-evidence-results-20261029`, commit `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650`. Phase 1 files and research artifacts remain unchanged. The new package is `manuscript/jdsa-phase2/`.

| Phase 1 location | Phase 2 destination | Treatment |
|---|---|---|
| Abstract and Introduction | Abstract and §1 | Abstract verbatim; Introduction changes only its final organizational roadmap. |
| §2 Related Work | §2 Related Work | Scientific text preserved; cited binding preprint explicitly identified in bibliography. |
| §3 Task Definition and Evaluation Assumptions | §3 Problem Setting and Evaluation Framework | Heading revised; definitions and assistance conditions preserved. |
| §4 Methods and Experimental Protocol | §4 Methods and Experimental Setup | Heading revised; setup, algorithm and figure preserved. |
| §5 Diagnostic Results | §5.1–§5.7 Experimental Results | Entire core body and Tables 1–7 preserved byte-for-byte; section title changes. |
| §6.1 common-18 comparison | §5.8.1 and S4 | Table 8 retained in main; denominator/precision interpretation foregrounded; configuration/correction details in S4. |
| §6.2 objective extraction and failures | Main §5.8.1 summary; S4.2–S4.3 | Round-first predicate moved to numbered Eq. S2; 31 terminal pairs, correction history, failures, precision witness preserved. |
| §6.3 relationship judgments | §5.8.3 and S6 | Table 9 and principal validity/agreement results remain main; model settings, pairwise counts, category marginals, recovery history and witnesses in S6. |
| §7.1 OptMATH extraction | §5.8.2 and S9.1 | Component findings retained; historical interval and automated classifier details in S9.1. Corrected 7.28% attribution explicitly documented. |
| §7.2 limitations | §6.3, with relevant endpoint qualifications in §5.8 | Consolidated into assistance/reference, historical evidence, dependence/transfer and semantic-assessment limitations. |
| §7.3 future directions | §6.4 and S6.3 | Future independent review distinguished from completed D1 construction; 36+12−2=46 protocol remains uncompleted. |
| §7.4 G1–G3 | §6.4 concise implication; S5 full accounting | Static audits remain methodological findings, not performance experiments. |
| §8 Conclusion | §7 Conclusion | Two coherent paragraphs answering retrieval, grounding and semantic-evidence questions. |
| D1 in S8.6 | Main §5.8.4; S9.2 | Full annotation caveats retained, all 12 pair summaries added in Tables S25–S26. S8.6 links to them. |
| S1–S8 remaining studies and tables | S1–S8 | All 24 existing supplementary tables unchanged. Existing scientific material retained; added availability subsection shifts some S3 subsection numbers. |
| Declarations | Declarations and S3.5 | Historical v1.2 distinguished from unreleased M5; D1 author review distinguished from an independent human study. |

The seven-section paper follows the progression retrieval → readiness → full scalar agreement → matched/controlled diagnostics → complementary evidence → implications/limitations/future work. Supplementary references are label-based and resolved by the two-document build. Every original table label, caption and body is preserved; new D1 tables are additive.

## All 47 registered investigation groups

These are investigation groups, not 47 completed performance experiments. Status and reconstruction grades remain those in the [Phase 1 register](../jdsa_revision_phase1_20261029/EXPERIMENT_EVIDENCE_REGISTER.md). The table assigns every group an explicit destination. Blocked/planned work is retained as a status/limit, never promoted to a scientific outcome.

| ID | Investigation | Retained status | Phase 2 publication destination |
|---|---|---|---|
| E01 | Lexical schema retrieval and ranked-list diagnostics | Completed | Main §5.1, Table 1; S1 and S8.2 ranked-list details |
| E02 | Pretrained dense retrieval | Completed | Main §5.1, Table 1; S3.3 provenance limits |
| E03 | Four-arm scalar-grounding benchmark | Completed | Main §§3.2–3.4, 5.2, Table 2; S3.4 assistance boundaries |
| E04 | Ratio-word extraction ablation | Completed | Main §5.6, Table 5; S7.4 statistical history |
| E05 | Matched grounding alternatives | Completed | Main §5.3, Table 3; S3.1 and S3.3 |
| E06 | Residual-error and full-scalar analyses | Completed offline investigation | Main §5.4, Table 4; S7.3 error interpretation |
| E07 | Frozen-score factorial assignment | Completed | Main §5.5; S3.1, Table S7 |
| E08 | Cached joint assignment pilot | Completed | Main §5.5; S3.1 and S8.3, Table S23 |
| E09 | Exact objective and beam-pruning diagnosis | Completed | Main §5.5; S3.1 objective/beam diagnosis |
| E10 | Value/type-distinct exact enumeration | Completed negative follow-up | Main §5.5; S3.1 exact enumeration |
| E11 | Cached candidate availability and Stage-A error accounting | Completed diagnosis | S3.2 and S8.3, Table S22; implications in main §6.1 |
| E12 | Reference conventions and frozen-prediction sensitivity | Completed correction-only audit | Main §6.3; S3.2, Table S8 |
| E13 | Corrected paired significance and factorial multiplicity | Completed statistical correction | Main §5.6, Table 6; S3.1 and S7.4 |
| E14 | Sanitization and overlap robustness | Completed, aggregate-supported in part | Main §5.6; S2 Table S4 and S7.4 |
| E15 | Historical 60-case coverage/type proxy | Completed, aggregate-supported | Main §5.7; S2 Table S5 and S7.1 (aggregate-supported) |
| E16 | Historical extended269 proxy | Completed, aggregate-supported | Main §5.7; S7.1 (aggregate-supported, overlap retained) |
| E17 | Historical20-case reference-program execution | Completed, aggregate-supported | Main §5.7; S2 Table S6 and S7.1 (reference-assisted, aggregate-supported) |
| E18 | Common18 formulation and terminal-objective audit | Completed generation; ORLM execution blocked; PaMOP agreement inconclusive | Main §5.8.1, Table 8; S4 Tables S11–S14, Eq. S2 |
| E19 | 217-item three-service relationship judgments and recovery | Completed; semantic accuracy unmeasured | Main §5.8.3, Table 9; S6.1–S6.2 Tables S16–S18, Eq. S3 |
| E20 | D1 semantic contrast construction and author audit | Completed construction/audit; provider pilot not executed | Main §5.8.4; S9.2 Tables S25–S26 (construction, not provider performance) |
| E21 | NL4Opt proposal, annotation and template audit | Completed audit; training blocked | S3.7; implications in main §6.3 |
| E22 | Historical DistilRoBERTa mention-slot learning | Completed training; comparative interpretation inconclusive | S3.6 and S8.4 (weak labels and evaluation-origin dependence) |
| E23 | P0 learned feature scorer and assignment | Completed | S3.6 and S8.4 (positive and negative outcomes retained) |
| E24 | GAMS auxiliary weak-label training | Completed, underpowered | S8.4 (tiny auxiliary test population) |
| E25 | Targeted synthetic auxiliary training | Completed, underpowered | S8.4 (tiny auxiliary test population) |
| E26 | Historical fine-tuned schema retrieval | Completed fitting; separate task | S8.4 (schema retrieval, not binding) |
| E27 | OptMATH external numeric extraction and audit | Completed; corrected CI only summary-supported | Main §5.8.2; S9.1 (extraction only; corrected constant-denominator attribution) |
| E28 | G1 autonomous inventory feasibility | Completed feasibility audit; experiment blocked | Main §6.4; S5 (feasibility audit, no performance claim) |
| E29 | G2 fixed-slot gold-filter intervention | Completed identifiability audit; experiment blocked | Main §6.4; S5 (intervention/identifiability audit) |
| E30 | G3 schema-only participation sets | Completed static audit; performance experiment not executed | Main §6.4; S5 Table S15 (static participation-set accounting) |
| E31 | Historical global compatibility family | Completed legacy campaign | S8.1, Table S19 (historical campaign, separate baseline vintage) |
| E32 | Historical relation-aware linking ablations | Completed legacy campaign | S8.1, Table S19 (positive/negative ablations) |
| E33 | Historical ambiguity-aware grounding | Completed legacy campaign | S8.1, Table S19 (readiness/abstention trade-offs) |
| E34 | Baseline staleness, hierarchical search and classical alternatives | Completed comparison audit | S8.1, Tables S19–S20 (baseline-vintage correction retained) |
| E35 | Top-k schema and reranking feasibility | Completed retrospective diagnostic | S8.2 (top-k feasibility, distinct from main top-1) |
| E36 | Selective schema-grounding reranking | Completed diagnostic | S8.2 (selection study, not unseen evaluation) |
| E37 | Role/quantity Stage-A separability and cascade | Completed negative feasibility diagnosis | S8.2 (diagnostic/cascade results) |
| E38 | Strict-failure diagnosis and quick-fix precursor | Completed diagnosis/prototype; other interventions only oracle projections | S8.3 (earlier error accounting, not a substitute prediction archive) |
| E39 | Archived OpenAI two-stage schema/slot baseline | Completed saved-record campaign | S8.5, Table S24 (archived scalar LLM metrics, distinct endpoint) |
| E40 | Archived Mistral two-stage orig baseline | Completed orig metric artifact; other variants unverified | S8.5, Table S24 (archived scalar LLM metrics, distinct endpoint) |
| E41 | Optional Gemini scalar-benchmark rerun | Partially prepared/blocked; no full completed benchmark established | S8.5 (preflight/infrastructure only; no authenticated completed full benchmark) |
| E42 | CloudRift Qwen relationship extension | Blocked before inference | S8.5 (blocked before inference; unattempted, not zero accuracy) |
| E43 | Independent relationship reference-label protocol | Proposed/prepared, not completed | Main §6.4; S6.3 (prepared independent review, no accepted labels) |
| E44 | Earlier synthetic global-consistency evaluation | Completed synthetic exercise; real benchmark blocked | S8.6 (synthetic archival context only; not real-data scientific comparison) |
| E45 | Neural/scorer scaffolds and proposed external supervision | Implemented/proposed; completed non-smoke runs not verified | S3.6 and S8.4 (scaffolds/proposals; no fabricated completed neural results) |
| E46 | Historical prediction recovery and source integrity | Completed archival investigation; recovery inconclusive/negative | Main §6.3; S3.3 (archive integrity and missing mappings) |
| E47 | Public metrics and release provenance check | Completed provenance audit; no performance experiment | Declarations; S3.5; reference/availability report (provenance, not performance) |
