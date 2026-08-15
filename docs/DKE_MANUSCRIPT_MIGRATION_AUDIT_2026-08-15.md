# DKE Manuscript Migration Audit (2026-08-15)

Evidence-based preparation and gap audit for migrating the manuscript "Retrieval-Assisted
Instantiation of Natural-Language Optimization Problems" to *Data & Knowledge Engineering*
(DKE, Elsevier). This is a **preparation audit only**: no manuscript prose, title, abstract,
or structure was rewritten, and no experiments were run. The method is
`FROZEN_FOR_RESUBMISSION`.

---

## 1. Scope and Input Evidence

Sources consulted (all read/fetched this session unless otherwise noted):

- `manuscript/main.tex` (727 lines) — KAIS-targeted manuscript, Springer `sn-jnl.cls`.
- `manuscript/submission_package/main.tex` + `references.bib` — identical source to top-level.
- `Retrieval_Assisted_Instantiation_of_Natural_Language_Optimization_Problems.zip` — original
  EAAI/ESWA Elsevier `elsarticle.cls` draft (historical only).
- `manuscript/references.bib` (29 entries).
- `results/final_resubmission_method/metrics.json`, `strict_metrics.json`, `README.md`,
  `metrics.csv`, `per_query.csv`, `transitions.json`, `summary.json`.
- `results/external_baseline_comparison/comparison.md`.
- `docs/SCIENTIFIC_STATE.md`, `docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`,
  `docs/RESUBMISSION_BASELINE_READINESS_2026-08-15.md`.
- Official DKE guide-for-authors (fetched 2026-08-15, ScienceDirect) and DKE aims/scope.
- Recent DKE papers (2024-2026): Rizzi et al. 2025 (102452), Atzeni et al. 2026 (102494),
  plus DKE vol. 156/159/161/162 tables of contents and the LLM&KG special-issue call.
- Web verification of baseline publications: PaMOP (IJCAI 2025), ORLM (arXiv 2405.17743,
  v5; Operations Research 2025), OptMATH (arXiv 2502.11102; ICML 2025), OR-R1 (arXiv
  2511.09092, AAAI 2026), DeepOR (AAAI 2026), OptiBench (arXiv 2407.09887, ICLR 2025),
  ORQA (AAAI 2025).

## 2. Repository State (at audit time)

- Branch `main`, HEAD `15e312b`, in sync with `origin/main` (0 ahead / 0 behind), clean
  except one untracked leftover `results/pamop/fidelity_diagnostic_gpt5/scaled18_extension.log`
  (0-byte; not referenced; do not commit).
- Prior session commits `e76b2c7` -> `b6a7630` (KAIS rename) -> `15e312b` (baseline
  comparison regeneration) are present and pushed.

## 3. Method Freeze and Authoritative Numbers

Method is frozen (`docs/METHOD_FREEZE_FOR_RESUBMISSION_2026-08-13.md`): TF-IDF top-1 schema
retrieval over the 335-document catalog + typed-greedy scalar grounding + deterministic
multiplicative (ratio-word) extraction patch (`_extract_multiplicative_ratio_tokens` in
`tools/nlp4lp_downstream_utility.py`). No changes to `_choose_token`, retrieval, or
thresholds.

Authoritative numbers (`results/final_resubmission_method/metrics.json`, n=331, orig queries):

| Metric | prepatch | patched (final) |
|---|---|---|
| Schema R@1 | 301/331 = 0.909366 | 301/331 = 0.909366 |
| Coverage | 0.879430 | 0.888566 |
| TypeMatch | 0.851545 | 0.866549 |
| InstantiationReady | 257/331 = 0.776435 | 265/331 = 0.800604 |
| StrictInstantiationReady | 247/331 = 0.746224 | 255/331 = 0.770393 |
| Exact5 | 0.219463 | 0.235536 |
| Exact20 | 0.244888 | 0.261391 |
| KeyOverlap | 0.921781 | 0.921781 |

Strict gain from patch = 8/331 = 0.024169 (strict_metrics.json).

### CRITICAL FINDING: manuscript downstream numbers are stale

The current `manuscript/main.tex` reports the *older* evaluation configuration: abstract and
text give InstantiationReady **0.5287** for TFIDF-TG and **0.5680** for Oracle-TG
(lines 30, 88, 348, 352, 372-375, 401-404, 468-471, 501-507, 522, 570-587, 672). These do
NOT match the frozen-method artifacts above (patched 0.8006 / strict 0.7704). The frozen
artifact values are strictly higher. Every downstream table, the abstract, the
contributions, the significance section, and the conclusions must be migrated to the frozen
artifact values during rewrite. **Oracle numbers in the manuscript (0.5680) have no
corresponding artifact in `results/final_resubmission_method/`** (no oracle metrics.json is
present there); the oracle control must be recomputed/verified against the frozen method or
its provenance documented before DKE submission.

Schema R@1 0.9094 (= 301/331) is consistent with frozen artifacts. Retrieval-variant
(noisy/short) numbers in Table `nlp4lp-retrieval-main` (0.9033/0.7795 etc.) are not present
in `results/final_resubmission_method/`; their provenance is in earlier result directories
and should be re-verified during rewrite.

## 4. Manuscript Inventory

- `manuscript/main.tex` — KAIS target, Springer `sn-jnl.cls`, Dec-2024 template. Authoritative
  editable source, most complete and scientifically current version. **Recommended DKE base.**
- `manuscript/submission_package/` — same `main.tex` + `references.bib`; `main.pdf` is newer
  build. Not an independent source.
- Repo-root `.zip` — original Elsevier `elsarticle.cls` EAAI/ESWA draft (2026-07-23). Historical.
- `manuscript/elsarticle.cls`, `manuscript/elsarticle-num.bst`, `manuscript/elsarticle-harv.bst`,
  `manuscript/sn-jnl.cls`, `manuscript/sn-basic.bst` — both template families available locally.
  For DKE, migrate to `elsarticle.cls` + `elsarticle-num.bst` (DKE's numbered-reference style).
  Note DKE's guide encourages the newer `els-cas-templates.zip`; `elsarticle.cls` remains an
  acceptable Elsevier class, but the rewrite should consider CAS. See Section 10.

## 5. DKE Aims and Scope Alignment

From the official guide-for-authors (fetched 2026-08-15):

DKE covers: (1) Representation and Manipulation of Data & Knowledge — conceptual data models,
knowledge representation, data/knowledge manipulation languages; (2) Architectures of
database/expert/knowledge-based systems; (3) Construction of data/knowledge bases —
methodologies and tools, data/knowledge acquisition; (4) Applications, case studies,
management issues; (5) **Tools for specifying and developing Data and Knowledge Bases using
tools based on Linguistics or Human Machine Interface principles**; (6) Communication aspects
of KBSs in cyberspace.

**DKE fit: GOOD.** The paper frames NL optimization-problem instantiation as a knowledge
engineering task: grounding natural-language numeric evidence into a structured, **typed
optimization schema** (a knowledge representation artifact), using **schema retrieval**
(IR over a knowledge base of 335 schema documents) and deterministic semantic grounding.
This maps most directly to topics 1 (knowledge representation, conceptual models) and 5
(linguistics/HMI-based tools for specifying knowledge bases). The DKE scope is explicitly
broad and includes the interface between data engineering and knowledge engineering, which
fits the paper's "knowledge-processing" framing.

The recent DKE special issue on "Large Language Models and Knowledge Graphs for
Semantics-driven Systems Engineering" (deadline 30 Mar 2025) and DKE papers such as Rizzi
et al. 2025 ("Conceptual design of multidimensional cubes with LLMs: An investigation",
DKE 102452) and Atzeni et al. 2026 ("Semantic-aware query answering with Large Language
Models", DKE 102494) confirm that DKE actively publishes empirical LLM/NL-to-structured-
model work. The paper's deterministic, non-LLM angle is a distinctive fit rather than a
mismatch.

## 6. DKE Article Type and Peer Review

- Article types: research articles (the relevant type).
- Peer review: **single-anonymized** (reviewers see author identity). Author identity already
  included in the manuscript title block — no anonymization needed.
- Minimum of two reviewers; initial editorial desk assessment.
- Open access supported (hybrid). No APC for subscription route; OA APC if chosen.
- Submission portal: https://submit.elsevier.com/DATAK (also editorialmanager.com/datak).

## 7. DKE Writing and Formatting Requirements (Guide for Authors)

Key requirements extracted verbatim from the official guide (accessed 2026-08-15):

- **File format:** editable source required; `.tex` for LaTeX. PDF alone not acceptable.
  Double-column only for LaTeX submissions.
- **LaTeX:** els-cas-templates encouraged; all relevant editable source files required at
  submission/revision.
- **Title page:** concise informative title (avoid abbreviations/formulae), author names,
  full affiliations with country, corresponding author with email.
- **Abstract:** concise, factual, **NOT exceeding 250 words**; standalone; avoid references;
  define non-standard abbreviations at first mention.
- **Keywords:** 1 to 7 keywords, English, avoid multi-word phrases joined by "and"/"of".
- **Highlights:** encouraged, 3-5 bullet points, each **max 85 characters** including spaces,
  as a separate file with "highlights" in the filename.
- **Tables:** editable text (not images); cite all in text; number consecutively; captions;
  notes below body; avoid vertical rules and shading; use sparingly (no duplication).
- **Figures/artwork:** separate files; cite all; number consecutively; logical naming
  (Figure_1, ...); vector as EPS/PDF; halftones min 300 dpi; bitmapped line drawings min
  1000 dpi; combos min 500 dpi; captions required; keep text in images minimal.
- **Supplementary material:** encouraged; cite in text; submit at same time.
- **Research data:** DKE uses **Option C — REQUIRED** data deposit: deposit research data in
  a repository, cite/link the dataset in the article, or state why data cannot be shared.
- **Article structure:** numbered sections, subsections 1.1/1.2; cross-referencing by number;
  abstract not included in section numbering.
- **Acknowledgements:** separate section directly before the reference list.
- **CRediT author contributions:** required for the corresponding author.
- **Appendices:** A, B, ...; separate equation/table/figure numbering (Eq. (A.1), Table A.1).
- **Vitae:** DKE requires a short biography (max 100 words) per author plus a passport-type
  photo as a separate figure (editable format).
- **References:** numbered in order of appearance `[1]`; number references by order of
  citation; journal names abbreviated per LTWA; DOIs encouraged; no strict format at
  submission but must be consistent; journal style applied at proof stage. Datasets and
  software should be cited with `[dataset]`/`[software]` tags.
- **Preprints:** mark clearly; use the published version once available; preprint DOIs
  acceptable where central to the work.
- **Ethics:** submission declaration (not previously published / not under consideration
  elsewhere), competing-interest declaration, funding statement, generative-AI disclosure
  statement (required if AI tools used; example template given).
- **Inclusive language**, SAGER reporting where relevant (not applicable here — no human
  subjects).

### Page/word limits

**No explicit page limit and no explicit word limit for the body** are stated anywhere in the
DKE guide for authors (checked the full "Writing and formatting" section). The only hard
numerical constraints are: abstract <= 250 words; 1-7 keywords; highlights 3-5 bullets each
<= 85 characters; author vitae <= 100 words each; figure resolution minima; video <=150 MB
per file / 1 GB total. This must be stated plainly (do not invent a limit).

## 8. Recent DKE Paper Conventions (2024-2026 sample)

Based on DKE vol. 156 (Mar 2025), 159 (Sep 2025), 161 (Jan 2026), 162 (Mar 2026) TOCs and the
open-access Rizzi et al. 2025 (102452):

- Papers are single-column Elsevier-style articles, numbered sections, concrete subsections.
- Empirical/LLM papers follow: Introduction -> related work / research process -> method
  -> experiments with research questions -> results -> discussion -> conclusions ->
  data availability -> acknowledgements -> CRediT -> references.
- DKE papers routinely include: keywords, highlights (optional file), a Data availability
  statement (e.g., Rizzi et al.: "Data will be made available on request."), CRediT, and
  author vitae with photos.
- Article numbers in DKE: 102xxx (e.g., 102452); DOIs `10.1016/j.datak.YYYY.NNNNNN`.
- Rizzi et al. 2025 is a good structural template: LLM empirical evaluation with explicit
  research questions, prompt templates, test cases, metrics; ~the same empirical-conventions
  style this manuscript already uses.
- DKE actively publishes deterministic/NL-work too (e.g., text summarization, NER, KG papers
  in vols. 159/162), so the paper's non-LLM pipeline is not out of place.

## 9. Novelty / Contribution Audit

Assessment: the paper's intermediate-task framing (schema-conditioned scalar grounding,
retrieval-dependent evaluation, oracle bottleneck diagnosis) is a defensible, distinct
contribution. It does not claim full NL-to-solver compilation. Strengths: transparent,
deterministic, reproducible, negative-result families (GCG/RAL/AAG), strict-metric
robustness, solver-backed subsets.

Risks for DKE reviewers:
- Scope is deliberately narrow (scalar slots only, fixed 335-catalog retrieval). Reviewers
  may ask "why is this not just NER + IR?" — the paper addresses this via retrieval-dependent
  evaluation and the oracle diagnosis, but the rewrite should sharpen the knowledge-engineering
  framing (typed schemas as a KR artifact; grounding as semantic slot filling) to match DKE's
  readership.
- The paper currently has NO head-to-head comparison with end-to-end LLM systems in the body
  (explicitly stated as a limitation, lines 697-699). DKE reviewers will likely see this as
  the biggest weakness. Mitigation available: the external baseline comparison
  (results/external_baseline_comparison/comparison.md) already contains 18-query PaMOP /
  ORLM / OptMATH / generic-LLM runs with fidelity labels; the rewrite should at least add
  related-work citations (PaMOP, ORLM, OptMATH, DeepOR) and may incorporate the common-18
  evidence with proper caveats. See Sections 15 and 23.
- Single-author, single-benchmark paper; DKE scope-fit framing (Section 5) helps.

## 10. Baselines / Comparison Audit

- Retrieval baselines: BM25, TF-IDF, LSA + random (1/335) + oracle control. Adequate and
  standard for the retrieval stage.
- Grounding families: typed-greedy (TFIDF/BM25/LSA), oracle typed-greedy, constrained,
  semantic-IR-repair, optimization-role-repair, acceptance-rerank, hierarchical-acceptance-
  rerank, plus GCG/RAL/AAG. Comprehensive for deterministic grounding.
- External LLM baselines: **NOT currently in the manuscript body.** The `comparison.md`
  (generated at HEAD 0197f0c, status PRELIMINARY_EXTERNAL_BASELINE_STATUS) documents
  common-18 runs: PaMOP 18/18 (gpt-5.4, AMPL/HiGHS, 8/11 obj-proxy), ORLM 18/18 official
  checkpoint but execution BLOCKED (coptpy missing; NOT_EVALUABLE, never zero), OptMATH 18/18
  official checkpoint (gurobipy 15/18 COMPLETED, 6/15 obj-proxy), generic Azure gpt-5.4
  (gurobipy 16/18, 10/16 obj-proxy), DeepOR/OR-R1 = 0 rows (UNAVAILABLE_ARTIFACT).
- Fidelity labels to preserve: OFFICIAL_IMPLEMENTATION / OFFICIAL_CHECKPOINT /
  INDEPENDENT_RECONSTRUCTION / GENERAL_PURPOSE_LLM / UNAVAILABLE_ARTIFACT. Do NOT fabricate
  cells, do NOT convert NOT_APPLICABLE to zero, do NOT build a misleading unified leaderboard
  across incompatible task definitions.
- **Baseline sufficiency verdict: MODERATE.** For the core (schema retrieval + deterministic
  grounding) task, baselines are sufficient. For end-to-end context, external evidence exists
  but is preliminary, blocked (ORLM), or unavailable (DeepOR/OR-R1). The rewrite should add
  the 4 missing citations (Section 23) and either (a) cite the common-18 comparison with
  full fidelity caveats and its status, or (b) keep the "no head-to-head" limitation but cite
  the missing systems. Decision left to rewrite (task 2).

## 11. Dataset / Workload Audit

- NLP4LP (`udell-lab/NLP4LP`, gated Hugging Face): 331 orig test queries; 335-schema
  retrieval catalog; single gold schema per query. Random top-1 chance 1/335 ~= 0.0030.
- Query variants: orig / noisy / short. Documented.
- Restricted subsets: 60-instance structural, 269-instance executable-attempt, 20-instance
  solver-backed (SciPy HiGHS shim). Documented with explicit caveats.
- **Data-access constraint:** gated dataset requires approved HF account/token. Committed
  artifacts allow reproduction of tables/figures without gated access; end-to-end rerun
  requires it. DKE Option C data policy is satisfied by: data statement + the committed
  result artifacts in the public repo; the rewrite must add a clear DKE data statement and
  ensure the benchmark citation uses the `[dataset]` tag. NLP4LP is cited as
  `nlp4lp_dataset` (misc) — should be a proper dataset reference with `[dataset]`.

## 12. Experimental Rigor Audit

- Strengths: full-denominator metrics, threshold-based InstantiationReady, paired bootstrap
  (B=1000), CI reporting, strict-metric robustness gate, sanitization/overlap stress tests,
  error taxonomy, negative-result families, solver-backed validation. Rigor is a clear
  strength.
- Concerns: (a) stale numbers (Section 3) undermine internal consistency until migrated;
  (b) significance section references "stale intermediate significance-summary snapshot"
  and documents offsets (Table nlp4lp-overlap) — good transparency, but the rewrite should
  consolidate these provenance notes into a single clean set matching frozen artifacts;
  (c) oracle values not re-derived under frozen method (Section 3).
- Statistical claims must be recomputed on frozen per-query artifacts (per_query.csv) so
  bootstrap CIs and p-values match the headline numbers.

## 13. Reproducibility Audit

- Code public (github.com/SoroushVahidi/combinatorial-opt-agent). Deterministic, CPU-based,
  no LLM in core pipeline. Hugging Face `datasets`, scikit-learn documented.
- Committed artifacts: metrics.json/csv, per_query.csv, transitions, summary — sufficient to
  regenerate tables without gated access.
- DKE requirements: editable source files, data statement (Option C), software references
  (cite code as `[software]` with PID/URL), CRediT. All are actionable during rewrite.
- Gap: no formal DOI/archival release of the artifact set is cited yet; consider a Zenodo
  deposit for the version-of-record to satisfy Option C robustly.

## 14. Technical Correctness (Manuscript vs Code) Audit

- Schema R@1 = 301/331 = 0.909366 matches both frozen rows and the manuscript's 0.9094. OK.
- Coverage/TypeMatch/InstantiationReady/Strict in the manuscript DO NOT match frozen metrics
  (Section 3). This is the primary manuscript-vs-code inconsistency and must be fixed.
- Error-taxonomy counts (Table nlp4lp-error-taxonomy: 30 / 5 / 230 / 50 / 20 / 10 / 15 / 25 /
  0) are marked "approximate diagnostic indicators" — acceptable as qualitative, but should be
  reconciled against frozen diagnostic outputs during rewrite.
- Algorithm 1 (instantiation pipeline) matches the described deterministic no-reuse greedy
  procedure. Figure 1 (pipeline) is consistent with the text.
- Retrieval-variant numbers and the overlap-table offsets (0.9063 vs 0.9094 TF-IDF) are
  disclosed but must be consolidated in the DKE version.

## 15. Writing / Presentation Audit

- Strengths: precise scope discipline, transparent caveats, measured claims, clear structure.
- Issues for DKE: (a) the paper is written for a KAIS/ML-audience voice; DKE readers are
  data/knowledge-engineering and IS researchers — the rewrite should foreground schema-as-
  knowledge-representation and grounding-as-knowledge-acquisition, and add DKE-anchored
  citations (e.g., recent DKE LLM/KG work); (b) some paragraphs are long with heavy hedging
  (e.g., Section on significance, lines 522-570); DKE style favors concise, declarative
  prose; (c) generative-AI note and Springer-specific Declarations block must be replaced
  with the Elsevier/DKE versions (AI-use disclosure template, funding, competing interests,
  CRediT, data statement).

## 16. Figures / Tables / Pseudocode Audit

- Figure 1 `nlp4lp_instantiation_pipeline_v2.png` — KEEP (re-export at required resolution
  as PDF/EPS preferred).
- Figure 2 (`figure3_engineering_validation_comparison.pdf`) — KEEP, verify it matches frozen
  60-instance numbers.
- Figure 3 (`figure4_final_solver_backed_subset.pdf`) — KEEP, verify against frozen 20-instance
  numbers.
- Algorithm 1 — KEEP (Elsevier style supports algorithms; ensure `algorithmicx`/`algpseudocode`
  compile under elsarticle; if not, convert to a numbered figure/table or use the CAS class).
- Tables: 14 total. All number-bearing downstream tables (nlp4lp-retrieval-main OK for
  R@1=0.9094; nlp4lp-downstream-main, nlp4lp-downstream-variants, nlp4lp-prefix-postfix,
  nlp4lp-newfamily-errorcheck, nlp4lp-significance, strict-instready, eng-structural,
  eng-executable, eng-solver) must be regenerated from frozen artifacts. Remove
  vertical rules/shading (booktabs is already clean). Ensure "tables used sparingly" — 14 is
  high; consider moving some (error taxonomy, variant tables) to supplementary material for
  DKE.

## 17. Limitations / Threats Audit

- Current limitations section is thorough and honest (fixed catalog, single benchmark, gated
  data, scalar-only, heuristic type/role, small solver subsets, no LLM head-to-head, no
  external datasets/deployment). KEEP as a strength.
- The "no direct experimental comparison against end-to-end LLM systems" limitation now
  partially conflicts with the existing external baseline evidence; the rewrite must reconcile
  this (either add the comparison or retain the limitation with updated citations).
- Add a DKE-relevant limitation: single benchmark; cross-dataset (Text2Zinc, CP-Bench, MAMO,
  IndustryOR, OptMATH-Bench) generalization untested.

## 18. Practical Impact Audit

- Framed around intelligent decision support and human-in-the-loop modeling. Good DKE
  resonance (HMI-based tools, topic 5).
- The deterministic pipeline's transparency (inspectable schema/slots/assignments) is a
  concrete practical selling point for DKE's engineering-audience.
- Impact evidence is benchmark-scoped; no user study or deployment. State this clearly (already
  done). DKE accepts case-study/application work, so a short "potential use case" passage
  (without new experiments) would strengthen fit during rewrite.

## 19. DKE Scope-Fit Assessment

GOOD — justified in Section 5. The paper should present itself as knowledge engineering of
optimization schemas (typed knowledge representation + linguistic grounding), which is
directly within DKE topics 1 and 5. A reader-facing title/abstract reframe emphasizing
"knowledge representation", "schema grounding", "intelligent information systems" is
recommended during rewrite (title wording itself should stay as-is per instructions — any
title change is a rewrite-task decision, not made here).

## 20. Reviewer-Style Assessment (A-L)

| # | Concern | Severity | Evidence / Note |
|---|---|---|---|
| A | Internal numeric inconsistency (stale 0.5287/0.5680 vs frozen 0.8006/0.7704) | HIGH | Manuscript lines 30, 88, 348, 352, 372-375, 401-404, 468-471, 501-507, 522, 570-587, 672 vs results/final_resubmission_method/* |
| B | Oracle numbers not reproduced under frozen method | HIGH | No oracle artifact in final_resubmission_method/ |
| C | Missing 2024-2026 baseline citations (PaMOP, ORLM, OptMATH, DeepOR) | MEDIUM-HIGH | None appear in main.tex `\cite` list; all four verified real (Section 23) |
| D | No head-to-head end-to-end comparison in body | MEDIUM-HIGH | Lines 697-699; external evidence exists but preliminary/blocked |
| E | Single benchmark (NLP4LP) generalization | MEDIUM | Limitations acknowledge; external datasets untested |
| F | Scalar-only scope | MEDIUM | Deliberate; justify for DKE KR audience more explicitly |
| G | Fixed 335-catalog retrieval overstates "schema retrieval" generality | MEDIUM | Disclosed; reframe as KB-identification task |
| H | Gated dataset (reproducibility friction) | MEDIUM | Option C data statement + artifact repo mitigates |
| I | 14 tables heavy; some could move to supplementary | LOW-MED | DKE "tables sparingly" guidance |
| J | Springer-specific Declarations/AI note must become Elsevier format | MEDIUM | Sections 7, 15 |
| K | Structural subset values must match frozen artifacts | MEDIUM | eng-structural/executable/solver tables |
| L | Statistical section provenance notes are tangled | LOW-MED | Consolidate into one clean significance block |

**Readiness score: 62 / 100** (GOOD fit, strong rigor/reproducibility backbone, but the
stale-number inconsistency, oracle gap, and missing baseline citations must be resolved
before submission; these are mechanical/reproducible fixes rather than scientific
re-openers).

## 21. Recommended DKE Manuscript Structure

1. Introduction (Background/Motivation; Related Work; Problem Scope; Contributions) — KEEP
   structure, tighten prose, foreground knowledge-engineering framing, add 4 baseline refs.
2. Methodology (Problem Formulation; Retrieval-Based Schema Identification; Deterministic
   Parameter Instantiation; Evaluation-Oriented Design Choices) — KEEP.
3. Experiments (Setup; Schema Retrieval Performance; Downstream Utility; Error Analysis and
   Ablation; Statistical Significance and Lexical-Overlap Robustness; Structural and
   Solver-Backed Validation) — KEEP; regenerate all numbers from frozen artifacts.
4. Conclusions (Limitations; Future Work) — KEEP; reconcile no-LLM-comparison limitation.
5. DKE front/back matter: Abstract (<=250 words), Keywords (1-7), Highlights (3-5 x <=85
   chars), Data availability statement, Acknowledgements, CRediT, Declarations (funding,
   competing interests, generative-AI disclosure in Elsevier template), Vitae (<=100 words +
   photo), References (elsarticle-num, `[1]`-style, DOIs, `[dataset]`/`[software]` tags).
6. Move 2-3 auxiliary tables (error taxonomy, cross-variant detail) to supplementary material.

## 22. Retainable / Rewrite Sections

Retain (modify numbers only): Problem Formulation; Retrieval-Based Schema Identification;
Deterministic Parameter Instantiation (incl. Algorithm 1); Evaluation-Oriented Design
Choices; Setup description; Limitations; Future Work; error-taxonomy narrative (recompute
counts).

Rewrite (prose-level, no new experiments): Abstract (fresh numbers, <=250 words); all
downstream tables + significance block (frozen artifacts); introduction framing for DKE
audience; related work (add PaMOP/ORLM/OptMATH/DeepOR + 1-2 DKE-anchored refs); the
no-LLM-comparison limitation reconciliation; Declarations/back matter to Elsevier format;
highlights/keywords/vitae (new); references restyle to elsarticle-num.

## 23. Missing References (verified 2026-08-15)

Four important 2024-2026 works are absent from `references.bib` and `\cite` list (all
verified as real):

1. **PaMOP** — "Guiding Large Language Models in Modeling Optimization Problems via Question
   Partitioning", IJCAI 2025 (https://www.ijcai.org/proceedings/2025/0296; DOI
   10.24963/ijcai.2025/296). Directly evaluates on NLP4LP; a key comparison system.
2. **ORLM** — "ORLM: A Customizable Framework in Training Large Models for Automated
   Optimization Modeling", arXiv:2405.17743 (v5; accepted Operations Research 2025).
3. **OptMATH** — "OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization
   Modeling", arXiv:2502.11102; ICML 2025 poster (PMLR 267:40769-40802).
4. **DeepOR** — "DeepOR: A Deep Reasoning Foundation Model for Optimization...", AAAI 2026
   (https://ojs.aaai.org/index.php/AAAI/article/view/40699). Also verify **OR-R1** (arXiv
   2511.09092, AAAI 2026) as a 5th optional citation since it is used in comparison docs.

Also verify/add if used in rewrite: **OptiBench** (arXiv 2407.09887, ICLR 2025), **ORQA**
(AAAI 2025), and optionally a recent DKE-anchored reference (e.g., Rizzi et al. 2025 DKE
102452) to demonstrate venue fit.

## 24. Baseline Sufficiency Verdict

- Core retrieval/grounding evaluation: SUFFICIENT.
- End-to-end context: MODERATE — evidence exists (common-18) but is preliminary, partially
  blocked, and must be reported with fidelity labels and status (PRELIMINARY_EXTERNAL_
  BASELINE_STATUS). Do not present a unified leaderboard; report per-system with
  NOT_EVALUABLE/UNAVAILABLE_ARTIFACT preserved.

## 25. Claim-Migration Checklist (for rewrite)

1. Migrate all downstream numbers to frozen patched artifacts (Section 3).
2. Recompute or document oracle control under frozen method; otherwise replace oracle claims
   with the strict-metric robustness result (which is in frozen artifacts).
3. Regenerate all 9 number-bearing tables + significance block from frozen per-query data.
4. Reconcile retrieval-variant numbers and overlap-table offsets into one provenance-clean set.
5. Add 4 verified missing references (Section 23).
6. Convert to elsarticle (or CAS) + elsarticle-num; add keywords/highlights/vitae/data
   statement; replace Declarations with Elsevier format; update abstract <=250 words.
7. Reconcile the no-LLM-comparison limitation with external baseline evidence.

## 26. Recommended Base Manuscript and Next Step

**RECOMMENDED_DKE_BASE_MANUSCRIPT = `manuscript/main.tex`** (most complete, scientifically
current, authoritative editable source; Springer `sn-jnl.cls` — migrate class to
`manuscript/elsarticle.cls` + `elsarticle-num.bst`, both already in-repo). The
`submission_package/` copy is identical source; the repo-root zip is historical and not used.

**Readiness score: 62 / 100.** DKE fit GOOD; rejection-level problems are mechanical
(stale numbers, oracle provenance, missing citations), not scientific.

**ONE next rewrite step:** Migrate `manuscript/main.tex` to `elsarticle.cls` and update the
abstract + all downstream tables/significance numbers to the frozen patched artifacts
(265/331 = 0.8006 InstantiationReady; 255/331 = 0.770393 Strict; Schema R@1 = 0.9094),
recomputing the oracle control from frozen per-query data and adding the four verified
baseline references — with no change to methodology or experimental claims.

---

*Prepared 2026-08-15. Method frozen. No manuscript prose was changed during this audit; the
only repository change is this document. Untracked leftover
`results/pamop/fidelity_diagnostic_gpt5/scaled18_extension.log` (0-byte) was not committed.*