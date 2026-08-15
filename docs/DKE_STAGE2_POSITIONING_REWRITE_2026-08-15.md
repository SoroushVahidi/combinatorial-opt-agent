# DKE Stage 2: Positioning Rewrite Audit (2026-08-15)

Audit of the Stage 2 manuscript change for the DKE resubmission: rewritten Introduction (7-paragraph structure) and Related Work promoted to a standalone top-level section with conceptual subsections and an explicit differentiation paragraph.

## 1. Introduction

The manuscript `manuscript/dke/main.tex` (formerly 584 lines, now 645 lines) received a targeted Stage 2 rewrite confined to the front matter: the old `\subsection{Background and Motivation}` / `\subsection{Related Work}` / `\subsection{Problem Scope and Proposed Perspective}` / `\subsection{Contributions and Practical Implications}` structure was replaced by a 7-paragraph Introduction and a standalone `\section{Related Work}` (Section 2) with four conceptual subsections. The frozen method, experiments, results, and all validated numbers were not modified. The manuscript compiles cleanly with tectonic 0.16.9; final PDF is 24 pages (was 23), a +1 page increase for the new Related Work section, within the allowed budget.

## 2. Contributions

The four contributions are preserved verbatim in substance and now stated once (in the Contributions paragraph), not repeated across intro subsections:

1. Problem formulation and evaluation framework: two-stage decomposition (schema retrieval; schema-conditioned scalar parameter grounding) with retrieval-dependent evaluation design.
2. Transparent methodology: fully deterministic pipeline (lexical schema retrieval, rule-based numeric extraction, coarse type inference, no-reuse compatibility-based slot assignment).
3. Comprehensive empirical diagnosis: oracle controls, strict-metric sensitivity, numeric-extraction ablation, paired bootstrap tests, lexical-overlap robustness tests, error taxonomy — identifying semantic quantity-to-role grounding as the dominant bottleneck.
4. Reproducibility and downstream validation: public implementation plus restricted structural and solver-backed evidence from frozen artifacts.

## 3. Related Work

Related Work is now a top-level section (Section 2, `\label{sec:related_work}`) with four subsections:

- `subsec:rw_nl_modeling` — Natural Language and Mathematical Optimization Modeling (auto-formulation, NL4Opt, LaTeX2Solver, Ner4Opt, Text2Zinc, CP-Bench, EHOP, surveys).
- `subsec:rw_llm_generation` — LLM-Based Optimization-Model Generation (OptiMUS, OptLLM, Chain-of-Experts, LLMOPT, OPT2CODE, Autoformulation, PaMOP, ORLM, OptMATH, ReSocratic/OptiBench, ORQA, MAMO, OR-R1, DeepOR). Distinguishes output types per method (formulations / solver code / executable outputs).
- `subsec:rw_retrieval` — Retrieval, Schema Selection, and Structured Grounding (BM25, TF-IDF, LSA, Spider, NLI/DB survey, ReMatch, Rizzi et al. DKE, Atzeni et al. DKE).
- `subsec:rw_position` — Position of the Present Work: explicit differentiation from end-to-end generation systems, component-level extraction (Ner4Opt), and explainability-oriented work; restates the empirical question (retrieval vs. grounding).

## 4. Closest prior work

- Autoformulation (ICML 2025): search over hierarchically decomposed formulation components with MCTS — nearest in goal (decomposing modeling) but produces formulations, not schema-conditioned scalar instantiation over a fixed catalog.
- OptiMUS / OptLLM / OPT2CODE / LLMOPT / Chain-of-Experts: end-to-end formulation/code generation — broader task, opaque intermediate decisions, not retrieval-gated grounding.
- Ner4Opt: entity/quantity extraction without retrieval-conditioned slot eligibility.
- ReMatch: schema matching via retrieval + LLMs over relational schemas — methodologically adjacent but different target (relational vs. optimization-model schemas).
- Rizzi et al. (DKE 159:102452) and Atzeni et al. (DKE 161:102494): DKE-community grounding of natural language into structured artifacts — closest DKE framing.

## 5. New references

Nine new entries added to `manuscript/dke/references.bib`, all verified against primary sources:

- `affolter2019nli` — Affolter et al., VLDB J 28(5):793–819 (DOI 10.1007/s00778-019-00567-8). Survey of NLI for databases.
- `yu2018spider` — Yu et al., EMNLP 2018, pp. 3911–3921 (DOI 10.18653/v1/D18-1425).
- `sheetrit2024rematch` — Sheetrit et al., arXiv:2403.01567. Schema matching with retrieval + LLMs.
- `astorga2025autoformulation` — Astorga, Liu, Xiao, Van Der Schaar, ICML 2025, PMLR 267:1864–1886.
- `yang2025optibench` — Yang et al., ICLR 2025 (ReSocratic + OptiBench).
- `mostajabdaveh2025orqa` — Mostajabdaveh et al., AAAI-25, 39(23):24902–24910 (DOI 10.1609/aaai.v39i23.34673).
- `ding2026orr1` — Ding et al., AAAI-26, 40(1):228–236 (DOI 10.1609/aaai.v40i1.36983).
- `rizzi2025conceptual` — Rizzi et al., DKE 159:102452 (DOI 10.1016/j.datak.2025.102452). LLMs for multidimensional conceptual design.
- `atzeni2026semantic` — Atzeni et al., DKE 161:102494 (DOI 10.1016/j.datak.2025.102494). LLM semantic layer for query answering.

Citation audit: 36 keys cited, 0 missing from the bib, 0 unused. All bib entries resolve. A minor pre-existing inconsistency in the DeepOR entry (booktitle "Thirty-Ninth AAAI" with `{AAAI}-26`) was corrected to "Fortieth AAAI" (AAAI-26 = volume 40).

## 6. DKE-specific references

Two DKE-community references were added and cited in `subsec:rw_retrieval` to ground the framing of knowledge-instantiation / grounding into structured artifacts:

- `rizzi2025conceptual` (Data & Knowledge Engineering 159:102452): LLM support for conceptual design of multidimensional cubes.
- `atzeni2026semantic` (Data & Knowledge Engineering 161:102494): LLM as intermediate semantic layer for query answering over heterogeneous data.

Both are used only to position the schema-catalog-as-structured-knowledge-base view, not to rename optimization concepts in DKE jargon. "Knowledge engineering" is used sparingly (abstract + framing paragraphs), never to overclaim.

## 7. Claims corrected / weakened

- Removed the "the remainder of the introduction positions this perspective relative to prior work" meta-sentence (obsolete after Related Work became a standalone section).
- Intro evaluation paragraph states the modest oracle gain and the numeric-extraction ablation as evidence that number-to-slot grounding is the bottleneck, with the explicit caveat "without claiming benchmark-wide solver readiness."
- Contribution 3's diagnostic list in the intro mirrors the frozen experiments; no new experimental claims.
- The strict-metric numbers (0.8006 → 0.8489 oracle; 0.7704 → 0.8489 schema-gated) are stated only in the Contribution 3 paragraph with the "statistically significant but practically modest gain" framing, matching the manuscript body.
- No "state-of-the-art", "novel", "no prior work", or "to the best of our knowledge" claims introduced. Grep for stale claims ("first", "general", "optimal", "state-of-the-art", "novel") found only benign usages.

## 8. Terminology changes

- "schema-conditioned scalar parameter instantiation" retained as the core term.
- Related Work now describes each LLM system by its output type (formulation vs. modeling-language code vs. solver code vs. schema/parameter instantiation) rather than lumping them as "end-to-end."
- New term "retrieval-assisted optimization-model support" used in the positioning paragraph; consistent with abstract framing.
- "knowledge-driven model instantiation" introduced once in the framing paragraph, tied to `nlp4lp_dataset`, OptiMUS, and ReMatch citations, and supported by the DKE refs.

## 9. Abstract consistency

The abstract was NOT modified (still 227 words). The rewritten Introduction is consistent with it: the abstract's claimed numbers (0.9094 Schema R@1, 0.8006 InstantiationReady, 0.7704 strict criterion, 0.8489 oracle) are reproduced identically in the intro. No contradictions introduced.

## 10. Citation audit

- Per-sentence citation rule (≤4 keys per sentence) holds: automated scan of the new Introduction + Related Work region found zero sentences with >4 citations after one fix (a sentence combining 3+3 keys was split into two sentences).
- 36 unique keys cited across the manuscript; all present in `references.bib`; no unused bib entries.
- Label audit: removed labels `subsec:intro_background` and `subsec:intro_scope` are not referenced anywhere (grep exit 1). `subsec:intro_contributions` re-added after "This paper makes four contributions." so the Conclusion's `\ref` resolves. Two `\ref{subsec:intro_related}` in Limitations updated to `\ref{sec:related_work}`. Compile reports 0 undefined references/citations.

## 11. Compile status

`tectonic main.tex` (0.16.9): success. Final PDF 24 pages (was 23). 0 undefined references/citations. Overfull hboxes remaining (all pre-existing in untouched sections): line 230 (0.29pt), 589 (1.11pt), 602 (4.54pt), 608 (3.20pt), 614 (29.44pt); underfull at line 303 (table area). The one new-content overfull at line 54 (10.53pt, caused by an unbreakable compound in the first intro paragraph) was eliminated by rewording.

## 12. Reviewer mini-audit

Target journal: Data & Knowledge Engineering.

| Dimension | Score /10 | Notes |
|---|---|---|
| Novelty clarity | 7 | Framing as knowledge-instantiation over a fixed catalog is now crisper; contribution list is unchanged and defensible. |
| Related-work completeness | 8 | Substantially improved: DKE-specific grounding refs, output-type differentiation across LLM systems, 2025–2026 coverage. |
| Difference from prior work | 8 | Explicit position subsection now contrasts end-to-end generation, component extraction, and relational schema matching. |
| DKE scope fit | 7 | Direct DKE refs added; risk remains that optimization framing reads as OR-flavored rather than DKE — mitigated by the retrieval/grounding framing. |
| Writing quality | 9 | Tighter prose, no redundant meta-text, consistent terminology, ≤4 citations per sentence. |

Strongest remaining concerns for the journal: (1) the DKE fit argument depends on the reviewer accepting "optimization schema catalog as a structured knowledge base" framing; (2) limited quantitative comparison against LLM systems (left to Stage 3 / future work), which DKE reviewers may probe; (3) benchmark-scoped evidence only (no external dataset / deployment study).

## 13. Remaining risks

- DKE reviewers may expect deeper knowledge-engineering terminology integration; the current usage is deliberately light to avoid mechanical DKE renaming.
- Cross-ref integrity depends on labels `sec:related_work`, `sec:method`, `sec:experiments`, `sec:conclusion` staying stable through later stages.
- The +1 page increase is within budget but should be re-checked after any later-stage additions.

## 14. Files changed

- `manuscript/dke/main.tex` — rewritten Introduction + standalone Related Work section; label fixes; 645 lines.
- `manuscript/dke/references.bib` — 9 new verified entries; DeepOR booktitle ordinal fix.
- `manuscript/dke/main.pdf` — recompiled artifact (24 pages).
- `docs/DKE_STAGE2_POSITIONING_REWRITE_2026-08-15.md` — this audit.

## 15. Git state

- `git diff --check` passes (no whitespace errors).
- Modified: `manuscript/dke/main.tex`, `manuscript/dke/main.pdf`, `manuscript/dke/references.bib`.
- Untracked (not to be committed): `results/pamop/fidelity_diagnostic_gpt5/scaled18_extension.log`.
- Suggested commit message: `manuscript: rewrite DKE introduction and related work`.

## 16. Next action

Commit the three manuscript files with `manuscript: rewrite DKE introduction and related work` and push to `origin/main` (do NOT add the untracked `scaled18_extension.log`). If the user prefers to review the diff first, stop here for review. After commit, Stage 2 is complete; Stage 3 (full external-baseline comparison) remains out of scope as planned.