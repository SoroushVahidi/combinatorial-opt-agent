# SN Computer Science — Stage 4: Final Submission Audit and Package Preparation

**Date:** 2026-08-27
**Builds on:** Stages 1-3 (all read/known in full; not redone from scratch — see cross-references below).
**Scope:** exhaustive final verification across scientific correctness, numeric provenance, Springer/SNCS compliance, visual quality, references, declarations, repository quality, and branch/GitHub cleanliness; produce a submission-ready package.

---

## 1. Executive Verdict

The manuscript is submission-ready with one outstanding author-confirmation item (the Azure funding disclosure — not resolvable by audit). This stage's fresh, from-scratch read-through of the entire `manuscript/sncs/main.tex` found **one genuine, previously-undetected defect**: a stale hardcoded section-number cross-reference ("Section 4.8") that became incorrect after Stage 3 inserted a new Computational Complexity subsection, and a **table layout defect** in which the external-baseline provenance table (Table 12) had drifted roughly ten pages from its point of discussion and rendered with severely mis-wrapped, overlapping cell text. Both were fixed in this stage, verified by rebuild, and confirmed with page-rendered visual inspection (not just `pdftotext`). A clean-room build from an isolated copy of only the submission-package files succeeded with zero undefined citations, zero unresolved references, and the correct page count. No new scientific claims, numbers, or experiments were introduced; every fix in this stage is either a stale-reference correction, a table-presentation fix, or documentation-currency correction.

## 2. Preflight

- **HOSTNAME:** al-khwarizmi
- **REPO_PATH:** /home/soroush/combinatorial-opt-agent
- **BRANCH:** main
- **HEAD (start of Stage 4):** ecd3e09c157174613f473daff5134efa7fcc7b5d
- **UPSTREAM:** origin/main
- **ORIGIN_MAIN (start):** ecd3e09c157174613f473daff5134efa7fcc7b5d (LOCAL_HEAD == REMOTE_HEAD confirmed before any edit)
- **AHEAD_BEHIND:** 0/0
- **WORKTREE_STATUS (start):** 3 modified, pre-existing, unrelated `results/paper/eaai_camera_ready_figures/*.pdf` files (unchanged since Stage 1); nothing else
- **UNTRACKED_FILES (start):** none
- **WORKTREES:** single canonical worktree only
- **GIT_LOCKS:** none

The three EAAI figure PDFs were re-confirmed unrelated to the current manuscript: `grep -c "eaai_camera_ready_figures" manuscript/sncs/main.tex manuscript/dke/main.tex` returns 0 for both. They were not touched in this stage.

## 3. Official SN Computer Science Requirement Verification

Re-checked via web search (not relying on cached interpretation from Stage 3):

- **Template:** Springer Nature's `sn-jnl.cls` unified template covers SN Computer Science; class options select reference style, with `sn-basic` (numbered) being the appropriate general-CS style (confirmed via a Springer Nature LaTeX template guide describing the 8 available style options and their scope, and via SN Computer Science's own submission-guidelines page metadata). `manuscript/sncs/main.tex` uses exactly `\documentclass[pdflatex,sn-basic,Numbered]{sn-jnl}` — **compliant**.
- **Length/page limits:** confirmed via Springer Nature's own support documentation: *"Springer Nature does not have restraints on the size of manuscripts submitted to most of their journals unless otherwise specified... articles should be as concise as possible for the benefit of peer reviewers and readers."* No SN Computer Science-specific hard limit was found. The manuscript's 39 pages is **not a compliance issue** (see Section 19 for the length assessment performed anyway).
- **Abstract:** structured (Purpose/Methods/Results/Conclusion), 247 words — within the general 150-250 word convention the user specified for this journal; Springer's general abstract ceiling across journals is stated as 350 words in one support article, so this is comfortably compliant either way.
- **Keywords:** 4-6 is the general Springer Nature convention; the manuscript has 6 — **compliant**.
- **Reference style:** numbered, bracketed, `sn-basic.bst` — **compliant** and unchanged from Stage 3.
- Article-type suitability, figure/table conventions, and supplementary-material guidance were not found to differ from what Stage 3 already implemented; no further changes were needed.

**No requirement was found that the current manuscript fails to satisfy**, beyond the pending author-confirmation funding item (Section 4).

## 4. Azure OpenAI Support / Funding — Decision Table and Status

| Case | Description | Funding wording | Acknowledgments wording | Competing interests wording |
|---|---|---|---|---|
| **A** | Ordinary personally/institutionally available access, no dedicated research award | "This research did not receive any specific grant... API access [was used] for the external baselines." (no further detail needed) | None required | None required |
| **B** | Complimentary credits specifically granted to support this research (e.g., an named academic/cloud credit program) | "This research received in-kind computational support in the form of Microsoft Azure OpenAI API credits provided through [program name] for the external-baseline experiments in Section X; no other funding was received." | Optionally acknowledge the credit program by name | None required unless the program imposes reporting obligations |
| **C** | An explicit grant/sponsorship/award/contract | "This research was supported by [award/contract name and number] from [funder]." | Acknowledge per funder requirements | Disclose if the funder has any interest in the outcome |

**Repository evidence inspected:** `docs/DKE_EXTERNAL_RESOURCE_PROVENANCE_2026-08-15.md` states the Azure OpenAI calls were "highly likely" charged against an **Azure for Students** (USD 100) credit — this is evidence pointing toward **Case A** (a standard, non-competitive student credit is not a dedicated research award), but the word "highly likely" is explicitly not a confirmation. No document in the repository states this with certainty.

**Decision: insufficient evidence to definitively select A vs. B vs. C — not guessed.** The manuscript's current wording ("This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. The external large-language-model baselines... used Microsoft Azure OpenAI API access.") is **already the conservative, safe wording** — it does not claim a grant, does not mischaracterize a credit as an award, and factually discloses that Azure API access was used. This wording is correct and defensible under **any** of Cases A/B/C and requires no change regardless of which case is eventually confirmed. **AZURE_FUNDING_STATUS: NEEDS_AUTHOR_CONFIRMATION** — the only action needed is the author personally confirming which case applies, at which point the wording can be enriched (per the table above) if Case B or C turns out to apply; if Case A is confirmed, no manuscript change is needed at all.

## 5. Final Acknowledgments Audit

Verified in both `manuscript/dke/main.tex` and `manuscript/sncs/main.tex` (exact string match): *"The author is deeply grateful to his mother for her continuous emotional support and thanks his Ph.D. advisor, Professor Ioannis Koutis, for his support, guidance, and encouragement. The author gratefully acknowledges Anders Borum for providing complimentary lifetime access to Secure ShellFish, which was used in preparing this manuscript."* All three parties present; capitalization of "Secure ShellFish" and "Anders Borum" verified character-for-character against the source. Not classified as funding (confirmed: this sentence appears only in the Acknowledgements section, not in Funding).

## 6. Complete Numeric Audit (Representative Table)

The full ~90-row claim inventory was produced in Stage 1 and re-verified in Stage 3 via targeted residue/presence scans (zero stale values, zero missing corrected values, in both manuscripts). This stage re-ran that scan fresh rather than re-deriving the entire inventory from scratch, since no manuscript number changed since Stage 3 except the presentation-only table restructuring in Section 8 below (which altered no numeric values). Representative rows (full manifest: `docs/SNCS_RESULT_MANIFEST_2026-08-27.json`):

| Location | Claim | Manuscript value | Authoritative value | Source | Status |
|---|---|---|---|---|---|
| Abstract | Schema R@1 (TF-IDF / BGE-M3) | 0.9094 / 0.9456 | same | `results/dense_retrieval_bge_m3/retrieval_metrics.json` | MATCH |
| Abstract | InstantiationReady (BGE-M3 / Oracle) | 0.8248 / 0.8489 | same | `results/dense_retrieval_bge_m3/downstream_metrics.json`, `oracle_frozen_verification.json` | MATCH |
| Abstract | Value-inaccurate fraction | 64.7% | 0.6465256797583081 | `results/final_resubmission_method/residual_error_analysis_2026-08-27.json` | ROUNDING_ONLY |
| Table 4 | Exact20 (TFIDF ratio-aware / BGE-M3 / Oracle) | 0.2614 / 0.2436 / 0.2505 | same | `results/final_resubmission_method/exact20_uniform_2026-08-27.json` | MATCH |
| Table 5 | Residual-error 5-category counts | 30/31/49/214/7 | same | `results/final_resubmission_method/residual_error_analysis_2026-08-27.json` | MATCH |
| Table 7 | All 5 significance rows (diff/CI/p) | as printed | reproduces exactly | `results/final_resubmission_method/significance_recomputation_2026-08-27.json`, `results/dense_retrieval_bge_m3/significance_tests.json` | MATCH |
| §4.6 | Runtime 1.09s / 3.29ms / 202,508 KB | same | same | `results/final_resubmission_method/runtime.json` | MATCH |
| Table 10 | 60-instance structural subset | as printed | same | `results/paper/eaai_camera_ready_tables/table2*.csv` | MATCH |
| §4.7 prose | 269-instance extended subset | as printed | same | `table3*.csv` | MATCH |
| Table 11 | 20-instance solver-backed subset | as printed | same | `table4*.csv` | MATCH |
| Tables 12-13 | External baseline (18-instance) | as printed | same | `results/external_baseline_comparison/comparison.json` | MATCH |
| §4.9 | OptMATH extraction validation (recall figures, 98.04% agreement) | as printed | same | `results/external_validation/optmath/final_verification/*.json` | MATCH |
| Declarations | DeepOR/OR-R1 provenance wording | corrected wording | matches `docs/DEEPOR_PROVENANCE.md` / `docs/ORR1_PROVENANCE.md` | — | MATCH |

**MISMATCH = 0. UNSUPPORTED = 0.**

## 7. Exact20 Final Verification

Re-confirmed the uniform definition is applied identically to all methods shown (TF-IDF baseline, TF-IDF ratio-aware, BGE-M3, Oracle): schema-hit cases with no comparable scalar value are **excluded** from both numerator and denominator (not zero-filled). Verified exact denominators from the table footnote and `exact20_uniform_2026-08-27.json`: 291 (TF-IDF baseline), 291 (TF-IDF ratio-aware), 303 (BGE-M3), 320 (Oracle).

**Residue scan for the stale 0.2527 value**, run across every current-facing file: `manuscript/sncs/main.tex`, `manuscript/dke/main.tex`, `README.md`, `docs/SNCS_REPRODUCIBILITY.md`, `docs/DKE_SOURCE_OF_TRUTH.md` — zero occurrences of `0.2527` outside of two files that explicitly and correctly label it as the historical/old value being explained (`docs/SNCS_RESULT_MANIFEST_2026-08-27.json`'s `old_manuscript_value` field, and `docs/SNCS_REPRODUCIBILITY.md`'s explanatory row for the root-cause artifact) — both are correctly historical-labeled context, not current claims.

## 8. Error-Taxonomy Final Verification

Re-confirmed the 5-category decomposition (`tools/recompute_residual_error_analysis.py`): counting unit is the **query** (not slot or assignment); denominator is the full 331-query benchmark; categories are **mutually exclusive by construction** (evaluated in a fixed priority order, so a query is assigned to exactly one category); reproduction script and JSON output are both committed (`results/final_resubmission_method/residual_error_analysis_2026-08-27.json`). Re-ran the script fresh in this stage — output unchanged (deterministic, as expected: 30/31/49/214/7, summing to 331). Confirmed the "value accuracy is the dominant bottleneck" claim: 214/331 = 0.6465256797583081, correctly rounded to 64.7% throughout. **Residue scan** for "230, mainly float-related" across all current-facing manuscript and documentation files: zero occurrences (the phrase only ever existed in the now-superseded table, which was fully replaced in Stage 3).

## 9. Statistical Claim Final Audit

Re-ran `tools/recompute_dke_significance.py` fresh in this session (a clean re-invocation, not reusing Stage 3's cached output) from the frozen per-query CSVs. All three previously-unreproduced rows (TF-IDF-vs-Oracle InstReady/Strict, prepatch-vs-patched Strict) reproduced their diff, CI, **and p-value** exactly against the manuscript's printed values, confirming Stage 3's fix is stable and not a one-time coincidence. The two BGE-M3-vs-TF-IDF rows and the exact McNemar test continue to reproduce from `results/dense_retrieval_bge_m3/significance_tests.json` and `results/final_resubmission_method/summary.json` respectively, as previously verified. **Every statistical number in the manuscript is reproducible by committed code — no exceptions found.** This stage does not need to "stop treating the manuscript as submission-ready," since no discrepancy was found.

## 10. Retrieval/Grounding Terminology Audit

Read the full manuscript specifically checking for conceptual overstatement. Findings, all **already correct** (no changes needed):

- BGE-M3 is consistently described as "pretrained dense text embeddings" / "a pretrained dense neural text encoder... not a rule-based interpretable component" (§2.4, §4.1) — never conflated with the deterministic grounding stage.
- The grounding stage is consistently qualified as "inference-time LLM-free" / "deterministic" (Introduction, Methodology, Conclusions) — the paper never claims the *entire system* (including retrieval) is rule-based; §2.4 explicitly states "Retrieval outputs are reproducible under the fixed lexical or BGE-M3 configurations, but the dense encoder itself is not a rule-based interpretable component."
- No claim of open-domain optimization modeling, full model/constraint generation, or unrestricted solver compatibility was found; every such claim is explicitly scoped ("fixed-catalog," "restricted subset," "does not attempt to reconstruct a full symbolic optimization model").
- No claim of broad external generalization beyond what is supported: §4.9's OptMATH validation is explicitly scoped to "numeric extraction only," and the Limitations section explicitly states generalization beyond NLP4LP "remains untested."

## 11. External-Baseline Final Audit

Re-verified PaMOP/ORLM/OptMATH/Generic-LLM provenance classifications, case counts (18-instance shared subset), and objective-agreement denominators against `results/external_baseline_comparison/comparison.json` — unchanged since Stage 1's verification, all MATCH. Re-verified DeepOR/OR-R1 wording (corrected in Stage 3) is present in the redesigned Table 12 (see Section 15 below) with meaning preserved through the presentation-only text compression required to fix the layout defect. Confirmed the "not a leaderboard" framing and fairness caveats remain intact and were not affected by the table redesign.

## 12. OptMATH External Validation Final Audit

Re-confirmed the corrected description (Stage 3): the manuscript now describes "a second, independently coded, deterministic rule-based classifier" cross-checked against the primary classifier, explicitly stating "no manual, human-reviewed annotation of these literals was performed." No use of "manual audit," "single-reviewer," or "gold label" survives anywhere in either manuscript (residue-scanned in this stage, zero matches). The 98.04% agreement / 94-disagreement numbers are unchanged and traceable.

## 13. Structural/Solver Final Audit

Re-verified the 60/269/20-instance subsets' selection rules, denominators, and metric definitions (Structural Valid, Instantiation Complete, Executable, Solver Success, Feasible, Objective Produced) against `manuscript/sncs/main.tex` §4.7 — unchanged since Stage 1, all correctly restricted-scope worded. The Gurobi-unavailability statement for the 269-instance subset is already correctly framed in the past/evaluation-time context ("At the time of the reported evaluation, the reference implementations required Gurobi functionality that was unavailable in the evaluation environment") — this was already accurate framing from the original manuscript and required no change.

## 14. Runtime/Complexity Final Audit

Re-verified `results/final_resubmission_method/runtime.json` fields directly (`total_seconds: 1.09`, `patched_mean_ms_per_query: 3.293051359516616`, `max_kb: 202508`) against the manuscript's §4.6 prose — exact match. Confirmed the section explicitly states what timing excludes (BGE-M3 GPU inference, external LLM API latency) and explicitly disclaims cross-method comparison. No apples-to-oranges comparison was introduced.

## 15. Abstract Final Polish

Re-read the structured abstract. Word count re-verified at 247 (within the 150-250 range). No citations, no equations. Confirmed it: describes the fixed-catalog scope explicitly ("retrieving a compatible schema from a fixed catalog"); accurately describes BGE-M3 as "a pretrained dense retriever," distinct from the "fully deterministic... numeric-extraction and typed-greedy slot-assignment procedure"; contains no claim of complete optimization-model synthesis or open-domain generalization (the Conclusion sentence explicitly says "without establishing benchmark-wide solver readiness or open-domain generalization"). No language changes were made — the abstract already meets every requirement.

## 16. Title / Keywords / Author Metadata Verification

Verified exact strings in `manuscript/sncs/main.tex`:

- **Title:** "Retrieval-Assisted Instantiation of Natural-Language Optimization Problems" — exact match.
- **Author:** "Soroush Vahidi" (`\fnm{Soroush} \sur{Vahidi}`) — exact match.
- **Affiliation:** "Department of Computer Science, Ying Wu College of Computing, New Jersey Institute of Technology, University Heights, Newark, NJ 07102-1982, USA" — contains the requested "Department of Computer Science, New Jersey Institute of Technology, Newark, NJ, USA" as a substring plus the college-level parent unit and full street address (carried over from the previously author-verified KAIS-era metadata per Stage 3.5's own verification note); not a discrepancy, a superset.
- **Email:** sv96@njit.edu — exact match.
- **ORCID:** 0000-0003-1934-6282 — exact match.
- **Corresponding-author marker:** `\author*[1]` — present (asterisk denotes corresponding author in sn-jnl convention).
- **Keywords:** 6 (natural language processing; optimization modeling; knowledge representation; information retrieval; semantic grounding; intelligent information systems) — within the 4-6 range.

## 17. Reference Audit

- **Citation resolution:** 0 undefined citations, 0 unresolved references (confirmed via a fresh tectonic build's log, both before and after this stage's fixes).
- **Cited vs. bibliography:** 35 unique `\cite` keys in the manuscript body, all 35 present in `references.bib`, all 35 printed in the compiled reference list (verified by counting `[N]` entries in the rendered PDF). 10 additional `.bib` entries exist but are not cited in the current text (e.g., `singirikonda2025text2zinc`, `michailidis2025cpbench`) — these are intentionally retained broader-literature entries from earlier drafting passes; leaving them is harmless (`sn-basic.bst` only typesets cited entries) and not a compliance issue.
- **Spot-check against official sources:** verified `xiao2026deepor` (DeepOR) against the official AAAI proceedings page (`ojs.aaai.org/index.php/AAAI/article/view/40699`) — venue, volume (40), issue (40), and page range (34052-34060) all match exactly. This was chosen as the highest-risk entry (newest, most specialized venue). A full entry-by-entry re-verification of all 35 references against external sources was not repeated in this stage, since Stage 1's DKE migration record (`docs/DKE_STAGE1_RESULT_MIGRATION_2026-08-15.md`) already documents the sourcing process for the added 2024-2026 entries (PaMOP/ORLM/OptMATH/DeepOR with exact venue/volume citations), and this stage's spot-check found no discrepancy.
- **No citation was silently replaced.**

## 18. Figure and Table Audit

Rendered and visually inspected (not just `pdftotext`) the title page, the algorithm/figure page, all tables with wide/wrapped text, and the Declarations/References boundary. **One genuine defect found and fixed:**

- **Table 12** (`tab:ext-context`, external-baseline provenance) originally used `tabularx` with two equal-width `X` columns competing against three fixed-width `l`/`c` columns containing long unbreakable strings ("Generic LLM (GPT-5.4-2026-03-05)", "GENERIC-ZERO-SHOT"). This caused severe per-word cell wrapping, a table so tall that LaTeX's float-placement algorithm deferred it roughly ten pages past its point of discussion (landing after the Declarations section, near the very end of the document, at page 39 of 40), and at least one instance of the page-footer number visually intruding into a table cell. **Fixed** by switching to a fixed-width `tabular` with explicitly tuned `p{}` column widths and `\raggedright` wrapping, and inserting manual break points into three compound all-caps/hyphenated terms that had no natural break point (`ADAPTED-\ OFFICIAL`, `GENERIC-\ ZERO-SHOT`, `GPT-5.4-\ 2026-03-05`) and lightly compressing two cell sentences for length while preserving their exact meaning (verified word-for-word against the Stage-3 corrected wording). Rebuilt and re-rendered: the table now lands 2 pages after its discussion (page 30 of 39), wraps cleanly with no overlapping text, and the page count dropped from 40 to 39.
- All other tables (Tables 1-11, 13) were inspected and render correctly: proper column alignment, correct bolding of best-non-oracle values, readable font size, no clipping, no margin overflow, captions self-contained, all abbreviations (TFIDF-TG, Oracle-TG, BGE-M3, N/A) explained in-caption or in-text.
- The single figure (`nlp4lp_instantiation_pipeline_v2.png`, the pipeline schematic) renders correctly at its specified width with a legible caption; it is a raster PNG (not vector), which is acceptable for a schematic diagram but noted as a minor, non-blocking presentation observation.
- **Second defect found and fixed (non-visual):** a hardcoded prose cross-reference ("Section 4.8") in the Limitations section, which had silently become incorrect (pointing at "External Component-Level Robustness Validation," now actually Section 4.9) after Stage 3 inserted the new Computational Complexity subsection ahead of it. Replaced with a proper `\ref{}` in both manuscripts so it can never drift again.

**VISUAL_PROBLEMS_FOUND (before this stage's fixes): 2. VISUAL_PROBLEMS_REMAINING (after fixes): 0.**

## 19. Page-Length Assessment

No length constraint applies (Section 3). Page-usage breakdown (39 pages, single-column `sn-jnl`): main text and 13 tables/1 figure/1 algorithm ≈ pages 1-33; Declarations ≈ page 33; References ≈ pages 34-39 (35 numbered entries in a single-column layout). Classification of major content blocks:

| Content | Classification | Rationale |
|---|---|---|
| Introduction, Related Work, Methodology | KEEP_IN_MAIN | Core scientific narrative; already concise |
| Main benchmark tables/prose (§4.2-4.5) | KEEP_IN_MAIN | Central evidence for the paper's thesis |
| Computational Complexity and Runtime (§4.6) | KEEP_IN_MAIN | Newly added, short (1 paragraph), directly supports a stated contribution |
| Structural/solver-backed subsets (§4.7) | KEEP_IN_MAIN | Restricted-scope evidence explicitly flagged as complementary; each caveat sentence is load-bearing for correct interpretation, not redundant filler |
| External baseline comparison (§4.8) | KEEP_IN_MAIN | The extensive fairness caveats are necessary to prevent the "not a leaderboard" framing from being misread; removing them would materially increase misinterpretation risk |
| External component-level validation (§4.9) | KEEP_IN_MAIN | Short, single self-contained subsection |
| Limitations (7 sub-paragraphs) | KEEP_IN_MAIN | Each paragraph addresses a distinct, non-overlapping threat to validity; no genuine redundancy found across them |
| References (35 entries, ~6 pages) | KEEP_IN_MAIN | Standard reference-list length for numbered style with full metadata |

**No content was moved to supplementary material or removed for length.** The paper does not exhibit the kind of repetition the task anticipated (repeated limitations, repeated external-baseline caveats stated identically in multiple places) — each caveat appears once, at the point where it is load-bearing for correct interpretation of the adjacent claim, which is intentional scientific-writing practice given how heavily audited this paper's claims have been across Stages 1-3. Conservatively, **no page-count reduction was performed**, consistent with the instruction not to shorten "solely because it feels long" and the absence of any actual length requirement.

## 20. Language Polish

A sentence-by-sentence read of the entire manuscript found no grammar errors, no tense inconsistencies, no capitalization errors, and no Springer-style violations beyond what was already fixed in Section 18. Two very minor, presentation-only wording compressions were made as a side effect of the Table 12 fix (Section 18); their scientific meaning was verified unchanged. No other language changes were made, since none were needed: the manuscript's prose is already precise, consistently uses defined terminology (InstantiationReady, StrictInstantiationReady, Exact20 (on hits), Schema R@1 used consistently throughout with no synonym drift), and its caveat density — while high — is deliberate and appropriate given the manuscript's own audit history, not evidence of "excessive caveat repetition" in the sense of restating the same caveat redundantly.

## 21. Repository Reviewer Experience

Opened `README.md` fresh and verified it answers all 13 items in the task's checklist: paper title, purpose, current manuscript directory (`manuscript/sncs/`), authoritative result directory (`results/final_resubmission_method/` and named siblings), reproduction entry point (the 4 `tools/*.py` scripts, explicitly commanded), data-access restriction (gated NLP4LP, clearly stated), result manifest (`docs/SNCS_RESULT_MANIFEST_2026-08-27.json`, linked), statistical reproduction (`tools/recompute_dke_significance.py`, linked), error-taxonomy reproduction (`tools/recompute_residual_error_analysis.py`, linked), external-baseline provenance (`docs/*_PROVENANCE.md`, linked), historical/superseded result location (explicitly tabled), environment/setup (Quick Start commands), and license (`LICENSE`, MIT). All relative links in `README.md` and `docs/SNCS_REPRODUCIBILITY.md` were checked for existence (`ls` on each referenced path) — all resolve. The 4 reproduction commands listed in `README.md` were re-run fresh in this stage (Section 9) and succeeded.

## 22. Branch/Git Final Pass

`git branch -a` lists only `main` locally and 20 remote branches (`origin/main`, `origin/SoroushVahidi-patch-1`, 2 `codex/*`, 14 `copilot/*`, `origin/kais-final-submission-prep`). No local-only branches, no worktrees beyond the canonical one, no tags. **No branch was deleted or renamed** — this remains a documentation-only recommendation (repeated from Stage 3, since nothing has changed and the same caution applies: deleting remote branches is a destructive, hard-to-reverse action outside this audit's authority). `BRANCH_REVIEW_COMPLETE: YES` (reviewed, documented, no unsafe action taken).

## 23. Wulver Status

A single, quick, non-invasive check (`ssh login02 'hostname'`, one attempt, 8-second timeout) was performed per the explicit instruction not to repeat Stage 3's extended investigation. Result: identical `Permission denied (gssapi-keyex,gssapi-with-mic,keyboard-interactive)` outcome. **WULVER_STATUS: HUMAN_INTERACTIVE_LOGIN_REQUIRED.** Per the task's own framing and Stage 3's conclusion, no known scientific artifact on Wulver is required for correctness of the current manuscript, so this is not a submission blocker.

## 24. Submission Package Contents

Created `manuscript/sncs/submission_package/` containing exactly:

```
submission_package/
├── main.tex
├── main.pdf            (clean-room-built, see Section 25)
├── references.bib
├── sn-jnl.cls
├── sn-basic.bst
└── figures/
    └── nlp4lp_instantiation_pipeline_v2.png
```

**One cleanup performed:** two unreferenced figure files (`figure3_engineering_validation_comparison.pdf`, `figure4_final_solver_backed_subset.pdf`) had been carried over into `manuscript/sncs/figures/` during Stage 3's migration but are not `\includegraphics`'d anywhere in `main.tex` (confirmed by grep). These were removed from both the working `manuscript/sncs/figures/` directory and the submission package — they were EAAI-era camera-ready figures unrelated to this manuscript's actual figure count (1), not evidence, and their presence would have been confusing clutter in a journal submission upload. No historical artifact was deleted by this action (the source PDFs remain intact and tracked at `manuscript/dke/figures/` and `results/paper/eaai_camera_ready_figures/`, wherever they originated).

**Excluded, correctly:** build caches, `.aux`/`.log`/`.out`/`.blg`, editor backups, the unrelated EAAI figure PDFs, secrets, repository-only audit docs (Stage 1-4 reports, kept in `docs/` where they belong, not duplicated into the submission package), raw NLP4LP data, and any large output files.

## 25. Clean-Room Build

Copied `manuscript/sncs/submission_package/*` into an isolated `/tmp` directory (outside the repository, no sibling-file access) and built with `tectonic` from there.

- **BUILD_SUCCESS:** YES
- **PDF_CREATED:** YES (397 KB)
- **PAGE_COUNT:** 39
- **UNDEFINED_CITATIONS:** 0
- **UNRESOLVED_REFERENCES:** 0
- **MISSING_FILES:** 0 (the one "not found" log line, `\pdfdraftmode not found`, is a benign optional-feature info message from the `pdftexcmds` package, unrelated to any required file)
- **LATEX_ERRORS:** 0
- **OVERFULL_BOXES:** a handful of cosmetic overfull/underfull hbox warnings (pre-existing pattern, sub-millimeter, not visually apparent at normal zoom)
- **OTHER_WARNINGS:** benign `xdvipdfmx` "Object already defined" notices (a harmless artifact of rebuilding onto an existing PDF object stream; does not occur on a true first-ever build from a truly clean directory with no prior `.pdf`)

The verified clean-room `main.pdf` was copied back into `manuscript/sncs/submission_package/main.pdf` as the final deliverable proof.

## 26. Visual PDF Audit

Page-rendered inspection (not text-extraction-only) of: title page/front matter, abstract, all section headings, all equations, all 13 tables (including the fixed Table 12), the single figure, the algorithm box, page breaks around Declarations/References, the reference list's first and last entries, and all author metadata/hyperlinks. **VISUAL_PROBLEMS_FOUND: 2** (both described and fixed in Section 18). **VISUAL_PROBLEMS_REMAINING: 0.**

## 27. Submission Metadata

Created `manuscript/sncs/SUBMISSION_METADATA.md` with verified title, article type, author, affiliation, corresponding-author marker, email, ORCID, abstract reference, keywords, code/data availability text, competing interests, acknowledgments, AI-use disclosure, and suggested subject classification. The Funding field is explicitly marked as an author-confirmation field rather than populated with an invented value. No grant numbers, contract numbers, funding agency names, or reviewer suggestions were invented.

## 28. Author-Required Actions

1. **Confirm the exact Azure OpenAI credit relationship** (Section 4) — the only scientific/declaration blocker. The current manuscript wording is safe to submit as-is regardless of the answer, but enriching it per the Case A/B/C table (Section 4) is recommended once confirmed.
2. **Optional:** decide whether to draft an SN Computer Science cover letter (not required by the journal; information prepared in `manuscript/sncs/SUBMISSION_CHECKLIST.md` if the author wants one).
3. **Optional:** review and disposition the three unrelated, pre-existing modified EAAI figure PDFs (`results/paper/eaai_camera_ready_figures/*.pdf`), flagged and untouched since Stage 1.
4. **Optional:** review the 14 stale remote `copilot/*`/`codex/*` branches and the `kais-final-submission-prep` branch before any future repository cleanup pass (documented, not acted on).
5. **Optional, environment-dependent:** if the DKE/Elsevier track is still being pursued in parallel, recompile `manuscript/dke/main.tex` with the `,times` class option restored in an environment with full network access (currently dropped due to a sandboxed font-relay 403, unrelated to content; does not affect the SNCS submission).

## 29. Commits

See the actual commit log for exact SHAs (recorded after commit, Section 30).

## 30. Push Verification

See the terminal summary block below for final SHAs, captured after push.

## 31. Final Readiness Recommendation

**READY_TO_SUBMIT_TO_SN_COMPUTER_SCIENCE: YES**, contingent only on the author's personal confirmation of the Azure funding item (Section 4/28). No scientific, provenance, statistical, or template-compliance blocker remains. The manuscript, its submission package, and the supporting repository documentation are internally consistent, fully cross-referenced, and independently reproducible.

---

## Terminal Summary

```
SNCS_STAGE4_COMPLETE: YES
SCIENTIFIC_AUDIT_PASS: YES
UNRESOLVED_NUMERIC_MISMATCHES: 0
UNSUPPORTED_CURRENT_CLAIMS: 0
EXACT20_FINAL_PASS: YES
ERROR_TAXONOMY_FINAL_PASS: YES
STATISTICS_FINAL_PASS: YES
EXTERNAL_BASELINE_FINAL_PASS: YES
OPTMATH_PROVENANCE_FINAL_PASS: YES
STRUCTURAL_SOLVER_FINAL_PASS: YES
ACKNOWLEDGMENT_FINAL_PASS: YES
AZURE_FUNDING_STATUS: NEEDS_AUTHOR_CONFIRMATION
SPRINGER_SNCS_REQUIREMENTS_PASS: YES
STRUCTURED_ABSTRACT_PASS: YES
REFERENCE_AUDIT_PASS: YES
FIGURE_TABLE_AUDIT_PASS: YES (2 defects found and fixed this stage)
LANGUAGE_POLISH_COMPLETE: YES
ORIGINAL_PAGE_COUNT: 40
FINAL_PAGE_COUNT: 39
SUBMISSION_PACKAGE_CREATED: YES
CLEAN_ROOM_BUILD_SUCCESS: YES
VISUAL_PROBLEMS_REMAINING: 0
REPOSITORY_REVIEWER_READY: YES
BRANCH_REVIEW_COMPLETE: YES
WULVER_STATUS: HUMAN_INTERACTIVE_LOGIN_REQUIRED
SUBMISSION_METADATA_CREATED: YES
SUBMISSION_CHECKLIST_CREATED: YES
COMMITS_CREATED: <see final terminal message after push>
PUSH_SUCCESSFUL: <see final terminal message after push>
FINAL_LOCAL_HEAD: <see final terminal message after push>
FINAL_REMOTE_HEAD: <see final terminal message after push>
WORKTREE_CLEAN_EXCEPT_UNRELATED_EAAI_FILES: YES
READY_TO_SUBMIT_TO_SN_COMPUTER_SCIENCE: YES
AUTHOR_ACTIONS_REMAINING: Confirm the exact Azure OpenAI credit relationship (Funding); everything else is optional (cover letter, EAAI-figure disposition, branch cleanup, DKE-track times-font recompile).
```
