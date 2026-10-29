# JDSA Phase 2 scientific revision report

**Status: scientific revision, verification and document builds completed; dedicated branch prepared for public review.** This is not journal submission. The branch commit and remote publication are reported in the final delivery; this report is part of that commit.

- Source branch: `manuscript/jdsa-phase1-evidence-results-20261029`
- Exact source commit: `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650`
- New isolated worktree: `/home/soroush/combinatorial-opt-agent-jdsa-phase2`
- Revision branch: `manuscript/jdsa-phase2-scientific-consolidation-20261029`
- Target repository: [SoroushVahidi/combinatorial-opt-agent](https://github.com/SoroushVahidi/combinatorial-opt-agent)
- No merge or push to main; Phase 1 is unchanged.

## Review artifacts

- [Complete main PDF — 30 pages](../../manuscript/jdsa-phase2/Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase2.pdf)
- [Complete supplement PDF — 32 pages](../../manuscript/jdsa-phase2/Retrieval_Assisted_Scalar_Grounding_JDSA_Phase2_Supplement.pdf)
- [Editable main source](../../manuscript/jdsa-phase2/main.tex), [supplement source](../../manuscript/jdsa-phase2/supplement.tex), [complete source package and build instructions](../../manuscript/jdsa-phase2/README.md)
- [Old-to-new section map and all 47 investigation destinations](SECTION_RESTRUCTURING_MAP.md)
- [Claim-to-evidence verification](CLAIM_TO_EVIDENCE_VERIFICATION.md), [machine-readable checks](SCIENTIFIC_CHECKS.json)
- [Primary reference and journal-convention verification](REFERENCE_VERIFICATION.md)
- [Remaining scientific and formatting issues](REMAINING_ISSUES.md)
- [Artifact hashes and build/visual receipt](ARTIFACT_VERIFICATION.json)

## Substantive changes

The paper now has seven sections. Experimental Results progresses from schema retrieval through readiness, full scalar agreement, matched grounding and controlled diagnostics to complementary formulation, extraction, relationship and semantic-contrast evidence. The new Discussion interprets those endpoints, explains why scalar gains can coexist with lower readiness, limits causal conclusions about search, and connects numerical binding to source fidelity and practical interfaces. Limitations are grouped by assistance/reference validity, historical reproducibility, sampling/dependence and semantic assessment. The Conclusion answers the principal questions without re-listing secondary model scores.

The Phase 1 Abstract is verbatim. The Introduction changes only its organizational roadmap. The whole Phase 1 core Results body is unchanged apart from the section title. **All nine original main tables and all 24 original supplementary tables are byte-for-byte preserved**, including Tables 1–6 and the common-18/relationship tables. All original figures, algorithms and supporting assets remain. No research artifact is removed or overwritten.

Common-18 remains a heterogeneous, selected formulation comparison with separate execution and objective endpoints. The 11/15 and 12/16 counts keep their distinct conditional denominators; PaMOP's aggregate is withheld and ORLM remains unavailable. Recovery stages in the relationship study remain explicit: 500 original-valid, 59 repaired and 72 valid historical replacements produce 631/651 structurally valid outputs; κ≈0.371 concerns 197 complete selected relationships, not semantic accuracy. D1 is integrated as completed author-validated, language-model-assisted construction, not provider performance or independent human gold.

The supplement retains full terminal-objective predicates, precision/failure accounting, service settings, recovery lineage, pairwise agreement and source clusters. It adds two self-contained tables summarizing all 12 D1 pairs and preserves their annotation limitations. G1–G3 detailed participation-set accounting stays in S5; the main paper states only its methodological implication. All 47 registered groups have destinations, including negative, partial, inconclusive, blocked and proposed work with their original status.

## Explicit evidence-attribution correction

The old OptMATH paragraph attributed 7.28% constants to the 150-instance secondary-classifier comparison. Saved category counts instead reproduce it as **2,310/31,736 across the 1,000-instance automatic audit**, including implementation, solver/API, index/loop and array/dimension categories. The percentage is retained; its population and category attribution are corrected in S9.1. The independent 150-instance classifier comparison remains 4,707/4,801=98.04% agreement. No measurement, table value or experimental observation was replaced. This is the only newly identified numerical-provenance correction; the missing corrected extraction counts and summary-only confidence interval remain disclosed.

## Verification performed

1. Inspected Git/worktrees/ancestry; started exactly from the authorized Phase 1 commit in a new worktree. Read the complete Phase 1 main/supplement, register, map, manifest, verification report and relevant M3/M5/D1/common-18/relationship/OptMATH/G1–G3 evidence.
2. Consulted official JDSA guidelines and the three requested published regular papers for organization. Verified primary references used in revised sections. The binding work is explicitly marked as an author preprint with author-reported COLM acceptance, rather than an authenticated separate proceedings version.
3. Re-ran only deterministic checks of saved records. All 65 Phase 1 manifest hashes match. Core and historical saved-metric checks passed, as did corrected common-18 arithmetic, relationship counts/κ, extraction counts, D1 label-pair preservation and the 47-group destination coverage.
4. Built both PDFs from clean temporary source with the established pdfLaTeX/BibTeX procedure. Final logs contain no unresolved citations/cross-references, missing characters or overfull boxes. All displayed equations are numbered. No auxiliary files or deliverable named `main.pdf` are included.
5. Inspected rendered contact sheets covering every main and supplement page, with enlarged checks of the common-18 and relationship tables, conclusion and D1 tables. Text, tables, figure, equations and bibliographies render within the page. All 35 external PDF links resolve to named destinations in the companion PDF. PDF text extraction has no replacement characters. Some inherited float-only supplementary pages retain whitespace; comprehensive formatting is deferred.
6. Checked exact repeated paragraphs of at least 45 words within each expanded document: none. Manually consolidated protocol repetition while retaining interpretive qualifications. The 227-word Abstract and six keywords remain consistent with the revised narrative.
7. Checked read-only publication metadata. Public Hugging Face v1.2 remains at `24a047fa5d661be6cfcf75980ac8f62e6c942243`; the M5 evidence candidate is local and not uploaded. The public manuscript source does not imply public release of all underlying records.
8. Verified main remains `a3f0dde397341a41842a30ddb946e7966706351b` and Phase 1 remains `a09032a23c4c44a5d6d2a5215b3a8c1ca3b1a650`. All 115 existing untracked user files in the main worktree retain their original hashes. Changes are additive under the two Phase 2 directories only.

## Scope and next step

Experiments executed: **NO**. Model inference/training/solver/Slurm execution: **NO**. New datasets collected: **NO**. Scientific observations fabricated or regenerated: **NO**. Hugging Face modified: **NO**. Main/Phase 1 modified: **NO**. Journal submission: **NO**.

Review the revised PDFs and this evidence-preservation record before authorizing any further work. Final two-column JDSA formatting and any separately approved additive evidence release remain future tasks. Phase 3 has not begun.
