# Manuscript: KAIS Submission Package

**Target journal:** *Knowledge and Information Systems* (KAIS), Springer Nature.
**Status:** Adapted from an EAAI/Elsevier (`elsarticle`) draft uploaded to the repository
on 2026-07-23 (`Retrieval_Assisted_Instantiation_of_Natural_Language_Optimization_Problems.zip`,
repo root). This folder is now the authoritative, editable manuscript source.

## Provenance

The original zip (still present at the repo root, untouched) contained an EAAI-targeted
draft: `\journal{Expert Systems with Applications}` (a stale macro left over from an even
earlier ESWA submission attempt), an author-year (`elsarticle-harv`) bibliography style, a
cover letter addressed to the EAAI Editor-in-Chief, and an anonymized `\author{}` field in
`main.tex` (consistent with `docs/EAAI_SOURCE_OF_TRUTH.md`'s note that the EAAI submission
was double-anonymized). This folder re-targets all of that content at KAIS.

## What changed and why

1. **Bibliography style → numbered.** KAIS's official submission guidelines
   (`link.springer.com/journal/10115/submission-guidelines`, fetched during this pass)
   specify numbered bracketed citations (`[1]`, `[1-3,7]`), not author-year. Switched
   `\bibliographystyle{elsarticle-harv}` → `elsarticle-num` and dropped the `authoryear`
   class option. All `\cite{}` calls in the text were already plain (no `\citet`/`\citep`),
   so this is a drop-in, verified-safe change.
2. **Removed the stale `\journal{Expert Systems with Applications}` macro** (leftover from
   an earlier ESWA draft; not applicable to KAIS and not used by the Springer template).
3. **Author identity restored in `main.tex`.** The EAAI draft left `\author{}` blank for
   double-anonymized review. **KAIS's blinding policy could not be confirmed** from the
   publicly accessible Springer Nature pages (see "Open item" below), so the manuscript
   now includes the author name/affiliation/ORCID directly, on the assumption of
   single-blind review (the common default for Springer computer-science journals using
   Editorial Manager). **Confirm this before submitting** -- if KAIS requires
   double-anonymous review, blank out `\author`/`\address` in `main.tex` again and remove
   the GitHub URL from the Declarations section.
4. **Cover letter rewritten** for KAIS (Editor-in-Chief addressed generically, since the
   named EIC changes over time; journal name, scope framing, and key numbers updated).
5. **New "Declarations" section** added before the bibliography (Funding, Competing
   interests, Ethics, Data availability, Code availability, Author contributions,
   Acknowledgements, Generative-AI disclosure), matching the structure Springer Nature's
   guidelines describe. The old Elsevier-specific `highlights.tex` and
   `declaration-of-interest.tex` files are kept only as optional supplementary material
   (KAIS does not require a "Highlights" file; competing-interest info is otherwise
   collected via the submission portal).
6. **Keywords revised** to 6 terms aligned with KAIS's scope (natural language
   processing, optimization modeling, knowledge representation, information retrieval,
   semantic grounding, intelligent information systems), replacing more Elsevier-flavored
   terms ("engineering decision support").
7. **Six new/expanded content pieces**, all sourced from already-computed, committed
   repository artifacts (no new experiments were run and no numbers were invented):
   - Fixed three instances of an internal naming bug: the manuscript's own footnotes
     referred to `tfidf_ambiguity_beam` / `tfidf_ambiguity_abstain`, but the actual
     measured result files (and repository result CSVs) use
     `tfidf_ambiguity_aware_beam` / `tfidf_ambiguity_aware_abstain`. The reported
     numbers were already correct; only the method-name strings were wrong.
   - Added a formal description of "optimization-role compatibility" and "admissibility"
     (Sec. 2.2), grounded in the actual `OPT_ROLE_WORDS` lexicon-based scoring logic in
     `tools/nlp4lp_downstream_utility.py`, explicitly framed as heuristic (no formal
     correctness guarantee claimed).
   - Added a "Statistical Significance and Lexical-Overlap Robustness" subsection
     (Sec. 3.4) with two new tables. The Experimental Setup section (Sec. 3.1) already
     *promised* bootstrap significance tests and overlap stress tests that were never
     delivered anywhere in the original draft -- this was a real, reviewer-visible
     inconsistency. Source data:
     `results/eswa_revision/15_significance/paired_significance.csv` and
     `results/eswa_revision/17_overlap_analysis/*.csv`.
   - Added an "Engineering-Oriented and Solver-Backed Validation" subsection (Sec. 3.5)
     with three new tables and two new figures, replacing vague, number-free prose in
     the original Conclusion ("the pipeline can often produce structurally valid...
     artifacts," with no instance counts or rates given). The original draft's own
     Setup section (Sec. 3.1) already promised these three studies but the results were
     never actually reported anywhere in the text. Source data:
     `results/paper/eaai_camera_ready_tables/table{2,3,4}_*.csv` and
     `results/paper/eaai_camera_ready_figures/figure{3,4}_*.pdf`.
   - Rewrote the Contributions paragraph (Sec. 1.4) from four to six explicit
     contributions to match what the paper (now, with the additions above) actually
     contains: decomposition, deterministic grounding framework, optimization-role
     reasoning, controlled bottleneck comparison, structural/solver-backed validation,
     and error/failure analysis.
   - Updated the Conclusion's engineering-subset paragraph to cite the new tables with
     actual numbers instead of qualitative-only language.
8. **Bibliography cleanup.** Removed 5 unused/duplicate entries from `references.bib`:
   `ahmaditeshnizi2023optimus` (an arXiv preprint of OptiMUS superseded by the
   peer-reviewed ICML version already cited, `ahmaditeshnizi2024optimus_icml`),
   `dakle2023ner4opt` (a conference-proceedings duplicate of the journal version already
   cited, `kadioglu2024ner4opt`), and `reimers2019sentencebert` / `wang2020minilm` /
   `lu2025optmath` (present in the .bib file but never cited anywhere in the manuscript
   text). No new references were added. All 20 remaining entries are cited exactly once
   and every in-text `\cite` resolves to an entry (verified programmatically).
9. **Removed 2 unreferenced, stale figure files** from the submission package
   (`nlp4lp_error_hitmiss_orig.png`, `nlp4lp_downstream_schema_vs_ready_orig.png`,
   `nlp4lp_instantiation_pipeline.png`, the pre-`_v2` pipeline diagram). The two "hitmiss"
   and "schema_vs_ready" images were not referenced anywhere in `main.tex`, and their
   `InstantiationReady` values (~0.06-0.08) match the **pre-bug-fix** numbers, not the
   corrected post-fix numbers (~0.52-0.57) the manuscript now reports -- including them
   would have silently reintroduced a stale, superseded result into the submission.

## Verified consistent (no change needed)

- Headline Table 1 numbers (TF-IDF Schema R@1 = 0.9094, Coverage = 0.8639, TypeMatch =
  0.7513, InstantiationReady = 0.5257; Oracle Coverage = 0.9151, TypeMatch = 0.8030,
  InstantiationReady = 0.5650) match `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`
  and its downstream copy `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`.
- The 335-candidate-catalog vs. 331-test-query distinction is used consistently
  throughout (random baseline = 1/335 ~ 0.0030 for retrieval; all benchmark metrics use
  a 331-query denominator).
- All 11 "new method family" (GCG/RAL/AAG) numbers in the manuscript's tables were
  checked against `results/eswa_revision/16_error_analysis/method_comparison_table.csv`
  and the underlying per-method JSON aggregates; all matched to 4 decimal places.
- No LaTeX structural errors: brace/label/ref/begin-end balance checked programmatically
  (no duplicate labels, no dangling `\ref`s, no unused `\includegraphics` targets).

## KAIS final-preparation pass (2026-07-23, second pass)

A second, deeper preparation pass resolved every open item listed below (previously
numbered 1-3) and made additional scientific and presentation corrections. See
`docs/KAIS_FINAL_PREPARATION_REPORT.md` for the full audit report. Summary of what
changed in this pass:

- **Official Springer Nature template.** `main.tex` now uses the real `sn-jnl.cls`
  (December 2024 version, downloaded directly from Springer Nature's own CMS resource
  host) with `[pdflatex,sn-basic,Numbered]` options, matching KAIS's own citation-style
  guidance exactly. `elsarticle.cls`/`elsarticle-num.bst` are kept only because the
  optional standalone `title.tex` still uses them (see Open Item 3 below); they are no
  longer used by `main.tex`.
- **Peer-review blinding confirmed.** KAIS's ethics-and-disclosures page
  (<https://link.springer.com/journal/10115/ethics-and-disclosures>) states review is
  single-anonymous. Author identity is correctly included in the title block; no
  anonymization is required.
- **`InstantiationReady` definition corrected.** The prose previously described an
  all-or-nothing rule ("schema matches gold AND all slots filled AND all correct
  type"); the actual implementation
  (`tools/nlp4lp_downstream_utility.py`, canonical per-query loop) is a per-query
  `Coverage >= 0.8 AND TypeMatch >= 0.8` threshold rule with no separate hard schema-match
  gate. Fixed in Sec. 3.1 with a proper equation.
- **Three new verified references added** to Related Work (Chain-of-Experts, ICLR 2024;
  LLMOPT, ICLR 2025; a 2025 IJCAI survey on optimization modeling + LLMs), each checked
  against official proceedings/OpenReview pages before adding.
- **Table formatting standardized**: all 14 tables converted to Springer's
  `\toprule`/`\midrule`/`\botrule` style; removed an unjustified `\resizebox`; fixed two
  inconsistently-`\scriptsize` simple tables to `\small`; fixed a real ~91pt tabularx
  overfull-hbox bug.
- **Two figures regenerated** (`tools/build_eaai_camera_ready_figures.py`) to remove a
  baked-in title that duplicated and numerically mismatched the LaTeX caption.
- **New "Limitations and Threats to Validity" subsection** added to the Conclusion
  (previously promised in the intro's roadmap sentence but never delivered as a
  consolidated section).
- Minor fixes: a lexical-overlap bucket mislabeled "low" instead of "medium"
  (data has no "low" bucket); two Related Work sentences with >4 citations restructured
  so each citation is attributed to a specific claim.
- Full clean compile verified: 3x `pdflatex` + `bibtex`, 36 pages, all fonts embedded,
  zero undefined citations/references, zero overfull hboxes.

## Open items requiring your manual action

1. **Enter declarations via the KAIS submission portal** as well as in the manuscript
   text -- Springer's guidelines state Author Contribution and Competing Interest
   information must be provided through the submission interface, and only that
   interface version is used in the final published record.
2. **Decide on the standalone `title.tex` / `highlights.tex` / `declaration-of-interest.tex`
   files.** None are required for KAIS (all their content is already folded into
   `main.tex`); include them only if the submission system specifically asks for a
   separate title page.
3. **Double-check the corresponding-author postal address.** The `sn-jnl` title-block
   template requires a `\street{}` field that the original `elsarticle` `\address` did
   not have; it is now filled in as "University Heights" (NJIT's official general
   mailing-address designation, verified against njit.edu's own contact/maps pages
   during this pass, zip 07102-1982). Replace with your department's specific street
   address/building if you prefer a more specific one on the title page.

## File manifest

| File | Role |
|------|------|
| `main.tex` | Full manuscript (title, abstract, keywords, body, Declarations, bibliography command) -- **sn-jnl.cls-based** |
| `main.pdf` | Compiled output of the current `main.tex` (36 pages) |
| `references.bib` | Bibliography (23 entries, all cited) |
| `sn-jnl.cls`, `sn-basic.bst` | Official Springer Nature template files (see header comment in `main.tex` for provenance) |
| `elsarticle.cls`, `elsarticle-num.bst` | Retained only for the optional standalone `title.tex`; not used by `main.tex` |
| `figures/nlp4lp_instantiation_pipeline_v2.png` | Figure 1 (pipeline overview) |
| `figures/figure3_engineering_validation_comparison.pdf` | Figure 2 (engineering subset; regenerated to drop a duplicate baked-in title) |
| `figures/figure4_final_solver_backed_subset.pdf` | Figure 3 (solver-backed subset; regenerated to drop a duplicate baked-in title) |
| `cover-letter.tex` | KAIS-targeted cover letter |
| `title.tex`, `highlights.tex`, `declaration-of-interest.tex` | Optional supplementary files (see Open Item 2) |
