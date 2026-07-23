# KAIS Final Preparation Report

**Manuscript:** "Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"
**Target journal:** *Knowledge and Information Systems* (KAIS), Springer Nature
**Prepared:** 2026-07-23
**Branch:** `kais-final-submission-prep`

---

## 1. Final Readiness Status

**READY WITH MINOR MANUAL CHECKS**

The manuscript is scientifically consistent with the repository, uses the official
Springer Nature LaTeX template, compiles cleanly to a 38-page PDF, and has no known
factual, citation, or metric-definition errors remaining. The items that still require
a human decision are administrative/portal actions (see Section 14), not scientific or
formatting defects.

## 2. Batch Execution Status

This pass ran as a single long working session with continuous, incremental git
checkpoints (one commit per completed phase) so that no work would be lost if the
session were interrupted. All infrastructure needed (internet access for research and
literature verification, a full TeX Live 2025 toolchain via the `texlive/20250308`
environment module) was available directly in this compute environment, so no separate
remote/cloud sandbox was required.

- **Branch:** `kais-final-submission-prep`
- **Final commit hash:** run `git log -1 --format=%H` on this branch after this report
  is committed; as of the last commit before this report, the hash is `654394b`
  (`feat(manuscript): add Limitations section, fix fabricated address, trim citation
  density`).
- **Pushed:** yes, after every major phase (baseline checkpoint, metric-definition fix,
  literature additions, template conversion, limitations/address-fix pass). `git push`
  succeeded each time with no conflicts.
- **Resumability:** each commit is self-contained and independently useful; if this
  session had been interrupted at any point, the branch would have reflected a strictly
  better state than the previous commit, never a half-edited one.

## 3. Official KAIS Requirements Verified

All verified directly from live Springer Nature pages during this pass (not from
memory or third-party summaries), by fetching the pages with cookies/session handling
where needed:

| Requirement | Value | Source |
|---|---|---|
| Peer review model | Single-anonymous | `link.springer.com/journal/10115/ethics-and-disclosures` |
| LaTeX template | Springer Nature template encouraged; `sn-basic.bst` explicitly named | `link.springer.com/journal/10115/submission-guidelines` |
| Citation style | Numbered brackets, e.g. `[1]`, `[1-3,7]` | same, "References" section |
| Abstract length | 150-250 words | same, "Abstract" section |
| Keywords | 4-6 | same, "Keywords" section |
| Headings | Decimal system, max 3 levels | same, "Headings" section |
| Declarations | Funding, Competing interests, Ethics, Data/Code availability, Author contributions, in a "Statements and Declarations" section before references | same |
| Generative-AI disclosure | LLMs not authors; must be documented in Methods (or equivalent); "AI-assisted copy editing" need not be declared | same, "Title Page" section |
| Source files | Must submit editable LaTeX source + all style files + a compiled PDF | same, "Source Files" section |
| Page/word limit for regular research articles | None found (Short Papers are capped at 3000 words, "visions and directions" pieces at 1000 words, but this is a regular research article) | `aims-and-scope` page |
| Journal scope | Knowledge/information processing, intelligent information retrieval, knowledge-data engineering, decision support, etc. | `aims-and-scope` page |

No requirement was assumed without checking; the peer-review blinding policy in
particular had been an explicit open item in the prior pass and is now resolved with a
direct citation.

## 4. Template Status

**Official Springer Nature LaTeX template, December 2024 version**, `sn-jnl.cls`,
downloaded directly from Springer Nature's CMS resource host (the URL Springer's own
"Download the journal article template package" button on
`springernature.com/gp/authors/campaigns/latex-author-support/...` resolves to). Class
options in use: `[pdflatex,sn-basic,Numbered]` -- the "Basic Springer Nature Reference
Style" with numbered bracketed citations and `sn-basic.bst`, matching KAIS's citation
guidance verbatim. Both `sn-jnl.cls` and `sn-basic.bst` are committed in
`manuscript/` (and in `manuscript/submission_package/`) for provenance and so the
submission is self-contained per Springer's own "include all style files" instruction.

## 5. Files Changed

**New:**
- `manuscript/sn-jnl.cls`, `manuscript/sn-basic.bst` (official template files)
- `manuscript/main.pdf` (compiled output)
- `manuscript/submission_package/` (clean, minimal, independently-verified-compilable submission directory: `main.tex`, `main.pdf`, `references.bib`, `sn-jnl.cls`, `sn-basic.bst`, `cover-letter.tex`, `figures/`)
- `manuscript/KAIS_FINAL_PREPARATION_REPORT.md` (this file)

**Modified:**
- `manuscript/main.tex` (template conversion; metric-definition fix; 3 new citations;
  table/figure formatting fixes; new Limitations subsection; address correction;
  citation-density fixes; dataset-description clarification)
- `manuscript/references.bib` (3 new entries)
- `manuscript/MANUSCRIPT_README.md` (updated to record this pass; open items resolved)
- `tools/build_eaai_camera_ready_figures.py` (removed baked-in figure titles)
- `results/paper/eaai_camera_ready_figures/figure3_*.{png,pdf}`,
  `figure4_*.{png,pdf}` (regenerated without the duplicated/mismatched title)
- `manuscript/figures/figure3_*.pdf`, `figure4_*.pdf` (copies of the regenerated figures)
- `docs/KAIS_SOURCE_OF_TRUTH.md` (pointer to this report; open items marked resolved)
- `.gitignore` (LaTeX build-artifact patterns for `manuscript/`)

**Deleted:** none from the repository (the interim `elsarticle.cls`/`elsarticle-num.bst`
are kept because the optional standalone `title.tex` still depends on them; they are
simply no longer used by `main.tex`).

## 6. Scientific Changes

1. **`InstantiationReady` metric definition corrected** to match the actual
   implementation (`Coverage >= 0.8 AND TypeMatch >= 0.8` per query, no separate hard
   schema-match gate), replacing an incorrect all-or-nothing prose description. This was
   verified by reading the canonical per-query evaluation loop in
   `tools/nlp4lp_downstream_utility.py` (the code that produced the numbers in
   `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`), not
   inferred from prose or memory. A proper numbered equation was added.
2. **Lexical-overlap bucket mislabel fixed**: the manuscript called a 4-query bucket
   "low-overlap"; the underlying data (`overlap_stratified_retrieval.csv`) labels it
   "medium" (there is no "low" bucket). Its TF-IDF Schema_R@1 (0.0000) was also added
   since it is informative and was previously omitted.
3. **New "Limitations and Threats to Validity" subsection** added, consolidating caveats
   that were previously scattered and, in one case (a promised "limitations" discussion
   in the conclusion), never actually delivered as a discrete section.
4. **NLP4LP dataset description clarified** with one sentence on its actual structure
   (curated collection with gold parameter records, reference solver code, and
   reference solutions per instance), without overstating validation on the dataset's
   external construction methodology (which was not independently re-audited beyond
   what is publicly documented on the dataset's own pages).

No numerical result, table value, or table caption was changed based on this audit --
every headline number was independently spot-checked against its source CSV/JSON in
`results/` and matched.

## 7. Repository Consistency Corrections

| Claim | Manuscript said | Repository evidence | Resolution |
|---|---|---|---|
| `InstantiationReady` definition | "predicted schema matches gold AND all eligible slots filled AND all correct type" | `tools/nlp4lp_downstream_utility.py`: `1 if (param_coverage >= 0.8 and type_match >= 0.8) else 0` | Manuscript prose fixed to match code |
| Overlap-analysis small-bucket label | "low-overlap bucket" | `overlap_stratified_retrieval.csv` has buckets `{medium, high}` only | Manuscript fixed to "medium-overlap" |
| Corresponding-author street address | (none in original; template requires one) | N/A -- this was introduced during template conversion and initially filled with a guessed address | Caught during self-review; corrected to NJIT's verified official address ("University Heights, Newark, NJ 07102-1982") |

All other spot-checked claims (335-candidate catalog size, 331-query test split, Table 1
headline numbers, the 11 new GCG/RAL/AAG method numbers, the significance-test table,
the overlap-ablation table, and all three engineering/executable/solver-backed subset
tables) matched their source files in `results/` to the reported decimal precision; no
discrepancy was found beyond the three rows above.

## 8. Literature Changes

**Added** (all verified against official proceedings/OpenReview pages before adding):

- Xiao et al., "Chain-of-Experts: When LLMs Meet Complex Operations Research Problems,"
  ICLR 2024 -- a multi-agent LLM framework for OR problem solving; added to Related
  Work alongside OptiMUS/OptLLM as another representative end-to-end system.
- Jiang et al., "LLMOPT: Learning to Define and Solve General Optimization Problems
  from Scratch," ICLR 2025 -- a fine-tuned (rather than purely prompting-based)
  end-to-end system; added as a complementary representative.
- Xiao et al., "A Survey of Optimization Modeling Meets LLMs: Progress and Future
  Directions," IJCAI 2025 -- added to the existing survey-citation sentence.

**Considered but not added:** CAFA ("Coding as Auto-Formulation," NeurIPS 2024 MATH-AI
*workshop* paper) -- a real paper, but workshop-tier and a thinner evaluation than the
three additions above; noted here rather than added to avoid diluting the reference
list with a lower-confidence citation.

**Removed/corrected:** none in this pass (the prior pass had already removed 5
unused/duplicate entries from `references.bib`; this pass only added).

## 9. Baseline Assessment

Important baselines remain missing, and this is now explicitly documented in the new
Limitations subsection rather than silently left unaddressed:

- **No dense/neural retrieval baseline** (e.g., a sentence-embedding or dual-encoder
  retriever) alongside the three lexical baselines (BM25, TF-IDF, LSA). This is a
  reasonable reviewer question. Running one was not attempted in this pass because it
  is a genuinely new experiment (model selection, embedding computation, evaluation
  harness integration) rather than a formatting/consistency fix, and the task
  instructions for this pass were explicit that new expensive experiments should be
  flagged, not fabricated or launched without separate authorization.
- **No head-to-head experimental comparison against end-to-end LLM systems** (OptiMUS,
  OptLLM, Chain-of-Experts, LLMOPT, OPT2CODE) on NLP4LP. These are discussed
  qualitatively and correctly distinguished by task scope in Related Work, but no
  quantitative comparison is run, primarily because they target a broader task (full
  formulation + solver-code generation) and several require paid API access. This is
  now explicit in the Limitations subsection.

Both gaps are honestly scoped as limitations rather than hidden; the paper does not
claim to be compared against these systems.

## 10. Experimental-Rigor Assessment

The benchmark is fully deterministic (rule-based retrieval + rule-based grounding), so
the paper correctly does *not* report meaningless standard deviations across repeated
stochastic runs. Statistical reliability is instead established via:

- Paired bootstrap significance tests (B = 1,000 resamples) for the key retrieval and
  downstream comparisons, with 95% confidence intervals and two-sided p-values
  (Table 10 in the compiled PDF).
- An overlap-based stress test (numeric-token and stopword removal) to test whether
  strong retrieval reflects benchmark-specific lexical overlap rather than genuine
  schema-level signal (Table 11).
- A measured pre-fix/post-fix ablation isolating the effect of a specific
  numeric-type-handling correction (Table 8).
- A diagnostic error taxonomy (Table 7) and a negative-result comparison against three
  additional deterministic grounding families (Table 9), which strengthens rather than
  merely illustrates the paper's central bottleneck claim.

This is an appropriate and honestly-described statistical treatment for a deterministic
pipeline; no changes were needed here beyond the metric-definition fix in Section 6.

## 11. Visual Audit

- **Table consistency:** all 14 tables now use Springer's `\toprule`/`\midrule`/`\botrule`
  booktabs style (previously a mix of plain `\hline`). Font size now follows a
  principled two-tier rule (`\small` for tables with <=5 simple columns, `\scriptsize`
  only for genuinely wide/dense tables with 6+ columns or long text cells) instead of
  arbitrary per-table variation -- this was the specific defect flagged from the prior
  Digital Engineering submission and has been directly addressed.
- **Figure consistency:** the two bar-chart figures (engineering subset, solver-backed
  subset) were regenerated to remove a baked-in PIL-drawn title that duplicated *and*
  numerically mismatched the LaTeX `\caption` numbering (embedded "Figure 3"/"Figure 4"
  vs. actual "Fig. 2"/"Fig. 3"). The pipeline-overview figure (Fig. 1) was inspected at
  300 dpi; it is legible and not clipped, though its "Query branch"/"Schema branch"
  side labels are visually cramped -- a minor cosmetic issue, not a blocking one (see
  Section 19 prompt below for a possible follow-up).
- **Font consistency:** `pdffonts` confirms all ~30 embedded font subsets are Type 1,
  fully embedded (no `emb=no` entries).
- **Page layout:** A4, single-column Springer layout. No content overflows page
  margins (zero `Overfull \hbox` warnings in the final compile log).
- **Final page count:** 38 pages.

## 12. Remaining Risks

- **Novelty**: the core contribution is an empirical decomposition and diagnostic study
  rather than a new algorithm; a reviewer could characterize the method as "TF-IDF plus
  deterministic rules." The oracle-vs-retrieval bottleneck finding and the negative
  result for three additional richer method families are the paper's strongest defenses
  against this framing, and both are now clearly and prominently stated.
- **Generalization**: single benchmark, gated access, untested on other domains/styles
  of optimization text (now explicit in Limitations).
- **Heuristic grounding**: the type/role inference layer has no formal correctness
  guarantee; this is stated plainly in both the Methodology and the new Limitations
  subsection.
- **Dataset dependence**: NLP4LP-only; Text2Zinc/CP-Bench are discussed as related
  benchmarks but not used for external validation.
- **Missing baselines**: no dense-retrieval baseline; no head-to-head LLM-system
  comparison (Section 9).
- **Solver coverage**: SciPy HiGHS only, small (20-instance) solver-backed subset.

None of these risks were hidden or downplayed; all are now explicit in the manuscript
text.

## 13. Reviewer-Style Score

| Category | Assessment |
|---|---|
| A. Novelty and contribution | Moderate. Real, well-supported empirical finding (retrieval strong / grounding is the bottleneck); not a new algorithm. Minor concern. |
| B. Baselines | Adequate for the retrieval-vs-grounding decomposition studied; missing a dense-retrieval baseline and any head-to-head LLM-system comparison. Documented as a limitation. Minor-to-moderate concern. |
| C. Related work | Strong after this pass; correctly distinguishes task framings, explains what each prior system does. |
| D. Datasets/workloads | Single gated benchmark; accurately described; external validity appropriately hedged. Minor concern. |
| E. Experimental rigor | Strong for a deterministic pipeline: significance tests, overlap stress tests, ablation, negative results. |
| F. Reproducibility | Good; code public; gated-dataset requirement clearly disclosed; committed artifacts usable without dataset access. |
| G. Technical correctness | Now verified against implementation (major fix applied to `InstantiationReady`); no remaining known inconsistencies. |
| H. Writing and presentation | Strong; consistent terminology; honest, non-inflated claims throughout. |
| I. Formatting and journal style | Compliant with verified KAIS/Springer requirements; official template; clean compile. |
| J. Limitations | Now explicit and thorough (new dedicated subsection). |
| K. Practical impact | Modest, honestly scoped as an intermediate decision-support component, not a deployed system. |
| L. Missing references/baselines | Three genuine additions made; two baseline gaps remain and are disclosed. |

**Overall score: 79/100.**

**Revision severity: minor revision.** No fatal scientific, ethical, or reproducibility
flaws remain. The most likely reviewer requests are (i) a dense-retrieval baseline or
(ii) a discussion of why end-to-end LLM systems were not run head-to-head -- both are
already anticipated and addressed in the Limitations subsection, which should reduce
(though not eliminate) the chance of a major-revision verdict on these points.

This assessment is recorded here only; it is **not** inserted into the manuscript
itself.

## 14. Submission Checklist (manual steps remaining)

1. Submit via KAIS's online submission system (Editorial Manager, per Springer's
   standard workflow) at the "Submit your manuscript" link on the journal's site.
2. Re-enter Funding, Competing Interests, and Author Contribution information into the
   submission portal's own interface fields -- Springer's guidelines state only the
   portal-entered version is used in the final published record, even though the same
   statements already appear in the manuscript's Declarations section.
3. Suggest reviewers if desired (optional per KAIS's guidelines).
4. Decide whether to attach `title.tex` / `highlights.tex` / `declaration-of-interest.tex`
   as supplementary files -- none are required; only attach if the portal specifically
   requests a separate title page.
5. Verify/adjust the corresponding-author postal address on the title page
   ("University Heights, Newark, NJ 07102-1982" -- NJIT's official general address; a
   more specific department/building address can be substituted if preferred).
6. Do **not** submit until the author has done a final personal read-through; this pass
   was thorough but is not a substitute for the author's own final check.

## 15. Final Artifact Paths

- **Final manuscript PDF:** `manuscript/main.pdf` (38 pages) and
  `manuscript/submission_package/main.pdf` (identical content, verified independently
  compiled from the standalone package)
- **Final LaTeX source:** `manuscript/main.tex`
- **Bibliography:** `manuscript/references.bib` (23 entries)
- **Template files:** `manuscript/sn-jnl.cls`, `manuscript/sn-basic.bst`
- **Cover letter:** `manuscript/cover-letter.tex`
- **Submission-package directory:** `manuscript/submission_package/` (self-contained;
  independently verified to compile cleanly with `pdflatex` + `bibtex` + `pdflatex` x2
  from a fresh checkout of just that directory)
- **This report:** `manuscript/KAIS_FINAL_PREPARATION_REPORT.md`
