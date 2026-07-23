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
- **Final commit hash:** run `git log -1 --format=%H` on this branch for the exact
  value; as of the commit immediately before this report's own final update, the hash
  is `a2f5940` (`fix(manuscript): number the eligible-slot-set, Coverage, and
  TypeMatch equations`). This report itself is committed after that, so the true final
  hash is one commit later -- check `git log --oneline -5` on `kais-final-submission-prep`.
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

### 11.1 Optional follow-up: Figure 1 relabeling (Cursor prompt)

Figure 1 (the pipeline-overview diagram, `figures/nlp4lp_instantiation_pipeline_v2.png`)
is legible and not clipped at 300 dpi, but its two small italic side labels
("Query branch" / "Schema branch") are visually cramped against the first box in each
row. This is cosmetic, not a compilation or content defect, and was left as-is in this
pass. If a cleaner version is wanted, the following prompt can be given to Cursor
(the figure is generated by a small Python/PIL script, not drawn by hand):

> The pipeline-overview figure used in the manuscript
> (`manuscript/figures/nlp4lp_instantiation_pipeline_v2.png`) is generated by
> `figures/gen_instantiation_pipeline.py` (a matplotlib script; see its "Branch labels"
> section, which places the "Query branch" / "Schema branch" captions "to the left of
> each first box"). At high zoom these two small italic labels sit flush against the
> border of the first box in each row and look cramped. Increase the horizontal gap
> between these branch-label text elements and the first box in each row (e.g., increase
> the x-offset used when placing the "Query branch"/"Schema branch" text, or increase
> `FS_LABEL` slightly), regenerate the figure at the same 300 dpi / aspect ratio, and
> overwrite `manuscript/figures/nlp4lp_instantiation_pipeline_v2.png` (and the
> `results/`-side copy if one exists) with the regenerated PNG. Do not change the
> diagram's content, box labels, arrows, or overall flow structure -- only the spacing
> of the two branch-label captions.

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

## 16. Targeted Follow-Up Update (2026-07-23, second session)

A subsequent, narrowly-scoped request asked for four specific changes only (no
re-audit of scientific content, tables, figures, metrics, or references). This section
documents what was done and confirms each requested item.

1. **Acknowledgements updated.** `main.tex`'s `\bmhead{Acknowledgements}` text now
   reads: "The author is deeply grateful to his mother for her continuous emotional
   support. The author also thanks his PhD advisor, Professor Ioannis Koutis, for his
   support, guidance, and encouragement." Confirmed by extracting text directly from
   the compiled PDF (`pdftotext`) -- the surname is spelled **Koutis** (not "Kotis"),
   and both the mother's emotional support and the PhD advisor's guidance are
   acknowledged, matching the requested wording exactly.
2. **Public GitHub URL confirmed present** in the manuscript: `main.tex` cites
   `https://github.com/SoroushVahidi/combinatorial-opt-agent` in both the **Data
   availability** statement (for committed camera-ready tables/figures/analysis
   reports) and the **Code availability** statement (for the retrieval/grounding/
   evaluation code), consistent with the repository's data being derived from a gated
   third-party dataset (NLP4LP) that is not itself redistributed.
3. **Repository public accessibility independently verified** (not assumed from the
   URL string) via three separate unauthenticated checks run from this environment:
   - `curl` to `https://github.com/SoroushVahidi/combinatorial-opt-agent` returned
     HTTP 200 with no authentication.
   - `git ls-remote` against the HTTPS remote succeeded anonymously
     (`GIT_TERMINAL_PROMPT=0`, no credential prompt, no stored token used), returning
     the current `HEAD` commit.
   - The GitHub REST API (`api.github.com/repos/SoroushVahidi/combinatorial-opt-agent`,
     unauthenticated) reports `"private": false` and `"visibility": "public"`.
4. **Recompiled after the change**: `pdflatex` x3 + `bibtex`, zero overfull hboxes,
   zero undefined citations or references, zero new LaTeX errors or warnings beyond
   the pre-existing ~29 minor underfull-hbox cosmetic warnings already noted in
   Section 11. Final PDF: 38 pages (unchanged page count). Verified visually by
   extracting and rendering the Acknowledgements page from the freshly compiled PDF.
5. No scientific results, tables, figures, metric definitions, or references were
   touched in this follow-up; the only content change in `main.tex` is the
   Acknowledgements paragraph.

## 17. Targeted Follow-Up: Figure Legend Fix (2026-07-23, third session)

A subsequent request asked specifically to move Fig. 2/3's legends outside the plot
area (Fig. 3's legend was overlapping a bar-value annotation) and to inspect Fig. 1 for
similar issues. Summary: replaced the PIL-based bar-chart renderer with a matplotlib
one for the two figures used in the manuscript (legend centered below the axes,
enlarged fonts, true vector PDF output instead of a rasterized PNG wrapped in a PDF);
fixed a cramped-label spacing issue in Fig. 1's generator
(`figures/gen_instantiation_pipeline.py`). Verified via rendered page images that the
legend no longer overlaps plot content and the manuscript recompiles clean. Commit
`1485f10`. See that commit's message for full detail; not repeated here.

## 18. FINAL ACCEPTANCE-ORIENTED CONSISTENCY PASS (2026-07-23, fourth session)

This section documents a deep scientific-consistency pass focused on internal
contradictions, statistical-claim accuracy, and a proactive reviewer-objection check,
per an explicit 20-part audit request. A claim-evidence matrix was built internally
(covering retrieval performance, Oracle behavior, all four core metrics, significance
tests, overlap robustness, the pre/post-fix ablation, the three engineering subsets,
reproducibility, and public-artifact availability) and used to drive every fix below;
it is not reproduced in full here, only its actionable findings.

### 18.1 New HEAD commit SHA

`ac9e84416b1a3c9013663de90e086f7be1df6217` (branch `kais-final-submission-prep`).

### 18.2 Exact files changed

`manuscript/main.tex`, `manuscript/cover-letter.tex`, `manuscript/main.pdf`, their
mirrors in `manuscript/submission_package/`, `tools/build_eaai_camera_ready_figures.py`
(font embedding), `tools/run_overlap_analysis.py` (LSA dimensionality fix), new
`tools/run_strict_instantiation_ready.py`, new
`results/eswa_revision/18_strict_instready/{strict_instantiation_ready.csv,
strict_vs_standard_significance.csv}`, regenerated
`results/eswa_revision/17_overlap_analysis/*` and
`results/paper/eaai_camera_ready_figures/figure{3,4}_*`, and a dated correction note in
`results/eswa_revision/14_reports/FINAL_REVISION_EXPERIMENT_SUMMARY.md`.

### 18.3 Known inconsistencies fixed

Two internal contradictions specifically flagged were found to be real and fixed
precisely as described (see 18.4 and the Oracle-narrative correction in Sec. 3.6
summary/conclusion: Oracle described as inconsistent across the three engineering
studies rather than uniformly "small but positive"). A third, previously undiscovered
inconsistency (the Table 3/Table 11 LSA gap) was root-caused and corrected (18.6).

### 18.4 Table 10 statistical correction

Prose previously claimed TF-IDF-TG "significantly outperforms representative
global-compatibility, relation-aware, and ambiguity-aware variants" -- contradicted by
Table 10's own RAL-Basic row ($p=0.218$). Corrected: TF-IDF-TG is statistically
indistinguishable from relation-aware (as well as acceptance-rerank and
hierarchical-acceptance-rerank), and significantly outperforms only
global-compatibility and ambiguity-aware ($p<0.001$ each). The relation-aware family's
lower point estimate is now described as directionally consistent but not itself
statistically confirmed, rather than folded into a blanket "significant" claim.

### 18.5 Oracle narrative corrections

Both the Sec. 3.6 summary paragraph and the conclusion's engineering-studies paragraph
previously implied a consistent (if modest) Oracle advantage across all three
restricted studies. The 20-instance solver-backed subset actually shows Oracle
*below* TF-IDF on solver-success/feasible/objective-produced (0.75 vs. 0.80 on each).
Both passages were rewritten to state this explicitly, attribute it plausibly to the
subset's small size and compatibility-filtering rule, and reframe the conclusion as
"Oracle does not consistently provide a substantial advantage" rather than a uniform
small-improvement narrative. Every use of "oracle upper bound" (abstract, contributions
paragraph, Sec. 2.4's methodological definition, Sec. 3.3, and the cover letter -- 5
occurrences total) was replaced with "oracle control" / "oracle reference" / "oracle
condition," since the solver-backed-subset counterexample shows Oracle is not a
formally guaranteed upper bound on every reported metric. A one-sentence formal caveat
was added at the oracle condition's first definition (Sec. 2.4) explaining why.

### 18.6 Table 3/Table 11 discrepancy: root cause, fix, final values

**Root cause:** `tools/run_overlap_analysis.py`'s LSA retriever capped
`TruncatedSVD(n_components=...)` at 100, while the canonical retriever
(`retrieval/baselines.py::LSABaseline`) defaults to 256. Schema-text construction,
corpus (`data/catalogs/nlp4lp_catalog.jsonl`, 335 candidates), query source, and the
TF-IDF vectorizer's `ngram_range=(1,2)` were already identical between the two scripts
-- confirmed by direct code comparison before making any change, per the task's
explicit "establish provenance before overwriting" instruction.

**Fix:** changed `n_comp = min(100, ...)` to `n_comp = min(256, ...)` (one line),
matching the canonical default exactly, with a code comment explaining why.

**Final values:** rerunning the corrected script reproduces the canonical Table 3 LSA
value **exactly** for the unsanitized baseline row: $0.8459$ (previously $0.7734$, an
artifact of the truncated latent space, not a property of LSA retrieval). The
stopword-removed LSA value is now $0.9184$ (previously $0.8731$) -- notably *above*
canonical TF-IDF. TF-IDF and BM25's small pre-existing offset from Table 3 ($\le
0.003$, schema-text-construction differences) is unchanged and was already honestly
disclosed; it was not chased further, consistent with the task's instruction not to
over-engineer a already-transparent minor discrepancy. A secondary, unrelated
single-query environment-sensitivity fluctuation was also caught by the same rerun (the
high-overlap-bucket TF-IDF rate shifted from $0.9205$ to $0.9174$, one query's
`schema_hit` flag flipping under the current `sklearn` version) and is now reported
with the fresher, currently-reproducible value rather than a stale one.

### 18.7 StrictInstantiationReady sensitivity results

New Eq. (7) and Table 12 (Sec. 3.5). Computed from `results/eswa_revision/
02_downstream_postfix/` per-query artifacts via the new
`tools/run_strict_instantiation_ready.py`, using the same paired-bootstrap methodology
as the canonical significance tests ($B=1{,}000$, seed$=42$).

| Method | InstReady | StrictInstReady | $\Delta$ | $n_{\text{differ}}$ |
|---|---|---|---|---|
| TFIDF-TG | 0.5287 | 0.5045 | 0.0242 | 8 |
| BM25-TG | 0.5196 | 0.4924 | 0.0272 | 9 |
| LSA-TG | 0.5076 | 0.4864 | 0.0211 | 7 |
| Oracle-TG | 0.5680 | 0.5680 | 0.0000 | 0 |

TF-IDF-vs-Oracle gap: $0.0393$ (standard) $\to$ $0.0634$ (strict); paired bootstrap
$p=0.004 \to p<0.001$. **The central conclusion is unchanged -- indeed strengthened**:
adding a hard schema-match gate does not shrink the Oracle advantage, it widens it and
makes it more significant, directly refuting the possible objection that the modest
oracle gain is a metric-definition artifact. As a validity check, recomputing the
*standard* (non-strict) metric from these same live files and paired-bootstrapping
TFIDF-TG vs. Oracle-TG reproduces the canonical Table 10 result exactly
($\Delta=-0.0393$, CI $[-0.0665,-0.0151]$, $p=0.004$), confirming the small absolute
offset from frozen Table 4 values (same cause as 18.6's TF-IDF/BM25 offset) cancels out
of paired differences and does not affect this conclusion.

### 18.8 "Oracle upper bound" audit outcome

All 5 occurrences reviewed and corrected (18.5). No remaining uses of "upper bound" in
an oracle context; the two remaining unrelated uses ("lower versus upper bounds" in
optimization-constraint semantics, error taxonomy) are correct as-is.

### 18.9 Table 13 decision

**Retained as a compact main-text table**, unmodified. Its own caption already states
"reported as a blocker study, not a success-rate study," which independently satisfies
the requirement to make an environment-dependency failure's purpose unmistakable
without presenting it as a methodological result. Moving it to supplementary material
was considered and rejected: the table is small (3 rows), directly motivates the
20-instance solver-backed subset that follows it, and its removal from the main text
would weaken (not strengthen) the paper's transparency about why that smaller subset
was necessary.

### 18.10 Figure font-embedding status

Fixed. Added `matplotlib.rcParams["pdf.fonttype"] = 42` / `ps.fonttype = 42` to the
figure generator; `pdffonts` now reports the embedded font as `CID TrueType` (Type 42)
for both Fig. 2 and Fig. 3, replacing the previous Type 3 embedding. No visual
regression (verified by rendering both figures before/after at 150 dpi).

### 18.11 KAIS acknowledgment placement status

Unchanged and still compliant: `\bmhead{Acknowledgements}` inside `\backmatter`,
immediately before `\section*{Declarations}`, matching the official Springer Nature
template's own structural convention (Sec. 13/16 of this report, prior sessions).
Acknowledgement wording preserved exactly, including the correct spelling "Ioannis
Koutis".

### 18.12 Generative-AI disclosure placement/status

Unchanged and still compliant with the policy verified directly from KAIS's title-page
guidelines in an earlier session (LLMs do not satisfy authorship criteria; substantive
AI use must be documented; "AI-assisted copy editing" need not be declared). The
existing Declarations paragraph accurately describes both AI-assisted writing and
AI-assisted coding and states the author's verification and responsibility. No changes
were needed or made in this pass.

### 18.13 Numerical consistency audit result

All previously-verified headline numbers (Table 1 core values, the 60/269/20-instance
subset tables, the significance table, catalog/query denominators) were re-confirmed
stable and were not touched. Three genuine issues were found and fixed: the Table 10
significance mischaracterization (18.4), the Oracle-narrative overgeneralization
(18.5), and the Table 3/Table 11 LSA discrepancy (18.6). No other numerical
contradictions were found across the abstract, contributions, main text, table values,
figure labels, captions, or conclusion in this pass.

### 18.14 Clean compilation status

Full clean rebuild (`pdflatex` x3 + `bibtex`) succeeded independently in both
`manuscript/` and `manuscript/submission_package/`: zero overfull hboxes (one was
introduced by the new Eq. (7) display and fixed by tightening `\wedge` spacing before
the final compile), zero undefined citations or references, bibliography resolves
correctly, all figures present, all fonts embedded (Type 1 for text via the Springer
template, Type 42/CID TrueType for the two regenerated vector figures).

### 18.15 Final page count

**39 pages** (up from 38, due to the new StrictInstantiationReady equation, table, and
discussion paragraph in Sec. 3.5).

### 18.16 Final reviewer-style score

**82/100** (revised up from the prior pass's 79/100). Rationale: the statistical and
Oracle-narrative corrections remove concrete, reviewer-detectable internal
contradictions that a careful KAIS reviewer would very plausibly have caught and used
to question the paper's rigor; the new StrictInstantiationReady check proactively and
convincingly answers the single most likely "is this a metric artifact?" objection with
a result that *strengthens* rather than merely defends the central claim. Revision
severity remains **minor revision**: no fatal flaws, and the paper is now free of the
specific contradictions most likely to draw a skeptical referee's attention.

### 18.17 Remaining acceptance risks

Unchanged from Section 12 of this report (novelty framing, single-benchmark
dependence, heuristic grounding, no dense-retrieval baseline, no head-to-head LLM-system
comparison, small solver-backed subset) -- all already disclosed in the Limitations
subsection. No new risks were introduced by this pass; if anything, risk C ("the Oracle
improvement is small because of the metric definition") is now substantially mitigated
by Section 18.7's sensitivity result.

### 18.18 Exact final PDF path

`manuscript/main.pdf` (39 pages, 632,182 bytes).

### 18.19 Exact submission-package path

`manuscript/submission_package/` (byte-identical `main.pdf` independently verified to
compile from a fresh checkout of just that directory).

### 18.20 Final readiness status

**READY WITH MINOR MANUAL CHECKS** -- unchanged category from Section 1, but on
materially stronger footing: the manuscript is now free of the specific statistical and
narrative contradictions a skeptical reviewer would most likely flag, and the central
finding has been stress-tested against its most probable metric-definition objection
and survived. Remaining items are the same administrative/portal actions listed in
Section 14 (nothing new).
