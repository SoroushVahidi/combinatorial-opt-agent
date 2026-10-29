# Phase 1 verification report — 2026-10-29

## Authoritative source and isolation

- Starting branch: `main`, commit `a3f0dde397341a41842a30ddb946e7966706351b`.
- Published Stage 1 derives from `manuscript/jdsa-abstract-introduction-stage1-20261029` at `67fb2b51700429e0338d2ac63eeebd321095846e`.
- Revision branch: `manuscript/jdsa-phase1-evidence-results-20261029`; isolated sibling worktree `combinatorial-opt-agent-jdsa-phase1`.
- Original complete manuscript: `manuscript/jdsa-stage1/main.tex`; original descriptive PDF has 32 pages and SHA256 `c58a8cacac78e1d969dc11e5fd19581f49bc1905cf8cceb91d486aaebca5b0ee`.
- New package: `manuscript/jdsa-phase1/`. Existing Stage 1 files and unrelated repository content remain untouched. All 141 pre-existing untracked files recorded in the main and Stage 1 worktrees retain their original hashes.

## Evidence and numerical checks

The register covers **47 investigation groups**, not 47 independent completed experiments. Each has a status, population, conditions, available results, evidence locators, reconstruction boundary and publication destination. All 65 source-manifest hashes were checked against the inspected files. Local-only evidence is explicitly distinguished from files authenticated by a source commit.

`python docs/jdsa_revision_phase1_20261029/verify_saved_metrics.py` performs only standard-library arithmetic on saved records. It checks ten core scalar arms, twelve retrieval cells, 72 historical metric files and four optional LLM files. It confirms:

- TF–IDF/BGE retrieval: **301/331 and 313/331**.
- Ratio-aware TF–IDF: **265/331** coarse readiness, **255/331** schema-gated readiness, **7/331** complete all-key scalar tolerance.
- Matched all-key successes: **7, 40, 37, 35, 46, 9**, each over 331. The constrained method has 41 eligible-only successes but 40 all-key successes; its paired all-key gains/losses are 36/3.
- Conditional Exact20 comparable denominators: **291, 291, 303, 320** for baseline/ratio/BGE/Oracle; semantic-IR has 285.
- Ratio residual partition: **30 + 31 + 49 + 214 + 7 = 331**.
- All **16** archived paired differences, gain/loss counts, exact McNemar tests, exact text-group swap tests and Holm-adjusted probabilities. Bootstrap intervals are retained from the archived correction; no new bootstrap or resampling was run.

`PRESERVED_PAIRED_STATISTICS.csv` is the aggregate correction from Stage 1 `results/sncs_corrections_2026-10-27/corrected_paired_statistics.csv`. `PRESERVED_TEXT_GROUPS.json` contains only the 331 public query-ID/group-ID mappings extracted from Stage 1 `results/joint_grounding_2026-10-27/grouping.json` (327 groups), with no query text. These permit public reconstruction of the exact paired tests.

Additional read-only ledger checks reproduced the common-18 OptMATH 11/15 and GPT-5.4 12/16 agreement counts, PaMOP 13 process successes and nine precision-unresolved cases, and ORLM 18 parsed/zero executed cases. PaMOP agreement remains withheld. The relationship ledger reproduces 217/217/197 valid objects, 631/651 total, 197 complete items and Fleiss kappa 0.370807178608. These are checks of archived outcomes, not new generations or solver runs. `CONSISTENCY_CHECKS.json` records aggregate receipts and source hashes; raw private ledgers are not copied.

## Source and evidence preservation

The updated Abstract and Introduction and all other pre-Results scientific text are byte-identical to Stage 1; the only pre-Results source change is the external supplement PDF filename. Sections 6–8, declarations and bibliography are unchanged. All eight direct main-table environments are byte-identical, as is the relationship table in its supporting input. Original supplement inputs, figures, table assets and Springer dependencies are preserved byte-for-byte.

The main Results section now leads with the retrieval/readiness/full-scalar gap, then the matched alternatives and residual partition. Tables 3 and 4 remain together on manuscript page 15. Fixed-score/search controls retain their negative findings and do not turn heterogeneous pipelines into a causal search comparison. Displaced protocol, runtime, occurrence and statistical-history detail remains in linked S7; additional archived campaigns and non-results are represented in S8. All 47 groups have destinations in the restructuring map.

## Build and visual inspection

Command: `sh manuscript/jdsa-phase1/build.sh`. The established clean temporary-directory pdfLaTeX/BibTeX build completed. Final main and supplement passes contain no warnings, undefined references/citations, multiply-defined labels or missing-glyph diagnostics. No overfull boxes were reported. Both bibliographies are rendered. All cross-document PDF targets use the two descriptive filenames.

All 59 pages were rendered and inspected in contact sheets; enlarged checks covered the prominent main tables and newly added supplementary tables. A punctuation-spacing correction in new supplementary prose was rebuilt and re-inspected. Tables, equations, figures and text are legible with no observed clipping or overlap. PDF text extraction found no unresolved `??` markers or text blocks outside page boundaries. The manuscript remains in the existing template, as requested.

| PDF | Pages | SHA256 |
|---|---:|---|
| `Retrieval_Assisted_Scalar_Grounding_JDSA_Phase1_Supplement.pdf` | 29 | `d9a57e110368874554dd3684e5a159045f5642127dd6ab7c578e2f05873f97c0` |
| `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_JDSA_Phase1.pdf` | 30 | `24ea7b4141694b99f05101737bea8057fefa512324a4106b44a73c763c93b8fc` |

## Public-release scope and remaining limitations

Only the two new revision directories are intended for this commit. The package contains manuscript sources, descriptive PDFs, bibliography/style/figure dependencies, aggregate evidence metadata and offline check code. Credential/private-link scans of the new text files found no matches. No restricted corpus, raw provider response, private author-review log or credentials were copied. Existing datasets, Hugging Face releases, experimental code, logs and branches are preserved.

Historical reconstruction remains incomplete: original headline scalar mappings, some subset manifests and rendered prompts, authenticated generation environments, full-precision PaMOP returns, and corrected per-instance OptMATH counts needed for the reported CI remain missing. Summary-only results are retained with that grade rather than promoted to independently reproducible predictions. Known catalog exposure, gold-assisted scalar participation, clustered/selected frames and absent human semantic truth remain explicit.

The archived ESWA narrative readiness count of 174 conflicts with 175 in its retained CSV; S8 uses the saved rows and labels their historical configuration. No current scientific result was replaced. Existing M5-specific availability/author-review wording awaits reconciliation in the next phase; its local release candidate is not uploaded here.

Recommended next phase: review this PDF and evidence register, then consolidate Sections 6–8 and supplementary placement according to the map, reconcile availability declarations, and separately handle JDSA formatting. No next-phase work or journal submission was started.

**New experiments, inference APIs, training, solver benchmarks, Slurm jobs, new data collection and Hugging Face modifications: none.**
