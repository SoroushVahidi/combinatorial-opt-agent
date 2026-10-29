# Final editorial-polish report

## A. Authoritative source

Branch: `manuscript/jdsa-phase4-final-consistency-20261029`.
Commit: `cea99e86d33ad83219b9bf711c310e929ae18409`.
Date: 29 October 2026.

## B. Final-polish branch and commit

Branch: `manuscript/jdsa-final-editorial-polish-20261029`, created in an isolated worktree directly from the exact source commit.
Manuscript/package commit: `df4fa55f4e572c1a2ed10f300f727953755f73bf` (direct child of the authoritative Phase 4 commit).
The final report commit adds verification documentation only. No earlier branch or existing artifact was modified.
The final documents and editable inputs are in `manuscript/jdsa-final-review/`.

## C. Exact editorial corrections

Only the following 15 prose/link substitutions were made. The snippets are literal LaTeX, preserving exact wording and link targets. [EDITORIAL_CHANGES.json](EDITORIAL_CHANGES.json) provides the machine-readable list; [EXACT_SOURCE_DIFF.patch](EXACT_SOURCE_DIFF.patch) provides the complete source comparison, including packaging identifiers.

### 1. `main.tex`

Remove pending-review language without changing the declaration.

Before:

```tex
Not applicable to participant data; the manuscript and submission declarations remain subject to the author's final review.
```

After:

```tex
Not applicable; this study did not involve participant data.
```

### 2. `main.tex`

Retain nonpublication status without internal inspection language.

Before:

```tex
This M5 candidate is prepared for author inspection and has not been uploaded.
```

After:

```tex
This M5 evidence bundle remains local and has not been uploaded.
```

### 3. `main.tex`

Replace preparation-phase prose with a precise public artifact link.

Before:

```tex
The present manuscript source and supplementary summaries are available on the dedicated JDSA Phase 4 branch.
```

After:

```tex
The manuscript source and supplementary material are available in the repository's \href{https://github.com/SoroushVahidi/combinatorial-opt-agent/tree/manuscript/jdsa-final-editorial-polish-20261029/manuscript/jdsa-final-review}{versioned source package}.
```

### 4. `main.tex`

Describe the analysis and its limits rather than a revision.

Before:

```tex
This revision uses frozen records and adds no model inference, solver execution or performance experiment.
```

After:

```tex
The reported analyses are based on frozen experimental records; their reanalysis involves no new model inference, solver execution or performance experiment.
```

### 5. `main.tex`

Preserve AI-use disclosure and responsibility; remove pending internal review status.

Before:

```tex
During preparation the author used ChatGPT and Gemini to improve readability and language. OpenAI Codex later assisted manuscript revisions, evidence organization and deterministic validation. The author reviewed and edited the original work and remains responsible for its content; the present journal-preparation revision and the separate M5 release candidate await author review.
```

After:

```tex
The author used ChatGPT and Gemini to improve readability and language. OpenAI Codex assisted manuscript editing, evidence organization and deterministic validation. The author reviewed and edited the original work and remains responsible for its content.
```

### 6. `relationship_evidence.tex`

Replace editorial timing in Table 9 caption with evidence provenance; table data unchanged.

Before:

```tex
No new calls were made for this revision.
```

After:

```tex
The counts are derived from archived responses.
```

### 7. `supplement.tex`

Remove internal review designation from Online Resource 1 identification.

Before:

```tex
International Journal of Data Science and Analytics (author-review manuscript).
```

After:

```tex
International Journal of Data Science and Analytics.
```

### 8. `supplement.tex`

Describe the evidence basis without editorial-process language.

Before:

```tex
No new model predictions or experiments were produced for this revision.
```

After:

```tex
The analyses reported here use archived predictions and experimental records.
```

### 9. `supervision_diagnostics.tex`

Preserve the no-training qualification without revision framing.

Before:

```tex
no model is trained from that packet for this revision.
```

After:

```tex
no model was trained from that packet in the reported analyses.
```

### 10. `reproduction_map.tex`

Preserve local-only evidence status without editorial review wording.

Before:

```tex
M5 is a local author-review candidate, not an uploaded release.
```

After:

```tex
M5 remains a local evidence bundle, not an uploaded release.
```

### 11. `reproduction_map.tex`

Preserve the evaluation-independence limitation in scientific language.

Before:

```tex
The absence of new fitting in this revision does not establish evaluation
independence.
```

After:

```tex
Reanalysis without additional model fitting does not establish evaluation
independence.
```

### 12. `reproduction_map.tex`

Replace obsolete editorial-phase framing while preserving the public/local distinction.

Before:

```tex
The Phase 2 branch publishes manuscript sources, supplementary summaries and deterministic aggregate checks; it does not publish the local M3/M5 evidence bundles or private author-review records.
```

After:

```tex
The public repository provides manuscript sources, supplementary summaries and deterministic aggregate checks; it does not publish the local M3/M5 evidence bundles or private author-review records.
```

### 13. `reproduction_map.tex`

State the public dataset version rather than repository-maintenance activity.

Before:

```tex
Upstream access conditions continue to apply; no dataset or Hugging Face record was changed for this manuscript revision.
```

After:

```tex
Upstream access conditions continue to apply; the public Hugging Face dataset remains the historical v1.2 release.
```

### 14. `additional_evidence.tex`

Remove revision history while preserving evidentiary scope and no-new-execution qualification.

Before:

```tex
The following additions preserve relevant work previously absent or only briefly mentioned; they do not replace the main matched comparison. No experiment, prediction, model call or solver run was performed for this revision.
```

After:

```tex
The following sections report complementary archived investigations; they do not replace the main matched comparison. These analyses use saved records without additional experiments, predictions, model calls or solver runs.
```

### 15. `main.tex`

Use descriptive hyperlink text at the unchanged dataset URL to prevent the long URL splitting across pages after declaration shortening.

Before:

```tex
Historical v1.2 derived metrics are public at \url{https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics}, under their existing CC BY-NC-SA 4.0 license.
```

After:

```tex
The \href{https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics}{historical v1.2 derived metrics} are public under their existing CC BY-NC-SA 4.0 license.
```

The two source comments now identify the final editorial-polish package; relative PDF links and output filenames use `JDSA_Final_Review`. The document build script, flat-package generator and README reflect the new artifact directory and names. No class, bibliography style, table sizing, font setting, section structure or figure was changed.

Author-reviewed D1 labels, private annotation conversations, the uncompleted independent reference protocol, historical numerical corrections, model/checkpoint revisions and pinned evidence paths remain explicit. These describe scientific provenance or actual access limitations, rather than pending manuscript approval. No previously confirmed author information was reopened.

## D. Scientific preservation checks

The exact replacement list reconstructs every changed LaTeX file from Phase 4. All other LaTeX bytes are identical. `main.tex` from the abstract through the conclusion is byte-identical; its included Table 9 caption has only the listed archival-provenance wording change. All 35 table bodies, 12 numbered equations (nine main, three supplementary), inline mathematics, algorithms, citation commands, labels and section headings are unchanged. All experimental numbers, populations, denominators, statistical findings, algorithmic details and conclusions are therefore preserved. No experimental scripts or results were edited or executed.

The 47-group destination manifest is byte-identical to the exact source commit. The complete D1 source is byte-identical, including author/language-model assistance, shared lineage and no-provider-performance qualifications. No negative, inconclusive or incomplete evidence was removed. Author name, affiliation, email, ORCID, acknowledgement wording, funding, credits, contributions, ethics and software-assistance declarations are preserved. Consent and writing-assistance declarations change only as itemized above.

[Preservation checks](PRESERVATION_CHECKS.json) record these comparisons. The document-only checker is [verify_documents.py](verify_documents.py); it performs no scientific reevaluation or inference.

## E. Bibliography preservation

`references.bib` is byte-identical to Phase 4, SHA-256 `6632d047b5c4933c4f55f09658c17ee98af921b86fa0f4337ab46732737a9b70`. All 48 editable records, citation commands and corrected publication metadata are retained. The same 41 distinct references are cited across the documents. The three preprint discussions and primary links remain unchanged. No reference investigation or replacement was introduced in this editorial task.

## F. PDF and editable-source verification

Both modular sources compile successfully with the unchanged Springer Nature class and two-column main layout. The ten-file flat ZIP was extracted into an empty temporary directory and compiled independently. Every rebuilt page matches the corresponding modular-build page in extracted text and 72-dpi rendered pixels. PDF byte identity between separate compilations is not claimed because build metadata can differ.

Final logs contain no unresolved citations, broken cross-references, missing glyphs, multiply defined labels or overfull boxes. The main document retains the same harmless hyperref bookmark-level adjustment warning observed in the Phase 4 build; the supplement has no package warnings. No warning was suppressed.

All 54 pages were rendered and compared with Phase 4. Forty-two pages remain pixel-identical: main pages 1–15 and 17; supplement pages 2–8, 13–23 and 25–32. This includes the main title page, ORCID and acknowledgements, all wide result tables except the caption-only change to Table 9, and D1 Tables S25–S26. The twelve changed pages were visually inspected: main 16 and 18–22; supplement 1, 9–12 and 24. Declarations, page/column flow, table captions, supplementary identification and references remain readable, with no clipping, overlaps or new excessive whitespace. The long dataset URL initially split across pages after the declaration shortening; item 15 replaces its display with descriptive linked text at the identical URL. No spacer or font reduction was introduced.

Internal and cross-document link destinations were checked against PDF destination tables, including the renamed supplementary PDF. The ORCID target remains `https://orcid.org/0000-0003-1934-6282`. Keep both PDFs in the same directory for cross-document navigation; browser support varies.

[Build verification and artifact checksums](BUILD_VERIFICATION.json) record page counts, rendered comparisons, PDF-link checks and log diagnostics. The [source-package generator](package_submission.py) produces the flat ZIP. To repeat the document QA from the repository root:

```sh
KEEP_BUILD=1 sh manuscript/jdsa-final-review/build.sh > /tmp/jdsa-final-polish-modular-build.log 2>&1
python docs/jdsa_final_editorial_polish_20261029/package_submission.py
python docs/jdsa_final_editorial_polish_20261029/verify_documents.py
```

## G. Page counts

Main manuscript: **22 pages**. Supplement: **32 pages**. Both are unchanged from Phase 4.

## H. Artifact availability

The public source commit and its complete GitHub tree were checked read-only. Historical code and metrics and Phase 4 document sources are public. The tree does not contain `reproducibility/m5/`, `verify_llm217_m5.py` or `COMMON18_CORRECTED_CASES.csv`. The final branch adds only the final document package and this report directory; it does not release M3/M5 evidence. Historical public aggregates remain distinct from missing original scalar mappings, incomplete prompts, full-precision outputs and other unreconstructible records.

The Hugging Face API returned HTTP 200 and the unchanged historical v1.2 revision `24a047fa5d661be6cfcf75980ac8f62e6c942243`, last modified 24 August 2026, with the existing CC BY-NC-SA 4.0 license. No dataset or model was modified. See [Hugging Face check](AVAILABILITY_CHECKS.json) and [GitHub check](GITHUB_AVAILABILITY_CHECK.json).

The dedicated branch was pushed successfully. Public HTTP 200 downloads of both PDFs and the ZIP matched the local build bytes and SHA-256 values. [Remote artifact verification](REMOTE_ARTIFACT_VERIFICATION.json) records immutable URLs and checksums at the manuscript/package commit. The final report and verification documentation are committed separately; the final branch HEAD and public report are checked after that documentation commit. The manuscript package is unchanged by that final documentation commit.

## I. Remaining genuine submission blocker

The previous overlapping **SN Computer Science submission must be formally withdrawn before any new journal submission**. No withdrawal or JDSA submission was performed here. Existing historical-evidence limitations and bibliographic uncertainties remain disclosed in the manuscript and Phase 4 reports; no new scientific or author-information blocker was identified.

No new experiments, solver benchmarks, inference, training, paid API calls, Hugging Face writes, restricted-material release, merge into main or subsequent revision phase occurred.
