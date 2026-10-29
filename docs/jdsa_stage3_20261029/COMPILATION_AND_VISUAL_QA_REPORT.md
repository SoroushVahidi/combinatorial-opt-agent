# Compilation and visual-QA report

**PASS.** Final main article: 15 pages. Online Resource 1: 37 pages. All 52 pages were visually inspected at 105 dpi; changed pages were inspected again after typography adjustments. The individual render hashes and review ledger are in [VISUAL_PAGE_REVIEW.csv](VISUAL_PAGE_REVIEW.csv).

## Build

Both documents were compiled in fresh `/tmp/jdsa-build.*` directories from flat source copies. Fig1 was regenerated from its editable TikZ source. The sequence was pdfLaTeX for the figure and each document, BibTeX for each document, then three paired supplement/main passes. The final editorial build directory was `/tmp/jdsa-build.EzbgRq`; preserved final logs are in [build_logs](build_logs). A separate clean build from the extracted submission ZIP is recorded in [PACKAGE_REBUILD_CHECKS.json](PACKAGE_REBUILD_CHECKS.json).

The environment was pdfTeX 1.40.25 / TeX Live 2023 (Debian), BibTeX 0.99d, with the baseline Springer class and local ORCID style. No margins or body font sizes were changed. Standard TeX distribution packages remain installation dependencies; all project-specific sources/styles are supplied.

## Checks

- Zero unresolved citations or cross-references, multiply defined labels, missing characters, overfull boxes, or BibTeX warnings in final logs.
- All fonts embedded in both PDFs; [main fonts](main-fonts.txt), [supplement fonts](supplement-fonts.txt).
- No extracted words outside the PDF page boxes, no replacement characters, no source placeholders, and correct sequential visible page numbers.
- All 31 companion-document link annotations resolve to existing filenames and named destinations: 20 from main to ESM_1 and 11 back. Internal link targets and bookmarks also resolve. External HTTP failures/access challenges are listed separately, not counted as unresolved LaTeX references.
- Populated title, author, subject, keyword and English-language metadata; searchable text and working contents navigation.
- All six main tables, 30 supplementary tables, the algorithm, pipeline figure, 11 main equations, three supplementary equations and both reference lists inspected. No visibly clipped/overlapping content found.

## Visual review findings

Main pages 1–6: title, affiliation, abstract, keyword wrap, decimal headings and metric definitions are legible. Page 7: algorithm steps and diagram arrows/labels are clear; color is not the only encoding. Pages 8–11: all main tables fit; conditional denominators and factorial arrows are readable. Pages 12–15: conclusion, declarations, DOI links and bibliography fit.

Supplement pages 1–2: linked contents and S10/S11 number columns are aligned. Pages 3–13: context/supporting tables, source conventions and reproduction instructions fit. Pages 14–18: common-18 provenance, precision predicates and individual dispositions are legible. Pages 19–23: request/validity counts, Fleiss equation and cluster table are clear. Pages 24–32: historical investigations, retained numeric tables and OptMATH micro/macro definitions fit. Pages 33–34: all twelve D1 pairs and their qualifications are legible. Pages 35–37: references render without a nearly empty trailing page.

The final review fixed two contents-number overflows, a stretched sentence and reference spillover. Underfull-box diagnostics from ordinary justified text are not represented as overflows. The PDF is not a certified tagged-PDF/UA artifact; captions, text extraction, font embedding and language metadata were checked. Companion-PDF link opening depends on viewer security settings; semantic destinations are validated, but no claim of universal browser behavior is made.
