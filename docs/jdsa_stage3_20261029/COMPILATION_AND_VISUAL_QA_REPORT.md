# Compilation and visual-QA report

**PASS: 15-page article and 37-page Online Resource 1, all 52 pages visually reviewed.** Final PDFs were built from clean temporary source copies. The flat source archive was then extracted outside every repository worktree and rebuilt with a cleared environment, a fresh temporary home, and the existing system TeX installation. Every rebuilt page has identical extracted text and 72-dpi rendered pixels to the final PDFs. PDF bytes differ only in build-dependent metadata/identifiers; byte identity is not claimed.

## Deliverable integrity

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Main PDF | 444821 | `275c93618bd2c68ca943603cab953c8a466eb9a0f6d641a92d92da280c5a9d68` |
| ESM_1.pdf | 429499 | `9b014d411b495c65c1c03f7e0e924ee574cb8f21e4745492589c41d8214965aa` |
| Flat source ZIP | 1022633 | `ef3df1770f7499311d15316fdf8c41cec23776054d372e00fbb755f496220e87` |

[SUBMISSION_PACKAGE_MANIFEST.json](SUBMISSION_PACKAGE_MANIFEST.json) lists all 33 ZIP members and hashes. The ZIP's `MANIFEST.sha256` verifies every other member. The source package contains all local inputs, editable diagram source, shared bibliography, class/style files, final PDFs, compiled bibliography/auxiliary files and instructions. No evidence originals or unnecessary directory hierarchy are included.

## Build results

- Engine: pdfTeX 1.40.25, TeX Live 2023/Debian, with BibTeX. Both document entry points are built before repeated paired passes resolve cross-document labels.
- Final main and supplement logs: **zero unresolved citations, unresolved cross-references, missing characters, overfull boxes or multiply-defined-label diagnostics**. Both final BibTeX logs have zero warnings.
- Underfull-box diagnostics remain (56 main, 41 supplement); these are spacing notices, not overfull/clipped material. Their corresponding rendered layouts were reviewed. No claim of a wholly warning-free TeX log is made.
- Every font is embedded. Text span bounds remain within page rectangles. Page numbering is sequential from 1 to 15 and 1 to 37. No missing-glyph replacement characters were found.
- All internal destinations resolve; **31 cross-document links** resolve to named destinations in the supplied companion PDF. Two previously inaccurate supplementary cross-references were corrected to local sections.
- PDF title, author, subject, keywords, English language and navigation bookmarks are present. Supplemental contents are linked and have correct S10/S11 spacing.
- The standalone editable `Fig1.tex` also compiles successfully. The existing figure PDF is preserved, not scientifically redrawn.

Detailed receipts: [PDF_QA.json](PDF_QA.json), [CLEAN_PACKAGE_BUILD_CHECK.json](CLEAN_PACKAGE_BUILD_CHECK.json), [VISUAL_PAGE_REVIEW.json](VISUAL_PAGE_REVIEW.json). The latter records a positive visual review for every page, not a sampled subset.

## Render review

Reviewed all pages as rendered contact sheets, with full-page inspection of dense equations, Algorithm 1/Figure 1 and the conditional-agreement table. Checked table captions, symbols, bibliography wrapping, page breaks and supplementary navigation. Removed an almost empty trailing bibliography page by omitting redundant printed publisher links while retaining their DOI destinations. No table, equation, evidence or qualification was removed to achieve this.

The final article retains the Springer two-column body; wide tables/algorithm/figure use their intended widths. Online Resource 1 remains a single-column 37-page resource with eleven linked sections and thirty tables. Long historical tables and D1 examples are legible and unclipped. Scientific object numbering remains unchanged from Stage 2.

## Practical limits

This uses a clean source/environment build on the installed TeX distribution, not a newly installed OS or hermetic container. No private files or network access are required by the build, but a new syscall-tracing audit was not performed in Stage 3. Earlier Stage 1 tracing concerns the evidence verifier and is not misrepresented as a trace of this LaTeX build.

Relative cross-PDF links require both final files in the same directory and a viewer that supports remote PDF destinations. A browser or journal platform may disable that feature; visible section/table references remain usable. External publisher access restrictions are documented separately. The PDFs are not PDF/UA-tagged; formal screen-reader conformance has not been established. Journal portal compilation and upload have not been performed.

Final source whitespace cleanup was rebuilt and compared page by page against the visually reviewed PDFs; all text and rendered pixels remain identical. Repository whitespace checks pass for edited sources; copied Springer/ORCID vendor files retain their byte-identical upstream formatting.
