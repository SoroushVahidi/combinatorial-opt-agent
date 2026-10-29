# Visual verification of the final Phase 3 PDFs

Date: 29 October 2026. This is an assistant visual/technical review, not author approval or independent scientific peer review. Final PDF hashes and page counts are recorded in [BUILD_AND_ARTIFACT_VERIFICATION.json](BUILD_AND_ARTIFACT_VERIFICATION.json).

## Coverage and method

Every page of the final 22-page main manuscript and 32-page supplement was rendered at approximately 119 dpi and inspected individually at readable resolution. Changed pages were re-rendered and re-inspected after the final equation, citation and float corrections. Difficult main tables/algorithm/figure were additionally examined in 180-dpi detail crops, and supplementary D1 tables in 216-dpi detail crops. The inspected artifacts are the repository PDFs, not an older Phase 2 build. Temporary render images and build logs are not included in the public source package.

| Document pages | Visual review focus and outcome |
|---|---|
| Main 1–4 | Title, author/email/ORCID, abstract/keywords, acknowledgements, Introduction and Related Work: readable, no overlap or missing text. |
| Main 5–8 | Definitions and setup: equations 1–9 remain numbered; multi-line expressions fit the columns. The initially awkward strict-readiness split and indicator glyph were corrected and re-inspected. |
| Main 9 | Full-width 17-step Algorithm 1 and pipeline figure: all labels/arrows/inputs readable, assistance boundary explicit, caption within margins. |
| Main 10–12 | Retrieval and diagnostic results; Tables 1–4: rows, denominators, headers and notes readable. Tables 2–4 inspected at higher resolution. |
| Main 13–16 | Controlled and complementary evidence; Tables 5–9: compact/full-width placements fit, captions define populations, Table 8 denominators and Table 9 recovery stages are legible. Tables 8–9 inspected at higher resolution. |
| Main 17–19 | Discussion, limitations, conclusion, declarations and bibliography transition: coherent flow, no clipped text or overlapping floats. |
| Main 20–22 | References: wrapped URLs/DOIs, correct arXiv decimal separators after the bibliography-field correction, readable numbered entries. |
| Supplement 1–4 | Title/contact metadata and S1–S6 tables: larger prior-work text and wrapped columns are readable. Some title/float whitespace remains. |
| Supplement 5–10 | Factorial, inventory, assistance and population evidence; S7–S10 and equations: all rows and links retained, no clipping. |
| Supplement 11–16 | Availability, archived implementation/formulation and common-18 evidence; S11–S14: long paths wrap and table data remain legible. |
| Supplement 17–21 | G3 and relationship protocols; S15–S18 and Fleiss equation S3: complete tables, readable notes, incomplete human-review status preserved. |
| Supplement 22–24 | Subset/statistical and historical negative results: readable prose and cross-references. |
| Supplement 25–28 | Historical result tables S19–S23 and auxiliary learning findings: complete, aligned numeric columns and adequate row spacing. |
| Supplement 29–30 | S24 and extraction/D1 qualifications: no missing rows; source/denominator and annotation limitations remain visible. |
| Supplement 31 | S25 and S26: all 12 controlled-pair summaries, labels and categories retained. Both tables separately inspected at 216 dpi; ragged category wrapping and row separation are readable. |
| Supplement 32 | All seven supplementary references rendered and readable. |

## Technical corroboration

Clean final logs contain no overfull boxes, missing characters, unresolved citations/references or multiply defined labels. All embedded PDF fonts were confirmed present. Extracted text blocks lie within page bounds. Named internal destinations and all relative cross-document destinations resolve to existing targets. The independently built flat source archive yields exactly identical extracted page text and 72-dpi rendered page pixels; PDF container bytes can differ because of timestamps/IDs.

No clipped text, overlapping objects, unreadably small table content, missing figure/table, broken equation or margin overflow was observed. This is a visual finding, not a conclusion drawn solely from the LaTeX log. The supplement's remaining cover/float whitespace is disclosed; evidence tables were not deleted or miniaturized to shorten it. Full-width floats on dedicated table-heavy pages are acceptable where readable.

External hyperlinks have narrower verification: primary bibliographic and publication endpoints were checked where accessible, public availability metadata was read, and final GitHub downloads are verified at delivery. Email ownership, access-controlled publisher pages and historical inference endpoints are not certified. No inference endpoint was contacted. Cross-PDF navigation may require a desktop reader and both PDFs saved together; printed section/table identifiers also resolve.
