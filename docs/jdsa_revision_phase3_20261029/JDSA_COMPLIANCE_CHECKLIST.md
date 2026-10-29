# JDSA formatting and submission-review checklist

Checked 29 October 2026 against the [official JDSA submission guidelines](https://link.springer.com/journal/41060/submission-guidelines), [Springer Nature LaTeX support](https://www.springernature.com/gp/authors/campaigns/latex-author-support), and the official template bundle linked there. These sources, rather than inferred journal page limits, govern the preparation.

## Published-paper conventions examined

The organization and visual presentation of [Nikitina and Mouriño, 2026](https://doi.org/10.1007/s41060-026-01295-3), [Alikhah et al., 2026](https://doi.org/10.1007/s41060-026-01336-x), and [Majadly and Last, 2025; online 2024](https://doi.org/10.1007/s41060-024-00665-z) were examined through publisher HTML/PDF. Their numbered citations, decimal headings, two-column prose, full-width evidence tables, and separation of methods, experiments and interpretation inform this layout. Interior table pages of the latter two were also inspected. Their wording and publisher typesetting ornaments were not copied; this package uses the submission template.

| Requirement | Current implementation / status |
|---|---|
| Springer Nature LaTeX, two columns | PASS: `sn-jnl` with `iicol,sn-basic,Numbered`; `iicol` is the journal's specific SN option. The vendor class is identical to the official bundle (SHA-256 `36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc`). |
| Title, author, affiliation, corresponding email | Present; carried from authoritative source. Author must confirm current affiliation/contact. |
| ORCID | Linked 16-digit identifier via the intended class hook; separate from email. Text fallback preserves accessibility because the bundle lacks the icon expected by the class. Author ownership still requires confirmation. |
| Abstract 150–250 words | PASS; unchanged Phase 2 abstract, approximately 227 words. Automated tokenization count is in document checks. |
| Keywords 4–6 | PASS: six. |
| Numbered citations | PASS: square-bracket numbers, no author–year citations; separately numbered supplement bibliography. |
| Displayed equations | PASS: main (1)–(9), supplement (S1)–(S3); numbered and referenced, no unnumbered display math. |
| Decimal sections, maximum three levels | PASS: seven principal sections; deepest numbered heading is a subsubsection. |
| Tables | PASS: main 1–9 and supplement S1–S26; captions, population/denominator notes, readable row spacing, values preserved. Wide tables span columns. |
| Figures | PASS for review: vector PDF with embedded fonts and editable source; readable labels, no clipping, bold numbered caption label, caption beneath figure without final punctuation. No new decorative figure. |
| Algorithm | PASS: full-width, readable, complete 17-step procedure with explicit assistance boundary. |
| Supplementary information | PASS: cited as Online Resource 1, title/journal/author/contact metadata, descriptive `ESM_1_...pdf`, complete document. Supplement is single-column; the main is two-column. |
| Acknowledgements | Present on title page, separate from declarations. People and credit programs preserved; factual scope requires author review. |
| Statements and Declarations | Correct heading; funding, interests, contributions, ethics/consent, data/code and assistance statements present. See author decisions below. |
| Availability | Public historical v1.2 and GitHub are identified; local M5 is not claimed public. Missing records remain missing. No release performed. |
| Complete editable source and flat upload | PASS: independently built nine-file ZIP without subfolders, alongside modular repository source. Includes vector/editable figure, class/style, bibliography and build instructions. |
| Formal page limit | **Not verified:** no applicable numerical regular-article page limit was found in the consulted instructions. No page-limit claim is made. |
| Prior overlapping submission | **SUBMISSION BLOCKER:** withdrawal confirmation is not established in the reviewed records. Do not submit until confirmed. |
| Author approval, funding/credit attribution, permissions | **PENDING AUTHOR:** this review package does not establish approval or grant/license decisions. |
| Preprint treatment | **PENDING AUTHOR/EDITOR:** explicitly identified preprints remain; acceptance must not be invented to satisfy the journal's published/accepted-reference instruction. |

## Limits of compliance verification

Both document builds and the independent archive build pass. Every rendered page was inspected. Internal PDF destinations and cross-document destinations resolve. Some browsers do not implement cross-PDF navigation; download both PDFs into one directory. Publisher access challenges and email ownership cannot be certified by a PDF link check. Historical inference endpoints were deliberately not contacted. See the reference report for restricted/uncertain source access.

Journal editorial acceptance, author approval, data-rights clearance and withdrawal confirmation are not established by formatting compliance. The package is ready for independent author review, not automatic submission.
