# JDSA compliance checklist

Reviewed 29 October 2026 against the [official JDSA submission guidelines](https://link.springer.com/journal/41060/submission-guidelines). Requirements below are paraphrased. This is an editorial/source-package check, not journal acceptance or a submission-system certification.

| Item | Result and evidence |
|---|---|
| Article type and scope | Regular Paper; methodological contribution and diagnostic evaluation stated explicitly. No autonomous formulation or novel algorithm claim. Editorial scope/acceptance remains the journal's decision. |
| Springer two-column source | PASS: `\documentclass[pdflatex,iicol,sn-basic,Numbered]{sn-jnl}`; main article uses the Springer two-column layout. Online Resource 1 uses the readable single-column companion layout. |
| Abstract and keywords | PASS: 225 words against the 150–250-word range; six keywords against the four-to-six range. No abstract citations. |
| Title, affiliation and correspondence | PASS: article title, Soroush Vahidi, NJIT affiliation and `sv96@njit.edu` present. Email delivery and present appointment were not independently authenticated. Author should confirm before submission. |
| ORCID | PASS: linked `0000-0003-1934-6282`; public ORCID name agrees. See `AUTHOR_METADATA_CHECK.json`. |
| Numbered headings | PASS: decimal headings, no more than three displayed levels; supplementary sections carry S prefixes. |
| Equations and notation | PASS: sequential compiled equation numbers, no unresolved references, selected-mention notation separated from query type score. Scientific definitions preserved. |
| Tables | PASS: six main and thirty supplementary tables; captions and notes preserve conditional denominators and protocol distinctions. No clipping/overfull diagnostics. |
| Figure and algorithm | PASS: editable vector diagram plus PDF; caption describes nodes, reference assistance and scope without relying on color alone; algorithm retained. Fonts embedded. No experimental figure reconstruction. |
| References | PASS with documented verification limits: numbered Springer bibliography and DOI links; corrected demonstrable errors from primary sources. See `REFERENCE_VERIFICATION_REPORT.md` for access failures and publisher date ambiguities. |
| Acknowledgements | PASS: identified after title/abstract area, before Introduction. Support attribution distinguishes this study from broader research. |
| Funding and competing interests | Present; educational credits and in-kind assistance distinguished from grants and from awards with unestablished use. Author attestation still required; no new funding assertion introduced. |
| Contributions and ethics/consent | Present in Statements and Declarations. Author must also enter contributions and competing interests in the submission portal as required. |
| Data/code availability | PASS: exact GitHub evidence tag and HF revision; historical v1.2, sanitized v2 projections and unreleased M5 originals distinguished. Gated/permission-unclear source content is not claimed public. |
| Reproducibility | PASS: saved-record reanalysis is distinguished from unavailable original inference reproduction. No new inference or solver experiment. |
| AI assistance | Present: writing, software assistance and scientific model calls separately described. The author's accountability is retained. Disclosures cover earlier assistance as well as this editorial pass; no inference that copyediting itself requires a new scientific author. |
| Supplement identity | PASS: Online Resource 1, article title, journal name, author, affiliation, email and ORCID included. File named `ESM_1.pdf`; suggested upload caption in `SUBMISSION_README.txt`. |
| Supplement navigation | PASS: linked contents and S numbering; every internal/cross-document target resolves in the compiled pair. |
| Editable upload | PASS: 33 files at ZIP root, no unnecessary subfolders. Main entry `main.tex`; bibliography, Springer class/style, local ORCID style, editable figure and all inputs included. |
| Clean compilation | PASS: extracted archive builds outside the repository with a cleared environment and installed TeX Live. All 52 rebuilt pages match final renderings and text. |
| Fonts, metadata and accessibility | Embedded fonts, English language metadata, title/author metadata, bookmarks and descriptive captions verified. PDFs are not PDF/UA-tagged; no formal accessibility certification is claimed. |
| External links and viewer behavior | Accessible links checked anonymously; blocked/rate-limited sites listed. Relative cross-PDF links need both files beside each other and a supporting PDF viewer. |
| Submission action | NOT PERFORMED, as requested. Portal-specific source processing has not been tested. Author must review final content, declarations and permissions before upload. |

The source package includes generated `.aux`/`.bbl` files to support cross-document references in single-entry journal compilation. They are intentional submission aids; the provided build script reconstructs them from source. The supplement is publication-ready as supplied because supplementary material may be published without journal copyediting.
