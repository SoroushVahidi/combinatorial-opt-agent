# JDSA compliance checklist

Checked 29 October 2026 against the [official JDSA instructions](https://link.springer.com/journal/41060/submission-guidelines). The journal's source requirements were checked live; local PDF/source checks are recorded separately. “Author confirmation” identifies information an editorial audit cannot attest on the author's behalf.

| Item | Finding |
|---|---|
| Article type | Regular research paper; diagnostic methodology and empirical application remain explicit |
| LaTeX layout | Springer `sn-jnl`, `pdflatex,iicol,sn-basic,Numbered`; two-column main body |
| Editable upload | Flat ZIP contains both entry points, included text, bibliography, BBLs, local styles and figure source/PDF; no subfolders |
| Abstract/keywords | 222 words by documented tokenizer; six keywords; no citations; artificial intelligence spelled out |
| Title/author | Informative title; Soroush Vahidi; full NJIT departmental affiliation in main; corresponding email shown in both |
| ORCID | Existing 0000-0003-1934-6282 retained, linked in both; public page requires JavaScript; identity and contact activity require author confirmation |
| Headings | Decimal numbering, no more than three levels; supplement uses S-prefixed section numbering |
| Equations | Editable TeX, sequential numbering, consistent endpoint notation; numerical predicates preserved |
| Figures | One vector TikZ diagram, editable source included; descriptive caption identifies inputs, assistance and arrows; labels/dashed border make meaning independent of color; fonts embedded |
| Tables | Six main tables and 30 supplementary tables; Arabic/S numbering; explicit callouts; captions define populations and measures |
| References | Square-bracket numbers; only cited entries printed; full DOI links when supplied; three unpublished preprints are discussed as such in text, outside printed lists |
| Acknowledgements | On first page; existing support distinctions preserved |
| Declarations | Funding, competing interests, ethics/consent, contributions, data/code availability and reproducibility are present; submission interface declarations still require author entry |
| AI assistance | Editing/development disclosures retained; scientific service protocols separately described; human accountability retained |
| Data access | Public v2 projections distinguished from gated/permission-unclear sources and unreleased originals; no claim of full historical reproduction |
| Supplement | `ESM_1.pdf`, title identifies Online Resource 1; article title, author, journal, affiliation/email, linked contents and consistent numbering present |
| Supplement caption | Main supplementary-information paragraph describes contents; upload caption supplied in source README |
| PDF QA | All 52 pages inspected; metadata populated, fonts embedded, page numbers correct, no clipping/overflow or unresolved references |
| Accessibility | Searchable text, descriptive figure caption, readable diagram, language metadata and bookmarks; not certified as tagged PDF/UA |
| Journal submission | Not performed; no acceptance or editorial decision implied |

The source ZIP is ready for author review. When uploading, identify the main TeX entry point and designate the companion PDF as Online Resource 1. Browser support for companion-PDF links varies; the target files and named destinations are validated. The submission system may rename or merge files, so its generated proof needs the author's final check.

The pinned release uses versioned repository URLs and a commit identifier, not a newly minted DOI. No DOI was invented. Bibliographic source-access and year discrepancies are documented in the reference report.
