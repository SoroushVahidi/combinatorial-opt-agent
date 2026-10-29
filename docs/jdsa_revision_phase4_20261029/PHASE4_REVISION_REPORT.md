# Phase 4 targeted revision report

Source branch: `manuscript/jdsa-phase3-journal-preparation-20261029` at exact commit `49827d274cae995005681fb03a299a5363583861`. New isolated branch: `manuscript/jdsa-phase4-final-consistency-20261029`. Scope: references, abstract self-containment, author-block presentation, consistency and review delivery. This is not a broad rewrite or journal submission.

## Exact changes and reasons

- Corrected MIPLIB-NL to the official ICML 2026/PMLR publication; corrected AlphaOPT to its authenticated ACM KDD 2026 proceedings/DOI/pages; corrected Gao et al. to official COLM 2026 pre-proceedings and its author metadata. Added five independently verified missing page ranges. See the [reference report](REFERENCE_CORRECTION_REPORT.md).
- Changed the supplementary context table's AlphaOPT year from 2025 to 2026, with an explanatory note. This is a bibliographic year, not an experimental measurement. Every result-table value remains unchanged.
- Kept ReMatch, OptimAI and execution-guided decoding in the discussion with explicit preprint identification and primary links, while excluding these unverified-publication entries from the printed reference list in accordance with JDSA. Their editable BibTeX records are retained. No method or scientific finding was silently removed.
- Added short benchmark descriptions for NLP4LP and NL4Opt in the abstract. It changes from 227 to **237 whitespace-delimited words**, preserving all counts, percentages, qualifications and conclusions. No other abstract rewrite.
- Replaced the standalone ORCID line beneath the keywords with a linked ORCID icon next to the author name, using `orcidlink` with the unmodified Springer Nature class. The identifier is unchanged. The style is supplied with its original license notice. Author identity, affiliation, email, funding/credits, contributions and acknowledgement wording are preserved; no new verification checklist is imposed.
- Retained the acknowledgements as their distinct title-page component. An attempted full-width placement produced a vertical overflow and was discarded; the final package retains the stable column flow and original wording. No warning suppression was used.
- Updated descriptive filenames, cross-document PDF links, build/package scripts and the current availability phase designation to Phase 4. The seven-section structure, all scientific Sections 3–7, and the supplementary scientific evidence remain unchanged apart from the bibliographic context-table correction described above.

## Preservation and verification

[Scientific consistency](SCIENTIFIC_CONSISTENCY_REPORT.md) documents deterministic checks against saved records, all 35 tables and all 47 investigation-group destinations. [Artifact consistency](ARTIFACT_CONSISTENCY_REPORT.md) distinguishes public v1.2, local M5, aggregate reconstruction and missing historical records. [Visual verification](VISUAL_VERIFICATION_REPORT.md) covers every final PDF page. [JDSA compliance](JDSA_FINAL_COMPLIANCE.md) records checked requirements and objective remaining conditions.

The delivered PDFs have **22 main pages and 32 supplementary pages**. The ten-file flat source ZIP builds independently; extracted text and rendered page pixels match the modular-source builds. Deliverables are in [manuscript/jdsa-phase4](../../manuscript/jdsa-phase4/README.md). Old packages, research artifacts, uncommitted user work and earlier branches are preserved.

## Remaining objective boundaries

Unpublished status remains unresolved for the three preprints, which are handled as text mentions. SIRL pagination conflicts between primary metadata sources, so no unsupported range is added. Historical missing mappings, corrected per-instance extraction counts, full-precision objectives and some protocols remain missing. No formal journal page limit was verified. Actual submission must await confirmation of withdrawal of the previous overlapping submission. Confirmed author information is not reopened.

No experiment, inference, training, solver benchmark, paid API call or Slurm job was run. No Hugging Face write or journal submission occurred. This Phase 4 revision awaits independent review; no further phase was begun.
