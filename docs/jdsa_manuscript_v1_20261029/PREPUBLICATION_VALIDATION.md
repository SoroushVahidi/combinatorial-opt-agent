# Manuscript v1.0.0 link and packaging validation

This report records checks completed before the new GitHub release is published. The proposed manuscript release URL is a reserved destination at this point; this report alone does not assert public availability. The release notes and subsequent independent public-download receipt record publication verification.

Baseline: `e12c413c18e6d1f068d29509e7f9108e7ef51f65`, verified locally and at origin. Separate worktree and branch: `manuscript/jdsa-permanent-release-v1-20261029`. The requested target tag was absent locally and remotely, and the authenticated release lookup returned 404 before preparation.

## Exact link changes

[EXACT_LINK_CHANGES.json](EXACT_LINK_CHANGES.json) records every TeX before/after edit. [SCIENTIFIC_PRESERVATION.json](SCIENTIFIC_PRESERVATION.json) reconstructs all 19 TeX files from the exact baseline using only those edits; every other byte matches. The bibliography, local styles and build script are unchanged.

1. Main Code availability replaces the two temporary Stage 3 branch links (manuscript source and audit directory) with `https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0`, using the descriptive text “manuscript release v1.0.0”. It identifies the public project repository explicitly and retains the immutable code/evidence release.
2. Main Data availability retains the exact Hugging Face URL/revision and changes only its visible label to “Hugging Face evidence at the exact revision”. All redistribution and historical limitations remain unchanged.
3. Supplement Section S4.3 retains the pinned evidence release, revisions and historical evidence boundaries; its data link is labelled “exact-version Hugging Face evidence”, and it links the versioned manuscript release for sources and the separate crosswalk/audit.
4. The two external-document bases now use the exact permanent PDF asset URLs. Imported anchors use the standard `#nameddest=` PDF parameter. All 31 cross-document destinations resolve in the final target PDFs; there are no local-path or temporary-branch hyperlink targets.
5. README replaces the temporary audit link and branch-dependent instructions with the four permanent destinations, package/build instructions, release filename mapping and explicit pre-submission/reanalysis qualifications.

## Reproducibility and scientific preservation

- Frozen verifier: **9,737 checks pass**, including the 172 authenticated manifest payloads. [Receipt](FROZEN_REANALYSIS_VERIFICATION.json).
- Applicable Stage 3 procedures, adapted only for baseline/output paths: **711 checks pass**, including both final build logs. [Receipt](NUMERICAL_CLAIM_CONSISTENCY.json).
- Independent readiness-condition derivations, exact float numerical-cell checks and source-number comparisons pass. The source-number check excludes the Code availability paragraph's release version and resource identifier; the stricter exact allowed-edit reconstruction separately checks that whole paragraph.
- The 7/265 and 7/255 findings, conditional Exact20 denominators, 40-versus-41 distinction, 49-query controls, OptMATH definitions, PaMOP withholding and correction families remain unchanged.
- No scientific claim or experimental input is altered. No model inference or solver experiment is performed.
- Crosswalk refreshed for final source hashes/lines and compiled object pages: all 37 original numbered objects, 183 numerical passages and 47 investigations, plus the unchanged Stage 2 derived claim, remain covered.

## Compilation, layout and package

Both PDFs compile in a clean temporary directory, with the figure regenerated from editable TikZ source. Final main: 15 pages; supplement: 37 pages. No unresolved citations/references, missing characters, overfull boxes or BibTeX warnings remain. Fonts are embedded; page numbering and internal targets pass. [Document checks](FINAL_DOCUMENT_CHECKS.json).

Every page was pixel-compared at 105 dpi with the previously inspected Stage 3 baseline. All 18 changed pages were visually inspected; the remaining 34 pages are pixel-identical. Changed pages are main 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15 and supplement 5, 6, 7, 9, 12, 13, 26. Most differences are the online-link color; the availability text reflows on main pages 13–15 and supplement page 9. All modified declarations, references, tables and mathematical content fit without clipping or overlap. [Page ledger](VISUAL_COMPARISON.json).

The 30-file flat ZIP is built from an explicit source/asset allowlist, with no datasets, raw records, conversations, credentials, logs or unrelated files. Its sources, bibliography, local styles, editable figure and final PDFs are complete. [Package manifest](SUBMISSION_PACKAGE_MANIFEST.json). An independent clean rebuild from the extracted ZIP is required before publication; its receipt is [PACKAGE_REBUILD_CHECKS.json](PACKAGE_REBUILD_CHECKS.json).

The existing public repository, evidence release, exact Hugging Face tree and manifest returned HTTP 200 anonymously; the downloaded Hugging Face manifest is byte-identical to the authenticated local manifest. [Existing destination checks](EXISTING_PUBLIC_DESTINATIONS.json). Public verification of the new release and its four assets is deliberately deferred until publication; `verify_public_release.py` checks the tag, all bytes/checksums and all four permanent destinations without credentials.

## Remaining boundaries

Opening online companion-PDF links requires network access and viewer support for named destinations. Journal renaming/rendering still needs proof review. Historical evidence gaps, gated/permission-unclear materials and the author-confirmation requirements from Stage 3 remain unchanged. Bibliographic/service URLs outside this task retain their documented access limitations; inaccessible sites are not evidence of an incorrect citation. No DOI, acceptance or journal publication is asserted.
