# Attribution correction v1 — PDFs for author review

These newly compiled PDFs contain the bounded attribution correction to manuscript release v1.0.0. They are **not a new release or an accepted/published journal article**.

- [Manuscript](manuscript.pdf)
- [Online Resource 1](ESM_1.pdf)
- [Correction note and exact changes](../../docs/jdsa_attribution_correction_v1/CORRECTION_NOTE_V1.md)
- [Validation report](../../docs/jdsa_attribution_correction_v1/VALIDATION_REPORT.md)

Download both PDFs into the same directory. Their cross-document links resolve to the corrected companion using PDF named destinations; viewer support varies. Internal references remain clickable. The old release assets remain untouched.

Editable sources: `../jdsa-stage3/` at this correction commit. Existing PDFs, `.bbl` files and ZIP in that source directory are preserved baseline artifacts, not updated outputs. Build the corrected pair with:

```sh
KEEP_BUILD=1 OUTPUT_DIR="$PWD/manuscript/jdsa-attribution-correction-v1" sh manuscript/jdsa-stage3/build.sh
```

The build uses a clean temporary directory and regenerates bibliographies from `references.bib`; it does not use the old `.bbl` files. No new submission ZIP or manuscript release was created.
