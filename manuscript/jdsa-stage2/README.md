# JDSA Stage 2 manuscript

Revised Regular Paper and Online Resource 1, based on Stage 1 report commit `3296f3bd20128077cd94a4eac82f0b5fe73e2946` and immutable `jdsa-evidence-v2.0.0`. Original manuscripts and evidence remain unchanged in their original directories.

- `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf`: 15-page, two-column main paper.
- `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization_Supplement.pdf`: 36-page supplement.
- `main.tex`, `supplement.tex` and the flat included sources: editable manuscript.
- `Fig1.pdf`, `Fig1.tex`, `pipeline_figure.tex`: preserved diagram assets/source.
- [Scientific restructuring and validation reports](../../docs/jdsa_stage2_20261029/SCIENTIFIC_RESTRUCTURING_REPORT.md).

Build both documents with an installed pdfLaTeX/BibTeX environment (validated with TeX Live 2023/Debian). No scientific inference or solver run is required:

```bash
sh manuscript/jdsa-stage2/build.sh
```

The script builds in a fresh temporary directory and copies PDFs only after final citation/reference/overflow checks pass. Set `KEEP_BUILD=1` to retain logs and auxiliary files. The script prints the build directory. A repeatable validation sequence, from the repository root, is:

```bash
KEEP_BUILD=1 sh manuscript/jdsa-stage2/build.sh
# Replace BUILD_DIRECTORY with the directory printed above.
python docs/jdsa_stage2_20261029/validate_stage2.py --build-dir BUILD_DIRECTORY
python docs/jdsa_stage2_20261029/build_crosswalk.py --build-dir BUILD_DIRECTORY
python docs/jdsa_stage2_20261029/measure_narrative.py
```

The evidence verifier needs NumPy 2.3.5 (as specified by the frozen release); conditional derivation uses the Python standard library. Validation writes Stage 2 JSON receipts and regenerates the derived conditional table from authenticated public records. It never changes the immutable release. `ready_conditioned_rows.tex` is the generated audit snippet; the identical rows are inlined in `main.tex` for the table build.

The versioned crosswalk is an overlay: original claim IDs and numbering in the immutable release remain unchanged. Saved-record PASS does not imply full historical reproduction or independently established semantic truth.
