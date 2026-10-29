# Final JDSA editorial manuscript

Stage 3 starts from verified Stage 2 commit `60785b0d144365413dd3b44b577bff901043ddd7`. Historical manuscripts and `jdsa-evidence-v2.0.0` remain unchanged.

- `Retrieval_Assisted_Scalar_Grounding_for_Natural_Language_Optimization.pdf`: final two-column main paper.
- `ESM_1.pdf`: Online Resource 1, with linked contents and supplementary evidence.
- `JDSA_Submission_Source_20261029.zip`: complete flat editable source package and compiled PDFs.
- [Editorial, reference, numerical and visual audit reports](../../docs/jdsa_stage3_20261029/FINAL_REPORT.md).

Build locally with a TeX Live installation providing pdfLaTeX, BibTeX and the standard packages used by the sources:

```bash
sh manuscript/jdsa-stage3/build.sh
```

Inside an extracted source ZIP, run `sh build.sh`. The script works independently of the repository, copies source to a fresh temporary directory, builds both documents and checks final logs before copying PDFs. `KEEP_BUILD=1` retains the build directory for inspecting logs. The build does not load data, run scientific experiments or fetch remote content.

The ZIP includes generated cross-document `.aux` files and bibliography `.bbl` files for journal compilation. These are useful upload inputs; the clean build script regenerates both documents without relying on them. `main.tex` is the article entry point; `supplement.tex` is Online Resource 1; `Fig1.tex` is the editable standalone figure source, not another article. Keep the two PDFs together for cross-document links.

Numerical checks, from the repository root, require the frozen release's NumPy dependency:

```bash
python docs/jdsa_stage3_20261029/validate_stage3.py --build-dir BUILD_DIRECTORY
python docs/jdsa_stage3_20261029/build_crosswalk.py --build-dir BUILD_DIRECTORY
```

Replace `BUILD_DIRECTORY` with the directory printed by a `KEEP_BUILD=1` build. These commands write Stage 3 audit outputs only. The inherited Stage 2 check implementation is unchanged. Reanalysis PASS does not establish full historical experimental reproduction or independent semantic truth. No journal submission is performed by these scripts.
