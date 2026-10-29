> HISTORICAL BUILD ONLY. This package is retired from current submission use. Use [the corrected Stage 3 build](../jdsa-stage3/README.md). The instructions below document the preserved older archive.

# Build the JDSA manuscript and supplement

Install TeX Live with pdfLaTeX and BibTeX, including amsmath, graphicx, tabularx,
booktabs, algorithm, algorithmicx, algpseudocode, xurl, microtype, xr-hyper,
hyperref, multirow, TikZ, placeins and dependencies of the included Springer class.
Python 3 is needed only to regenerate the source ZIP.

From this directory run:

```sh
sh build.sh
```

The build uses a clean temporary directory and repeated pdfLaTeX/BibTeX passes.
It rejects undefined citations/references, missing characters, duplicate labels
and overfull boxes. Set `KEEP_BUILD=1` to retain the temporary logs. The inherited
hyperref bookmark-level adjustment warning is harmless and is not suppressed.
Outputs are the two descriptive PDF filenames in this directory. `main.pdf`
is an intermediate file in the temporary build directory only.

The checked-in PDFs originate from the authoritative final review; only link
annotations were updated for the canonical paths. A rebuild may change PDF
metadata and serialization. All pages must retain the verified text and rendering.
Main: 22 pages. Supplement: 32 pages.

## Independent editable-source archive

Extract the Submission_Source.zip into an empty directory and run `sh build.sh`
there. Its ten flat files contain all document inputs, with included LaTeX inlined,
the bibliography, Springer class/style, ORCID style, editable figure and figure PDF.
No repository checkout or experimental data is needed. Keep the output PDFs together
for cross-document navigation. To regenerate the archive from this directory:

```sh
python3 package_source.py
```

The modular editable sources are in `source/`, with figure sources/assets in
`source/figures/` and supporting tables in `source/tables/`. Fig1.pdf is used
directly; regenerating its TikZ/standalone source is optional. Original class,
style and figure notices are retained.
