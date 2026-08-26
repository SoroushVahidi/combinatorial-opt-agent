# SN Computer Science — Editorial Manager Upload Order

**Paper:** Retrieval-Assisted Instantiation of Natural-Language Optimization Problems  
**Frozen HEAD:** `d2adcb6a884b3bea837a0c72c873764e4fbd2c3f`  
**Source ZIP:** `manuscript/sncs/Retrieval-Assisted-Instantiation_SN-Computer-Science_LaTeX-Source.zip`  
**Reference PDF:** `manuscript/sncs/Retrieval-Assisted-Instantiation_SN-Computer-Science_Manuscript.pdf`

This guide is **outside** the submission ZIP. Do not upload this file to Editorial Manager.

## Intended item types

Upload the five flat source files individually if Editorial Manager requires item classification. The ZIP is a convenient complete source bundle; do **not** treat the ZIP itself as the Manuscript item unless the system explicitly accepts a LaTeX source archive that way.

| # | File | Item type | Notes |
|---|---|---|---|
| 1 | `main.tex` | **Manuscript** | Upload first. Flat package copy (figure path at ZIP root). |
| 2 | `references.bib` | **LaTeX Supporting File** / manuscript-supporting source | Use the exact option provided by Editorial Manager. |
| 3 | `sn-jnl.cls` | **LaTeX Supporting File** | Springer Nature journal class. |
| 4 | `sn-basic.bst` | **LaTeX Supporting File** | Numbered bibliography style. |
| 5 | `nlp4lp_instantiation_pipeline_v2.png` | **Figure** | Same folder level as `main.tex` (no `figures/` subfolder). |
| 6 | `Retrieval-Assisted-Instantiation_SN-Computer-Science_Manuscript.pdf` | **Supplementary Material** | Reference PDF only. Upload only if the system requests or accepts it. |

## Important constraints from Editorial Manager

- All LaTeX source files must be at the **same folder level** (no nested directories).
- Do **not** add `%!TEX TS-program = xelatex` — this manuscript uses **pdflatex**.
- Do **not** put `main.pdf` inside the LaTeX source ZIP.

## Quick verification before upload

- ZIP root contains exactly: `main.tex`, `references.bib`, `sn-jnl.cls`, `sn-basic.bst`, `nlp4lp_instantiation_pipeline_v2.png`
- Separate PDF SHA256: `98d12f1c39d248fc90e5c02736196678313168e8d06072770513e4a989035ff1`
- Page count: 25
