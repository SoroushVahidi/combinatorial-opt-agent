# Final Upload Manifest — SN Computer Science

**Date:** 2026-08-27 (Stage 8 post-compression integrity)  
**Source package directory:** `manuscript/sncs/submission_package/`  
**Authoritative manuscript:** `manuscript/sncs/main.tex`

Upload only the files listed below to Editorial Manager (plus the generated PDF if the system requests/accepts it). Do **not** upload audit reports, Stage docs, raw gated NLP4LP data, or EAAI camera-ready materials.

## Files to upload

| FILE | ROLE | REQUIRED/OPTIONAL | SHA256 | SIZE (bytes) |
|---|---|---|---|---:|
| `main.tex` | Manuscript source | REQUIRED | `ee272512dc3c0a667251da5795df665607bbe2fc5b32f8345219b9b277bfcba5` | 65616 |
| `references.bib` | Bibliography | REQUIRED | `ae527bad629cc1b535b9c6a7b0fef250088de9e23342775f085ebc39d3adabad` | 24552 |
| `sn-jnl.cls` | Springer Nature journal class | REQUIRED (if EM does not supply it) | `36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc` | 55857 |
| `sn-basic.bst` | Numbered bibliography style | REQUIRED (if EM does not supply it) | `4b368414cc5593169907933b417aacfdb0ce905866a39bdf55d21aad65e9d46c` | 35515 |
| `figures/nlp4lp_instantiation_pipeline_v2.png` | Pipeline figure | REQUIRED | `9e2f5787a1d73cc5fca475ce7a4ac0f29dae986bf89a6f3cd4d3bb98c8a0b1b5` | 125799 |
| `main.pdf` | Compiled manuscript PDF | OPTIONAL (upload if EM requests/accepts PDF) | `98d12f1c39d248fc90e5c02736196678313168e8d06072770513e4a989035ff1` | 351282 |

## Package hashes

| Item | Value |
|---|---|
| FINAL_PDF_SHA256 | `98d12f1c39d248fc90e5c02736196678313168e8d06072770513e4a989035ff1` |
| SUBMISSION_PACKAGE_SHA256 | `87d0984d9a31da877949e5b7493bd80e1051c410ed12318c261502b452fe5991` |
| FINAL_GIT_COMMIT | `origin/main` tip after Stage-8 push (verify with `git rev-parse origin/main`) |
| CLEAN_ROOM_PAGE_COUNT | 25 |
| UNDEFINED_CITATIONS | 0 |
| UNRESOLVED_REFERENCES | 0 |
| OVERFULL_BOXES | 0 |

## Notes

- Unrelated EAAI PDFs under `results/paper/eaai_camera_ready_figures/` are **not** part of this upload.
- Internal Stage reports under `docs/` are **not** part of this upload.
- Stage 8 restored essential structural/solver definitions and readability; scientific numbers unchanged from Stage 7.
