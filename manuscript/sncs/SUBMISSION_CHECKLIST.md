# SN Computer Science — Final Submission Checklist

Updated 2026-08-27 (Stage 6). See `docs/SNCS_STAGE6_FINAL_SUBMISSION_FREEZE_2026-08-27.md` for the freeze report.

| Item | Status | Notes |
|---|---|---|
| Springer `sn-jnl` template, `sn-basic` numbered style | ✅ DONE | `\documentclass[pdflatex,sn-basic,Numbered]{sn-jnl}` |
| Structured abstract (Purpose/Methods/Results/Conclusion) | ✅ DONE | Self-contained; no citations |
| Keywords (4–6) | ✅ DONE | 6 keywords |
| Author metadata (name, affiliation, email) | ✅ DONE | Verified against manuscript source |
| ORCID | ✅ DONE | 0000-0003-1934-6282 |
| Corresponding author marker | ✅ DONE | `\author*` |
| Figures | ✅ DONE | 1 figure (`nlp4lp_instantiation_pipeline_v2.png`) |
| Tables | ✅ DONE | Includes matched same-task grounding-baseline table; significance table overfull cleared |
| References | ✅ DONE | 0 undefined citations, 0 unresolved references |
| Declarations (Funding/Competing/Ethics/CRediT/Data/Code/AI) | ✅ DONE | Funding Case A; AI software disclosure includes Claude, Cursor, GitHub Copilot |
| Funding | ✅ DONE | Azure for Students educational credit wording |
| Competing interests | ✅ DONE | None declared |
| Data availability | ✅ DONE | Gated NLP4LP disclosed; derived artifacts on GitHub |
| Code availability | ✅ DONE | GitHub repository linked |
| AI-use disclosure | ✅ DONE | Writing: ChatGPT, Gemini. Software: Claude, Cursor, GitHub Copilot |
| Acknowledgment | ✅ DONE | Mother, Prof. Ioannis Koutis, Anders Borum (Secure ShellFish) |
| Source package | ✅ DONE | `manuscript/sncs/submission_package/` |
| Upload manifest | ✅ DONE | `manuscript/sncs/FINAL_UPLOAD_MANIFEST.md` |
| Clean-room compilation | ✅ DONE | Isolated `/tmp` copy; **24 pages** after Stage-7 compression; 0 undefined citations/references; 0 overfull boxes |
| Same-task grounding baselines | ✅ DONE | Stage-6 matched rerun; typed greedy best on InstReady/Strict |
| Reproducibility verification | ✅ DONE | `docs/SNCS_REPRODUCIBILITY.md` + Stage-6 summary JSON |
| GitHub synchronization | ✅ DONE (after Stage-6 push) | |
| Cover letter status | ⚠️ NOT DRAFTED | Optional; SNCS does not mandate one |
| Supplementary material | ✅ N/A | Evidence in linked GitHub repository |
| Azure-support confirmation | ✅ DONE | Case A |
| STOP EDITING | ✅ YES | Only EM requirements / factual errors / reviewer requests thereafter |

## Cover letter readiness (information only — no letter drafted)

SN Computer Science's public submission guidance does not mandate a cover letter for standard research articles. If the author chooses to write one, use verified facts from `manuscript/sncs/SUBMISSION_METADATA.md`. Do not repurpose `manuscript/cover-letter.tex` (old EAAI-targeted letter) without a full rewrite.
