# SN Computer Science — Final Submission Checklist

Generated 2026-08-27 (Stage 4). See `docs/SNCS_STAGE4_FINAL_SUBMISSION_AUDIT.md` for full detail behind each row.

| Item | Status | Notes |
|---|---|---|
| Springer `sn-jnl` template, `sn-basic` numbered style | ✅ DONE | `\documentclass[pdflatex,sn-basic,Numbered]{sn-jnl}`; verified against current SN Computer Science / Springer Nature guidance |
| Structured abstract (Purpose/Methods/Results/Conclusion) | ✅ DONE | 247 words |
| Keywords (4–6) | ✅ DONE | 6 keywords |
| Author metadata (name, affiliation, email) | ✅ DONE | Verified against manuscript source |
| ORCID | ✅ DONE | 0000-0003-1934-6282 |
| Corresponding author marker | ✅ DONE | `\author*` |
| Figures | ✅ DONE | 1 figure (`nlp4lp_instantiation_pipeline_v2.png`), referenced and rendering correctly |
| Tables | ✅ DONE | 13 tables; one layout defect found and fixed (Table 12 had drifted ~10 pages and mis-wrapped; now renders cleanly 2 pages after its discussion) |
| References | ✅ DONE | 35 cited entries, 0 undefined citations, 0 unresolved references; spot-checked newest/highest-risk entry (DeepOR) against the official AAAI proceedings page — exact match |
| Declarations (Funding/Competing/Ethics/CRediT/Data/Code/AI) | ⚠️ MOSTLY DONE | All present and accurate except Funding, which has one pending author-confirmation TODO (see below) |
| Funding | ⚠️ NEEDS AUTHOR CONFIRMATION | Current wording is conservative and does not claim grant funding; author must confirm the exact Azure credit relationship (see `SUBMISSION_METADATA.md`) |
| Competing interests | ✅ DONE | None declared |
| Data availability | ✅ DONE | Gated NLP4LP disclosed; derived artifacts on GitHub |
| Code availability | ✅ DONE | GitHub repository linked |
| AI-use disclosure | ✅ DONE | Two separate declarations (writing assistance; software development assistance) |
| Acknowledgment | ✅ DONE | Mother, Prof. Ioannis Koutis, Anders Borum (Secure ShellFish, not classified as funding) |
| Source package | ✅ DONE | `manuscript/sncs/submission_package/` (main.tex, references.bib, sn-jnl.cls, sn-basic.bst, figures/, main.pdf) |
| Clean-room compilation | ✅ DONE | Built from an isolated `/tmp` copy of only the submission package; 39 pages, 0 undefined citations/references, 0 missing files |
| Repository URL | ✅ DONE | https://github.com/SoroushVahidi/combinatorial-opt-agent |
| Repository commit SHA (at submission prep time) | See `docs/SNCS_STAGE4_FINAL_SUBMISSION_AUDIT.md` §30 for the exact final SHA | |
| Reproducibility verification | ✅ DONE | 4 deterministic scripts reproduce every corrected number; see `docs/SNCS_REPRODUCIBILITY.md` |
| GitHub synchronization | ✅ DONE (after this stage's push) | See Stage-4 report push verification |
| Cover letter status | ⚠️ NOT DRAFTED | See "Cover letter readiness" note below; SN Computer Science does not mandate one, drafting deferred to author's discretion |
| Supplementary material | ✅ N/A | None planned; all evidence lives in the linked GitHub repository, which is standard practice for this journal and already disclosed in Data/Code availability |
| Azure-support confirmation | ⚠️ **AUTHOR ACTION REQUIRED** | The only remaining blocker; see `SUBMISSION_METADATA.md` Funding section |

## Cover letter readiness (information only — no letter drafted)

SN Computer Science's public submission guidance does not mandate a cover letter for standard research articles (unlike some clinical/medical Springer journals); one is optional and primarily useful to state fit/scope for the editor. If the author chooses to write one, the following verified facts are available to build it from:

- **Title:** Retrieval-Assisted Instantiation of Natural-Language Optimization Problems
- **Journal fit:** SN Computer Science covers applied and foundational computer science, including NLP, information retrieval, and knowledge engineering applications — this paper's schema-retrieval + deterministic-grounding framing fits that scope.
- **Key contribution:** A retrieval-conditioned, inference-time-LLM-free scalar-grounding formulation for natural-language optimization problems, with a diagnostic decomposition showing schema retrieval is comparatively strong while downstream value-to-slot correctness is the dominant residual bottleneck.
- **Prior submission history:** This manuscript was previously prepared (not published) for EAAI/Elsevier and then Knowledge and Information Systems (KAIS)/Springer Nature and Data & Knowledge Engineering (DKE)/Elsevier before being retargeted to SN Computer Science; none of these were completed submissions to the author's knowledge based on repository evidence, and no claim of prior publication or rejection should be stated without the author's own confirmation of what, if anything, was actually submitted externally.
- **Code/data availability:** already covered in Declarations.

**Do not repurpose** `manuscript/cover-letter.tex` (an old EAAI-targeted cover letter) for this submission without a full rewrite — it names a different journal and editor.
