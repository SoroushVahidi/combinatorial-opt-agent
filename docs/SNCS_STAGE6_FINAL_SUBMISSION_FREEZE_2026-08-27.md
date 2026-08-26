# SN Computer Science — Stage 6: Final Same-Task Baseline Completion + Submission Freeze

**Date:** 2026-08-27  
**Authoritative manuscript:** `manuscript/sncs/main.tex`  
**Starting HEAD:** `d8bbc0c9fa82326fd9e012d4cfc74b080b43583e` (= `origin/main`)

---

## 1. Starting state

| Field | Value |
|---|---|
| HOSTNAME | al-khwarizmi |
| REPO_PATH | `/home/soroush/combinatorial-opt-agent` |
| BRANCH | main |
| LOCAL/ORIGIN HEAD | `d8bbc0c` (0 ahead / 0 behind) |
| WORKTREE | Stage-5 SHA note pending; unrelated EAAI PDFs present as local noise |
| GIT_LOCKS | none |

Stage-5 SHA self-reference corrected to distinguish scientific/manuscript HEAD `b3ca802...` from actual Stage-5 final repository HEAD `d8bbc0c...`.

---

## 2. Baseline-completion decision

**YES — rerun already-implemented same-task grounding alternatives under the frozen codebase.**

External end-to-end systems (OptimAI / AlphaOPT / ORThought / OptiMUS / ORLM / OptMATH, etc.) remain non-comparable for InstantiationReady / Exact20 (Stage 5 conclusion unchanged).

Predeclared matched set (mechanism diversity; selected **before** scores):

1. typed greedy (production reference)
2. constrained assignment
3. maximum-weight bipartite matching
4. optimization-role repair
5. search-structured grounding
6. semantic-IR repair (optional distinct repair)

Fairness audit: `results/stage6_matched_grounding_baselines_2026-08-27/fairness_audit.json`.

Typed / constrained share the final ratio-aware extractor. Opt-role / search / semantic-IR families retain intrinsic enriched extractors required by their scoring definitions (surgical injection of production tokens would change algorithmic semantics).

---

## 3. Matched baseline results

Artifact: `results/stage6_matched_grounding_baselines_2026-08-27/matched_grounding_baselines_summary.json`  
Runner: `tools/run_stage6_matched_grounding_baselines.py` (`PYTHONHASHSEED=0`, committed gold cache).

| Method | Coverage | TypeMatch | Exact20 | InstReady | Strict |
|---|---:|---:|---:|---:|---:|
| Typed greedy | 0.8886 | 0.8665 | 0.2614 | **0.8006** | **0.7704** |
| Constrained | 0.8531 | 0.8745 | 0.5772 | 0.7613 | 0.7311 |
| Max-weight matching | 0.8436 | 0.8740 | 0.5225 | 0.7432 | 0.7130 |
| Opt-role repair | 0.8436 | 0.8724 | 0.5210 | 0.7372 | 0.7069 |
| Search-structured | 0.8195 | 0.8795 | 0.6397 | 0.7039 | 0.6737 |
| Semantic-IR repair | 0.8131 | 0.8483 | 0.4949 | 0.7160 | 0.6949 |

Typed greedy remains best on InstantiationReady and StrictInstantiationReady (paired bootstrap / McNemar vs each alternative, $B=10{,}000$, seed 42). Some alternatives achieve higher conditional Exact20 by abstaining more (lower Coverage). **Production method not replaced.**

Manuscript table added: `tab:nlp4lp-grounding-baselines`.

---

## 4. Direct-peer literature sanity check

Targeted 2024–2026 search for fixed-catalog optimization-schema retrieval / scalar value-to-slot / InstantiationReady peers (arXiv / publisher surfaces). Related IE/tool-schema work exists, but **no** system exposes a genuinely comparable InstantiationReady / Exact20 fixed-catalog NLP4LP scalar-instantiation evaluation.

**NO_DIRECT_PEER_BASELINE_FOUND**

OptimAI/AlphaOPT/ORThought were not re-run (Stage 5 task/metric incompatibility stands).

---

## 5. KAIS-style comparison-risk classification

| Question | Answer |
|---|---|
| Recent strong literature? | YES (Stage 5 OptimAI/AlphaOPT/ORThought + landscape table) |
| Modern task-level comparison landscape? | YES |
| Direct same-task grounding alternatives? | YES (Stage 6 matched table) |
| End-to-end systems contextual rather than omitted? | YES |
| Absence of external InstantiationReady peer due to task novelty? | YES |

**KAIS_COMPARISON_RISK: MEDIUM**

Residual risk is primarily **reviewer preference** for head-to-head end-to-end numbers despite metric incompatibility — not a missing same-task evaluation defect.

---

## 6. Scientific-regression check

| Check | Result |
|---|---|
| Exact20 / residual 64.7% / significance / retrieval / downstream | unchanged vs Stage-5 authoritative artifacts |
| Structural 60 / 269 / solver 20 / common-18 / OptMATH | unchanged |
| Stage-6 typed greedy matches final ratio-aware metrics.json | YES |
| UNRESOLVED_NUMERIC_MISMATCHES | 0 |
| UNSUPPORTED_CURRENT_CLAIMS | 0 |

---

## 7. Visual / build status

| Check | Result |
|---|---|
| Clean-room build | SUCCESS |
| FINAL_PAGE_COUNT | 41 (+1 vs Stage 5 from matched baseline table) |
| UNDEFINED_CITATIONS | 0 |
| UNRESOLVED_REFERENCES | 0 |
| OVERFULL_BOXES | 0 (Stage-5 ~4.6pt significance-table overfull eliminated) |
| TABLE_OVERLAP / FIGURE_OVERLAP | 0 |
| ADEQUATE_TABLE_ROW_SPACING | YES |

---

## 8. Submission package

`manuscript/sncs/submission_package/` contains only journal-needed files.  
Upload manifest: `manuscript/sncs/FINAL_UPLOAD_MANIFEST.md`.

FINAL_PDF_SHA256: `cdfabcc13db478d109a63d8dadac081ac728dc71e47e8e9b8995e53f7fcdf018`

---

## 9. Exact final manuscript / page count

- Source: `manuscript/sncs/main.tex`
- Clean-room PDF: 41 pages

---

## 10. Final commit

Stage-6 tip: `554484a916ab9a8bb0afdb85b55d49803dd62978` (`origin/main`).

Primary Stage-6 commits:

1. `ba3905e` — matched same-task grounding baseline artifacts/runner
2. `3e6e2dc` — manuscript baseline table + significance layout + submission package
3. `00e89c7` — Stage-6 freeze report + Stage-5 SHA note + reproducibility/manifest
4. follow-ups through `554484a` — upload/freeze pointer stabilization

`FINAL_LOCAL_HEAD` == `FINAL_REMOTE_HEAD` after push.

---

## 11. Remaining non-blocking limitations

1. Reviewers may still prefer end-to-end InstantiationReady-style numbers against OptimAI/AlphaOPT despite incompatibility.
2. Single-benchmark (NLP4LP) dependence.
3. Dense-retriever embeddings remain non-rule-interpretable (disclosed).
4. Common-18 external context remains small by design.
5. Opt-role / search / semantic-IR rows use intrinsic extractors (disclosed in table footnote).

---

## 12. STOP-EDITING recommendation

**STOP_EDITING_RECOMMENDED = YES**

Do not open another general manuscript-improvement stage. Future edits only for:

- Editorial Manager / journal system requirements;
- discovered factual errors;
- editor/reviewer requests.

**RECOMMENDED_SUBMISSION_TAG:** `sncs-submission-2026-08-27`  
(No prior manuscript submission-tag convention found in the repository; tag **not** created.)
