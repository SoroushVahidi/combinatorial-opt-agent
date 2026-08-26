# SN Computer Science — Stage 8: Post-Compression Integrity and Readability Audit

**Date:** 2026-08-27  
**Starting HEAD:** `7f7b44b687a427c16623a8ff1ce2c6808c72f82e`  
**Authoritative manuscript:** `manuscript/sncs/main.tex`  
**Mode:** integrity / readability restoration only (no new science; no general re-compression)

---

## Length summary

| Metric | Stage 6 | Stage 7 | Stage 8 |
|---|---:|---:|---:|
| Pages | 41 | 24 | 25 |
| Words (comment-stripped TeX tokens; Stage-7 method) | 17159 | 8102 | 8495 |
| Tables | 15 | 15 | 15 |
| Figures | 1 | 1 | 1 |
| Displayed equations | 9 | 9 | 9 |
| Citation keys | 38 | 38 | 38 |

**STAGE6_PAGES:** 41  
**STAGE7_PAGES:** 24  
**STAGE8_PAGES:** 25  

**STAGE6_WORDS:** 17159  
**STAGE7_WORDS:** 8102  
**STAGE8_WORDS:** 8495  

**ESSENTIAL_EXPLANATIONS_RESTORED:** 4  
**TRANSITIONS_RESTORED:** 5  
**DEFINITIONS_RESTORED_OR_CLARIFIED:** 6  
**REPORT_LIKE_SENTENCES_REWRITTEN:** 8  

Page growth (+1) is from restored definitions and journal prose, not new experiments. Target range 24–26; clarity preferred over forcing 24.

---

## Preflight

| Check | Result |
|---|---|
| Expected starting HEAD | `7f7b44b687a427c16623a8ff1ce2c6808c72f82e` |
| LOCAL_HEAD at start | match |
| ORIGIN_MAIN at start | match |
| AHEAD_BEHIND | 0 / 0 |
| Unrelated EAAI PDFs | untouched |

---

## Semantic diff audit (Stage 6 → Stage 7), with Stage-8 actions

Classification legend: PURE_REPETITION, SAFE_COMPRESSION, MOVED_TO_CROSS_REFERENCE, UNIQUE_BUT_NONESSENTIAL_DETAIL, UNIQUE_ESSENTIAL_EXPLANATION_LOST, TRANSITION_LOST, DEFINITION_WEAKENED, REPRODUCIBILITY_DETAIL_LOST, LIMITATION_WEAKENED, CLAIM_SCOPE_WEAKENED.

| SECTION | DELETED_CONTENT_SUMMARY | CLASSIFICATION | RESTORE_ANYTHING | WHY |
|---|---|---|---|---|
| Introduction | Repeated reformulation / RQ / pipeline before contributions | PURE_REPETITION / SAFE_COMPRESSION | No | Concise Intro still answers problem, gap, RQ, method, novelty, findings, contributions |
| Related Work — Binding | Minor tightening of Gao complementarity | SAFE_COMPRESSION | No | Binding vs Position still complementary |
| Related Work — Position | Restatement of deterministic / InstantiationReady claims | PURE_REPETITION | Minor prose only | Softened slash-heavy InstantiationReady/Exact20; no expansion |
| Method — Problem / Retrieval | Repeated fixed-catalog / not-complete-model caveats | PURE_REPETITION / SAFE_COMPRESSION | No | Schema, catalog, oracle, eligible slots remain defined |
| Method — Instantiation | Verbose restatements of scope after each step | SAFE_COMPRESSION | No | Typed greedy, no-reuse, ratio-aware, Algorithm, figure intact |
| Method — Design | Duplicate oracle / predicted-schema caveats | SAFE_COMPRESSION | No | Design choices remain clear |
| Experimental Setup | Long block duplicating Table exp_blocks; metric implications after each equation | SAFE_COMPRESSION / MOVED_TO_CROSS_REFERENCE | Yes (bootstrap) | Restored explicit $B=10{,}000$, seed $42$ in Setup so significance is not repository-only |
| Retrieval / Downstream / Grounding / Error / Significance | Cell-by-cell narration; open-domain caveats; Exact20 re-explanation | SAFE_COMPRESSION / TRANSITION_LOST (mild) | Yes (openings) | Rewrote report-like “Table X reports…” openings into takeaway-led prose |
| Runtime | Stage-6 asymptotic $O(\mathrm{nnz}(q)+M)$ retained then oversimplified | DEFINITION_WEAKENED (correctness) | Yes | Replaced misleading asymptotic with implementation-faithful sparse cosine scoring description |
| Structural / solver | Full metric definitions → “runtime stages as named”; paragraph shorthand “Table X: …” | UNIQUE_ESSENTIAL_EXPLANATION_LOST / DEFINITION_WEAKENED / REPORT-LIKE | Yes | Restored concise Boolean definitions and journal prose for 60 / 269 / 20 subsets |
| External common-18 | Fairness essay → “(i)…(v)…” dump; outcome metrics “as defined in the harness” | DEFINITION_WEAKENED / REPORT-LIKE | Yes | Two short fairness paragraphs; caption defines Parse / Executable / Feasible / Objective agreement |
| OptMATH validation | Verbosity cut | SAFE_COMPRESSION | No | Numbers and interpretation remain |
| Limitations | Consequence sentences compressed toward labels | LIMITATION_WEAKENED | Yes | Restored short consequence clauses without restating distant numbers |
| Conclusion / Future | Length cut | SAFE_COMPRESSION | No | Still states study, findings, importance, next direction |

**ESSENTIAL_CONTENT_LOST_IN_STAGE7:** YES (structural/solver definitions and a few setup/external clarifications) — restored in Stage 8.  
**CLAIM_SCOPE_WEAKENED items requiring restore:** none beyond the structural/external wording above.

---

## Actual restorations (rationale)

1. **Structural/solver metric definitions** — Stage 7’s “runtime stages as named” was insufficient; restored concise Boolean definitions for structural validity, instantiation completeness, executable, solver success, feasible, objective produced.
2. **60 / 269 / 20 subset narratives** — Rewrote shorthand table annotations into research prose: what each subset is, why it exists, how selected, metrics, inconclusive 0.80 vs 0.75, supported conclusion.
3. **TF–IDF complexity wording** — Removed oversimplified $O(\mathrm{nnz}(q)+M)$; describe fit-once + per-query sparse cosine against fixed $M=335$ catalog (sklearn-faithful).
4. **External fairness paragraph** — Split cognitively dense `(i)–(v)` list into two short journal paragraphs; preserved all fairness qualifications.
5. **External outcomes caption** — Define Parse / Executable / Feasible / Objective agreement so metrics are not harness-only.
6. **Setup bootstrap convention** — State $B=10{,}000$, seed $42$, and McNemar cross-ref in Experimental Setup.
7. **Results openings** — Downstream, grounding baselines, error taxonomy, significance, and retrieval closing rewritten away from table-annotation tone.
8. **Limitations** — Each major limitation retains a consequence clause (open-domain untested; cross-style untested; gated rerun constraint; scalars ≠ full model; etc.).
9. **Position of Present Work** — Minor InstantiationReady / Exact20 wording into normal prose with Setup cross-ref.

---

## Journal-prose / method / setup checks

| Check | Result |
|---|---|
| Journal prose (no notes/audit/README tone where flagged) | PASS after rewrites |
| Method self-contained (schema, catalog, metrics, oracle, typed greedy, ratio-aware, Strict) | PASS |
| Experimental design understandable without GitHub | PASS (commands may remain in repo) |
| Results takeaways per major experiment | PASS |
| Structural/solver clarity | PASS |
| External baseline readability | PASS |
| Limitations sufficient | PASS |
| TF–IDF complexity wording | PASS |
| Captions not miniature methods | PASS |
| Cross-refs resolve to real definitions | PASS |
| Scientific numbers changed | 0 |
| Experiments removed | 0 |
| Unsupported claims introduced | 0 |

---

## Readability tests

| Read | Question | Result |
|---|---|---|
| EDITOR | Contribution/scope clear from title+abstract+intro+conclusion? | **PASS** |
| REVIEWER | Technically qualified reviewer can follow experiments/metrics without GitHub? | **PASS** |
| REPRODUCIBILITY | Conceptual reproduction from Method+Setup; GitHub for implementation only? | **PASS** |

---

## Build / visual validation

| Check | Result |
|---|---|
| Normal build | SUCCESS |
| Clean-room build | SUCCESS |
| Pages rendered | 25 / 25 |
| UNDEFINED_CITATIONS | 0 |
| UNRESOLVED_REFERENCES | 0 |
| OVERFULL_BOXES | 0 |
| TABLE_OVERLAP | 0 |
| FIGURE_OVERLAP | 0 |
| SERIOUS_VISUAL_PROBLEMS | 0 |
| TABLE_ROW_SPACING_PASS | YES |
| FINAL_PDF_SHA256 | `98d12f1c39d248fc90e5c02736196678313168e8d06072770513e4a989035ff1` |

Note: last PDF page is a near-empty bibliography widow (ref.\ [38] only). Acceptable within 24–26; not forced back to 24 by cutting definitions.

---

## Scientific freeze checksum

All Stage-6/7 scientific values checked present and unchanged (retrieval, Coverage/TypeMatch/Exact20/InstReady/Strict, 64.7%, same-task baselines, bootstrap/McNemar headlines, 60/269/20 structural/solver rates, common-18, OptMATH, runtime 1.09s / 3.29ms).

**SCIENTIFIC_NUMBERS_CHANGED:** 0  
**FACTUAL_ERRORS_FOUND:** 0 (TF–IDF complexity was wording accuracy, not a numeric result)

---

## Package

Submission package synced from authoritative `manuscript/sncs/main.tex`; EAAI camera-ready PDFs untouched.
