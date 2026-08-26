# SN Computer Science — Stage 7: Compression and Repetition Removal

**Date:** 2026-08-27  
**Starting HEAD:** `069cf3f4543e99fd3e300a838ccf24a028e0658f`  
**Authoritative manuscript:** `manuscript/sncs/main.tex`  
**Mode:** compression only (no new science)

---

## Length summary

| Metric | Baseline | Final | Change |
|---|---:|---:|---:|
| Pages | 41 | 24 | −17 |
| Words (approx., comment-stripped TeX tokens) | 17159 | 8102 | −9057 (−52.8%) |
| Tables | 15 | 15 | 0 |
| Figures | 1 | 1 | 0 |
| Displayed equations | 9 | 9 | 0 |
| Citation keys | 38 | 38 | 0 |

**WORDS_REMOVED:** 9057  
**PERCENT_WORD_REDUCTION:** 52.8%  
**PAGES_REMOVED:** 17

Reduction came from deleting repeated scope/caveat/interpretation prose, not from removing experiments, tables, figures, equations, or references. All Stage-6 scientific numbers were checksum-verified present after compression.

---

## Section reduction table

Approximate section word counts (prose + captions within section spans; not including shared preamble/declarations):

| SECTION | BEFORE_WORDS (approx.) | AFTER_WORDS (approx.) | REDUCTION | MAIN_REPETITION_REMOVED |
|---|---:|---:|---:|---|
| Introduction | ~1450 | ~620 | ~57% | Reformulation / RQ / pipeline restated before contributions |
| Related Work (esp. Position) | ~1100 | ~720 | ~35% | Position restated Binding distinction and deterministic claims |
| Methodology | ~4200 | ~1950 | ~54% | Fixed-catalog / scalar-only / not-complete-model restated in every subsection |
| Experimental Setup | ~2200 | ~950 | ~57% | Prose duplicating Table exp_blocks; metric implications restated after each equation |
| Results (retrieval→error) | ~3800 | ~1450 | ~62% | Open-domain caveats; Exact20 re-explanation; cell-by-cell narration |
| Significance / Strict | ~1100 | ~450 | ~59% | Full CI restatement of table; long Strict motivation |
| Structural / solver | ~1400 | ~520 | ~63% | Repeated restricted-subset warnings + Summary restating all rates |
| External + OptMATH | ~1200 | ~480 | ~60% | Duplicate fairness essay after tables; OptMATH verbosity |
| Limitations + Conclusion + Future | ~1600 | ~550 | ~66% | Limitations restating earlier numbers; Conclusion restating Intro/Results |

---

## Scientific-content checksum

| Item | Preserved? |
|---|---|
| Schema R@1 values | YES |
| Coverage / TypeMatch / Exact20 / InstReady / Strict | YES |
| 64.7% / 214 value-inaccurate | YES |
| Same-task grounding baseline table values | YES |
| Bootstrap / McNemar headline results | YES |
| Structural 60 / 269 / solver 20 | YES |
| Common-18 external outcomes | YES |
| OptMATH extraction validation numbers | YES |
| Runtime 1.09s / 3.29ms | YES |

**SCIENTIFIC_NUMBERS_CHANGED:** 0  
**EXPERIMENTS_REMOVED:** 0  
**REFERENCES_REMOVED:** 0 (temporarily dropped `lhoest2021datasets` then restored)  
**TABLES_REMOVED:** 0  
**FIGURES_REMOVED:** 0

---

## Build / package

| Check | Result |
|---|---|
| Clean-room build | SUCCESS |
| Pages | 24 |
| Undefined citations | 0 |
| Unresolved references | 0 |
| Overfull boxes | 0 |
| FINAL_PDF_SHA256 | `681c6e01691b83187d12d910a61102de4051073f5bd6daf8a315c1116e17a458` |

---

## Stop-editing

Compression complete. Scientific freeze from Stage 6 preserved. Further edits only for EM requirements, factual errors, or reviewer requests.
