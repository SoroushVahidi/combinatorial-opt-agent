# Failed / Skipped Sources Report

*Generated: 2026-03-16 13:24 UTC*

This file documents every source considered during collection that could **not** be collected or was only partially useful.

---

## DCP-Bench-Open (sample_test.jsonl)

- **URL:** https://github.com/DCP-Bench/DCP-Bench-Open
- **Type:** benchmark
- **License:** Apache-2.0
- **Outcome:** `accessible_not_useful`
- **Notes:** sample_test.jsonl only contains 5 items that are solver code (CPMPy models), not NL problem statements. Full dataset is not available in the public repo sample. Skipped — no NL descriptions in the accessible portion.

---

## MAMO (FreedomIntelligence/Mamo)

- **URL:** https://github.com/FreedomIntelligence/Mamo
- **Type:** benchmark
- **License:** unknown (see repo)
- **Outcome:** `inaccessible`
- **Notes:** HTTP 404 for expected data path benchmark/MathBench.json. Repository structure may have changed since last update. Not collected.

---

## NEOS Guide Case Studies

- **URL:** https://neos-guide.org/
- **Type:** case_studies
- **License:** public (educational)
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed for neos-guide.org in this environment (no address associated with hostname). Source is publicly accessible in normal internet conditions. Cannot collect in sandboxed environment.

---

## MIT OpenCourseWare (15.053 Optimization Methods)

- **URL:** https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/
- **Type:** course_notes
- **License:** CC BY-NC-SA 4.0
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed for ocw.mit.edu in sandboxed environment. Problem sets and lecture notes are publicly available at the URL above under CC BY-NC-SA 4.0 in normal conditions.

---

## GAMS Model Library (web catalog)

- **URL:** https://www.gams.com/latest/gamslib_ml/libhtml/
- **Type:** model_library
- **License:** GAMS proprietary (models publicly described)
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed for www.gams.com in sandboxed environment. Web catalog is publicly viewable. Model list already captured in data/sources/gams_models.json.

---

## OR-Library (J.E. Beasley)

- **URL:** http://people.brunel.ac.uk/~mastjjb/jeb/info.html
- **Type:** test_data_sets
- **License:** public domain / free research use
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed in sandboxed environment. Provides test instance data files (not NL descriptions). Problem type metadata already captured in data/sources/or_library.json.

---

## LibreTexts Mathematics (LP/OR chapters)

- **URL:** https://math.libretexts.org/
- **Type:** open_textbook
- **License:** CC BY-SA 4.0
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed in sandboxed environment. Open textbook platform with LP/OR examples under CC BY-SA 4.0.

---

## AMPL Book Examples

- **URL:** https://ampl.com/resources/the-ampl-book/
- **Type:** textbook
- **License:** free to view online (proprietary book)
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed in sandboxed environment. Book is freely viewable online (PDF). Problem statements would require paraphrasing (copyrighted).

---

## Pyomo Documentation Examples

- **URL:** https://pyomo.readthedocs.io/
- **Type:** documentation
- **License:** BSD-3-Clause
- **Outcome:** `accessible_not_useful`
- **Notes:** Pyomo docs provide solver code, not NL problem statements. Metadata already in data/sources/pyomo_examples.json.

---

## CSPLib (Constraint Satisfaction Problem Library)

- **URL:** https://www.csplib.org/
- **Type:** benchmark_library
- **License:** CC BY 4.0
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed in sandboxed environment. Provides CP problem descriptions; would be useful for constraint-focus.

---

## MiniZinc Challenge Benchmark

- **URL:** https://github.com/MiniZinc/minizinc-benchmarks
- **Type:** benchmark
- **License:** MIT
- **Outcome:** `accessible_not_useful`
- **Notes:** Contains .mzn/.dzn constraint models without natural-language problem descriptions. Not useful for NL-to-formulation tasks.

---

## MIPLIB 2017

- **URL:** https://miplib.zib.de/
- **Type:** benchmark_instances
- **License:** public (see individual instance licenses)
- **Outcome:** `inaccessible`
- **Notes:** DNS resolution failed in sandboxed environment. Provides MPS/LP instance files, not NL descriptions. Metadata in data/sources/miplib.json.

---
