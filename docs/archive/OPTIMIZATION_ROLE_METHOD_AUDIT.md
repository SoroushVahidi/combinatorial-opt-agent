# Audit: Optimization-Aware Downstream Method — Current Repo State

**Date:** 2025-03-08  
**Scope:** Verify whether the requested optimization-role method was implemented, still exists, and has current results. Evidence from repo files only.

---

## 1. Verdict

**The optimization-role-aware downstream method is fully implemented, still present in the current codebase, and has current results in the downstream summary and per-query/output files.**

- **Name in repo:** `optimization_role_repair` (assignment mode); baseline names: `tfidf_optimization_role_repair`, `oracle_optimization_role_repair`, etc.
- **Status:** Implemented and active. Reachable from CLI; all requested doc filenames exist; results exist in `results/paper/nlp4lp_downstream_summary.csv` and in per-query JSON/CSV for orig, noisy, and short with tfidf and oracle.
- **Acceptance rerank:** Separate from this method. It is retrieval-side (schema acceptance reranking); optimization_role_repair is a downstream assignment mode. Both coexist; docs state assignment modes (including optimization_role_repair) run after the chosen retrieval/rerank.

---

## 2. Evidence of implementation in code

**File:** `tools/nlp4lp_downstream_utility.py`

| Component | Location / name | Implements |
|-----------|------------------|------------|
| **CLI assignment mode** | `main()` line 2520: `choices=("typed", "untyped", "constrained", "semantic_ir_repair", "optimization_role_repair")` | New mode in CLI |
| **Effective baseline name** | Lines 2579–2580: `elif args.assignment_mode == "optimization_role_repair": effective_baseline = f"{args.baseline}_optimization_role_repair"` | Output baseline naming |
| **Run-one branch** | Lines 2215–2247: `if assignment_mode == "optimization_role_repair" and expected_scalar:` → `_run_optimization_role_repair(...)` | Entry from pipeline |
| **Constants** | Lines 973–1007: `OPT_ROLE_WORDS` (objective_coeff, unit_profit, unit_cost, capacity_limit, demand_requirement, total_budget, lower_bound, upper_bound, ratio_constraint, percentage_constraint, penalty, setup_cost, etc.) | Optimization-role cues |
| **Patterns** | Lines 1008–1013: `PER_UNIT_PATTERNS`, `OBJECTIVE_PATTERNS`, `BOUND_PATTERNS`, `TOTAL_RESOURCE_PATTERNS`, `RATIO_PATTERNS`, `REQUIREMENT_PATTERNS` | Per-unit, objective, bounds, total resource, ratio, requirement |
| **Weights** | Lines 1023–1045: `OPT_ROLE_WEIGHTS`, `OPT_REPAIR_WEIGHTS` | Scoring and repair |
| **MentionOptIR** | Lines 1049–1066: dataclass with role_tags, operator_tags, unit_tags, fragment_type, is_per_unit, is_total_like, nearby_entity/resource/product_tokens | Mention IR with optimization-role fields |
| **SlotOptIR** | Lines 1069–1084: slot_role_tags, operator_preference, unit_preference, is_objective_like, is_bound_like, is_total_like, is_coefficient_like | Slot-side optimization-role priors |
| **Tagging / classification** | `_context_to_opt_role_tags` (1086), `_classify_fragment_type` (1098), `_detect_opt_unit_tags` (1123) | Objective-vs-constraint/resource/ratio/bound fragment classification |
| **Mention extraction** | `_extract_opt_role_mentions` (1140) | Optimization-aware mention extraction |
| **Slot expansion** | `_slot_opt_role_expansion` (1209) | Slot-side role expansion from param name |
| **Slot IR build** | `_build_slot_opt_irs` (1243) | Build SlotOptIR list |
| **Scoring** | `_score_mention_slot_opt` (1289) | Global assignment scoring with opt-role overlap, fragment compat, etc. |
| **Global assignment** | `_opt_role_global_assignment` (1371) | Maximum-weight bipartite matching (scipy linear_sum_assignment or DP) |
| **Validation** | `_opt_role_validate_one` (1453) | Single assignment validation (type, role, bound plausibility, total-vs-coeff) |
| **Repair** | `_opt_role_validate_and_repair` (1471) | Drop weak assignments; fill unfilled slots from debug list with type-compatible, role-plausible mentions |
| **Orchestrator** | `_run_optimization_role_repair` (1523) | Calls extract → build slots → global assignment → validate_and_repair; returns filled_values, filled_mentions, filled_in_repair |

All of the above are present and used in the single code path triggered by `--assignment-mode optimization_role_repair`.

---

## 3. Assignment modes currently available

From `tools/nlp4lp_downstream_utility.py` line 2520:

- `typed`
- `untyped`
- `constrained`
- `semantic_ir_repair`
- `optimization_role_repair`

All five are in `choices` and are reachable from the CLI. No mode named `optimization_role_ir` or `optrole_semantic_assignment` exists; the implemented name is `optimization_role_repair`.

---

## 4. Optimization-role features found

| Requested feature | Present? | Where |
|-------------------|----------|--------|
| Optimization-role-aware assignment mode | Yes | `optimization_role_repair` in CLI and `run_one` |
| Optimization-role cues for mentions | Yes | `OPT_ROLE_WORDS`, `_context_to_opt_role_tags`, `_extract_opt_role_mentions` |
| Mention IR / Slot IR | Yes | `MentionOptIR`, `SlotOptIR` (names match doc; no separate “MentionIR”/“SlotIR” for this path—those exist for semantic_ir_repair) |
| Slot-side optimization-role priors | Yes | `SlotOptIR.slot_role_tags`, `_slot_opt_role_expansion`, `_build_slot_opt_irs` |
| Objective-vs-constraint (or similar) classification | Yes | `_classify_fragment_type` → "objective", "constraint", "resource", "ratio", "bound" |
| Global assignment with optimization-role scoring | Yes | `_opt_role_global_assignment` using `_score_mention_slot_opt` |
| Deterministic validation-and-repair with optimization logic | Yes | `_opt_role_validate_one`, `_opt_role_validate_and_repair` |
| Per-unit patterns | Yes | `PER_UNIT_PATTERNS` |
| Objective patterns | Yes | `OBJECTIVE_PATTERNS` |
| Bounds | Yes | `BOUND_PATTERNS`, lower_bound/upper_bound in OPT_ROLE_WORDS |
| Total resource | Yes | `TOTAL_RESOURCE_PATTERNS`, total_budget, total_available in OPT_ROLE_WORDS |
| Ratio / requirement | Yes | `RATIO_PATTERNS`, `REQUIREMENT_PATTERNS`; ratio_constraint, percentage_constraint, demand_requirement, etc. |
| capacity_limit, demand_requirement, total_budget, lower_bound, upper_bound, unit_profit, unit_cost, penalty, setup_cost | Yes | All in `OPT_ROLE_WORDS` (lines 974–1007) |

---

## 5. Documentation found

| Requested doc | Exists? | Path | Summary |
|---------------|---------|------|---------|
| NLP4LP_OPTIMIZATION_ROLE_METHOD_IMPLEMENTATION.md | Yes | `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_IMPLEMENTATION.md` | Describes file, entry point, CLI, OPT_ROLE_WORDS, MentionOptIR/SlotOptIR, scoring, validation/repair, baseline names. Matches current code. |
| NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md | Yes | `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md` | Tables for orig TF-IDF and Oracle: typed, constrained, semantic_ir_repair, optimization_role_repair; comparison text. |
| NLP4LP_OPTIMIZATION_ROLE_METHOD_EXAMPLES.md | Yes | `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_EXAMPLES.md` | Five examples: total budget vs per-unit cost, min/max bounds, demand/capacity, ratio/percentage, failing ambiguous coefficient. |

All three requested filenames exist under `docs/` and describe the same method and naming as in the code.

---

## 6. Results/artifacts found

**Summary CSV:**  
`results/paper/nlp4lp_downstream_summary.csv` contains rows for:

- `orig,tfidf_optimization_role_repair` — param_coverage 0.8218, type_match 0.2427, exact20_on_hits 0.2772, instantiation_ready 0.0604
- `orig,oracle_optimization_role_repair` — param_coverage 0.8691, type_match 0.2688, exact20_on_hits 0.2702, instantiation_ready 0.0695
- `noisy,tfidf_optimization_role_repair`
- `short,tfidf_optimization_role_repair`

**Per-query / aggregate JSON:**  
Present in `results/paper/`:

- `nlp4lp_downstream_orig_tfidf_optimization_role_repair.json`
- `nlp4lp_downstream_per_query_orig_tfidf_optimization_role_repair.csv`
- `nlp4lp_downstream_orig_oracle_optimization_role_repair.json`
- `nlp4lp_downstream_per_query_orig_oracle_optimization_role_repair.csv`
- `nlp4lp_downstream_noisy_tfidf_optimization_role_repair.json`
- `nlp4lp_downstream_per_query_noisy_tfidf_optimization_role_repair.csv`
- `nlp4lp_downstream_short_tfidf_optimization_role_repair.json`
- `nlp4lp_downstream_per_query_short_tfidf_optimization_role_repair.csv`

**Types summary:**  
`results/paper/nlp4lp_downstream_types_summary.csv` also references optimization_role_repair (grep hit).

So the method has been run and results are present for orig, noisy, and short with tfidf and for orig with oracle.

---

## 7. What matches the original request

- **Name:** Implemented as `optimization_role_repair` (not optimization_role_ir or optrole_semantic_assignment). Baseline names: `tfidf_optimization_role_repair`, `oracle_optimization_role_repair`, etc.
- **Optimization-aware downstream assignment mode:** Yes; CLI and pipeline branch.
- **Optimization-role cues for mentions:** Yes; OPT_ROLE_WORDS and pattern lists.
- **Mention IR / Slot IR:** Yes; MentionOptIR and SlotOptIR with role tags, fragment_type, is_per_unit, is_total_like, slot priors.
- **Slot-side optimization-role priors:** Yes; SlotOptIR fields and _slot_opt_role_expansion.
- **Objective-vs-constraint (and similar) classification:** Yes; _classify_fragment_type.
- **Global assignment with optimization-role scoring:** Yes; _opt_role_global_assignment + _score_mention_slot_opt.
- **Deterministic validation-and-repair:** Yes; _opt_role_validate_one and _opt_role_validate_and_repair.
- **Patterns:** per-unit, objective, bounds, total resource, ratio, requirement (and capacity_limit, demand_requirement, total_budget, lower_bound, upper_bound, unit_profit, unit_cost, penalty, setup_cost) — all present.
- **Documentation:** All three requested docs exist with matching content.
- **Results:** Summary and per-query outputs exist for the method under the names above.

---

## 8. What is missing or only partial

- **Alternate names:** No mode named `optimization_role_ir` or `optrole_semantic_assignment`; only `optimization_role_repair` is implemented. This is a naming choice, not a missing implementation.
- **Paper “final” tables:** The artifact script’s final/main orig table uses a fixed list of baselines that does not include `*_optimization_role_repair` (see RESULTS_VS_CODE_VERIFICATION.md). So the method is in the full summary and per-query files but not in the main paper table CSV; that is a table subset choice, not removal of the method.
- **Swap repair:** Implementation doc notes “Swap repair: Not implemented as a separate pass”; that remains the case in the code.

---

## 9. Whether this method still exists in the current repo

**Yes.** The optimization-role method exists in the current repo:

- **Code:** All of the logic in §2 is in `tools/nlp4lp_downstream_utility.py` and is on the active path for `--assignment-mode optimization_role_repair`.
- **CLI:** The mode is in the argparse choices and sets `effective_baseline` to `{baseline}_optimization_role_repair`.
- **Docs:** The three requested docs are present and aligned with the code.
- **Results:** The downstream summary and the listed per-query/JSON files in `results/paper/` contain optimization_role_repair results. They are the current trusted outputs for this method (summary is the source of truth; paper tables are a subset and omit it by design).

---

## 10. Appendix: file-by-file evidence list

| Evidence | Path | What was inspected |
|----------|------|--------------------|
| CLI and run_one branch | `tools/nlp4lp_downstream_utility.py` | Lines 2215–2247, 2516–2521, 2577–2580 |
| OPT_ROLE_WORDS, patterns, weights | `tools/nlp4lp_downstream_utility.py` | Lines 973–1045 |
| MentionOptIR, SlotOptIR | `tools/nlp4lp_downstream_utility.py` | Lines 1049–1084 |
| Tagging and classification | `tools/nlp4lp_downstream_utility.py` | _context_to_opt_role_tags, _classify_fragment_type, _detect_opt_unit_tags (1086–1139) |
| Extraction and slot build | `tools/nlp4lp_downstream_utility.py` | _extract_opt_role_mentions (1140), _slot_opt_role_expansion (1209), _build_slot_opt_irs (1243) |
| Scoring and assignment | `tools/nlp4lp_downstream_utility.py` | _score_mention_slot_opt (1289), _opt_role_global_assignment (1371) |
| Validation and repair | `tools/nlp4lp_downstream_utility.py` | _opt_role_validate_one (1453), _opt_role_validate_and_repair (1471), _run_optimization_role_repair (1523) |
| Implementation doc | `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_IMPLEMENTATION.md` | Full file |
| Results doc | `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_RESULTS.md` | Full file |
| Examples doc | `docs/NLP4LP_OPTIMIZATION_ROLE_METHOD_EXAMPLES.md` | First 80 lines |
| Acceptance rerank doc | `docs/NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md` | Lines 1–30 (clarifies acceptance_rerank vs assignment modes) |
| Downstream summary | `results/paper/nlp4lp_downstream_summary.csv` | Rows for tfidf_optimization_role_repair, oracle_optimization_role_repair (orig, noisy, short) |
| Per-query / JSON outputs | `results/paper/` | Files named *optimization_role_repair*.json and *optimization_role_repair*.csv (grep list) |
| GAMSPy reference | `tools/collect_gams_examples_manifest.py` | Line 111: “optimization_role_extraction” (separate usage string) |

---

## Summary answers to the six questions

1. **What exactly from the original request is present?**  
   The full optimization-role-aware downstream method: assignment mode `optimization_role_repair`, optimization-role mention extraction and tagging, MentionOptIR/SlotOptIR, slot-side priors, objective-vs-constraint (and resource/ratio/bound) classification, global assignment with opt-role scoring, deterministic validation-and-repair, and support for per-unit, objective, bounds, total resource, ratio, requirement, and the listed slot/role patterns. All three requested docs exist. Results exist in the summary and per-query files.

2. **What is missing?**  
   No alternate names (optimization_role_ir, optrole_semantic_assignment). Optional “swap repair” pass is not implemented. The method is omitted from the paper “final” table by design (subset of baselines).

3. **What is present under different names?**  
   The requested method is implemented under the name `optimization_role_repair` (and baseline names like `tfidf_optimization_role_repair`). The doc names use “Optimization-Role Method”; the code uses “optimization_role_repair”.

4. **Are there current result files proving it was run?**  
   Yes. `results/paper/nlp4lp_downstream_summary.csv` and the per-query CSV/JSON files listed in §6 prove runs for orig, noisy, and short with tfidf and for orig with oracle.

5. **Are there docs proving it was intended but maybe not completed?**  
   The docs describe an implemented method and match the code. They support that it was intended and completed, not only intended.

6. **If acceptance-rerank exists, is that separate from this optimization-role method or did it replace it?**  
   Acceptance rerank is separate. It is a retrieval-side reranker (schema acceptance); optimization_role_repair is a downstream assignment mode. Both exist; assignment modes (including optimization_role_repair) run after the chosen retrieval/rerank (see docs/NLP4LP_ACCEPTANCE_RERANK_IMPLEMENTATION.md).
