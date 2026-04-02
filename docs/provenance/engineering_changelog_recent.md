> **Provenance notice:** Engineering notes only — **not** canonical benchmark numbers. See [`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md) and [`docs/RESULTS_PROVENANCE.md`](../RESULTS_PROVENANCE.md).

# Recent engineering changes (preserved notes)

These items were previously summarized on the repository `README.md` and are kept here so the root stays concise.

## Min/max ordering and bound pairing

- `_is_partial_admissible` rejects partial assignments where a min-slot value exceeds the paired max-slot (e.g. `MinDemand > MaxDemand`), reducing `lower_vs_upper_bound` failures.
- `_slot_stem()` pairs bound slots by quantity stem (`MinDemand`/`MaxDemand` → `"demand"`, etc.).

## Float type-match and token routing

- `_is_type_match("float","int")` treats int tokens as compatible with float slots where appropriate.
- `_expected_type` refinements reduce misclassification of quantity keywords as currency.
- Large non-monetary numbers are less often mis-tagged as currency.
- `_choose_token` priority adjustments for currency vs integer/float slots.

## Short-query retrieval

- `_DOMAIN_EXPANSION_MAP` extended for additional problem families (LP/MIP/ILP, QP, portfolio, bipartite matching, inventory, cutting/packing).

## Structural LP checks

- `formulation/verify.py` flags invalid objective sense and missing variable symbols without a solver.

## Bound-role annotation

- Deterministic min/max operator-phrase recognition, `bound_role` on `MentionOptIR`, range expressions (`between X and Y`), wrong-direction penalties, bound-flip swap repair.

**Verification:** Targeted pytest coverage for these paths; see `tests/` and commit history.
