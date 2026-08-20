# Baseline staleness audit — raw artifacts

Full record and interpretation: `docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`.

- `nlp4lp_downstream_orig_*.json` / `nlp4lp_downstream_per_query_orig_*.csv` —
  fresh (2026-08-12, commit `0f0b24e`) reruns of all 12 methods on the
  `orig` variant, produced via `run_single_setting()` (the same function
  `training/external/run_full_downstream_benchmark.py` calls), not by
  reusing any committed summary CSV as input.
- `regen_results_all_variants.json` — the same 12 methods across all 3
  variants (`orig`/`noisy`/`short`).
- `significance_vs_fresh_typed_greedy.json` — paired bootstrap (B=1000,
  seed=42) of every method against the *fresh* `tfidf_typed_greedy` number
  (0.7764), not the stale committed one (0.5287).

Reproduction: `PYTHONHASHSEED=0` + the commands in
`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md` §8.

**Do not treat these as replacing `results/eswa_revision/13_tables/postfix_main_metrics.csv`
or the camera-ready manuscript tables** — see
`docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md` for why that decision
is deliberately left to the paper's author.
