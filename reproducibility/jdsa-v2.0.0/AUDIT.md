# Claim-to-artifact audit

Source of truth: editorial commit `9d8c501c58415e766d0d5773cf7cbb576da66cfa`, canonically published at `d3071ba31eaf7babdf1e7fe234d8527191ed93de`. Only public PDF crosslinks/source URL changed during publication. No manuscript content is edited in this stage.

`claims.json` indexes every numbered table, figure and algorithm, every numeral-containing source paragraph, and all 47 registered investigations. It records exact source lines, hashes, original evidence locators/commits, released inputs and limitations. Dates in source manifests are filesystem observations unless explicitly recorded as run timestamps.

## Numbered object coverage

| Object | Classification | Evidence family | Limit |
|---|---|---|---|
| Table 1 (main) | REANALYZABLE | retrieval | Top-1 IDs/metrics retained. Full ranked lists and complete BGE historical environment are not authenticated. |
| Table 2 (main) | REANALYZABLE | core | Original scalar mappings absent: checks aggregate frozen query metrics, not source-level scalar correctness. |
| Table 3 (main) | REANALYZABLE | core | Original scalar mappings absent: checks aggregate frozen query metrics, not source-level scalar correctness. |
| Table 4 (main) | REANALYZABLE | core | Original scalar mappings absent: checks aggregate frozen query metrics, not source-level scalar correctness. |
| Table 5 (main) | REANALYZABLE | core | Original scalar mappings absent: checks aggregate frozen query metrics, not source-level scalar correctness. |
| Table 6 (main) | REANALYZABLE | stats | 327 frozen groups, seed42, B10000, exact group swaps and Holm16; grouping construction still needs source access. |
| Table 7 (main) | REANALYZABLE | core | Original scalar mappings absent: checks aggregate frozen query metrics, not source-level scalar correctness. |
| Table 8 (main) | REANALYZABLE | common18 | Terminal numerical token/classification reanalysis only. Original code/logs omitted from this redistributable subset. PaMOP precision unresolved, no aggregate objective accuracy. |
| Table 9 (main) | REANALYZABLE | relationships | Label/validity/request counts, kappa, pairwise agreement and16 clusters. Raw response/schema validity checked locally, not independently revalidated from these projections. No human accuracy gold. |
| Figure 1 (main) | SPECIFICATION_ONLY | specification | Editable figure/algorithm/metric definitions are versioned in canonical manuscript and code. They are not empirical observations. |
| Algorithm 1 (main) | SPECIFICATION_ONLY | specification | Editable figure/algorithm/metric definitions are versioned in canonical manuscript and code. They are not empirical observations. |
| Table S1 (supplement) | SPECIFICATION_ONLY | specification | Editable figure/algorithm/metric definitions are versioned in canonical manuscript and code. They are not empirical observations. |
| Table S2 (supplement) | SPECIFICATION_ONLY | specification | Editable figure/algorithm/metric definitions are versioned in canonical manuscript and code. They are not empirical observations. |
| Table S3 (supplement) | SPECIFICATION_ONLY | specification | Editable figure/algorithm/metric definitions are versioned in canonical manuscript and code. They are not empirical observations. |
| Table S4 (supplement) | HISTORICAL_SUMMARY_ONLY | overlap | No authenticated per-condition predictions for sanitization/overlap; use exact source locators/hashes in investigation manifest. |
| Table S5 (supplement) | HISTORICAL_SUMMARY_ONLY | historical_subsets | Original 60/269/20 selection manifests and per-case outcomes unavailable; do not reconstruct memberships or claim disjointness. |
| Table S6 (supplement) | HISTORICAL_SUMMARY_ONLY | historical_subsets | Original 60/269/20 selection manifests and per-case outcomes unavailable; do not reconstruct memberships or claim disjointness. |
| Table S7 (supplement) | REANALYZABLE | factorial | Frozen score/choice/metric projections; source candidate texts, values and slot names excluded. Recomputed bootstrap and Holm240, no new solves. |
| Table S8 (supplement) | MIXED | references | Ten saved correction records and pre-ratio fixed-choice count reanalysis; source-based semantic validation requires gated witnesses. No replacement headline predictions. |
| Table S9 (supplement) | PROVENANCE_ONLY | public_provenance | Historical HF v1.2 remains distinct from the original unreleased M5 and this curated v2 release. |
| Table S10 (supplement) | SPECIFICATION_ONLY | specification | Editable figure/algorithm/metric definitions are versioned in canonical manuscript and code. They are not empirical observations. |
| Table S11 (supplement) | REANALYZABLE | common18 | Terminal numerical token/classification reanalysis only. Original code/logs omitted from this redistributable subset. PaMOP precision unresolved, no aggregate objective accuracy. |
| Table S12 (supplement) | REANALYZABLE | common18 | Terminal numerical token/classification reanalysis only. Original code/logs omitted from this redistributable subset. PaMOP precision unresolved, no aggregate objective accuracy. |
| Table S13 (supplement) | REANALYZABLE | common18 | Terminal numerical token/classification reanalysis only. Original code/logs omitted from this redistributable subset. PaMOP precision unresolved, no aggregate objective accuracy. |
| Table S14 (supplement) | REANALYZABLE | common18 | Terminal numerical token/classification reanalysis only. Original code/logs omitted from this redistributable subset. PaMOP precision unresolved, no aggregate objective accuracy. |
| Table S15 (supplement) | MIXED | g3 | G3 count reanalysis; source key-set semantics need upstream access. G1/G2 are recorded identifiability findings, not newly executable performance experiments. |
| Table S16 (supplement) | REANALYZABLE | relationships | Label/validity/request counts, kappa, pairwise agreement and16 clusters. Raw response/schema validity checked locally, not independently revalidated from these projections. No human accuracy gold. |
| Table S17 (supplement) | REANALYZABLE | relationships | Label/validity/request counts, kappa, pairwise agreement and16 clusters. Raw response/schema validity checked locally, not independently revalidated from these projections. No human accuracy gold. |
| Table S18 (supplement) | REANALYZABLE | relationships | Label/validity/request counts, kappa, pairwise agreement and16 clusters. Raw response/schema validity checked locally, not independently revalidated from these projections. No human accuracy gold. |
| Table S19 (supplement) | REANALYZABLE | legacy | 72 original historical metric files, configuration versions distinct from ratio-aware headline. |
| Table S20 (supplement) | REANALYZABLE | legacy | 72 original historical metric files, configuration versions distinct from ratio-aware headline. |
| Table S21 (supplement) | HISTORICAL_SUMMARY_ONLY | ranked | Missing authenticated full original rankings; do not replace with current predictions. |
| Table S22 (supplement) | REANALYZABLE | pilot | Reaggregate saved numeric outcomes; rebuilding pools or correctness labels requires source and original local cache. |
| Table S23 (supplement) | REANALYZABLE | pilot | Reaggregate saved numeric outcomes; rebuilding pools or correctness labels requires source and original local cache. |
| Table S24 (supplement) | MIXED | optional | Four saved metric rows independently reaggregate; exact historical prompts/provider versions missing. Gemini scalar/Qwen extensions and synthetic harness have no authenticated full results. |
| Table S25 (supplement) | INDEPENDENTLY_INSPECTABLE | d1 | All24 original author/AI-assisted fixtures and12 final pair decisions retained. Deterministic counts; semantic labels are author judgments, not independent blinded ground truth or provider outcomes. |
| Table S26 (supplement) | INDEPENDENTLY_INSPECTABLE | d1 | All24 original author/AI-assisted fixtures and12 final pair decisions retained. Deterministic counts; semantic labels are author judgments, not independent blinded ground truth or provider outcomes. |

## Missing evidence and required qualifications

- The ten core arms are reaggregatable saved query metrics. Original per-slot headline assignments and complete historical rendered prompts/runtime settings were not recovered. No new prediction substitutes for them.
- Conditional Exact20 denominators (291 for most lexical arms;303 BGE;320 Oracle;285 semantic IR) differ from331. Constrained all-key success is40; its eligible-only count41 is not equivalent. Legacy175 differs from an obsolete174 narrative. Final JDSA uses the corrected values.
- 60/269/20-case engineering results remain historical summaries. Exact memberships and original per-case outcomes are missing; their samples cannot be assumed disjoint.
- Common18 supports execution/classification and retained objective-token reanalysis. PaMOP precision ambiguity remains unresolved and its aggregate objective accuracy is withheld. Different models/prompts/solver budgets prevent an equal-treatment leaderboard.
- Relationship final validity is217/217/197 (631/651), with197 complete cases;119 determinate unanimous,77 disagreement,1 unresolved,20 invalid. Raw schema/literal-quote validation passed locally; public label projections alone cannot independently establish that validation. Provider assertions are not human semantic ground truth.
- D1 contains12 author/AI-assisted pairs,24 samples,eight categories.20 labels were directly AI-exposed and12 pairs batch-approved; no independent blinded gold or provider predictions exist.
- Higher-rank retrieval, sanitization, single-run timing, some learned/auxiliary studies and corrected OptMATH uncertainty remain summary-only. The corrected OptMATH micro ratio is23241/28799; macro mean0.8135 is a different estimand. Earlier JSON names reverse micro/macro; do not reuse those names without correction.
- Main availability paragraphs and supplement availability section will need a precise v2-release update during Stage2: the complete M5 bundle remains unreleased; only the curated subset is public after publication. Statements implying all candidate identities/raw responses are publicly released need qualification.
- No newly discovered main-table numerical contradiction is asserted. PASS verifies the stated saved-record computations, not original data acquisition, semantic truth, causal identification or full experimental reproduction.

## Publication boundary

The original candidate ZIP is not suitable for wholesale publication: assignment traces include NLP4LP context tokens/values and reference witnesses. This v2 projection omits those fields, all raw provider prose, private conversations, original third-party source datasets, model weights and private paths. Local originals are preserved. See `PUBLICATION_REVIEW.md` and `LIMITATIONS.md`.

## Stage2 readiness

Safe to begin manuscript clarification based on this bounded release after remote integrity verification. Do not claim full reproduction or independence, remove negative results, or promote blocked studies into outcomes.
