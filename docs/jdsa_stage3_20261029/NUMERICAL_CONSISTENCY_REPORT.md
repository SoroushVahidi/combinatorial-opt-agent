# Numerical and scientific consistency report

**PASS.** The immutable saved-record verifier passed **9,737 checks**, the unchanged Stage 2 validator retargeted to the Stage 3 sources passed **699 checks**, and the Stage 3 preservation extension passed **56 checks**. No model inference, paid service call, solver experiment or unsupported historical reconstruction was performed.

## Evidence and method

Baseline: `60785b0d144365413dd3b44b577bff901043ddd7`. Frozen evidence: `jdsa-evidence-v2.0.0`, commit `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`; HF revision `f94012872211ab6a828147bb138fbe8ab695b982`. The verifier output is byte-identical to the published report, SHA-256 `1bacf16801ae0a3ab10e175a8f41965c6318a2b43d33abee1d66717d2a93350b`.

[NUMERICAL_CONSISTENCY.json](NUMERICAL_CONSISTENCY.json) records the checks and original validator hash. The inherited validator's `baseline` and numeric-literal delta refer to the earlier Stage 1-to-Stage 2 comparison, not to purported scientific edits in Stage 3. The `stage3_extension` separately compares this pass directly with Stage 2. The numerical-token check is a coverage aid, supplemented by per-object comparisons, equation preservation, saved-record recomputation and manual claim review; it is not presented as a complete semantic proof.

Ten scientific endpoint equations and every Stage 2 table/figure object's numeric content pass preservation checks. Equation (11)'s selected-mention notation is the documented notation-only exception. Both independent readiness-conditional implementations agree on ten arms, with five boundary tests. Source hashes and final page/line positions are in the [crosswalk](CLAIM_LOCATION_CROSSWALK_V1.json): all 37 original named objects, 183 numerical passages and 47 investigations remain mapped, plus the preserved Stage 2 derived claim `STAGE2-D01`.

## Protected findings and interpretations

| Finding | Retained check / qualification |
|---|---|
| RQ1 retrieval | TF–IDF 301/331; BGE-M3 313/331; known 335-schema catalog, no unseen-schema inference. |
| Conditional complete agreement | 7/265 = 2.64% of ordinary ready outputs; 7/255 = 2.75% of schema-gated ready outputs. Method-specific denominators and descriptive/dependent-population limits retained. |
| Exact20 | Schema-hit, comparable-value conditional mean; Table 2 counts 291, 301, 303 and 320, not the unconditional 331. Excluding incomparable/abstained slots can change this metric. |
| Complete endpoint | Every required stored-reference scalar must satisfy the historical tolerance. Equal values and tolerant agreement do not establish occurrence identity or semantic fidelity. |
| Residual partition | Mutually exclusive ordered retrieval/coverage/type/value failures; not independently adjudicated causal errors. |
| Six matched procedures | Complete successes 7, 40, 37, 35, 46 and 9 on 331 queries. Constrained assignment's 40 all-key successes differ from 41 eligible-only successes; query 102 explains the boundary. |
| Frozen-score controls | Separate 49 queries / 277 slots; contextual one-to-one exact counts 106→136 and 100→133; full-tolerance remains 5/49 with one gain/one loss. No model-ranking or broad search-superiority conclusion. |
| Statistics | Primary 327 groups, B=10,000, seed 42 and separate 16-test Holm family; historical factorial B=2,000, Python seeded generator/order-statistic intervals and separate 240-test family. Procedures are not pooled or modernized. |
| OptMATH | Micro 23241/28799; 0.8135 is the macro mean. Historical JSON label reversal remains explained. Missing corrected per-instance records prevent independent re-bootstrap of the corrected interval. |
| PaMOP | Aggregate objective agreement remains withheld because saved precision is unresolved. |
| Relationship judgments | 631/651 structurally valid; 197 complete cases, kappa 0.371; 119 determinate unanimous, 77 disputed, one unresolved. Agreement is not semantic accuracy. |
| D1 contrasts | 12 pairs, 24 fixtures, eight categories; ten meaning-changing and two meaning-preserving pairs. Author/AI assistance remains explicit; no independent gold/provider ranking. |
| Historical auxiliary evidence | Legacy 175 versus obsolete 174 distinction, archived subsets, feasibility/unexecuted studies and missing observations retain their original status. |

## Current main-paper object classification

| Object | Classification and limit |
|---|---|
| Table 1: retrieval | Public saved-record reanalysis; historical acquisition/full rankings not reconstructed. |
| Table 2: downstream diagnostics | Public saved-query-metric reanalysis; original per-slot assignments unavailable. |
| Table 3: readiness-conditioned agreement | Deterministic reanalysis derived from authenticated metrics in Stage 2; no new Stage 3 observation. |
| Table 4: residual partition | Public saved-record ordered partition; no semantic adjudication. |
| Table 5: six procedures | Public saved-record reanalysis; whole procedures, not controlled decoder ablations. |
| Table 6: assignment controls | Frozen-score saved-record reanalysis on the selected 49-query population; historical candidate/source restrictions persist. |
| Figure 1 and Algorithm 1 | Inspectable specification and editable diagram; not evidence of complete historical runtime reproduction. |

All supplementary objects retain their immutable classifications through the crosswalk. No unresolved scientific inconsistency requiring a changed measurement was found. Existing evidence weaknesses are listed in the [limitations report](REMAINING_LIMITATIONS_AND_SUBMISSION_RISKS.md).

## Repeat the checks

```sh
KEEP_BUILD=1 sh manuscript/jdsa-stage3/build.sh
python docs/jdsa_stage3_20261029/validate_stage3.py --build-dir <printed-build-directory>
python docs/jdsa_stage3_20261029/build_crosswalk.py --build-dir <printed-build-directory>
```

These checks use the repository's frozen public evidence and installed reanalysis dependencies. They do not require private M5 originals. The editable source ZIP is a manuscript compilation package; the evidence release remains a separate versioned package.
