# Public reproducibility and claim navigation

The final [pre-submission manuscript](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-manuscript-v1.0.0) and the
[immutable evidence release v2.0.0](https://github.com/SoroushVahidi/combinatorial-opt-agent/releases/tag/jdsa-evidence-v2.0.0) are separate versioned artifacts.
The evidence tag points to `209fa982c0a0e2087bd021cbc7a9f57d94256c8f`; the
[exact Hugging Face version](https://huggingface.co/datasets/SoroushVahidi/nl-optimization-instantiation-metrics/tree/f94012872211ab6a828147bb138fbe8ab695b982/releases/jdsa-evidence-v2.0.0) is pinned to
`f94012872211ab6a828147bb138fbe8ab695b982`. Prefer release assets to mutable branches.

The authenticated archive contains 172 files: 170 manifest payloads plus
`release_manifest.json` and `CHECKSUMS.sha256`. Use its own README and verifier from
an extracted temporary directory. Do not overwrite repository records. The frozen
verifier performs 9,737 checks; successful checks authenticate saved-record
consistency, not new inference, original solver reproduction or semantic truth.

## Evidence boundaries

- Saved-record reanalysis recomputes outcomes from retained rows/choices. It cannot
  recover unavailable original predictions or establish missing execution settings.
- Statistical reanalysis uses retained paired inputs and keeps distinct correction
  families separate. It is not an additional experiment.
- Original experimental reproduction requires the original data, versions, prompts,
  models, access and execution environment; these are not all public or recoverable.
- Independent semantic validation requires source-level review beyond numerical
  agreement and author-designed validity records.
- Historical summary-only studies remain explicitly limited when observations,
  source records or execution details are unavailable.

## Claim-to-artifact navigation

This table is derived from the authenticated [claims index](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json) and
[release manifest](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/release_manifest.json). It is navigation, not a new claim
verification. The full index retains all 47 investigation groups, including secondary
studies. Passage locations there reflect the evidence release's manuscript stage;
use the final manuscript release's crosswalk for final section/object locations.

| Investigation | Manifest-backed artifact or classified source mapping | Evidence limitation |
|---|---|---|
| E01: Lexical schema retrieval and ranked-list diagnostics | [results/eswa_revision/01_retrieval/retrieval_results.json](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/results/eswa_revision/01_retrieval/retrieval_results.json); [results/paper/nlp4lp_downstream_per_query_orig_tfidf.csv](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/results/paper/nlp4lp_downstream_per_query_orig_tfidf.csv) | Known-catalog identification; evaluation-source templates and repeated development exposure. Sanitization uses another index. |
| E02: Pretrained dense retrieval | [results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/results/dense_retrieval_bge_m3/nlp4lp_downstream_per_query_orig_bge_m3.csv) | Metadata does not authenticate runtime device, serving weights, hash seed or full environment; no unseen-template claim. |
| E03: Four-arm scalar-grounding benchmark | [results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/results/final_resubmission_method/nlp4lp_downstream_per_query_orig_tfidf.csv) | Gold-assisted participation; conditional Exact20 differs from all-query success; Oracle is not an upper bound on other grounders. |
| E04: Ratio-word extraction ablation | [claims.json: E04](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json) (consult source classifications) | Improves coarse diagnostics; does not imply eight additional numerically correct problems. |
| E05: Matched grounding alternatives | [summaries/results/joint_grounding_2026-10-27/completed_run_reanalysis.json](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/summaries/results/joint_grounding_2026-10-27/completed_run_reanalysis.json) | Candidates, contexts, scores, abstention and decoding differ; no isolated search effect. Descriptive full-success ordering; no new significance test. |
| E06: Residual-error and full-scalar analyses | [claims.json: E06](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json) (consult source classifications) | Stage-of-failure accounting, not adjudicated semantic cause frequencies; the 0.20 rule is absolute for magnitudes below one. |
| E07: Frozen-score factorial assignment | [claims.json: E07](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json) (consult source classifications) | Partially assisted template inventory; ordinal score cardinalization; exploratory, exposed small pilot. No whole-instance equivalence. |
| E08: Cached joint assignment pilot | [results/joint_grounding_2026-10-27/pilot_summary.csv](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/results/joint_grounding_2026-10-27/pilot_summary.csv) | No-op arithmetic branch is not a test of meaningful constraints; source metadata remains assisted; differences are conditional. |
| E09: Exact objective and beam-pruning diagnosis | [claims.json: E09](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json) (consult source classifications) | Correctness-restricted solves are oracle diagnostics. Ties and retained reference criterion do not establish source semantics. |
| E10: Value/type-distinct exact enumeration | [claims.json: E10](https://github.com/SoroushVahidi/combinatorial-opt-agent/blob/209fa982c0a0e2087bd021cbc7a9f57d94256c8f/reproducibility/jdsa-v2.0.0/claims.json) (consult source classifications) | Degenerate empirical interval is not population equivalence. No new run or full-dataset expansion. |

Secondary baseline and validation studies remain in the full claims index; do not
omit their negative results or historical qualifications. Corrected OptMATH micro
and macro definitions and withheld PaMOP objective agreement remain authoritative.
The 7/265 and 7/255 conditional findings, Exact20 conditional denominators, 40-versus-41
constrained distinction, and 49-query frozen-score population must not be conflated.

[Access and redistribution restrictions](DATA_ACCESS_AND_LICENSING.md) apply to
upstream-required material. Sanitized evidence does not publish complete M5 originals.
[Historical SNCS reanalysis documentation](SNCS_REPRODUCIBILITY.md) remains available
within its original scope; older claims are not promoted to current findings.
