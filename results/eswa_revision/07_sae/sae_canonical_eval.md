# SAE Canonical Evaluation

**Date:** 2026-03-10

## Status: PARTIAL — blocked by HF_TOKEN

Slot-aware extraction (SAE) is fully implemented in
`tools/nlp4lp_downstream_utility.py` and integrated into the assignment pipeline.
It can be activated via `--assignment-mode slot_aware_extraction` or by calling
`slot_aware_extraction()` directly.

## Structurally verified results

From `docs/archive_internal_status/publish_now_decision_report.md §2.3` (locally verified, NOT gold-data-dependent):

| Evaluation | Result | Evidence |
|-----------|--------|---------|
| 24 hand-crafted benchmark cases | **24/24 = 100% exact-match** | Locally verified |
| NLP4LP orig queries, slot coverage | **62.9% of schema slots filled** | Locally verified |

## What is blocked

End-to-end SAE evaluation on the canonical 331-query benchmark (TypeMatch, InstReady)
requires the gold parameter values from `udell-lab/NLP4LP`, which needs HF_TOKEN.

## How to run end-to-end

```bash
export HF_TOKEN=hf_...
# Ensure gold cache is populated:
export NLP4LP_GOLD_CACHE=results/paper/nlp4lp_gold_cache.json
python tools/nlp4lp_downstream_utility.py \
    --variant orig \
    --baseline tfidf \
    --assignment-mode slot_aware_extraction
# Or compare all modes at once:
python tools/run_nlp4lp_focused_eval.py --variant orig
```

## Expected structure of comparison

Once HF_TOKEN is available, compare:
- `tfidf_typed_greedy` (baseline)
- `tfidf_optimization_role_repair` (best balance)
- `tfidf_slot_aware_extraction` (if registered in focused_eval)

Metrics: Coverage, TypeMatch, Exact20, InstReady

## For the manuscript

Current claim that can be made honestly:
> "The slot-aware extraction method fills 62.9% of expected schema slots on orig queries
> and achieves 100% exact-match on our hand-crafted benchmark (24/24 cases)."

Cannot yet claim end-to-end TypeMatch or InstReady improvement without gold data run.
