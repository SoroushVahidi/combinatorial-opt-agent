# NLP4LP Pairwise Feature Analysis

**Analysis mode**: corpus_proxy

**Eval items**: 331

> Pairwise data not available. Showing corpus-level proxy statistics.

## Corpus-Level Proxy Stats

| Statistic | Value |
|-----------|-------|
| avg_numeric_mentions_per_query | 7.559 |
| fraction_with_lower_cue | 0.662 |
| fraction_with_upper_cue | 0.462 |
| fraction_with_both_cues | 0.302 |
| fraction_with_entity_cue | 0.979 |
| fraction_with_multi_numeric | 0.870 |

## Feature Notes

### type_match

Whether the mention's inferred type matches the slot's expected type (1/0).
> Requires slot type annotations; not computable from eval corpus alone.

### operator_cue_match (corpus freq: 2.4%)

Whether operator cues (≤/≥/=) near the mention match the slot role.
> Fraction of eval queries containing operator cues.

### lower_cue_present (corpus freq: 66.2%)

Whether lower-bound language ('at least', 'minimum') precedes the mention.
> Fraction of eval queries containing lower-bound cues.

### upper_cue_present (corpus freq: 46.2%)

Whether upper-bound language ('at most', 'maximum') precedes the mention.
> Fraction of eval queries containing upper-bound cues.

### slot_mention_overlap

Lexical overlap (Jaccard) between slot name tokens and mention context.
> Requires slot names; not computable from eval corpus alone.

### entity_match (corpus freq: 97.9%)

Whether an entity/name near the mention matches entity cues in the slot name.
> Fraction of eval queries containing entity/name cues.

### sentence_proximity

How close (in sentences) the mention is to the slot's description sentence.
> Requires slot-sentence alignment; not computable from eval corpus alone.

