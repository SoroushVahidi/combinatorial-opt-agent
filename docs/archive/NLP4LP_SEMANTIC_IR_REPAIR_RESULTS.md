# NLP4LP Semantic IR + Repair — Results

## Metric comparison (orig queries)

Summary rows read from `results/paper/nlp4lp_downstream_summary.csv` for `variant=orig`.

### TF-IDF on orig

| Metric | typed (greedy) | constrained | semantic_ir_repair |
|--------|----------------|-------------|--------------------|
| schema_R1 | 0.906 | 0.906 | 0.906 |
| param_coverage | 0.822 | 0.772 | 0.778 |
| type_match | 0.231 | 0.195 | **0.254** |
| key_overlap | 0.919 | 0.919 | 0.919 |
| exact20_on_hits | 0.205 | 0.325 | 0.261 |
| instantiation_ready | 0.073 | 0.027 | **0.063** |

### Oracle on orig

| Metric | typed (greedy) | constrained | semantic_ir_repair |
|--------|----------------|-------------|--------------------|
| schema_R1 | 1.0 | 1.0 | 1.0 |
| param_coverage | 0.869 | 0.819 | 0.825 |
| type_match | 0.240 | 0.209 | **0.280** |
| key_overlap | 0.995 | 0.995 | 0.995 |
| exact20_on_hits | 0.204 | 0.321 | 0.258 |
| instantiation_ready | 0.082 | 0.021 | **0.069** |

## Interpretation

- **TypeMatch:** semantic_ir_repair is higher than both typed greedy and constrained for both TF-IDF and Oracle on orig. The semantic IR (role tags, operator/unit alignment) and repair pass help type-aligned assignments.
- **Coverage:** semantic_ir_repair is slightly above constrained (0.778 vs 0.772 for TF-IDF; 0.825 vs 0.819 for Oracle) but below typed greedy. The repair pass fills some unfilled slots with type-compatible fallbacks, improving coverage over the stricter constrained method.
- **InstantiationReady:** **Yes — InstantiationReady improved** versus the constrained method. For TF-IDF orig: 0.063 vs 0.027 (constrained); for Oracle orig: 0.069 vs 0.021 (constrained). So the new method roughly doubles (or more) the fraction of queries that are instantiation-ready compared to constrained, while staying a bit below typed greedy (0.073 and 0.082). The design goal of improving end-task readiness without over-conservative matching is met relative to constrained.
- **Exact20:** semantic_ir_repair sits between typed and constrained (constrained remains best on Exact20; semantic_ir_repair trades a bit of Exact20 for better Coverage, TypeMatch, and InstantiationReady).

## Note on InstantiationReady

**InstantiationReady improved** when moving from the constrained assignment to the semantic_ir_repair assignment on orig, for both TF-IDF and Oracle. It remains below the typed greedy baseline, but the gap to typed is much smaller than for constrained, and the balance across Coverage, TypeMatch, and InstantiationReady is better than constrained.
