# NLP4LP Optimization-Role Method — Results

## Metric comparison (orig queries)

Data from `results/paper/nlp4lp_downstream_summary.csv` for `variant=orig`.

### TF-IDF on orig

| Metric | typed | constrained | semantic_ir_repair | optimization_role_repair |
|--------|--------|-------------|--------------------|---------------------------|
| schema_R1 | 0.906 | 0.906 | 0.906 | 0.906 |
| param_coverage | 0.822 | 0.772 | 0.778 | **0.822** |
| type_match | 0.231 | 0.195 | 0.254 | **0.243** |
| key_overlap | 0.919 | 0.919 | 0.919 | 0.919 |
| exact20_on_hits | 0.205 | 0.325 | 0.261 | **0.277** |
| instantiation_ready | 0.073 | 0.027 | 0.063 | 0.060 |

### Oracle on orig

| Metric | typed | constrained | semantic_ir_repair | optimization_role_repair |
|--------|--------|-------------|--------------------|---------------------------|
| schema_R1 | 1.0 | 1.0 | 1.0 | 1.0 |
| param_coverage | 0.869 | 0.819 | 0.825 | **0.869** |
| type_match | 0.240 | 0.209 | 0.280 | **0.269** |
| key_overlap | 0.995 | 0.995 | 0.995 | 0.995 |
| exact20_on_hits | 0.204 | 0.321 | 0.258 | **0.270** |
| instantiation_ready | 0.082 | 0.021 | 0.069 | **0.069** |

## Comparison versus typed

- **Coverage:** optimization_role_repair **preserves** coverage (TF-IDF 0.822 vs 0.822; Oracle 0.869 vs 0.869).
- **TypeMatch:** improvement (TF-IDF 0.243 vs 0.231; Oracle 0.269 vs 0.240).
- **Exact20:** improvement (TF-IDF 0.277 vs 0.205; Oracle 0.270 vs 0.204).
- **InstantiationReady:** slightly lower (TF-IDF 0.060 vs 0.073; Oracle 0.069 vs 0.082).

## Comparison versus constrained

- **Coverage:** large improvement (TF-IDF 0.822 vs 0.772; Oracle 0.869 vs 0.819).
- **TypeMatch:** improvement (TF-IDF 0.243 vs 0.195; Oracle 0.269 vs 0.209).
- **Exact20:** lower than constrained (TF-IDF 0.277 vs 0.325; Oracle 0.270 vs 0.321) but still above typed.
- **InstantiationReady:** large improvement (TF-IDF 0.060 vs 0.027; Oracle 0.069 vs 0.021).

## Comparison versus semantic_ir_repair

- **Coverage:** higher (TF-IDF 0.822 vs 0.778; Oracle 0.869 vs 0.825).
- **TypeMatch:** TF-IDF slightly lower (0.243 vs 0.254), Oracle slightly lower (0.269 vs 0.280).
- **Exact20:** higher (TF-IDF 0.277 vs 0.261; Oracle 0.270 vs 0.258).
- **InstantiationReady:** TF-IDF slightly lower (0.060 vs 0.063), Oracle same (0.069 vs 0.069).

## Whether InstantiationReady improved

- **Versus typed:** InstantiationReady is **slightly lower** for optimization_role_repair (TF-IDF 0.060 vs 0.073; Oracle 0.069 vs 0.082).
- **Versus constrained:** InstantiationReady **improved** (TF-IDF 0.060 vs 0.027; Oracle 0.069 vs 0.021).
- **Versus semantic_ir_repair:** Roughly similar (TF-IDF 0.060 vs 0.063; Oracle 0.069 vs 0.069).

**Summary:** The optimization-role method **preserves coverage** (unlike constrained and semantic_ir_repair) and improves TypeMatch and Exact20 over typed, while InstantiationReady is slightly below typed but much above constrained. So we get a better balance of coverage + type + exact match, with a small trade-off on instantiation_ready vs typed.
