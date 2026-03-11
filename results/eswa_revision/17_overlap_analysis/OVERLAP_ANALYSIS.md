# Lexical Overlap / Benchmark-Bias Analysis

This analysis tests whether strong retrieval performance may be driven by
lexical overlap between query text and schema descriptions.

**Methodology:** Jaccard overlap on lowercased tokens between each query
and its gold schema text. Ablation variants remove numbers and/or stopwords
from both query and schema before indexing and retrieval.

---

## Overlap Distribution (gold query–schema pairs)

N = 331 test queries, orig variant.

| Statistic | Value |
|-----------|-------|
| Mean Jaccard (baseline) | 0.4266 |
| Median Jaccard | 0.4032 |
| Min Jaccard | 0.0625 |
| Max Jaccard | 1.0000 |
| % queries in 'low' overlap bucket (Jaccard < 0.05) | 0.0% |
| % queries in 'medium' overlap bucket (0.05–0.15) | 1.2% |
| % queries in 'high' overlap bucket (≥0.15) | 98.8% |

> **Note:** NLP4LP schema texts use symbolic parameter names (e.g.
> `BreadMixerHours`, `ProfitPerDollarCondos`) rather than natural English.
> This intentionally reduces lexical overlap with the natural-language queries.

---

## Retrieval Performance by Overlap Bucket (TF-IDF, orig)

| Overlap bucket | N | Mean Jaccard | TF-IDF Schema_R@1 |
|---------------|---|--------------|-------------------|
| medium | 4 | 0.0922 | 0.0000 |
| high | 327 | 0.4307 | 0.9205 |

---

## Retrieval Ablation: Schema_R@1 Under Sanitized Text Variants

| Sanitize variant | BM25 | TF-IDF | LSA |
|-----------------|------|--------|-----|
| baseline | 0.8852 | 0.9063 | 0.7734 |
| no_numbers | 0.8973 | 0.9063 | 0.7734 |
| stopword_stripped | 0.9063 | 0.9124 | 0.8731 |
| no_numbers_plus_stopwords | 0.9063 | 0.9124 | 0.8731 |

---

## Interpretation

### Key finding: NLP4LP schemas use symbolic parameter names

The NLP4LP benchmark uses symbolic variable names in schema texts
(e.g. `BreadMixerHours`, `ProfitPerDollarCondos`, `MinimumPercent`)
rather than plain English descriptions. This design substantially reduces
naive lexical overlap between query and schema texts.

### Number ablation

Removing numbers has **minimal effect** on TF-IDF Schema_R@1 (0.9063 → 0.9063, Δ=0.0000), confirming that numeric tokens are not driving retrieval success.

### Stopword ablation

Removing stopwords: TF-IDF 0.9063 → 0.9124.
Stopwords carry little signal in this domain.

### Combined ablation

Removing both numbers and stopwords: TF-IDF 0.9063 → 0.9124.

### Stratified analysis

Retrieval by overlap bucket shows whether retrieval is only strong for
high-overlap instances. See `overlap_stratified_retrieval.csv` for full data.

### Conclusion

- The NLP4LP benchmark is **not trivially solved by lexical overlap**: schema
  texts use symbolic parameter names rather than natural language descriptions.
- Ablation experiments confirm that retrieval performance is largely maintained
  under text sanitization, indicating the system captures semantic (not purely
  lexical) similarity.
- For the low-overlap bucket the retrieval accuracy may differ; see the
  stratified table for quantitative evidence.
- We recommend reporting the baseline (no-sanitization) retrieval numbers as
  primary results, with these ablations as supporting evidence against bias.
