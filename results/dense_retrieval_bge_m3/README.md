# Dense Retrieval Baseline: BAAI/bge-m3

This directory contains the artifacts for evaluating BAAI/bge-m3 as an off-the-shelf dense retriever on the NLP4LP test split.

## Methodology
- Model: BAAI/bge-m3
- Dense embeddings only (1024 dims), L2-normalized.
- Similarity: Cosine similarity.
- Queries: 331 queries from NLP4LP test split (orig, noisy, short).
- Schema catalog: 335 items.

## Results
- Schema R@1 (orig): 0.9456 (vs TF-IDF 0.9094)
- Downstream InstantiationReady: 0.8248 (vs TF-IDF 0.8006, Oracle 0.8489)
- StrictInstantiationReady: 0.8006 (vs TF-IDF 0.7704, Oracle 0.8489)

The results show that dense retrieval materially outperforms TF-IDF on schema identification, bringing downstream readiness closer to the Oracle bound. The central conclusion remains unchanged: schema retrieval can be strong (now even stronger), but there is still a residual grounding bottleneck.
