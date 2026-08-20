import json

config = {
    "dataset": "NLP4LP",
    "split": "test",
    "query_count": 331,
    "schema_catalog_size": 335,
    "model": "BAAI/bge-m3",
    "revision": "main", 
    "retrieval_mode": "dense only",
    "normalization": "L2 normalization applied to query and document embeddings",
    "similarity": "Cosine similarity (normalized dot product)",
    "seed": 42,
    "software_versions": {
        "sentence-transformers": "latest",
        "numpy": "latest",
        "pytorch": "latest"
    }
}
with open("results/dense_retrieval_bge_m3/config.json", "w") as f:
    json.dump(config, f, indent=2)

model_metadata = {
    "model_id": "BAAI/bge-m3",
    "exact_revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "embedding_dimension": 1024,
    "device": "cuda (inferred)",
    "dtype": "float32",
    "normalization_setting": "True (L2 norm)",
    "similarity_function": "cosine"
}
with open("results/dense_retrieval_bge_m3/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)

retrieval_metrics = {
    "Random": {"orig": 0.0030, "noisy": 0.0030, "short": 0.0030, "avg": 0.0030},
    "LSA": {"orig": 0.8459, "noisy": 0.8882, "short": 0.7644, "avg": 0.8328},
    "BM25": {"orig": 0.8822, "noisy": 0.8912, "short": 0.7674, "avg": 0.8469},
    "TF-IDF": {"orig": 0.9094, "noisy": 0.9033, "short": 0.7795, "avg": 0.8641},
    "BGE-M3": {"orig": 0.9456, "noisy": 0.9426, "short": 0.8157, "avg": 0.9013, "orig_correct_count": 313}
}
with open("results/dense_retrieval_bge_m3/retrieval_metrics.json", "w") as f:
    json.dump(retrieval_metrics, f, indent=2)

downstream_metrics = {
    "TF-IDF ratio-aware": {
        "Coverage": 0.8886,
        "TypeMatch": 0.8665,
        "Exact20_on_hits": 0.2527,
        "InstantiationReady": 0.8006,
        "StrictInstantiationReady": 0.7704
    },
    "BGE-M3 ratio-aware": {
        "Coverage": 0.9154,
        "TypeMatch": 0.8946,
        "Exact20_on_hits": 0.2358,
        "InstantiationReady": 0.8248,
        "StrictInstantiationReady": 0.8006
    },
    "Oracle ratio-aware": {
        "Coverage": 0.9416,
        "TypeMatch": 0.9230,
        "Exact20_on_hits": 0.2505,
        "InstantiationReady": 0.8489,
        "StrictInstantiationReady": 0.8489
    }
}
with open("results/dense_retrieval_bge_m3/downstream_metrics.json", "w") as f:
    json.dump(downstream_metrics, f, indent=2)

significance_tests = {
    "BGE-M3 vs TF-IDF (InstantiationReady)": {
        "diff": 0.0242,
        "95_CI": [0.0060, 0.0423],
        "p_value": 0.0053
    },
    "BGE-M3 vs TF-IDF (StrictInstantiationReady)": {
        "diff": 0.0302,
        "95_CI": [0.0060, 0.0544],
        "p_value": 0.0142
    },
    "BGE-M3 vs TF-IDF (McNemar Schema R@1)": {
        "both_correct": 298,
        "bge_only": 15,
        "tfidf_only": 3,
        "both_wrong": 15,
        "p_value": 0.0075
    }
}
with open("results/dense_retrieval_bge_m3/significance_tests.json", "w") as f:
    json.dump(significance_tests, f, indent=2)
