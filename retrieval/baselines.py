"""
Retrieval baselines with a unified interface: fit(catalog), rank(query, top_k) -> [(problem_id, score)].
Used for paper-ready comparison: BM25, TF-IDF, SBERT (SentenceTransformer).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from retrieval.utils import expand_short_query


def _searchable_text(problem: dict) -> str:
    """Same as retrieval.search._searchable_text."""
    parts = [problem.get("name", "")]
    parts.extend(problem.get("aliases") or [])
    parts.append(problem.get("description", ""))
    return " ".join(p for p in parts if p)


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    import re
    return re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()


class RetrievalBaseline(ABC):
    """Unified interface: fit(catalog), rank(query, top_k) -> [(problem_id, score)]."""

    @abstractmethod
    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        """Build index from catalog. Catalog items must have 'id' and searchable name/aliases/description."""
        ...

    @abstractmethod
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return list of (problem_id, score) sorted by score descending."""
        ...


class BM25Baseline(RetrievalBaseline):
    """BM25 retrieval using rank_bm25."""

    def __init__(self) -> None:
        self._problem_ids: list[str] = []
        self._index = None

    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        from rank_bm25 import BM25Okapi
        texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        tokenized = [_tokenize_for_bm25(t) for t in texts]
        self._index = BM25Okapi(tokenized)
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Call fit(catalog) first")
        import numpy as np
        q_tokens = _tokenize_for_bm25(expand_short_query(query))
        scores = self._index.get_scores(q_tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._problem_ids[i], float(scores[i])) for i in idx]


class TfidfBaseline(RetrievalBaseline):
    """TF-IDF + cosine similarity using sklearn."""

    def __init__(self) -> None:
        self._problem_ids: list[str] = []
        self._vectorizer = None
        self._matrix = None

    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(texts)
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._matrix is None or self._vectorizer is None:
            raise RuntimeError("Call fit(catalog) first")
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        q_vec = self._vectorizer.transform([expand_short_query(query)])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._problem_ids[i], float(scores[i])) for i in idx]


class LSABaseline(RetrievalBaseline):
    """LSA/LSI: TF-IDF (1-2 grams) -> TruncatedSVD -> cosine in latent space. CPU-only, no torch."""

    def __init__(self, n_components: int = 256) -> None:
        self._n_components = n_components
        self._problem_ids: list[str] = []
        self._vectorizer = None
        self._svd = None
        self._matrix = None

    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        import warnings
        import numpy as np
        texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = self._vectorizer.fit_transform(texts)
        n_comp = min(self._n_components, X.shape[1], X.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        # TruncatedSVD emits a RuntimeWarning ("invalid value encountered in
        # divide") when computing explained_variance_ratio_ on very small corpora
        # because the total variance is effectively zero.  The decomposition
        # result itself is correct; only the ratio statistic is undefined.
        # Suppress it so callers (and the test suite) are not cluttered.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._matrix = self._svd.fit_transform(X)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._matrix = self._matrix / norms
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._matrix is None or self._vectorizer is None or self._svd is None:
            raise RuntimeError("Call fit(catalog) first")
        import numpy as np
        q_vec = self._vectorizer.transform([expand_short_query(query)])
        q_latent = self._svd.transform(q_vec).flatten()
        q_norm = np.linalg.norm(q_latent) or 1
        q_latent = q_latent / q_norm
        scores = (self._matrix @ q_latent).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._problem_ids[i], float(scores[i])) for i in idx]


class SBERTBaseline(RetrievalBaseline):
    """SentenceTransformer (SBERT) retrieval; same as retrieval/search."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._problem_ids: list[str] = []
        self._embeddings = None
        self._model = None

    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        if self._model_path is None:
            from retrieval.search import _default_model_path
            self._model_path = _default_model_path()
        self._model = SentenceTransformer(self._model_path)
        texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        self._embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # L2 normalize for cosine = dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = self._embeddings / norms
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._embeddings is None or self._model is None:
            raise RuntimeError("Call fit(catalog) first")
        import numpy as np
        q_vec = self._model.encode(
            [expand_short_query(query)],
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
        scores = (self._embeddings @ q_vec.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._problem_ids[i], float(scores[i])) for i in idx]


class E5Baseline(RetrievalBaseline):
    """Dense retrieval using intfloat/e5-base-v2.

    E5 requires asymmetric prefixes: passages are prefixed with "passage: " and
    queries with "query: " before encoding.
    """

    _MODEL_NAME = "intfloat/e5-base-v2"
    _QUERY_PREFIX = "query: "
    _PASSAGE_PREFIX = "passage: "

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self._MODEL_NAME
        self._problem_ids: list[str] = []
        self._embeddings = None
        self._model = None

    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._model = SentenceTransformer(self._model_name)
        raw_texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        # E5 passage prefix
        passages = [self._PASSAGE_PREFIX + t for t in raw_texts]
        self._embeddings = self._model.encode(passages, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = self._embeddings / norms
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._embeddings is None or self._model is None:
            raise RuntimeError("Call fit(catalog) first")
        import numpy as np
        # E5 query prefix
        q_text = self._QUERY_PREFIX + expand_short_query(query)
        q_vec = self._model.encode([q_text], show_progress_bar=False, convert_to_numpy=True)
        q_norm = np.linalg.norm(q_vec) or 1
        q_vec = q_vec / q_norm
        scores = (self._embeddings @ q_vec.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._problem_ids[i], float(scores[i])) for i in idx]


class BGEBaseline(RetrievalBaseline):
    """Dense retrieval using BAAI/bge-large-en-v1.5.

    BGE uses a task-specific query instruction prefix for retrieval:
    "Represent this sentence for searching relevant passages: <query>"
    Passages/schemas are encoded without any prefix.
    """

    _MODEL_NAME = "BAAI/bge-large-en-v1.5"
    _QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self._MODEL_NAME
        self._problem_ids: list[str] = []
        self._embeddings = None
        self._model = None

    def fit(self, catalog: list[dict]) -> RetrievalBaseline:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._model = SentenceTransformer(self._model_name)
        texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        # BGE passages: no prefix
        self._embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = self._embeddings / norms
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._embeddings is None or self._model is None:
            raise RuntimeError("Call fit(catalog) first")
        import numpy as np
        # BGE query instruction prefix
        q_text = self._QUERY_INSTRUCTION + expand_short_query(query)
        q_vec = self._model.encode([q_text], show_progress_bar=False, convert_to_numpy=True)
        q_norm = np.linalg.norm(q_vec) or 1
        q_vec = q_vec / q_norm
        scores = (self._embeddings @ q_vec.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._problem_ids[i], float(scores[i])) for i in idx]


def bm25() -> BM25Baseline:
    """Factory: BM25 baseline. Call fit(catalog) with list of problem dicts."""
    return BM25Baseline()


def tfidf() -> TfidfBaseline:
    """Factory: TF-IDF baseline. Call fit(catalog) with list of problem dicts."""
    return TfidfBaseline()


def sbert(model_path: str | None = None) -> SBERTBaseline:
    """Factory: SBERT baseline. model_path defaults to _default_model_path()."""
    return SBERTBaseline(model_path=model_path)


def e5(model_name: str | None = None) -> E5Baseline:
    """Factory: E5 dense retrieval baseline (intfloat/e5-base-v2).

    Uses asymmetric prefixes: 'passage: ' for schema texts, 'query: ' for queries.
    """
    return E5Baseline(model_name=model_name)


def bge(model_name: str | None = None) -> BGEBaseline:
    """Factory: BGE dense retrieval baseline (BAAI/bge-large-en-v1.5).

    Uses instruction prefix 'Represent this sentence for searching relevant passages: '
    for queries; passages are encoded without any prefix.
    """
    return BGEBaseline(model_name=model_name)


def get_baseline(name: str, model_path: str | None = None) -> RetrievalBaseline:
    """Get baseline by name: 'bm25', 'tfidf', 'lsa', 'sbert', 'e5', 'bge'."""
    name = (name or "").strip().lower()
    if name == "bm25":
        return BM25Baseline()
    if name == "tfidf":
        return TfidfBaseline()
    if name == "lsa":
        return LSABaseline()
    if name == "sbert":
        return SBERTBaseline(model_path=model_path)
    if name == "e5":
        return E5Baseline(model_name=model_path)
    if name == "bge":
        return BGEBaseline(model_name=model_path)
    raise ValueError(f"Unknown baseline: {name!r}. Use bm25, tfidf, lsa, sbert, e5, or bge.")
