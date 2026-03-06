"""
Retrieval baselines with a unified interface: fit(catalog), rank(query, top_k) -> [(problem_id, score)].
Used for paper-ready comparison: BM25, TF-IDF, SBERT (SentenceTransformer).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


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
        q_tokens = _tokenize_for_bm25(query)
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
        q_vec = self._vectorizer.transform([query])
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
        import numpy as np
        texts = [_searchable_text(p) for p in catalog]
        self._problem_ids = [p.get("id", "") for p in catalog]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        X = self._vectorizer.fit_transform(texts)
        n_comp = min(self._n_components, X.shape[1], X.shape[0] - 1)
        if n_comp < 1:
            n_comp = 1
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self._matrix = self._svd.fit_transform(X)
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._matrix = self._matrix / norms
        return self

    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._matrix is None or self._vectorizer is None or self._svd is None:
            raise RuntimeError("Call fit(catalog) first")
        import numpy as np
        q_vec = self._vectorizer.transform([query])
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
        q_vec = self._model.encode([query], show_progress_bar=False, convert_to_numpy=True)
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
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


def get_baseline(name: str, model_path: str | None = None) -> RetrievalBaseline:
    """Get baseline by name: 'bm25', 'tfidf', 'lsa', 'sbert'."""
    name = (name or "").strip().lower()
    if name == "bm25":
        return BM25Baseline()
    if name == "tfidf":
        return TfidfBaseline()
    if name == "lsa":
        return LSABaseline()
    if name == "sbert":
        return SBERTBaseline(model_path=model_path)
    raise ValueError(f"Unknown baseline: {name!r}. Use bm25, tfidf, lsa, or sbert.")
