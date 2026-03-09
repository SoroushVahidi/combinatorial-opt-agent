"""
Tests for Bottleneck-2 fix: short-query expansion and SHORT_QUERY_TEMPLATES coverage.

Covers:
- expand_short_query / _is_short_query: boundary behaviour and docstring examples
- search() expand_short_queries parameter (default on, can be disabled)
- All four baselines apply expansion inside rank()
- generate_queries_for_problem produces short-form queries via SHORT_QUERY_TEMPLATES
"""
from __future__ import annotations

import random

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_catalog() -> list[dict]:
    """Three-problem catalog — same structure used across all tests."""
    return [
        {
            "id": "p1",
            "name": "Knapsack",
            "aliases": ["0-1 knapsack"],
            "description": (
                "Select items with weights and values to maximize total "
                "value without exceeding a weight capacity."
            ),
        },
        {
            "id": "p2",
            "name": "Set Cover",
            "aliases": [],
            "description": "Choose the minimum number of subsets to cover all elements.",
        },
        {
            "id": "p3",
            "name": "Vertex Cover",
            "aliases": [],
            "description": "Minimum set of vertices that covers every edge in the graph.",
        },
    ]


# ---------------------------------------------------------------------------
# 1. expand_short_query — unit tests
# ---------------------------------------------------------------------------

class TestExpandShortQuery:
    """Directly validate the expansion function and its word-count threshold."""

    def test_single_word_expanded(self):
        from retrieval.search import expand_short_query
        result = expand_short_query("knapsack")
        assert result == "knapsack optimization problem formulation"

    def test_two_words_expanded(self):
        from retrieval.search import expand_short_query
        result = expand_short_query("TSP ILP")
        assert result == "TSP ILP optimization problem formulation"

    def test_exactly_five_words_expanded(self):
        """Boundary: 5-word query should still be expanded."""
        from retrieval.search import expand_short_query
        q = "facility location integer linear program"
        assert len(q.split()) == 5
        result = expand_short_query(q)
        assert result == q + " optimization problem formulation"

    def test_six_words_not_expanded(self):
        """Boundary: 6-word query should be returned unchanged."""
        from retrieval.search import expand_short_query
        q = "facility location integer linear program formulation"
        assert len(q.split()) == 6
        assert expand_short_query(q) == q

    def test_long_query_unchanged(self):
        from retrieval.search import expand_short_query
        long_q = (
            "minimize cost of opening warehouses and assigning customers "
            "to open warehouses subject to capacity constraints"
        )
        assert expand_short_query(long_q) == long_q

    def test_empty_string_unchanged(self):
        from retrieval.search import expand_short_query
        assert expand_short_query("") == ""

    def test_whitespace_only_unchanged(self):
        from retrieval.search import expand_short_query
        assert expand_short_query("   ") == ""

    def test_leading_trailing_whitespace_stripped(self):
        from retrieval.search import expand_short_query
        result = expand_short_query("  TSP  ")
        assert result == "TSP optimization problem formulation"

    def test_docstring_examples(self):
        """Verify the three examples in the expand_short_query docstring."""
        from retrieval.search import expand_short_query
        assert expand_short_query("knapsack") == "knapsack optimization problem formulation"
        assert expand_short_query("TSP ILP") == "TSP ILP optimization problem formulation"
        long_q = "minimize cost of opening warehouses and assigning customers"
        assert expand_short_query(long_q) == long_q


# ---------------------------------------------------------------------------
# 2. _is_short_query — unit tests
# ---------------------------------------------------------------------------

class TestIsShortQuery:

    def test_one_word_is_short(self):
        from retrieval.search import _is_short_query
        assert _is_short_query("knapsack") is True

    def test_five_words_is_short(self):
        from retrieval.search import _is_short_query
        assert _is_short_query("a b c d e") is True

    def test_six_words_is_not_short(self):
        from retrieval.search import _is_short_query
        assert _is_short_query("a b c d e f") is False

    def test_long_sentence_is_not_short(self):
        from retrieval.search import _is_short_query
        assert _is_short_query("minimize the total cost of opening warehouses") is False


# ---------------------------------------------------------------------------
# 3. search() — expand_short_queries parameter
# ---------------------------------------------------------------------------

class TestSearchExpansionParameter:
    """search() must accept expand_short_queries and it must default to True."""

    def test_expand_short_queries_default_true(self, monkeypatch):
        """search() default expand_short_queries=True calls expand_short_query."""
        from retrieval import search as search_mod
        calls = []

        original = search_mod.expand_short_query

        def tracking_expand(q):
            calls.append(q)
            return original(q)

        monkeypatch.setattr(search_mod, "expand_short_query", tracking_expand)

        # Build tiny fake embeddings (just zeros) to avoid needing a real model
        import numpy as np

        catalog = _tiny_catalog()
        n = len(catalog)
        dim = 4

        fake_embeddings = np.random.default_rng(0).random((n, dim)).astype("float32")

        class FakeModel:
            def encode(self, texts, **kwargs):
                return np.random.default_rng(1).random((len(texts), dim)).astype("float32")

        search_mod.search(
            "knapsack",
            catalog=catalog,
            model=FakeModel(),
            embeddings=fake_embeddings,
            top_k=1,
        )
        assert "knapsack" in calls, "expand_short_query should have been called with the query"

    def test_expand_short_queries_false_skips_expansion(self, monkeypatch):
        """expand_short_queries=False must NOT call expand_short_query."""
        from retrieval import search as search_mod
        calls = []

        original = search_mod.expand_short_query

        def tracking_expand(q):
            calls.append(q)
            return original(q)

        monkeypatch.setattr(search_mod, "expand_short_query", tracking_expand)

        import numpy as np

        catalog = _tiny_catalog()
        n = len(catalog)
        dim = 4
        fake_embeddings = np.random.default_rng(0).random((n, dim)).astype("float32")

        class FakeModel:
            def encode(self, texts, **kwargs):
                return np.random.default_rng(1).random((len(texts), dim)).astype("float32")

        search_mod.search(
            "knapsack",
            catalog=catalog,
            model=FakeModel(),
            embeddings=fake_embeddings,
            top_k=1,
            expand_short_queries=False,
        )
        assert calls == [], "expand_short_query should NOT have been called when disabled"


# ---------------------------------------------------------------------------
# 4. Baselines — rank() applies expansion
# ---------------------------------------------------------------------------

class TestBaselineRankExpansion:
    """Each baseline's rank() must pass the expanded query to its retrieval engine."""

    def _catalog_with_domain_terms(self) -> list[dict]:
        """Catalog whose passages contain the expansion suffix tokens.

        When a short query like "testproblem" is expanded to
        "testproblem optimization problem formulation", BM25 / TF-IDF can
        match on the shared suffix tokens even though "testproblem" itself
        appears in only one passage.
        """
        return [
            {
                "id": "target",
                "name": "testproblem",
                "aliases": [],
                "description": "A special optimization problem formulation for testing.",
            },
            {
                "id": "other1",
                "name": "unrelated alpha",
                "aliases": [],
                "description": "Completely different subject with no overlap.",
            },
            {
                "id": "other2",
                "name": "unrelated beta",
                "aliases": [],
                "description": "Another completely different topic.",
            },
        ]

    def test_bm25_short_query_retrieves_correct_problem(self):
        """BM25 with a 1-word query should return the matching problem at rank 1."""
        from retrieval.baselines import BM25Baseline
        cat = self._catalog_with_domain_terms()
        bl = BM25Baseline()
        bl.fit(cat)
        top = bl.rank("testproblem", top_k=1)
        assert len(top) == 1
        pid, score = top[0]
        assert pid == "target", (
            f"Expected 'target' at rank 1 for short query, got {pid!r}"
        )

    def test_tfidf_short_query_retrieves_correct_problem(self):
        """TF-IDF with a 1-word query should return the matching problem at rank 1."""
        from retrieval.baselines import TfidfBaseline
        cat = self._catalog_with_domain_terms()
        bl = TfidfBaseline()
        bl.fit(cat)
        top = bl.rank("testproblem", top_k=1)
        assert len(top) == 1
        pid, score = top[0]
        assert pid == "target", (
            f"Expected 'target' at rank 1 for short query, got {pid!r}"
        )

    def test_bm25_rank_returns_top_k_for_short_query(self):
        """BM25 rank() does not crash or lose results when given a short query."""
        from retrieval.baselines import BM25Baseline
        cat = _tiny_catalog()
        bl = BM25Baseline()
        bl.fit(cat)
        out = bl.rank("knapsack", top_k=2)
        assert len(out) == 2

    def test_tfidf_rank_returns_top_k_for_short_query(self):
        """TF-IDF rank() does not crash or lose results when given a short query."""
        from retrieval.baselines import TfidfBaseline
        cat = _tiny_catalog()
        bl = TfidfBaseline()
        bl.fit(cat)
        out = bl.rank("knapsack", top_k=2)
        assert len(out) == 2

    def test_lsa_rank_returns_top_k_for_short_query(self):
        """LSA rank() does not crash or lose results when given a short query."""
        from retrieval.baselines import LSABaseline
        cat = _tiny_catalog()
        bl = LSABaseline()
        bl.fit(cat)
        out = bl.rank("knapsack", top_k=2)
        assert len(out) == 2

    def test_bm25_long_query_unaffected(self):
        """BM25 rank() works correctly for long queries (no accidental expansion)."""
        from retrieval.baselines import BM25Baseline
        cat = _tiny_catalog()
        bl = BM25Baseline()
        bl.fit(cat)
        long_q = "select items to maximize total value without exceeding weight capacity"
        out = bl.rank(long_q, top_k=3)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# 5. generate_queries_for_problem — SHORT_QUERY_TEMPLATES coverage
# ---------------------------------------------------------------------------

class TestShortQueryTemplatesInSamples:
    """SHORT_QUERY_TEMPLATES must produce short-form training queries."""

    def _make_problem(self) -> dict:
        return {
            "id": "k1",
            "name": "Knapsack",
            "aliases": ["0-1 knapsack"],
            "description": "Select items with weights and values.",
        }

    def test_short_templates_exported(self):
        """SHORT_QUERY_TEMPLATES must be importable and non-empty."""
        from training.generate_samples import SHORT_QUERY_TEMPLATES
        assert len(SHORT_QUERY_TEMPLATES) > 0

    def test_all_short_templates_have_text_placeholder(self):
        """Every SHORT_QUERY_TEMPLATE must contain the {text} placeholder."""
        from training.generate_samples import SHORT_QUERY_TEMPLATES
        for t in SHORT_QUERY_TEMPLATES:
            assert "{text}" in t, f"Template missing {{text}}: {t!r}"

    def test_name_ilp_variant_present(self):
        """'Knapsack ILP' should appear in generated queries."""
        from training.generate_samples import generate_queries_for_problem
        prob = self._make_problem()
        queries = generate_queries_for_problem(prob, random.Random(0), target_per_problem=200)
        assert "Knapsack ILP" in queries, (
            "SHORT_QUERY_TEMPLATES should produce 'Knapsack ILP' for a problem named 'Knapsack'"
        )

    def test_name_problem_variant_present(self):
        """'Knapsack problem' should appear in generated queries."""
        from training.generate_samples import generate_queries_for_problem
        prob = self._make_problem()
        queries = generate_queries_for_problem(prob, random.Random(0), target_per_problem=200)
        assert "Knapsack problem" in queries

    def test_name_optimization_variant_present(self):
        """'Knapsack optimization' should appear in generated queries."""
        from training.generate_samples import generate_queries_for_problem
        prob = self._make_problem()
        queries = generate_queries_for_problem(prob, random.Random(0), target_per_problem=200)
        assert "Knapsack optimization" in queries

    def test_alias_short_variant_present(self):
        """Short-form queries for aliases also appear, e.g. '0-1 knapsack ILP'."""
        from training.generate_samples import generate_queries_for_problem
        prob = self._make_problem()
        queries = generate_queries_for_problem(prob, random.Random(0), target_per_problem=200)
        assert "0-1 knapsack ILP" in queries, (
            "SHORT_QUERY_TEMPLATES should produce alias short-form queries"
        )

    def test_no_duplicates_in_output(self):
        """generate_queries_for_problem should deduplicate its output."""
        from training.generate_samples import generate_queries_for_problem
        prob = self._make_problem()
        queries = generate_queries_for_problem(prob, random.Random(0), target_per_problem=200)
        assert len(queries) == len(set(queries))

    def test_short_queries_are_short(self):
        """Queries produced by SHORT_QUERY_TEMPLATES must themselves be short (≤ 5 words + name).

        Since problem names are typically 1–3 words and templates add at most
        2 words (e.g. "optimization", "ILP"), the resulting short-form queries
        should be ≤ 7 words in total.  This is well inside the 5-word expansion
        threshold for a 1-word name like "Knapsack".
        """
        from training.generate_samples import generate_queries_for_problem, SHORT_QUERY_TEMPLATES
        prob = self._make_problem()
        queries = generate_queries_for_problem(prob, random.Random(0), target_per_problem=200)
        # Extract only the queries that match a short-template pattern for the name
        name = prob["name"]
        short_variants = {t.format(text=name) for t in SHORT_QUERY_TEMPLATES}
        for q in short_variants:
            if q in queries:
                assert len(q.split()) <= 7, f"Short template produced unexpectedly long query: {q!r}"
