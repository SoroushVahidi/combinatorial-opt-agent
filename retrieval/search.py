"""
Natural-language search over the combinatorial optimization problem catalog.
Uses sentence-transformers for embeddings and cosine similarity for retrieval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from retrieval.utils import _is_short_query, expand_short_query  # re-exported for callers


def _project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root


def _default_model_path() -> str:
    """Use fine-tuned model if present, else the default."""
    finetuned = _project_root() / "data" / "models" / "retrieval_finetuned" / "final"
    if finetuned.exists():
        return str(finetuned)
    return "sentence-transformers/all-MiniLM-L6-v2"


def _load_catalog() -> list[dict]:
    """Load the problem catalog.

    Prefers ``all_problems_extended.json`` (merged base + custom + enriched)
    when it exists, but guards against a **stale extended catalog**: if the
    extended file contains fewer problems than the base file, it was likely
    built from an older version of the base and is missing most of the catalog.
    In that case a warning is printed and the (larger) base catalog is used
    instead, so that the search is never silently restricted to a tiny subset.

    To fix a stale extended catalog, run::

        python build_extended_catalog.py

    from the project root.
    """
    root = _project_root()
    extended = root / "data" / "processed" / "all_problems_extended.json"
    base = root / "data" / "processed" / "all_problems.json"

    if extended.exists() and base.exists():
        with open(extended, encoding="utf-8") as f:
            ext_catalog = json.load(f)
        with open(base, encoding="utf-8") as f:
            base_catalog = json.load(f)
        if len(ext_catalog) < len(base_catalog):
            import warnings
            warnings.warn(
                f"all_problems_extended.json has {len(ext_catalog)} problems "
                f"but all_problems.json has {len(base_catalog)}. "
                "The extended catalog is stale. "
                "Run `python build_extended_catalog.py` to rebuild it. "
                "Falling back to all_problems.json.",
                # stacklevel=2: points the warning at the _load_catalog() call
                # site (one frame up), which is the actionable location for the
                # developer to investigate.
                stacklevel=2,
            )
            return base_catalog
        return ext_catalog

    path = extended if extended.exists() else base
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _searchable_text(problem: dict, multi_view: bool = False) -> str:
    """Single string used for embedding: name + aliases + description.

    When *multi_view=True* (ablation flag), also appends slot vocabulary
    extracted from the formulation (variable descriptions, constraint descriptions).
    This gives the embedding model more signal for queries that mention
    role/slot words (e.g. "shipped amount", "profit coefficient").
    """
    parts = [problem.get("name", "")]
    parts.extend(problem.get("aliases") or [])
    parts.append(problem.get("description", ""))
    if multi_view:
        # Append slot vocabulary without duplicating the full description
        from retrieval.reranking import _extract_slot_vocabulary
        slot_words = " ".join(_extract_slot_vocabulary(problem))
        if slot_words:
            parts.append(slot_words)
    return " ".join(p for p in parts if p)


def _embed(texts: list[str], model) -> np.ndarray:
    """Embed a list of strings; returns (n, dim) array."""
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def build_index(catalog: list[dict], model, multi_view: bool = False) -> np.ndarray:
    """Build embedding matrix (n_problems, dim) for the catalog.

    Args:
        multi_view: if True, include slot vocabulary in each schema's text
            (ablation flag; see ``_searchable_text``).
    """
    texts = [_searchable_text(p, multi_view=multi_view) for p in catalog]
    return _embed(texts, model)


def search(
    query: str,
    catalog: list[dict] | None = None,
    embeddings: np.ndarray | None = None,
    model=None,
    top_k: int = 3,
    validate: bool = False,
    expand_short_queries: bool = True,
    rerank: bool = False,
    rerank_weight: float = 0.3,
    grounding_rerank: bool = False,
    grounding_lambda: float = 0.15,
    report_ambiguity: bool = False,
    verbose_rerank: bool = False,
) -> list[tuple[dict, float]]:
    """
    Return top_k (problem, score) pairs for the natural-language query.
    score is cosine similarity in [0, 1] (assuming normalized vectors).

    Args:
        validate: if True, each problem dict gets ``"_validation"`` with schema/formulation errors.
        expand_short_queries: if True (default), short keyword queries (≤ 5 words) are
            expanded with domain context before embedding (see ``expand_short_query()``).
        rerank: if True, apply deterministic lexical reranking on top of first-stage scores
            (alias overlap, slot-name overlap, role-cue overlap; see ``retrieval.reranking``).
        rerank_weight: weight for the lexical reranker in the combined score (default 0.3).
        grounding_rerank: if True, apply an optional second-stage grounding-consistency
            rerank after lexical reranking.  Uses quantity-role cue compatibility to
            re-sort the top-k candidates.  Ablation flag: default False.
        grounding_lambda: weight for the grounding-consistency term (default 0.15).
        report_ambiguity: if True, print a warning when top-1 and top-2 scores are very close.
        verbose_rerank: if True, print per-candidate reranking feature scores.
    """
    if catalog is None:
        catalog = _load_catalog()
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from None
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            _device = "cpu"
        model = SentenceTransformer(_default_model_path(), device=_device)

    if embeddings is None:
        embeddings = build_index(catalog, model)

    # Normalize so that cosine similarity is dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_n = embeddings / norms

    effective_query = expand_short_query(query) if expand_short_queries else query
    q_vec = _embed([effective_query], model)
    q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
    scores = (embeddings_n @ q_vec.T).flatten()

    # Retrieve more candidates when reranking so the reranker has a larger pool
    fetch_k = (top_k * 3) if rerank else top_k
    idx = np.argsort(scores)[::-1][:fetch_k]
    results: list[tuple[dict, float]] = [(catalog[i], float(scores[i])) for i in idx]

    # Optional deterministic lexical reranking
    if rerank:
        from retrieval.reranking import rerank as _rerank
        results = _rerank(
            query,
            results,
            retrieval_weight=1.0 - rerank_weight,
            rerank_weight=rerank_weight,
            verbose=verbose_rerank,
        )
        results = results[:top_k]

    # Optional grounding-consistency second-stage rerank
    if grounding_rerank:
        from retrieval.reranking import grounding_rerank as _grounding_rerank
        results = _grounding_rerank(
            query,
            results,
            grounding_lambda=grounding_lambda,
            verbose=verbose_rerank,
        )

    # Optional ambiguity / low-margin detection
    if report_ambiguity:
        from retrieval.reranking import detect_ambiguity
        amb = detect_ambiguity(results)
        if amb is not None and amb.is_ambiguous:
            import sys
            print(
                f"[retrieval] AMBIGUOUS: top schema '{amb.top_id}' (score={amb.top_score:.3f}) "
                f"vs '{amb.second_id}' (score={amb.second_score:.3f}), "
                f"margin={amb.margin:.4f}",
                file=sys.stderr,
            )

    if validate:
        try:
            from formulation.verify import run_all_problem_checks
        except Exception:
            run_all_problem_checks = None
        if run_all_problem_checks:
            for problem, _ in results:
                problem["_validation"] = run_all_problem_checks(problem)

    return results


def format_problem_and_ip(problem: dict, score: float | None = None) -> str:
    """Format one problem and its integer program for display."""
    lines = [
        f"## {problem['name']}",
        "",
        problem.get("description", ""),
        "",
    ]

    formulation = problem.get("formulation") or {}
    variables = formulation.get("variables") or []
    constraints = formulation.get("constraints") or []
    has_formulation = bool(variables or constraints or formulation.get("objective"))

    if not has_formulation:
        lines.append(
            "> **Formulation not yet available.** "
            "This problem is in the catalog but its structured ILP has not been added yet. "
            "The description above may still help you understand the problem structure."
        )
        if score is not None:
            lines.insert(0, f"(relevance: {score:.3f})\n")
        return "\n".join(lines)

    # Collapsible variables section
    lines.append("<details><summary><strong>Variables</strong></summary>")
    lines.append("")
    for v in variables:
        symbol = v.get("symbol", "")
        description = v.get("description", "")
        domain = v.get("domain", "")
        # Try to render symbols/domains with LaTeX if they look like math.
        symbol_tex = f"$ {symbol} $" if symbol else ""
        domain_tex = f"$ {domain} $" if domain else ""
        if symbol_tex or domain_tex:
            lines.append(f"- {symbol_tex}: {description} ({domain_tex})")
        else:
            lines.append(f"- {symbol}: {description} ({domain})")
    lines.append("")
    lines.append("</details>")
    obj = formulation.get("objective") or {}
    lines.extend(
        [
            "",
            "<details><summary><strong>Objective</strong></summary>",
            "",
            f"**Sense:** {obj.get('sense', '')}",
            "",
            obj.get("expression", ""),
            "",
            "</details>",
            "",
            "<details><summary><strong>Constraints</strong></summary>",
            "",
        ]
    )
    for c in problem.get("formulation", {}).get("constraints", []):
        lines.append(f"- {c.get('expression', '')} — {c.get('description', '')}")
    lines.append("")
    lines.append("</details>")
    if problem.get("formulation_latex"):
        latex = problem["formulation_latex"]
        lines.extend(
            [
                "",
                "<details><summary><strong>LaTeX (rendered)</strong></summary>",
                "",
                f"$$ {latex} $$",
                "",
                "</details>",
                "",
            ]
        )
    if problem.get("complexity"):
        lines.extend(["", f"*Complexity: {problem['complexity']}*", ""])
    if score is not None:
        lines.insert(0, f"(relevance: {score:.3f})\n")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m retrieval.search \"<natural language query>\" [top_k]")
        print('Example: python -m retrieval.search "warehouses and customers minimize cost"')
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    catalog = _load_catalog()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers is required: pip install sentence-transformers")
        sys.exit(1)

    try:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        _device = "cpu"
    model = SentenceTransformer(_default_model_path(), device=_device)
    results = search(query, catalog=catalog, model=model, top_k=top_k)

    print(f"Query: \"{query}\"\n")
    for i, (problem, score) in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(format_problem_and_ip(problem, score))
        print()


if __name__ == "__main__":
    main()
