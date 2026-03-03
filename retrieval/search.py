"""
Natural-language search over the combinatorial optimization problem catalog.
Uses sentence-transformers for embeddings and cosine similarity for retrieval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


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
    root = _project_root()
    extended = root / "data" / "processed" / "all_problems_extended.json"
    base = root / "data" / "processed" / "all_problems.json"

    path = extended if extended.exists() else base
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _searchable_text(problem: dict) -> str:
    """Single string used for embedding: name + aliases + description."""
    parts = [problem.get("name", "")]
    parts.extend(problem.get("aliases") or [])
    parts.append(problem.get("description", ""))
    return " ".join(p for p in parts if p)


def _embed(texts: list[str], model) -> np.ndarray:
    """Embed a list of strings; returns (n, dim) array."""
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def build_index(catalog: list[dict], model) -> np.ndarray:
    """Build embedding matrix (n_problems, dim) for the catalog."""
    texts = [_searchable_text(p) for p in catalog]
    return _embed(texts, model)


def search(
    query: str,
    catalog: list[dict] | None = None,
    embeddings: np.ndarray | None = None,
    model=None,
    top_k: int = 3,
) -> list[tuple[dict, float]]:
    """
    Return top_k (problem, score) pairs for the natural-language query.
    score is cosine similarity in [0, 1] (assuming normalized vectors).
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
        model = SentenceTransformer(_default_model_path())

    if embeddings is None:
        embeddings = build_index(catalog, model)

    # Normalize so that cosine similarity is dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_n = embeddings / norms

    q_vec = _embed([query], model)
    q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
    scores = (embeddings_n @ q_vec.T).flatten()

    # Top-k indices
    idx = np.argsort(scores)[::-1][:top_k]
    return [(catalog[i], float(scores[i])) for i in idx]


def format_problem_and_ip(problem: dict, score: float | None = None) -> str:
    """Format one problem and its integer program for display."""
    lines = [
        f"## {problem['name']}",
        "",
        problem.get("description", ""),
        "",
    ]

    # Collapsible variables section
    lines.append("<details><summary><strong>Variables</strong></summary>")
    lines.append("")
    for v in problem.get("formulation", {}).get("variables", []):
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
    obj = problem.get("formulation", {}).get("objective", {})
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

    model = SentenceTransformer(_default_model_path())
    results = search(query, catalog=catalog, model=model, top_k=top_k)

    print(f"Query: \"{query}\"\n")
    for i, (problem, score) in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(format_problem_and_ip(problem, score))
        print()


if __name__ == "__main__":
    main()
