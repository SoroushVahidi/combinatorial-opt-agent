"""
Simple web UI: type your combinatorial problem in natural language,
get the best-matching problem(s) and their integer programs.
Run: python app.py  then open the URL in your browser.
"""
from pathlib import Path

import gradio as gr

# Add project root so imports work when run as python app.py
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval.search import (
    _load_catalog,
    search,
    format_problem_and_ip,
)

# Load catalog once at startup
CATALOG = _load_catalog()
MODEL = None


def get_model():
    """Lazy-load the embedding model once (so app starts fast)."""
    global MODEL
    if MODEL is None:
        from sentence_transformers import SentenceTransformer
        MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return MODEL


def answer(query: str, top_k: int) -> str:
    if not query or not query.strip():
        return "Please type a short description of your optimization problem (e.g. *minimize cost of opening warehouses and assigning customers*)."
    # Streaming response so users see that work is in progress.
    yield "Searching for matching problems. This may take a few seconds, especially on the first query..."

    model = get_model()
    results = search(
        query.strip(),
        catalog=CATALOG,
        model=model,
        top_k=max(1, min(10, top_k)),
    )
    if not results:
        yield "No matching problems found."
        return

    # Simple confidence hint based on the top score.
    top_score = results[0][1]
    if top_score < 0.4:
        yield (
            "The match is not very confident (top relevance score is below 0.4). "
            "If the result does not look right, try adding more detail to your description."
        )

    out = []
    for i, (problem, score) in enumerate(results, 1):
        out.append(f"### Result {i} (relevance: {score:.3f})")
        out.append(format_problem_and_ip(problem, score=score))
        out.append("---")
    yield "\n".join(out)


def main():
    title = "Combinatorial Optimization Bot"
    description = (
        "Describe your problem in plain English. The bot finds the closest matching "
        "combinatorial optimization problem and shows its **integer program** (variables, "
        "objective, constraints).\n\n"
        "Tip: click an example below to fill the inputs, then press **Submit** to run it."
    )
    iface = gr.Interface(
        fn=answer,
        inputs=[
            gr.Textbox(
                label="Your problem (natural language)",
                placeholder="e.g. I want to minimize cost of opening warehouses and assigning each customer to one warehouse",
                lines=3,
            ),
            gr.Slider(1, 10, value=3, step=1, label="Number of results to show"),
        ],
        outputs=gr.Markdown(label="Answer"),
        title=title,
        description=description,
        examples=[
            ["minimize cost of opening warehouses and assigning customers", 3],
            ["knapsack with weights and values", 2],
            ["schedule jobs on machines to minimize makespan", 3],
            ["shortest path from source to sink", 1],
        ],
    )
    iface.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
