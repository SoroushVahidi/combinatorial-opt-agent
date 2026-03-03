"""
Simple web UI: type your combinatorial problem in natural language,
get the best-matching problem(s) and their integer programs.
Run: python app.py  then open the URL in your browser.
"""
from pathlib import Path
import os

# Disable Gradio analytics to avoid extra thread (fixes "can't start new thread" on HPC)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# Disable Hugging Face Xet backend (spawns worker threads -> "Resource temporarily unavailable" on login nodes)
os.environ["HF_HUB_DISABLE_XET"] = "1"

# On Wulver, FastAPI/Starlette run sync route handlers in a thread pool -> "can't start new thread".
# Patch run_in_threadpool to run in the current thread instead (no new threads).
import starlette.concurrency as _starlette_concurrency
_orig_run_in_threadpool = _starlette_concurrency.run_in_threadpool
async def _run_in_threadpool_same_thread(func, *args, **kwargs):
    return func(*args, **kwargs)  # run in event-loop thread; blocks during request
_starlette_concurrency.run_in_threadpool = _run_in_threadpool_same_thread

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
    """Lazy-load the embedding model once (so app starts fast). Uses fine-tuned model if present."""
    global MODEL
    if MODEL is None:
        from sentence_transformers import SentenceTransformer
        from retrieval.search import _default_model_path
        MODEL = SentenceTransformer(_default_model_path())
    return MODEL


async def answer(query: str, top_k: int) -> str:
    # Async so FastAPI runs this in the event loop (no thread pool).
    # Avoids "can't start new thread" on Wulver when user clicks Submit.
    if not query or not query.strip():
        return "Please type a short description of your optimization problem (e.g. *minimize cost of opening warehouses and assigning customers*)."
    model = get_model()
    results = search(query.strip(), catalog=CATALOG, model=model, top_k=max(1, min(10, top_k)))
    if not results:
        return "No matching problems found."
    out = []
    for i, (problem, score) in enumerate(results, 1):
        out.append(f"### Result {i} (relevance: {score:.3f})")
        out.append(format_problem_and_ip(problem, score=None))
        out.append("---")
    return "\n".join(out)


def main():
    title = "Combinatorial Optimization Bot"
    description = (
        "Describe your problem in plain English. The bot finds the closest matching "
        "combinatorial optimization problem and shows its **integer program** (variables, "
        "objective, constraints). Use the **Flag** button to mark bad or surprising results; "
        "flagged examples are saved for later review."
    )
    iface = gr.Interface(
        fn=answer,
        inputs=[
            gr.Textbox(
                label="Your problem (natural language)",
                placeholder=(
                    "e.g. I manage a network of potential warehouse locations and customer zones with known "
                    "demands. Each warehouse has an opening cost and capacity, and shipping costs to each "
                    "customer. I want to choose which warehouses to open and how to assign each customer to "
                    "one open warehouse to minimize total opening plus shipping cost."
                ),
                lines=5,
            ),
            gr.Slider(1, 10, value=3, step=1, label="Number of results to show"),
        ],
        outputs=gr.Markdown(label="Answer"),
        title=title,
        description=description,
        examples=[
            [
                "I manage a logistics network with candidate warehouse locations and a set of customer zones "
                "with known demands. Each warehouse has a fixed opening cost and limited capacity, and there is "
                "a shipping cost for serving each customer from each warehouse. I want to decide which warehouses "
                "to open and how to assign each customer to exactly one open warehouse so that total opening plus "
                "shipping cost is minimized without violating capacities.",
                3,
            ],
            [
                "I have a list of items, each with a weight and a profit, and a single knapsack with a maximum "
                "weight capacity. I need to choose a subset of items to put in the knapsack so that the total "
                "weight does not exceed the capacity and the total profit is as large as possible. Each item can "
                "be taken at most once.",
                3,
            ],
            [
                "There are several jobs, each consisting of a fixed sequence of operations that must run on specific "
                "machines with given processing times. Each machine can process at most one operation at a time. I "
                "want to schedule the start times of all operations so that machine-capacity and job-order constraints "
                "are respected and the time when the last job finishes (the makespan) is as small as possible.",
                3,
            ],
            [
                "Given a weighted directed graph with a designated source and sink node, I want to find the lowest-cost "
                "path from the source to the sink, where the cost of a path is the sum of the edge costs along that path.",
                1,
            ],
        ],
    )
    # Preload model so first Submit doesn't block the server for 30–60s (avoids "keeps loading").
    print("Loading embedding model (one-time, ~30s on first run)...", flush=True)
    get_model()
    print("Model ready. Starting server...", flush=True)

    # On HPC (Wulver) Gradio's launch() uses a thread for the server -> "can't start new thread".
    # Run via FastAPI + uvicorn in the main thread instead (no extra threads).
    # Access from laptop: ssh -L 7860:localhost:7860 USER@wulver.njit.edu then open http://127.0.0.1:7860
    from fastapi import FastAPI
    import uvicorn
    app = FastAPI()
    gr.mount_gradio_app(app, iface, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
