"""
Simple web UI: type your combinatorial problem in natural language,
get the best-matching problem(s) and their integer programs.
Run: python app.py  then open the URL in your browser.
User queries are logged to data/collected_queries/user_queries.jsonl for training.

PDF support: you can also upload a PDF file (e.g. a paper or problem spec).
Its text will be extracted and placed in the query box so you can edit it
before running the search.
"""
from pathlib import Path
import json
import os
from datetime import datetime, timezone

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
    build_index,
    search,
    format_problem_and_ip,
)
from retrieval.pdf_utils import extract_text_from_pdf

# Load catalog once at startup
CATALOG = _load_catalog()
MODEL = None
# Pre-built embedding matrix for the catalog (built in get_model()).
EMBEDDINGS = None

# Where to save user prompts on Wulver (or any run) for later training
COLLECTED_QUERIES_DIR = Path(__file__).resolve().parent / "data" / "collected_queries"
USER_QUERIES_FILE = COLLECTED_QUERIES_DIR / "user_queries.jsonl"


def _log_user_query(query: str, top_k: int, results: list) -> None:
    """Append one JSONL record so you can use it for training. Safe for concurrent appends."""
    if not query or not query.strip():
        return
    try:
        COLLECTED_QUERIES_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "query": query.strip(),
            "top_k": int(top_k),
            "results": [
                {"name": p.get("name", ""), "score": float(s)}
                for p, s in results
            ],
        }
        with open(USER_QUERIES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # don't break the app if logging fails


def get_model():
    """Lazy-load the embedding model and pre-build the catalog index once.

    Building the index here (alongside the model) means every call to search()
    can pass the pre-built ``EMBEDDINGS`` array and skip re-encoding all
    catalog problems — the main per-query runtime bottleneck.
    """
    global MODEL, EMBEDDINGS
    if MODEL is None:
        from sentence_transformers import SentenceTransformer
        from retrieval.search import _default_model_path
        MODEL = SentenceTransformer(_default_model_path())
        EMBEDDINGS = build_index(CATALOG, MODEL)
    return MODEL


async def answer(query: str, top_k: int, validate: bool = False) -> str:
    # Async so FastAPI runs this in the event loop (no thread pool).
    # Avoids "can't start new thread" on Wulver when user clicks Submit.
    if not query or not query.strip():
        return "Please type a short description of your optimization problem (e.g. *minimize cost of opening warehouses and assigning customers*)."
    model = get_model()
    k = max(1, min(10, top_k))
    results = search(
        query.strip(),
        catalog=CATALOG,
        embeddings=EMBEDDINGS,
        model=model,
        top_k=k,
        validate=validate,
    )
    _log_user_query(query.strip(), k, results)
    if not results:
        return "No matching problems found."
    out = []
    for i, (problem, score) in enumerate(results, 1):
        out.append(f"### Result {i} (relevance: {score:.3f})")
        out.append(format_problem_and_ip(problem, score=None))
        if validate and problem.get("_validation"):
            v = problem["_validation"]
            errs = (v.get("schema_errors") or []) + (v.get("formulation_errors") or [])
            if errs:
                out.append(f"**Validation:** ⚠ {len(errs)} issue(s): " + "; ".join(errs[:3]) + (" ..." if len(errs) > 3 else ""))
            else:
                out.append("**Validation:** ✓ Schema and formulation OK")
        out.append("---")
    return "\n".join(out)


def handle_pdf_upload(file_path: str) -> tuple[str, str]:
    """Gradio event handler for PDF file upload.

    Extracts text from the uploaded PDF and returns a ``(query_text, status)``
    tuple consumed by the ``[query_in, pdf_status]`` output components.
    When *file_path* is falsy (file cleared) both outputs are reset.
    """
    if not file_path:
        return "", "*Upload a PDF to extract its text into the query box above.*"
    text = extract_text_from_pdf(file_path)
    if text.startswith("(Could not extract PDF text:"):
        return text, f"⚠ {text}"
    return text, "*PDF loaded — text extracted into the query box. Edit as needed, then click Search.*"

def main():
    n_problems = len(CATALOG)
    # Softer theme and compact custom CSS for a cleaner, more readable UI
    custom_css = """
    .header-box { text-align: center; padding: 1rem 0 0.5rem; margin-bottom: 0.5rem; }
    .header-box h1 { margin: 0; font-size: 1.6rem; }
    .header-box p { margin: 0.4rem 0 0; opacity: 0.9; font-size: 0.95rem; }
    .foot-note { font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem; }
    """
    try:
        theme = gr.themes.Soft(primary_hue="slate", secondary_hue="blue")
    except Exception:
        theme = None
    with gr.Blocks(
        title="Combinatorial Optimization Bot",
        theme=theme,
        css=custom_css,
    ) as iface:
        gr.HTML(
            f'<div class="header-box">'
            '<h1>Combinatorial Optimization Bot</h1>'
            "<p>Describe your problem in plain English. Get the closest matching problem and its "
            "integer program (variables, objective, constraints).</p>"
            "</div>"
        )
        with gr.Row():
            query_in = gr.Textbox(
                label="Your problem (natural language)",
                placeholder=(
                    "e.g. I have warehouses and customers; I want to choose which warehouses to open "
                    "and assign each customer to one warehouse to minimize total cost."
                ),
                lines=4,
                scale=3,
            )
            with gr.Column(scale=1):
                top_k_in = gr.Slider(
                    1, 10, value=3, step=1,
                    label="Number of results",
                )
                validate_in = gr.Checkbox(value=False, label="Validate outputs")
                gr.Markdown("*Tip: first query may take a few seconds.*")
        with gr.Accordion("📄 Upload a PDF (optional)", open=False):
            pdf_upload = gr.File(
                label="Upload a PDF problem description",
                file_types=[".pdf"],
                type="filepath",
            )
            pdf_status = gr.Markdown("*Upload a PDF to extract its text into the query box above.*")
            pdf_upload.change(
                fn=handle_pdf_upload,
                inputs=pdf_upload,
                outputs=[query_in, pdf_status],
            )
        with gr.Row():
            submit_btn = gr.Button("Search", variant="primary")
        out_md = gr.Markdown(label="Results", value="*Enter a problem above and click Search.*")
        submit_btn.click(
            fn=answer,
            inputs=[query_in, top_k_in, validate_in],
            outputs=out_md,
        )
        gr.Examples(
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
            inputs=[query_in, top_k_in],
            label="Try an example",
        )
        gr.Markdown(
            f'<div class="foot-note">'
            f"**Flag** bad or surprising results for review. Catalog: {n_problems} problems."
            "</div>"
        )
    # Preload model and build embedding index so first Submit doesn't block the server
    # for 30–60s (avoids "keeps loading").  Without this, build_index() would run on
    # every query because EMBEDDINGS would be None.
    print("Loading embedding model and building catalog index (one-time, ~30s on first run)...", flush=True)
    get_model()
    print(f"Model ready. Index built for {len(CATALOG)} problems. Starting server...", flush=True)

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