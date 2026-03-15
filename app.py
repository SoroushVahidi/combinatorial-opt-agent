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
import html as _html
import json
import os
from datetime import datetime, timezone

# Disable Gradio analytics to avoid extra thread (fixes "can't start new thread" on HPC)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# Disable Hugging Face Xet backend (spawns worker threads -> "Resource temporarily unavailable" on login nodes)
os.environ["HF_HUB_DISABLE_XET"] = "1"

# On Wulver, FastAPI/Starlette run sync route handlers in a thread pool -> "can't start new thread".
# Patch run_in_threadpool to run in the current thread instead (no new threads).
# Guarded so app.py can be imported in environments where starlette is not installed
# (e.g. lightweight test environments).
try:
    import starlette.concurrency as _starlette_concurrency
    async def _run_in_threadpool_same_thread(func, *args, **kwargs):
        return func(*args, **kwargs)  # run in event-loop thread; blocks during request
    _starlette_concurrency.run_in_threadpool = _run_in_threadpool_same_thread
except ImportError:
    pass  # starlette not installed; HPC thread-pool patch skipped

# gradio is required to run the web UI (main()), but the core logic functions
# (_log_user_query, get_model, answer, search helpers) work without it.
# Guard the import so that app.py can be imported in lightweight test environments
# where gradio is not installed.
try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore[assignment]  # UI will not be available

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
                {"id": p.get("id", ""), "name": p.get("name", ""), "score": float(s)}
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


def _relevance_color(score: float) -> tuple[str, str]:
    """Return (hex_color, label) based on similarity score."""
    if score >= 0.80:
        return "#15803d", "Strong match"   # green-700
    if score >= 0.65:
        return "#0369a1", "Good match"     # sky-700
    if score >= 0.50:
        return "#b45309", "Partial match"  # amber-700
    return "#6b7280", "Weak match"         # gray-500


def _format_result_html(problem: dict, score: float, index: int, validation: dict | None = None) -> str:
    """Render one search result as a self-contained styled HTML card."""
    color, label = _relevance_color(score)
    pct = min(100, int(score * 100))
    name = _html.escape(problem.get("name", "Unknown"))
    description = _html.escape(problem.get("description", ""))
    complexity = _html.escape(problem.get("complexity", ""))
    source = _html.escape(problem.get("source", ""))

    formulation = problem.get("formulation") or {}
    variables = formulation.get("variables") or []
    constraints = formulation.get("constraints") or []
    objective = formulation.get("objective") or {}
    has_formulation = bool(variables or constraints or objective)

    # ── Header ────────────────────────────────────────────────────────────────
    card = f"""
<div class="coa-card" id="coa-result-{index}">
  <div class="coa-card-header">
    <div class="coa-title-row">
      <span class="coa-rank">#{index}</span>
      <span class="coa-name">{name}</span>
    </div>
    <div class="coa-meta-row">
      <span class="coa-badge" style="background:{color}">{label}&nbsp;{pct}%</span>
      {"<span class='coa-complexity'>⏱ " + complexity + "</span>" if complexity else ""}
      {"<span class='coa-source'>📖 " + source + "</span>" if source else ""}
    </div>
  </div>
  <div class="coa-relevance-bar">
    <div class="coa-relevance-fill" style="width:{pct}%;background:{color}"></div>
  </div>
  <p class="coa-description">{description}</p>
"""

    # ── Formulation sections ──────────────────────────────────────────────────
    if has_formulation:
        # Variables
        if variables:
            rows = ""
            for v in variables:
                sym = _html.escape(v.get("symbol", ""))
                desc = _html.escape(v.get("description", ""))
                domain = _html.escape(v.get("domain", ""))
                rows += (
                    f"<tr><td class='coa-sym'><code>{sym}</code></td>"
                    f"<td class='coa-vdesc'>{desc}</td>"
                    f"<td class='coa-domain'>{domain}</td></tr>"
                )
            card += f"""
  <details class="coa-section">
    <summary>📊 Variables <span class="coa-count">({len(variables)})</span></summary>
    <table class="coa-var-table"><thead>
      <tr><th>Symbol</th><th>Description</th><th>Domain</th></tr>
    </thead><tbody>{rows}</tbody></table>
  </details>"""

        # Objective
        if objective:
            sense = _html.escape(objective.get("sense", "").capitalize())
            expr = _html.escape(objective.get("expression", ""))
            card += f"""
  <details class="coa-section" open>
    <summary>🎯 Objective</summary>
    <div class="coa-obj">
      <span class="coa-sense-badge">{sense}</span>
      <code class="coa-expr">{expr}</code>
    </div>
  </details>"""

        # Constraints
        if constraints:
            items = ""
            for c in constraints:
                expr = _html.escape(c.get("expression", ""))
                desc = _html.escape(c.get("description", ""))
                items += (
                    f"<li><code class='coa-expr'>{expr}</code>"
                    + (f"<span class='coa-cdesc'> — {desc}</span>" if desc else "")
                    + "</li>"
                )
            card += f"""
  <details class="coa-section">
    <summary>📋 Constraints <span class="coa-count">({len(constraints)})</span></summary>
    <ul class="coa-constraint-list">{items}</ul>
  </details>"""
    else:
        card += """
  <div class="coa-no-formulation">
    ⚠ Formulation not yet available — the description above may still help.
  </div>"""

    # ── Validation banner (optional) ──────────────────────────────────────────
    if validation is not None:
        errs = (
            (validation.get("schema_errors") or [])
            + (validation.get("formulation_errors") or [])
            + (validation.get("lp_consistency_errors") or [])
        )
        if errs:
            snippet = _html.escape("; ".join(errs[:3])) + (" …" if len(errs) > 3 else "")
            card += f"""
  <div class="coa-validation coa-validation-warn">
    ⚠ {len(errs)} validation issue(s): {snippet}
  </div>"""
        else:
            card += """
  <div class="coa-validation coa-validation-ok">
    ✓ Schema, formulation, and LP consistency OK
  </div>"""

    card += "\n</div>\n"
    return card


async def answer(query: str, top_k: int, validate: bool = False) -> str:
    """Return styled HTML cards for the top-k matching problems.

    Async so FastAPI runs this in the event loop (no thread pool).
    Avoids 'can't start new thread' on Wulver when user clicks Submit.
    """
    if not query or not query.strip():
        return (
            '<div class="coa-empty-state">'
            "<p>✏️ Describe your optimization problem above and click <strong>Search</strong>.</p>"
            "<p style='font-size:0.9rem;opacity:0.7'>Example: <em>minimize cost of opening warehouses and assigning customers</em></p>"
            "</div>"
        )
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
        return (
            '<div class="coa-empty-state coa-empty-warn">'
            "⚠ No matching problems found — try rephrasing your description."
            "</div>"
        )
    cards = []
    for i, (problem, score) in enumerate(results, 1):
        val = problem.get("_validation") if validate else None
        cards.append(_format_result_html(problem, score, i, validation=val))

    header = (
        f'<p class="coa-result-summary">'
        f"Found <strong>{len(results)}</strong> result{'s' if len(results) != 1 else ''} "
        f"— ordered by semantic similarity to your query."
        f"</p>"
    )
    return header + "\n".join(cards)


def handle_pdf_upload(file_path: str) -> tuple[str, str]:
    """Gradio event handler for PDF file upload.

    Extracts text from the uploaded PDF and returns a ``(query_text, status)``
    tuple consumed by the ``[query_in, pdf_status]`` output components.
    When *file_path* is falsy (file cleared) both outputs are reset.
    """
    if not file_path:
        return "", "📄 Upload a PDF to extract its text into the query box above."
    text = extract_text_from_pdf(file_path)
    if text.startswith("(Could not extract PDF text:"):
        return text, f"⚠ {text}"
    return text, "✅ PDF loaded — text extracted into the query box. Edit as needed, then click **Search**."


# ---------------------------------------------------------------------------
# CSS – applied to the entire Gradio Blocks page
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
/* ── Page-level adjustments ──────────────────────────────────────────────── */
.gradio-container { max-width: 960px !important; margin: 0 auto; }

/* ── Hero header ─────────────────────────────────────────────────────────── */
.coa-hero {
  background: linear-gradient(135deg, #1e3a5f 0%, #0f6cbf 60%, #1a7fc1 100%);
  border-radius: 12px;
  padding: 1.4rem 2rem 1.2rem;
  margin-bottom: 1rem;
  color: #fff;
}
.coa-hero h1 {
  margin: 0 0 0.35rem;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: -0.5px;
}
.coa-hero p {
  margin: 0;
  font-size: 0.97rem;
  opacity: 0.88;
  line-height: 1.5;
}

/* ── Result summary line ─────────────────────────────────────────────────── */
.coa-result-summary {
  font-size: 0.9rem;
  color: #4b5563;
  margin: 0 0 0.75rem;
  padding-left: 2px;
}

/* ── Individual result card ──────────────────────────────────────────────── */
.coa-card {
  background: #fff;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  margin-bottom: 1.1rem;
  overflow: hidden;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  transition: box-shadow 0.15s ease;
}
.coa-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.10); }

/* Card header */
.coa-card-header {
  padding: 0.8rem 1rem 0.4rem;
  border-bottom: 1px solid #f1f5f9;
}
.coa-title-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  margin-bottom: 0.4rem;
}
.coa-rank {
  background: #1e3a5f;
  color: #fff;
  font-size: 0.72rem;
  font-weight: 700;
  border-radius: 4px;
  padding: 1px 6px;
  flex-shrink: 0;
}
.coa-name {
  font-size: 1.05rem;
  font-weight: 600;
  color: #1e3a5f;
}
.coa-meta-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.coa-badge {
  color: #fff;
  font-size: 0.72rem;
  font-weight: 600;
  border-radius: 999px;
  padding: 2px 10px;
  white-space: nowrap;
}
.coa-complexity, .coa-source {
  font-size: 0.78rem;
  color: #6b7280;
}

/* Thin relevance bar below header */
.coa-relevance-bar {
  height: 4px;
  background: #f1f5f9;
}
.coa-relevance-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.4s ease;
}

/* Problem description */
.coa-description {
  margin: 0.7rem 1rem 0.5rem;
  font-size: 0.92rem;
  color: #374151;
  line-height: 1.6;
}

/* ── Collapsible formulation sections ────────────────────────────────────── */
.coa-section {
  border-top: 1px solid #f1f5f9;
  padding: 0;
}
.coa-section summary {
  cursor: pointer;
  padding: 0.55rem 1rem;
  font-size: 0.88rem;
  font-weight: 600;
  color: #374151;
  user-select: none;
  list-style: none;
}
.coa-section summary::-webkit-details-marker { display: none; }
.coa-section summary::before { content: "▶ "; font-size: 0.65rem; opacity: 0.55; }
.coa-section[open] summary::before { content: "▼ "; }
.coa-section summary:hover { background: #f8fafc; }
.coa-count { font-weight: 400; color: #9ca3af; font-size: 0.82rem; }

/* Variables table */
.coa-var-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.83rem;
  margin: 0 0 0.6rem;
}
.coa-var-table th {
  background: #f8fafc;
  padding: 5px 12px;
  text-align: left;
  color: #6b7280;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 1px solid #e2e8f0;
}
.coa-var-table td {
  padding: 5px 12px;
  border-bottom: 1px solid #f1f5f9;
  vertical-align: top;
}
.coa-sym code { font-size: 0.88rem; color: #1e3a5f; background: #eff6ff; padding: 1px 5px; border-radius: 4px; }
.coa-vdesc { color: #374151; }
.coa-domain { color: #6b7280; font-size: 0.8rem; }

/* Objective block */
.coa-obj {
  padding: 0.5rem 1rem 0.65rem;
  display: flex;
  align-items: flex-start;
  gap: 0.6rem;
  flex-wrap: wrap;
}
.coa-sense-badge {
  background: #eff6ff;
  color: #1d4ed8;
  font-size: 0.75rem;
  font-weight: 700;
  border-radius: 6px;
  padding: 2px 9px;
  text-transform: uppercase;
  white-space: nowrap;
  flex-shrink: 0;
}
.coa-expr {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.85rem;
  color: #1e3a5f;
  background: #f8fafc;
  padding: 2px 6px;
  border-radius: 4px;
  white-space: pre-wrap;
  word-break: break-all;
}

/* Constraints list */
.coa-constraint-list {
  padding: 0.4rem 1rem 0.65rem 1.5rem;
  margin: 0;
  list-style: disc;
}
.coa-constraint-list li {
  margin-bottom: 0.35rem;
  font-size: 0.85rem;
  color: #374151;
  line-height: 1.5;
}
.coa-cdesc { color: #6b7280; }

/* Missing formulation notice */
.coa-no-formulation {
  margin: 0.6rem 1rem 0.8rem;
  font-size: 0.87rem;
  color: #b45309;
  background: #fffbeb;
  border: 1px solid #fde68a;
  border-radius: 6px;
  padding: 0.4rem 0.75rem;
}

/* Validation banners */
.coa-validation {
  margin: 0;
  padding: 0.4rem 1rem;
  font-size: 0.83rem;
  font-weight: 500;
  border-top: 1px solid #f1f5f9;
}
.coa-validation-ok { background: #f0fdf4; color: #15803d; }
.coa-validation-warn { background: #fffbeb; color: #b45309; }

/* ── Empty / error states ─────────────────────────────────────────────────── */
.coa-empty-state {
  text-align: center;
  padding: 2.5rem 1rem;
  color: #6b7280;
  font-size: 0.95rem;
  border: 2px dashed #e2e8f0;
  border-radius: 10px;
  background: #f9fafb;
}
.coa-empty-warn { color: #b45309; border-color: #fcd34d; background: #fffbeb; }

/* ── Footer ──────────────────────────────────────────────────────────────── */
.coa-footer {
  text-align: center;
  font-size: 0.78rem;
  color: #9ca3af;
  margin-top: 0.75rem;
  padding-top: 0.5rem;
  border-top: 1px solid #f1f5f9;
}

/* ── Mobile / iPhone responsive ──────────────────────────────────────────── */
@media (max-width: 640px) {
  /* Respect iPhone notch and home-bar safe areas */
  body {
    padding-left:  env(safe-area-inset-left);
    padding-right: env(safe-area-inset-right);
  }
  .gradio-container { padding: 0.5rem !important; }

  /* Stack the input row vertically on small screens */
  .gr-row { flex-direction: column !important; }

  /* Larger touch targets for buttons */
  .coa-card-header { padding: 0.9rem 0.85rem 0.5rem; }
  .coa-section summary { padding: 0.7rem 0.85rem; font-size: 0.92rem; }
  .coa-description { font-size: 0.94rem; margin: 0.7rem 0.85rem 0.5rem; }
  .coa-constraint-list { padding-left: 1.2rem; }
  .coa-var-table td, .coa-var-table th { padding: 5px 8px; }

  /* Hero banner — smaller on mobile */
  .coa-hero { padding: 1rem 1rem 0.9rem; }
  .coa-hero h1 { font-size: 1.3rem; }
  .coa-hero p  { font-size: 0.88rem; }

  /* Buttons: full-width on mobile */
  button { min-height: 44px; }
}

/* ── PWA standalone mode tweaks ─────────────────────────────────────────── */
@media (display-mode: standalone) {
  /* Add top padding so content clears the iOS status bar */
  body { padding-top: env(safe-area-inset-top); }
  .coa-hero { border-radius: 0 0 12px 12px; }
}
"""


# ── PWA: <head> tags injected into every Gradio page ────────────────────────
# These turn the web app into a Progressive Web App that iPhone users can
# "Add to Home Screen" from Safari — it then runs fullscreen with an icon and
# splash screen, indistinguishable from a native app.
_PWA_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#1e3a5f">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Opt Bot">
<meta name="description" content="Match your optimization problem to a structured integer program from a catalog of 1600+ classical formulations.">

<!-- PWA manifest (Android + iOS Safari install prompt) -->
<link rel="manifest" href="/manifest.json">

<!-- iOS home-screen icons (Safari ignores manifest icons) -->
<link rel="apple-touch-icon"              href="/static/icons/icon-180.png">
<link rel="apple-touch-icon" sizes="120x120" href="/static/icons/icon-120.png">
<link rel="apple-touch-icon" sizes="152x152" href="/static/icons/icon-152.png">
<link rel="apple-touch-icon" sizes="167x167" href="/static/icons/icon-167.png">
<link rel="apple-touch-icon" sizes="180x180" href="/static/icons/icon-180.png">

<!-- Favicon -->
<link rel="icon" type="image/x-icon" href="/favicon.ico">

<!-- Service-worker registration (enables offline + install prompt) -->
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('/sw.js', { scope: '/' })
        .catch(function (err) {
          console.warn('SW registration failed:', err);
        });
    });
  }
</script>
"""

# Minimal offline fallback page served by the service worker when the network
# is unavailable and no cached version of the page exists.
_OFFLINE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="theme-color" content="#1e3a5f">
  <title>Combinatorial Optimization Bot — Offline</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #1e3a5f;
      color: #fff;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
      padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
      text-align: center;
    }
    .icon { font-size: 4rem; margin-bottom: 1rem; }
    h1 { font-size: 1.5rem; margin: 0 0 0.5rem; }
    p  { font-size: 0.95rem; opacity: 0.8; max-width: 320px; line-height: 1.5; }
    button {
      margin-top: 1.5rem;
      background: #fff;
      color: #1e3a5f;
      border: none;
      border-radius: 8px;
      padding: 0.75rem 2rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      min-height: 44px;
    }
  </style>
</head>
<body>
  <div class="icon">🔢</div>
  <h1>You're offline</h1>
  <p>The Combinatorial Optimization Bot needs an internet connection to run the search model.</p>
  <button onclick="window.location.reload()">Try again</button>
</body>
</html>"""


def main():
    n_problems = len(CATALOG)
    try:
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky")
    except Exception:
        theme = None
    with gr.Blocks(
        title="Combinatorial Optimization Bot",
    ) as iface:

        # ── Hero header ───────────────────────────────────────────────────────
        gr.HTML(
            '<div class="coa-hero">'
            "<h1>🔢 Combinatorial Optimization Bot</h1>"
            "<p>Describe your problem in plain English. "
            "Get the closest matching formulation from a curated catalog of "
            f"{n_problems} classical integer programs — variables, objective, and constraints.</p>"
            "</div>"
        )

        # ── Input panel ───────────────────────────────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                query_in = gr.Textbox(
                    label="Problem description",
                    placeholder=(
                        "e.g. I have a set of warehouses with fixed opening costs and a set of "
                        "customers with known demands. I want to decide which warehouses to open "
                        "and how to assign customers to open warehouses to minimise total cost."
                    ),
                    lines=5,
                    max_lines=12,
                )
            with gr.Column(scale=1, min_width=180):
                top_k_in = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Number of results",
                )
                validate_in = gr.Checkbox(value=False, label="Validate outputs")
                gr.Markdown(
                    "<small>💡 First query takes ~30 s to load the model; subsequent queries are fast.</small>"
                )

        # ── PDF accordion ─────────────────────────────────────────────────────
        with gr.Accordion("📄 Upload a PDF problem description (optional)", open=False):
            with gr.Row():
                pdf_upload = gr.File(
                    label="PDF file",
                    file_types=[".pdf"],
                    type="filepath",
                    scale=1,
                )
                with gr.Column(scale=2):
                    pdf_status = gr.Markdown(
                        "📄 Upload a PDF to extract its text into the query box above.",
                    )
            pdf_upload.change(
                fn=handle_pdf_upload,
                inputs=pdf_upload,
                outputs=[query_in, pdf_status],
            )

        # ── Action buttons ────────────────────────────────────────────────────
        with gr.Row():
            submit_btn = gr.Button("🔍 Search", variant="primary", scale=3)
            clear_btn = gr.Button("✕ Clear", variant="secondary", scale=1)

        # ── Results pane ──────────────────────────────────────────────────────
        out_html = gr.HTML(
            value=(
                '<div class="coa-empty-state">'
                "✏️ Describe your optimization problem above and click <strong>Search</strong>."
                "</div>"
            ),
        )

        # ── Button wiring ─────────────────────────────────────────────────────
        submit_btn.click(
            fn=answer,
            inputs=[query_in, top_k_in, validate_in],
            outputs=out_html,
        )
        # Trigger search on Shift+Enter or Enter inside the textbox
        query_in.submit(
            fn=answer,
            inputs=[query_in, top_k_in, validate_in],
            outputs=out_html,
        )
        clear_btn.click(
            fn=lambda: ("", 3, False,
                        '<div class="coa-empty-state">'
                        "✏️ Describe your optimization problem above and click <strong>Search</strong>."
                        "</div>"),
            inputs=[],
            outputs=[query_in, top_k_in, validate_in, out_html],
        )

        # ── Examples ──────────────────────────────────────────────────────────
        gr.Examples(
            label="📌 Try an example (click to load)",
            examples=[
                [
                    "Facility location: choose which warehouses to open and assign customers "
                    "to minimise fixed opening costs plus per-unit shipping cost, subject to "
                    "warehouse capacity constraints.",
                    3,
                ],
                [
                    "0-1 knapsack: a list of items each with a weight and a profit; "
                    "select a subset that fits in a weight-limited knapsack and maximises total profit.",
                    2,
                ],
                [
                    "Job-shop scheduling: jobs consist of ordered operations on specific machines; "
                    "each machine handles one operation at a time; minimise makespan.",
                    2,
                ],
                [
                    "Shortest path in a weighted directed graph from a source node to a sink node.",
                    1,
                ],
                [
                    "Bin packing: pack items of various sizes into the minimum number of "
                    "fixed-capacity bins.",
                    2,
                ],
                [
                    "Set cover: select the minimum number of subsets so that every element "
                    "of the universe is covered by at least one selected subset.",
                    2,
                ],
            ],
            inputs=[query_in, top_k_in],
        )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML(
            f'<div class="coa-footer">'
            f"Catalog: <strong>{n_problems}</strong> problems · "
            "Press <kbd>Enter</kbd> in the text box or click Search · "
            "Queries are logged locally for training"
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
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    _STATIC_DIR = Path(__file__).resolve().parent / "static"
    app = FastAPI()

    # ── PWA asset routes ───────────────────────────────────────────────────────
    # Mount static files directory (icons, etc.)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/manifest.json", include_in_schema=False)
    async def pwa_manifest() -> Response:
        return FileResponse(
            str(_STATIC_DIR / "manifest.json"),
            media_type="application/manifest+json",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/sw.js", include_in_schema=False)
    async def service_worker() -> Response:
        return FileResponse(
            str(_STATIC_DIR / "sw.js"),
            media_type="application/javascript",
            # Service workers must be served from the root scope;
            # Service-Worker-Allowed header grants scope "/" even when SW is at /sw.js.
            headers={
                "Service-Worker-Allowed": "/",
                "Cache-Control": "no-cache",
            },
        )

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        ico = _STATIC_DIR / "favicon.ico"
        if ico.exists():
            return FileResponse(str(ico), media_type="image/x-icon")
        return Response(status_code=404)

    @app.get("/offline", include_in_schema=False)
    async def offline_page() -> HTMLResponse:
        return HTMLResponse(_OFFLINE_HTML, status_code=200)

    # ── Mount Gradio with PWA head tags injected ───────────────────────────────
    gr.mount_gradio_app(app, iface, path="/", theme=theme, css=_CUSTOM_CSS, head=_PWA_HEAD)
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()