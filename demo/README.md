# Demo / Application Layer

This directory and the root-level files listed below contain the **interactive demo
application** for this project.  They are **outside the scope of the EAAI manuscript**
and are provided for demonstration and exploratory use only.

---

## Demo-related files (at repository root, not in paper scope)

| File | Purpose |
|------|---------|
| `app.py` | Gradio web UI — interactive schema search + LLM generation path |
| `feedback_server.py` | Lightweight feedback collection server for the web UI |
| `analyze_feedback.py` | Offline analysis of collected feedback logs |
| `deploy_to_hf.py` | Deploys the app to a Hugging Face Space |
| `launch_and_capture_url.py` | Helper to launch app and capture its public URL |
| `run_app_wulver.sh` | Shell wrapper to run the app on NJIT Wulver HPC |
| `telemetry.py` | Optional query logging for training data collection |
| `README_Spaces.md` (this dir) | Instructions for deploying to Hugging Face Spaces |

---

## Quickstart — run the demo app

```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:7860 in your browser
```

The app lets you:
- Type a natural-language description of an optimization problem
- Retrieve the best-matching problem schema from the catalog
- View the ILP/LP formulation (variables, objective, constraints)
- Optionally use an LLM to generate a formulation for unknown problems (**outside paper scope**)

---

## What is NOT in the benchmark evaluation

The following demo features have **not** been evaluated in the EAAI manuscript:

- LLM-based formulation generation for unknown problem types
- Gradio UI interactions
- Hugging Face Spaces deployment
- Feedback server / telemetry collection
- GAMSPy / Gurobi solver code generation

For the benchmark-evaluated pipeline and paper artifacts, see:
- `README.md` (top-level)
- `docs/EAAI_SOURCE_OF_TRUTH.md`
- `results/paper/`
- `analysis/eaai_*`

---

## Data collection notice

When running the app, every user query is logged locally to
`data/collected_queries/user_queries.jsonl` (gitignored).  Optional telemetry to a
private GitHub repository is enabled only when `TELEMETRY_REPO` and `TELEMETRY_TOKEN`
environment variables are set.  See the top-level `README.md` for details.
