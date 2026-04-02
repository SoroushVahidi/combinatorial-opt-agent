# Retrieval-Assisted Optimization Schema Grounding

Companion research repository for the **EAAI** manuscript: **retrieval-assisted schema grounding** and **deterministic scalar instantiation** on the **fixed-catalog NLP4LP** benchmark, plus **restricted** engineering and **solver-backed subset** studies (SciPy HiGHS shim, 20 instances).

---

## Repository status

| | |
|---|---|
| **Intent** | Public, reviewer-facing companion codebase — not a production product |
| **Canonical summary** | **[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)** — one-page truth: headline pointers, validated vs auxiliary, limitations |
| **Manuscript authority** | **[`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md)** — scope, authoritative file list |
| **Paper-core pipeline** | NLP4LP `orig` (331 queries): fixed-catalog **schema retrieval** → **deterministic scalar grounding**; camera-ready tables in **`results/paper/eaai_camera_ready_tables/`** |
| **Restricted subsets** | Structural (60), executable-attempt (269, with blockers), solver-backed (20) — see Table 2–4 |
| **Infrastructure / reruns** | Slurm batch scripts under **`batch/learning/`**; optional LLM baselines via **`tools/llm_baselines.py`**; Gemini workflow **[`docs/GEMINI_RERUN_REPORT.md`](docs/GEMINI_RERUN_REPORT.md)** |
| **Demo / app** | **`app.py`** (Gradio), **`demo/`** — **outside** paper-evaluated claims unless explicitly documented |
| **Archived / provenance** | **`docs/archive/`**, **`docs/archive_internal_status/`**, **`docs/provenance/`**, **`analysis/archive/`** — history only |

**Table 1 headline (TF-IDF + typed greedy, `orig`):** Schema R@1 **0.9094**; Coverage **0.8639**; TypeMatch **0.7513**; InstantiationReady **0.5257** — from **`results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`** (see **[`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md)**; do not mix with other CSV column definitions).

---

## What this repository does not claim

- **Arbitrary NL → solver-ready compilation** for the full benchmark (solver-backed claims are **subset-only**).
- **Benchmark-wide** end-to-end execution — Tables 2–4 define **restricted denominators**.
- **Dense retrieval (E5/BGE) as primary results** — supplementary; TF-IDF is the primary retrieval baseline in the paper.
- **Learned retrieval beating the rule baseline** on held-out eval — it does not (**[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)**).
- **Gurobi** for paper numbers — Table 4 uses a **SciPy HiGHS** path on 20 instances.
- **Mistral (or other extra providers)** — not configured in this repo; optional APIs are **OpenAI** and **Gemini** only.

---

## Reproducibility / access requirements

| Goal | Requirements |
|------|----------------|
| **Read official numbers** | No keys — use committed **`results/paper/`** |
| **Structural / repo checks** | **`python scripts/paper/run_repo_validation.py`**, pytest |
| **Recompute NLP4LP metrics** | Gated HF dataset **`udell-lab/NLP4LP`** + **`HF_TOKEN`** |
| **EAAI subset experiments** | Scripts **`tools/run_eaai_*.py`** (see **[`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md)**) |
| **Optional LLM baselines** | **`OPENAI_API_KEY`** / **`GEMINI_API_KEY`**; outputs may go to **`results/rerun/`** for Gemini |
| **HPC (NJIT Wulver)** | **[`docs/wulver.md`](docs/wulver.md)** |

---

## LLM provider wiring (optional, non-paper-core)

| Provider | Role in repo | Result status |
|----------|--------------|---------------|
| **OpenAI** | `tools/llm_baselines.py` + `batch/learning/run_openai_llm_baselines.sbatch` | **Downstream artifacts** for several variants exist under **`results/paper/`** (historical successful batch). |
| **Gemini** | `google.genai` client, `batch/learning/run_gemini_llm_baselines.sbatch`, **`scripts/gemini_preflight.py`** | **Infrastructure stabilized** (preflight, cache, partial save). **Full benchmark completion** is **not** asserted in docs unless **`results/rerun/gemini/…`** artifacts and logs exist for your run. |
| **Mistral** | — | **Not present** in `configs/` or baseline code paths. |

Camera-ready **Tables 1–5** remain the authority for the manuscript; LLM baseline CSVs are **auxiliary**.

---

## Reviewer-first navigation

| Read first | Purpose |
|------------|---------|
| **[`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md)** | Orientation, canonical paths, limitations |
| **[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)** | Single status page |
| **[`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md)** | Manuscript scope + file authority |
| **[`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md)** | Metric definitions + provenance chain |
| **[`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md)** | Step-by-step rerun |
| **[`REPO_STRUCTURE.md`](REPO_STRUCTURE.md)** | Annotated tree (canonical vs demo vs archive) |
| **[`docs/README.md`](docs/README.md)** | Full documentation index |

**Doc integrity (local):** `python scripts/check_docs_integrity.py`

---

## Pipeline (high level)

```
NL query → Schema retrieval (TF-IDF / BM25 / LSA, …) → top-1 schema
         → Deterministic scalar grounding (tools/nlp4lp_downstream_utility.py)
         → Structural LP check (formulation/verify.py)
         → [Optional] Solver on restricted subset (SciPy HiGHS shim)
```

---

## Quick start

```bash
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Web demo (out of paper scope):** `python app.py` — see **[`demo/README.md`](demo/README.md)**.

**CLI search:** `python -m retrieval.search "your query" 3`

**Regenerate figures from committed tables:** `python tools/build_eaai_camera_ready_figures.py` (requires Pillow).

---

## Paper artifacts

Tables **1–5** and figures **1–5** live under **`results/paper/eaai_camera_ready_tables/`** and **`results/paper/eaai_camera_ready_figures/`**.  
Experiment write-ups: **`analysis/eaai_*_report.md`**.

**Engineering changelog (non-canonical):** [`docs/provenance/engineering_changelog_recent.md`](docs/provenance/engineering_changelog_recent.md)

---

## HuggingFace (gated NLP4LP)

```bash
cp .env.example .env   # add HF_TOKEN=hf_... (never commit .env)
# Request access: https://huggingface.co/datasets/udell-lab/NLP4LP
```

---

## Tech stack (short)

Python 3.10+ · scikit-learn / rank-bm25 · optional sentence-transformers · SciPy · Gradio · pytest · GitHub Actions · SLURM on Wulver (optional).

---

## License · acknowledgments · contact

**License:** [MIT](LICENSE) · © Soroush Vahidi  

**Acknowledgments:** NL4Opt, Gurobi (examples), GAMS, MIPLIB, OR-Library, Pyomo; NJIT Computer Science.

**Contact:** [sv96@njit.edu](mailto:sv96@njit.edu) · [@SoroushVahidi](https://github.com/SoroushVahidi)

**Contributing / remotes:** [CONTRIBUTING.md](CONTRIBUTING.md)
