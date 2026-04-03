# Retrieval-assisted optimization schema grounding

**EAAI companion repo:** fixed-catalog **NLP4LP** benchmark, **deterministic scalar grounding**, and **restricted** engineering / **solver-backed subset** (SciPy HiGHS, 20 instances)—not a production product.

---

## Repository status (short)

| | |
|---|---|
| **Validated paper core** | NLP4LP `orig` (331 queries): retrieval → grounding; **Tables 1–5** in `results/paper/eaai_camera_ready_tables/` |
| **Canonical one-pager** | [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) |
| **Infrastructure / reruns** | Slurm `batch/learning/`; optional LLM baselines; Gemini [`docs/GEMINI_RERUN_REPORT.md`](docs/GEMINI_RERUN_REPORT.md); Mistral [`docs/MISTRAL_RERUN_REPORT.md`](docs/MISTRAL_RERUN_REPORT.md) (**infra ≠ completed reruns** unless your `results/rerun/` proves it) |
| **Demo / app** | `app.py`, `demo/` — **outside** paper-evaluated claims unless explicitly scoped |
| **Archives** | `docs/archive/`, `docs/archive_internal_status/`, `docs/provenance/`, `analysis/archive/` — **provenance only** |
| **External validation (non–paper-core)** | **Text2Zinc** + **CP-Bench** (DCP-Bench-Open): adapters + staging docs — **no new camera-ready metrics** until runs exist ([`docs/DATASET_EXPANSION_STATUS.md`](docs/DATASET_EXPANSION_STATUS.md)) |

**Table 1 headline** (TF-IDF + typed greedy, `orig`): Schema R@1 **0.9094**; Coverage **0.8639**; TypeMatch **0.7513**; InstantiationReady **0.5257** — source: `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` (see [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md)).

---

## Read these first (repo map)

1. [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md) — what is official vs auxiliary  
2. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — validated vs experimental, limitations  
3. [`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md) — manuscript authority / scope  
4. [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) — metrics and provenance chain  
5. [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) — rerun commands  
6. [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — blockers and design tensions  
7. [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) — annotated tree (canonical vs demo vs archive)  

**Index:** [`docs/README.md`](docs/README.md) · **External datasets plan:** [`docs/DATASET_EXPANSION_PLAN.md`](docs/DATASET_EXPANSION_PLAN.md) · **Doc check:** `python scripts/check_docs_integrity.py`

---

## What this repository does not claim

- **Arbitrary NL → solver-ready** for the full benchmark (solver claims are **subset-only**).  
- **Benchmark-wide** end-to-end execution — Tables 2–4 use **restricted denominators**.  
- **Dense retrieval (E5/BGE) as primary results** — supplementary; TF-IDF is the main retrieval baseline in the paper.  
- **Learned retrieval beating the rule baseline** on held-out eval — it does not ([`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)).  
- **Gurobi** for paper numbers — Table 4 uses **SciPy HiGHS** on 20 instances.  
- **Completed Mistral (or other) LLM benchmark reruns** — **not** claimed unless committed artifacts under `results/rerun/…` match [`docs/MISTRAL_RERUN_REPORT.md`](docs/MISTRAL_RERUN_REPORT.md) / [`docs/GEMINI_RERUN_REPORT.md`](docs/GEMINI_RERUN_REPORT.md).

---

## Reproducibility / access (short)

| Goal | Needs |
|------|--------|
| **Read official numbers** | Committed `results/paper/` (no keys) |
| **Structural checks** | `python scripts/paper/run_repo_validation.py`, pytest |
| **Recompute NLP4LP metrics** | Gated HF dataset `udell-lab/NLP4LP` + `HF_TOKEN` |
| **EAAI subset experiments** | `tools/run_eaai_*.py` — [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) |
| **Optional LLM baselines** | `OPENAI_API_KEY` / `GEMINI_API_KEY` / `MISTRAL_API_KEY` (see provider docs below) |
| **HPC (NJIT Wulver)** | [`docs/wulver.md`](docs/wulver.md) |

---

## LLM providers (optional, non–paper-core)

| Provider | Role | Result status |
|----------|------|----------------|
| **OpenAI** | `tools/llm_baselines.py`, `batch/learning/run_openai_llm_baselines.sbatch` | Historical downstream artifacts under `results/paper/` |
| **Gemini** | `google.genai`, Slurm batch, `scripts/gemini_preflight.py` | **Infra stabilized**; **full benchmark completion not asserted** without your `results/rerun/gemini/…` artifacts |
| **Mistral** | `tools/llm_baselines.py`, `batch/learning/run_mistral_llm_baselines.sbatch`, `scripts/mistral_preflight.py` | **Infra present**; **full completion not asserted** without `results/rerun/mistral/…` — [`docs/MISTRAL_RERUN_REPORT.md`](docs/MISTRAL_RERUN_REPORT.md) |

Camera-ready **Tables 1–5** remain manuscript authority; LLM CSVs are **auxiliary**.

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

**Web demo (out of paper scope):** `python app.py` — [`demo/README.md`](demo/README.md).  
**CLI search:** `python -m retrieval.search "your query" 3`  
**Figures from committed tables:** `python tools/build_eaai_camera_ready_figures.py` (requires Pillow).

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

**Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

**Full experiment log (history + pre-EAAI work):** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) · **CI benchmark workflow:** [`docs/HOW_TO_RUN_BENCHMARK.md`](docs/HOW_TO_RUN_BENCHMARK.md)
