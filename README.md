# Retrieval-assisted optimization schema grounding

**Companion repo for a manuscript submitted to Knowledge and Information Systems (KAIS), Springer Nature** (manuscript source: [`manuscript/`](manuscript/); see [`docs/KAIS_SOURCE_OF_TRUTH.md`](docs/KAIS_SOURCE_OF_TRUTH.md)): fixed-catalog **NLP4LP** benchmark, **deterministic scalar grounding**, and **restricted** engineering / **solver-backed subset** (SciPy HiGHS, 20 instances)—not a production product.

**New agent or contributor? Start at [`PROJECT_STATUS.md`](PROJECT_STATUS.md)** — single up-to-date entry point (scientific goal, pipeline, authoritative results, what's implemented/failed, PaMOP baseline-reproduction status, next steps).

---

## Repository status (short)

| | |
|---|---|
| **Validated paper core** | NLP4LP `orig` (331 queries): retrieval → grounding; **Tables 1–5** in `results/paper/eaai_camera_ready_tables/` |
| **Canonical one-pager** | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) (see also [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)) |
| **External baseline in progress** | **PaMOP** (IJCAI 2025) reproduction — see [`docs/PAMOP_REPRODUCTION_PLAN.md`](docs/PAMOP_REPRODUCTION_PLAN.md), `baselines/pamop/`; ORLM/OptMATH/DeepOR/OR-R1 all implemented/reconstructed for lightweight inference-preparation (no baseline has a runnable empirical result yet) |
| **Infrastructure / reruns** | Slurm `batch/learning/`; optional LLM baselines; Gemini [`docs/GEMINI_RERUN_REPORT.md`](docs/GEMINI_RERUN_REPORT.md); Mistral [`docs/MISTRAL_RERUN_REPORT.md`](docs/MISTRAL_RERUN_REPORT.md) (**infra ≠ completed reruns** unless your `results/rerun/` proves it) |
| **Demo / app** | `app.py`, `demo/` — **outside** paper-evaluated claims unless explicitly scoped |
| **Archives** | `docs/archive/`, `docs/archive_internal_status/`, `docs/provenance/`, `analysis/archive/` — **provenance only** |
| **External validation (non–paper-core)** | **Text2Zinc** + **CP-Bench** (DCP-Bench-Open): adapters + staging docs — **no new camera-ready metrics** until runs exist ([`docs/DATASET_EXPANSION_STATUS.md`](docs/DATASET_EXPANSION_STATUS.md)) |

**Table 1 headline** (TF-IDF + typed greedy, `orig`): Schema R@1 **0.9094**; Coverage **0.8609**; TypeMatch **0.7453**; InstantiationReady **0.5287** — matches `manuscript/main.tex` **as submitted**. **This number does not reproduce from the current codebase** (2026-08-12 finding): a fresh rerun of the identical method gives InstantiationReady **0.7764**, because 49 commits of grounding fixes landed after `results/eswa_revision/13_tables/postfix_main_metrics.csv` was last generated and it was never regenerated. Full audit: [`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`](docs/BASELINE_STALENESS_AUDIT_2026-08-12.md) — **read this before citing 0.5287 as a current-code baseline for anything.** `manuscript/main.tex` and the camera-ready tables were **not** modified; see [`docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md`](docs/MANUSCRIPT_INTEGRATION_DECISION_2026-08-12.md).

**Correction (2026-08-12, same day):** an earlier note here claimed an existing assignment mode (`max_weight_matching`) reaches InstantiationReady 0.7432 and beats typed greedy. That comparison used the stale 0.5287 baseline above; against a freshly rerun typed greedy (0.7764), `max_weight_matching` **loses** (p=0.042), as do `search_structured_grounding`/`hierarchical_structured_grounding`. These are negative results — see [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) NR12 and the staleness audit above for the full correction.

---

## Read these first (repo map)

0. [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — **start here**: goal, pipeline, authoritative results, what's implemented/failed, PaMOP status, next steps
0a. [`docs/BASELINE_STALENESS_AUDIT_2026-08-12.md`](docs/BASELINE_STALENESS_AUDIT_2026-08-12.md) — **read second**: the manuscript's headline number does not reproduce from current code
1. [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md) — what is official vs auxiliary  
2. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — validated vs experimental, limitations  
3. [`docs/KAIS_SOURCE_OF_TRUTH.md`](docs/KAIS_SOURCE_OF_TRUTH.md) — current manuscript authority / scope (see also [`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md) for unchanged benchmark facts)  
4. [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) — metrics and provenance chain  
5. [`docs/HOW_TO_REPRODUCE.md`](docs/HOW_TO_REPRODUCE.md) — rerun commands  
6. [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — blockers and design tensions  
7. [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) / [`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md) — annotated tree (canonical vs demo vs archive)
8. [`docs/PAMOP_REPRODUCTION_PLAN.md`](docs/PAMOP_REPRODUCTION_PLAN.md) — current external-baseline work in progress
9. [`docs/BASELINE_IMPLEMENTATION_ROADMAP.md`](docs/BASELINE_IMPLEMENTATION_ROADMAP.md) — ORLM/OptMATH/DeepOR/OR-R1 planning and implementation status
10. [`docs/SCIENTIFIC_STATE.md`](docs/SCIENTIFIC_STATE.md) / [`NEXT_STEPS.md`](NEXT_STEPS.md) — detailed scientific handoff and the operational execution queue

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
