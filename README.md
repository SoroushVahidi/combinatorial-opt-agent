# Retrieval-Assisted Optimization Schema Grounding

> **Research companion codebase** for the EAAI manuscript:
> *"Retrieval-Assisted Instantiation of Natural-Language Optimization Problems"*

The pipeline combines a **schema retrieval stage** (NL query → problem schema) with a
**deterministic scalar grounding stage** (NL numeric mentions → parameter slots),
enabling retrieval-assisted instantiation of optimization problems from natural-language
descriptions.

---

## Repository status

| Item | Details |
|------|---------|
| **Benchmark scope** | NLP4LP `orig` variant, 331 test queries |
| **Core task** | Schema retrieval + deterministic scalar parameter grounding |
| **Solver-backed results** | Restricted to a 20-instance subset via SciPy HiGHS shim |
| **Gurobi** | NOT required; paper results use SciPy only |
| **HF dataset** | Full benchmark rerun requires the gated `udell-lab/NLP4LP` dataset |
| **LLM generation** | Present in demo app; outside the scope of the paper evaluation |

---

## Current evidence-based results

All results are benchmark-scoped (NLP4LP `orig` variant, 331 test queries).

### Retrieval

| Method | Schema R@1 (orig) | Schema R@1 (noisy) | Schema R@1 (short) |
|--------|------------------|-------------------|-------------------|
| TF-IDF | **0.9094** | 0.9033 | 0.7855 |
| BM25 | 0.8822 | — | — |
| Oracle | 1.000 | — | — |

### Downstream grounding

| Method | Schema | Coverage | TypeMatch | Exact20 | InstReady |
|--------|--------|----------|-----------|---------|-----------|
| Typed Greedy | TF-IDF | 0.822 | 0.226 | 0.233 | 0.076 |
| Typed Greedy | Oracle | 0.870 | 0.240 | 0.204 | 0.082 |
| Opt-Role Repair | TF-IDF | 0.822 | 0.243 | 0.277 | — |
| Constrained | TF-IDF | 0.772 | 0.195 | 0.328 | 0.027 |

### Engineering-oriented validation subsets

| Subset | Instances | Key result |
|--------|-----------|-----------|
| Structural subset (Table 2) | 60 | End-to-end structural consistency |
| Executable-attempt subset (Table 3) | 269 | Full execution with documented blockers |
| Solver-backed subset (Table 4) | 20 | TF-IDF: **80% feasible** via SciPy HiGHS |

Source: `results/paper/eaai_camera_ready_tables/` — see `docs/RESULTS_PROVENANCE.md`.

---

## What this repo does and does not claim

**Claims:**
- Schema retrieval is strong: TF-IDF Schema R@1 = 0.9094 on NLP4LP `orig`
- Deterministic scalar grounding (typed greedy, optimization-role repair) is implemented and benchmarked
- Structural LP validation (without a live solver) is reproducible
- A 20-instance restricted subset achieves real nonzero solver outcomes

**Does not claim:**
- Full natural-language-to-optimization compilation for arbitrary problems
- Benchmark-wide solver readiness
- That the learned retrieval model beats the rule baseline (it does not — see `KNOWN_ISSUES.md`)
- That Gurobi is available or required

---

## Architecture

```
Natural-language query
        │
        ▼
┌──────────────────────┐
│   Schema Retrieval   │  TF-IDF / BM25 / LSA (primary); SBERT / E5 / BGE (supplementary)
│   retrieval/         │  → top-1 schema ID  (Schema R@1 = 0.9094)
└─────────┬────────────┘
          │  predicted schema (slot names + types)
          ▼
┌──────────────────────┐
│   Numeric Mention    │  regex tokenisation, type tagging, operator-cue detection,
│   Extraction         │  bound-role annotation, range expressions
│   tools/nlp4lp…      │
└─────────┬────────────┘
          │  MentionOptIR list
          ▼
┌──────────────────────┐
│   Slot Assignment    │  typed greedy · constrained DP · semantic IR repair
│   + Repair           │  · optimization-role repair · GCG (experimental)
└─────────┬────────────┘
          │  slot → value mapping
          ▼
┌──────────────────────┐
│   Output             │  ILP/LP formulation + instantiated parameters;
│                      │  LP structural consistency check; solver (restricted subset)
└──────────────────────┘
```

---

## Quick start

### Demo app (paper scope + LLM generation)

```bash
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py           # Open http://127.0.0.1:7860
```

### Command-line retrieval

```bash
python run_search.py "minimize cost of opening warehouses and assigning customers"
python run_search.py "knapsack" 2
```

### Reproduce paper results

See **[`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md)** for step-by-step instructions.

---

## Repository map

| Path | Purpose | Authority |
|------|---------|-----------|
| `results/paper/` | Camera-ready tables and figures | ★ Authoritative |
| `analysis/eaai_*` | EAAI experiment reports | ★ Authoritative |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Master paper framing | ★ Authoritative |
| `docs/RESULTS_PROVENANCE.md` | Canonical metrics + provenance chain | ★ Authoritative |
| `tools/nlp4lp_downstream_utility.py` | Core grounding pipeline | Core source |
| `retrieval/` | Schema retrieval methods | Core source |
| `formulation/verify.py` | LP structural validation | Core source |
| `tests/` | Pytest suite (1 400+ tests, CPU-only) | Core source |
| `scripts/paper/` | Paper support scripts | Core source |
| `demo/` | Demo app documentation | Demo only |
| `docs/archive/` | Historical dev notes | ⚠ Non-authoritative |
| `docs/eswa_revision/` | ESWA-era materials | ⚠ Non-authoritative |
| `docs/audits/` | Legacy audit/decision reports | ⚠ Non-authoritative |
| `results/eswa_revision/` | Earlier experiment results | ⚠ Non-authoritative |

See **[`REPO_STRUCTURE.md`](REPO_STRUCTURE.md)** for the full annotated directory map.

---

## Paper artifacts

All camera-ready artifacts live under `results/paper/`.

| Artifact | File |
|---------|------|
| Table 1 | `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` |
| Table 2 | `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv` |
| Table 3 | `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv` |
| Table 4 | `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv` |
| Table 5 | `results/paper/eaai_camera_ready_tables/table5_failure_taxonomy.csv` |
| Figures 1–5 | `results/paper/eaai_camera_ready_figures/figure*.{png,pdf}` |

---

## HuggingFace dataset access

Several downstream scripts require the gated `udell-lab/NLP4LP` dataset.

```bash
cp .env.example .env
# Set HF_TOKEN=hf_... in .env
# Request access: https://huggingface.co/datasets/udell-lab/NLP4LP
export $(grep -v '^#' .env | xargs)
```

Or authenticate via the HuggingFace CLI:
```bash
pip install huggingface_hub && huggingface-cli login
```

---

## Documentation

| Topic | Doc |
|-------|-----|
| Reproduction steps | [`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md) |
| Experiments overview | [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Known issues | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |
| Results provenance | [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) |
| Paper vs demo scope | [`docs/paper_vs_demo_scope.md`](docs/paper_vs_demo_scope.md) |
| Repo structure | [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md) |
| Paper framing | [`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| HPC (Wulver) | [`docs/wulver.md`](docs/wulver.md) |
| Demo app | [`demo/README.md`](demo/README.md) |

---

## Data collection notice

When the demo app (`app.py`) is running, every user query is logged locally to
`data/collected_queries/user_queries.jsonl` (gitignored, never committed).
Optional telemetry to a private GitHub repository is activated only when both
`TELEMETRY_REPO` and `TELEMETRY_TOKEN` environment variables are set.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Retrieval | TF-IDF (scikit-learn), BM25 (rank-bm25), LSA, Sentence-Transformers, E5, BGE |
| Grounding | Typed greedy, constrained DP, semantic IR repair, optimization-role repair, GCG |
| Solver | SciPy HiGHS shim (restricted 20-instance subset); demo also supports GAMSPy/Pyomo/PuLP |
| Web UI | Gradio |
| CI | GitHub Actions |
| HPC | NJIT Wulver (SLURM) |

---

## License

MIT License. See [LICENSE](LICENSE). Copyright © Soroush Vahidi.

## Acknowledgments

- [NL4Opt Competition](https://nl4opt.github.io/) — NeurIPS 2022
- [Gurobi Optimization](https://www.gurobi.com/) — OptiMods & Modeling Examples
- [GAMS Development Corp.](https://www.gams.com/) — Model Library
- MIPLIB, OR-Library, Pyomo Project
- NJIT Department of Computer Science

## Contact

**Soroush Vahidi** — NJIT · sv96@njit.edu · [@SoroushVahidi](https://github.com/SoroushVahidi)
