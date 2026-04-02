# Retrieval-Assisted Optimization Schema Grounding

A research codebase for **retrieval-assisted instantiation of natural-language optimization
problems**, as described in the companion **EAAI** manuscript. The **paper-evaluated path**
is: **schema retrieval** (NL query → catalog schema) → **deterministic scalar grounding**
(numeric mentions → parameter slots) on the **NLP4LP** benchmark, plus **restricted**
engineering and solver-backed studies documented in the camera-ready tables.

## Repository status

> **Companion research repository** (EAAI manuscript), not a production product.

**Single source of truth for reviewers:** **[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)**  
Headline numbers, limitations, and what is validated vs auxiliary are summarized there and in
**[`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md)**.

| | |
|---|---|
| **Paper / core pipeline** | NLP4LP `orig` (331 queries): retrieval + deterministic grounding; camera-ready tables in **`results/paper/eaai_camera_ready_tables/`** |
| **Auxiliary (demo / app / optional APIs)** | Gradio **`app.py`**, optional **OpenAI / Gemini** baselines (`tools/llm_baselines.py`), telemetry hooks — **not** the main evaluated claim set unless stated otherwise |
| **Experimental / non-canonical** | Learned retrieval fine-tuning, GCG and other extended grounding modes, HPC learning runs — see **`EXPERIMENTS.md`** and `docs/learning_runs/` |

**Manuscript-aligned headline (Table 1, TF-IDF + typed greedy):** Schema R@1 = **0.9094**; downstream columns in the same row are **0.8639 / 0.7513 / 0.5257** (coverage / type match / instantiation-ready as defined in that table — see CSV).  
Do **not** mix these with ad hoc columns from other CSVs without checking definitions (**[`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)**).

## Project scope (EAAI manuscript)

This system retrieves a compatible optimization schema from a fixed catalog, deterministically
instantiates scalar parameters from numeric evidence in NL text, and supports restricted
engineering-oriented downstream validation on subsets of the NLP4LP benchmark.

**The paper story in brief:**
- Main benchmark on NLP4LP (331 test queries, `orig` variant)
- Retrieval is strong on `orig` (TF-IDF Schema R@1 = **0.9094** in Table 1)
- Main bottleneck is downstream number-to-slot grounding
- Three engineering-oriented validations: structural subset (60 instances), executable-attempt subset (269 instances with documented blockers), and a solver-backed restricted subset (20 instances, SciPy HiGHS shim)

> **Demo / open-domain:** The Gradio app can surface **LLM-based formulation paths** for unknown
> problem types. That path is **outside** the main paper evaluation unless explicitly documented
> elsewhere. The **benchmarked** system assumes a **fixed catalog** of known schemas.

## What this repo does not claim

- Full natural-language-to-optimization **compilation** for arbitrary problems (solver-ready output is a restricted subset only)
- **Benchmark-wide solver readiness** — not all NLP4LP instances are executable end-to-end
- **Dense retrieval (E5/BGE) as primary results** — supplementary; TF-IDF is the paper's primary retrieval baseline
- That the **learned retrieval model beats the rule baseline** — it does not (see **`KNOWN_ISSUES.md`**)
- That **Gurobi is available or required** — paper solver-backed results use a **SciPy HiGHS shim** on a small subset

## Reproducibility and requirements

| Mode | Needs |
|------|--------|
| **Offline / no HF** | Unit tests (`pytest`), structural checks, repo validation scripts (e.g. `scripts/paper/run_repo_validation.py`), figure rebuild from **existing** tables |
| **Gated HuggingFace NLP4LP** | `HF_TOKEN` (or CLI login) + dataset access — required to **recompute** full benchmark metrics and gold-parameter grounding |
| **Solver-backed paper subset (Table 4)** | SciPy stack as in `requirements.txt`; **no** Gurobi for those results |
| **Paper-validated artifacts** | Tables 1–5 + figures under **`results/paper/`** and reports under **`analysis/eaai_*.md`** |
| **Demo-only** | Gradio app; optional LLM API keys for non-benchmark tooling |

Intermediate audits and internal decision logs are in **`docs/archive_internal_status/`** (provenance; not the canonical headline source).

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

Source: `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv`
(provenance: `results/eswa_revision/13_tables/deterministic_method_comparison_orig.csv`)

| Method | Schema | Coverage | TypeMatch | Exact20 | InstReady |
|--------|--------|----------|-----------|---------|-----------|
| Typed Greedy | TF-IDF | 0.8639 | 0.7513 | 0.1991 | 0.5257 |
| Typed Greedy | BM25 | 0.8509 | 0.7386 | 0.2057 | 0.5196 |
| Typed Greedy | Oracle | 0.9151 | 0.8030 | 0.1882 | 0.5650 |
| Opt-Role Repair | TF-IDF | 0.8248 | 0.7036 | 0.2847 | 0.4411 |
| Constrained | TF-IDF | 0.8112 | 0.7113 | 0.3293 | 0.4230 |

### Engineering-oriented validation subsets

| Subset | Instances | Key result |
|--------|-----------|-----------|
| Structural subset (Table 2) | 60 | End-to-end structural consistency |
| Executable-attempt subset (Table 3) | 269 | Full execution with documented blockers |
| Solver-backed subset (Table 4) | 20 | TF-IDF: **80% feasible** via SciPy HiGHS |

---

## Recent improvements

- **Min/max ordering enforcement** — `_is_partial_admissible` now rejects partial
  assignments where the value placed in a min-slot exceeds the paired max-slot value
  (e.g. `MinDemand > MaxDemand`), directly eliminating the `lower_vs_upper_bound`
  failure family.  A new `_slot_stem()` helper pairs bound slots by quantity stem
  (`MinDemand`/`MaxDemand` → `"demand"`, `LowerBound`/`UpperBound` → `"bound"`).
- **Float type-match fixes** — Five root causes of near-zero float TypeMatch were
  resolved: `_is_type_match("float","int")` now returns `True`; `_expected_type`
  no longer misclassifies quantity-constraint keywords as currency; large non-monetary
  numbers are no longer mis-tagged as currency; `_choose_token` gives integer/float
  tokens correct priority for currency slots.  Verified by 43 targeted tests.
- **Short-query retrieval** — `_DOMAIN_EXPANSION_MAP` extended with six new problem
  families (LP/MIP/ILP, QP, portfolio, bipartite matching, inventory, cutting/packing).
- **LP structural consistency checks** — `formulation/verify.py` catches invalid
  objective sense and missing variable symbols without requiring a solver.
- **Bound-role annotation layer** — Deterministic min/max operator-phrase recognition,
  fine-grained `bound_role` field on `MentionOptIR`, range-expression detection
  (`between X and Y`), wrong-direction penalties, and bound-flip swap repair.

---

## Architecture

```
Natural-language query
        │
        ▼
┌───────────────────┐
│  Schema Retrieval │  TF-IDF / BM25 / LSA / SBERT / E5 / BGE
│  (retrieval/)     │  → top-1 schema ID  (Schema R@1 ≈ 0.9094 on NLP4LP orig, Table 1)
└────────┬──────────┘
         │  predicted schema (slot names + types)
         ▼
┌───────────────────┐
│  Numeric Mention  │  regex tokenisation, type tagging (int/float/
│  Extraction       │  currency/percent), operator-cue detection,
│  (tools/nlp4lp…) │  bound-role annotation, range expressions
└────────┬──────────┘
         │  MentionOptIR list
         ▼
┌───────────────────┐
│  Slot Assignment  │  typed greedy │ constrained DP │ semantic IR
│  + Repair         │  repair │ optimization-role repair │ GCG
└────────┬──────────┘
         │  slot → value mapping
         ▼
┌───────────────────┐
│  Output           │  ILP/LP formulation + instantiated parameter slots;
│                   │  structural consistency check; solver code (demo only)
└───────────────────┘
```

For **known problems** the formulation is fetched from the catalog.
Structural validation checks LP consistency without a live solver.
Solver execution is supported on a restricted subset via the SciPy HiGHS shim
(see `tools/run_eaai_final_solver_attempt.py` and Table 4 in the paper).

---

## Quick start

1. **Clone and install**
   ```bash
   git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
   cd combinatorial-opt-agent
   python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Run the web app**
   ```bash
   python app.py
   ```
   Open the URL shown (e.g. **http://127.0.0.1:7860**). Type a short description of your optimization problem and click **Search**. The bot returns the best-matching problem(s) and their integer program (variables, objective, constraints).
3. **Optional — command line**
   ```bash
   python -m retrieval.search "minimize cost of opening warehouses and assigning customers" 3
   ```

**Note:** When the app is run (e.g. on a server), every search is logged to `data/collected_queries/user_queries.jsonl` so you can use real user prompts for training. See [training/README.md](training/README.md).

---

## Data Collection (demo / app only)

> **Note:** Data collection is part of the **demo application** (`app.py`), not the
> paper-evaluated pipeline. It is irrelevant for benchmark reproduction.

When a user submits a query via the web UI, the app logs the interaction to a local
file (`data/collected_queries/user_queries.jsonl`, gitignored). Remote telemetry
is opt-in via `TELEMETRY_REPO` / `TELEMETRY_TOKEN` env vars. No PII is collected.
Set `.env.example` → `.env` to configure. See `telemetry.py` for full details.

---

## Paper artifacts / EAAI manuscript support

All camera-ready artifacts for the EAAI manuscript live under `results/paper/`.

### Tables

| Table | File | Description |
|-------|------|-------------|
| Table 1 | `results/paper/eaai_camera_ready_tables/table1_main_benchmark_summary.csv` | Main NLP4LP benchmark (TF-IDF, BM25, Oracle) |
| Table 2 | `results/paper/eaai_camera_ready_tables/table2_engineering_structural_subset.csv` | Engineering structural subset (60 instances) |
| Table 3 | `results/paper/eaai_camera_ready_tables/table3_executable_attempt_with_blockers.csv` | Executable-attempt study with blockers (269 instances) |
| Table 4 | `results/paper/eaai_camera_ready_tables/table4_final_solver_backed_subset.csv` | Final solver-backed subset (20 instances, SciPy shim) |
| Table 5 | `results/paper/eaai_camera_ready_tables/table5_failure_taxonomy.csv` | Cross-experiment failure taxonomy |
| Bundle | `results/paper/eaai_camera_ready_tables/camera_ready_tables.md` | All tables in Markdown |

### Figures

| Figure | Files | Description |
|--------|-------|-------------|
| Figure 1 | `results/paper/eaai_camera_ready_figures/figure1_pipeline_overview.{png,pdf}` | Pipeline overview schematic |
| Figure 2 | `results/paper/eaai_camera_ready_figures/figure2_main_benchmark_comparison.{png,pdf}` | Main benchmark comparison |
| Figure 3 | `results/paper/eaai_camera_ready_figures/figure3_engineering_validation_comparison.{png,pdf}` | Engineering validation subset |
| Figure 4 | `results/paper/eaai_camera_ready_figures/figure4_final_solver_backed_subset.{png,pdf}` | Final solver-backed subset |
| Figure 5 | `results/paper/eaai_camera_ready_figures/figure5_failure_breakdown.{png,pdf}` | Failure breakdown (appendix) |

To regenerate figures from the authoritative tables:

```bash
pip install Pillow>=9.0.0
python tools/build_eaai_camera_ready_figures.py
```

### Experiment reports (provenance)

- `analysis/eaai_engineering_subset_report.md` — Engineering subset (60 instances)
- `analysis/eaai_executable_subset_report.md` — Executable-attempt study (269 instances)
- `analysis/eaai_final_solver_attempt_report.md` — Final solver-backed subset (20 instances)
- `analysis/eaai_tables_build_report.md` — Table provenance and conflict resolution
- `analysis/eaai_figures_build_report.md` — Figure build notes
- `analysis/eaai_figures_reproduction_report.md` — Figure reproduction log
- `docs/EAAI_SOURCE_OF_TRUTH.md` — Canonical paper framing and authoritative file list
- `docs/RESULTS_PROVENANCE.md` — Canonical metrics + full provenance chain

### Historical note

Older docs in `docs/archive/` contain earlier ESWA-era experiment records and development
notes. They are preserved for history but should not be cited as authoritative for the
current EAAI manuscript.

---

## How to reproduce the main paper artifacts

> Full benchmark rerun requires the gated `udell-lab/NLP4LP` dataset (HuggingFace
> access required). See [HuggingFace dataset access](#huggingface-dataset-access).

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate repo integrity (does NOT need HF token)
python scripts/paper/run_repo_validation.py

# 3. Re-run EAAI experiments (requires HF_TOKEN)
python tools/run_eaai_engineering_subset_experiment.py   # Table 2 (60 instances)
python tools/run_eaai_executable_subset_experiment.py    # Table 3 (269 instances)
python tools/run_eaai_final_solver_attempt.py            # Table 4 (20 instances)

# 4. Regenerate figures from tables
python tools/build_eaai_camera_ready_figures.py
```

See **[`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md)** for the full step-by-step guide.

---

## Repository map

| Path | Purpose | Authority |
|------|---------|-----------|
| `results/paper/` | Camera-ready tables and figures | ★ Authoritative |
| `analysis/eaai_*` | EAAI experiment reports | ★ Authoritative |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Master paper framing | ★ Authoritative |
| `docs/RESULTS_PROVENANCE.md` | Canonical metrics + provenance chain | ★ Authoritative |
| `docs/CURRENT_STATUS.md` | Reviewer-facing status | ★ Summary |
| `tools/nlp4lp_downstream_utility.py` | Core grounding pipeline | Core source |
| `retrieval/` | Schema retrieval methods | Core source |
| `formulation/verify.py` | LP structural validation | Core source |
| `tests/` | Pytest suite (1 400+ tests, CPU-only) | Core source |
| `scripts/paper/` | Paper support scripts | Core source |
| `demo/` | Demo app documentation | Demo only |
| `docs/archive_internal_status/` | Internal audits / decision logs | ⚠ Provenance only |
| `docs/archive/` | Historical dev notes | ⚠ Non-authoritative |
| `docs/eswa_revision/` | ESWA-era materials | ⚠ Non-authoritative |
| `results/eswa_revision/` | Earlier experiment results | ⚠ Non-authoritative |

See **[`REPO_STRUCTURE.md`](REPO_STRUCTURE.md)** for the full annotated directory map.

---

## Documentation

| Topic | Doc |
|-------|-----|
| **Reviewer guide** | [`docs/REVIEWER_GUIDE.md`](docs/REVIEWER_GUIDE.md) — what to look at, headline metrics, limitations |
| **Current status (reviewers)** | [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — headline metrics pointer, validated vs auxiliary |
| **Manuscript authority** | [`docs/EAAI_SOURCE_OF_TRUTH.md`](docs/EAAI_SOURCE_OF_TRUTH.md) — paper framing and authoritative file list |
| **Results provenance** | [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md) — canonical metrics + provenance chain |
| **Reproduction steps** | [`HOW_TO_REPRODUCE.md`](HOW_TO_REPRODUCE.md) |
| **Experiments overview** | [`EXPERIMENTS.md`](EXPERIMENTS.md) — retrieval, grounding methods, learning |
| **Known issues** | [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) |
| **Paper vs demo scope** | [`docs/paper_vs_demo_scope.md`](docs/paper_vs_demo_scope.md) |
| **Repo structure** | [`REPO_STRUCTURE.md`](REPO_STRUCTURE.md) |
| **Wulver (HPC)** | [`docs/wulver.md`](docs/wulver.md) — NJIT cluster setup and batch jobs |
| **Training** | [`training/README.md`](training/README.md) — retrieval fine-tuning |
| **Learning (NLP4LP)** | [`docs/learning_runs/`](docs/learning_runs/README.md) — benchmark-safe splits, experiment records |
| **Internal status archives** | [`docs/archive_internal_status/`](docs/archive_internal_status/README.md) — provenance only |
| **Historical docs** | [`docs/archive/`](docs/archive/README.md) — development notes (non-authoritative) |
| **Contributing** | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| **Demo app** | [`demo/README.md`](demo/README.md) |

---

## HuggingFace dataset access

Several downstream scripts require the gated `udell-lab/NLP4LP` dataset.

```bash
# Safe setup — never paste your token into code or a chat:
cp .env.example .env   # .env is gitignored — it will NOT be committed
# Edit .env: set HF_TOKEN=hf_...
# Request access: https://huggingface.co/datasets/udell-lab/NLP4LP
export $(grep -v '^#' .env | xargs)
```

Or authenticate once with the HuggingFace CLI:
```bash
pip install huggingface_hub && huggingface-cli login
```

**For CI / GitHub Actions** — add `HF_TOKEN` as a repository secret (Settings → Secrets and variables → Actions → New repository secret).

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

**Soroush Vahidi** — NJIT  
- Email: [sv96@njit.edu](mailto:sv96@njit.edu)  
- GitHub: [@SoroushVahidi](https://github.com/SoroushVahidi)

---

## Developer workflows

For second remotes, worktrees, or AI coding assistants against another clone, see **[CONTRIBUTING.md](CONTRIBUTING.md)** and your tool's docs. Use `git remote -v` and provider authentication (`gh auth login`, SSH) as usual.
