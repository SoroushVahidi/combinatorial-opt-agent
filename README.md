# Retrieval-Assisted Optimization Schema Grounding

A research codebase for **retrieval-assisted instantiation of natural-language optimization
problems**, as described in the companion EAAI manuscript.  The pipeline combines a
**schema retrieval stage** (NL query → problem schema) with a **deterministic scalar
grounding stage** (NL numeric mentions → parameter slots) so that the retrieved
formulation is numerically instantiated from the user's own problem data.

## Repository status

> **This is a companion research repository** for an EAAI submission, not a production tool.

| Item | Details |
|------|---------|
| **Benchmark scope** | Fixed-catalog NLP4LP (`orig` variant, 331 test queries) |
| **Core evaluated task** | Schema retrieval + deterministic scalar parameter grounding |
| **Solver-backed results** | Restricted to a 20-instance subset via SciPy HiGHS shim |
| **Gurobi** | NOT required; paper results use SciPy only |
| **HF dataset** | Full benchmark rerun requires the gated `udell-lab/NLP4LP` dataset |
| **LLM generation path** | Present in demo app; outside the scope of the paper evaluation |

**Core capabilities:**

| Capability | What it does |
|---|---|
| **Problem recognition** | TF-IDF / dense retrieval matches queries to 90+ known problem types (facility location, knapsack, scheduling, …) at **Schema R@1 = 0.906** |
| **Formulation retrieval** | Returns the ILP/LP formulation (variables, objective, constraints) for the matched schema |
| **Deterministic scalar grounding** | Assigns numeric mentions from the NL query to schema parameter slots; best method: **Optimization-Role Repair** (Coverage 0.822 · TypeMatch 0.243 · Exact20 0.277) |
| **Structural validation** | LP structural consistency checks (invalid objective sense, missing variable symbols) without requiring a live solver |
| **Restricted solver execution** | Real solver outcomes on a 20-instance subset via SciPy HiGHS shim (see Table 4) |
| **Demo app** | Gradio web UI for interactive schema search; GAMSPy/GAMS example collection (outside paper scope) |

## Project scope (EAAI manuscript)

This system retrieves a compatible optimization schema from a fixed catalog, deterministically
instantiates scalar parameters from numeric evidence in NL text, and supports restricted
engineering-oriented downstream validation on subsets of the NLP4LP benchmark.

**The paper story in brief:**
- Main benchmark on NLP4LP (331 test queries, `orig` variant)
- Retrieval is already strong (TF-IDF Schema R@1 = 0.9094)
- Main bottleneck is downstream number-to-slot grounding
- Three engineering-oriented validations: structural subset (60 instances), executable-attempt subset (269 instances with documented blockers), and a solver-backed restricted subset (20 instances, real nonzero outcomes)

> **Note:** The demo app also supports querying unknown problem types via an LLM-generation
> path. This feature is outside the scope of the EAAI manuscript and is provided for
> demonstration purposes only.

## What this repo does not claim

- Full natural-language-to-optimization **compilation** for arbitrary problems (solver-ready output is a restricted subset only)
- **Benchmark-wide solver readiness** — not all NLP4LP instances are executable
- **Dense retrieval (E5/BGE) as primary results** — these are supplementary; TF-IDF is the paper's primary retrieval baseline
- That the **learned retrieval model beats the rule baseline** — it does not (see `KNOWN_ISSUES.md`)
- That **Gurobi is available or required** — paper results use SciPy HiGHS shim for the solver subset

## Current evidence-based status

| Component | Status | Key metric |
|---|---|---|
| Problem retrieval (TF-IDF, orig queries) | ✅ Strong | Schema R@1 = **0.906** |
| Problem retrieval (short / first-sentence queries) | ⚠️ Degraded | Schema R@1 = 0.786 |
| Downstream grounding — typed greedy (primary baseline) | ✅ Reproducible | Coverage 0.822, TypeMatch 0.226, InstReady 0.076 |
| Downstream grounding — optimization-role repair (recommended) | ✅ Best deterministic | Coverage 0.822, TypeMatch 0.243, Exact20 0.277 |
| Downstream grounding — global consistency grounding (GCG) | 🔬 Experimental | Unit-tested; full-run needs HF gold data |
| Learned retrieval fine-tuning | ⚠️ Future work | Real-data run did not beat rule baseline |
| Structural validation (no live solver) | ✅ Available | LP objective/variable consistency checks in `formulation/verify.py` |
| Solver-backed restricted subset (20 instances, SciPy HiGHS shim) | ✅ Real outcomes | TF-IDF: 80% feasible; Oracle: 75% feasible |

All claims are **benchmark-scoped** (NLP4LP `orig` variant, 331 test queries). See the
[EAAI paper artifacts](#paper-artifacts--eaai-manuscript-support) section for camera-ready
tables and figures.

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

## Architecture

```
Natural-language query
        │
        ▼
┌───────────────────┐
│  Schema Retrieval │  TF-IDF / BM25 / LSA / SBERT / E5 / BGE
│  (retrieval/)     │  → top-1 schema ID  (Schema R@1 = 0.906)
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

## Quick start (use the bot)

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

**Note:** When the app is run (e.g. on a server), every search is logged to `data/collected_queries/user_queries.jsonl` so you can use real user prompts for training. See [Training the retrieval model](#training-the-retrieval-model) and [training/README.md](training/README.md).

## Connect Codex to another repository

If you want to run Codex against a different project folder/repository, use one of the
workflows below.

### Option A: Open Codex in a new local clone

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
codex
```

### Option B: Add a second remote to the current repository

```bash
git remote add upstream https://github.com/<owner>/<repo>.git
git fetch upstream
git remote -v
```

### Option C: Worktree for side-by-side repositories/branches

```bash
git worktree add ../<repo>-alt <branch-or-ref>
cd ../<repo>-alt
codex
```

### Authentication notes

- For private repositories, authenticate first via your Git provider CLI (for example,
  `gh auth login`) or SSH keys.
- Confirm access with `git ls-remote <repo-url>` before launching Codex.

## Data Collection

> **⚠️ Please read this section before deploying or distributing the application.**

When a user submits a query in the web UI, the application logs the interaction
for training purposes.  Logging happens at **two levels**:

| Level | Where | When |
|---|---|---|
| **Local file** | `data/collected_queries/user_queries.jsonl` on the server's disk | Always (every query) |
| **Private GitHub repository** | `queries/<date>/<session-id>.jsonl` inside the repo named by `TELEMETRY_REPO` | Only when `TELEMETRY_REPO` **and** `TELEMETRY_TOKEN` env vars are set |

### What is collected

Each record contains **only**:

```json
{
  "ts":      "2026-03-16T18:21:36.000000+00:00",
  "query":   "minimize cost of opening warehouses and assigning customers",
  "top_k":   3,
  "results": [
    {"id": "facility_location", "name": "Facility Location", "score": 0.93},
    ...
  ]
}
```

No personally identifiable information (PII) is ever collected — no IP addresses,
browser fingerprints, user accounts, session cookies, or any other identifying data.

### How to enable remote telemetry

1. Create a **private** GitHub repository (e.g. `YourOrg/opt-agent-telemetry`).
2. Generate a GitHub personal-access-token with **`repo` scope** for that repository
   (or a fine-grained token with *Contents: Read & write*).
   See <https://github.com/settings/tokens>.
3. Copy `.env.example` → `.env` and fill in:
   ```
   TELEMETRY_REPO=YourOrg/opt-agent-telemetry
   TELEMETRY_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
   ```
4. The app reads these variables at startup.  The Gradio footer will show
   *"Queries are also pushed to a private GitHub repository for training"*
   when telemetry is active.

### How to opt out

Simply leave `TELEMETRY_REPO` and `TELEMETRY_TOKEN` unset (or empty).  When these
variables are absent the module (`telemetry.py`) is a complete no-op — no network
request is made and no data leaves the machine beyond the local log file.

The local log file (`data/collected_queries/user_queries.jsonl`) is listed in
`.gitignore` and is never committed to the public repository.

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

### Historical note

Older docs in `docs/archive/` contain earlier ESWA-era experiment records and development
notes. They are preserved for history but should not be cited as authoritative for the
current EAAI manuscript.

## How to reproduce the main paper artifacts

> Full benchmark rerun requires the gated `udell-lab/NLP4LP` dataset (HuggingFace
> access required). See [Benchmark access requirements](#huggingface-dataset-access).

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

## Repo map

| Path | Purpose | Authority |
|------|---------|-----------|
| `results/paper/` | Camera-ready tables and figures | ★ Authoritative |
| `analysis/eaai_*` | EAAI experiment reports | ★ Authoritative |
| `docs/EAAI_SOURCE_OF_TRUTH.md` | Master paper framing | ★ Authoritative |
| `tools/nlp4lp_downstream_utility.py` | Core grounding pipeline | Core source |
| `retrieval/` | Schema retrieval methods | Core source |
| `formulation/verify.py` | LP structural validation | Core source |
| `tests/` | Pytest suite (1400+ tests, CPU-only) | Core source |
| `scripts/paper/` | Paper support scripts | Core source |
| `docs/archive/` | Historical dev notes | ⚠ Non-authoritative |
| `docs/eswa_revision/` | Earlier ESWA materials | ⚠ Non-authoritative |
| `results/eswa_revision/` | Earlier experiment results | ⚠ Non-authoritative |

## Documentation

| Topic | Doc |
|-------|-----|
| **Experiments** | [EXPERIMENTS.md](EXPERIMENTS.md) — Consolidated overview of all experiments (retrieval, grounding methods, learning, copilot comparison) |
| **Data sources** | `data/sources/` — Machine-readable manifests |
| **Wulver (HPC)** | [docs/wulver.md](docs/wulver.md) — NJIT cluster setup and batch jobs |
| **Training** | [training/README.md](training/README.md) — retrieval fine-tuning; mention-slot scorer in `training/` |
| **Learning (NLP4LP)** | [docs/learning_runs/](docs/learning_runs/README.md) — benchmark-safe splits, real-data-only check, experiment records |
| **Historical docs** | [docs/archive/](docs/archive/README.md) — development notes (non-authoritative) |

Private data (GAMSPy models, license-related files) live under **`data_private/`** (gitignored). Manifests and catalogs are in `data_private/gams_models/manifests/` and `catalog/`.

## Data sources

The benchmark dataset is NLP4LP (`udell-lab/NLP4LP` on HuggingFace; gated access required).
The problem catalog is built from public sources including NL4Opt, OR-Library, and Gurobi
modeling examples. Machine-readable manifests are in `data/sources/`.

## HuggingFace dataset access

Several scripts (e.g. `training/external/nlp4lp_loader.py`, `training/external/build_nlp4lp_benchmark.py`)
load gated datasets such as `udell-lab/NLP4LP` from the HuggingFace Hub.
To use them you need a HuggingFace account with access approved on the dataset page,
and a personal access token configured locally.

**Safe setup — never paste your token into code or a chat:**

```bash
# 1. Copy the example env file
cp .env.example .env          # .env is gitignored — it will NOT be committed

# 2. Edit .env and replace the placeholder with your real token
#    Get a (read-only) token at: https://huggingface.co/settings/tokens
#    Then set:  HF_TOKEN=hf_...

# 3. Load the variable into your shell (or let your IDE load .env automatically)
export $(grep -v '^#' .env | xargs)

# 4. Verify
python -c "import os; print('HF_TOKEN set:', bool(os.environ.get('HF_TOKEN')))"
```

Alternatively, authenticate once with the HuggingFace CLI (token is stored in
`~/.cache/huggingface/token` and is never written to the repo):

```bash
pip install huggingface_hub
huggingface-cli login        # paste your token at the prompt; choose read-only
```

All scripts automatically pick up the token from `HF_TOKEN`, `HUGGINGFACE_TOKEN`,
or the cached CLI token — in that priority order.

**For CI / GitHub Actions** — add the token as a repository secret (token never appears in logs):

1. Go to **Settings → Secrets and variables → Actions → New repository secret**
2. Name: `HF_TOKEN`, Value: your token (starts with `hf_...`)
3. The `NLP4LP benchmark` workflow (`.github/workflows/nlp4lp.yml`) will pick it up automatically.
   Trigger it from the **Actions** tab → **NLP4LP benchmark** → **Run workflow**.

## How to run

1. Clone the repo and install dependencies (see `requirements.txt` or project docs).
2. Run the agent (e.g. CLI or web interface) and input a natural-language description of your optimization problem.
3. Use the generated formulation and solver code with your preferred solver.

See in-repo documentation for API keys (if using hosted LLMs), environment variables, and example prompts.

### Option A: GitHub Codespaces

1. **Code → Codespaces → Create codespace on main**
2. Wait for the environment to build (~2 minutes)
3. In the terminal: `python pipeline/run_collection.py`

### Option B: Local Setup

```bash
# Clone the repo
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the collection pipeline
python pipeline/run_collection.py
```

### Refresh catalog (add public data)

To fetch public NL4Opt data and merge it into the bot's catalog (1,100+ problems):

```bash
bash setup_catalog.sh
# or step by step:
pip install -r requirements.txt
python pipeline/run_collection.py
```

### Try the retrieval (query → problem + IP)

**Option 1 — Web UI (recommended):** A window in your browser where you type and get an answer.

```bash
pip install -r requirements.txt
python app.py
```

Then open the URL shown (e.g. http://127.0.0.1:7860), type your problem in the text box, and click Submit. The bot shows the best-matching problem(s) and their integer program.

**Option 2 — Command line:**

```bash
pip install -r requirements.txt
python run_search.py "minimize cost of opening warehouses and assigning customers"
python run_search.py "knapsack" 2
```

First run will download the sentence-transformers model (~90MB). Results show the best-matching problem(s) and their integer program (variables, objective, constraints).

### Option C: HPC (Wulver @ NJIT)

You can run the same retrieval bot on NJIT’s Wulver cluster. See **[docs/wulver.md](docs/wulver.md)** for step-by-step setup. Summary:

```bash
# SSH to Wulver, then:
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
module load python/3.10
python -m venv venv
source venv/bin/activate   # or: venv/Scripts/activate on Windows
pip install -r requirements.txt

# Run a query (interactive)
python run_search.py "minimize cost of opening warehouses"

# Or submit as a batch job (optional)
sbatch scripts/run_search.slurm
```

## Training

- **Retrieval model** — Fine-tune the sentence-transformers model so it better matches NL queries to problems in the catalog.  
  **Full guide:** [training/README.md](training/README.md)  
  Steps: generate synthetic (query, passage) pairs → optional: add collected user queries → run training (local or GPU batch on Wulver, see [docs/wulver.md](docs/wulver.md)).  
  **Evaluation:** `python -m training.evaluate_retrieval --regenerate --num 500` for Precision@1 / Precision@5 on 500 held-out instances.

- **Mention–slot scorer** (NLP4LP) — For constrained assignment (NL numeric mentions → schema slots). Generate pairs with `training/generate_mention_slot_pairs.py` and train with `training/train_mention_slot_scorer.py`.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data processing | pandas, json, BeautifulSoup |
| Retrieval | TF-IDF (scikit-learn), BM25 (rank-bm25), LSA, Sentence-Transformers, E5, BGE |
| NLP / extraction | Regex-based numeric tokenisation, operator-cue detection, bound-role annotation |
| Assignment | Typed greedy, constrained DP, semantic IR repair, optimization-role repair, GCG |
| Optimization solvers | SciPy HiGHS shim (restricted subset, paper results); GAMSPy/Pyomo/PuLP (demo only, outside paper scope) |
| Web UI | Gradio |
| HPC | NJIT Wulver (SLURM) |
| CI/CD | GitHub Actions |
| Dev environment | GitHub Codespaces |

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text. Copyright (c) Soroush Vahidi.

## 🙏 Acknowledgments

- [NL4Opt Competition](https://nl4opt.github.io/) — NeurIPS 2022
- [Gurobi Optimization](https://www.gurobi.com/) — OptiMods & Modeling Examples
- [GAMS Development Corp.](https://www.gams.com/) — Model Library
- [MIPLIB](https://miplib.zib.de/) — Benchmark instances
- [J.E. Beasley](http://people.brunel.ac.uk/~mastjjb/jeb/info.html) — OR-Library
- [Pyomo Project](https://www.pyomo.org/) — Open-source optimization
- NJIT Department of Computer Science

## 📬 Contact

**Soroush Vahidi** — NJIT  
- Email: [sv96@njit.edu](mailto:sv96@njit.edu)  
- GitHub: [@SoroushVahidi](https://github.com/SoroushVahidi)

---

**Repository description** (for GitHub **Settings → General → Description**; update manually if needed):  
*Retrieval-assisted optimization schema grounding: NLP4LP benchmark, deterministic scalar instantiation (typed greedy, optimization-role repair, GCG), engineering-oriented validation. EAAI manuscript companion codebase.*
