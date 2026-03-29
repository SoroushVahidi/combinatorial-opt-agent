# Combinatorial Optimization AI Agent

An **AI-powered agent** for **natural-language optimization**: describe a problem in plain English and get problem recognition, formulations, and solver-ready code.

**Capabilities:**

- **Problem recognition** — Match queries to a catalog of known combinatorial optimization problems (facility location, knapsack, scheduling, etc.).
- **Formulation retrieval** — Return ILP/LP formulations (variables, objective, constraints) for matched problems.
- **NLP4LP pipeline** — Schema retrieval, acceptance-aware reranking, constrained assignment (mention→slot), and optimization-role extraction for NL-to-optimization workflows.
- **GAMSPy integration** — Local GAMSPy/GAMS example collection and catalog for problem-family grouping and evaluation (see [GAMSPy docs](#documentation)).
- **Solver-ready code** — Pyomo, Gurobi, PuLP, and GAMSPy when applicable.

## Project vision

You describe a problem in plain English; the agent identifies the problem type (e.g. Uncapacitated Facility Location), returns the ILP/LP formulation, and can generate code you run with standard solvers. The project also supports research on **NL-to-optimization** (NLP4LP): schema acceptance, parameter instantiation, and optimization-role extraction.

## Architecture (high level)

- **Natural language** → entity and constraint extraction.
- **Problem classifier** (embeddings + LLM) matches against a **database of known problems** (90+ types).
- For **known problems:** retrieve formulation from the DB.
- For **unknown problems:** generate formulation via LLM and verify.
- **Output:** ILP formulation, LP relaxation, solver code (Pyomo/Gurobi/PuLP), and complexity class when applicable.
An **AI-powered agent** that translates plain-English optimization problem descriptions into
structured ILP/LP formulations and solver-ready code.  It combines a **retrieval pipeline**
(query → problem schema) with a **downstream grounding pipeline** (NL numeric mentions →
parameter slots) so that the generated formulation is both syntactically correct and
numerically instantiated from the user's own problem data.

**Core capabilities:**

| Capability | What it does |
|---|---|
| **Problem recognition** | TF-IDF / dense retrieval matches queries to 90+ known problem types (facility location, knapsack, scheduling, …) at **Schema R@1 = 0.906** |
| **Formulation retrieval** | Returns the ILP/LP formulation (variables, objective, constraints) for the matched schema |
| **Downstream grounding** | Assigns numeric mentions from the NL query to schema parameter slots; best method: **Optimization-Role Repair** (Coverage 0.822 · TypeMatch 0.243 · Exact20 0.277) |
| **GAMSPy integration** | Local GAMSPy/GAMS example collection, catalog grouping, and evaluation |
| **Solver-ready code** | Pyomo, Gurobi, PuLP, and GAMSPy output when applicable |

## Project vision

You describe a problem in plain English; the agent identifies the problem type (e.g.
Uncapacitated Facility Location), extracts the numeric parameters from your description,
and returns both the ILP/LP formulation and solver code ready to run.  The project also
advances research on **NL-to-optimization** (NLP4LP): schema acceptance, parameter
instantiation, optimization-role extraction, and incremental admissibility-constrained
decoding.

## Current evidence-based status

| Component | Status | Key metric |
|---|---|---|
| Problem retrieval (TF-IDF, orig queries) | ✅ Strong | Schema R@1 = **0.906** |
| Problem retrieval (short / first-sentence queries) | ⚠️ Degraded | Schema R@1 = 0.786 |
| Downstream grounding — typed greedy (primary baseline) | ✅ Reproducible | Coverage 0.822, TypeMatch 0.226, InstReady 0.076 |
| Downstream grounding — optimization-role repair (recommended) | ✅ Best deterministic | Coverage 0.822, TypeMatch 0.243, Exact20 0.277 |
| Downstream grounding — global consistency grounding (GCG) | 🔬 Experimental | Unit-tested; full-run needs HF gold data |
| Learned retrieval fine-tuning | ⚠️ Future work | Real-data run did not beat rule baseline |
| Solver-based output validation | ⚠️ Partial | Structural consistency checks only; no LP solver |

**Downstream grounding is the active research frontier** — all five assignment methods
(typed greedy, constrained, semantic IR repair, optimization-role repair, GCG) are
implemented and benchmarked; see [EXPERIMENTS.md](EXPERIMENTS.md) for full tables.

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
│  Output           │  ILP/LP formulation, LP relaxation,
│                   │  solver code (Pyomo / Gurobi / PuLP / GAMSPy)
└───────────────────┘
```

For **known problems** the formulation is fetched from the catalog; for **unknown
problems** it is generated via LLM and structurally verified.

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

## Documentation

| Topic | Doc |
|-------|-----|
| **Experiments** | [EXPERIMENTS.md](EXPERIMENTS.md) — Consolidated overview of all experiments (retrieval, grounding methods, learning, copilot comparison) |
| **Data sources** | [docs/data_sources.md](docs/data_sources.md) — OR-Library, Gurobi examples/OptiMods, NL4Opt, etc. |
| **GAMSPy** | [docs/GAMSPY_SETUP_AND_LICENSE.md](docs/GAMSPY_SETUP_AND_LICENSE.md), [GAMSPY_LOCAL_EXAMPLES_COLLECTION.md](docs/GAMSPY_LOCAL_EXAMPLES_COLLECTION.md), [GAMSPY_LOCAL_EXAMPLES_NEXT_STEPS.md](docs/GAMSPY_LOCAL_EXAMPLES_NEXT_STEPS.md) — setup, license, local example collection and manifests |
| **NLP4LP** | Acceptance rerank, constrained assignment, optimization-role method, semantic IR — see `docs/NLP4LP_*.md` |
| **Wulver (HPC)** | [docs/wulver.md](docs/wulver.md) — NJIT cluster setup and batch jobs |
| **Training** | [training/README.md](training/README.md) — retrieval fine-tuning; mention-slot scorer in `training/` |
| **Evaluation / paper** | `docs/BASELINE_TABLE_CLI.md`, `docs/PATCH_LEAK_FREE_EVAL.md`, and other experiment docs in `docs/` |
| **Learning (NLP4LP)** | [docs/learning_runs/](docs/learning_runs/README.md) — benchmark-safe splits, real-data-only check, experiment records |

Private data (GAMSPy models, license-related files) live under **`data_private/`** (gitignored). Manifests and catalogs are in `data_private/gams_models/manifests/` and `catalog/`.

## Data sources

The dataset is built from multiple authoritative sources:

- **[docs/data_sources.md](docs/data_sources.md)** — Canonical list with URLs, sizes, and problem/example names (OR-Library, Gurobi modeling examples, Gurobi OptiMods, etc.).
- **data/sources/** — Machine-readable manifests: `or_library.json`, `gurobi_modeling_examples.json`, `gurobi_optimods.json`, `index.json`.

Notable sources: [NL4Opt](https://github.com/nl4opt/nl4opt-competition) (NL→formulation), NLP4LP/OptiMUS, and GAMSPy examples (see [GAMSPy collection](docs/GAMSPY_LOCAL_EXAMPLES_COLLECTION.md)).

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

## Testing on iPhone (or any phone / tablet)

The web app is a full **Progressive Web App (PWA)** — it works in Safari on iOS just like a native app, including an "Add to Home Screen" icon, fullscreen mode, and offline fallback.

### Steps

1. **Start the server on your laptop/desktop** (the server and iPhone must be on the same Wi-Fi):
   ```bash
   python app.py
   ```
   The startup output now prints two URLs, for example:
   ```
     Local:   http://127.0.0.1:7860
     Network: http://192.168.1.42:7860  ← open this on your iPhone/phone
   ```

2. **On your iPhone, open Safari** and navigate to the **Network URL** shown (e.g. `http://192.168.1.42:7860`).  
   *(Other browsers such as Chrome or Firefox for iOS also work, but Safari is required for the "Add to Home Screen" feature.)*

3. **Use the app** — type an optimization problem in the text box and tap **Search**.

4. **Optional — install as a home-screen app:**
   - Tap the **Share** icon (the box with an arrow pointing up) at the bottom of Safari.
   - Scroll down and tap **Add to Home Screen**.
   - Give it a name (or keep "Opt Bot") and tap **Add**.
   - The app icon appears on your home screen and opens fullscreen, just like a native app.

### Troubleshooting

| Symptom | Fix |
|---|---|
| iPhone can't reach the URL | Make sure your laptop and iPhone are on the **same Wi-Fi network**. Check that no firewall blocks port 7860: on Linux run `sudo ufw allow 7860`; on Windows add an inbound rule in Windows Firewall; on macOS go to **System Settings → Network → Firewall → Options** and allow incoming connections for Python. |
| No "Network" URL printed at startup | The LAN IP detection may fail (e.g., no default gateway). Run `ipconfig` (Windows) or `ifconfig` / `ip addr` (macOS/Linux) to find your machine's local IP, then open `http://<your-ip>:7860` on the iPhone. |
| Page loads but search hangs | First-time startup downloads the embedding model (~90 MB). Wait until the terminal says *"Model ready"* before running queries. |
| "Add to Home Screen" not in Share sheet | The option only appears in **Safari** (not Chrome or Firefox on iOS). |

## Training

- **Retrieval model** — Fine-tune the sentence-transformers model so it better matches NL queries to problems in the catalog.  
  **Full guide:** [training/README.md](training/README.md)  
  Steps: generate synthetic (query, passage) pairs → optional: add [collected user queries](training/README.md#6-collect-real-user-prompts-for-training) from the app → run training (local or GPU batch on Wulver).  
  **Evaluation:** `python -m training.evaluate_retrieval --regenerate --num 500` for Precision@1 / Precision@5 on 500 held-out instances.

- **Mention–slot scorer** (NLP4LP) — For constrained assignment (NL numeric mentions → schema slots). Generate pairs with `training/generate_mention_slot_pairs.py` and train with `training/train_mention_slot_scorer.py`. See `docs/NLP4LP_CONSTRAINED_ASSIGNMENT_*.md`.

## 📋 Project Phases

### ✅ Phase 1: Data Collection & Processing (Current)
- [x] Define unified data schema
- [x] Collect NL4Opt dataset (1,101 NL→LP pairs) — public data added via `pipeline/run_collection.py`
- [ ] Collect Gurobi Modeling Examples (40+ notebooks)
- [ ] Collect Gurobi OptiMods (15+ documented mods)
- [ ] Parse all sources into unified JSON format
- [ ] Generate `all_problems.json`

### 🔲 Phase 2: Expand Dataset
- [ ] Parse GAMS Model Library (400+ models, needs GAMS license)
- [ ] Download & parse MIPLIB 2017 instances
- [ ] Scrape OR-Library problem families
- [ ] Extract Pyomo example formulations
- [ ] Manual additions from Williams' textbook

### 🔲 Phase 3: Problem Recognition Engine
- [ ] Generate embeddings for all problem descriptions
- [ ] Build similarity search index (FAISS/ChromaDB)
- [ ] Train/fine-tune problem classifier
- [ ] Implement disambiguation (clarifying questions)

### 🔲 Phase 4: Formulation Generation
- [ ] Retrieval pipeline for known problems
- [ ] LLM-based generation for novel problems
- [ ] LaTeX + solver code output formatting
- [ ] Validation against benchmark instances

### 🔲 Phase 5: Conversational Agent
- [ ] Conversation flow design
- [ ] Backend API (FastAPI)
- [ ] Frontend (Streamlit/Gradio)
- [ ] Deployment

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | pandas, json, BeautifulSoup |
| Embeddings | Sentence-Transformers / OpenAI |
| Vector Search | FAISS / ChromaDB |
| LLM | GPT-4 / Claude / LLaMA (fine-tuned on Wulver) |
| Optimization Solvers | Gurobi, Pyomo, PuLP, OR-Tools |
| Math Rendering | LaTeX / KaTeX |
| Backend | FastAPI |
| Frontend | Streamlit / Gradio |
| HPC | NJIT Wulver (SLURM) |
| CI/CD | GitHub Actions |
| Dev Environment | GitHub Codespaces |

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
| Data processing | pandas, json, BeautifulSoup |
| Retrieval | TF-IDF (scikit-learn), BM25 (rank-bm25), LSA, Sentence-Transformers, E5, BGE |
| NLP / extraction | Regex-based numeric tokenisation, operator-cue detection, bound-role annotation |
| Assignment | Typed greedy, constrained DP, semantic IR repair, optimization-role repair, GCG |
| Optimization solvers | Gurobi, Pyomo, PuLP, GAMSPy (output only; no live solver in CI) |
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

**Soroush Vahidi** — NJIT Student  
**Soroush Vahidi** — NJIT  
- Email: [sv96@njit.edu](mailto:sv96@njit.edu)  
- GitHub: [@SoroushVahidi](https://github.com/SoroushVahidi)

---

**Repository description** (for GitHub **Settings → General → Description**):  
*NL-to-optimization agent: problem recognition, formulation retrieval, GAMSPy/NLP4LP pipelines, and solver code generation.*
**Repository description** (for GitHub **Settings → General → Description**; update manually if needed):  
*NL-to-optimization agent: plain-English → ILP/LP formulation + solver code. Schema retrieval (R@1 0.906), deterministic grounding (typed greedy, optimization-role repair, GCG), and bound-role annotation pipeline.*
