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

## Current evidence-based status

- **Problem recognition / retrieval** — Strong; embeddings + catalog match queries to known problem types.
- **Downstream grounding** (NL mentions → schema slots for instantiation) — Main bottleneck; active research.
- **Deterministic methods** (rule-based scoring, typed greedy assignment) — Currently the main trusted, reproducible results for grounding.
- **Learning** — Infrastructure exists (benchmark-safe train/dev/test splits, split-integrity checks, pairwise ranker training and evaluation). A real-data-only benchmark run showed the current learned formulation did not outperform the deterministic rule baseline; learning is documented as future work. See [docs/learning_runs/](docs/learning_runs/README.md).

## Architecture (high level)

- **Natural language** → entity and constraint extraction.
- **Problem classifier** (embeddings + LLM) matches against a **database of known problems** (90+ types).
- For **known problems:** retrieve formulation from the DB.
- For **unknown problems:** generate formulation via LLM and verify.
- **Output:** ILP formulation, LP relaxation, solver code (Pyomo/Gurobi/PuLP), and complexity class when applicable.

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

## Documentation

| Topic | Doc |
|-------|-----|
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
*NL-to-optimization agent: problem recognition, formulation retrieval, NLP4LP grounding (deterministic methods primary), and solver code generation.*
