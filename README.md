# 🧠 Combinatorial Optimization AI Agent

An AI-powered agent that helps users describe mathematical optimization problems in **natural language** and automatically provides:

- ✅ Recognition of whether it's a **known combinatorial optimization problem**
- ✅ The **Integer Linear Program (ILP)** formulation
- ✅ The **Linear Program (LP) relaxation**
- ✅ Solver-ready code (Pyomo, Gurobi, PuLP)

## 🎯 Project Vision

```
User: "I have 5 warehouses and 20 customers. I want to open some warehouses 
       and assign each customer to an open warehouse to minimize total cost."

Agent: "This is the **Uncapacitated Facility Location Problem** (UFL).
        Here is the ILP formulation..."
        
        min  Σᵢ fᵢyᵢ + Σᵢⱼ cᵢⱼxᵢⱼ
        s.t. Σᵢ xᵢⱼ = 1          ∀j ∈ Customers
             xᵢⱼ ≤ yᵢ             ∀i ∈ Warehouses, ∀j ∈ Customers
             xᵢⱼ ∈ {0,1}, yᵢ ∈ {0,1}
```

## 🏗️ Architecture

```
User (Natural Language)
        │
        ▼
┌─────────────────────┐
│  NL Understanding    │ ← Extract entities, variables, constraints
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────��┐
│  Problem Classifier  │────►│  Known Problem DB     │
│  (Embedding + LLM)   │     │  (90+ problem types)  │
└────────┬────────────┘     └──────────────────────┘
         │
    ┌────┴────┐
    │         │
 Known    Unknown
    │         │
    ▼         ▼
┌────────┐ ┌────────────┐
│Retrieve│ │Generate via │
│from DB │ │LLM + Verify │
└───┬────┘ └─────┬──────┘
    │             │
    ▼             ▼
┌─────────────────────┐
│  Output:             │
│  • ILP Formulation   │
│  • LP Relaxation     │
│  • Solver Code       │
│  • Complexity Class  │
└─────────────────────┘
```

## 📊 Data Sources

Our dataset is built from multiple authoritative sources. **The full list of libraries and their problem names is in the project:**

- **[docs/data_sources.md](docs/data_sources.md)** — Canonical list of all sources with URLs, sizes, and explicit problem/example names (OR-Library families, Gurobi modeling example folders, Gurobi OptiMods mods, etc.).
- **data/sources/** — Machine-readable manifests: `or_library.json`, `gurobi_modeling_examples.json`, `gurobi_optimods.json`, `index.json`.

| Source | Type | Size | What We Extract |
|--------|------|------|-----------------|
| [NL4Opt](https://github.com/nl4opt/nl4opt-competition) | NL → Formulation | 1,101 problems | Natural language + LP formulations |
| [NLP4LP / OptiMUS](https://github.com/teshnizi/OptiMUS) | NL → Formulation | 269 problems | NL + LP/MILP formulations |
| [Gurobi OptiMods](https://github.com/Gurobi/gurobi-optimods) | Documented Models | ~15 mods | Math formulations + Python code |
| [Gurobi Modeling Examples](https://github.com/Gurobi/modeling-examples) | Notebooks | 40+ examples | NL + LaTeX + Gurobi code |
| [GAMS Model Library](https://www.gams.com/latest/gamslib_ml/libhtml/) | Model Catalog | 400+ models | Categorized formulations in GAMS |
| [MIPLIB 2017](https://miplib.zib.de/) | Benchmark Instances | 1,000+ instances | Real-world MILP in MPS format |
| [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/info.html) | Problem Families | 90+ types | Labeled families + test instances |
| [Pyomo Examples](https://github.com/Pyomo/pyomo) | Code Examples | 15+ models | Python optimization models |

## 🗂️ Project Structure

```
combinatorial-opt-agent/
│
├── README.md
├── requirements.txt
├── .devcontainer/              # Codespace configuration
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       └── collect_data.yml    # GitHub Actions for automated collection
│
├── schema/
│   └── problem_template.json   # Unified data schema
│
├── collectors/                 # Scripts to pull from each source
│   ├── __init__.py
│   ├── collect_nl4opt.py       # NL4Opt dataset
│   ├── collect_gurobi_optimods.py  # Gurobi OptiMods
│   ├── collect_gurobi_examples.py  # Gurobi Modeling Examples
│   ├── collect_gamslib.py      # GAMS Model Library (Phase 2)
│   ├── collect_miplib.py       # MIPLIB 2017 (Phase 2)
│   ├── collect_or_library.py   # OR-Library (Phase 2)
│   └── collect_pyomo.py        # Pyomo examples
│
├── parsers/                    # Transform raw data → unified schema
│   ├── __init__.py
│   ├── notebook_parser.py      # Parse Jupyter notebooks
│   ├── rst_parser.py           # Parse RST documentation
│   ├── mps_parser.py           # Parse MPS files
│   └── latex_formatter.py      # Generate LaTeX from schema
│
├── pipeline/                   # Orchestration
│   ├── __init__.py
│   └── run_collection.py       # Master pipeline script
│
├── data/
│   ├── raw/                    # Downloaded raw data (git-ignored)
│   ├── processed/              # Unified schema JSON files
│   │   ├── all_problems.json
│   │   └── by_category/
│   └── embeddings/             # Pre-computed embeddings (Phase 3)
│
├── validation/                 # Verify formulations (Phase 2-3)
│   └── solve_and_verify.py
│
└── notebooks/                  # Exploration & analysis
    └── explore_dataset.ipynb
```

### Extending the catalog with more problems

The retrieval engine searches over `data/processed/all_problems.json` (or an extended version if present). You can add more problems (from new libraries, textbooks, etc.) without changing any model weights:

1. Copy the template:

   ```bash
   cp data/processed/custom_problems.template.json data/processed/custom_problems.json
   ```

2. Edit `data/processed/custom_problems.json` and **append** more problems following the schema (see `schema/problem_schema.json`).
3. Build the extended catalog:

   ```bash
   python build_extended_catalog.py
   ```

   This writes `data/processed/all_problems_extended.json`, which is automatically picked up by the web UI and CLI search.

## 🚀 Quick Start

### Option A: GitHub Codespaces (Recommended — zero local setup)

1. Click **Code → Codespaces → Create codespace on main**
2. Wait for the environment to build (~2 minutes)
3. In the terminal, run:
   ```bash
   python pipeline/run_collection.py
   ```

### Option B: Local Setup

```bash
# Clone the repo
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (includes sentence-transformers, gradio, numpy)
pip install -r requirements.txt

# If you see a "Numpy is not available" error on macOS/Python 3.12+, run:
pip install --force-reinstall "numpy<2" "sentence-transformers==2.5.1" "transformers==4.44.2"

# (Optional) Run the data collection pipeline
python pipeline/run_collection.py
```

### Try the retrieval (query → problem + IP)

#### Option 1 — Web UI (recommended)

A browser app where you type in natural language and see the closest matching problem and its integer program.

```bash
source venv/bin/activate    # if not already
python app.py
```

Then:

1. Open the URL printed in the terminal (e.g. `http://127.0.0.1:7860`).
2. Type your optimization problem in plain English in the textbox.
3. (Optional) Click one of the **Examples** — this only fills the inputs.
4. Click **Submit** to actually run the search.
5. A short message appears (“Searching for matching problems…”) while the model runs, then the results are shown:
   - Problem name
   - Natural-language description
   - **Variables**, **objective**, **constraints** (with LaTeX-style math)
   - LaTeX formulation and complexity, when available

#### Option 2 — Command line

```bash
source venv/bin/activate    # if not already
python run_search.py "minimize cost of opening warehouses and assigning customers"
python run_search.py "knapsack" 2
```

The first run will download the sentence-transformers model (~90MB). Results show the best-matching problem(s) and their integer program (variables, objective, constraints).

## 🔍 Feedback & logging (privacy)

When you run `python app.py`, the web UI:

- **Stores your queries and the corresponding answers locally** in `data/feedback/chat_logs.jsonl` on the machine where you run the app.
- **Optionally sends these logs to a remote server** controlled by the app operator *only if* the environment variable `REMOTE_FEEDBACK_ENDPOINT` is set.
  - In that case, each interaction is sent as a JSON `POST` payload to `REMOTE_FEEDBACK_ENDPOINT` so it can be used to improve the model and catalog over time.
- The **Flag** button in the UI stores flagged interactions (with a reason such as “wrong problem”, “bad formulation”, etc.) in CSV files under `data/feedback/`.

If you deploy this app for other people to use, you should:

- Make sure they understand that their text inputs and the model’s responses are being logged, and
- Tell them whether `REMOTE_FEEDBACK_ENDPOINT` is configured so that logs are also sent to your server.

### Running a central feedback server (optional)

If you want to aggregate logs from many users/machines:

1. **Start the feedback server on a machine you control**

   ```bash
   cd combinatorial-opt-agent
   python feedback_server.py
   ```

   This starts a small HTTP server on `http://0.0.0.0:8000/collect` and appends all received logs to:

   - `data/feedback/remote_logs.jsonl`

2. **Configure each deployment of the web app to send logs**

   On every machine where you run `app.py` and want to send logs to your server:

   ```bash
   export REMOTE_FEEDBACK_ENDPOINT="http://YOUR_SERVER_IP:8000/collect"
   python app.py
   ```

   Replace `YOUR_SERVER_IP` with the public or internal IP / hostname of the feedback server machine.

3. **Keep the feedback server running on a remote VM**

   On a Linux VM you can keep the server alive using `tmux` or `screen` (simplest option):

   ```bash
   ssh user@your-vm
   cd combinatorial-opt-agent
   tmux new -s feedback
   python feedback_server.py
   # Press Ctrl+B, then D to detach the tmux session
   ```

   The server continues running in the background even if you disconnect. Later you can reattach with:

   ```bash
   tmux attach -t feedback
   ```

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

## 📋 Project Phases

### ✅ Phase 1: Data Collection & Processing (Current)
- [x] Define unified data schema
- [ ] Collect NL4Opt dataset (1,101 NL→LP pairs)
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
- GitHub: [@SoroushVahidi](https://github.com/SoroushVahidi)  
- Email: `sv96@njit.edu` (send suggestions, recommendations, or problem ideas)
