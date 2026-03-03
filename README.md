# Combinatorial Optimization AI Agent

An **AI-powered agent** that helps users describe **mathematical optimization problems** in **natural language** and automatically provides:

- Recognition of whether it’s a **known combinatorial optimization problem**
- The **Integer Linear Program (ILP)** formulation
- The **Linear Program (LP)** relaxation
- **Solver-ready code** (e.g. Pyomo, Gurobi, PuLP)

## Project vision

You describe a problem in plain English; the agent identifies the problem type (e.g. Uncapacitated Facility Location), returns the ILP/LP formulation, and generates code you can run with standard solvers.

## Architecture (high level)

- **Natural language** → entity and constraint extraction.
- **Problem classifier** (embeddings + LLM) matches against a **database of known problems** (90+ types).
- For **known problems:** retrieve formulation from the DB.
- For **unknown problems:** generate formulation via LLM and verify.
- **Output:** ILP formulation, LP relaxation, solver code (Pyomo/Gurobi/PuLP), and complexity class when applicable.

## Data sources

The dataset is built from multiple authoritative sources. The full list of libraries and problem names is in the project:

- **[docs/data_sources.md](docs/data_sources.md)** — Canonical list of all sources with URLs, sizes, and problem/example names (OR-Library, Gurobi modeling examples, Gurobi OptiMods, etc.).
- **data/sources/** — Machine-readable manifests: `or_library.json`, `gurobi_modeling_examples.json`, `gurobi_optimods.json`, `index.json`.

Notable sources include [NL4Opt](https://github.com/nl4opt/nl4opt-competition) (NL → formulation) and [NLP4LP / OptiMUS](https://github.com/...) for natural language optimization.

## How to run

1. Clone the repo and install dependencies (see `requirements.txt` or project docs).
2. Run the agent (e.g. CLI or web interface) and input a natural-language description of your optimization problem.
3. Use the generated formulation and solver code with your preferred solver.

See in-repo documentation for API keys (if using hosted LLMs), environment variables, and example prompts.

## License

See the `LICENSE` file in the repository. Contributions are welcome via pull requests.
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
>>>>>>> 5e985ef (Expand catalog, training pipeline, and web UI)
