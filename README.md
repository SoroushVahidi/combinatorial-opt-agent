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

Our dataset is built from multiple authoritative sources:

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the collection pipeline
python pipeline/run_collection.py
```

### Option C: HPC (Wulver @ NJIT) — For Phase 4+

```bash
# On Wulver
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
module load python/3.10
pip install --user -r requirements.txt

# For GPU-intensive tasks (model training)
sbatch scripts/train_classifier.slurm
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
