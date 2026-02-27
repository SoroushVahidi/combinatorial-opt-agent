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
