# Running the project on Wulver (NJIT HPC)

This guide gets the combinatorial-optimization bot (retrieval + catalog) running on **Wulver**, NJIT’s HPC cluster.

---

## 1. Log in to Wulver

From your laptop or a campus machine:

```bash
ssh your_netid@wulver.njit.edu
```

Use your NJIT credentials. If you use a different login node name, use that instead.

---

## 2. Clone the repo

```bash
cd ~   # or cd /home/your_netid, or a project dir
git clone https://github.com/SoroushVahidi/combinatorial-opt-agent.git
cd combinatorial-opt-agent
```

To use your latest local commits, push from your PC first (`git push origin main`), then clone or `git pull` on Wulver.

---

## 3. Load Python and create an environment

Wulver uses **modules** for software. Load a recent Python and use a virtual environment so you don’t need admin rights:

```bash
module load python/3.10
python -m venv venv
source venv/bin/activate
```

Your prompt should show `(venv)`.

---

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first time, this downloads `sentence-transformers` and the embedding model (~90MB). It may take a few minutes. The retrieval runs on **CPU** by default, so you don’t need a GPU for the bot.

---

## 5. Run the bot

**Interactive (login or compute node):**

```bash
python run_search.py "minimize cost of opening warehouses and assigning customers"
python run_search.py "knapsack" 2
```

**As a SLURM job (optional):**

Edit the query in `scripts/run_search.slurm` (see below), then:

```bash
sbatch scripts/run_search.slurm
```

Output will be in the file SLURM prints (e.g. `slurm-12345.out`).

---

## 6. Optional: SLURM script for batch queries

The repo includes `scripts/run_search.slurm`. It requests a single core and runs one query. Edit the script to change the query or the number of results, then:

```bash
sbatch scripts/run_search.slurm
```

Use `squeue -u $USER` to see your job; results appear in the `.out` file when the job finishes.

---

## 7. Updating the project on Wulver

After you push new changes from your PC:

```bash
cd ~/combinatorial-opt-agent
git pull origin main
```

If you only changed Python code or data, no need to re-run `pip install` unless `requirements.txt` changed.

---

## 8. Where to run

- **Login nodes**: Fine for quick tests (e.g. one or two `run_search.py` calls). Don’t run long or heavy jobs here.
- **Compute nodes**: For many queries or heavier use, use `sbatch scripts/run_search.slurm` (or an interactive session with `srun` / `salloc`) so the job runs on a compute node.

---

## Troubleshooting

- **`module: command not found`** — You may need to run `source /etc/profile.d/modules.sh` or log in again; ask your HPC support if modules still don’t load.
- **`python: command not found`** — Run `module load python/3.10` (or another available Python module: `module avail python`).
- **Out of memory** — The default retrieval is small; if you hit memory limits, try running in a job with more memory or use a smaller sentence-transformers model later.
- **Network / pip install fails** — Some clusters restrict the internet from compute nodes; run `pip install -r requirements.txt` on a **login node** so the model is cached in your environment, then run `run_search.py` via SLURM as needed.

If `sbatch` says the partition is invalid, run `sinfo` on Wulver and change `#SBATCH --partition=shared` in `scripts/run_search.slurm` to a valid partition (e.g. `normal`, `cpu`). For Wulver-specific policies (queue names, time limits, storage), see [NJIT’s HPC documentation](https://www.njit.edu/research/hpc).
