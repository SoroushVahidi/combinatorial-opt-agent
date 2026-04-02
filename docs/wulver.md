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

**Command-line (login or compute node):**

```bash
python run_search.py "minimize cost of opening warehouses and assigning customers"
python run_search.py "knapsack" 2
```

**Web app (`python app.py`) — run on a compute node**

The Gradio app uses the embedding model and Hugging Face’s downloader, which can spawn threads. **Login nodes** on Wulver have a strict thread limit, so the app often fails there with “can’t start new thread” or “Resource temporarily unavailable” when loading the model.

- **Option A — Run the app on a compute node:** Get an interactive session (use your usual partition, e.g. `srun --pty bash` or the partition name from `sinfo`), then in that shell run `python app.py`. Use SSH port forwarding to that compute node’s hostname if your cluster allows it, or run the app on your laptop for the web UI.
- **Option B — Pre-download the model on a compute node, then run app on login:** In an interactive compute session, run once:  
  `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`  
  so the model is cached under `~/.cache`. Then on the **login node**, run `python app.py`; it may load from cache and avoid the download thread. If you still see thread/resource errors on the login node, use Option A.

From your laptop, forward the port and open the app:

```bash
ssh -L 7860:localhost:7860 your_netid@wulver.njit.edu
```

Then in the browser open **http://127.0.0.1:7860** (not 0.0.0.0).

---

## Gemini LLM baselines (optional)

If you use **`GEMINI_API_KEY`** and `batch/learning/run_gemini_llm_baselines.sbatch`, the Google client may print **`pthread_create failed: Resource temporarily unavailable`** on some nodes. **`preflight`** can still succeed; see **`docs/gemini_api_quota.md`** (section *Wulver / HPC: pthread*) and optional **`export GEMINI_LIMIT_RUNTIME_THREADS=1`** before Python.

`run_gemini_llm_baselines.sbatch` defaults to **`GEMINI_LIMIT_RUNTIME_THREADS=1`**, **`GEMINI_AUTO_PICK_MODEL` on** (run `pick-model` then `preflight`), and the yaml default model **`gemini-2.5-flash-lite`**. Always run **`python tools/llm_baselines.py preflight`** in the same venv when debugging; quota is **account-specific** (see `configs/llm_baselines.yaml`, `docs/gemini_api_quota.md`).

**If you see `ImportError: cannot import name 'HfFolder'`:** Upgrade the `datasets` package:  
`pip install --upgrade datasets --user`

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
