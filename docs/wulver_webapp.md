# Running the web app on Wulver (login node thread limit)

The app needs to load the SentenceTransformer model. On the **login node**, Hugging Face’s downloader tries to spawn a thread and gets blocked (`Resource temporarily unavailable`). So you have to either run the app on a **compute node** or **pre-fill the model cache** and run on the login node.

---

## Option A: Run the app on a compute node (recommended)

You’ll run the app on a compute node and use an SSH reverse tunnel so your laptop’s browser can reach it.

### 1. Get a compute node and start the app in the background

On Wulver (in the terminal where you’re logged in). You **must** specify partition and QOS or you may get "Invalid qos specification". Use **low** QOS (no SU charge):

```bash
srun --pty -p general --qos=low bash
```

Or use the debug partition (short jobs, no SU charge; 4 CPUs, 16 GB):

```bash
srun --pty -p debug --qos=debug bash
```

Other Wulver partitions: `gpu`, `bigmem`. Run `sinfo` to see what’s available.

When you get a prompt on the compute node:

```bash
cd /mmfs1/home/sv96/combinatorial-opt-agent
bash run_app_wulver.sh &
```

Wait until you see the server listening (e.g. “Starting server…”). The app is now running in the background.

### 2. Create a reverse tunnel so the login node forwards to the app

In the **same** terminal (still on the compute node), run:

```bash
ssh -R 7860:localhost:7860 login02
```

Use your login node name if different (e.g. `login01`; you were on `login02` when you ran `srun`). Log in if asked. **Leave this SSH session open** — it keeps the tunnel active so that when your laptop connects to the login node’s port 7860, traffic is forwarded to the app on the compute node.

### 3. On your laptop: forward port and open the app

On your **laptop**, in a new terminal:

```bash
ssh -L 7860:localhost:7860 sv96@wulver.njit.edu
```

Log in and keep this session open. In your browser open:

**http://127.0.0.1:7860**

Traffic flows: laptop → login node (your SSH -L) → compute node (reverse tunnel -R) → app.

---

## Option B: Pre-download the model on your laptop, then run on the login node

If the model is already in the cache, the app may load it without spawning the download thread, so it might work on the login node.

### 1. On your laptop (one-time)

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

This fills the cache (e.g. `~/.cache/huggingface/hub` on Linux/macOS).

### 2. Copy the cache to Wulver

From your laptop:

```bash
scp -r ~/.cache/huggingface/hub sv96@wulver.njit.edu:~/.cache/huggingface/
```

(If `~/.cache/huggingface` doesn’t exist on Wulver, run `mkdir -p ~/.cache/huggingface` there first.)

### 3. On Wulver login node

```bash
cd /mmfs1/home/sv96/combinatorial-opt-agent
bash run_app_wulver.sh
```

If it still fails with “spawn thread” or “Resource temporarily unavailable”, the loader is still using a thread; use **Option A** instead.

### 4. On your laptop

```bash
ssh -L 7860:localhost:7860 sv96@wulver.njit.edu
```

Then open **http://127.0.0.1:7860** in your browser.
