# Wulver (NJIT HPC) — Resource Inventory for Training

**Date:** Inventory run from current environment.  
**Current node:** `login01` (login node).  
**Important:** This inventory was collected on the **login node**. Login nodes have **no GPUs** and are not representative of GPU compute nodes. For GPU training you must use the **SLURM `gpu` partition** and run your jobs on compute nodes (see Section C and “Inspect GPU nodes” below).

---

## A) Summary table

| Resource   | Finding |
|-----------|---------|
| **CPU**   | 8 logical cores (8 sockets × 1 core), Intel Xeon (Sapphire Rapids), 2800 MHz. VM (KVM). |
| **RAM**   | 46 Gi total, ~33 Gi available; 15 Gi swap. |
| **GPU**   | **None on this node** (login node). Cluster has GPU partition: A100 (40G/20G/10G) and L40. |
| **VRAM**  | N/A on login node. On GPU nodes: A100 up to 40 GB per GPU; L40 typically 48 GB. |
| **Storage** | Root 40G; **/mmfs1** (GPFS) 1.3 P total, 881 TiB avail; /home, /scratch → /mmfs1. User quota on /tmp: 400G. |
| **Python** | 3.11.5 (Anaconda3/2023.09-0). Conda and pip available. |
| **PyTorch** | 2.7.0 (user site-packages), built with CUDA 12 dependencies. |
| **CUDA**  | Not available on login node (no nvidia-smi). GPU nodes have drivers; torch uses CUDA 12. |
| **SLURM** | Available. Partitions: general (CPU), **gpu** (A100, L40), bigmem, debug, debug_gpu, course. |

---

## B) Detailed results (commands and interpretation)

### 1) CPU

**Commands:** `lscpu`, `cat /proc/cpuinfo`

**Findings:**
- **Sockets / physical CPUs:** 8 (each socket has 1 core — typical of a small VM).
- **Logical cores:** 8 (Thread(s) per core = 1).
- **Model:** Intel Xeon Processor (Sapphire Rapids), 2800 MHz.
- **Virtualization:** KVM hypervisor (full virtualization) — this is a **login node VM**, not a bare-metal compute node.

**Interpretation:** Login node has limited, shared CPU; sufficient for light scripting and submission, not for serious training. Compute nodes (e.g. GPU partition) have 128 cores per node.

---

### 2) RAM / memory

**Commands:** `free -h`, `cat /proc/meminfo`

**Findings:**
- **Total:** 46 Gi (49258768 kB).
- **Available:** ~33 Gi (34995028 kB).
- **Swap:** 15 Gi total, ~15 Gi free.

**Interpretation:** Adequate for running small scripts and SLURM clients on the login node. GPU nodes have 514 GB RAM per node for training jobs.

---

### 3) GPU

**Commands:** `which nvidia-smi`, `nvidia-smi`, `nvcc --version`

**Findings:**
- **nvidia-smi:** Not in PATH (command not found). **No GPU on this node.**
- **nvcc:** Not in PATH.

**Interpretation:** You are on a **login node**; GPUs are only on compute nodes in the **gpu** (and debug_gpu / course_gpu) partitions. To see GPUs and CUDA, run a job or interactive session on a GPU node (see “Inspect GPU nodes” at the end).

**From SLURM (GPU partition):**
- **gpu** partition: 26 nodes total.
  - Some nodes: **4× A100** per node (e.g. n0003–n0005, n0045–n0049, n0066–n0069, …).
  - Some nodes: **4× A100_40G**, **4× A100_20G**, **8× A100_10G** (mixed, e.g. n0002, n0089, n0091).
  - Some nodes: **2× L40** (e.g. n1541, n1543, n1547).
- **debug_gpu:** 1 node (n0111), A100_40g/20g/10g, 12 h time limit.
- **course_gpu:** 3 nodes, A100_10g.

---

### 4) Disk / storage

**Commands:** `df -h`, `pwd`, `du -sh .`, `ls /mmfs1`, `quota -s`, `lfs quota`

**Findings:**
- **Root (/):** 40 G total, ~32 G avail.
- **/mmfs1:** GPFS, **1.3 P** total, **881 TiB** avail (31% used). This is the main shared filesystem.
- **/home** → `/mmfs1/home`, **/scratch** → `/mmfs1/scratch` (symlinks).
- **Project path:** `/mmfs1/home/sv96/combinatorial-opt-agent` (~13 G).
- **/tmp:** 868 G local, 861 G avail; **user quota 400 G** (lv_local).
- **/var:** 80 G.

**Interpretation:** Use **/mmfs1/home/sv96** for code and small datasets; use **/mmfs1/scratch/sv96** (or similar) for large runs and checkpoints if your site recommends scratch. Respect 400 G quota on /tmp if you use it.

---

### 5) SLURM

**Commands:** `hostname`, `which squeue sinfo scontrol`, `squeue -u $USER`, `sinfo`, `scontrol show partition`, `scontrol show node n0049`, `scontrol show node n1541`, `sinfo -o "%P %G %c %m %N"`, `scontrol show config`

**Findings:**
- **Current host:** `login01` → **login node.**
- **SLURM:** Available at `/apps/slurm/current/bin/` (squeue, sinfo, scontrol).
- **Partitions (summary):**

| Partition   | Use      | Nodes      | Time limit | Notes                          |
|------------|----------|------------|------------|--------------------------------|
| general    | CPU      | 96         | Unlimited  | Default partition, no GPU      |
| gpu        | GPU      | 26         | Unlimited  | A100 (10g/20g/40g), L40       |
| bigmem     | CPU      | 2          | Unlimited  | 128 cores, ~2 TB RAM/node     |
| debug      | CPU      | 2          | 8 h        | QoS=debug                     |
| debug_gpu  | GPU      | 1 (n0111)  | 12 h       | QoS=debug, A100 mix           |
| course     | CPU      | 62         | Unlimited  | courses account                |
| course_gpu | GPU      | 3          | Unlimited  | courses, A100_10g              |

- **GPU node examples:**
  - **n0049:** 128 CPUs, 514 GB RAM, **4× A100**, 4× MPS (A100).
  - **n1541:** 128 CPUs, 514 GB RAM, **2× L40**, 2× MPS (L40).
- **Billing:** TRES includes `gres/gpu:a100`, `gres/gpu:a100_40g`, `gres/gpu:a100_20g`, `gres/gpu:a100_10g`, `gres/gpu:l40`. MaxJobCount 1000001.

**Interpretation:** For training you want the **gpu** partition (or **debug_gpu** for short tests). Request the appropriate GRES (e.g. `--gres=gpu:a100_40g:1` or `--gres=gpu:l40:1`) in your job script.

---

### 6) Python / ML environment

**Commands:** `which python python3`, `python --version`, `which conda pip`, `pip list | grep -E "torch|transformers|..."`, `pip show torch`

**Findings:**
- **Python:** `/apps/easybuild/software/Anaconda3/2023.09-0/bin/python` → **3.11.5**.
- **Conda:** Available (same Anaconda install).
- **pip:** Available.
- **Packages (from pip list):**
  - **torch** 2.7.0 (installed in **user** site-packages: `/home/sv96/.local/lib/python3.11/site-packages`).
  - **transformers** 5.2.0, **datasets** 4.6.1, **sentence-transformers** 5.2.3.
  - **scipy** 1.11.1, **scikit-learn** 1.3.0.
  - **torch-geometric**, **torch_scatter**, **torch_sparse** (with cu126 in name → CUDA 12.6).
- **Torch dependencies:** nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, etc. → **PyTorch built for CUDA 12.**
- **Not seen in pip list:** accelerate, peft, sentencepiece, bitsandbytes (may be in another env or need installing in job env).
- **CUDA from Python:** On the login node, `import torch` did not succeed in the default shell (e.g. module/path); **on a GPU compute node**, the same user env would be expected to see `torch.cuda.is_available() == True` and a non-zero `torch.cuda.device_count()`.

**Interpretation:** Python 3.11 and PyTorch 2.7 with CUDA 12 are available; ensure your SLURM job uses the same environment (e.g. activate conda or use the same `python` that has torch). For bf16/fp16: A100 and L40 support both; check with `torch.cuda.get_device_capability()` on a GPU node.

---

## C) Recommended training scale and next steps

### Recommended training scales on Wulver GPU nodes

- **LoRA / small adapter fine-tuning (e.g. 7B):** Realistic on **1× A100 40G** or **1× L40**; batch size 4–16 depending on seq length and LoRA rank.
- **Cross-encoder / small transformer:** Fine on 1 GPU; can use A100_10g or L40.
- **Full fine-tuning of 7B:** Possible on **1–2× A100 40G** with gradient checkpointing and modest batch size; 70B would need multi-node/multi-GPU and is a larger project.
- **Batch size:** Start with 4–8 per A100 40G for 7B LoRA; scale up if memory allows.
- **CPU-only training:** Would be **too slow** for transformer fine-tuning; use only for tiny experiments or data prep. Use the **gpu** partition for real training.

### Missing or to verify on a GPU node

- **Accurate VRAM and driver on compute nodes:** Run `nvidia-smi` and `nvcc --version` (if needed) **inside a GPU job**.
- **Torch CUDA in job:** Run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"` in a GPU job.
- **accelerate / peft / sentencepiece:** Install in the environment used by your job if you need them.
- **Account / QOS:** Confirm your account has access to `gpu` (and optionally `debug_gpu`) with `sacctmgr show assoc user=$USER` or ask support.

---

## Inspect GPU nodes (recommended next steps)

Because this inventory was taken on the **login node**, you do **not** see GPUs or CUDA here. To inspect a **GPU compute node**:

**1) Interactive session on one GPU (recommended):**
```bash
srun --partition=gpu --gres=gpu:a100_40g:1 --mem=32G --time=0:30:00 --pty bash
```
Then on that node:
```bash
nvidia-smi
nvcc --version 2>/dev/null || true
python -c "import torch; print('cuda', torch.cuda.is_available()); print('count', torch.cuda.device_count()); print('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A')"
```

**2) One-line check from login node (if your cluster allows):**
```bash
srun --partition=gpu --gres=gpu:1 --time=0:05:00 nvidia-smi
```

**3) Example batch script to run the same Python check and save to a file:**
```bash
#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_40g:1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=gpu_check_%j.out

nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"
```

Submit with: `sbatch gpu_check.slurm`, then read the output file to confirm GPU and CUDA availability in your environment.
