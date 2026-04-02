#!/bin/bash
#SBATCH --job-name=paper_artifacts
#SBATCH --output=logs/paper_artifacts_%j.out
#SBATCH --error=logs/paper_artifacts_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Run paper artifact generation on a compute node (avoids login-node fork limits).
# Submit from project root (or from an srun session if login node blocks sbatch):
#   sbatch jobs/run_paper_artifacts.slurm
# Or from an interactive compute node:
#   srun --pty -p shared bash   # then: cd .../combinatorial-opt-agent && sbatch jobs/run_paper_artifacts.slurm
# Output: logs/paper_artifacts_<jobid>.out

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$ROOT}"
mkdir -p logs
module load python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true
python tools/make_nlp4lp_paper_artifacts.py
