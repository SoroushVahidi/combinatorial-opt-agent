#!/usr/bin/env bash
#SBATCH --job-name=pamop_pilot
#SBATCH --output=logs/pamop/pamop_pilot_%j.out
#SBATCH --error=logs/pamop/pamop_pilot_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=standard

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs/pamop results/pamop/pilot

module load python/3.10 2>/dev/null || true
if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
  source "${REPO_ROOT}/venv/bin/activate"
fi

export PAMOP_AMPLPY_PYTHON="${PAMOP_AMPLPY_PYTHON:-/home/soroush/.venvs/gurobi/bin/python}"
export PYTHONUNBUFFERED=1

python tools/pamop_pilot_benchmark.py \
  --output-dir results/pamop/pilot \
  --deployment gpt-4.1-mini \
  --temperature 0.2 \
  --ampl-python "${PAMOP_AMPLPY_PYTHON}"
