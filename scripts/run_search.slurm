#!/bin/bash
#SBATCH --job-name=run_search
#SBATCH --output=run_search_%j.out
#SBATCH --error=run_search_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

# Run the retrieval bot with one query. Edit QUERY (and optionally TOP_K) below.
# Usage: sbatch scripts/run_search.slurm

QUERY="minimize cost of opening warehouses and assigning customers"
TOP_K=3

cd "${SLURM_SUBMIT_DIR:-.}"
module load python/3.10
source venv/bin/activate 2>/dev/null || true
python run_search.py "${QUERY}" ${TOP_K}
