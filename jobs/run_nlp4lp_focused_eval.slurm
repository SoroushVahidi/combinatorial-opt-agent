#!/bin/bash
#SBATCH --job-name=nlp4lp_focused
#SBATCH --output=logs/nlp4lp_focused_eval_%j.out
#SBATCH --error=logs/nlp4lp_focused_eval_%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:45:00
# Run focused downstream eval (default: 4 evidence-supported methods) + per-instance comparison + disagreement + failure audit + three-bottlenecks analysis.
# Submit from project root: sbatch jobs/run_nlp4lp_focused_eval.slurm
#
# Optional: export NLP4LP_EXPERIMENTAL=1 to include archived methods (anchor_linking, bottomup_beam, entity_semantic_beam; 7 methods total).
# Use safe mode on Wulver/low-resource nodes: export NLP4LP_SAFE=1 before sbatch, or run steps manually with --safe (see docs/NLP4LP_LOW_RESOURCE_RUN_GUIDE.md).

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${SLURM_SUBMIT_DIR:-$ROOT}"
mkdir -p logs results/paper
module load python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

# Conservative thread limits for constrained compute nodes (avoids BlockingIOError / "can't start new thread")
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
# Gold cache: step 1 writes it, step 2 reads it (avoids loading HuggingFace dataset twice + reduces threads)
export NLP4LP_GOLD_CACHE="${NLP4LP_GOLD_CACHE:-$ROOT/results/paper/nlp4lp_gold_cache.json}"
SAFE_ARGS=""
if [ -n "${NLP4LP_SAFE}" ] || [ "${NLP4LP_SAFE:-0}" = "1" ]; then
  SAFE_ARGS="--safe"
fi
EXP_ARGS=""
if [ -n "${NLP4LP_EXPERIMENTAL}" ] && [ "${NLP4LP_EXPERIMENTAL}" = "1" ]; then
  EXP_ARGS="--experimental"
fi

set -e
echo "=== 1. Run focused methods (4 default, 7 with NLP4LP_EXPERIMENTAL=1) and build summary CSV ==="
python tools/run_nlp4lp_focused_eval.py --variant orig $SAFE_ARGS $EXP_ARGS

echo "=== 2. Build per-instance comparison (opt_repair + relation_repair; add experimental columns if EXP_ARGS) ==="
python tools/build_nlp4lp_per_instance_comparison.py --variant orig $SAFE_ARGS $EXP_ARGS

echo "=== 3. Label disagreements ==="
python tools/analyze_nlp4lp_downstream_disagreements.py

echo "=== 4. Failure audit (patterns, hard cases, per-query family, report) ==="
python tools/build_nlp4lp_failure_audit.py

echo "=== 5. Three-bottlenecks comparison (wrong_var, multiple_float, lower_vs_upper) ==="
python tools/analyze_nlp4lp_three_bottlenecks.py

echo "Done. Check results/paper/nlp4lp_focused_*.csv, nlp4lp_downstream_failure_*.csv/.md, nlp4lp_three_bottlenecks_comparison.md"
