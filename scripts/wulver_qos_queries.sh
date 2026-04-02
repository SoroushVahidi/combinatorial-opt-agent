#!/usr/bin/env bash
# Query Wulver cluster for QoS (Quality of Service) and partition info.
# Run on a login node: bash scripts/wulver_qos_queries.sh
# Use output to fix sbatch "Invalid qos specification" (e.g. set SBATCH_QOS in batch files).
set -euo pipefail

echo "=== 1. Slurm QoS (sacctmgr show qos) ==="
sacctmgr show qos format=Name,Priority,MaxWall,MaxJobs,MaxSubmitJobs,MaxTRES,State -p 2>/dev/null || echo "sacctmgr failed or not Slurm"

echo ""
echo "=== 2. Partition info (scontrol show partition) ==="
scontrol show partition 2>/dev/null | head -80 || echo "scontrol failed"

echo ""
echo "=== 3. Node/partition summary (sinfo -s) ==="
sinfo -s 2>/dev/null || true

echo ""
echo "=== 4. Default QoS (if any) ==="
scontrol show config 2>/dev/null | grep -i qos || true

echo ""
echo "=== 5. Your associations (sacctmgr show assoc user=$USER format=User,Account,Partition,QOS) ==="
sacctmgr show assoc user="$USER" format=User,Account,Partition,QOS -p 2>/dev/null || echo "sacctmgr show assoc failed"

echo ""
echo "=== Recommendation ==="
echo "Use a QoS that appears in your User|QOS list above. For partition 'general' or 'gpu', avoid DenyQos (e.g. debug is denied on general)."
echo "Example: sbatch --qos=standard batch/learning/run_stage3_experiments.sbatch"
echo "Or set SBATCH_QOS=standard and add #SBATCH --qos=\${SBATCH_QOS} to your batch file."
