#!/bin/bash
set -euo pipefail

cd /home/soroush/combinatorial-opt-agent
echo "Starting external validation at $(date -u)"
echo "Git SHA: $(git rev-parse HEAD)"
echo "Command: python3 scripts/run_external_validation_optmath.py"

python3 scripts/run_external_validation_optmath.py
echo "Finished external validation at $(date -u)"