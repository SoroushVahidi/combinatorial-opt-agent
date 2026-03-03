#!/usr/bin/env bash
# Run the Gradio app on Wulver. Use from a compute node:
#   srun --pty -p general --qos=low bash
#   then: cd .../combinatorial-opt-agent && bash run_app_wulver.sh
# From laptop: ssh -L 7860:localhost:7860 YOUR_NETID@wulver.njit.edu  and open http://127.0.0.1:7860

set -e
cd "$(dirname "$0")"

# Avoid Xet backend so Hugging Face doesn't spawn worker threads (fails on login nodes)
export HF_HUB_DISABLE_XET=1

echo "Loading embedding model (one-time if not cached)..."
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
print('Model ready.')
"

echo "Starting server at http://0.0.0.0:7860 ..."
echo "From your laptop open: http://127.0.0.1:7860 (with ssh -L 7860:localhost:7860)"
exec python app.py
