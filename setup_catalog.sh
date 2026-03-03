#!/usr/bin/env bash
# One-shot setup: install deps, fetch public data, merge into catalog.
# Run from repo root: bash setup_catalog.sh

set -e
cd "$(dirname "$0")"

echo "== Installing dependencies =="
pip install -q -r requirements.txt

echo "== Collecting public data (NL4Opt) =="
python pipeline/run_collection.py

echo "== Verifying catalog =="
python -c "
from retrieval.search import _load_catalog
c = _load_catalog()
print(f'  Catalog: {len(c)} problems')
nl4 = [p for p in c if p.get(\"source\") == \"NL4Opt\"]
print(f'  NL4Opt:  {len(nl4)}')
"

echo "== Done. Try: python app.py   or   python run_search.py \"your problem\""
