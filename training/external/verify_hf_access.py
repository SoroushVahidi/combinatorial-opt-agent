"""Quick smoke-test: verify that HF_TOKEN is set and the NLP4LP dataset is reachable.

Exit 0  → token is valid and dataset is accessible.
Exit 1  → token is missing or dataset cannot be reached.

Usage (local):
    export HF_TOKEN=hf_...
    python training/external/verify_hf_access.py

Usage (CI): called automatically by .github/workflows/check-hf-access.yml and
    as a fast-fail preflight step in .github/workflows/nlp4lp.yml.
"""
from __future__ import annotations

import os
import sys


DATASET_ID = "udell-lab/NLP4LP"


def _get_token() -> str:
    for var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return value
    return ""


def verify() -> bool:
    token = _get_token()
    if not token:
        print("ERROR: No HuggingFace token found in environment.")
        print("  Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) and re-run.")
        print("  In GitHub Actions: add HF_TOKEN as a repository secret.")
        return False

    print(f"HF_TOKEN is set (length={len(token)}).")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    api = HfApi(token=token)
    try:
        info = api.dataset_info(DATASET_ID)
        print(f"Dataset reachable: {info.id}")
        return True
    except Exception as e:
        print(f"ERROR: Could not reach dataset '{DATASET_ID}': {e}")
        print("Check that:")
        print("  1. Your token has read access (https://huggingface.co/settings/tokens)")
        print(f"  2. You have been granted access to {DATASET_ID}")
        return False


def main() -> None:
    ok = verify()
    if ok:
        print("HuggingFace access: OK")
    else:
        print("HuggingFace access: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
