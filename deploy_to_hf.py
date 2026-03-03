"""Create HF Space and upload app + catalog. Requires HF_TOKEN or huggingface-cli login."""
import os
import subprocess
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parent
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

def main():
    if not token:
        # Try cached token from huggingface-cli
        cache = Path.home() / ".cache" / "huggingface" / "token"
        if cache.exists():
            token_str = cache.read_text().strip()
            if token_str:
                os.environ["HF_TOKEN"] = token_str
                global token
                token = token_str
    if not token:
        print("Set HF_TOKEN or run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    who = api.whoami()
    username = who["name"]
    repo_id = f"{username}/combinatorial-opt-agent"
    sdk = "gradio"

    create_repo(repo_id, repo_type="space", space_sdk=sdk, token=token, exist_ok=True)

    # Upload only what the Space needs
    files_to_upload = [
        ("app.py", ROOT / "app.py"),
        ("requirements.txt", ROOT / "requirements.txt"),
        ("retrieval/search.py", ROOT / "retrieval" / "search.py"),
        ("retrieval/__init__.py", ROOT / "retrieval" / "__init__.py"),
        ("data/processed/all_problems.json", ROOT / "data" / "processed" / "all_problems.json"),
    ]
    for path_in_repo, local_path in files_to_upload:
        if not local_path.exists():
            print(f"Skip (missing): {local_path}", file=sys.stderr)
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
        print(f"Uploaded {path_in_repo}")

    # Space README for Gradio SDK
    readme_path = ROOT / "data" / "processed" / "README_space.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("""---
title: Combinatorial Optimization Bot
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
---
# Combinatorial Optimization Bot
Describe your problem in plain English; get matching optimization formulations.
""")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    out = ROOT / "data" / "processed" / "public_url.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(url)
    print(url)

if __name__ == "__main__":
    main()
