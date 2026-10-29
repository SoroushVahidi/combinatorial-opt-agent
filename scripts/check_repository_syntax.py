#!/usr/bin/env python3
"""Parse tracked Python and bash scripts without importing or executing experiments."""
import ast
import subprocess
import warnings
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    counts, errors = {"python": 0, "shell": 0}, []
    for name in paths:
        path = root / name
        try:
            if path.suffix == ".py":
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    ast.parse(path.read_bytes(), filename=name)
                counts["python"] += 1
            elif path.suffix in (".sh", ".sbatch", ".slurm"):
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(result.stderr.strip())
                counts["shell"] += 1
        except (SyntaxError, Warning) as exc:
            errors.append(f"{name}: {exc}")
    print(counts)
    for error in errors:
        print(error)
    return bool(errors)

if __name__ == "__main__":
    raise SystemExit(main())
