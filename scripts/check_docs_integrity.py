#!/usr/bin/env python3
"""Check tracked Markdown repository links and real conflict markers, offline.

External URLs and fragment anchors are not checked. This is a lightweight parser,
not a full CommonMark renderer. Untracked and ignored research files are not read.
"""
from __future__ import annotations
import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]

def tracked_files(root):
    return set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")) - {""}

def markdown_targets(text):
    text = re.sub(r"(?ms)^ *(`{3,}|~{3,})[^\n]*\n.*?^ *\1[^\n]*$", "", text)
    text = re.sub(r"`[^`\n]*`", "", text)
    # Inline destinations (including images), and reference definitions.
    pattern = r"!?\[[^]\n]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+[^)]*)?\)|^ {0,3}\[[^]\n]+\]:\s*(<[^>]*>|\S+)"
    for match in re.finditer(pattern, text, re.M):
        yield (match.group(1) or match.group(2)).strip("<>")

def check(root):
    root = root.resolve()
    files = tracked_files(root)
    errors, checked = [], 0
    for name in sorted(files):
        path = root / name
        if path.suffix not in (".md", ".py", ".yml", ".yaml", ".sh") or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"(?m)^(?:<{7}|>{7}|\|{7}) .+$", text):
            errors.append(f"{name}: unresolved conflict marker")
        if path.suffix != ".md":
            continue
        for target in markdown_targets(text):
            if target.startswith(("#", "//")) or urlsplit(target).scheme:
                continue
            dest = unquote(urlsplit(target).path)
            if not dest:
                continue
            checked += 1
            resolved = (root / dest.lstrip("/") if dest.startswith("/") else path.parent / dest).resolve()
            try:
                rel = resolved.relative_to(root).as_posix()
            except ValueError:
                errors.append(f"{name}: link outside repository: {target}")
                continue
            if rel != "." and rel not in files and not any(f.startswith(rel.rstrip("/") + "/") for f in files):
                errors.append(f"{name}: missing tracked target: {target}")
    return checked, errors

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    count, errors = check(REPO_ROOT)
    for error in errors:
        print(error)
    print(f"Repository-relative destinations checked: {count}; errors: {len(errors)}")
    return bool(errors)

if __name__ == "__main__":
    raise SystemExit(main())
