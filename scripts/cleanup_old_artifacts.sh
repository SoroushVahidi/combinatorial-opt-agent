#!/usr/bin/env bash
# Safe cleanup of old artifacts. Deletes only paths listed in the manifest.
# Run from repository root. Skips nonexistent paths; prints each path before delete.
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${REPO_ROOT}/docs/REPO_CLEANUP_DELETED_FILES_MANIFEST.txt"

cd "$REPO_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 1
fi

echo "Reading paths from $MANIFEST"
while IFS= read -r path || [[ -n "$path" ]]; do
  path="${path%%#*}"
  path="${path// /}"
  [[ -z "$path" ]] && continue
  full="${REPO_ROOT}/${path}"
  if [[ -e "$full" ]]; then
    echo "Deleting: $path"
    rm -rf "$full"
  else
    echo "Skip (not found): $path"
  fi
done < "$MANIFEST"

echo "Done."
