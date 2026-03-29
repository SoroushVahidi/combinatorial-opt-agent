"""Optional telemetry: push user interaction records to a private GitHub repository.

Telemetry is **opt-in** via environment variables.  When ``TELEMETRY_REPO``
and ``TELEMETRY_TOKEN`` are not set this module is a no-op.  When set, each
interaction record (query, matched problems, similarity scores) is pushed to a
per-session JSONL file inside the private repository.

Environment variables
---------------------
TELEMETRY_REPO
    GitHub repository in ``"owner/repo"`` format that will receive the data,
    e.g. ``"SoroushVahidi/opt-agent-telemetry"``.  The repository must already
    exist and be accessible with the given token.
TELEMETRY_TOKEN
    A GitHub personal-access-token (classic) with ``repo`` scope on
    ``TELEMETRY_REPO``, or a fine-grained token with *Contents: Read & write*
    permission on that specific repository.

Data collected
--------------
Each pushed record contains exactly:
- ``ts``      : UTC ISO-8601 timestamp of the interaction
- ``query``   : the user's natural-language problem description
- ``top_k``   : the number of results requested
- ``results`` : list of ``{"id", "name", "score"}`` for each returned problem

No personally identifiable information (IP address, browser fingerprint, user
account, etc.) is ever collected.  All data is appended to files under
``queries/<YYYY-MM-DD>/<session-id>.jsonl`` inside the private repository so
that each application session writes to its own file (avoiding push conflicts).

Full disclosure is also present in the project README and in the application's
footer bar.

Opt-out
-------
Simply do **not** set ``TELEMETRY_REPO`` / ``TELEMETRY_TOKEN``.  When these
variables are absent the module does nothing; no network request is made and
no data leaves the machine.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any
import urllib.error
import urllib.request


# ── Configuration ─────────────────────────────────────────────────────────────

_TELEMETRY_REPO: str = os.environ.get("TELEMETRY_REPO", "").strip()
_TELEMETRY_TOKEN: str = os.environ.get("TELEMETRY_TOKEN", "").strip()

# Unique ID for the current process/session.  Multiple users running the app
# on different machines each get a different ID so their writes land in
# separate files and never conflict.
_SESSION_ID: str = uuid.uuid4().hex

# GitHub Contents API base
_GH_API = "https://api.github.com"

# Maximum in-flight pushes; extra records are silently dropped rather than
# blocking the application.
_MAX_WORKERS = 4
_semaphore = threading.Semaphore(_MAX_WORKERS)

# Enabled only when both env vars are present.
_ENABLED: bool = bool(_TELEMETRY_REPO and _TELEMETRY_TOKEN)


# ── Public API ────────────────────────────────────────────────────────────────

def push_record(record: dict[str, Any]) -> None:
    """Fire-and-forget push of one interaction record to the private repo.

    Returns immediately; the actual network I/O happens in a daemon thread.
    All errors (network, auth, rate-limit) are silently ignored so the caller
    is never affected.  If the push queue is full (> ``_MAX_WORKERS`` concurrent
    pushes already in flight) the record is silently dropped.
    """
    if not _ENABLED:
        return
    if not _semaphore.acquire(blocking=False):
        # Too many pushes in flight — drop this record rather than queuing
        return
    t = threading.Thread(target=_push_worker, args=(record,), daemon=True)
    t.start()


def is_enabled() -> bool:
    """Return True if telemetry is configured and will actually push data."""
    return _ENABLED


# ── Internal ──────────────────────────────────────────────────────────────────

def _push_worker(record: dict[str, Any]) -> None:
    """Worker that runs in a daemon thread and releases the semaphore when done."""
    try:
        _push_to_github(record)
    except Exception as exc:  # noqa: BLE001
        # Print to stderr so operators can diagnose auth/permission problems without
        # crashing the application.  Users never see this output.
        import sys
        print(f"[telemetry] push failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        _semaphore.release()


def _push_to_github(record: dict[str, Any]) -> None:
    """Append *record* to the session JSONL file in the private repo.

    File path inside the repo: ``queries/<YYYY-MM-DD>/<session_id>.jsonl``

    The GitHub Contents API is used for a read-then-update cycle:
    1. GET the current file content (to obtain the current SHA and existing lines).
    2. Append the new JSON line.
    3. PUT the updated content back.

    If the file does not yet exist it is created with just this record.
    """
    date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    file_path = f"queries/{date_str}/{_SESSION_ID}.jsonl"
    url = f"{_GH_API}/repos/{_TELEMETRY_REPO}/contents/{file_path}"

    headers = {
        "Authorization": f"Bearer {_TELEMETRY_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "combinatorial-opt-agent-telemetry/1.0",
        "Content-Type": "application/json",
    }

    new_line = json.dumps(record, ensure_ascii=False) + "\n"

    # --- Step 1: try to read existing file content and SHA ---------------
    current_sha: str | None = None
    existing_content: str = ""
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            current_sha = body.get("sha")
            raw_b64 = body.get("content", "")
            # GitHub returns base64 with newlines; strip before decoding
            existing_content = base64.b64decode(
                raw_b64.replace("\n", "")
            ).decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # File does not exist yet — will be created below
            pass
        else:
            raise

    # --- Step 2: build updated content -----------------------------------
    updated_content = existing_content + new_line
    encoded = base64.b64encode(updated_content.encode("utf-8")).decode("ascii")

    commit_message = f"telemetry: {date_str}/{_SESSION_ID}"
    payload: dict[str, Any] = {
        "message": commit_message,
        "content": encoded,
    }
    if current_sha:
        payload["sha"] = current_sha

    # --- Step 3: create or update ----------------------------------------
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    # GitHub returns 200 (update) or 201 (create); both are success
    with urllib.request.urlopen(req, timeout=15) as resp:
        resp.read()  # drain response body
