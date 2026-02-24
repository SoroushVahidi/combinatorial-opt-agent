"""
Minimal HTTP server to collect remote feedback logs from the web UI.

Usage (local example):
    python feedback_server.py

Then, in another terminal (or on another machine), run the app with:
    export REMOTE_FEEDBACK_ENDPOINT="http://localhost:8000/collect"
    python app.py

Each interaction will be sent as a JSON POST to /collect and appended
to data/feedback/remote_logs.jsonl on the server.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Tuple


ROOT = Path(__file__).resolve().parent
FEEDBACK_DIR = ROOT / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
REMOTE_LOG_PATH = FEEDBACK_DIR / "remote_logs.jsonl"


class FeedbackHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_POST(self) -> None:  # type: ignore[override]
        if self.path != "/collect":
            self._set_headers(404)
            self.wfile.write(b'{"status":"not_found"}')
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""

        try:
            record = json.loads(body.decode("utf-8"))
        except Exception:
            self._set_headers(400)
            self.wfile.write(b'{"status":"error","message":"invalid json"}')
            return

        try:
            with REMOTE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            self._set_headers(500)
            self.wfile.write(b'{"status":"error","message":"could_not_write"}')
            return

        self._set_headers(200)
        self.wfile.write(b'{"status":"ok"}')

    # Silence default logging to keep the server output clean.
    def log_message(self, format: str, *args) -> None:  # type: ignore[override]
        return


def run_server(address: Tuple[str, int] = ("0.0.0.0", 8000)) -> None:
    httpd = HTTPServer(address, FeedbackHandler)
    host, port = address
    print(f"Feedback server listening on http://{host}:{port}/collect")
    print(f"Logs will be appended to {REMOTE_LOG_PATH}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()

