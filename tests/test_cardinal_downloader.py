"""Offline fixtures exercise raw-snapshot staging and failure preservation."""
import io
import json
from urllib.error import HTTPError, URLError

import pytest
from scripts import get_cardinal_nl4opt as downloader


@pytest.fixture
def payload():
    return b'{"en_question":"fixture only", "en_answer":"retained raw"}\n\n{"id":"second"}\n'


def test_success_preserves_raw_fields_and_reports_counts(tmp_path, monkeypatch, payload):
    calls = []
    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)
    monkeypatch.setattr(downloader, "urlopen", fetch)
    assert downloader.main(["--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "provenance.json").read_text())
    assert report["row_counts"] == dict.fromkeys(("train", "dev", "test"), 2)
    assert len(calls) == 3 and all(timeout == 30 for _, timeout in calls)
    assert all((tmp_path / f"{s}.jsonl").read_bytes() == payload for s in report["splits"])


@pytest.mark.parametrize("payload", [b"", b" \n", b"<html>denied</html>", b"{}\n", b"[]\n", b"null\n", b'{"ok":1}\nnot-json', b'\xff'])
def test_invalid_download_never_replaces_existing_split(tmp_path, monkeypatch, payload):
    old = b'{"original":true}\n'
    (tmp_path / "train.jsonl").write_bytes(old)
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: io.BytesIO(payload))
    assert downloader.main(["--output-dir", str(tmp_path), "--force"]) == 1
    assert (tmp_path / "train.jsonl").read_bytes() == old
    assert not (tmp_path / "dev.jsonl").exists()


@pytest.mark.parametrize("error", [HTTPError("fixture", 403, "forbidden", {}, None), URLError("offline"), TimeoutError("timeout")])
def test_network_errors_are_nonzero_without_split_files(tmp_path, monkeypatch, error):
    def fetch(*a, **k): raise error
    monkeypatch.setattr(downloader, "urlopen", fetch)
    assert downloader.main(["--output-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.glob("*.jsonl"))
    assert len(json.loads((tmp_path / "provenance.json").read_text())["errors"]) == 3


def test_existing_files_and_provenance_are_preserved_without_network(tmp_path, monkeypatch, payload):
    for split in downloader.SPLIT_URLS:
        (tmp_path / f"{split}.jsonl").write_bytes(payload)
    (tmp_path / "provenance.json").write_bytes(b'{"historical":"keep"}\n')
    def unexpected(*a, **k): pytest.fail("existing snapshots must not contact upstream")
    monkeypatch.setattr(downloader, "urlopen", unexpected)
    assert downloader.main(["--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "provenance.json").read_bytes() == b'{"historical":"keep"}\n'


def test_invalid_existing_file_requires_explicit_force(tmp_path, monkeypatch):
    for split in downloader.SPLIT_URLS:
        (tmp_path / f"{split}.jsonl").write_bytes(b"broken")
    monkeypatch.setattr(downloader, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert downloader.main(["--output-dir", str(tmp_path)]) == 1
    assert (tmp_path / "train.jsonl").read_bytes() == b"broken"
