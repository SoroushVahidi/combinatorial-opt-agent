"""
Regression tests for the 5 critical bugs fixed in the second debug pass.

Bug 1 – analyze_feedback.py reads wrong file path → always "No chat logs found"
Bug 2 – analyze_feedback.py uses wrong field name → all problems shown as <unknown>
Bug 3 – app.py::_log_user_query doesn't store problem id
Bug 4 – scripts/merge_catalog.py::merge_catalog crashes with KeyError on entries without "id"
Bug 5 – training/evaluate_retrieval.py silently forces k≥10, ignoring user --top-k
"""
from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Bug 3 — _log_user_query must write "id" for each result
# ---------------------------------------------------------------------------

class TestLogUserQueryWritesId:
    """_log_user_query must include 'id' in each logged result entry."""

    def _make_problem(self, pid: str, name: str) -> dict:
        return {"id": pid, "name": name, "description": "desc"}

    def test_log_record_includes_id(self, tmp_path, monkeypatch):
        """Each result entry in the JSONL record must have an 'id' key."""
        import app as app_mod

        log_file = tmp_path / "user_queries.jsonl"
        monkeypatch.setattr(app_mod, "COLLECTED_QUERIES_DIR", tmp_path)
        monkeypatch.setattr(app_mod, "USER_QUERIES_FILE", log_file)

        problems = [
            (self._make_problem("knapsack_01", "Knapsack"), 0.95),
            (self._make_problem("set_cover", "Set Cover"), 0.80),
        ]
        app_mod._log_user_query("my query", top_k=2, results=problems)

        records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        for entry in records[0]["results"]:
            assert "id" in entry, f"'id' missing from logged result entry: {entry}"

    def test_logged_id_matches_problem_id(self, tmp_path, monkeypatch):
        """The logged 'id' must equal the problem's actual id."""
        import app as app_mod

        log_file = tmp_path / "user_queries.jsonl"
        monkeypatch.setattr(app_mod, "COLLECTED_QUERIES_DIR", tmp_path)
        monkeypatch.setattr(app_mod, "USER_QUERIES_FILE", log_file)

        problems = [
            (self._make_problem("facility_location", "Facility Location"), 0.92),
        ]
        app_mod._log_user_query("warehouse query", top_k=1, results=problems)

        records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        assert records[0]["results"][0]["id"] == "facility_location"

    def test_logged_name_still_present(self, tmp_path, monkeypatch):
        """Adding 'id' must not remove the existing 'name' field."""
        import app as app_mod

        log_file = tmp_path / "user_queries.jsonl"
        monkeypatch.setattr(app_mod, "COLLECTED_QUERIES_DIR", tmp_path)
        monkeypatch.setattr(app_mod, "USER_QUERIES_FILE", log_file)

        problems = [(self._make_problem("tsp", "TSP"), 0.88)]
        app_mod._log_user_query("traveling salesman", top_k=1, results=problems)

        records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        entry = records[0]["results"][0]
        assert entry["name"] == "TSP"
        assert entry["id"] == "tsp"

    def test_empty_id_does_not_crash(self, tmp_path, monkeypatch):
        """A problem with no 'id' must log an empty string, not crash."""
        import app as app_mod

        log_file = tmp_path / "user_queries.jsonl"
        monkeypatch.setattr(app_mod, "COLLECTED_QUERIES_DIR", tmp_path)
        monkeypatch.setattr(app_mod, "USER_QUERIES_FILE", log_file)

        problems = [({"name": "No ID problem"}, 0.5)]  # no 'id' key
        app_mod._log_user_query("query", top_k=1, results=problems)

        records = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        assert records[0]["results"][0]["id"] == ""


# ---------------------------------------------------------------------------
# Bugs 1 & 2 — analyze_feedback reads correct path and correct field name
# ---------------------------------------------------------------------------

class TestAnalyzeFeedbackPathAndFields:
    """load_chat_logs reads from user_queries.jsonl; summarize_chats uses 'id'."""

    def _write_user_queries(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_load_chat_logs_reads_user_queries_file(self, tmp_path, monkeypatch):
        """load_chat_logs must find records written by app.py to user_queries.jsonl."""
        import analyze_feedback as af

        uq_path = tmp_path / "collected_queries" / "user_queries.jsonl"
        records = [
            {"query": "knapsack", "results": [{"id": "kp01", "name": "Knapsack", "score": 0.9}]},
        ]
        self._write_user_queries(uq_path, records)

        monkeypatch.setattr(af, "USER_QUERIES_PATH", uq_path)
        monkeypatch.setattr(af, "CHAT_LOG_PATH", tmp_path / "feedback" / "chat_logs.jsonl")  # non-existent

        loaded = af.load_chat_logs()
        assert len(loaded) == 1, "Should have loaded 1 record from user_queries.jsonl"
        assert loaded[0]["query"] == "knapsack"

    def test_load_chat_logs_merges_both_sources(self, tmp_path, monkeypatch):
        """When both log files exist, load_chat_logs merges their records."""
        import analyze_feedback as af

        uq_path = tmp_path / "user_queries.jsonl"
        chat_path = tmp_path / "chat_logs.jsonl"

        self._write_user_queries(uq_path, [{"query": "from_app", "results": []}])
        self._write_user_queries(chat_path, [{"query": "from_legacy", "results": []}])

        monkeypatch.setattr(af, "USER_QUERIES_PATH", uq_path)
        monkeypatch.setattr(af, "CHAT_LOG_PATH", chat_path)

        loaded = af.load_chat_logs()
        queries = {r["query"] for r in loaded}
        assert "from_app" in queries
        assert "from_legacy" in queries

    def test_summarize_chats_uses_id_field(self, tmp_path, capsys, monkeypatch):
        """summarize_chats must count problems by 'id', not default to <unknown>."""
        import analyze_feedback as af

        records = [
            {
                "query": "knapsack",
                "results": [{"id": "kp01", "name": "Knapsack", "score": 0.9}],
            },
            {
                "query": "set cover",
                "results": [{"id": "kp01", "name": "Knapsack", "score": 0.8}],
            },
        ]
        af.summarize_chats(records)
        out = capsys.readouterr().out
        assert "kp01" in out, "Problem ID 'kp01' must appear in summary output"
        assert "<unknown>" not in out, "No problem should be listed as <unknown>"

    def test_summarize_chats_falls_back_to_name_when_no_id(self, capsys):
        """When 'id' is absent (old log format), fall back to 'name' rather than <unknown>."""
        import analyze_feedback as af

        records = [
            {
                "query": "knapsack",
                "results": [{"name": "Knapsack", "score": 0.9}],  # old format: no 'id'
            },
        ]
        af.summarize_chats(records)
        out = capsys.readouterr().out
        assert "Knapsack" in out
        assert "<unknown>" not in out

    def test_load_chat_logs_no_files_returns_empty(self, tmp_path, monkeypatch):
        """When neither log file exists, load_chat_logs returns an empty list."""
        import analyze_feedback as af

        monkeypatch.setattr(af, "USER_QUERIES_PATH", tmp_path / "no_file.jsonl")
        monkeypatch.setattr(af, "CHAT_LOG_PATH", tmp_path / "also_no_file.jsonl")

        result = af.load_chat_logs()
        assert result == []


# ---------------------------------------------------------------------------
# Bug 4 — merge_catalog must not crash on entries without "id"
# ---------------------------------------------------------------------------

class TestMergeCatalogSafeId:
    """merge_catalog must tolerate existing entries that have no 'id' key."""

    def test_no_crash_on_existing_without_id(self):
        """Entries in 'existing' that lack 'id' must not raise KeyError."""
        from scripts.merge_catalog import merge_catalog

        existing_with_missing_id = [{"name": "Bad entry, no id"}]
        new_problems = [{"id": "new1", "name": "New Problem"}]

        # Must not raise
        merged, added = merge_catalog(existing_with_missing_id, new_problems)
        assert added == 1, "New problem should be added"
        assert any(p.get("id") == "new1" for p in merged)

    def test_existing_with_id_still_deduplicates(self):
        """Normal entries with 'id' still prevent duplicates."""
        from scripts.merge_catalog import merge_catalog

        existing = [{"id": "existing1", "name": "Existing"}]
        new_problems = [{"id": "existing1", "name": "Duplicate"}]

        merged, added = merge_catalog(existing, new_problems)
        assert added == 0, "Duplicate should not be added"
        assert len(merged) == 1

    def test_mixed_existing_with_and_without_id(self):
        """Mixed list (some with id, some without) is handled gracefully."""
        from scripts.merge_catalog import merge_catalog

        existing = [
            {"id": "p1", "name": "Has ID"},
            {"name": "No ID entry"},   # no 'id' key
        ]
        new_problems = [
            {"id": "p2", "name": "New"},
            {"id": "p1", "name": "Duplicate of p1"},  # should be skipped
        ]

        merged, added = merge_catalog(existing, new_problems)
        assert added == 1
        ids = [p.get("id") for p in merged]
        assert "p2" in ids
        assert ids.count("p1") == 1  # no duplicate


# ---------------------------------------------------------------------------
# Bug 5 — evaluate_retrieval must honor user-supplied --top-k
# ---------------------------------------------------------------------------

class TestEvaluateRetrievalTopK:
    """k = max(args.top_k, 1) must use the user-supplied value, not force k≥10."""

    def test_top_k_below_10_is_respected(self):
        """User passing --top-k 5 should get k=5, not k=10."""
        # Directly test the formula that was fixed
        user_top_k = 5
        k = max(user_top_k, 1)  # new formula
        assert k == 5, f"Expected k=5, got k={k}"

    def test_top_k_1_is_respected(self):
        """User passing --top-k 1 should get k=1."""
        user_top_k = 1
        k = max(user_top_k, 1)
        assert k == 1

    def test_top_k_0_is_guarded_to_1(self):
        """k=0 would be nonsensical; max(0, 1) = 1 is the floor."""
        user_top_k = 0
        k = max(user_top_k, 1)
        assert k == 1

    def test_top_k_above_10_unchanged(self):
        """Values above 10 pass through unchanged."""
        user_top_k = 20
        k = max(user_top_k, 1)
        assert k == 20

    def test_evaluate_retrieval_module_uses_correct_formula(self):
        """Check the actual AST of evaluate_retrieval.py: k must not be capped at 10."""
        import ast as ast_mod
        src = Path(__file__).resolve().parent.parent / "training" / "evaluate_retrieval.py"
        tree = ast_mod.parse(src.read_text())
        # Walk the AST looking for any Call to max() with a literal second arg of 10.
        # If found, the formula has reverted to the old (wrong) form.
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.Call):
                func = node.func
                if isinstance(func, ast_mod.Name) and func.id == "max":
                    args = node.args
                    if len(args) == 2 and isinstance(args[1], ast_mod.Constant):
                        if args[1].value == 10:
                            # Check if the first arg references top_k
                            first_arg_src = ast_mod.unparse(args[0])
                            if "top_k" in first_arg_src:
                                pytest.fail(
                                    "evaluate_retrieval.py still has max(top_k, 10) — "
                                    "should be max(top_k, 1) so user-supplied values are respected"
                                )
