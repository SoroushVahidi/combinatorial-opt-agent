"""
Tests for the synthetic data generation + training pipeline (training/run_pipeline.py).

Covers:
- dry-run mode: no files written, correct output messages
- skip-train mode: splits and pairs written but no training
- split coverage: all catalog IDs appear in exactly one split
- pair quality: every pair's passage belongs to a train-split problem
- step_build_splits: returns disjoint splits covering all catalog IDs
- step_generate_pairs: output JSONL is valid and uses only train problems
- main() CLI: end-to-end invocation with --skip-train
- GPU / device detection helpers: _detect_device, _print_gpu_info
- fp16 and dataloader_workers flags propagate correctly to step_train dry-run
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_catalog() -> list[dict]:
    """Return a minimal 20-problem catalog for fast tests."""
    return [
        {
            "id": f"prob_{i}",
            "name": f"Problem {i}",
            "aliases": [f"alias_{i}"],
            "description": f"Description for problem {i}. It has constraints.",
            "source": "classic" if i % 2 == 0 else "nl4opt",
        }
        for i in range(20)
    ]


# ---------------------------------------------------------------------------
# step_build_splits
# ---------------------------------------------------------------------------

class TestStepBuildSplits:
    def test_disjoint(self, tmp_path):
        from training.run_pipeline import step_build_splits

        splits_path = tmp_path / "splits.json"
        catalog_path = tmp_path / "catalog.json"
        catalog_path.write_text(json.dumps(_tiny_catalog()), encoding="utf-8")

        # Monkey-patch load_catalog so the step uses our tiny catalog
        import training.splits as splits_mod
        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        try:
            splits = step_build_splits(
                splits_path=splits_path,
                seed=0,
                train_ratio=0.70,
                dev_ratio=0.15,
                test_ratio=0.15,
                dry_run=False,
            )
        finally:
            splits_mod.load_catalog = orig

        train = set(splits["train"])
        dev = set(splits["dev"])
        test = set(splits["test"])
        assert train & dev == set()
        assert train & test == set()
        assert dev & test == set()

    def test_covers_all_ids(self, tmp_path):
        from training.run_pipeline import step_build_splits
        import training.splits as splits_mod

        splits_path = tmp_path / "splits.json"
        catalog = _tiny_catalog()
        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: catalog
        try:
            splits = step_build_splits(
                splits_path=splits_path,
                seed=42,
                train_ratio=0.70,
                dev_ratio=0.15,
                test_ratio=0.15,
                dry_run=False,
            )
        finally:
            splits_mod.load_catalog = orig

        all_ids = {p["id"] for p in catalog}
        split_ids = set(splits["train"]) | set(splits["dev"]) | set(splits["test"])
        assert split_ids == all_ids

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        from training.run_pipeline import step_build_splits
        import training.splits as splits_mod

        splits_path = tmp_path / "splits.json"
        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        try:
            step_build_splits(
                splits_path=splits_path,
                seed=0,
                train_ratio=0.70,
                dev_ratio=0.15,
                test_ratio=0.15,
                dry_run=True,
            )
        finally:
            splits_mod.load_catalog = orig

        assert not splits_path.exists(), "dry-run must not write any file"
        out = capsys.readouterr().out
        assert "dry-run" in out.lower()

    def test_file_written(self, tmp_path):
        from training.run_pipeline import step_build_splits
        import training.splits as splits_mod

        splits_path = tmp_path / "splits.json"
        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        try:
            step_build_splits(
                splits_path=splits_path,
                seed=7,
                train_ratio=0.60,
                dev_ratio=0.20,
                test_ratio=0.20,
                dry_run=False,
            )
        finally:
            splits_mod.load_catalog = orig

        assert splits_path.exists()
        data = json.loads(splits_path.read_text())
        assert "train" in data and "dev" in data and "test" in data


# ---------------------------------------------------------------------------
# step_generate_pairs
# ---------------------------------------------------------------------------

class TestStepGeneratePairs:
    def _make_splits_from_catalog(self, catalog: list[dict]) -> dict[str, list[str]]:
        ids = [p["id"] for p in catalog]
        n = len(ids)
        return {
            "train": ids[: n * 7 // 10],
            "dev": ids[n * 7 // 10 : n * 7 // 10 + n * 15 // 100],
            "test": ids[n * 7 // 10 + n * 15 // 100 :],
        }

    def test_pairs_only_from_train_problems(self, tmp_path):
        from training.run_pipeline import step_generate_pairs
        from training.generate_samples import searchable_text

        catalog = _tiny_catalog()
        splits = self._make_splits_from_catalog(catalog)
        pairs_path = tmp_path / "pairs.jsonl"

        step_generate_pairs(
            splits=splits,
            pairs_path=pairs_path,
            instances_per_problem=5,
            seed=0,
            dry_run=False,
        )

        assert pairs_path.exists()
        train_passages = {searchable_text(p) for p in catalog if p["id"] in splits["train"]}
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                assert obj["passage"] in train_passages, (
                    "All passages must come from train-split problems"
                )

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        from training.run_pipeline import step_generate_pairs

        catalog = _tiny_catalog()
        splits = self._make_splits_from_catalog(catalog)
        pairs_path = tmp_path / "pairs.jsonl"

        count = step_generate_pairs(
            splits=splits,
            pairs_path=pairs_path,
            instances_per_problem=10,
            seed=0,
            dry_run=True,
        )

        assert not pairs_path.exists(), "dry-run must not write any file"
        assert count > 0, "dry-run should return a positive estimated count"
        out = capsys.readouterr().out
        assert "dry-run" in out.lower()

    def test_output_is_valid_jsonl(self, tmp_path):
        from training.run_pipeline import step_generate_pairs

        catalog = _tiny_catalog()
        splits = self._make_splits_from_catalog(catalog)
        pairs_path = tmp_path / "pairs.jsonl"

        n = step_generate_pairs(
            splits=splits,
            pairs_path=pairs_path,
            instances_per_problem=3,
            seed=42,
            dry_run=False,
        )

        lines = pairs_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == n
        for line in lines:
            obj = json.loads(line)
            assert "query" in obj and "passage" in obj
            assert obj["query"] and obj["passage"]

    def test_no_dev_or_test_passages(self, tmp_path):
        from training.run_pipeline import step_generate_pairs
        from training.generate_samples import searchable_text

        catalog = _tiny_catalog()
        splits = self._make_splits_from_catalog(catalog)
        pairs_path = tmp_path / "pairs.jsonl"

        step_generate_pairs(
            splits=splits,
            pairs_path=pairs_path,
            instances_per_problem=5,
            seed=0,
            dry_run=False,
        )

        eval_ids = set(splits["dev"]) | set(splits["test"])
        eval_passages = {searchable_text(p) for p in catalog if p["id"] in eval_ids}
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                passage = json.loads(line)["passage"]
                assert passage not in eval_passages, (
                    "No dev/test problem passage must appear in training pairs"
                )


# ---------------------------------------------------------------------------
# main() CLI integration
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_dry_run_end_to_end(self, tmp_path, capsys):
        """main() --dry-run should print messages and write no files."""
        import training.splits as splits_mod
        from training.run_pipeline import main

        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        splits_path = tmp_path / "splits.json"
        pairs_path = tmp_path / "pairs.jsonl"
        model_out = tmp_path / "model"
        try:
            main([
                "--dry-run",
                "--splits-out", str(splits_path),
                "--pairs-out", str(pairs_path),
                "--model-out", str(model_out),
                "--instances-per-problem", "5",
            ])
        finally:
            splits_mod.load_catalog = orig

        assert not splits_path.exists()
        assert not pairs_path.exists()
        out = capsys.readouterr().out
        assert "dry-run" in out.lower()
        assert "pipeline" in out.lower()

    def test_skip_train_generates_data(self, tmp_path):
        """main() --skip-train must write splits.json and training_pairs.jsonl."""
        import training.splits as splits_mod
        from training.run_pipeline import main

        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        splits_path = tmp_path / "splits.json"
        pairs_path = tmp_path / "pairs.jsonl"
        model_out = tmp_path / "model"
        try:
            main([
                "--skip-train",
                "--splits-out", str(splits_path),
                "--pairs-out", str(pairs_path),
                "--model-out", str(model_out),
                "--instances-per-problem", "3",
            ])
        finally:
            splits_mod.load_catalog = orig

        assert splits_path.exists(), "splits.json must be written"
        assert pairs_path.exists(), "training_pairs.jsonl must be written"
        assert not (model_out / "final").exists(), "No model should be trained with --skip-train"

        splits = json.loads(splits_path.read_text())
        assert "train" in splits and "dev" in splits and "test" in splits
        train = set(splits["train"])
        dev = set(splits["dev"])
        test = set(splits["test"])
        assert train & dev == set() and train & test == set() and dev & test == set()

    def test_skip_splits_reuses_existing(self, tmp_path):
        """--skip-splits must reuse the existing splits file and not overwrite it."""
        import training.splits as splits_mod
        from training.run_pipeline import main

        # Pre-write a tiny splits.json
        ids = [f"prob_{i}" for i in range(20)]
        pre_splits = {"train": ids[:14], "dev": ids[14:17], "test": ids[17:]}
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(json.dumps(pre_splits), encoding="utf-8")
        mtime_before = splits_path.stat().st_mtime

        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        pairs_path = tmp_path / "pairs.jsonl"
        model_out = tmp_path / "model"
        try:
            main([
                "--skip-train",
                "--skip-splits",
                "--splits-out", str(splits_path),
                "--pairs-out", str(pairs_path),
                "--model-out", str(model_out),
                "--instances-per-problem", "2",
            ])
        finally:
            splits_mod.load_catalog = orig

        # splits.json must not have been replaced
        assert splits_path.stat().st_mtime == mtime_before, (
            "--skip-splits must not overwrite the existing splits.json"
        )
        assert pairs_path.exists(), "Pairs must still be generated"


# ---------------------------------------------------------------------------
# GPU / device detection
# ---------------------------------------------------------------------------

class TestDeviceDetection:
    def test_detect_device_returns_string(self):
        """_detect_device must return 'cuda' or 'cpu' without raising."""
        from training.run_pipeline import _detect_device
        device = _detect_device()
        assert device in ("cuda", "cpu")

    def test_print_gpu_info_returns_string(self, capsys):
        """_print_gpu_info must return a device string and print something."""
        from training.run_pipeline import _print_gpu_info
        device = _print_gpu_info()
        assert device in ("cuda", "cpu")
        out = capsys.readouterr().out
        assert "[device]" in out

    def test_no_fp16_flag_is_honoured(self, tmp_path, capsys):
        """--no-fp16 dry-run output must show fp16=False regardless of hardware."""
        import training.splits as splits_mod
        from training.run_pipeline import main

        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        splits_path = tmp_path / "splits.json"
        pairs_path = tmp_path / "pairs.jsonl"
        model_out = tmp_path / "model"
        try:
            main([
                "--dry-run",
                "--no-fp16",
                "--splits-out", str(splits_path),
                "--pairs-out", str(pairs_path),
                "--model-out", str(model_out),
                "--instances-per-problem", "2",
            ])
        finally:
            splits_mod.load_catalog = orig

        out = capsys.readouterr().out
        # The dry-run step_train message must report fp16=False
        assert "fp16=False" in out

    def test_fp16_flag_is_honoured(self, tmp_path, capsys):
        """--fp16 dry-run output must show fp16=True regardless of hardware."""
        import training.splits as splits_mod
        from training.run_pipeline import main

        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        splits_path = tmp_path / "splits.json"
        pairs_path = tmp_path / "pairs.jsonl"
        model_out = tmp_path / "model"
        try:
            main([
                "--dry-run",
                "--fp16",
                "--splits-out", str(splits_path),
                "--pairs-out", str(pairs_path),
                "--model-out", str(model_out),
                "--instances-per-problem", "2",
            ])
        finally:
            splits_mod.load_catalog = orig

        out = capsys.readouterr().out
        assert "fp16=True" in out

    def test_dataloader_workers_flag_appears_in_dry_run(self, tmp_path, capsys):
        """--dataloader-workers is accepted and flows into the pipeline without error."""
        import training.splits as splits_mod
        from training.run_pipeline import main

        orig = splits_mod.load_catalog
        splits_mod.load_catalog = lambda *a, **kw: _tiny_catalog()
        splits_path = tmp_path / "splits.json"
        pairs_path = tmp_path / "pairs.jsonl"
        model_out = tmp_path / "model"
        try:
            # Should not raise; dry-run so no actual training happens
            main([
                "--dry-run",
                "--dataloader-workers", "2",
                "--splits-out", str(splits_path),
                "--pairs-out", str(pairs_path),
                "--model-out", str(model_out),
                "--instances-per-problem", "2",
            ])
        finally:
            splits_mod.load_catalog = orig
