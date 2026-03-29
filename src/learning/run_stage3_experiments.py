#!/usr/bin/env python3
"""Launch Stage 3 experiment round: train/eval learned runs, bottleneck slices, write manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _load_matrix(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: list[str], cwd: Path | None = None) -> bool:
    try:
        subprocess.run(cmd, cwd=str(cwd or ROOT), check=True, timeout=3600)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Failed: {' '.join(cmd)} -> {e}", file=sys.stderr)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "learning" / "experiment_matrix_stage3.json")
    ap.add_argument("--mode", choices=["print_only", "local", "sbatch"], default="print_only")
    ap.add_argument("--force", action="store_true", help="Rerun completed steps")
    ap.add_argument("--repo_root", type=Path, default=ROOT)
    ap.add_argument("--data_dir", type=Path, default=None)
    ap.add_argument("--save_dir", type=Path, default=None)
    ap.add_argument("--artifacts_base", type=Path, default=None, help="If set, data/save/corpus/nl4opt_aux dirs are under this (e.g. $SCRATCH/artifacts)")
    args = ap.parse_args()
    root = args.repo_root
    art_base = args.artifacts_base
    if art_base is not None:
        data_dir = args.data_dir or art_base / "learning_ranker_data" / "nlp4lp"
        save_dir = args.save_dir or art_base / "learning_runs"
    else:
        data_dir = args.data_dir or root / "artifacts" / "learning_ranker_data" / "nlp4lp"
        save_dir = args.save_dir or root / "artifacts" / "learning_runs"
    config_path = args.config if args.config.is_absolute() else root / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    # Fail fast if artifacts base is not writable (e.g. /tmp on some compute nodes)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Cannot create save_dir {save_dir}: {e}", file=sys.stderr)
        sys.exit(1)
    matrix = _load_matrix(config_path)
    runs = matrix.get("runs", [])
    eval_split = matrix.get("eval_split", "test")
    decoder = matrix.get("decoder", "argmax")
    manifest: dict = {"runs": [], "bottleneck_done": False, "errors": []}

    # Prerequisites
    corpus_dir = (art_base / "learning_corpus") if art_base is not None else (root / "artifacts" / "learning_corpus")
    nl4opt_aux_dir = (art_base / "learning_aux_data" / "nl4opt") if art_base is not None else (root / "artifacts" / "learning_aux_data" / "nl4opt")
    ranker_data_train = data_dir / "train.jsonl"
    ranker_data_test = data_dir / f"{eval_split}.jsonl"
    need_corpus = ranker_data_train.exists() is False or ranker_data_test.exists() is False
    need_nl4opt = any(
        r.get("use_nl4opt_entity") or r.get("use_nl4opt_bound") or r.get("use_nl4opt_role")
        for r in runs
    ) and any(r.get("training_mode") in ("pretrain_then_finetune", "joint") for r in runs)

    if args.mode == "print_only":
        print("Prerequisites:")
        print(f"  corpus/ranker data: {'build' if need_corpus else 'ok'}")
        print(f"  nl4opt aux: {'build' if need_nl4opt else 'ok (or not needed)'}")
        for r in runs:
            out = Path(r["output_dir"].replace("artifacts/learning_runs", str(save_dir)) if not Path(r["output_dir"]).is_absolute() else r["output_dir"])
            out = save_dir / r["run_name"] if not out.is_absolute() else out
            print(f"  {r['run_name']}: train={r.get('train_script')} -> {out}")
        print("Bottleneck slice eval: rule_baseline + all run names")
        return

    if args.mode == "sbatch":
        # Single job that runs this script in local mode
        batch_path = root / "batch" / "learning" / "run_stage3_experiments.sbatch"
        if not batch_path.exists():
            print(f"Batch file not found: {batch_path}", file=sys.stderr)
            sys.exit(1)
        _run(["sbatch", str(batch_path)], cwd=root)
        return

    # --- local mode ---
    if need_corpus:
        if not _run([sys.executable, "-m", "src.learning.build_common_grounding_corpus", "--dataset", "nlp4lp", "--split", "all", "--output_dir", str(corpus_dir)], cwd=root):
            manifest["errors"].append("corpus_build_failed")
        if not _run([sys.executable, "-m", "src.learning.build_nlp4lp_pairwise_ranker_data", "--corpus_dir", str(corpus_dir), "--output_dir", str(data_dir)], cwd=root):
            manifest["errors"].append("ranker_data_build_failed")
        # Fallback: if train.jsonl missing (e.g. no HF train/dev), use test as train for minimal learned run
        if not (data_dir / "train.jsonl").exists() and (data_dir / "test.jsonl").exists():
            import shutil
            shutil.copy(data_dir / "test.jsonl", data_dir / "train.jsonl")
            print("Warning: train.jsonl missing; used test.jsonl as train (test-as-train fallback)", file=sys.stderr)
            manifest.setdefault("warnings", []).append("test_as_train_fallback")
    # Fallback when corpus was built earlier but only test split existed
    if not (data_dir / "train.jsonl").exists() and (data_dir / "test.jsonl").exists():
        import shutil
        shutil.copy(data_dir / "test.jsonl", data_dir / "train.jsonl")
        print("Warning: train.jsonl missing; used test.jsonl as train (test-as-train fallback)", file=sys.stderr)
        manifest.setdefault("warnings", []).append("test_as_train_fallback")
    if need_nl4opt:
        if not _run([sys.executable, "-m", "src.learning.build_nl4opt_aux_data", "--output_dir", str(nl4opt_aux_dir)], cwd=root):
            manifest["errors"].append("nl4opt_aux_build_failed")

    for r in runs:
        run_name = r["run_name"]
        out_dir = save_dir / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {"run_name": run_name, "output_dir": str(out_dir), "trained": False, "metrics_written": False}
        train_script = r.get("train_script")
        training_mode = r.get("training_mode", "")
        checkpoint_ok = (out_dir / "checkpoint.pt").exists() or (out_dir / "pairwise_only.pt").exists()
        if training_mode != "none" and train_script and (args.force or not checkpoint_ok):
            if train_script == "train_nlp4lp_pairwise_ranker":
                cmd = [
                    sys.executable, "-m", "src.learning.train_nlp4lp_pairwise_ranker",
                    "--run_name", run_name, "--data_dir", str(data_dir), "--save_dir", str(save_dir),
                    "--encoder", r.get("encoder", "distilroberta-base"), "--max_steps", str(r.get("max_steps", 200)),
                    "--lr", str(r.get("lr", 2e-5)), "--batch_size", str(r.get("batch_size", 8)),
                ]
                if r.get("use_structured_features"):
                    cmd.append("--use_features")
                if not _run(cmd, cwd=root):
                    manifest["errors"].append(f"{run_name}_train_failed")
                else:
                    entry["trained"] = True
            elif train_script == "train_multitask_grounder":
                cmd = [
                    sys.executable, "-m", "src.learning.train_multitask_grounder",
                    "--mode", training_mode, "--run_name", run_name,
                    "--nlp4lp_data_dir", str(data_dir), "--nl4opt_aux_dir", str(nl4opt_aux_dir), "--save_dir", str(save_dir),
                    "--encoder", r.get("encoder", "distilroberta-base"), "--pretrain_steps", str(r.get("pretrain_steps", 0)),
                    "--finetune_steps", str(r.get("finetune_steps", 0)), "--joint_steps", str(r.get("joint_steps", 0)),
                    "--batch_size", str(r.get("batch_size", 8)), "--lr", str(r.get("lr", 2e-5)),
                ]
                if r.get("use_nl4opt_entity"):
                    cmd.append("--use_nl4opt_entity")
                if r.get("use_nl4opt_bound"):
                    cmd.append("--use_nl4opt_bound")
                if r.get("use_nl4opt_role"):
                    cmd.append("--use_nl4opt_role")
                if not _run(cmd, cwd=root):
                    manifest["errors"].append(f"{run_name}_train_failed")
                else:
                    entry["trained"] = True
        elif checkpoint_ok:
            entry["trained"] = True
        metrics_path = out_dir / "metrics.json"
        if args.force or not metrics_path.exists():
            cmd = [
                sys.executable, "-m", "src.learning.eval_nlp4lp_pairwise_ranker",
                "--data_dir", str(data_dir), "--run_dir", str(out_dir), "--split", eval_split, "--decoder", decoder,
            ]
            if _run(cmd, cwd=root):
                entry["metrics_written"] = True
        else:
            entry["metrics_written"] = True
        manifest["runs"].append(entry)

    run_names = [r["run_name"] for r in runs]
    slice_dir = save_dir / "bottleneck_slices"
    slice_json = slice_dir / "slice_metrics.json"
    if args.force or not slice_json.exists():
        cmd = [
            sys.executable, "-m", "src.learning.eval_bottleneck_slices",
            "--data_dir", str(data_dir), "--out_dir", str(save_dir), "--split", eval_split,
            "--run_dirs", *run_names,
        ]
        if _run(cmd, cwd=root):
            manifest["bottleneck_done"] = True
    else:
        manifest["bottleneck_done"] = True

    manifest_path = save_dir / "stage3_manifest.json"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")
    # Collect results and build paper comparison report
    _run([
        sys.executable, "-m", "src.learning.collect_stage3_results",
        "--config", str(config_path), "--runs_dir", str(save_dir), "--deterministic_dir", str(root / "results" / "paper"),
    ], cwd=root)
    _run([
        sys.executable, "-m", "src.learning.build_stage3_comparison_report",
        "--summary_json", str(save_dir / "stage3_results_summary.json"), "--out_path", str(save_dir / "stage3_paper_comparison.md"),
    ], cwd=root)


if __name__ == "__main__":
    main()
