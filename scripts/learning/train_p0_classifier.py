#!/usr/bin/env python3
"""Train the P0 local-compatibility classifier(s) and select on validation only.

Preregistered small model set (see docs/LEARNED_GROUNDING_P0.md "Classifier"):
  - logistic regression (linear, calibrated via sklearn's default logistic loss)
  - gradient-boosted trees (sklearn HistGradientBoostingClassifier)

No transformer fine-tuning, no broad hyperparameter search. Model selection
uses the DEV split's group-level (slot-selection) accuracy only -- the TEST
split is not touched by this script.

Usage:
    python3 scripts/learning/train_p0_classifier.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tools.learned_local_scorer import ALL_FEATURE_NAMES, feature_dict_to_vector  # noqa: E402

DATA_DIR = ROOT / "artifacts" / "learning_ranker_data" / "nlp4lp_p0"
OUT_DIR = ROOT / "artifacts" / "learning_runs" / "p0"


def load_split(split: str) -> list[dict]:
    rows = []
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def row_vector(row: dict, feature_names: tuple[str, ...]) -> list[float]:
    feats = dict(row["engineered_features"])
    feats["embedding_similarity"] = row.get("embedding_similarity", 0.0)
    return feature_dict_to_vector(feats, feature_names)


def group_accuracy(rows: list[dict], scores: list[float]) -> dict[str, float]:
    """Slot-selection accuracy: for each group_id (slot), did argmax score match gold?"""
    by_group: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        by_group[row["group_id"]].append((row["mention_id"], score, row["label"]))
    correct = 0
    total = 0
    for gid, candidates in by_group.items():
        gold_mid = next((mid for mid, _, lab in candidates if lab == 1), None)
        if gold_mid is None:
            continue
        total += 1
        best_mid = max(candidates, key=lambda x: x[1])[0]
        if best_mid == gold_mid:
            correct += 1
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def train_model(kind: str, X_train, y_train):
    if kind == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42))
    elif kind == "gradient_boosted_trees":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.08, random_state=42, class_weight="balanced")
    else:
        raise ValueError(kind)
    model.fit(X_train, y_train)
    return model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = load_split("train")
    dev_rows = load_split("dev")

    feature_sets = {
        "combined": ALL_FEATURE_NAMES,
        "engineered_only": tuple(n for n in ALL_FEATURE_NAMES if n != "embedding_similarity"),
        "embedding_only": ("embedding_similarity",),
    }
    model_kinds = ["logistic_regression", "gradient_boosted_trees"]

    y_train = [r["label"] for r in train_rows]
    y_dev = [r["label"] for r in dev_rows]

    # Step 1: model-type selection on the COMBINED feature set (dev only).
    selection_results = {}
    trained_models = {}
    for kind in model_kinds:
        X_train = [row_vector(r, feature_sets["combined"]) for r in train_rows]
        X_dev = [row_vector(r, feature_sets["combined"]) for r in dev_rows]
        model = train_model(kind, X_train, y_train)
        dev_scores = [float(p[1]) for p in model.predict_proba(X_dev)]
        acc = group_accuracy(dev_rows, dev_scores)
        selection_results[kind] = acc
        trained_models[(kind, "combined")] = model
        print(f"[model selection] {kind} (combined features): dev slot-selection accuracy = {acc['accuracy']:.4f} ({acc['correct']}/{acc['total']})")

    best_kind = max(selection_results, key=lambda k: selection_results[k]["accuracy"])
    print(f"Selected model type (validation-only): {best_kind}")

    # Step 2: ablation for the SELECTED model type across feature sets (dev only, informational).
    ablation_results = {}
    for fname, fnames in feature_sets.items():
        key = (best_kind, fname)
        if key not in trained_models:
            X_train = [row_vector(r, fnames) for r in train_rows]
            trained_models[key] = train_model(best_kind, X_train, y_train)
        X_dev = [row_vector(r, fnames) for r in dev_rows]
        dev_scores = [float(p[1]) for p in trained_models[key].predict_proba(X_dev)]
        ablation_results[fname] = group_accuracy(dev_rows, dev_scores)
        print(f"[ablation] {best_kind} / {fname}: dev slot-selection accuracy = {ablation_results[fname]['accuracy']:.4f}")

    # Persist: the frozen combined-feature model of the selected type is P0's final config.
    import joblib

    final_model = trained_models[(best_kind, "combined")]
    joblib.dump(final_model, OUT_DIR / "p0_model.joblib")

    config = {
        "selected_model_kind": best_kind,
        "feature_set": "combined",
        "feature_names": list(ALL_FEATURE_NAMES),
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "model_selection_dev_results": selection_results,
        "ablation_dev_results_for_selected_model": ablation_results,
        "train_size": len(train_rows),
        "dev_size": len(dev_rows),
        "seed": 42,
    }
    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model + config to {OUT_DIR}")


if __name__ == "__main__":
    main()
