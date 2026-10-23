"""Train and cross-validate the baseline fault classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from .schema import FEATURE_COLUMNS


RANDOM_STATE = 42


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )


def train(features_path: Path, model_path: Path, metrics_path: Path) -> dict:
    data = pd.read_csv(features_path)
    required = set(FEATURE_COLUMNS) | {"fault_type", "recording_id"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing processed columns: {missing}")

    x = data[FEATURE_COLUMNS]
    y = data["fault_type"]
    groups = data["recording_id"]

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    predicted = cross_val_predict(
        build_model(),
        x,
        y,
        groups=groups,
        cv=splitter,
        method="predict",
        n_jobs=1,
    )

    labels = sorted(y.unique())
    report = classification_report(
        y,
        predicted,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y, predicted, labels=labels).tolist()
    metrics = {
        "evaluation": "3-fold stratified group cross-validation by recording_id",
        "rows": int(len(data)),
        "recordings": int(groups.nunique()),
        "features": FEATURE_COLUMNS,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
    }

    model = build_model()
    model.fit(x, y)
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": list(model.classes_),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=project_dir / "data" / "processed" / "features.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_dir / "models" / "fault_classifier.joblib",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=project_dir / "models" / "baseline_metrics.json",
    )
    args = parser.parse_args()
    metrics = train(args.features, args.model, args.metrics)
    accuracy = metrics["classification_report"]["accuracy"]
    macro_f1 = metrics["classification_report"]["macro avg"]["f1-score"]
    print(f"Grouped CV accuracy: {accuracy:.3f}")
    print(f"Grouped CV macro F1: {macro_f1:.3f}")
    print(f"Saved model to {args.model}")
    print(f"Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()

