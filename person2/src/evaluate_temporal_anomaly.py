"""Evaluate anomaly detection with chronological normal-run holdouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .evaluate_two_stage import ANOMALY_FEATURES, RANDOM_STATE


def split_normal_recordings(normal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = []
    validation_parts = []
    test_parts = []
    for _, recording in normal.groupby("recording_id", sort=True):
        recording = recording.sort_values("window_id")
        first = int(len(recording) * 0.60)
        second = int(len(recording) * 0.80)
        train_parts.append(recording.iloc[:first])
        validation_parts.append(recording.iloc[first:second])
        test_parts.append(recording.iloc[second:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(validation_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def evaluate(features_path: Path, output_path: Path) -> dict:
    data = pd.read_csv(features_path)
    normal = data[data["fault_type"] == "normal"]
    abnormal = data[data["fault_type"] != "normal"]
    train_normal, validation_normal, test_normal = split_normal_recordings(normal)

    scaler = StandardScaler().fit(train_normal[ANOMALY_FEATURES])
    train_x = scaler.transform(train_normal[ANOMALY_FEATURES])
    validation_x = scaler.transform(validation_normal[ANOMALY_FEATURES])
    detector = IsolationForest(
        n_estimators=500,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(train_x)

    # Five percent of later healthy validation windows may be treated as
    # anomalous to establish an operating threshold without using fault labels.
    validation_scores = -detector.score_samples(validation_x)
    threshold = float(np.quantile(validation_scores, 0.95))

    evaluation = pd.concat([test_normal, abnormal], ignore_index=True)
    truth = (evaluation["fault_type"] != "normal").astype(int).to_numpy()
    scores = -detector.score_samples(scaler.transform(evaluation[ANOMALY_FEATURES]))
    predicted = (scores > threshold).astype(int)

    result = {
        "protocol": {
            "normal_train": "first 60% of each normal recording",
            "normal_validation": "next 20% of each normal recording",
            "normal_test": "final 20% of each normal recording",
            "abnormal_test": "all fault recordings",
            "threshold": "95th percentile of validation-normal anomaly scores",
        },
        "counts": {
            "normal_train": int(len(train_normal)),
            "normal_validation": int(len(validation_normal)),
            "normal_test": int(len(test_normal)),
            "abnormal_test": int(len(abnormal)),
        },
        "threshold": threshold,
        "report": classification_report(
            truth,
            predicted,
            labels=[0, 1],
            target_names=["normal", "abnormal"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    result = evaluate(
        project_dir / "data" / "processed" / "features.csv",
        project_dir / "models" / "temporal_anomaly_metrics.json",
    )
    report = result["report"]
    print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Normal recall: {report['normal']['recall']:.3f}")
    print(f"Abnormal recall: {report['abnormal']['recall']:.3f}")
    print(f"Confusion matrix: {result['confusion_matrix']}")


if __name__ == "__main__":
    main()
