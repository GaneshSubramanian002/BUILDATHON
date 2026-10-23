"""Evaluate an anomaly-first, fault-second model without recording leakage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .schema import FEATURE_COLUMNS


RANDOM_STATE = 42
ANOMALY_FEATURES = [name for name in FEATURE_COLUMNS if name != "load"]


def evaluate(features_path: Path, output_path: Path) -> dict:
    data = pd.read_csv(features_path)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    binary_truth = (data["fault_type"] != "normal").astype(int).to_numpy()
    binary_prediction = np.zeros(len(data), dtype=int)
    fault_prediction = np.full(len(data), "normal", dtype=object)
    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data, data["fault_type"], data["recording_id"]), start=1
    ):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        normal_train = train[train["fault_type"] == "normal"]
        abnormal_train = train[train["fault_type"] != "normal"]

        scaler = StandardScaler().fit(normal_train[ANOMALY_FEATURES])
        normal_x = scaler.transform(normal_train[ANOMALY_FEATURES])
        test_x = scaler.transform(test[ANOMALY_FEATURES])

        detector = IsolationForest(
            n_estimators=400,
            contamination=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ).fit(normal_x)
        # Higher values mean more anomalous. Calibrate from training-normal data.
        train_scores = -detector.score_samples(normal_x)
        threshold = float(np.quantile(train_scores, 0.95))
        test_scores = -detector.score_samples(test_x)
        fold_binary = (test_scores > threshold).astype(int)
        binary_prediction[test_idx] = fold_binary

        fault_model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        ).fit(abnormal_train[FEATURE_COLUMNS], abnormal_train["fault_type"])
        abnormal_positions = np.flatnonzero(fold_binary == 1)
        if len(abnormal_positions):
            predicted_faults = fault_model.predict(
                test.iloc[abnormal_positions][FEATURE_COLUMNS]
            )
            fault_prediction[test_idx[abnormal_positions]] = predicted_faults

        fold_details.append(
            {
                "fold": fold,
                "train_recordings": int(train["recording_id"].nunique()),
                "test_recordings": int(test["recording_id"].nunique()),
                "normal_train_recordings": int(normal_train["recording_id"].nunique()),
                "threshold": threshold,
            }
        )

    binary_labels = [0, 1]
    multiclass_labels = sorted(data["fault_type"].unique())
    result = {
        "evaluation": "3-fold stratified group cross-validation by recording_id",
        "anomaly_features": ANOMALY_FEATURES,
        "folds": fold_details,
        "binary_labels": ["normal", "abnormal"],
        "binary_report": classification_report(
            binary_truth,
            binary_prediction,
            labels=binary_labels,
            target_names=["normal", "abnormal"],
            output_dict=True,
            zero_division=0,
        ),
        "binary_confusion_matrix": confusion_matrix(
            binary_truth, binary_prediction, labels=binary_labels
        ).tolist(),
        "multiclass_labels": multiclass_labels,
        "multiclass_report": classification_report(
            data["fault_type"],
            fault_prediction,
            labels=multiclass_labels,
            output_dict=True,
            zero_division=0,
        ),
        "multiclass_confusion_matrix": confusion_matrix(
            data["fault_type"], fault_prediction, labels=multiclass_labels
        ).tolist(),
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    result = evaluate(
        project_dir / "data" / "processed" / "features.csv",
        project_dir / "models" / "two_stage_metrics.json",
    )
    binary = result["binary_report"]
    multiclass = result["multiclass_report"]
    print(f"Anomaly macro F1: {binary['macro avg']['f1-score']:.3f}")
    print(f"Normal recall: {binary['normal']['recall']:.3f}")
    print(f"Abnormal recall: {binary['abnormal']['recall']:.3f}")
    print(f"End-to-end macro F1: {multiclass['macro avg']['f1-score']:.3f}")


if __name__ == "__main__":
    main()
