"""Test whether real cross-speed data improves the KAIST healthy/fault gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.55


def model(class_weight="balanced"):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, class_weight=class_weight, max_iter=2000),
    )


def report(y_true, scores):
    predicted = scores >= THRESHOLD
    result = classification_report(
        y_true,
        predicted,
        target_names=["normal", "abnormal"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "threshold": THRESHOLD,
        "normal_recall": result["normal"]["recall"],
        "abnormal_recall": result["abnormal"]["recall"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
    }


def grouped_scores(train_source, test_source, feature_columns, folds, supplement=None, supplement_weight=1.0):
    scores = np.zeros(len(test_source), dtype=float)
    for train_index, test_index in folds:
        train = train_source.iloc[train_index]
        x_train = train[feature_columns]
        y_train = train["binary_label"]
        if supplement is not None:
            scaler = StandardScaler().fit(x_train)
            real_scaled = np.clip(scaler.transform(x_train), -5, 5)
            # The machines have very different absolute spectral domains.
            # Clip auxiliary standardized values so a few extreme dimensions
            # cannot redefine the KAIST decision boundary.
            external_scaled = np.clip(scaler.transform(supplement[feature_columns]), -5, 5)
            combined_x = np.vstack([real_scaled, external_scaled])
            combined_y = pd.concat([y_train, supplement["binary_label"]], ignore_index=True)
            real_counts = combined_y.iloc[:len(train)].value_counts()
            external_counts = combined_y.iloc[len(train):].value_counts()
            real_weights = combined_y.iloc[:len(train)].map(
                {label: len(train) / (2 * count) for label, count in real_counts.items()}
            ).to_numpy()
            external_weights = combined_y.iloc[len(train):].map(
                {label: supplement_weight * len(train) / (2 * count) for label, count in external_counts.items()}
            ).to_numpy()
            classifier_weights = np.concatenate([real_weights, external_weights])
            fitted = LogisticRegression(C=1, class_weight=None, max_iter=2000).fit(
                combined_x, combined_y, sample_weight=classifier_weights
            )
            test_scaled = np.clip(scaler.transform(test_source.iloc[test_index][feature_columns]), -5, 5)
            scores[test_index] = fitted.predict_proba(test_scaled)[:, 1]
        else:
            fitted = model().fit(x_train, y_train)
            scores[test_index] = fitted.predict_proba(test_source.iloc[test_index][feature_columns])[:, 1]
    return scores


def main():
    kaist = pd.read_csv(ROOT / "data/processed/targeted_vibration.csv")
    external_path = ROOT / "data/processed/cross_speed_4ch_features.csv"
    if not external_path.exists():
        raise SystemExit(
            "Missing cross_speed_4ch_features.csv. Extract the dataset and run "
            "python -m person2.src.import_cross_speed_dataset first."
        )
    external = pd.read_csv(external_path)
    features = [column for column in kaist if column.startswith("vibration_")]
    missing = sorted(set(features) - set(external))
    if missing:
        raise ValueError(f"External features do not match v2 contract: {missing}")
    kaist["binary_label"] = kaist.fault_type.ne("normal").astype(int)
    external["binary_label"] = external.condition.ne("normal").astype(int)

    kaist_folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            kaist, kaist.fault_type, kaist.recording_id
        )
    )
    baseline_scores = grouped_scores(kaist, kaist, features, kaist_folds)
    transfer_sweep = {}
    for speed in (600, 900, 1200, "all"):
        supplement = external if speed == "all" else external[external.speed_rpm.eq(speed)]
        for weight in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            combined_scores = grouped_scores(
                kaist,
                kaist,
                features,
                kaist_folds,
                supplement=supplement,
                supplement_weight=weight,
            )
            transfer_sweep[f"speed={speed};weight={weight}"] = report(kaist.binary_label, combined_scores)

    external_folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            external, external.condition, external.recording_id
        )
    )
    external_scores = grouped_scores(external, external, features, external_folds, supplement=kaist)
    results = {
        "dataset": "10.17632/y8mv28hjwb.1",
        "external_rows": len(external),
        "external_recordings": int(external.recording_id.nunique()),
        "external_counts": external.condition.value_counts().to_dict(),
        "evaluation_warning": "Development comparison; folds may not be used as an untouched final test.",
        "kaist_baseline": report(kaist.binary_label, baseline_scores),
        "kaist_transfer_sweep": transfer_sweep,
        "external_grouped_with_kaist_training": report(external.binary_label, external_scores),
    }
    output = ROOT / "reports/cross_dataset_audit.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
