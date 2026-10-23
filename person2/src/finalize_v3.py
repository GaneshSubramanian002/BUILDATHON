"""Finalize v2.1 with low-weight real cross-machine transfer data."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "2.1.0"
DECISION_THRESHOLD = 0.55
AUXILIARY_SPEED_RPM = 600
AUXILIARY_WEIGHT = 0.01


def fault_model():
    return ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def fit_transfer_gate(real_x, real_y, auxiliary_x, auxiliary_y):
    scaler = StandardScaler().fit(real_x)
    real_scaled = np.clip(scaler.transform(real_x), -5, 5)
    auxiliary_scaled = np.clip(scaler.transform(auxiliary_x), -5, 5)
    combined_x = np.vstack([real_scaled, auxiliary_scaled])
    combined_y = pd.concat(
        [pd.Series(real_y).reset_index(drop=True), pd.Series(auxiliary_y).reset_index(drop=True)],
        ignore_index=True,
    )

    real_counts = pd.Series(real_y).value_counts()
    auxiliary_counts = pd.Series(auxiliary_y).value_counts()
    real_weights = pd.Series(real_y).map(
        {label: len(real_y) / (2 * count) for label, count in real_counts.items()}
    ).to_numpy()
    auxiliary_weights = pd.Series(auxiliary_y).map(
        {
            label: AUXILIARY_WEIGHT * len(real_y) / (2 * count)
            for label, count in auxiliary_counts.items()
        }
    ).to_numpy()
    classifier = LogisticRegression(C=1, max_iter=2000).fit(
        combined_x,
        combined_y,
        sample_weight=np.concatenate([real_weights, auxiliary_weights]),
    )
    return Pipeline(
        [
            ("standardscaler", scaler),
            ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -5, "a_max": 5})),
            ("logisticregression", classifier),
        ]
    )


def main():
    data = pd.read_csv(ROOT / "data/processed/targeted_vibration.csv")
    original = pd.read_csv(ROOT / "data/processed/features.csv")
    data = data.merge(
        original[["recording_id", "window_id", "severity_rank", "severity_value"]],
        on=["recording_id", "window_id"],
        validate="one_to_one",
    )
    external = pd.read_csv(ROOT / "data/processed/cross_speed_4ch_features.csv")
    auxiliary = external[external.speed_rpm.eq(AUXILIARY_SPEED_RPM)].copy()
    columns = [column for column in data if column.startswith("vibration_")]
    x = data[columns]
    y = data.fault_type
    binary = y.ne("normal")
    auxiliary_binary = auxiliary.condition.ne("normal")
    condition = np.where(data.severity_rank.eq(1), "WARNING", "CRITICAL")
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(x, y, data.recording_id)
    )

    scores = np.zeros(len(data), dtype=float)
    faults = np.full(len(data), "normal", dtype=object)
    conditions = np.full(len(data), "NORMAL", dtype=object)
    manifest = []
    for train, test in folds:
        gate = fit_transfer_gate(
            x.iloc[train],
            binary.iloc[train],
            auxiliary[columns],
            auxiliary_binary,
        )
        scores[test] = gate.predict_proba(x.iloc[test])[:, 1]
        flagged = scores[test] >= DECISION_THRESHOLD
        abnormal_train = train[binary.iloc[train].to_numpy()]
        flagged_indices = test[flagged]
        fm = fault_model().fit(x.iloc[abnormal_train], y.iloc[abnormal_train])
        cm = fault_model().fit(x.iloc[abnormal_train], condition[abnormal_train])
        faults[flagged_indices] = fm.predict(x.iloc[flagged_indices])
        conditions[flagged_indices] = cm.predict(x.iloc[flagged_indices])
        train_groups = sorted(data.iloc[train].recording_id.unique())
        test_groups = sorted(data.iloc[test].recording_id.unique())
        assert not set(train_groups) & set(test_groups)
        manifest.append({"train_recordings": train_groups, "test_recordings": test_groups})

    flagged = scores >= DECISION_THRESHOLD
    binary_report = classification_report(
        binary,
        flagged,
        target_names=["normal", "abnormal"],
        output_dict=True,
        zero_division=0,
    )
    labels = sorted(y.unique())
    final_gate = fit_transfer_gate(x, binary, auxiliary[columns], auxiliary_binary)
    external_holdout = {}
    for speed in (900, 1200):
        subset = external[external.speed_rpm.eq(speed)]
        expected = subset.condition.ne("normal")
        predicted = final_gate.predict_proba(subset[columns])[:, 1] >= DECISION_THRESHOLD
        result = classification_report(
            expected,
            predicted,
            target_names=["normal", "abnormal"],
            output_dict=True,
            zero_division=0,
        )
        external_holdout[str(speed)] = {
            "rows": len(subset),
            "normal_recall": result["normal"]["recall"],
            "abnormal_recall": result["abnormal"]["recall"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro avg"]["f1-score"],
            "confusion_matrix": confusion_matrix(expected, predicted).tolist(),
        }

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_type": "development grouped CV; transfer speed/weight selected on these folds; not untouched final test",
        "decision_threshold": DECISION_THRESHOLD,
        "auxiliary_dataset": "10.17632/y8mv28hjwb.1",
        "auxiliary_speed_rpm": AUXILIARY_SPEED_RPM,
        "auxiliary_rows": len(auxiliary),
        "auxiliary_weight": AUXILIARY_WEIGHT,
        "folds": manifest,
        "binary_report": binary_report,
        "binary_confusion_matrix": confusion_matrix(binary, flagged).tolist(),
        "end_to_end_fault_report": classification_report(y, faults, output_dict=True, zero_division=0),
        "fault_labels": labels,
        "fault_confusion_matrix": confusion_matrix(y, faults, labels=labels).tolist(),
        "condition_report": classification_report(
            np.where(binary, condition, "NORMAL"), conditions, output_dict=True, zero_division=0
        ),
        "external_speed_holdout": external_holdout,
    }

    artifact_path = ROOT / "models/final_pipeline.joblib"
    if not (ROOT / "models/final_pipeline_v2.joblib").exists():
        shutil.copy2(artifact_path, ROOT / "models/final_pipeline_v2.joblib")
        shutil.copy2(ROOT / "models/final_metrics.json", ROOT / "models/final_metrics_v2.json")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "gate": final_gate,
        "fault_model": fault_model().fit(x.loc[binary], y.loc[binary]),
        "condition_model": fault_model().fit(x.loc[binary], condition[binary]),
        "feature_columns": columns,
        "training_rows": len(data),
        "auxiliary_training_rows": len(auxiliary),
        "auxiliary_weight": AUXILIARY_WEIGHT,
        "input_basis": "four-channel normalized vibration spectrum; low-weight real cross-machine transfer",
    }
    joblib.dump(artifact, artifact_path)
    (ROOT / "models/final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    handoff = ROOT / "handoff"
    contract = json.loads((handoff / "input_contract.json").read_text(encoding="utf-8"))
    contract["schema_version"] = SCHEMA_VERSION
    contract["notes"] = [
        "Detector uses normalized four-channel vibration spectra only.",
        "The binary gate includes low-weight real 600-RPM auxiliary training data; fault type and condition models use KAIST only.",
        "Temperature/current are monitoring context, not trained inputs.",
        "load is validated context, not a model feature.",
        "Do not pass fault_type/severity labels into predict().",
        "Health and condition severity remain prototype assumptions.",
    ]
    (handoff / "input_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "binary": binary_report,
                "fault_macro_f1": metrics["end_to_end_fault_report"]["macro avg"]["f1-score"],
                "condition_macro_f1": metrics["condition_report"]["macro avg"]["f1-score"],
                "external_speed_holdout": external_holdout,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
