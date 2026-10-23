"""Finalize v2.2: fixed-frequency plus RPM-order probability ensemble."""
from __future__ import annotations

import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from .audit_raw_signals import ROOT
from .ensemble import ProbabilityEnsembleGate
from .evaluate_feature_fusion import prepare
from .evaluate_order_transfer import fit_gate
from .finalize_v3 import fault_model


SCHEMA_VERSION = "2.2.0"
DECISION_THRESHOLD = 0.50
ORDER_SHARE = 0.25
FIXED_SPEEDS = (600,)
FIXED_WEIGHT = 0.01
ORDER_SPEEDS = (600, 900)
ORDER_WEIGHT = 0.10


def make_gate(kaist, expected, external, fixed_columns, order_columns):
    fixed_auxiliary = external[external.speed_rpm.isin(FIXED_SPEEDS)]
    order_auxiliary = external[external.speed_rpm.isin(ORDER_SPEEDS)]
    fixed_gate = fit_gate(
        kaist[fixed_columns], expected,
        fixed_auxiliary[fixed_columns], fixed_auxiliary.condition.ne("normal"), FIXED_WEIGHT,
    )
    order_gate = fit_gate(
        kaist[order_columns], expected,
        order_auxiliary[order_columns], order_auxiliary.condition.ne("normal"), ORDER_WEIGHT,
    )
    return ProbabilityEnsembleGate(
        fixed_gate, order_gate, fixed_columns, order_columns, ORDER_SHARE
    )


def main():
    data, external, feature_sets = prepare()
    severity = pd.read_csv(ROOT / "data/processed/features.csv")
    data = data.merge(
        severity[["recording_id", "window_id", "severity_rank", "severity_value"]],
        on=["recording_id", "window_id"],
        validate="one_to_one",
    )
    fixed_columns = feature_sets["fixed_hz"]
    order_columns = feature_sets["order"]
    all_columns = fixed_columns + order_columns
    expected = data.fault_type.ne("normal")
    condition = np.where(data.severity_rank.eq(1), "WARNING", "CRITICAL")
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            data, data.fault_type, data.recording_id
        )
    )

    scores = np.zeros(len(data))
    faults = np.full(len(data), "normal", dtype=object)
    conditions = np.full(len(data), "NORMAL", dtype=object)
    manifest = []
    for train, test in folds:
        gate = make_gate(
            data.iloc[train], expected.iloc[train], external, fixed_columns, order_columns
        )
        scores[test] = gate.predict_proba(data.iloc[test][all_columns])[:, 1]
        flagged_test = scores[test] >= DECISION_THRESHOLD
        abnormal_train = train[expected.iloc[train].to_numpy()]
        flagged_indices = test[flagged_test]
        fault_classifier = fault_model().fit(
            data.iloc[abnormal_train][fixed_columns], data.iloc[abnormal_train].fault_type
        )
        condition_classifier = fault_model().fit(
            data.iloc[abnormal_train][fixed_columns], condition[abnormal_train]
        )
        faults[flagged_indices] = fault_classifier.predict(data.iloc[flagged_indices][fixed_columns])
        conditions[flagged_indices] = condition_classifier.predict(data.iloc[flagged_indices][fixed_columns])
        train_groups = sorted(data.iloc[train].recording_id.unique())
        test_groups = sorted(data.iloc[test].recording_id.unique())
        assert not set(train_groups) & set(test_groups)
        manifest.append({"train_recordings": train_groups, "test_recordings": test_groups})

    flagged = scores >= DECISION_THRESHOLD
    binary_report = classification_report(
        expected, flagged, target_names=["normal", "abnormal"], output_dict=True, zero_division=0
    )
    final_gate = make_gate(data, expected, external, fixed_columns, order_columns)
    heldout = external[external.speed_rpm.eq(1200)]
    heldout_expected = heldout.condition.ne("normal")
    heldout_predicted = final_gate.predict_proba(heldout[all_columns])[:, 1] >= DECISION_THRESHOLD
    heldout_report = classification_report(
        heldout_expected, heldout_predicted,
        target_names=["normal", "abnormal"], output_dict=True, zero_division=0,
    )
    labels = sorted(data.fault_type.unique())
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_type": "development grouped CV; ensemble settings selected on these folds; 1200 RPM external validation excluded from selection",
        "decision_threshold": DECISION_THRESHOLD,
        "order_probability_share": ORDER_SHARE,
        "folds": manifest,
        "binary_report": binary_report,
        "binary_confusion_matrix": confusion_matrix(expected, flagged).tolist(),
        "end_to_end_fault_report": classification_report(data.fault_type, faults, output_dict=True, zero_division=0),
        "fault_labels": labels,
        "fault_confusion_matrix": confusion_matrix(data.fault_type, faults, labels=labels).tolist(),
        "condition_report": classification_report(
            np.where(expected, condition, "NORMAL"), conditions, output_dict=True, zero_division=0
        ),
        "external_1200_rpm_validation": {
            "rows": len(heldout),
            "normal_recall": heldout_report["normal"]["recall"],
            "abnormal_recall": heldout_report["abnormal"]["recall"],
            "accuracy": heldout_report["accuracy"],
            "macro_f1": heldout_report["macro avg"]["f1-score"],
            "confusion_matrix": confusion_matrix(heldout_expected, heldout_predicted).tolist(),
        },
        "training": {
            "fixed_gate_auxiliary_speeds_rpm": list(FIXED_SPEEDS),
            "fixed_gate_auxiliary_weight": FIXED_WEIGHT,
            "order_gate_auxiliary_speeds_rpm": list(ORDER_SPEEDS),
            "order_gate_auxiliary_weight": ORDER_WEIGHT,
        },
    }

    artifact_path = ROOT / "models/final_pipeline.joblib"
    shutil.copy2(artifact_path, ROOT / "models/final_pipeline_v2_1.joblib")
    shutil.copy2(ROOT / "models/final_metrics.json", ROOT / "models/final_metrics_v2_1.json")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "gate": final_gate,
        "fault_model": fault_model().fit(data.loc[expected, fixed_columns], data.loc[expected, "fault_type"]),
        "condition_model": fault_model().fit(data.loc[expected, fixed_columns], condition[expected]),
        "feature_columns": all_columns,
        "fault_feature_columns": fixed_columns,
        "required_context": ["load", "speed_rpm"],
        "training_rows": len(data),
        "input_basis": "four-channel fixed-Hz and RPM-normalized vibration spectra",
    }
    joblib.dump(artifact, artifact_path)
    (ROOT / "models/final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    handoff = ROOT / "handoff"
    contract = {
        "schema_version": SCHEMA_VERSION,
        "required_numeric_features": all_columns,
        "required_context": ["load", "speed_rpm"],
        "valid_loads": [0, 2, 4],
        "valid_speed_rpm": {"minimum_exclusive": 0, "maximum_inclusive": 1536000},
        "sample_rate_hz": 25600,
        "window_seconds": 1,
        "units": {
            column: ("Hz" if column.endswith("dominant_hz") else "dimensionless")
            for column in all_columns
        },
        "faults": labels,
        "optional_monitoring_fields": ["temperature_A_mean", "temperature_B_mean", "current_U_rms"],
        "feature_functions": [
            "person2.src.audit_raw_signals.targeted",
            "person2.src.order_features.order_features",
        ],
        "notes": [
            "Compute both feature families from the same four 25.6-kHz, one-second vibration channels.",
            "order_features requires measured or simulator-known speed_rpm for that window.",
            "Temperature/current are monitoring context, not trained inputs.",
            "Do not pass fault_type or severity labels into predict().",
        ],
    }
    (handoff / "input_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")

    examples = []
    for (fault, load, rank), group in data.groupby(["fault_type", "load", "severity_rank"]):
        row = {column: float(group.iloc[0][column]) for column in all_columns}
        row.update(
            load=float(load), speed_rpm=3010.0,
            equipment_id="MOTOR_01", timestamp="2026-10-23T10:30:00+05:30",
        )
        examples.append({"source_fault": fault, "source_severity_rank": int(rank), "sensor_data": row})
    (handoff / "integration_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    data.to_csv(handoff / "reference_features.csv", index=False)
    data.groupby(["fault_type", "load", "severity_rank"])[all_columns].agg(
        ["min", "max", "mean", "std"]
    ).to_csv(handoff / "simulator_ranges.csv")
    print(json.dumps({
        "binary": binary_report,
        "fault_macro_f1": metrics["end_to_end_fault_report"]["macro avg"]["f1-score"],
        "condition_macro_f1": metrics["condition_report"]["macro avg"]["f1-score"],
        "external_1200_rpm_validation": metrics["external_1200_rpm_validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
