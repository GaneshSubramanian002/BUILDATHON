"""Compare fixed-Hz, RPM-order, and fused gates on identical grouped folds."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from .audit_raw_signals import ROOT
from .evaluate_order_transfer import fit_gate


SOURCE_SPEEDS = ((600,), (900,), (600, 900))
WEIGHTS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
THRESHOLDS = tuple(np.round(np.arange(0.35, 0.81, 0.01), 2))


def prepare():
    fixed_k = pd.read_csv(ROOT / "data/processed/targeted_vibration.csv")
    order_k = pd.read_csv(ROOT / "data/processed/kaist_order_features.csv")
    fixed_e = pd.read_csv(ROOT / "data/processed/cross_speed_4ch_features.csv")
    order_e = pd.read_csv(ROOT / "data/processed/cross_speed_order_features.csv")
    order_k = order_k.rename(
        columns={c: f"order__{c}" for c in order_k if c.startswith("vibration_")}
    )
    order_e = order_e.rename(
        columns={c: f"order__{c}" for c in order_e if c.startswith("vibration_")}
    )
    kaist = fixed_k.merge(
        order_k.drop(columns=["fault_type", "load", "speed_rpm"]),
        on=["recording_id", "window_id"],
        validate="one_to_one",
    )
    external = fixed_e.merge(
        order_e.drop(columns=["condition", "speed_rpm"]),
        on=["recording_id", "window_id"],
        validate="one_to_one",
    )
    fixed = [c for c in fixed_k if c.startswith("vibration_")]
    order = [c for c in kaist if c.startswith("order__")]
    return kaist, external, {"fixed_hz": fixed, "order": order, "fused": fixed + order}


def summarize(expected, probabilities, threshold):
    predicted = probabilities >= threshold
    report = classification_report(
        expected,
        predicted,
        target_names=["normal", "abnormal"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "threshold": threshold,
        "normal_recall": report["normal"]["recall"],
        "abnormal_recall": report["abnormal"]["recall"],
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(expected, predicted).tolist(),
    }


def main():
    kaist, external, feature_sets = prepare()
    expected = kaist.fault_type.ne("normal")
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            kaist, kaist.fault_type, kaist.recording_id
        )
    )
    candidates = []
    for feature_name, columns in feature_sets.items():
        for speeds in SOURCE_SPEEDS:
            auxiliary = external[external.speed_rpm.isin(speeds)]
            auxiliary_expected = auxiliary.condition.ne("normal")
            for weight in WEIGHTS:
                probabilities = np.zeros(len(kaist))
                for train, test in folds:
                    gate = fit_gate(
                        kaist.iloc[train][columns],
                        expected.iloc[train],
                        auxiliary[columns],
                        auxiliary_expected,
                        weight,
                    )
                    probabilities[test] = gate.predict_proba(kaist.iloc[test][columns])[:, 1]
                for threshold in THRESHOLDS:
                    result = summarize(expected, probabilities, threshold)
                    result.update(
                        feature_set=feature_name,
                        auxiliary_speeds_rpm=list(speeds),
                        auxiliary_weight=weight,
                    )
                    candidates.append(result)

    eligible = [candidate for candidate in candidates if candidate["abnormal_recall"] >= 0.93]
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["macro_f1"],
            candidate["normal_recall"],
            candidate["accuracy"],
        ),
    )
    columns = feature_sets[selected["feature_set"]]
    auxiliary = external[external.speed_rpm.isin(selected["auxiliary_speeds_rpm"])]
    gate = fit_gate(
        kaist[columns],
        expected,
        auxiliary[columns],
        auxiliary.condition.ne("normal"),
        selected["auxiliary_weight"],
    )
    heldout = external[external.speed_rpm.eq(1200)]
    validation = summarize(
        heldout.condition.ne("normal"),
        gate.predict_proba(heldout[columns])[:, 1],
        selected["threshold"],
    )
    output = {
        "selection_rule": "highest grouped-CV macro F1 with abnormal recall >= 0.93",
        "selected": selected,
        "external_1200_rpm_validation": {"rows": len(heldout), **validation},
        "all_candidates": candidates,
    }
    (ROOT / "reports/feature_fusion_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in output.items() if key != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
