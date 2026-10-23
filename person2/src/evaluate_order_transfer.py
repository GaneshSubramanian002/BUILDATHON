"""Evaluate RPM-normalized features before changing the deployed artifact.

Selection uses grouped KAIST development folds. Cross-speed 600/900-RPM rows
are auxiliary training data; 1200 RPM remains untouched until one candidate is
selected and is then reported once as an external validation result.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .audit_raw_signals import ROOT


WEIGHTS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)
THRESHOLDS = tuple(np.round(np.arange(0.30, 0.76, 0.01), 2))


def fit_gate(real_x, real_y, auxiliary_x, auxiliary_y, auxiliary_weight):
    scaler = StandardScaler().fit(real_x)
    x_parts = [np.clip(scaler.transform(real_x), -5, 5)]
    y_parts = [pd.Series(real_y).reset_index(drop=True)]
    real_counts = pd.Series(real_y).value_counts()
    sample_weights = [
        pd.Series(real_y)
        .map({label: len(real_y) / (2 * count) for label, count in real_counts.items()})
        .to_numpy()
    ]
    if auxiliary_weight:
        x_parts.append(np.clip(scaler.transform(auxiliary_x), -5, 5))
        y_parts.append(pd.Series(auxiliary_y).reset_index(drop=True))
        auxiliary_counts = pd.Series(auxiliary_y).value_counts()
        sample_weights.append(
            pd.Series(auxiliary_y)
            .map(
                {
                    label: auxiliary_weight * len(real_y) / (2 * count)
                    for label, count in auxiliary_counts.items()
                }
            )
            .to_numpy()
        )
    classifier = LogisticRegression(C=1, max_iter=2000).fit(
        np.vstack(x_parts),
        pd.concat(y_parts, ignore_index=True),
        sample_weight=np.concatenate(sample_weights),
    )
    return Pipeline(
        [
            ("standardscaler", scaler),
            ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -5, "a_max": 5})),
            ("logisticregression", classifier),
        ]
    )


def score(expected, probability, threshold):
    predicted = probability >= threshold
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
    kaist = pd.read_csv(ROOT / "data/processed/kaist_order_features.csv")
    external = pd.read_csv(ROOT / "data/processed/cross_speed_order_features.csv")
    columns = [column for column in kaist if column.startswith("vibration_")]
    expected = kaist.fault_type.ne("normal")
    auxiliary = external[external.speed_rpm.isin([600, 900])]
    auxiliary_expected = auxiliary.condition.ne("normal")
    heldout = external[external.speed_rpm.eq(1200)]
    heldout_expected = heldout.condition.ne("normal")
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            kaist[columns], kaist.fault_type, kaist.recording_id
        )
    )

    candidates = []
    probabilities_by_weight = {}
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
        probabilities_by_weight[weight] = probabilities
        for threshold in THRESHOLDS:
            result = score(expected, probabilities, threshold)
            result["auxiliary_weight"] = weight
            candidates.append(result)

    # Maintain at least 90% fault recall, then maximize the weaker class recall;
    # macro F1 and accuracy resolve ties.
    eligible = [candidate for candidate in candidates if candidate["abnormal_recall"] >= 0.90]
    selected = max(
        eligible,
        key=lambda candidate: (
            min(candidate["normal_recall"], candidate["abnormal_recall"]),
            candidate["macro_f1"],
            candidate["accuracy"],
        ),
    )
    gate = fit_gate(
        kaist[columns], expected, auxiliary[columns], auxiliary_expected, selected["auxiliary_weight"]
    )
    heldout_probability = gate.predict_proba(heldout[columns])[:, 1]
    heldout_result = score(heldout_expected, heldout_probability, selected["threshold"])
    report = {
        "evaluation_type": "grouped KAIST development CV; 600/900 RPM auxiliary training; 1200 RPM one-time external validation",
        "feature_type": "RPM-normalized vibration order spectrum",
        "feature_count": len(columns),
        "auxiliary_training_rows": len(auxiliary),
        "selected": selected,
        "external_1200_rpm_validation": {"rows": len(heldout), **heldout_result},
        "all_candidates": candidates,
    }
    path = ROOT / "reports/order_transfer_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
