"""Test an ensemble of the proven fixed-Hz gate and the RPM-order gate."""
from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .audit_raw_signals import ROOT
from .evaluate_feature_fusion import prepare, summarize
from .evaluate_order_transfer import fit_gate


def main():
    kaist, external, feature_sets = prepare()
    expected = kaist.fault_type.ne("normal")
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=42).split(
            kaist, kaist.fault_type, kaist.recording_id
        )
    )
    specs = {
        "fixed": (feature_sets["fixed_hz"], (600,), 0.01),
        "order": (feature_sets["order"], (600, 900), 0.1),
    }
    probabilities = {name: np.zeros(len(kaist)) for name in specs}
    for train, test in folds:
        for name, (columns, speeds, weight) in specs.items():
            auxiliary = external[external.speed_rpm.isin(speeds)]
            gate = fit_gate(
                kaist.iloc[train][columns],
                expected.iloc[train],
                auxiliary[columns],
                auxiliary.condition.ne("normal"),
                weight,
            )
            probabilities[name][test] = gate.predict_proba(kaist.iloc[test][columns])[:, 1]

    candidates = []
    for order_share in np.round(np.arange(0, 0.51, 0.05), 2):
        combined = (1 - order_share) * probabilities["fixed"] + order_share * probabilities["order"]
        for threshold in np.round(np.arange(0.35, 0.81, 0.01), 2):
            result = summarize(expected, combined, float(threshold))
            result["order_probability_share"] = float(order_share)
            candidates.append(result)
    # v2.1 grouped-CV recalls.  A replacement must improve both classes, so a
    # higher aggregate score cannot hide a regression in healthy detection.
    baseline_normal_recall = 418 / 540
    baseline_abnormal_recall = 3751 / 3960
    eligible = [
        candidate
        for candidate in candidates
        if candidate["normal_recall"] > baseline_normal_recall
        and candidate["abnormal_recall"] > baseline_abnormal_recall
    ]
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["macro_f1"],
            candidate["normal_recall"],
            candidate["accuracy"],
        ),
    )

    heldout = external[external.speed_rpm.eq(1200)]
    heldout_probability = np.zeros(len(heldout))
    for name, (columns, speeds, weight) in specs.items():
        auxiliary = external[external.speed_rpm.isin(speeds)]
        gate = fit_gate(
            kaist[columns],
            expected,
            auxiliary[columns],
            auxiliary.condition.ne("normal"),
            weight,
        )
        share = selected["order_probability_share"] if name == "order" else 1 - selected["order_probability_share"]
        heldout_probability += share * gate.predict_proba(heldout[columns])[:, 1]
    validation = summarize(
        heldout.condition.ne("normal"), heldout_probability, selected["threshold"]
    )
    output = {
        "selection_rule": "highest grouped-CV macro F1 among candidates that strictly improve both v2.1 class recalls",
        "v2_1_baseline": {
            "normal_recall": baseline_normal_recall,
            "abnormal_recall": baseline_abnormal_recall,
        },
        "selected": selected,
        "external_1200_rpm_validation": {"rows": len(heldout), **validation},
        "all_candidates": candidates,
    }
    (ROOT / "reports/gate_ensemble_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in output.items() if key != "all_candidates"}, indent=2))


if __name__ == "__main__":
    main()
