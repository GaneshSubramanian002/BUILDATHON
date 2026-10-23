"""Compare fixed feature families using identical recording-grouped folds.

These are development CV results: previous results have already informed
feature design. They are not an untouched final test or industrial validation.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from .schema import FEATURE_COLUMNS


def main():
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data/processed/features.csv")
    derived = [c for c in data if c.startswith(("vibration_", "current_", "temperature_"))]
    families = {
        "basic": FEATURE_COLUMNS,
        "expanded": derived + ["load"],
        "signal_shape": [c for c in derived if not c.startswith("temperature_")] + ["load"],
    }
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(data, data.fault_type, data.recording_id))
    for train, test in folds:
        assert not set(data.iloc[train].recording_id) & set(data.iloc[test].recording_id)
    results = {}
    for name, columns in families.items():
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        pred = cross_val_predict(model, data[columns], data.fault_type, cv=folds)
        report = classification_report(data.fault_type, pred, output_dict=True, zero_division=0)
        labels = sorted(data.fault_type.unique())
        results[name] = {"features": columns, "report": report, "labels": labels, "confusion_matrix": confusion_matrix(data.fault_type, pred, labels=labels).tolist()}
        print(name, 'macro F1', round(report['macro avg']['f1-score'], 4), 'normal recall', round(report['normal']['recall'], 4), flush=True)
    best = max(results, key=lambda k: results[k]['report']['macro avg']['f1-score'])
    # Keep all three sensor modalities when the score difference is negligible.
    if results['expanded']['report']['macro avg']['f1-score'] >= results[best]['report']['macro avg']['f1-score'] - 0.01:
        best = 'expanded'
    columns = results[best]['features']
    model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(data[columns], data.fault_type)
    # Condition labels are an explicit demo policy, not labels supplied by the
    # original experiment. Rank one means the smallest defect per fault type.
    condition = np.where(data.fault_type == 'normal', 'NORMAL', np.where(data.severity_rank == 1, 'WARNING', 'CRITICAL'))
    condition_model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    condition_pred = cross_val_predict(condition_model, data[columns], condition, cv=folds)
    results['condition_policy_report'] = classification_report(condition, condition_pred, output_dict=True, zero_division=0)
    condition_model.fit(data[columns], condition)
    joblib.dump({"model": model, "condition_model": condition_model, "feature_columns": columns, "classes": list(model.classes_), "evaluation": "development grouped CV"}, root / 'models/improved_classifier.joblib')
    results['selected_family'] = best
    (root / 'models/improved_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    # Joint rows are provided for simulator realism; held-out simulator testing
    # is a distribution compatibility check, not independent real-world proof.
    data.groupby(['fault_type', 'load', 'severity_rank'])[columns].agg(['min','max','mean','std']).to_csv(root / 'data/processed/simulator_ranges.csv')
    data.to_json(root / 'data/processed/simulator_reference.json', orient='records')
    examples = []
    for fault, frame in data.groupby('fault_type'):
        sample = {c: float(frame.iloc[0][c]) for c in columns}
        sample.update(equipment_id='MOTOR_01', timestamp='2026-10-23T10:30:00+05:30')
        examples.append({'source_fault': fault, 'sensor_data': sample})
    (root / 'data/test/model_input_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    (root / 'models/input_contract.json').write_text(json.dumps({'required_numeric_features': columns, 'window_seconds': 1, 'load_values': [0, 2, 4], 'note': 'Examples are public training rows for integration only, not independent test data.'}, indent=2), encoding='utf-8')
    print('Selected', best, flush=True)


if __name__ == '__main__':
    main()
