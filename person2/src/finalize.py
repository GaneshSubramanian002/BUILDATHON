"""Reproducible development comparison and frozen Person 2 handoff exports."""
from pathlib import Path
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'


def candidates():
    return {
        'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1),
        'extra_trees': ExtraTreesClassifier(n_estimators=300, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1),
        'scaled_svm': make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    }


def main():
    data = pd.read_csv(ROOT / 'data/processed/features.csv')
    columns = [c for c in data if c.startswith(('temperature_', 'current_', 'vibration_'))] + ['load']
    x = data[columns]
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite source features')
    y = data.fault_type
    binary = y.ne('normal').astype(int)
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(x, y, data.recording_id))
    fold_manifest = []
    for train, test in folds:
        train_groups = sorted(data.iloc[train].recording_id.unique())
        test_groups = sorted(data.iloc[test].recording_id.unique())
        assert not set(train_groups) & set(test_groups)
        fold_manifest.append({'train_recordings': train_groups, 'test_recordings': test_groups})
    results = {}
    binary_preds = {}
    for name, model in candidates().items():
        pred = cross_val_predict(model, x, binary, cv=folds)
        binary_preds[name] = pred
        report = classification_report(binary, pred, output_dict=True, zero_division=0)
        results[name] = report
        print(name, 'binary macro F1', round(report['macro avg']['f1-score'], 4), 'normal recall', round(report['0']['recall'], 4), flush=True)
    selected = max(results, key=lambda n: results[n]['macro avg']['f1-score'])
    gate = candidates()[selected]
    # Fault-only model: grouped OOF predictions use the same folds as the gate.
    oof_fault = np.full(len(data), 'normal', dtype=object)
    oof_condition = np.full(len(data), 'NORMAL', dtype=object)
    severity_policy = np.where(data.severity_rank.eq(1), 'WARNING', 'CRITICAL')
    abnormal = binary.eq(1)
    for train, test in folds:
        fault_train = train[abnormal.iloc[train].to_numpy()]
        fault_model = candidates()['extra_trees']
        fault_model.fit(x.iloc[fault_train], y.iloc[fault_train])
        condition_model = candidates()['extra_trees']
        condition_model.fit(x.iloc[fault_train], severity_policy[fault_train])
        flagged = test[binary_preds[selected][test] == 1]
        if len(flagged):
            oof_fault[flagged] = fault_model.predict(x.iloc[flagged])
            oof_condition[flagged] = condition_model.predict(x.iloc[flagged])
    labels = sorted(y.unique())
    metrics = {
        'schema_version': VERSION,
        'evaluation_type': 'development cross-validation; reused for model selection, not untouched test',
        'folds': fold_manifest,
        'binary_comparison': results,
        'selected_gate': selected,
        'end_to_end_fault_report': classification_report(y, oof_fault, output_dict=True, zero_division=0),
        'fault_labels': labels,
        'fault_confusion_matrix': confusion_matrix(y, oof_fault, labels=labels).tolist(),
        'condition_report': classification_report(np.where(abnormal, severity_policy, 'NORMAL'), oof_condition, output_dict=True, zero_division=0),
        'condition_labels': ['NORMAL', 'WARNING', 'CRITICAL'],
        'condition_confusion_matrix': confusion_matrix(np.where(abnormal, severity_policy, 'NORMAL'), oof_condition, labels=['NORMAL', 'WARNING', 'CRITICAL']).tolist(),
    }
    gate.fit(x, binary)
    fault_model = candidates()['extra_trees'].fit(x.loc[abnormal], y.loc[abnormal])
    condition_model = candidates()['extra_trees'].fit(x.loc[abnormal], severity_policy[abnormal])
    artifact = {'schema_version': VERSION, 'gate': gate, 'fault_model': fault_model, 'condition_model': condition_model, 'feature_columns': columns, 'python_version': platform.python_version(), 'sklearn_version': sklearn.__version__, 'training_rows': len(data)}
    joblib.dump(artifact, ROOT / 'models/final_pipeline.joblib')
    (ROOT / 'models/final_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    handoff = ROOT / 'handoff'
    handoff.mkdir(exist_ok=True)
    def unit(c):
        if c == 'load': return 'Nm'
        if c.endswith(('kurtosis', 'crest')) or '_band_' in c: return 'dimensionless'
        if c.endswith('spectral_centroid'): return 'Hz'
        if c.startswith('temperature_'): return 'degC'
        if c.startswith('current_'): return 'A'
        return 'source vibration numeric scale; physical conversion requires verification'
    contract = {'schema_version': VERSION, 'required_numeric_features': columns, 'units': {c: unit(c) for c in columns}, 'window_seconds': 1, 'sample_rate_hz': 25600, 'valid_loads': [0, 2, 4], 'faults': labels, 'metadata': ['equipment_id', 'timestamp'], 'excluded_labels': ['fault_type', 'severity_rank', 'severity_value', 'condition'], 'condition_policy': 'normal=NORMAL; smallest defect rank=WARNING; all larger defect ranks=CRITICAL', 'health_policy': {'NORMAL': 90, 'WARNING': 60, 'CRITICAL': 25}, 'warning': 'Health is a fixed display index. Confidence is uncalibrated. No RUL or failure probability.'}
    (handoff / 'input_contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    examples = []
    for (fault, load, rank), group in data.groupby(['fault_type', 'load', 'severity_rank']):
        row = {c: float(group.iloc[0][c]) for c in columns}
        row.update(equipment_id='MOTOR_01', timestamp='2026-10-23T10:30:00+05:30')
        examples.append({'source_fault': fault, 'source_severity_rank': int(rank), 'sensor_data': row})
    (handoff / 'integration_examples.json').write_text(json.dumps(examples, indent=2), encoding='utf-8')
    data[['recording_id','window_id'] + columns + ['fault_type','severity_rank','severity_value']].to_csv(handoff / 'reference_features.csv', index=False)
    stats = data.groupby(['fault_type','load','severity_rank'])[columns].agg(['min','max','mean','std'])
    stats.to_csv(handoff / 'simulator_ranges.csv')
    (handoff / 'environment.json').write_text(json.dumps({'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__}, indent=2), encoding='utf-8')
    print('Frozen gate:', selected, 'features:', len(columns), flush=True)


if __name__ == '__main__':
    main()
