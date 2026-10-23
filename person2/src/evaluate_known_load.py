"""Known-load, healthy-baseline protocol; normal test is same-run temporal.

This is intentionally a different question from unseen-recording evaluation.
Fault runs are held out entirely. Healthy late windows are tested after a 20%
time gap, but are not independent acquisition sessions. Report both protocols.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = pd.read_csv(ROOT / 'data/processed/features.csv')
    columns = [c for c in data if c.startswith(('vibration_', 'current_'))]
    reports = []
    for load, group in data.groupby('load'):
        healthy = group[group.fault_type.eq('normal')].sort_values('window_id')
        n = len(healthy)
        normal_train = healthy.iloc[:int(n*.6)]
        normal_test = healthy.iloc[int(n*.8):]
        fault = group[~group.fault_type.eq('normal')]
        # Assign complete recordings by severity order within each fault.
        assignments = {}
        for _, fault_group in fault.groupby('fault_type'):
            recordings = fault_group[['recording_id', 'severity_rank']].drop_duplicates().sort_values('severity_rank')
            assignments.update({r: i % 3 for i, r in enumerate(recordings.recording_id)})
        for fold in range(3):
            held = [r for r, k in assignments.items() if k == fold]
            fault_test = fault[fault.recording_id.isin(held)]
            train = pd.concat([normal_train, fault[~fault.recording_id.isin(held)]])
            test = pd.concat([normal_test, fault_test])
            model = ExtraTreesClassifier(n_estimators=200, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1)
            model.fit(train[columns], train.fault_type.ne('normal'))
            pred = model.predict(test[columns])
            report = classification_report(test.fault_type.ne('normal'), pred, labels=[False,True], target_names=['normal','abnormal'], output_dict=True, zero_division=0)
            reports.append({'load': load, 'fold': fold, 'normal_train_windows': len(normal_train), 'normal_test_windows': len(normal_test), 'test_fault_recordings': held, 'report': report})
            print('load',load,'fold',fold,'normal',round(report['normal']['recall'],3),'fault',round(report['abnormal']['recall'],3),flush=True)
    output = {'protocol': 'known load; first60% normal baseline; 20% gap; final20% normal same-run test; fault recordings held out', 'warning': 'Normal test is not an independent recording. Healthy test windows reused across fault folds. Do not pool as independent samples.', 'features': columns, 'folds': reports, 'mean_normal_recall': float(np.mean([r['report']['normal']['recall'] for r in reports])), 'mean_fault_recall': float(np.mean([r['report']['abnormal']['recall'] for r in reports])), 'mean_macro_f1': float(np.mean([r['report']['macro avg']['f1-score'] for r in reports]))}
    (ROOT / 'reports/known_load_metrics.json').write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
