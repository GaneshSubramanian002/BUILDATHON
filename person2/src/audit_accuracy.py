"""Controlled feature ablation on fixed recording-held-out development folds."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targeted', action='store_true')
    args = parser.parse_args()
    data = pd.read_csv(ROOT / 'data/processed/features.csv')
    sets = {
        'vibration': [c for c in data if c.startswith('vibration_')],
        'vibration_current': [c for c in data if c.startswith(('vibration_', 'current_'))],
        'all_sensors': [c for c in data if c.startswith(('vibration_', 'current_', 'temperature_'))],
        'vibration_shape': [c for c in data if c.startswith('vibration_') and any(s in c for s in ['kurtosis', 'crest', 'band_', 'spectral_centroid'])],
    }
    if args.targeted:
        extra = pd.read_csv(ROOT / 'data/processed/targeted_vibration.csv')
        spectral = [c for c in extra if c.startswith('vibration_')]
        data = data.merge(extra[['recording_id','window_id'] + spectral], on=['recording_id','window_id'], validate='one_to_one')
        sets = {'targeted_spectrum': spectral, 'targeted_with_rms': spectral + [c for c in data if c.endswith('_rms')] + ['load']}
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(data, data.fault_type, data.recording_id))
    y = data.fault_type.ne('normal').astype(int)
    scores = []
    for name, columns in sets.items():
        for algorithm in ['extra_trees', 'svm', 'knn', 'logistic']:
            model = {
                'extra_trees': ExtraTreesClassifier(n_estimators=150, class_weight='balanced', min_samples_leaf=2, random_state=42, n_jobs=-1),
                'svm': make_pipeline(StandardScaler(), SVC(C=1, class_weight='balanced')),
                'knn': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights='distance')),
                'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=2000)),
            }[algorithm]
            pred = cross_val_predict(model, data[columns], y, cv=folds)
            report = classification_report(y, pred, output_dict=True, zero_division=0)
            entry = {'family': name, 'algorithm': algorithm, 'columns': columns, 'report': report}
            scores.append(entry)
            print(name, algorithm, 'normal', round(report['0']['recall'],3), 'fault', round(report['1']['recall'],3), 'macro_f1', round(report['macro avg']['f1-score'],3), flush=True)
    output_name = 'targeted_accuracy_audit.json' if args.targeted else 'accuracy_audit.json'
    (ROOT / 'reports' / output_name).write_text(json.dumps(scores, indent=2), encoding='utf-8')
    data.groupby(['recording_id','fault_type','load']).mean(numeric_only=True).to_csv(ROOT / 'reports/recording_means.csv')


if __name__ == '__main__':
    main()
