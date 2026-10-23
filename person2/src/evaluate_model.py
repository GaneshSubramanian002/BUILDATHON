"""Evaluate Person 1's labeled CSV without retraining the frozen model."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from .prediction import predict_many, load_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path('person2/reports/person1_evaluation.json'))
    parser.add_argument('--kind', choices=['external-synthetic', 'integration-smoke'], default='external-synthetic')
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    if data.empty:
        raise ValueError('Test CSV is empty')
    if not {'fault_type', 'severity_rank'} <= set(data.columns):
        raise ValueError('Test CSV must contain fault_type and severity_rank ground truth')
    labels = ['inner_race_fault', 'misalignment', 'normal', 'outer_race_fault', 'rotor_unbalance']
    if not set(data.fault_type) <= set(labels):
        raise ValueError('Unknown fault_type labels')
    for fault, rank in zip(data.fault_type, data.severity_rank):
        allowed = {0} if fault == 'normal' else (set(range(1, 6)) if fault == 'rotor_unbalance' else {1, 2, 3})
        if pd.isna(rank) or rank not in allowed:
            raise ValueError(f'Invalid severity rank {rank} for {fault}')
    # Strip labels before inference.
    columns = load_artifact()['feature_columns']
    if 'load' not in columns:
        columns = columns + ['load']
    pred = pd.DataFrame(predict_many(data[columns].to_dict(orient='records')))
    true_condition = ['NORMAL' if f == 'normal' else ('WARNING' if r == 1 else 'CRITICAL') for f, r in zip(data.fault_type, data.severity_rank)]
    results = {
        'evaluation_type': args.kind,
        'interpretation': 'Synthetic compatibility evidence only; integration-smoke on public rows is not accuracy evidence.',
        'rows': len(data),
        'fault_report': classification_report(data.fault_type, pred.possible_fault, labels=labels, output_dict=True, zero_division=0),
        'fault_labels': labels,
        'fault_confusion_matrix': confusion_matrix(data.fault_type, pred.possible_fault, labels=labels).tolist(),
        'binary_report': classification_report(data.fault_type.ne('normal'), pred.anomaly, output_dict=True, zero_division=0),
        'condition_report': classification_report(true_condition, pred.condition, output_dict=True, zero_division=0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    pd.concat([data.reset_index(drop=True), pred.add_prefix('predicted_')], axis=1).to_csv(args.output.with_suffix('.csv'), index=False)
    print(f'Evaluated {len(data)} rows. Results: {args.output}')


if __name__ == '__main__':
    main()

