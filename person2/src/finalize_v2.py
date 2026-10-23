"""Train spectral v2 using source vibration only; preserve v1 evaluation."""
import json
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DECISION_THRESHOLD = 0.55


def gate_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=2000))


def fault_model():
    return ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)


def main():
    data = pd.read_csv(ROOT / 'data/processed/targeted_vibration.csv')
    original = pd.read_csv(ROOT / 'data/processed/features.csv')
    data = data.merge(original[['recording_id','window_id','severity_rank','severity_value']], on=['recording_id','window_id'], validate='one_to_one')
    columns = [c for c in data if c.startswith('vibration_')]
    x, y = data[columns], data.fault_type
    binary = y.ne('normal')
    condition = np.where(data.severity_rank.eq(1), 'WARNING', 'CRITICAL')
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=42).split(x,y,data.recording_id))
    gate_scores = cross_val_predict(gate_model(), x, binary, cv=folds, method='predict_proba')[:, 1]
    flagged = gate_scores >= DECISION_THRESHOLD
    faults = np.full(len(data), 'normal', dtype=object)
    conditions = np.full(len(data), 'NORMAL', dtype=object)
    manifest = []
    for train,test in folds:
        tr = train[binary.iloc[train].to_numpy()]
        indices = test[flagged[test]]
        fm = fault_model().fit(x.iloc[tr], y.iloc[tr])
        cm = fault_model().fit(x.iloc[tr], condition[tr])
        faults[indices] = fm.predict(x.iloc[indices])
        conditions[indices] = cm.predict(x.iloc[indices])
        train_groups = sorted(data.iloc[train].recording_id.unique())
        test_groups = sorted(data.iloc[test].recording_id.unique())
        assert not set(train_groups) & set(test_groups)
        manifest.append({'train_recordings':train_groups,'test_recordings':test_groups})
    report = classification_report(binary, flagged, target_names=['normal','abnormal'], output_dict=True, zero_division=0)
    labels = sorted(y.unique())
    metrics = {'schema_version':'2.0.0','evaluation_type':'development grouped CV; feature/model/threshold selection reused folds; not untouched test', 'decision_threshold':DECISION_THRESHOLD, 'folds':manifest, 'binary_report':report, 'binary_confusion_matrix':confusion_matrix(binary,flagged).tolist(), 'end_to_end_fault_report':classification_report(y,faults,output_dict=True,zero_division=0), 'fault_labels':labels, 'fault_confusion_matrix':confusion_matrix(y,faults,labels=labels).tolist(), 'condition_report':classification_report(np.where(binary,condition,'NORMAL'),conditions,output_dict=True,zero_division=0)}
    artifact_path = ROOT / 'models/final_pipeline.joblib'
    if not (ROOT / 'models/final_pipeline_v1.joblib').exists():
        shutil.copy2(artifact_path,ROOT / 'models/final_pipeline_v1.joblib')
        shutil.copy2(ROOT / 'models/final_metrics.json', ROOT / 'models/final_metrics_v1.json')
    artifact = {'schema_version':'2.0.0', 'decision_threshold':DECISION_THRESHOLD, 'gate':gate_model().fit(x,binary), 'fault_model':fault_model().fit(x.loc[binary],y.loc[binary]), 'condition_model':fault_model().fit(x.loc[binary],condition[binary]), 'feature_columns':columns, 'training_rows':len(data), 'input_basis':'vibration spectrum only; no cross-device fusion'}
    joblib.dump(artifact,artifact_path)
    (ROOT / 'models/final_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    handoff = ROOT / 'handoff'
    contract = {'schema_version':'2.0.0','required_numeric_features':columns,'required_context':['load'],'valid_loads':[0,2,4],'sample_rate_hz':25600,'window_seconds':1,'units':{c:('Hz' if c.endswith('dominant_hz') else 'dimensionless') for c in columns},'faults':labels,'optional_monitoring_fields':['temperature_A_mean','temperature_B_mean','current_U_rms'],'feature_function':'person2.src.audit_raw_signals.targeted','notes':['Detector uses normalized vibration spectra only. Temperature/current are monitoring context, not trained inputs.','load is validated context, not a model feature.','Do not pass fault_type/severity labels into predict().','Health and condition severity remain prototype assumptions.']}
    (handoff / 'input_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    examples = []
    for (fault, load, rank), group in data.groupby(['fault_type','load','severity_rank']):
        row = {c:float(group.iloc[0][c]) for c in columns}
        row.update(load=float(load),equipment_id='MOTOR_01',timestamp='2026-10-23T10:30:00+05:30')
        examples.append({'source_fault':fault,'source_severity_rank':int(rank),'sensor_data':row})
    (handoff / 'integration_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    data.to_csv(handoff / 'reference_features.csv',index=False)
    data.groupby(['fault_type','load','severity_rank'])[columns].agg(['min','max','mean','std']).to_csv(handoff / 'simulator_ranges.csv')
    print(json.dumps({'binary':report,'fault_macro_f1':metrics['end_to_end_fault_report']['macro avg']['f1-score'],'condition_macro_f1':metrics['condition_report']['macro avg']['f1-score']},indent=2))


if __name__ == '__main__':
    main()
