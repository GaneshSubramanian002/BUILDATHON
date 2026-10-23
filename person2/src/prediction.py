"""Prediction contract used by the backend."""

from collections.abc import Mapping
from numbers import Real
from typing import Any
from pathlib import Path
from functools import lru_cache
import math
import joblib
import pandas as pd

from .schema import VALID_LOADS


def validate_sensor_data(sensor_data: Mapping[str, Any]) -> dict[str, float]:
    """Return validated numeric model features in the fixed training order."""
    if not isinstance(sensor_data, Mapping):
        raise ValueError('Sensor data must be a JSON object')
    forbidden = {'fault_type', 'severity', 'severity_rank', 'severity_value', 'condition', 'scenario', 'expected_condition'}
    if forbidden.intersection(sensor_data):
        raise ValueError('Remove ground-truth labels from model input')
    artifact = load_artifact()
    fields = artifact['feature_columns']
    fields = fields + [field for field in artifact.get('required_context', ['load']) if field not in fields]
    missing = [field for field in fields if field not in sensor_data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    validated: dict[str, float] = {}
    for field in fields:
        value = sensor_data[field]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field} must be numeric")
        validated[field] = float(value)
        if not math.isfinite(validated[field]):
            raise ValueError(f"{field} must be finite")

    if validated["load"] not in VALID_LOADS:
        raise ValueError("load must be one of 0.0, 2.0, or 4.0 Nm")
    if "speed_rpm" in validated and not 0 < validated["speed_rpm"] <= 1_536_000:
        raise ValueError("speed_rpm must be positive and compatible with the sample rate")

    for field in fields:
        if field.startswith(("current_", "vibration_")) and validated[field] < 0:
            raise ValueError(f"{field} cannot be negative")
        if field.startswith('temperature_') and validated[field] < -273.15:
            raise ValueError(f'{field} is below absolute zero')
        if '_band_' in field and not 0 <= validated[field] <= 1:
            raise ValueError(f'{field} must be between 0 and 1')
        if any(s in field for s in ['_energy_', '_harmonic_']) and not 0 <= validated[field] <= 1:
            raise ValueError(f'{field} must be between 0 and 1')
        if field.endswith('dominant_hz') and validated[field] > 12800:
            raise ValueError(f'{field} exceeds Nyquist frequency')
        if field.endswith('spectral_centroid') and validated[field] > 12800:
            raise ValueError(f'{field} exceeds Nyquist frequency')
        if field.endswith('dominant_order') and not 0 <= validated[field] <= 256:
            raise ValueError(f'{field} must be between 0 and 256')

    return validated


@lru_cache(maxsize=1)
def load_artifact():
    path = Path(__file__).resolve().parents[1] / 'models/final_pipeline.joblib'
    if not path.is_file():
        raise RuntimeError('Missing final model. Run python -m person2.src.finalize')
    return joblib.load(path)


def predict_many(readings: list[Mapping]) -> list[dict]:
    """Vectorized inference with strict validation and no label inputs."""
    if not readings:
        return []
    artifact = load_artifact()
    frame = pd.DataFrame([validate_sensor_data(r) for r in readings], columns=artifact['feature_columns'])
    gate = artifact['gate']
    scores = gate.predict_proba(frame)[:, list(gate.classes_).index(1)]
    anomaly = scores >= artifact.get('decision_threshold', 0.5)
    faults = ['normal'] * len(frame)
    conditions = ['NORMAL'] * len(frame)
    confidences = 1 - scores
    indices = [i for i, flag in enumerate(anomaly) if flag]
    if indices:
        fault_columns = artifact.get('fault_feature_columns', artifact['feature_columns'])
        subset = frame.iloc[indices][fault_columns]
        fp = artifact['fault_model'].predict(subset)
        cp = artifact['condition_model'].predict(subset)
        confidence = artifact['fault_model'].predict_proba(subset).max(axis=1)
        for index, fault, condition, conf in zip(indices, fp, cp, confidence):
            faults[index] = str(fault)
            conditions[index] = str(condition)
            confidences[index] = float(conf) * scores[index]
    recommendations = {
        'normal': 'Continue routine monitoring.',
        'inner_race_fault': 'Inspect the bearing inner race and lubrication.',
        'outer_race_fault': 'Inspect the bearing outer race and mounting.',
        'misalignment': 'Inspect shaft and coupling alignment.',
        'rotor_unbalance': 'Inspect rotor balance and attachments.',
    }
    return [{
        'schema_version': artifact['schema_version'],
        'anomaly': bool(anomaly[i]),
        'anomaly_score': round(float(scores[i]), 4),
        'possible_fault': faults[i],
        'confidence': round(float(confidences[i]), 4),
        'condition': conditions[i],
        'risk': {'NORMAL': 'LOW', 'WARNING': 'MEDIUM', 'CRITICAL': 'CRITICAL'}[conditions[i]],
        'health_score': {'NORMAL': 90, 'WARNING': 60, 'CRITICAL': 25}[conditions[i]],
        'health_score_basis': 'fixed prototype condition index, not measured health or failure probability',
        'recommendation': recommendations[faults[i]],
    } for i in range(len(frame))]


def predict(sensor_data: Mapping[str, Any]) -> dict[str, Any]:
    return predict_many([sensor_data])[0]

