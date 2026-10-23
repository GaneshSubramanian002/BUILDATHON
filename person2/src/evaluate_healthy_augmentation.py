"""Generate realistic healthy vibration augmentations and evaluate them safely.

Synthetic rows are derived only from healthy recordings in each training fold.
Every score is calculated on untouched real recordings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audit_raw_signals import targeted
from .extract_features import canonical_stem, parse_recording_name


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42


def gate_model(class_weight: dict[int, float] | str = "balanced"):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1, class_weight=class_weight, max_iter=2000),
    )


def augment_window(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small operating and sensor changes to one four-channel window."""
    n = len(source)
    speed = rng.uniform(0.985, 1.015)
    sample_positions = (np.arange(n) * speed + rng.uniform(0, n)) % n
    left = np.floor(sample_positions).astype(int)
    fraction = sample_positions - left
    warped = source[left] * (1 - fraction[:, None]) + source[(left + 1) % n] * fraction[:, None]

    # Small sensor cross-coupling and independent channel gain changes.
    mixing = np.eye(4) * rng.uniform(0.96, 1.04, 4)
    off_diagonal = rng.normal(0, 0.008, (4, 4))
    np.fill_diagonal(off_diagonal, 0)
    warped = warped @ (mixing + off_diagonal)

    # Add low-amplitude measurement noise at a realistic 35-50 dB SNR.
    rms = np.sqrt(np.mean(warped**2, axis=0)).clip(1e-12)
    snr_db = rng.uniform(35, 50, 4)
    noise_scale = rms / (10 ** (snr_db / 20))
    return warped + rng.normal(size=warped.shape) * noise_scale


def generate(per_recording: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []
    paths = sorted((ROOT / "data/raw/vibration").glob("*.mat"))
    healthy_paths = [p for p in paths if parse_recording_name(canonical_stem(p)).fault_type == "normal"]
    if len(healthy_paths) != 3:
        raise RuntimeError(f"Expected 3 healthy source recordings, found {len(healthy_paths)}")

    for path in healthy_paths:
        label = parse_recording_name(canonical_stem(path))
        print(f"Generating from {label.recording_id}", flush=True)
        signal = loadmat(path, squeeze_me=True, struct_as_record=False)["Signal"]
        raw = np.asarray(signal.y_values.values, dtype=np.float64)
        fs = float(1 / signal.x_values.increment)
        transform = signal.y_values.quantity.unit_transformation
        raw = raw * float(transform.factor) + float(transform.offset)
        n = int(round(fs))
        windows = len(raw) // n
        for synthetic_id in range(per_recording):
            source_window = int(rng.integers(windows))
            values = raw[source_window * n : (source_window + 1) * n]
            values = augment_window(values, rng)
            row = {
                "recording_id": f"SYNTH_{label.recording_id}_{synthetic_id:04d}",
                "source_recording_id": label.recording_id,
                "window_id": synthetic_id,
                "fault_type": "normal",
                "load": label.load,
            }
            for channel in range(4):
                row.update(targeted(values[:, channel], fs, f"vibration_{channel + 1}"))
            rows.append(row)
    frame = pd.DataFrame(rows)
    output = ROOT / "data/processed/synthetic_healthy_vibration.csv"
    frame.to_csv(output, index=False)
    return frame


def evaluate(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    columns = [c for c in real if c.startswith("vibration_")]
    target = real.fault_type.ne("normal").astype(int)
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_SEED).split(
            real[columns], real.fault_type, real.recording_id
        )
    )
    experiments: dict[str, dict] = {}
    for per_source in (0, 50, 100, 200, 300):
        predictions = np.zeros(len(real), dtype=int)
        probabilities = np.zeros(len(real), dtype=float)
        for train_index, test_index in folds:
            train = real.iloc[train_index]
            train_groups = set(train.recording_id)
            augmented = synthetic[
                synthetic.source_recording_id.isin(train_groups)
                & synthetic.window_id.lt(per_source)
            ]
            x_train = pd.concat([train[columns], augmented[columns]], ignore_index=True)
            y_train = np.concatenate(
                [train.fault_type.ne("normal").astype(int), np.zeros(len(augmented), dtype=int)]
            )
            model = gate_model().fit(x_train, y_train)
            probability = model.predict_proba(real.iloc[test_index][columns])[:, 1]
            probabilities[test_index] = probability
            predictions[test_index] = probability >= 0.5

        report = classification_report(target, predictions, target_names=["normal", "abnormal"], output_dict=True, zero_division=0)
        threshold_sweep = {}
        for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            threshold_predictions = probabilities >= threshold
            threshold_report = classification_report(
                target,
                threshold_predictions,
                target_names=["normal", "abnormal"],
                output_dict=True,
                zero_division=0,
            )
            threshold_sweep[f"{threshold:.2f}"] = {
                "normal_recall": threshold_report["normal"]["recall"],
                "abnormal_recall": threshold_report["abnormal"]["recall"],
                "accuracy": threshold_report["accuracy"],
                "macro_f1": threshold_report["macro avg"]["f1-score"],
            }
        experiments[str(per_source)] = {
            "synthetic_per_training_source": per_source,
            "synthetic_generation": "time warp 0.985-1.015, channel gain/coupling, 35-50 dB noise",
            "evaluation": "three-fold recording-grouped CV on real rows only",
            "normal_recall": report["normal"]["recall"],
            "abnormal_recall": report["abnormal"]["recall"],
            "binary_accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "confusion_matrix": confusion_matrix(target, predictions).tolist(),
            "threshold_sweep": threshold_sweep,
        }
    return experiments


def main():
    real = pd.read_csv(ROOT / "data/processed/targeted_vibration.csv")
    synthetic_path = ROOT / "data/processed/synthetic_healthy_vibration.csv"
    synthetic = pd.read_csv(synthetic_path) if synthetic_path.exists() else generate()
    results = evaluate(real, synthetic)
    output = ROOT / "reports/healthy_augmentation_audit.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
