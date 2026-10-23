"""Audit source metadata and test targeted vibration features separately from v1."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from nptdms import TdmsFile
from .extract_features import pair_recordings, canonical_stem, parse_recording_name

ROOT = Path(__file__).resolve().parents[1]


def text(value):
    if isinstance(value, np.ndarray):
        return ''.join(value.astype(str).flat)
    return str(value)


def targeted(values, fs, prefix):
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(len(values)))) ** 2
    frequencies = np.fft.rfftfreq(len(values), 1 / fs)
    total = max(float(power.sum()), 1e-20)
    weights = power / total
    features = {}
    for low, high in [(0, 40), (40, 60), (60, 90), (90, 110), (110, 140), (140, 160), (160, 250), (250, 500), (500, 1000), (1000, 3000), (3000, 6000), (6000, 12801)]:
        features[f'{prefix}_energy_{low}_{high}'] = float(weights[(frequencies >= low) & (frequencies < high)].sum())
    for order in [1, 2, 3, 4]:
        mask = (frequencies >= order * 50.17 - 5) & (frequencies <= order * 50.17 + 5)
        features[f'{prefix}_harmonic_{order}'] = float(weights[mask].sum())
    features[f'{prefix}_entropy'] = float(-np.dot(weights, np.log(weights + 1e-20)))
    features[f'{prefix}_dominant_hz'] = float(frequencies[np.argmax(power)])
    return features


def main():
    pairs = pair_recordings(ROOT / 'data/raw/current_temp', ROOT / 'data/raw/vibration')
    audit, rows = [], []
    for i, (tdms_path, mat_path) in enumerate(pairs):
        print(i + 1, canonical_stem(mat_path), flush=True)
        signal = loadmat(mat_path, squeeze_me=True, struct_as_record=False)['Signal']
        raw = signal.y_values.values
        fs = 1 / signal.x_values.increment
        quantity = signal.y_values.quantity
        factor = float(quantity.unit_transformation.factor)
        offset = float(quantity.unit_transformation.offset)
        td = TdmsFile.read_metadata(tdms_path)
        channels = td['Log'].channels()
        label = parse_recording_name(canonical_stem(mat_path))
        audit.append({
            'recording': label.recording_id, 'vibration_shape': list(raw.shape),
            'vibration_fs': fs, 'vibration_display_unit': str(quantity.label),
            'vibration_storage_note': str(quantity.info), 'display_factor': factor, 'display_offset': offset,
            'tdms_channels': [{'name': c.name, 'samples': len(c), 'dt': c.properties.get('wf_increment'), 'start': str(c.properties.get('wf_start_time'))} for c in channels],
            'mat_creation': [text(v) for v in np.atleast_1d(signal.function_record.creation_time)],
            'mat_absolute_time': [text(v) for v in np.atleast_1d(signal.function_record.TL_export_properties_annotation.absolute_time)],
        })
        n = int(round(fs))
        for window in range(len(raw) // n):
            values = raw[window*n:(window+1)*n] * factor + offset
            row = {'recording_id': label.recording_id, 'window_id': window, 'fault_type': label.fault_type, 'load': label.load}
            for ch in range(4):
                row.update(targeted(values[:, ch], fs, f'vibration_{ch+1}'))
            rows.append(row)
    (ROOT / 'reports/raw_metadata_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(ROOT / 'data/processed/targeted_vibration.csv', index=False)


if __name__ == '__main__':
    main()
