"""Build cached RPM-normalized features for both public datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .audit_raw_signals import ROOT
from .extract_features import canonical_stem, pair_recordings, parse_recording_name
from .import_cross_speed_dataset import scalar, text, CONDITION_ALIASES, VALID_CONDITIONS
from .order_features import order_features


KAIST_SPEED_RPM = 3010.0


def build_kaist() -> pd.DataFrame:
    rows = []
    pairs = pair_recordings(ROOT / "data/raw/current_temp", ROOT / "data/raw/vibration")
    for index, (_, path) in enumerate(pairs, start=1):
        print(f"KAIST recording {index}/{len(pairs)}", flush=True)
        signal = loadmat(path, squeeze_me=True, struct_as_record=False)["Signal"]
        raw = np.asarray(signal.y_values.values, dtype=np.float64)
        sample_rate = float(1 / signal.x_values.increment)
        quantity = signal.y_values.quantity
        raw = raw * float(quantity.unit_transformation.factor) + float(quantity.unit_transformation.offset)
        label = parse_recording_name(canonical_stem(path))
        window_size = int(round(sample_rate))
        for window in range(len(raw) // window_size):
            values = raw[window * window_size : (window + 1) * window_size]
            row = {
                "recording_id": label.recording_id,
                "window_id": window,
                "fault_type": label.fault_type,
                "load": label.load,
                "speed_rpm": KAIST_SPEED_RPM,
            }
            for channel in range(4):
                row.update(order_features(values[:, channel], sample_rate, KAIST_SPEED_RPM, f"vibration_{channel + 1}"))
            rows.append(row)
    return pd.DataFrame(rows)


def build_cross_speed(data_root: Path) -> pd.DataFrame:
    paths = sorted(data_root.rglob("*.mat"))
    if not paths:
        raise ValueError(f"No cross-speed MAT files found below {data_root}")
    rows = []
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"Cross-speed sample {index}/{len(paths)}", flush=True)
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        signals = np.asarray(mat["signals"], dtype=np.float64)
        if signals.shape == (4, 25600):
            signals = signals.T
        sample_rate = float(scalar(mat["fs"]))
        speed = int(round(float(scalar(mat["speed_rpm"]))))
        condition = CONDITION_ALIASES.get(text(mat["condition"]).strip().lower(), text(mat["condition"]).strip().lower())
        if signals.shape != (25600, 4) or condition not in VALID_CONDITIONS:
            raise ValueError(f"Unexpected sample in {path}")
        group = text(mat["original_file"]).strip()
        second = int(round(float(scalar(mat.get("second_id", 0)))))
        row = {
            "recording_id": f"CROSSSPEED_{speed}_{group}",
            "window_id": second,
            "condition": condition,
            "speed_rpm": speed,
        }
        for channel in range(4):
            row.update(order_features(signals[:, channel], sample_rate, speed, f"vibration_{channel + 1}"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cross-root",
        type=Path,
        default=ROOT / "data/raw/Four-Channel Vibration Dataset for Cross-Speed Rot/SRO-Cross-Speed-Dataset/extracted",
    )
    args = parser.parse_args()
    output = ROOT / "data/processed"
    kaist = build_kaist()
    cross = build_cross_speed(args.cross_root)
    kaist.to_csv(output / "kaist_order_features.csv", index=False)
    cross.to_csv(output / "cross_speed_order_features.csv", index=False)
    print(f"Wrote {len(kaist)} KAIST rows and {len(cross)} cross-speed rows")


if __name__ == "__main__":
    main()
