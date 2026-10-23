"""Convert the Qasim/Almubarak four-channel dataset into v2 features.

Dataset DOI: 10.17632/y8mv28hjwb.1
Expected MAT fields: signals, fs, speed_rpm, condition, original_file, second_id.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .audit_raw_signals import targeted


ROOT = Path(__file__).resolve().parents[1]
VALID_CONDITIONS = {"normal", "imbalance", "misalignment"}
CONDITION_ALIASES = {"healthy": "normal"}


def scalar(value):
    array = np.asarray(value).squeeze()
    return array.item() if array.ndim == 0 else array


def text(value) -> str:
    value = scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return "".join(value.astype(str).flat)
    return str(value)


def load_sample(path: Path) -> dict:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    missing = [name for name in ("signals", "fs", "speed_rpm", "condition", "original_file") if name not in mat]
    if missing:
        raise ValueError(f"{path.name}: missing fields {missing}")

    signals = np.asarray(mat["signals"], dtype=np.float64)
    if signals.shape == (4, 25600):
        signals = signals.T
    if signals.shape != (25600, 4):
        raise ValueError(f"{path.name}: expected signals 4x25600, got {signals.shape}")
    fs = float(scalar(mat["fs"]))
    speed = int(round(float(scalar(mat["speed_rpm"]))))
    source_condition = text(mat["condition"]).strip().lower()
    condition = CONDITION_ALIASES.get(source_condition, source_condition)
    if fs != 25600 or speed not in {600, 900, 1200} or condition not in VALID_CONDITIONS:
        raise ValueError(f"{path.name}: unexpected fs/speed/condition {fs}/{speed}/{condition}")

    group = text(mat["original_file"]).strip()
    second = int(round(float(scalar(mat.get("second_id", 0)))))
    row = {
        "recording_id": f"CROSSSPEED_{speed}_{group}",
        "window_id": second,
        "condition": condition,
        "speed_rpm": speed,
        "source_file": path.name,
    }
    for channel in range(4):
        row.update(targeted(signals[:, channel], fs, f"vibration_{channel + 1}"))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_root",
        nargs="?",
        type=Path,
        default=ROOT / "data/raw/cross_speed_4ch",
        help="Folder containing extracted MAT files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/cross_speed_4ch_features.csv",
    )
    args = parser.parse_args()
    paths = sorted(args.data_root.rglob("*.mat"))
    if not paths:
        raise SystemExit(f"No MAT files found below {args.data_root}")
    rows = []
    for index, path in enumerate(paths, start=1):
        if index == 1 or index % 100 == 0 or index == len(paths):
            print(f"Processing {index}/{len(paths)}", flush=True)
        rows.append(load_sample(path))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["recording_id", "window_id"]).any():
        raise ValueError("Duplicate acquisition/second identifiers found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.groupby(["condition", "speed_rpm"]).size().to_string())
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
