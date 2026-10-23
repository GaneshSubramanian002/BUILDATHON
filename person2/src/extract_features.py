"""Create one-second, window-level features from paired MAT and TDMS files."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.io import loadmat


SAMPLE_RATE = 25_600
WINDOW_SAMPLES = SAMPLE_RATE

TDMS_CHANNELS = {
    "temperature_A": "cDAQ9185-1F486B5Mod1/ai0",
    "temperature_B": "cDAQ9185-1F486B5Mod1/ai1",
    "current_U": "cDAQ9185-1F486B5Mod2/ai0",
}

FAULT_NAMES = {
    "normal": "normal",
    "bpfi": "inner_race_fault",
    "bpfo": "outer_race_fault",
    "misalign": "misalignment",
    "unbalance": "rotor_unbalance",
}


@dataclass(frozen=True)
class RecordingLabel:
    recording_id: str
    load: float
    fault_type: str
    severity_value: float
    severity_rank: int


def canonical_stem(path: Path) -> str:
    """Normalize the one known typo in the published vibration filenames."""
    return path.stem.replace("Unbalalnce", "Unbalance")


def parse_recording_name(stem: str) -> RecordingLabel:
    """Decode load, fault, and severity from a published recording name."""
    normalized = stem.replace("Unbalalnce", "Unbalance")
    match = re.fullmatch(
        r"(?P<load>[024])Nm_(?P<fault>Normal|BPFI|BPFO|Misalign|Unbalance)"
        r"(?:_(?P<severity>\d+)(?P<unit>mg)?)?",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Unrecognized recording filename: {stem}")

    fault_key = match.group("fault").lower()
    severity_text = match.group("severity")
    severity_value = float(severity_text or 0)

    if fault_key == "normal":
        severity_rank = 0
    elif fault_key in {"bpfi", "bpfo"}:
        severity_rank = {3: 1, 10: 2, 30: 3}[int(severity_value)]
    elif fault_key == "misalign":
        severity_rank = {1: 1, 3: 2, 5: 3}[int(severity_value)]
    else:
        severity_rank = {
            583: 1,
            1169: 2,
            1751: 3,
            2239: 4,
            3318: 5,
        }[int(severity_value)]

    return RecordingLabel(
        recording_id=normalized,
        load=float(match.group("load")),
        fault_type=FAULT_NAMES[fault_key],
        severity_value=severity_value,
        severity_rank=severity_rank,
    )


def pair_recordings(current_temp_dir: Path, vibration_dir: Path) -> list[tuple[Path, Path]]:
    """Return paired TDMS/MAT paths, rejecting missing or duplicate recordings."""
    tdms = {canonical_stem(path): path for path in current_temp_dir.glob("*.tdms")}
    mats = {canonical_stem(path): path for path in vibration_dir.glob("*.mat")}
    if not tdms or not mats:
        raise FileNotFoundError("Both TDMS and MAT source files are required")

    current_only = sorted(tdms.keys() - mats.keys())
    vibration_only = sorted(mats.keys() - tdms.keys())
    if current_only or vibration_only:
        raise ValueError(
            f"Unpaired recordings: current_only={current_only}, "
            f"vibration_only={vibration_only}"
        )
    return [(tdms[key], mats[key]) for key in sorted(tdms)]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def signal_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Shape and spectral features of one window; kurtosis uses Pearson form."""
    centered = values - np.mean(values)
    std = float(np.std(values))
    peak = float(np.max(np.abs(values)))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(values)))) ** 2
    frequencies = np.fft.rfftfreq(len(values), 1 / SAMPLE_RATE)
    total = float(power.sum())
    result = {
        f"{prefix}_std": std,
        f"{prefix}_peak": peak,
        f"{prefix}_ptp": float(np.ptp(values)),
        f"{prefix}_kurtosis": float(np.mean(centered ** 4) / std ** 4) if std else 0.0,
        f"{prefix}_crest": peak / max(rms(values), 1e-12),
        f"{prefix}_spectral_centroid": float(np.dot(frequencies, power) / total) if total else 0.0,
    }
    for low, high in [(0, 200), (200, 1000), (1000, 5000), (5000, 12801)]:
        mask = (frequencies >= low) & (frequencies < high)
        result[f"{prefix}_band_{low}_{high}"] = float(power[mask].sum() / total) if total else 0.0
    return result


def extract_recording(tdms_path: Path, mat_path: Path) -> pd.DataFrame:
    """Extract aligned one-second summary features for one recording pair."""
    label = parse_recording_name(canonical_stem(tdms_path))
    vibration = loadmat(
        mat_path,
        squeeze_me=True,
        struct_as_record=False,
        variable_names=["Signal"],
    )["Signal"].y_values.values
    if vibration.ndim != 2 or vibration.shape[1] != 4:
        raise ValueError(f"Expected four vibration channels in {mat_path.name}")

    rows: list[dict[str, float | int | str]] = []
    with TdmsFile.open(tdms_path) as tdms:
        group = tdms["Log"]
        # Read each required channel once. Repeated slices can cross an unusual
        # final TDMS segment in the long 0 Nm normal recording and trigger an
        # npTDMS chunk-boundary error.
        channels = {
            key: np.asarray(group[name][:], dtype=float)
            for key, name in TDMS_CHANNELS.items()
        }
        available_seconds = min(
            vibration.shape[0] // WINDOW_SAMPLES,
            *(len(channel) // WINDOW_SAMPLES for channel in channels.values()),
        )

        for window_id in range(available_seconds):
            start = window_id * WINDOW_SAMPLES
            stop = start + WINDOW_SAMPLES
            vib_window = vibration[start:stop]
            temp_a = channels["temperature_A"][start:stop]
            temp_b = channels["temperature_B"][start:stop]
            current_u = channels["current_U"][start:stop]

            rows.append(
                {
                    "recording_id": label.recording_id,
                    "window_id": window_id,
                    "temperature_A_mean": float(np.mean(temp_a)),
                    "temperature_B_mean": float(np.mean(temp_b)),
                    "current_U_rms": rms(current_u),
                    "vibration_A_x_rms": rms(vib_window[:, 0]),
                    "vibration_A_y_rms": rms(vib_window[:, 1]),
                    "vibration_B_x_rms": rms(vib_window[:, 2]),
                    "vibration_B_y_rms": rms(vib_window[:, 3]),
                    "load": label.load,
                    "fault_type": label.fault_type,
                    "severity_value": label.severity_value,
                    "severity_rank": label.severity_rank,
                }
            )
            rows[-1]["temperature_difference"] = float(abs(np.mean(temp_a) - np.mean(temp_b)))
            rows[-1].update(signal_features(current_u, "current_U"))
            for index, prefix in enumerate(["vibration_A_x", "vibration_A_y", "vibration_B_x", "vibration_B_y"]):
                rows[-1].update(signal_features(vib_window[:, index], prefix))
    return pd.DataFrame(rows)


def extract_dataset(current_temp_dir: Path, vibration_dir: Path, output: Path) -> pd.DataFrame:
    """Extract all paired recordings and save one processed CSV."""
    pairs = pair_recordings(current_temp_dir, vibration_dir)
    frames = []
    for index, (tdms_path, mat_path) in enumerate(pairs, start=1):
        print(f"[{index:02d}/{len(pairs)}] {canonical_stem(tdms_path)}", flush=True)
        frames.append(extract_recording(tdms_path, mat_path))
    result = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-temp-dir",
        type=Path,
        default=project_dir / "data" / "raw" / "current_temp",
    )
    parser.add_argument(
        "--vibration-dir",
        type=Path,
        default=project_dir / "data" / "raw" / "vibration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "processed" / "features.csv",
    )
    args = parser.parse_args()
    data = extract_dataset(args.current_temp_dir, args.vibration_dir, args.output)
    print(f"Saved {len(data):,} windows to {args.output}")


if __name__ == "__main__":
    main()

