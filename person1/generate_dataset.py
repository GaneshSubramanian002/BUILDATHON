"""Command-line generator for independent Person 1 sensor readings."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .sensor_simulator import FAULT_SEVERITIES, VALID_LOADS, SensorSimulator


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    parser.add_argument("--sequences-per-class", type=int, default=1)
    parser.add_argument("--seconds-per-sequence", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--start-time", default="2026-10-23T10:00:00+05:30")
    args = parser.parse_args()
    if args.sequences_per_class < 1 or args.seconds_per_sequence < 1:
        raise SystemExit("Sequence counts must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    waveform_dir = args.output / "waveforms"
    waveform_dir.mkdir(exist_ok=True)
    simulator = SensorSimulator(seed=args.seed)
    start = datetime.fromisoformat(args.start_time)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    summaries = []
    truth = []
    global_step = 0
    for class_index, (fault, severities) in enumerate(FAULT_SEVERITIES.items()):
        for sequence_number in range(args.sequences_per_class):
            load = VALID_LOADS[(class_index + sequence_number) % len(VALID_LOADS)]
            severity = severities[sequence_number % len(severities)]
            sequence_id = f"SEQ_{fault.upper()}_{sequence_number + 1:03d}"
            for sequence_step in range(args.seconds_per_sequence):
                reading_id = f"READING_{global_step + 1:06d}"
                reading = simulator.generate_reading(
                    reading_id=reading_id,
                    fault_type=fault,
                    severity_rank=severity,
                    load_nm=load,
                    timestamp=start + timedelta(seconds=global_step),
                    sequence_id=sequence_id,
                    sequence_step=sequence_step,
                )
                relative_waveform = Path("waveforms") / f"{reading_id}.npz"
                reading.save_waveforms(args.output / relative_waveform)
                summaries.append(reading.sensor_summary(relative_waveform.as_posix()))
                truth.append(reading.ground_truth())
                global_step += 1

    with (args.output / "sensor_readings.jsonl").open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    with (args.output / "ground_truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    counts = Counter(row["fault_type"] for row in truth)
    manifest = {
        "generator": "person1.sensor_simulator.SensorSimulator",
        "seed": args.seed,
        "sample_rate_hz": simulator.sample_rate_hz,
        "window_seconds": 1,
        "reading_count": len(summaries),
        "counts_by_fault": dict(counts),
        "labels_are_separate": True,
        "connected_to_ml_model": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Generated files in {args.output}")


if __name__ == "__main__":
    main()
