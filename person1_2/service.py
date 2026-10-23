"""One-second sensor generation and Person 2 inference bridge.

The simulator remains independent: labels are retained only in ``raw_reading``
and are never included in the dictionary sent to the ML model.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Any

from person1.sensor_simulator import SensorSimulator
from person2.src.audit_raw_signals import targeted
from person2.src.order_features import order_features
from person2.src.prediction import predict


@dataclass(frozen=True)
class ProcessedReading:
    """A model result plus telemetry and offline-only simulator truth."""

    raw_reading: Any
    sensor_data: dict[str, float]
    telemetry: dict[str, Any]
    prediction: dict[str, Any]

    def public_json(self) -> dict[str, Any]:
        """Return the safe payload for a backend or frontend consumer."""
        return {
            "reading_id": self.raw_reading.reading_id,
            "telemetry": self.telemetry,
            "prediction": self.prediction,
        }


def build_model_input(reading) -> dict[str, float]:
    """Extract the exact v2.2 model fields from a raw simulator reading."""
    payload: dict[str, float] = {
        "load": float(reading.load_nm),
        "speed_rpm": float(reading.internal_shaft_frequency_hz * 60.0),
    }
    for channel, waveform in enumerate(reading.vibration_g, start=1):
        prefix = f"vibration_{channel}"
        payload.update(targeted(waveform, reading.sample_rate_hz, prefix))
        order = order_features(
            waveform,
            reading.sample_rate_hz,
            reading.internal_shaft_frequency_hz * 60.0,
            prefix,
        )
        payload.update({f"order__{name}": value for name, value in order.items()})
    return payload


def build_telemetry(reading) -> dict[str, Any]:
    """Return dashboard values without simulator labels."""
    current_rms = (reading.current_a**2).mean(axis=1) ** 0.5
    vibration_rms = (reading.vibration_g**2).mean(axis=1) ** 0.5
    return {
        "equipment_id": reading.equipment_id,
        "timestamp": reading.timestamp,
        "sequence_id": reading.sequence_id,
        "sequence_step": reading.sequence_step,
        "load_nm": reading.load_nm,
        "speed_rpm": reading.internal_shaft_frequency_hz * 60.0,
        "sample_rate_hz": reading.sample_rate_hz,
        "vibration_rms_g": [float(value) for value in vibration_rms],
        "current_rms_a": [float(value) for value in current_rms],
        "temperature_c": [float(value) for value in reading.temperature_c],
    }


class ReadingService:
    """Generate and classify one reading per interval on a background thread."""

    def __init__(
        self,
        *,
        fault_type: str = "normal",
        severity_rank: int = 0,
        load_nm: float = 2.0,
        interval_seconds: float = 1.0,
        seed: int = 20261023,
        equipment_id: str = "MOTOR_01",
        callback: Callable[[ProcessedReading], None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        SensorSimulator.validate_scenario(fault_type, severity_rank, load_nm)
        self.fault_type = fault_type
        self.severity_rank = severity_rank
        self.load_nm = load_nm
        self.interval_seconds = float(interval_seconds)
        self.equipment_id = equipment_id
        self.callback = callback
        self.simulator = SensorSimulator(seed=seed)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._counter = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def generate_once(self) -> ProcessedReading:
        """Generate one raw reading and synchronously return its ML result."""
        now = datetime.now(timezone.utc)
        self._counter += 1
        reading = self.simulator.generate_reading(
            reading_id=f"LIVE_{self._counter:06d}",
            fault_type=self.fault_type,
            severity_rank=self.severity_rank,
            load_nm=self.load_nm,
            timestamp=now,
            sequence_id="LIVE_SEQUENCE",
            sequence_step=self._counter - 1,
            equipment_id=self.equipment_id,
        )
        sensor_data = build_model_input(reading)
        result = ProcessedReading(
            raw_reading=reading,
            sensor_data=sensor_data,
            telemetry=build_telemetry(reading),
            prediction=predict(sensor_data),
        )
        if self.callback is not None:
            self.callback(result)
        return result

    def _run(self, max_readings: int | None) -> None:
        completed = 0
        next_tick = time.monotonic()
        while not self._stop_event.is_set() and (max_readings is None or completed < max_readings):
            self.generate_once()
            completed += 1
            next_tick += self.interval_seconds
            self._stop_event.wait(max(0.0, next_tick - time.monotonic()))

    def start(self, max_readings: int | None = None) -> None:
        """Start generation; call ``stop()`` to end an unlimited run."""
        if self.running:
            raise RuntimeError("reading service is already running")
        if max_readings is not None and max_readings < 1:
            raise ValueError("max_readings must be positive")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(max_readings,), name="person1-2-reading-service", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        """Stop generation promptly and wait for the worker to finish."""
        self._stop_event.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        if self._thread is not None and not self._thread.is_alive():
            self._thread = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one-second readings and run Person 2 ML inference")
    parser.add_argument("--fault", default="normal", choices=["normal", "inner_race_fault", "outer_race_fault", "misalignment", "rotor_unbalance"])
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--load", type=float, default=2.0, choices=[0.0, 2.0, 4.0])
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=None, help="Stop after this many readings; omit for continuous mode")
    args = parser.parse_args()

    def print_reading(item: ProcessedReading) -> None:
        print(json.dumps(item.public_json(), allow_nan=False), flush=True)

    service = ReadingService(
        fault_type=args.fault,
        severity_rank=args.severity,
        load_nm=args.load,
        interval_seconds=args.interval,
        callback=print_reading,
    )
    service.start(max_readings=args.count)
    try:
        while service.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping reading service...", flush=True)
        service.stop()


if __name__ == "__main__":
    main()
