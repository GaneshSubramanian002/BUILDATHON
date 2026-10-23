"""Independent rotating-machine sensor simulator.

This module deliberately does not import Person 2's feature or prediction code.
It generates raw one-second sensor measurements for later integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


FAULT_SEVERITIES = {
    "normal": (0,),
    "inner_race_fault": (1, 2, 3),
    "outer_race_fault": (1, 2, 3),
    "misalignment": (1, 2, 3),
    "rotor_unbalance": (1, 2, 3, 4, 5),
}
VALID_LOADS = (0.0, 2.0, 4.0)


@dataclass(frozen=True)
class RawReading:
    reading_id: str
    equipment_id: str
    timestamp: str
    sequence_id: str
    sequence_step: int
    load_nm: float
    sample_rate_hz: int
    vibration_g: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    fault_type: str
    severity_rank: int
    internal_shaft_frequency_hz: float

    def sensor_summary(self, waveform_file: str | None = None) -> dict[str, Any]:
        """Return a JSON-safe payload without ground-truth fault labels."""
        vibration_rms = np.sqrt(np.mean(self.vibration_g**2, axis=1))
        current_rms = np.sqrt(np.mean(self.current_a**2, axis=1))
        payload: dict[str, Any] = {
            "reading_id": self.reading_id,
            "equipment_id": self.equipment_id,
            "timestamp": self.timestamp,
            "sequence_id": self.sequence_id,
            "sequence_step": self.sequence_step,
            "load": self.load_nm,
            "sample_rate_hz": self.sample_rate_hz,
            "vibration_1_rms_g": float(vibration_rms[0]),
            "vibration_2_rms_g": float(vibration_rms[1]),
            "vibration_3_rms_g": float(vibration_rms[2]),
            "vibration_4_rms_g": float(vibration_rms[3]),
            "current_U_rms_a": float(current_rms[0]),
            "current_V_rms_a": float(current_rms[1]),
            "current_W_rms_a": float(current_rms[2]),
            "temperature_A_c": float(self.temperature_c[0]),
            "temperature_B_c": float(self.temperature_c[1]),
        }
        if waveform_file is not None:
            payload["waveform_file"] = waveform_file
        return payload

    def ground_truth(self) -> dict[str, Any]:
        return {
            "reading_id": self.reading_id,
            "sequence_id": self.sequence_id,
            "fault_type": self.fault_type,
            "severity_rank": self.severity_rank,
            "load": self.load_nm,
            "internal_shaft_frequency_hz": self.internal_shaft_frequency_hz,
        }

    def save_waveforms(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vibration_g=self.vibration_g.astype(np.float32),
            current_a=self.current_a.astype(np.float32),
            temperature_c=self.temperature_c.astype(np.float32),
            sample_rate_hz=np.int32(self.sample_rate_hz),
        )


class SensorSimulator:
    """Generate repeatable, physically motivated raw readings."""

    def __init__(self, seed: int = 20261023, sample_rate_hz: int = 25600):
        if sample_rate_hz != 25600:
            raise ValueError("This project uses 25,600 Hz one-second vibration windows")
        self.sample_rate_hz = sample_rate_hz
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def validate_scenario(fault_type: str, severity_rank: int, load_nm: float) -> None:
        if fault_type not in FAULT_SEVERITIES:
            raise ValueError(f"Unknown fault_type: {fault_type}")
        if severity_rank not in FAULT_SEVERITIES[fault_type]:
            raise ValueError(f"Invalid severity_rank {severity_rank} for {fault_type}")
        if float(load_nm) not in VALID_LOADS:
            raise ValueError(f"load must be one of {VALID_LOADS}")

    def _vibration(self, fault_type: str, severity: int, load: float, shaft_hz: float) -> np.ndarray:
        t = np.arange(self.sample_rate_hz) / self.sample_rate_hz
        phases = self.rng.uniform(0, 2 * np.pi, (4, 3))
        channel_scale = self.rng.uniform(0.88, 1.12, 4) * (1 + 0.06 * load)
        vibration = np.empty((4, self.sample_rate_hz), dtype=np.float64)
        for channel in range(4):
            base = (
                0.010 * np.sin(2 * np.pi * shaft_hz * t + phases[channel, 0])
                + 0.003 * np.sin(2 * np.pi * 2 * shaft_hz * t + phases[channel, 1])
                + 0.0015 * np.sin(2 * np.pi * 3 * shaft_hz * t + phases[channel, 2])
            )
            vibration[channel] = channel_scale[channel] * base

        if fault_type == "rotor_unbalance":
            for channel in range(4):
                vibration[channel] += (0.010 + 0.006 * severity) * channel_scale[channel] * np.sin(
                    2 * np.pi * shaft_hz * t + phases[channel, 0] + 0.2
                )
        elif fault_type == "misalignment":
            for channel in range(4):
                vibration[channel] += channel_scale[channel] * (
                    (0.005 + 0.005 * severity) * np.sin(2 * np.pi * 2 * shaft_hz * t + phases[channel, 1])
                    + (0.002 + 0.0025 * severity) * np.sin(2 * np.pi * 3 * shaft_hz * t + phases[channel, 2])
                )
        elif fault_type in {"inner_race_fault", "outer_race_fault"}:
            pass_frequency = (5.4 if fault_type == "inner_race_fault" else 3.6) * shaft_hz
            resonance_hz = 2600 if fault_type == "inner_race_fault" else 1850
            envelope = np.maximum(np.sin(2 * np.pi * pass_frequency * t), 0) ** 14
            for channel in range(4):
                resonance = np.sin(2 * np.pi * resonance_hz * t + phases[channel, 1])
                vibration[channel] += channel_scale[channel] * (0.008 + 0.007 * severity) * envelope * resonance

        # Independent sensor noise and a small common structural component.
        common = self.rng.normal(0, 0.0012, self.sample_rate_hz)
        vibration += common
        vibration += self.rng.normal(0, 0.0025 + 0.00025 * load, vibration.shape)
        vibration *= self.rng.uniform(0.97, 1.03, (4, 1))
        return vibration

    def _current(self, fault_type: str, severity: int, load: float, shaft_hz: float) -> np.ndarray:
        t = np.arange(self.sample_rate_hz) / self.sample_rate_hz
        electrical_hz = 50.0
        peak = np.sqrt(2) * (1.6 + 0.32 * load)
        phase = np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])[:, None]
        current = peak * np.sin(2 * np.pi * electrical_hz * t + phase)
        if fault_type != "normal":
            modulation_depth = 0.006 * severity
            modulation = 1 + modulation_depth * np.sin(2 * np.pi * shaft_hz * t)
            current *= modulation
        current += self.rng.normal(0, 0.015 + 0.003 * load, current.shape)
        return current

    def generate_reading(
        self,
        *,
        reading_id: str,
        fault_type: str,
        severity_rank: int,
        load_nm: float,
        timestamp: datetime,
        sequence_id: str,
        sequence_step: int,
        equipment_id: str = "MOTOR_01",
    ) -> RawReading:
        self.validate_scenario(fault_type, severity_rank, load_nm)
        shaft_hz = 50.17 + self.rng.normal(0, 0.18) - 0.025 * load_nm
        vibration = self._vibration(fault_type, severity_rank, load_nm, shaft_hz)
        current = self._current(fault_type, severity_rank, load_nm, shaft_hz)
        warmup = min(max(sequence_step, 0), 120) / 120
        fault_heat = severity_rank * (0.45 if fault_type != "normal" else 0)
        base_temperature = 25.5 + 1.15 * load_nm + 1.8 * warmup + fault_heat
        temperature = base_temperature + np.array([0.0, 0.55]) + self.rng.normal(0, 0.08, 2)
        return RawReading(
            reading_id=reading_id,
            equipment_id=equipment_id,
            timestamp=timestamp.isoformat(),
            sequence_id=sequence_id,
            sequence_step=sequence_step,
            load_nm=float(load_nm),
            sample_rate_hz=self.sample_rate_hz,
            vibration_g=vibration,
            current_a=current,
            temperature_c=temperature,
            fault_type=fault_type,
            severity_rank=severity_rank,
            internal_shaft_frequency_hz=float(shaft_hz),
        )
