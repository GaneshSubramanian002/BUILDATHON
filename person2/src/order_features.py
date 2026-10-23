"""RPM-normalized vibration features for cross-speed experiments.

An order is frequency divided by shaft frequency.  A peak at 1x rotation is
therefore at order 1 whether the machine runs at 600, 900, or 3010 RPM.
"""
from __future__ import annotations

import numpy as np


ORDER_BANDS = (
    (0.0, 0.75),
    (0.75, 1.25),
    (1.25, 1.75),
    (1.75, 2.25),
    (2.25, 3.5),
    (3.5, 5.0),
    (5.0, 8.0),
    (8.0, 12.0),
    (12.0, 20.0),
    (20.0, 40.0),
    (40.0, 80.0),
    (80.0, float("inf")),
)


def order_features(values, sample_rate_hz: float, speed_rpm: float, prefix: str) -> dict[str, float]:
    """Return 18 normalized spectral features for one vibration channel."""
    if sample_rate_hz <= 0 or speed_rpm <= 0:
        raise ValueError("sample rate and RPM must be positive")
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(len(values)))) ** 2
    frequency = np.fft.rfftfreq(len(values), 1 / sample_rate_hz)
    order = frequency / (speed_rpm / 60.0)
    weights = power / max(float(power.sum()), 1e-20)

    result: dict[str, float] = {}
    for low, high in ORDER_BANDS:
        high_name = "inf" if np.isinf(high) else str(high).replace(".", "p")
        low_name = str(low).replace(".", "p")
        mask = (order >= low) & (order < high)
        result[f"{prefix}_order_energy_{low_name}_{high_name}"] = float(weights[mask].sum())
    # A width of 0.20 order contains several bins even at 600 RPM and one
    # neighboring bin around the 3010-RPM source machine's harmonics.
    for harmonic in (1, 2, 3, 4):
        mask = np.abs(order - harmonic) <= 0.20
        result[f"{prefix}_order_harmonic_{harmonic}"] = float(weights[mask].sum())
    result[f"{prefix}_order_entropy"] = float(-np.dot(weights, np.log(weights + 1e-20)))
    result[f"{prefix}_dominant_order"] = float(min(order[np.argmax(power)], 256.0))
    return result

