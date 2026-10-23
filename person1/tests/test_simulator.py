from datetime import datetime, timezone
from pathlib import Path
import unittest

import numpy as np

from person1.sensor_simulator import SensorSimulator


class SimulatorTests(unittest.TestCase):
    def make(self, simulator, fault="normal", severity=0):
        return simulator.generate_reading(
            reading_id="R1",
            fault_type=fault,
            severity_rank=severity,
            load_nm=2,
            timestamp=datetime(2026, 10, 23, tzinfo=timezone.utc),
            sequence_id="S1",
            sequence_step=0,
        )

    def test_shapes_finite_and_label_separation(self):
        reading = self.make(SensorSimulator(7))
        self.assertEqual(reading.vibration_g.shape, (4, 25600))
        self.assertEqual(reading.current_a.shape, (3, 25600))
        self.assertEqual(reading.temperature_c.shape, (2,))
        self.assertTrue(np.isfinite(reading.vibration_g).all())
        summary = reading.sensor_summary()
        self.assertNotIn("fault_type", summary)
        self.assertNotIn("severity_rank", summary)

    def test_seed_is_repeatable(self):
        first = self.make(SensorSimulator(12)).vibration_g
        second = self.make(SensorSimulator(12)).vibration_g
        np.testing.assert_array_equal(first, second)

    def test_fault_has_more_vibration_than_normal_for_same_seed(self):
        normal = self.make(SensorSimulator(21)).vibration_g
        fault = self.make(SensorSimulator(21), "rotor_unbalance", 3).vibration_g
        self.assertGreater(np.sqrt(np.mean(fault**2)), np.sqrt(np.mean(normal**2)))

    def test_invalid_scenario_rejected(self):
        simulator = SensorSimulator(1)
        with self.assertRaises(ValueError):
            self.make(simulator, "normal", 2)
        with self.assertRaises(ValueError):
            simulator.generate_reading(
                reading_id="R", fault_type="normal", severity_rank=0, load_nm=3,
                timestamp=datetime.now(timezone.utc), sequence_id="S", sequence_step=0,
            )

    def test_npz_round_trip(self):
        reading = self.make(SensorSimulator(4))
        path = Path(__file__).resolve().parents[1] / "generated" / "test_round_trip.npz"
        try:
            reading.save_waveforms(path)
            with np.load(path) as stored:
                self.assertEqual(stored["vibration_g"].shape, (4, 25600))
                self.assertEqual(int(stored["sample_rate_hz"]), 25600)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
