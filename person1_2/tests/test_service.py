import time
import unittest

from person1_2.service import ReadingService, build_model_input


class LiveIntegrationTests(unittest.TestCase):
    def test_one_reading_has_complete_model_input_and_prediction(self):
        service = ReadingService(interval_seconds=0.01)
        item = service.generate_once()
        self.assertEqual(len(item.sensor_data), 146)  # 144 features + load + speed
        self.assertEqual(item.prediction["schema_version"], "2.2.0")
        self.assertNotIn("fault_type", item.sensor_data)
        self.assertNotIn("fault_type", item.public_json())

    def test_start_and_stop(self):
        items = []
        service = ReadingService(interval_seconds=0.02, callback=items.append, calibrated_demo=False)
        service.start()
        time.sleep(0.08)
        service.stop()
        self.assertFalse(service.running)
        self.assertGreaterEqual(len(items), 2)
        count = len(items)
        time.sleep(0.05)
        self.assertEqual(len(items), count)

    def test_cycle_mode_changes_scenarios_without_exposing_labels(self):
        service = ReadingService(interval_seconds=0.01, scenario_mode="cycle")
        readings = [service.generate_once() for _ in range(5)]
        scores = [item.prediction["anomaly_score"] for item in readings]
        self.assertGreater(max(scores) - min(scores), 0.001)
        for item in readings:
            self.assertNotIn("fault_type", item.public_json())

    def test_calibrated_demo_matches_generated_fault(self):
        service = ReadingService(
            fault_type="outer_race_fault", severity_rank=3, load_nm=2.0, scenario_mode="fixed"
        )
        item = service.generate_once()
        self.assertTrue(item.prediction["anomaly"])
        self.assertEqual(item.prediction["possible_fault"], "outer_race_fault")


if __name__ == "__main__":
    unittest.main()
