"""Regression checks for the frozen prediction interface and feature math."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from person2.src.prediction import predict, predict_many, load_artifact
from person2.src.extract_features import signal_features, rms, pair_recordings
from person2.src.audit_raw_signals import targeted
from person2.src.order_features import order_features

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = json.loads((ROOT / 'handoff/integration_examples.json').read_text())

    def test_feature_math_sine(self):
        values = 2 * np.sin(2 * np.pi * 50 * np.arange(25600) / 25600)
        features = signal_features(values, 'test')
        self.assertAlmostEqual(rms(values), np.sqrt(2), places=6)
        self.assertAlmostEqual(features['test_kurtosis'], 1.5, places=6)
        self.assertAlmostEqual(features['test_spectral_centroid'], 50, places=2)
        self.assertAlmostEqual(sum(v for k, v in features.items() if '_band_' in k), 1)
        spectrum = targeted(values, 25600, 'vibration_1')
        self.assertAlmostEqual(spectrum['vibration_1_dominant_hz'], 50)
        self.assertGreater(spectrum['vibration_1_harmonic_1'], .999)
        self.assertAlmostEqual(sum(v for k, v in spectrum.items() if '_energy_' in k), 1)
        orders = order_features(values, 25600, 3000, 'vibration_1')
        self.assertAlmostEqual(orders['vibration_1_dominant_order'], 1)
        self.assertGreater(orders['vibration_1_order_harmonic_1'], .999)
        self.assertAlmostEqual(sum(v for k, v in orders.items() if '_order_energy_' in k), 1)

    def test_examples_output_invariants(self):
        results = predict_many([e['sensor_data'] for e in self.examples])
        self.assertEqual(len(results), 45)
        for result in results:
            json.dumps(result, allow_nan=False)
            self.assertEqual(result['anomaly'], result['possible_fault'] != 'normal')
            self.assertEqual(result['condition'] == 'NORMAL', not result['anomaly'])
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)
            self.assertIn(result['health_score'], [90, 60, 25])
        self.assertEqual(predict(self.examples[0]['sensor_data']), results[0])

    def test_invalid_inputs(self):
        sample = self.examples[0]['sensor_data']
        first = load_artifact()['feature_columns'][0]
        cases = []
        for bad in [float('nan'), float('inf'), True, '52', None]:
            row = copy.deepcopy(sample)
            row[first] = bad
            cases.append(row)
        row = dict(sample)
        row.pop(first)
        cases.append(row)
        row = dict(sample)
        row[first] = -1
        cases += [dict(sample, load=3), dict(sample, fault_type='normal'), row]
        cases += [[], None]
        for case in cases:
            with self.subTest(case=str(case)[:40]):
                with self.assertRaises(ValueError):
                    predict(case)

    def test_group_manifest_has_no_leakage(self):
        metrics = json.loads((ROOT / 'models/final_metrics.json').read_text())
        seen = []
        for fold in metrics['folds']:
            self.assertFalse(set(fold['train_recordings']) & set(fold['test_recordings']))
            seen += fold['test_recordings']
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), 45)


if __name__ == '__main__':
    unittest.main()
