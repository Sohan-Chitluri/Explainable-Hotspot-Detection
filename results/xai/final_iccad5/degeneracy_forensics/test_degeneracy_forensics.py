"""CPU-only tests; synthetic arrays are NOT checkpoint evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts/improvements/degeneracy_forensics.py"
spec = importlib.util.spec_from_file_location("forensics", SCRIPT)
assert spec is not None and spec.loader is not None
f = importlib.util.module_from_spec(spec)
if SCRIPT.exists():
    spec.loader.exec_module(f)

class StatisticsTests(unittest.TestCase):
    def test_array_statistics_and_undefined_inputs(self):
        self.assertTrue(hasattr(f, "array_stats"), "array_stats is not implemented")
        s = f.array_stats(np.array([-2., 0., 2.]))
        self.assertTrue(s["valid"])
        self.assertAlmostEqual(s["l2_norm"], np.sqrt(8))
        self.assertEqual(s["mean"], 0.)
        self.assertAlmostEqual(s["variance"], 8/3)
        self.assertEqual((s["min"], s["max"], s["ptp"]), (-2., 2., 4.))
        self.assertEqual(s["nonzero_count"], 2)
        for a, reason in [(np.array([]), "empty_array"), (np.array([np.nan]), "nonfinite_array")]:
            s = f.array_stats(a)
            self.assertFalse(s["valid"])
            self.assertEqual(s["reason"], reason)
            self.assertIsNone(s["mean"])
        self.assertNotIn("tensorflow", sys.modules)

    def test_cancellation_is_distinct_from_negative_cam(self):
        self.assertTrue(hasattr(f, "cam_statistics"), "cam_statistics is not implemented")
        a = np.ones((2, 2, 1))
        g = np.array([1., -1., 1., -1.]).reshape(a.shape)
        cancelled = f.cam_statistics(a, g)
        self.assertEqual(cancelled["gap_weights"], [0.])
        self.assertEqual(cancelled["diagnostic"]["gap_spatial_retention"]["value"], 0.)
        self.assertFalse(cancelled["diagnostic"]["relu_erases_negative_cam"])
        negative = f.cam_statistics(a, -np.ones_like(a))
        self.assertEqual(negative["gap_weights"], [-1.])
        self.assertEqual(negative["diagnostic"]["gap_spatial_retention"]["value"], 1.)
        self.assertTrue(negative["diagnostic"]["relu_erases_negative_cam"])
        self.assertEqual(negative["cam_pre_relu"]["mean"], -1.)
        self.assertEqual(negative["cam_post_relu"]["l2_norm"], 0.)
        zero = f.cam_statistics(a, np.zeros_like(a))
        self.assertIsNone(zero["diagnostic"]["gap_spatial_retention"]["value"])
        self.assertFalse(zero["diagnostic"]["gap_spatial_retention"]["valid"])
        self.assertTrue(zero["diagnostic"]["gradient_exactly_zero"])
        channels = f.cam_statistics(np.ones((2, 2, 2)), np.tile([1., -1.], (2, 2, 1)))
        self.assertEqual(channels["diagnostic"]["channel_cam_retention"]["value"], 0.)
        self.assertEqual(channels["diagnostic"]["gap_spatial_retention"]["value"], 1.)
        self.assertEqual(channels["feature_spatial_variance_per_channel"], [0., 0.])
        positive = f.cam_statistics(a, np.ones_like(a))
        self.assertFalse(positive["diagnostic"]["relu_erases_negative_cam"])
        self.assertEqual(positive["cam_post_relu"]["mean"], 1.)
        for aa, gg in [(a, g[:1]), (a, np.full_like(a, np.nan))]:
            self.assertFalse(f.cam_statistics(aa, gg)["valid"])

    def test_historical_audit_counts_and_no_gradient_claim(self):
        self.assertTrue(hasattr(f, "historical_audit"), "historical_audit is not implemented")
        result = f.historical_audit()
        self.assertEqual(result["geometry_rows"], 132)
        self.assertEqual(result["flagged_rows"], 6)
        self.assertEqual(result["flagged_sample_ids"], ["NNHSCAD5999_6", "NNHSCAD5999_9"])
        self.assertEqual(len(result["samples"]), 11)
        self.assertFalse(result["direct_gradient_probe"]["valid"])
        self.assertIsNone(result["direct_gradient_probe"]["statistics"])
        self.assertNotIn("tensorflow", sys.modules)

    def test_probe_requires_authorization_before_tensorflow_import(self):
        self.assertTrue(hasattr(f, "gpu_probe"), "gpu_probe is not implemented")
        with self.assertRaisesRegex(RuntimeError, "lead authorization"):
            f.gpu_probe(None, None)
        self.assertNotIn("tensorflow", sys.modules)

    def test_manifest_blocks_unvalidated_preprocessing(self):
        self.assertTrue(hasattr(f, "validate_manifest"), "validate_manifest is not implemented")
        with self.assertRaisesRegex(ValueError, "B prediction validation"):
            f.validate_manifest({})
        with self.assertRaisesRegex(ValueError, "nearest"):
            f.validate_manifest({"b_prediction_validation_passed": True, "preprocessing": "bicubic"})
        self.assertNotIn("tensorflow", sys.modules)

if __name__ == "__main__":
    unittest.main(verbosity=2)
