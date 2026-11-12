"""Characterization guard: corrected nearest preprocessing must differ from legacy bicubic.

Complements tests/test_xai_preprocessing.py (the equivalence RED test). This test would
fail if someone silently reverted the fix to bicubic, by asserting the two paths produce
different tensors on a known fine-grained raster.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.xai_engine import XAIEngine


class LegacyVsCorrectedTest(unittest.TestCase):
    def test_corrected_tensor_differs_from_legacy_bicubic_on_fine_raster(self):
        rows, cols = np.indices((1200, 1200))
        raster = (((cols % 31) < 7) | ((rows % 47) < 5)).astype(np.uint8) * 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.png"
            Image.fromarray(raster).save(path)
            corrected, _ = XAIEngine.preprocess_image(path)
            with Image.open(path) as source:
                legacy = np.asarray(
                    source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC),
                    dtype=np.uint8,
                )
        legacy_tensor = np.expand_dims(legacy.astype(np.float32) / 255.0, axis=0)
        self.assertFalse(np.array_equal(corrected, legacy_tensor))
        self.assertGreater(int(np.count_nonzero(corrected[0] != legacy_tensor[0])), 0)


if __name__ == "__main__":
    unittest.main()
