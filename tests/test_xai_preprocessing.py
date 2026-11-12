"""Preprocessing regression test; no model construction or inference."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from src.xai_engine import XAIEngine


class XAIPreprocessingTest(unittest.TestCase):
    def test_binary_raster_matches_official_classifier_preprocessing(self):
        # Deterministic fine geometry makes interpolation differences observable.
        rows, cols = np.indices((1200, 1200))
        raster = (((cols % 31) < 7) | ((rows % 47) < 5)).astype(np.uint8) * 255
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.png"
            Image.fromarray(raster).save(path)
            official_image = load_img(path, color_mode="rgb", target_size=(224, 224),
                                      interpolation="nearest")
            official = ImageDataGenerator(rescale=1.0 / 255).standardize(
                img_to_array(official_image, dtype="float32"))
            actual, raw_rgb = XAIEngine.preprocess_image(path)

        self.assertEqual(actual.shape, (1, 224, 224, 3))
        self.assertEqual(actual.dtype, np.dtype("float32"))
        self.assertEqual(raw_rgb.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(raw_rgb, np.asarray(official_image))
        np.testing.assert_array_equal(actual[0], official)


if __name__ == "__main__":
    unittest.main()
