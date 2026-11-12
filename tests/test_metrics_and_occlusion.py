"""
Unit tests for core metrics and OcclusionSensitivity.

Validates:
1. binary_metrics:
   - Empty inputs
   - All-zero and all-one predictions
   - Highly imbalanced confusion-matrix edge cases
   - Numerical bounds and mathematical consistency
2. plot_balanced_accuracy_curves & BestBalancedAccuracyCheckpoint:
   - Zero positive cases (guard against ZeroDivisionError)
   - Zero predicted positive cases
   - Hotspot (class 0) sensitivity vs specificity mapping
   - History curve extraction and output files
3. OcclusionSensitivity:
   - Images smaller than the occlusion window
   - Boundary tiling and matching spatial dimensions
   - Finite, correctly shaped, [0, 1]-bounded normalized maps
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from src.metrics import (
    binary_metrics,
    plot_balanced_accuracy_curves,
    BestBalancedAccuracyCheckpoint,
)
from src.gradcam import GradCAM
from src.occlusion import OcclusionSensitivity
from src.xai_engine import XAIEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_binary_model():
    """Lightweight fully-convolutional 2D CNN accepting arbitrary image sizes."""
    inputs = tf.keras.Input(shape=(None, None, 3), name="input_layout")
    x = tf.keras.layers.Conv2D(
        filters=2,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv_final_2",
    )(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(4, activation="relu", name="dense_penultimate")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="dummy_model")


# ---------------------------------------------------------------------------
# 1. binary_metrics Tests
# ---------------------------------------------------------------------------

class TestBinaryMetrics:
    def test_empty_inputs(self):
        m = binary_metrics([], [], positive_label=0)
        assert m["confusion_matrix"] == {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        assert m["precision"] == 0.0
        assert m["recall_sensitivity"] == 0.0
        assert m["specificity"] == 0.0
        assert m["f1_score"] == 0.0
        assert m["balanced_accuracy"] == 0.0
        assert m["accuracy"] == 0.0
        assert m["positive_label"] == 0

    def test_all_zero_predictions(self):
        # 4 HS (0) and 4 NHS (1), all predicted as 0 (HS)
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 0, 0, 0, 0, 0]
        m = binary_metrics(y_true, y_pred, positive_label=0)
        assert m["confusion_matrix"] == {"tp": 4, "tn": 0, "fp": 4, "fn": 0}
        assert m["recall_sensitivity"] == 1.0  # 4/4
        assert m["specificity"] == 0.0         # 0/4
        assert m["precision"] == 0.5           # 4/8
        assert m["balanced_accuracy"] == 0.5   # 0.5 * (1.0 + 0.0)

    def test_all_one_predictions(self):
        # 4 HS (0) and 4 NHS (1), all predicted as 1 (NHS)
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [1, 1, 1, 1, 1, 1, 1, 1]
        m = binary_metrics(y_true, y_pred, positive_label=0)
        assert m["confusion_matrix"] == {"tp": 0, "tn": 4, "fp": 0, "fn": 4}
        assert m["recall_sensitivity"] == 0.0  # 0/4
        assert m["specificity"] == 1.0         # 4/4
        assert m["precision"] == 0.0           # 0/(0+0) guarded to 0.0
        assert m["balanced_accuracy"] == 0.5

    def test_no_actual_positives(self):
        # Only class 1 present
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 0]
        m = binary_metrics(y_true, y_pred, positive_label=0)
        assert m["confusion_matrix"] == {"tp": 0, "tn": 2, "fp": 2, "fn": 0}
        assert m["recall_sensitivity"] == 0.0  # 0/0 guarded
        assert m["specificity"] == 0.5         # 2/4
        assert m["balanced_accuracy"] == 0.25

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        m = binary_metrics(y_true, y_pred, positive_label=0)
        assert m["precision"] == 1.0
        assert m["recall_sensitivity"] == 1.0
        assert m["specificity"] == 1.0
        assert m["f1_score"] == 1.0
        assert m["balanced_accuracy"] == 1.0
        assert m["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# 2. Balanced Accuracy & Checkpoint Tests
# ---------------------------------------------------------------------------

class TestBalancedAccuracyCurvesAndCallback:
    def test_zero_positives_does_not_crash_plotting(self, tmp_path):
        out_png = tmp_path / "balanced_accuracy.png"
        # History where true positives or true negatives are 0
        history = {
            "true_positives": [0, 0],
            "true_negatives": [0, 10],
            "false_positives": [0, 0],
            "false_negatives": [0, 0],
            "val_true_positives": [0, 5],
            "val_true_negatives": [0, 0],
            "val_false_positives": [0, 0],
            "val_false_negatives": [0, 0],
        }
        train_bal, val_bal = plot_balanced_accuracy_curves(
            history, out_png, title="Test Curve"
        )
        assert out_png.exists()
        assert len(train_bal) == 2
        assert len(val_bal) == 2
        # Epoch 0 has all 0s, bal_acc should be 0.0 without ZeroDivisionError
        assert train_bal[0] == 0.0
        assert val_bal[0] == 0.0

    def test_sensitivity_maps_to_hotspot_class(self, tmp_path):
        out_png = tmp_path / "test_sens.png"
        # In Keras binary metrics:
        # Class 0 is HS, Class 1 is NHS.
        # Keras TrueNegatives = HS predicted HS
        # Keras FalsePositives = HS predicted NHS
        # If tn=8, fp=2 -> HS Recall = 8/10 = 0.8
        # If tp=10, fn=0 -> NHS Recall = 10/10 = 1.0
        history = {
            "true_positives": [10],
            "true_negatives": [8],
            "false_positives": [2],
            "false_negatives": [0],
            "val_true_positives": [10],
            "val_true_negatives": [8],
            "val_false_positives": [2],
            "val_false_negatives": [0],
        }
        train_bal, val_bal = plot_balanced_accuracy_curves(history, out_png, title="Sens Test")
        # Balanced accuracy = 0.5 * (0.8 + 1.0) = 0.9
        assert abs(train_bal[0] - 0.9) < 1e-6
        assert abs(val_bal[0] - 0.9) < 1e-6

    def test_checkpoint_callback_best_selection(self):
        cb = BestBalancedAccuracyCheckpoint()
        cb.model = make_dummy_binary_model()
        initial_weights = [w.copy() for w in cb.model.get_weights()]

        # Epoch 0: val HS recall = 0.5, val NHS recall = 0.5 -> bal_acc = 0.5
        cb.on_epoch_end(
            0,
            logs={
                "val_true_negatives": 5,   # HS TP
                "val_false_positives": 5,  # HS FN
                "val_true_positives": 50,  # NHS TN
                "val_false_negatives": 50, # NHS FP
            },
        )
        assert cb.best_epoch == 0
        assert abs(cb.best_value - 0.5) < 1e-6

        # Epoch 1: val HS recall = 0.8 (tn=8, fp=2), val NHS recall = 1.0 (tp=100, fn=0)
        # bal_acc = 0.5 * (0.8 + 1.0) = 0.9 -> new best
        cb.on_epoch_end(
            1,
            logs={
                "val_true_negatives": 8,
                "val_false_positives": 2,
                "val_true_positives": 100,
                "val_false_negatives": 0,
            },
        )
        assert cb.best_epoch == 1
        assert abs(cb.best_value - 0.9) < 1e-6

        # Epoch 2: bal_acc = 0.7 -> should not overwrite best_epoch 1
        cb.on_epoch_end(
            2,
            logs={
                "val_true_negatives": 4,
                "val_false_positives": 6,
                "val_true_positives": 100,
                "val_false_negatives": 0,
            },
        )
        assert cb.best_epoch == 1
        assert abs(cb.best_value - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# 3. OcclusionSensitivity Tests
# ---------------------------------------------------------------------------

class TestOcclusionSensitivity:
    def test_image_smaller_than_occlusion_window(self):
        model = make_dummy_binary_model()
        engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
        # Standard window is 32x32, input here is 16x16
        small_occ = OcclusionSensitivity(engine, window_size=(32, 32), stride=(16, 16))
        small_input = np.ones((1, 16, 16, 3), dtype=np.float32)

        res = small_occ.explain(small_input, target_class=0)
        assert res["raw_map"].shape == (16, 16)
        assert res["norm_map"].shape == (16, 16)
        assert np.all(np.isfinite(res["raw_map"]))
        assert np.all(np.isfinite(res["norm_map"]))
        assert 0.0 <= np.max(res["norm_map"]) <= 1.0

    def test_boundary_tiling_standard_input(self):
        model = make_dummy_binary_model()
        engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
        occ = OcclusionSensitivity(engine, window_size=(32, 32), stride=(16, 16))
        standard_input = np.ones((1, 224, 224, 3), dtype=np.float32)

        res = occ.explain(standard_input, target_class=0)
        assert res["raw_map"].shape == (224, 224)
        assert res["norm_map"].shape == (224, 224)
        assert np.all(np.isfinite(res["raw_map"]))
        assert np.all(np.isfinite(res["norm_map"]))
        assert 0.0 <= np.min(res["norm_map"])
        assert np.max(res["norm_map"]) <= 1.0 + 1e-6
        assert "centroid_x" in res
        assert "fraction_above_050" in res


# ---------------------------------------------------------------------------
# 4. GradCAM Submodel Re-use and Layer Resolution Tests
# ---------------------------------------------------------------------------

class TestGradCAMSubmodelReuse:
    def test_gradcam_submodel_reused_without_reconstruction(self):
        model = make_dummy_binary_model()
        gcam = GradCAM(model, target_layer_name="conv_final_2")
        initial_submodel = gcam.sub_model

        img = np.ones((1, 224, 224, 3), dtype=np.float32)
        raw1, res1, info1 = gcam.compute_heatmap(img, target_class="HS")
        raw2, res2, info2 = gcam.compute_heatmap(img, target_class="NHS")

        # Submodel reference must remain identical across calls
        assert gcam.sub_model is initial_submodel
        assert raw1.shape == (224, 224)
        assert res1.shape == (224, 224)
        assert np.all(np.isfinite(res1))
        assert np.all(np.isfinite(res2))
        assert info1["target_class"] == "HS"
        assert info2["target_class"] == "NHS"
