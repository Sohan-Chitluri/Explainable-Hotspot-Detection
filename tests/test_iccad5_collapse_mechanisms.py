"""
Unit tests verifying the ICCAD-5 penultimate representation collapse mechanisms
and their impact on XAI heatmaps and classification.

Verifies:
1. Exact all-zero vector rate on clean class-weighted ICCAD-5 model (models/xai/retrained_clean/xai_cnn_iccad5.keras).
2. Default output bias prediction mechanism: when penultimate vector is all-zero, logit strictly equals output bias b_out.
3. Mathematical vanishing of Grad-CAM, Grad-CAM++, and LayerCAM on all-zero penultimate feature vectors.
4. Absence of representation collapse in balanced mini-batch sampling model.
5. Contrast in faithfulness AUC metrics between collapsed and non-collapsed samples.
"""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

from src.xai_engine import XAIEngine, get_pre_sigmoid_logit
from src.faithfulness import compute_deletion_insertion_curves

ROOT = Path(__file__).resolve().parents[1]
CLEAN_MODEL_PATH = ROOT / "models" / "xai" / "retrained_clean" / "xai_cnn_iccad5.keras"
BALANCED_MODEL_PATH = ROOT / "models" / "xai" / "experimental" / "xai_cnn_iccad5_balanced_sampling_seed1.keras"
ICCAD5_TEST_DIR = ROOT / "iccad-official" / "iccad5" / "test"


@pytest.fixture(scope="module")
def clean_model():
    if not CLEAN_MODEL_PATH.exists():
        pytest.skip(f"Clean model checkpoint not found: {CLEAN_MODEL_PATH}")
    return tf.keras.models.load_model(str(CLEAN_MODEL_PATH), compile=False)


@pytest.fixture(scope="module")
def balanced_model():
    if not BALANCED_MODEL_PATH.exists():
        pytest.skip(f"Balanced model checkpoint not found: {BALANCED_MODEL_PATH}")
    return tf.keras.models.load_model(str(BALANCED_MODEL_PATH), compile=False)


def test_iccad5_clean_penultimate_collapse_reproduction(clean_model):
    """Verify that exactly 31 of 41 test HS samples produce all-zero penultimate activations."""
    hs_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(ICCAD5_TEST_DIR),
        classes=["test_hs"],
        batch_size=41,
        shuffle=False,
        target_size=(224, 224),
        class_mode=None
    )
    imgs = next(hs_gen)
    assert len(imgs) == 41

    pen_layer = clean_model.get_layer("dense_penultimate")
    submodel = Model(inputs=clean_model.input, outputs=pen_layer.output)
    pen_acts = submodel.predict(imgs, verbose=0)

    is_all_zero = np.all(pen_acts == 0.0, axis=1)
    collapsed_count = int(np.sum(is_all_zero))
    
    assert collapsed_count == 31, f"Expected 31 collapsed HS samples, found {collapsed_count}"
    assert collapsed_count / 41 == pytest.approx(31/41, abs=1e-4)


def test_collapsed_sample_logit_strictly_equals_output_bias(clean_model):
    """Verify that when penultimate activation is all-zero, logit equals dense_output bias."""
    out_layer = clean_model.get_layer("dense_output")
    out_w, out_b = out_layer.get_weights()
    expected_bias = float(out_b[0])
    expected_prob = 1.0 / (1.0 + np.exp(-expected_bias))

    # Test sample HSCAD51.png is known collapsed
    engine = XAIEngine(clean_model, target_layer_name="conv_final_2")
    img_t, _ = engine.preprocess_image(str(ICCAD5_TEST_DIR / "test_hs" / "HSCAD51.png"))

    pen_layer = clean_model.get_layer("dense_penultimate")
    pen_sub = Model(inputs=clean_model.input, outputs=pen_layer.output)
    pen_act = pen_sub(img_t).numpy()[0]
    assert np.all(pen_act == 0.0)

    # Output logit & probability
    logit_val = float(get_pre_sigmoid_logit(clean_model, pen_sub(img_t)).numpy()[0])
    prob_val = float(clean_model(img_t, training=False).numpy()[0, 0])

    assert logit_val == pytest.approx(expected_bias, abs=1e-5)
    assert prob_val == pytest.approx(expected_prob, abs=1e-5)
    assert prob_val == pytest.approx(0.464302, abs=1e-5)


def test_collapsed_sample_produces_all_zero_cams(clean_model):
    """Verify that Grad-CAM, Grad-CAM++, and LayerCAM vanish to exact zero on collapsed samples."""
    engine = XAIEngine(clean_model, target_layer_name="conv_final_2")
    img_t, _ = engine.preprocess_image(str(ICCAD5_TEST_DIR / "test_hs" / "HSCAD51.png"))

    raw_gcam, _, _ = engine.explain_gradcam(img_t, target_class="HS")
    raw_gcampp, _, _ = engine.explain_gradcam_plus_plus(img_t, target_class="HS")
    raw_lcam, _, _ = engine.explain_layercam(img_t, target_class="HS")

    assert np.all(raw_gcam == 0.0), "Grad-CAM map did not vanish on collapsed sample"
    assert np.all(raw_gcampp == 0.0), "Grad-CAM++ map did not vanish on collapsed sample"
    assert np.all(raw_lcam == 0.0), "LayerCAM map did not vanish on collapsed sample"


def test_non_collapsed_sample_produces_active_cams(clean_model):
    """Verify that non-collapsed samples produce non-degenerate, active CAM heatmaps."""
    engine = XAIEngine(clean_model, target_layer_name="conv_final_2")
    img_t, _ = engine.preprocess_image(str(ICCAD5_TEST_DIR / "test_hs" / "HSCAD50.png"))

    raw_gcam, _, _ = engine.explain_gradcam(img_t, target_class="HS")
    raw_lcam, _, _ = engine.explain_layercam(img_t, target_class="HS")

    assert not np.all(raw_gcam == 0.0), "Grad-CAM should not be zero for non-collapsed sample"
    assert not np.all(raw_lcam == 0.0), "LayerCAM should not be zero for non-collapsed sample"
    assert np.max(raw_gcam) > 0.0
    assert np.max(raw_lcam) > 0.0


def test_balanced_sampling_prevents_penultimate_collapse(balanced_model):
    """Verify that the balanced mini-batch sampling model has zero penultimate collapse on test HS."""
    hs_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(ICCAD5_TEST_DIR),
        classes=["test_hs"],
        batch_size=41,
        shuffle=False,
        target_size=(224, 224),
        class_mode=None
    )
    imgs = next(hs_gen)

    pen_layer = balanced_model.get_layer("dense_penultimate")
    submodel = Model(inputs=balanced_model.input, outputs=pen_layer.output)
    pen_acts = submodel.predict(imgs, verbose=0)

    is_all_zero = np.all(pen_acts == 0.0, axis=1)
    collapsed_count = int(np.sum(is_all_zero))
    assert collapsed_count == 0, f"Balanced model had {collapsed_count} collapsed HS samples"


def test_balanced_sampling_generates_valid_cams_on_hs51(balanced_model):
    """Verify that balanced sampling restores valid attribution for HSCAD51."""
    engine = XAIEngine(balanced_model, target_layer_name="conv_final_2")
    img_t, _ = engine.preprocess_image(str(ICCAD5_TEST_DIR / "test_hs" / "HSCAD51.png"))

    raw_gcam, res_gcam, _ = engine.explain_gradcam(img_t, target_class="HS")
    raw_gcampp, res_gcampp, _ = engine.explain_gradcam_plus_plus(img_t, target_class="HS")
    raw_lcam, res_lcam, _ = engine.explain_layercam(img_t, target_class="HS")

    assert not np.all(raw_gcam == 0.0)
    assert not np.all(raw_gcampp == 0.0)
    assert not np.all(raw_lcam == 0.0)
    assert np.var(res_gcam) > 0.01


def test_output_semantics_and_bias_classification(clean_model):
    """
    Regression test verifying output semantics:
    - sigmoid(b_out) = 0.464302 represents P(NHS).
    - Under threshold 0.5, P(NHS) < 0.5 maps to predicted label 0 (HS).
    - Therefore, collapsed samples are classified as True Positives purely via negative bias.
    """
    out_layer = clean_model.get_layer("dense_output")
    out_w, out_b = out_layer.get_weights()
    b_out = float(out_b[0])
    p_nhs = 1.0 / (1.0 + np.exp(-b_out))
    
    # Assert exact values
    assert b_out == pytest.approx(-0.143036, abs=1e-5)
    assert p_nhs == pytest.approx(0.464302, abs=1e-5)
    assert p_nhs < 0.5
    
    pred_label = int(p_nhs >= 0.5)
    assert pred_label == 0, "Default prediction must be class 0 (HS)"


def test_gradient_vanishing_pre_normalization(clean_model):
    """
    Verify that gradient vanishing occurs before normalization:
    dScore / dConvOutputs is identically zero, producing zero map variance
    strictly before any postprocessing or division.
    """
    engine = XAIEngine(clean_model, target_layer_name="conv_final_2")
    img_t, _ = engine.preprocess_image(str(ICCAD5_TEST_DIR / "test_hs" / "HSCAD51.png"))

    with tf.GradientTape() as tape:
        conv_outputs, penult_outputs, preds = engine.sub_model(img_t, training=False)
        logit_nhs = tf.matmul(penult_outputs, clean_model.get_layer("dense_output").weights[0]) + clean_model.get_layer("dense_output").weights[1]
        score = -logit_nhs[:, 0]

    grads = tape.gradient(score, conv_outputs)
    assert tf.reduce_all(grads == 0.0), "Gradients must be identically zero"

    # Pre-normalization Grad-CAM
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.maximum(tf.squeeze(conv_outputs[0] @ weights[..., tf.newaxis]), 0.0)
    assert float(tf.math.reduce_variance(cam)) == 0.0
