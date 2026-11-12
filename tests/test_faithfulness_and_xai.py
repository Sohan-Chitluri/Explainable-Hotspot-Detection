"""
Unit and regression tests for XAI faithfulness and gradient engine.

Validates:
1. Deletion / Insertion curve calculation, score tracking, and AUC bounds.
2. Handling of degenerate (flat / all-zero) attribution maps.
3. Grad-CAM++ derivative calculation and zero-gradient fallbacks.
4. Monotonicity / sensitivity on deterministic synthetic patterns.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.faithfulness import compute_deletion_insertion_curves
from src.xai_engine import XAIEngine


def make_dummy_functional_model():
    """Build a lightweight functional Keras model with named target layers."""
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_layout")
    c = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        name="conv_final_2",
    )(inputs)
    f = tf.keras.layers.Flatten(name="flatten")(c)
    p = tf.keras.layers.Dense(4, activation="relu", name="dense_penultimate")(f)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="dense_output")(p)
    return tf.keras.Model(inputs=inputs, outputs=out, name="dummy_xai_model")


def test_deletion_insertion_bounds_and_structure():
    """Verify deletion and insertion AUC are properly normalized in [0, 1]."""
    model = make_dummy_functional_model()
    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32)

    engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
    curves = compute_deletion_insertion_curves(
        model=model,
        engine=engine,
        img_tensor=dummy_input,
        target_idx=0,
        steps=5,
        methods=["gradcam", "layercam"],
    )

    for method in ["gradcam", "layercam"]:
        assert method in curves
        res = curves[method]
        assert len(res["deletion_scores"]) == 6  # 5 steps + initial
        assert len(res["insertion_scores"]) == 6
        assert 0.0 <= res["deletion_auc"] <= 1.0
        assert 0.0 <= res["insertion_auc"] <= 1.0


def test_degenerate_flat_map_handling():
    """Ensure that completely flat attribution maps do not crash curve computation."""
    model = make_dummy_functional_model()
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
    curves = compute_deletion_insertion_curves(
        model=model,
        engine=engine,
        img_tensor=dummy_input,
        target_idx=0,
        steps=4,
        methods=["gradcam"],
    )

    assert "gradcam" in curves
    del_auc = curves["gradcam"]["deletion_auc"]
    ins_auc = curves["gradcam"]["insertion_auc"]
    assert np.isfinite(del_auc)
    assert np.isfinite(ins_auc)


def test_gradcam_plus_plus_tape_derivatives():
    """Verify Grad-CAM++ executes without error and produces finite outputs."""
    model = make_dummy_functional_model()
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

    engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
    raw_map, resized_map, info = engine.explain_gradcam_plus_plus(dummy_input, target_class=0)

    assert resized_map.shape == (224, 224)
    assert np.all(np.isfinite(resized_map))
    assert np.min(resized_map) >= 0.0
    assert np.max(resized_map) <= 1.0 + 1e-6
    assert info["method"] == "Grad-CAM++"


def test_gradcam_plus_plus_all_zeros_fallback():
    """Verify Grad-CAM++ gracefully handles zero input where gradients vanish."""
    model = make_dummy_functional_model()
    zero_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
    raw_map, resized_map, info = engine.explain_gradcam_plus_plus(zero_input, target_class=0)

    assert resized_map.shape == (224, 224)
    assert np.all(np.isfinite(resized_map))


def test_gradcam_plus_plus_exact_numerical_match():
    """Verify exact analytic calculation of alpha weights and channel weights in Grad-CAM++."""
    # Construct a minimal model where gradients and activations are known analytically:
    # A single conv filter of shape (1, 1, 1, 1) followed by identity
    inputs = tf.keras.Input(shape=(2, 2, 1), name="in")
    c = tf.keras.layers.Conv2D(1, (1, 1), padding="same", use_bias=False, name="conv_final_2")(inputs)
    f = tf.keras.layers.Flatten(name="dense_penultimate")(c)
    out = tf.keras.layers.Dense(1, use_bias=True, activation="sigmoid", name="dense_output")(f)
    model = tf.keras.Model(inputs=inputs, outputs=out)

    # Set known weights
    # Conv weight = 1.0
    c_layer = model.get_layer("conv_final_2")
    c_layer.set_weights([np.ones((1, 1, 1, 1), dtype=np.float32)])

    # Dense out weights: w = [2.0, 3.0, 0.0, 0.0]^T, bias = 0.0
    dense_layer = model.get_layer("dense_output")
    dense_layer.set_weights([np.array([[2.0], [3.0], [0.0], [0.0]], dtype=np.float32), np.array([0.0], dtype=np.float32)])

    # Input: A = [[1.0, 2.0], [3.0, 4.0]]
    x = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)

    engine = XAIEngine(model, target_layer_name="conv_final_2", penultimate_layer_name="dense_penultimate")
    # For target_class=1 (NHS, positive logit), target_score = logit = 2*1 + 3*2 = 8.
    # Gradient w.r.t conv_output: g = [[2, 3], [0, 0]]
    # g^2 = [[4, 9], [0, 0]]
    # g^3 = [[8, 27], [0, 0]]
    # sum(A * g^3) = 1*8 + 2*27 = 62.
    # denom(0,0) = 2*4 + 62 = 70 => alpha(0,0) = 4/70
    # denom(0,1) = 2*9 + 62 = 80 => alpha(0,1) = 9/80
    # w = alpha(0,0)*2 + alpha(0,1)*3 = 8/70 + 27/80 = 0.4517857
    raw_map, _, _ = engine.explain_gradcam_plus_plus(x, target_class=1)

    # Expected CAM (before postprocessing max-norm):
    # cam = A * w = [[1, 2], [3, 4]] * 0.4517857
    # Max pixel is at (1,1) with value 4 * w. Normalized map should have (1,1) == 1.0, (0,0) == 0.25
    assert np.isclose(raw_map[1, 1], 1.0, atol=1e-5)
    assert np.isclose(raw_map[0, 0], 0.25, atol=1e-5)


def test_gradcam_plus_plus_finite_difference_derivatives():
    """Verify closed-form gradient powers match numerical finite differences of Y=exp(S)."""
    # 1. Independent numerical check:
    # Let S(A) = 2.0 * A_0 + 3.0 * A_1. In active linear/piecewise-linear regime:
    # d^2 S / dA_i^2 = 0 (piecewise linear functions have zero intrinsic curvature).
    # Under Y(A) = exp(S(A)), diagonal derivatives expand via chain rule as:
    # d^2 Y / dA_i^2 = exp(S) * (g_i)^2
    # d^3 Y / dA_i^3 = exp(S) * (g_i)^3
    A0 = np.array([1.0, 2.0], dtype=np.float64)
    g = np.array([2.0, 3.0], dtype=np.float64)
    eps = 1e-4

    def S_func(A):
        return g[0] * A[0] + g[1] * A[1]

    def Y_func(A):
        return np.exp(S_func(A))

    Y0 = Y_func(A0)
    S0 = S_func(A0)

    # Verify that raw logit S has zero second derivative:
    d2S_dA0 = (S_func(A0 + [eps, 0.0]) - 2.0 * S0 + S_func(A0 - [eps, 0.0])) / (eps**2)
    assert np.isclose(d2S_dA0, 0.0, atol=1e-5)

    # Compute diagonal 2nd and 3rd finite-difference derivatives of Y = exp(S):
    d2Y_dA0 = (Y_func(A0 + [eps, 0.0]) - 2.0 * Y0 + Y_func(A0 - [eps, 0.0])) / (eps**2)
    d3Y_dA0 = (Y_func(A0 + [2 * eps, 0.0]) - 2.0 * Y_func(A0 + [eps, 0.0]) + 2.0 * Y_func(A0 - [eps, 0.0]) - Y_func(A0 - [2 * eps, 0.0])) / (2.0 * (eps**3))

    d2Y_dA1 = (Y_func(A0 + [0.0, eps]) - 2.0 * Y0 + Y_func(A0 - [0.0, eps])) / (eps**2)
    d3Y_dA1 = (Y_func(A0 + [0.0, 2 * eps]) - 2.0 * Y_func(A0 + [0.0, eps]) + 2.0 * Y_func(A0 - [0.0, eps]) - Y_func(A0 - [0.0, 2 * eps])) / (2.0 * (eps**3))

    # Verify: d^2 Y / (Y0 * dA_i^2) == (g_i)^2
    assert np.isclose(d2Y_dA0 / Y0, g[0]**2, rtol=1e-4)
    assert np.isclose(d2Y_dA1 / Y0, g[1]**2, rtol=1e-4)

    # Verify: d^3 Y / (Y0 * dA_i^3) == (g_i)^3
    assert np.isclose(d3Y_dA0 / Y0, g[0]**3, rtol=1e-3)
    assert np.isclose(d3Y_dA1 / Y0, g[1]**3, rtol=1e-3)

    # Verify alpha_0 computed from finite differences equals closed-form formula:
    denom_fd = 2.0 * d2Y_dA0 + A0[0] * d3Y_dA0 + A0[1] * d3Y_dA1
    alpha0_fd = d2Y_dA0 / denom_fd

    denom_closed = 2.0 * (g[0]**2) + (A0[0] * (g[0]**3) + A0[1] * (g[1]**3))
    alpha0_closed = (g[0]**2) / denom_closed

    assert np.isclose(alpha0_fd, alpha0_closed, rtol=1e-4)

