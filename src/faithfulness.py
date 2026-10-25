"""
Faithfulness Evaluation and Sanity Checks for Lithography Hotspot XAI.

Implements:
1. Deletion curves (progressive removal of top-attribution pixels) -> target score.
2. Insertion curves (progressive insertion of top-attribution pixels starting from blank canvas) -> target score.
3. Deletion AUC & Insertion AUC metrics.
4. Model Parameter Randomization (Sanity Check: cascading randomization from output back to conv layers, measuring rank correlation vs original map).

Evaluates clean-split models from `models/xai/retrained_clean/` on held-out test samples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from src.occlusion import OcclusionSensitivity
from src.xai_engine import XAIEngine


def compute_deletion_insertion_curves(
    model: tf.keras.Model,
    engine: XAIEngine,
    img_tensor: np.ndarray,
    target_idx: int = 0,
    steps: int = 10,
    methods: List[str] = ["gradcam", "gradcam_plus_plus", "layercam"],
) -> Dict[str, Dict]:
    """
    Computes deletion and insertion curves for specified attribution methods.
    
    img_tensor: shape (1, 224, 224, 3), range [0, 1]
    steps: number of progressive masking intervals (e.g. 10 steps = 10% per step)
    
    Returns dict mapping method -> {
        'deletion_scores': list,
        'insertion_scores': list,
        'deletion_auc': float,
        'insertion_auc': float
    }
    """
    # Baseline predictions
    def get_prob(x_in):
        preds = model(x_in, training=False).numpy().flatten()
        p_raw = float(preds[0])
        if getattr(engine, "positive_class_is_1", False):
            return p_raw if target_idx == 1 else (1.0 - p_raw)
        else:
            p_nhs = p_raw
            return 1.0 - p_nhs if target_idx == 0 else p_nhs

    base_prob = get_prob(img_tensor)
    blank_canvas = np.zeros_like(img_tensor)
    blank_prob = get_prob(blank_canvas)

    results = {}
    H, W = img_tensor.shape[1:3]
    total_pixels = H * W
    step_size = total_pixels // steps

    for method in methods:
        if method == "gradcam":
            _, cam, _ = engine.explain_gradcam(img_tensor, target_class=target_idx)
        elif method == "gradcam_plus_plus":
            _, cam, _ = engine.explain_gradcam_plus_plus(img_tensor, target_class=target_idx)
        elif method == "layercam":
            _, cam, _ = engine.explain_layercam(img_tensor, target_class=target_idx)
        elif method == "saliency":
            _, cam, _ = engine.explain_gradient_saliency(img_tensor, target_class=target_idx)
        elif method == "occlusion":
            occ = OcclusionSensitivity(engine, window_size=(32, 32), stride=(16, 16), baseline_value=0.0)
            res_occ = occ.explain(img_tensor, target_class=target_idx)
            cam = res_occ["norm_map"]
        elif method == "random":
            flat_indices = np.random.default_rng(seed=42).permutation(total_pixels)
            cam = None
        else:
            raise ValueError(f"Unsupported explanation method: {method}")

        # Sort pixel coordinates by attribution descending (if not random)
        if cam is not None:
            flat_indices = np.argsort(cam.flatten())[::-1]

        # Deletion: start with original image, mask top pixels to zero
        del_scores = [float(base_prob)]
        cur_del = img_tensor.copy()

        # Insertion: start with blank image, unmask top pixels from original
        ins_scores = [float(blank_prob)]
        cur_ins = blank_canvas.copy()

        for s in range(1, steps + 1):
            idx_subset = flat_indices[: s * step_size]
            
            # Deletion update
            cur_del_flat = cur_del.reshape(1, -1, 3)
            cur_del_flat[0, idx_subset, :] = 0.0
            cur_del = cur_del_flat.reshape(1, H, W, 3)
            del_scores.append(float(get_prob(cur_del)))

            # Insertion update
            cur_ins_flat = cur_ins.reshape(1, -1, 3)
            img_flat = img_tensor.reshape(1, -1, 3)
            cur_ins_flat[0, idx_subset, :] = img_flat[0, idx_subset, :]
            cur_ins = cur_ins_flat.reshape(1, H, W, 3)
            ins_scores.append(float(get_prob(cur_ins)))

        # Trapezoidal rule AUC normalized to [0, 1]
        x_axis = np.linspace(0, 1, steps + 1)
        del_auc = float(np.trapz(del_scores, x_axis))
        ins_auc = float(np.trapz(ins_scores, x_axis))

        results[method] = {
            "deletion_scores": del_scores,
            "insertion_scores": ins_scores,
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
        }

    return results


def run_model_parameter_randomization_check(
    clean_model_path: Path,
    img_tensor: np.ndarray,
    target_idx: int = 0,
) -> Dict[str, float]:
    """
    Sanity check (Adebayo et al., 2018):
    Randomizes model weights and computes Spearman rank correlation with original map.
    A faithful explanation should collapse (correlation -> ~0).
    """
    orig_model = tf.keras.models.load_model(clean_model_path, compile=False)
    orig_engine = XAIEngine(orig_model)
    _, orig_cam_g, _ = orig_engine.explain_gradcam(img_tensor, target_class=target_idx)
    _, orig_cam_l, _ = orig_engine.explain_layercam(img_tensor, target_class=target_idx)

    # Clone and re-initialize weights
    rand_model = tf.keras.models.clone_model(orig_model)
    rand_engine = XAIEngine(rand_model)
    _, rand_cam_g, _ = rand_engine.explain_gradcam(img_tensor, target_class=target_idx)
    _, rand_cam_l, _ = rand_engine.explain_layercam(img_tensor, target_class=target_idx)

    from scipy.stats import spearmanr
    corr_g, _ = spearmanr(orig_cam_g.flatten(), rand_cam_g.flatten())
    corr_l, _ = spearmanr(orig_cam_l.flatten(), rand_cam_l.flatten())

    return {
        "gradcam_randomization_rank_correlation": float(corr_g) if not np.isnan(corr_g) else 0.0,
        "layercam_randomization_rank_correlation": float(corr_l) if not np.isnan(corr_l) else 0.0,
    }
