"""
Occlusion-based Sensitivity and Attribution for Lithography Hotspot Detection.

Implements perturbation-based spatial importance mapping:
  - Systematically occludes spatial regions of layout images using sliding windows.
  - Measures the drop in pre-sigmoid logit target score (importance = S_orig - S_occluded).
  - Accumulates overlapping window contributions with precise coverage-count normalization.
  - Preserves raw signed numerical sensitivity scores alongside normalized visual heatmaps.
  - Computes quantitative spatial diagnostics and comparative correlation metrics against CAM methods.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf

from src.xai_diagnostics import compute_attribution_diagnostics
from src.xai_engine import XAIEngine, get_pre_sigmoid_logit


def compute_map_comparison(
    map_a: np.ndarray,
    map_b: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute rigorous mathematical comparison metrics between two normalized 2D heatmaps.

    Metrics:
      - pearson_corr: Linear correlation between flattened 2D spatial attributions.
      - spearman_corr: Monotonic rank correlation between spatial attributions.
      - cosine_sim: Cosine similarity between attribution vectors.
      - iou_overlap: Intersection-over-Union between binary masks at the given activation threshold.
    """
    a = np.clip(np.asarray(map_a, dtype=np.float32), 0.0, 1.0).ravel()
    b = np.clip(np.asarray(map_b, dtype=np.float32), 0.0, 1.0).ravel()

    std_a = np.std(a)
    std_b = np.std(b)

    if std_a < 1e-8 or std_b < 1e-8:
        p_val = 0.0
        s_val = 0.0
    else:
        p_val = float(pearsonr(a, b)[0])
        s_val = float(spearmanr(a, b)[0])
        if np.isnan(p_val):
            p_val = 0.0
        if np.isnan(s_val):
            s_val = 0.0

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(a, b) / (norm_a * norm_b))

    mask_a = (map_a >= threshold)
    mask_b = (map_b >= threshold)
    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    iou = float(intersection / union) if union > 0 else 0.0

    return {
        "pearson_corr": round(p_val, 4),
        "spearman_corr": round(s_val, 4),
        "cosine_sim": round(cos_sim, 4),
        "iou_at_50": round(iou, 4),
    }


class OcclusionSensitivity:
    """
    Perturbation-based explainability engine for binary CNN classifiers.
    """

    def __init__(
        self,
        engine: XAIEngine,
        window_size: Tuple[int, int] = (32, 32),
        stride: Tuple[int, int] = (16, 16),
        baseline_value: float = 0.0,
        batch_size: int = 64,
    ):
        """
        Initialize OcclusionSensitivity analyzer.

        Args:
          engine: Initialized XAIEngine with loaded model.
          window_size: (win_h, win_w) spatial occlusion patch dimensions in pixels.
          stride: (stride_y, stride_x) sliding window step sizes in pixels.
          baseline_value: Pixel value assigned to occluded patches (0.0 = empty layout background).
          batch_size: Forward pass evaluation batch size on GPU.
        """
        self.engine = engine
        self.model = engine.model
        self.window_size = window_size
        self.stride = stride
        self.baseline_value = baseline_value
        self.batch_size = batch_size

    def _generate_window_coordinates(self, height: int, width: int) -> List[Tuple[int, int, int, int]]:
        """Compute top-left and bottom-right bounding box coordinates covering the full grid."""
        win_h, win_w = self.window_size
        str_y, str_x = self.stride

        y_coords = list(range(0, height - win_h + 1, str_y))
        x_coords = list(range(0, width - win_w + 1, str_x))

        # Ensure right and bottom borders are fully covered
        if y_coords[-1] + win_h < height:
            y_coords.append(height - win_h)
        if x_coords[-1] + win_w < width:
            x_coords.append(width - win_w)

        coords = []
        for y in y_coords:
            for x in x_coords:
                coords.append((y, x, y + win_h, x + win_w))
        return coords

    def explain(
        self,
        img_tensor: np.ndarray,
        target_class: Union[str, int] = "predicted",
    ) -> Dict[str, Union[np.ndarray, float, int, str, Dict]]:
        """
        Compute occlusion sensitivity importance map for a single input image tensor.

        Args:
          img_tensor: Normalized float32 image tensor of shape (1, 224, 224, 3).
          target_class: 'predicted', 'HS' (0), or 'NHS' (1).

        Returns:
          Dictionary containing raw map, normalized map, importance scores, diagnostics, and metadata.
        """
        t0 = time.perf_counter()
        p_hs, p_nhs, pred_label, pred_idx = self.engine.predict(img_tensor)

        # Resolve target class
        target_idx, target_name = self.engine._resolve_target_idx(img_tensor, target_class)

        # Compute original baseline score (pre-sigmoid logit)
        _, penult_orig, _ = self.engine.sub_model(img_tensor, training=False)
        orig_logit_nhs = float(get_pre_sigmoid_logit(self.model, penult_orig)[0])
        orig_score = -orig_logit_nhs if target_idx == 0 else orig_logit_nhs

        H, W = img_tensor.shape[1], img_tensor.shape[2]
        windows = self._generate_window_coordinates(H, W)
        num_windows = len(windows)

        # Build batch of occluded images
        occluded_list = []
        for y1, x1, y2, x2 in windows:
            occ_img = img_tensor[0].copy()
            occ_img[y1:y2, x1:x2, :] = self.baseline_value
            occluded_list.append(occ_img)

        batch_arr = np.array(occluded_list, dtype=np.float32)

        # Batched forward passes through penultimate feature extractor
        all_occ_scores = []
        for i in range(0, num_windows, self.batch_size):
            chunk = batch_arr[i : i + self.batch_size]
            _, penult_chunk, _ = self.engine.sub_model(chunk, training=False)
            chunk_logits = get_pre_sigmoid_logit(self.model, penult_chunk).numpy()
            chunk_scores = -chunk_logits if target_idx == 0 else chunk_logits
            all_occ_scores.append(chunk_scores)

        occ_scores = np.concatenate(all_occ_scores, axis=0)

        # Importance = Drop in target score caused by occluding the window
        # Positive importance: feature supports the target class (hiding it harms target score).
        # Negative importance: feature suppresses the target class (hiding it helps target score).
        window_importance = orig_score - occ_scores

        # Aggregate overlapping windows onto 2D spatial grid
        accum_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        for (y1, x1, y2, x2), imp_val in zip(windows, window_importance):
            accum_map[y1:y2, x1:x2] += imp_val
            count_map[y1:y2, x1:x2] += 1.0

        raw_map = np.where(count_map > 0, accum_map / count_map, 0.0)

        # Positive attribution map (ReLU of sensitivity drop)
        pos_map = np.maximum(raw_map, 0.0)
        max_pos = float(np.max(pos_map))
        norm_map = pos_map / (max_pos + 1e-10)

        elapsed_sec = time.perf_counter() - t0

        # Quantitative spatial diagnostics on the normalized positive attribution map
        diag = compute_attribution_diagnostics(norm_map)

        # Raw importance statistics
        mean_pos_imp = float(np.mean(pos_map[pos_map > 0])) if np.any(pos_map > 0) else 0.0
        min_raw = float(np.min(raw_map))
        max_raw = float(np.max(raw_map))

        result = {
            "method": "Occlusion",
            "p_hs": p_hs,
            "p_nhs": p_nhs,
            "pred_label": pred_label,
            "pred_idx": pred_idx,
            "target_class": target_name,
            "target_idx": target_idx,
            "orig_target_score": round(orig_score, 4),
            "max_occlusion_importance": round(max_raw, 4),
            "min_occlusion_importance": round(min_raw, 4),
            "mean_positive_importance": round(mean_pos_imp, 4),
            "window_size_h": self.window_size[0],
            "window_size_w": self.window_size[1],
            "stride_y": self.stride[0],
            "stride_x": self.stride[1],
            "baseline_value": self.baseline_value,
            "num_windows": num_windows,
            "runtime_sec": round(elapsed_sec, 3),
            "raw_map": raw_map,
            "norm_map": norm_map,
            **diag,
        }

        return result

    @staticmethod
    def overlay_heatmap(
        raw_rgb: np.ndarray,
        heatmap_224: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Blend colormapped normalized heatmap with original layout image."""
        try:
            cmap = matplotlib.colormaps[colormap]
        except AttributeError:
            cmap = cm.get_cmap(colormap)
        colored_cam = cmap(np.clip(heatmap_224, 0.0, 1.0))[:, :, :3]
        colored_cam_uint8 = np.uint8(colored_cam * 255)
        overlay = (1.0 - alpha) * raw_rgb.astype(np.float32) + alpha * colored_cam_uint8.astype(np.float32)
        return np.clip(overlay, 0, 255).astype(np.uint8)
