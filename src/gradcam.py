"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for
lithography hotspot detection custom CNN models.

Reference:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


def enable_gpu_memory_growth():
    """Enable memory growth on available GPUs to prevent OOM / full memory lock."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass


class GradCAM:
    """
    Grad-CAM interpreter for sequential / functional Keras models.

    Tailored for binary sigmoid classifiers where:
      - Class 0 (test_hs): Hotspot (score = 1 - p_sigmoid)
      - Class 1 (test_nhs): Non-Hotspot (score = p_sigmoid)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target_layer_name: Optional[str] = None,
    ):
        """
        Initialize GradCAM with a trained Keras model and target layer.

        If target_layer_name is None, automatically finds the last Conv2D layer.
        """
        self.model = model

        if target_layer_name is None:
            # Auto-detect final Conv2D layer
            conv_layers = [
                l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)
            ]
            if not conv_layers:
                raise ValueError("No Conv2D layer found in the model.")
            self.target_layer_name = conv_layers[-1]
        else:
            self.target_layer_name = target_layer_name

        # Verify target layer exists
        try:
            target_layer = self.model.get_layer(self.target_layer_name)
        except ValueError as e:
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found in model: {e}"
            )

        # Build submodel for gradient tracking
        self.grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, self.model.output],
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        target_layer_name: Optional[str] = "conv2d_5",
    ) -> "GradCAM":
        """Load model from .keras checkpoint and return GradCAM instance."""
        enable_gpu_memory_growth()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = tf.keras.models.load_model(str(checkpoint_path))
        return cls(model, target_layer_name=target_layer_name)

    def preprocess_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess image matching baseline pipeline (rescale 1/255).

        Returns:
          - img_array: Normalized float32 batch array of shape (1, H, W, 3)
          - raw_rgb: Original uint8 RGB numpy array of shape (H, W, 3)
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                # Float array in [0, 1]
                pil_img = Image.fromarray((image_input * 255).astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        raw_rgb = np.array(pil_img.resize(target_size), dtype=np.uint8)
        img_array = raw_rgb.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
        return img_tensor, raw_rgb

    def predict(
        self,
        img_tensor: np.ndarray,
    ) -> Tuple[float, float, str, int]:
        """
        Compute model prediction probabilities for a single image tensor.

        Returns:
          - p_hs: Probability of HS (class 0)
          - p_nhs: Probability of NHS (class 1)
          - pred_label: 'HS' or 'NHS'
          - pred_class_idx: 0 (HS) or 1 (NHS)
        """
        preds = self.model(img_tensor, training=False).numpy()
        p_nhs = float(preds[0, 0])
        p_hs = float(1.0 - p_nhs)
        pred_class_idx = 1 if p_nhs >= 0.5 else 0
        pred_label = "NHS" if pred_class_idx == 1 else "HS"
        return p_hs, p_nhs, pred_label, pred_class_idx

    def compute_heatmap(
        self,
        img_tensor: np.ndarray,
        target_class: Union[str, int] = "predicted",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[float, str, int]]]:
        """
        Compute Grad-CAM heatmap for the specified target class.

        Args:
          img_tensor: Preprocessed image tensor of shape (1, 224, 224, 3)
          target_class: 'HS' (0), 'NHS' (1), or 'predicted'

        Returns:
          - heatmap_raw: 2D float32 array at target layer resolution (e.g. 15x15) in [0, 1]
          - heatmap_resized: 2D float32 array resized to input resolution (224x224) in [0, 1]
          - info: Dict of prediction details and target class explained
        """
        p_hs, p_nhs, pred_label, pred_idx = self.predict(img_tensor)

        # Resolve target class
        if isinstance(target_class, str):
            tc_upper = target_class.upper()
            if tc_upper == "PREDICTED":
                target_idx = pred_idx
                target_name = pred_label
            elif tc_upper == "HS":
                target_idx = 0
                target_name = "HS"
            elif tc_upper == "NHS":
                target_idx = 1
                target_name = "NHS"
            else:
                raise ValueError(f"Unknown target class string: {target_class}")
        else:
            target_idx = int(target_class)
            target_name = "HS" if target_idx == 0 else "NHS"

        # Target score computation using pre-sigmoid logit to avoid vanishing gradients
        # from sigmoid saturation on high-confidence samples (Selvaraju et al., ICCV 2017).
        penultimate_layer = self.model.get_layer("dense")
        dense_1_layer = self.model.get_layer("dense_1")
        w_dense1 = dense_1_layer.weights[0]
        b_dense1 = dense_1_layer.weights[1]

        with tf.GradientTape() as tape:
            # We track target conv layer and penultimate dense layer
            sub_model = tf.keras.models.Model(
                inputs=[self.model.inputs],
                outputs=[self.grad_model.outputs[0], penultimate_layer.output, self.model.output],
            )
            conv_outputs, penult_outputs, predictions = sub_model(img_tensor, training=False)
            
            # Logit z for class NHS (1)
            logit_nhs = tf.matmul(penult_outputs, w_dense1) + b_dense1
            logit_nhs = logit_nhs[:, 0]

            if target_idx == 0:  # HS score = -logit_nhs
                target_score = -logit_nhs
            else:  # NHS score = +logit_nhs
                target_score = logit_nhs

        # Compute gradients d(target_score) / d(conv_outputs)
        grads = tape.gradient(target_score, conv_outputs)
        if grads is None:
            raise RuntimeError("Gradient computation returned None.")

        # Global average pooling of gradients: alpha_k = (1/Z) * sum_i,j (dScore / dA_ij^k)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)

        # Linear combination: sum_k alpha_k * A^k
        conv_out = conv_outputs[0]  # shape: (H, W, channels)
        cam = conv_out @ pooled_grads[..., tf.newaxis]  # shape: (H, W, 1)
        cam = tf.squeeze(cam)  # shape: (H, W)

        # Apply ReLU to capture features with positive contribution
        cam = tf.maximum(cam, 0.0)

        # Normalize to [0, 1]
        max_val = tf.math.reduce_max(cam)
        if max_val > 0:
            cam = cam / (max_val + 1e-10)

        heatmap_raw = cam.numpy().astype(np.float32)

        # Resize to input resolution (224x224) using bilinear interpolation
        heatmap_resized = tf.image.resize(
            cam[..., tf.newaxis],
            (img_tensor.shape[1], img_tensor.shape[2]),
            method="bilinear",
        ).numpy().squeeze().astype(np.float32)

        # Re-normalize resized heatmap to ensure strict [0, 1] bounds
        h_max = float(np.max(heatmap_resized))
        if h_max > 0:
            heatmap_resized = heatmap_resized / h_max

        info = {
            "p_hs": p_hs,
            "p_nhs": p_nhs,
            "pred_label": pred_label,
            "pred_idx": pred_idx,
            "target_class": target_name,
            "target_idx": target_idx,
            "target_layer": self.target_layer_name,
        }

        return heatmap_raw, heatmap_resized, info

    @staticmethod
    def overlay_heatmap(
        raw_rgb: np.ndarray,
        heatmap_224: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        """
        Superimpose colored heatmap on the original layout image.

        Args:
          raw_rgb: uint8 RGB numpy array (224, 224, 3)
          heatmap_224: float32 2D array in [0, 1] (224, 224)
          alpha: overlay blending factor in [0, 1]
          colormap: matplotlib colormap name (default 'jet')

        Returns:
          overlay: uint8 RGB numpy array (224, 224, 3)
        """
        try:
            cmap = matplotlib.colormaps[colormap]
        except AttributeError:
            cmap = cm.get_cmap(colormap)
        colored_cam = cmap(heatmap_224)[:, :, :3]  # drop alpha channel -> float32 [0, 1]
        colored_cam_uint8 = np.uint8(colored_cam * 255)

        # Blend: (1 - alpha) * original + alpha * colored_cam
        overlay = (1.0 - alpha) * raw_rgb.astype(np.float32) + alpha * colored_cam_uint8.astype(np.float32)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay

    @staticmethod
    def compute_spatial_statistics(heatmap_224: np.ndarray) -> Dict[str, float]:
        """
        Compute quantitative spatial activation statistics from a normalized 224x224 heatmap.

        Metrics computed:
          - total_activation: Sum of all pixel activation values
          - mean_activation: Mean activation intensity across layout
          - peak_activation: Maximum activation intensity (usually 1.0 for non-zero heatmap)
          - top10_threshold: Activation value at the 90th percentile (top 10% brightest pixels)
          - top10_activation_fraction: Fraction of total activation mass in the top 10% pixels
          - centroid_x: X-coordinate of activation center of mass [0, 223]
          - centroid_y: Y-coordinate of activation center of mass [0, 223]
          - spatial_spread: Weighted standard deviation (radius) of activation around centroid
          - activated_area_fraction_gt05: Fraction of layout pixels with activation >= 0.5
        """
        h = np.clip(heatmap_224, 0.0, 1.0)
        h_sum = float(np.sum(h))
        total_pixels = float(h.size)

        if h_sum <= 1e-9:
            return {
                "total_activation": 0.0,
                "mean_activation": 0.0,
                "peak_activation": 0.0,
                "top10_threshold": 0.0,
                "top10_activation_fraction": 0.0,
                "centroid_x": float(h.shape[1] / 2.0),
                "centroid_y": float(h.shape[0] / 2.0),
                "spatial_spread": 0.0,
                "activated_area_fraction_gt05": 0.0,
            }

        mean_act = float(np.mean(h))
        peak_act = float(np.max(h))

        # Top 10% highest intensity pixels
        top10_thresh = float(np.percentile(h, 90.0))
        top10_mask = h >= top10_thresh
        top10_mass = float(np.sum(h[top10_mask]))
        top10_fraction = top10_mass / h_sum

        # Center of mass (centroid)
        y_indices, x_indices = np.indices(h.shape)
        centroid_x = float(np.sum(x_indices * h) / h_sum)
        centroid_y = float(np.sum(y_indices * h) / h_sum)

        # Spatial spread (weighted radial variance from centroid)
        sq_dist = (x_indices - centroid_x) ** 2 + (y_indices - centroid_y) ** 2
        spatial_spread = float(np.sqrt(np.sum(sq_dist * h) / h_sum))

        # Fraction of area with high activation (>= 0.5)
        act_area_gt05 = float(np.sum(h >= 0.5) / total_pixels)

        return {
            "total_activation": round(h_sum, 4),
            "mean_activation": round(mean_act, 4),
            "peak_activation": round(peak_act, 4),
            "top10_threshold": round(top10_thresh, 4),
            "top10_activation_fraction": round(top10_fraction, 4),
            "centroid_x": round(centroid_x, 2),
            "centroid_y": round(centroid_y, 2),
            "spatial_spread": round(spatial_spread, 2),
            "activated_area_fraction_gt05": round(act_area_gt05, 4),
        }

    def explain_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        target_class: Union[str, int] = "predicted",
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> Dict[str, any]:
        """
        Full end-to-end explanation pipeline for a single layout image.

        Returns dictionary with:
          - raw_rgb: (224, 224, 3) uint8 image
          - heatmap_raw: (15, 15) float32 array
          - heatmap_224: (224, 224) float32 array
          - overlay: (224, 224, 3) uint8 overlay image
          - prediction info (p_hs, p_nhs, pred_label, etc.)
          - spatial statistics
        """
        img_tensor, raw_rgb = self.preprocess_image(image_input)
        heatmap_raw, heatmap_224, info = self.compute_heatmap(
            img_tensor, target_class=target_class
        )
        overlay = self.overlay_heatmap(raw_rgb, heatmap_224, alpha=alpha, colormap=colormap)
        stats = self.compute_spatial_statistics(heatmap_224)

        result = {
            "raw_rgb": raw_rgb,
            "heatmap_raw": heatmap_raw,
            "heatmap_224": heatmap_224,
            "overlay": overlay,
            **info,
            **stats,
        }
        return result


def save_comparison_panel(
    raw_rgb: np.ndarray,
    heatmap_224: np.ndarray,
    overlay: np.ndarray,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    stats: Optional[Dict[str, float]] = None,
):
    """
    Save a side-by-side 3-panel figure: Original | Grad-CAM Heatmap | Overlay.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=150)

    # 1. Original
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Original Layout", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Heatmap
    im1 = axes[1].imshow(heatmap_224, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (α=0.45)", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
