"""
Explainable AI (XAI) Engine for Lithography Hotspot Detection.

Implements:
  1. Vanilla Grad-CAM (Selvaraju et al., ICCV 2017)
  2. Grad-CAM++ (Chattopadhay et al., WACV 2018)
  3. LayerCAM (Jiang et al., IEEE TIP 2021)

Features:
  - Binary classifier pre-sigmoid logit targets to eliminate sigmoid gradient saturation.
  - Multi-target attribution: True-Class and Predicted-Class explanations.
  - Multi-layer support: Early, Intermediate, and Final convolutional layers.
  - Colormap overlay generation with customizable alpha blending.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import tensorflow as tf


def get_pre_sigmoid_logit(model: tf.keras.Model, penultimate_output: tf.Tensor) -> tf.Tensor:
    """Compute pre-sigmoid logit for the binary classifier."""
    dense_out_layer = model.get_layer("dense_output") if "dense_output" in [l.name for l in model.layers] else model.get_layer("dense_1")
    w = dense_out_layer.weights[0]
    b = dense_out_layer.weights[1]
    logit = tf.matmul(penultimate_output, w) + b
    return logit[:, 0]


class XAIEngine:
    """
    Comprehensive XAI interpreter supporting Grad-CAM, Grad-CAM++, and LayerCAM.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target_layer_name: Optional[str] = None,
        penultimate_layer_name: Optional[str] = None,
    ):
        self.model = model
        conv_layers = [
            l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            raise ValueError("No Conv2D layers found in the model.")

        self.target_layer_name = target_layer_name or conv_layers[-1]
        self.penultimate_layer_name = penultimate_layer_name or (
            "dense_penultimate" if "dense_penultimate" in [l.name for l in model.layers] else "dense"
        )

        # Build feature extractor submodel
        target_layer = self.model.get_layer(self.target_layer_name)
        penult_layer = self.model.get_layer(self.penultimate_layer_name)

        self.sub_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, penult_layer.output, self.model.output],
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        target_layer_name: Optional[str] = None,
    ) -> "XAIEngine":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        model = tf.keras.models.load_model(str(checkpoint_path))
        return cls(model, target_layer_name=target_layer_name)

    def set_target_layer(self, layer_name: str):
        """Switch target convolutional layer for multi-layer comparison."""
        target_layer = self.model.get_layer(layer_name)
        penult_layer = self.model.get_layer(self.penultimate_layer_name)
        self.target_layer_name = layer_name
        self.sub_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[target_layer.output, penult_layer.output, self.model.output],
        )

    def list_conv_layers(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """Return list of (layer_name, output_shape) for all Conv2D layers."""
        layers_info = []
        for l in self.model.layers:
            if isinstance(l, tf.keras.layers.Conv2D):
                layers_info.append((l.name, l.output_shape[1:]))
        return layers_info

    @staticmethod
    def preprocess_image(
        image_input: Union[str, Path, np.ndarray, Image.Image],
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray((image_input * 255).astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        raw_rgb = np.array(pil_img.resize(target_size), dtype=np.uint8)
        img_array = raw_rgb.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)
        return img_tensor, raw_rgb

    def predict(self, img_tensor: np.ndarray) -> Tuple[float, float, str, int]:
        preds = self.model(img_tensor, training=False).numpy()
        p_nhs = float(preds[0, 0])
        p_hs = float(1.0 - p_nhs)
        pred_idx = 1 if p_nhs >= 0.5 else 0
        pred_label = "NHS" if pred_idx == 1 else "HS"
        return p_hs, p_nhs, pred_label, pred_idx

    def _resolve_target_idx(self, img_tensor: np.ndarray, target_class: Union[str, int]) -> Tuple[int, str]:
        p_hs, p_nhs, pred_label, pred_idx = self.predict(img_tensor)
        if isinstance(target_class, str):
            t_str = target_class.upper()
            if t_str == "PREDICTED":
                return pred_idx, pred_label
            elif t_str in ["HS", "0", "HOTSPOT"]:
                return 0, "HS"
            elif t_str in ["NHS", "1", "NON-HOTSPOT", "NON_HOTSPOT"]:
                return 1, "NHS"
            else:
                raise ValueError(f"Unknown target class: {target_class}")
        else:
            target_idx = int(target_class)
            return target_idx, ("HS" if target_idx == 0 else "NHS")

    def explain_gradcam(
        self,
        img_tensor: np.ndarray,
        target_class: Union[str, int] = "predicted",
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Vanilla Grad-CAM: Global Average Pooling of gradients * feature maps.
        """
        target_idx, target_name = self._resolve_target_idx(img_tensor, target_class)

        with tf.GradientTape() as tape:
            conv_outputs, penult_outputs, preds = self.sub_model(img_tensor, training=False)
            logit_nhs = get_pre_sigmoid_logit(self.model, penult_outputs)
            # HS score = -logit, NHS score = +logit
            target_score = -logit_nhs if target_idx == 0 else logit_nhs

        grads = tape.gradient(target_score, conv_outputs)
        if grads is None:
            raise RuntimeError("Grad-CAM gradient computation failed.")

        # GAP weights: alpha_k = (1/Z) * sum_i,j (dScore / dA_ij^k)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
        conv_out = conv_outputs[0]  # (H, W, C)
        cam = conv_out @ weights[..., tf.newaxis]
        cam = tf.squeeze(cam)
        cam = tf.maximum(cam, 0.0)  # ReLU

        raw_map, resized_map = self._postprocess_cam(cam, img_tensor.shape[1:3])
        info = self._build_info_dict(img_tensor, target_idx, target_name, "Grad-CAM")
        return raw_map, resized_map, info

    def explain_gradcam_plus_plus(
        self,
        img_tensor: np.ndarray,
        target_class: Union[str, int] = "predicted",
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Grad-CAM++: Pixel-weighted higher-order gradients for multiple instances & precise localization.
        """
        target_idx, target_name = self._resolve_target_idx(img_tensor, target_class)

        with tf.GradientTape(persistent=True) as tape_grad2:
            with tf.GradientTape() as tape_grad1:
                conv_outputs, penult_outputs, preds = self.sub_model(img_tensor, training=False)
                logit_nhs = get_pre_sigmoid_logit(self.model, penult_outputs)
                target_score = -logit_nhs if target_idx == 0 else logit_nhs
            grads_1 = tape_grad1.gradient(target_score, conv_outputs)

        grads_2 = tape_grad2.gradient(grads_1, conv_outputs)
        grads_3 = tape_grad2.gradient(grads_2, conv_outputs)
        del tape_grad2

        if grads_1 is None:
            raise RuntimeError("Grad-CAM++ gradient computation failed.")

        # Replace None with zeros if higher gradients vanish
        if grads_2 is None:
            grads_2 = tf.zeros_like(grads_1)
        if grads_3 is None:
            grads_3 = tf.zeros_like(grads_1)

        conv_out = conv_outputs[0]  # (H, W, C)
        g1 = grads_1[0]
        g2 = grads_2[0]
        g3 = grads_3[0]

        # Grad-CAM++ alpha weighting:
        # alpha_ij^k = g2 / (2 * g2 + sum(A * g3) + eps)
        denom = 2.0 * g2 + tf.reduce_sum(conv_out * g3, axis=(0, 1), keepdims=True)
        denom = tf.where(denom != 0.0, denom, tf.ones_like(denom) * 1e-10)
        alphas = g2 / denom

        # Channel weights: w_k = sum_i,j (alpha_ij^k * ReLU(g1_ij^k))
        pos_g1 = tf.maximum(g1, 0.0)
        weights = tf.reduce_sum(alphas * pos_g1, axis=(0, 1))

        cam = conv_out @ weights[..., tf.newaxis]
        cam = tf.squeeze(cam)
        cam = tf.maximum(cam, 0.0)

        raw_map, resized_map = self._postprocess_cam(cam, img_tensor.shape[1:3])
        info = self._build_info_dict(img_tensor, target_idx, target_name, "Grad-CAM++")
        return raw_map, resized_map, info

    def explain_layercam(
        self,
        img_tensor: np.ndarray,
        target_class: Union[str, int] = "predicted",
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        LayerCAM: Spatial pixel-wise gradient weighting preserving fine-grained layout geometry.
        """
        target_idx, target_name = self._resolve_target_idx(img_tensor, target_class)

        with tf.GradientTape() as tape:
            conv_outputs, penult_outputs, preds = self.sub_model(img_tensor, training=False)
            logit_nhs = get_pre_sigmoid_logit(self.model, penult_outputs)
            target_score = -logit_nhs if target_idx == 0 else logit_nhs

        grads = tape.gradient(target_score, conv_outputs)
        if grads is None:
            raise RuntimeError("LayerCAM gradient computation failed.")

        conv_out = conv_outputs[0]  # (H, W, C)
        g = grads[0]  # (H, W, C)

        # LayerCAM spatial weight: w_ij^k = ReLU(g_ij^k)
        spatial_weights = tf.maximum(g, 0.0)

        # Pixel-wise combination: M_ij = ReLU( sum_k w_ij^k * A_ij^k )
        cam = tf.reduce_sum(spatial_weights * conv_out, axis=-1)
        cam = tf.maximum(cam, 0.0)

        raw_map, resized_map = self._postprocess_cam(cam, img_tensor.shape[1:3])
        info = self._build_info_dict(img_tensor, target_idx, target_name, "LayerCAM")
        return raw_map, resized_map, info

    def _postprocess_cam(self, cam: tf.Tensor, target_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        max_v = tf.math.reduce_max(cam)
        if max_v > 0:
            cam = cam / (max_v + 1e-10)

        raw_map = cam.numpy().astype(np.float32)

        # Bilinear interpolation upsampling
        resized = tf.image.resize(
            cam[..., tf.newaxis],
            target_hw,
            method="bilinear",
        ).numpy().squeeze().astype(np.float32)

        r_max = float(np.max(resized))
        if r_max > 0:
            resized = resized / r_max

        return raw_map, resized

    def _build_info_dict(self, img_tensor: np.ndarray, target_idx: int, target_name: str, method: str) -> Dict:
        p_hs, p_nhs, pred_label, pred_idx = self.predict(img_tensor)
        return {
            "method": method,
            "p_hs": p_hs,
            "p_nhs": p_nhs,
            "pred_label": pred_label,
            "pred_idx": pred_idx,
            "target_class": target_name,
            "target_idx": target_idx,
            "target_layer": self.target_layer_name,
        }

    @staticmethod
    def overlay_heatmap(
        raw_rgb: np.ndarray,
        heatmap_224: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> np.ndarray:
        try:
            cmap = matplotlib.colormaps[colormap]
        except AttributeError:
            cmap = cm.get_cmap(colormap)
        colored_cam = cmap(heatmap_224)[:, :, :3]
        colored_cam_uint8 = np.uint8(colored_cam * 255)
        overlay = (1.0 - alpha) * raw_rgb.astype(np.float32) + alpha * colored_cam_uint8.astype(np.float32)
        return np.clip(overlay, 0, 255).astype(np.uint8)
