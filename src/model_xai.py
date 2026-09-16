"""
XAI-Ready CNN architecture for Lithography Hotspot Detection.

Designed with high-resolution spatial feature maps (28x28 at final conv layer)
using 'same' padding and regular 2x2 max-pooling to avoid aggressive resolution
loss, enabling precise Grad-CAM, Grad-CAM++, and LayerCAM attribution.
"""

from __future__ import annotations

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_xai_cnn(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> tf.keras.Model:
    """
    Build the XAI-ready CNN with modular multi-scale conv stages.

    Layer Hierarchy for XAI:
      - Early conv layer:        `conv_early_2` (224x224x16) / after pool: 112x112
      - Intermediate conv layer: `conv_mid_2`   (112x112x24) / after pool: 56x56
      - Final conv layer:        `conv_final_2` (28x28x32)   <-- High resolution XAI target
    """
    inputs = layers.Input(shape=input_shape, name="input_layout")

    # Stage 1: Early spatial features (224x224 -> 112x112)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="elu", name="conv_early_1")(inputs)
    x = layers.Conv2D(16, (3, 3), padding="same", activation=None, name="conv_early_2")(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name="bn_early")(x)
    x = layers.Activation("elu", name="act_early")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_early")(x)  # -> (112, 112, 16)

    # Stage 2: Intermediate features (112x112 -> 56x56)
    x = layers.Conv2D(24, (3, 3), padding="same", activation="elu", name="conv_mid_1")(x)
    x = layers.Conv2D(24, (3, 3), padding="same", activation=None, name="conv_mid_2")(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name="bn_mid")(x)
    x = layers.Activation("elu", name="act_mid")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_mid")(x)  # -> (56, 56, 24)

    # Stage 3: High-Resolution Final Conv stage (56x56 -> 28x28)
    x = layers.MaxPooling2D((2, 2), name="pool_final_pre")(x)  # -> (28, 28, 24)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="elu", name="conv_final_1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation=None, name="conv_final_2")(x)  # (28, 28, 32)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001, name="bn_final")(x)
    x = layers.Activation("elu", name="act_final")(x)

    # Classifier Head
    x = layers.MaxPooling2D((2, 2), name="pool_classifier")(x)  # -> (14, 14, 32)
    x = layers.Flatten(name="flatten")(x)  # 14*14*32 = 6,272
    x = layers.Dropout(0.3, name="dropout")(x)
    x = layers.Dense(16, activation="relu", name="dense_penultimate")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="dense_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="xai_cnn")
    return model
