"""Evaluation metrics for HS/NHS classification."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf


def binary_metrics(y_true, y_pred, positive_label=0):
    """
    Compute confusion matrix and standard binary metrics.

    By default positive_label=0 treats HS (alphabetically first folder) as the
    positive class, which matches the research goal of hotspot detection.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    pos = int(positive_label)
    neg = 1 - pos

    tp = int(np.sum((y_true == pos) & (y_pred == pos)))
    tn = int(np.sum((y_true == neg) & (y_pred == neg)))
    fp = int(np.sum((y_true == neg) & (y_pred == pos)))
    fn = int(np.sum((y_true == pos) & (y_pred == neg)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    balanced_accuracy = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,  # reported but not primary for imbalanced data
        "positive_label": pos,
    }


def save_confusion_matrix_png(metrics: dict, out_path: Path, title: str):
    cm = metrics["confusion_matrix"]
    # rows = actual HS/NHS, cols = predicted HS/NHS (HS positive)
    matrix = np.array([[cm["tp"], cm["fn"]], [cm["fp"], cm["tn"]]], dtype=float)
    df_cm = pd.DataFrame(
        matrix,
        index=["Actual HS", "Actual NHS"],
        columns=["Pred HS", "Pred NHS"],
    )
    plt.figure(figsize=(4.5, 4))
    sn.heatmap(df_cm, annot=True, fmt=".0f", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_metrics_json(metrics: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_balanced_accuracy_curves(history, out_path: Path, title: str):
    """Notebook-style balanced accuracy vs epoch from Keras history metrics."""
    ttpl = history.history["true_positives"]
    ttnl = history.history["true_negatives"]
    tfpl = history.history["false_positives"]
    tfnl = history.history["false_negatives"]
    vtpl = history.history["val_true_positives"]
    vtnl = history.history["val_true_negatives"]
    vfpl = history.history["val_false_positives"]
    vfnl = history.history["val_false_negatives"]

    sensitivity = [ttpl[i] / (ttpl[i] + tfnl[i]) for i in range(len(tfnl))]
    specificity = [ttnl[i] / (tfpl[i] + ttnl[i]) for i in range(len(tfnl))]
    val_sensitivity = [vtpl[i] / (vtpl[i] + vfnl[i]) for i in range(len(vfnl))]
    val_specificity = [vtnl[i] / (vfpl[i] + vtnl[i]) for i in range(len(vfnl))]

    bal_acc = [(sensitivity[i] + specificity[i]) / 2 for i in range(len(sensitivity))]
    bal_acc_val = [
        (val_sensitivity[i] + val_specificity[i]) / 2 for i in range(len(val_sensitivity))
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(bal_acc, label="Balanced Accuracy (training)")
    plt.plot(bal_acc_val, label="Balanced Accuracy (validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.legend(loc="lower right")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

    return bal_acc, bal_acc_val


class BestBalancedAccuracyCheckpoint(tf.keras.callbacks.Callback):
    """
    Selects the best-epoch weights by validation balanced accuracy instead of leaving
    whatever the final training epoch produced.

    Balanced accuracy is computed epoch-by-epoch as 0.5 * (sensitivity + specificity)
    from Keras's native `val_true_positives` / `val_true_negatives` / `val_false_positives`
    / `val_false_negatives` metrics already present in `TRAIN_METRICS`
    (src/train_xai.py) -- the identical formula already used post-hoc by
    `plot_balanced_accuracy_curves` above, just evaluated live per epoch instead of only
    after training completes. No new metric is defined or compiled.

    At `on_train_end`, the model's weights are reset to the best epoch's weights, so the
    caller's existing `model.save(...)` call (unchanged) persists the best checkpoint
    rather than the final epoch. Ties keep the earliest epoch (strict `>` comparison).
    """

    def __init__(self):
        super().__init__()
        self.best_value = -1.0
        self.best_epoch = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tp = logs.get("val_true_positives")
        tn = logs.get("val_true_negatives")
        fp = logs.get("val_false_positives")
        fn = logs.get("val_false_negatives")
        if None in (tp, tn, fp, fn):
            return
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = 0.5 * (sensitivity + specificity)
        if bal_acc > self.best_value:
            self.best_value = bal_acc
            self.best_epoch = epoch
            self.best_weights = [w.copy() for w in self.model.get_weights()]

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
