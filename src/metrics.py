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
    """Notebook-style balanced accuracy vs epoch from Keras history metrics.

    Note on Keras binary classification metrics:
      Alphabetical folder ordering assigns 0 to 'train_hs'/'test_hs' (Hotspot)
      and 1 to 'train_nhs'/'test_nhs' (Non-Hotspot).
      Keras considers class 1 as positive, meaning:
        - Keras TruePositives (tp): actual NHS predicted NHS
        - Keras FalseNegatives (fn): actual NHS predicted HS
        - Keras TrueNegatives (tn): actual HS predicted HS
        - Keras FalsePositives (fp): actual HS predicted NHS
      Therefore, Hotspot Sensitivity (HS Recall) = tn / (tn + fp),
      and Hotspot Specificity (NHS Recall) = tp / (tp + fn).
      Balanced Accuracy = 0.5 * (sensitivity + specificity).
    """
    h = history.history if hasattr(history, "history") else history
    ttpl = h.get("true_positives", [])
    ttnl = h.get("true_negatives", [])
    tfpl = h.get("false_positives", [])
    tfnl = h.get("false_negatives", [])
    vtpl = h.get("val_true_positives", [])
    vtnl = h.get("val_true_negatives", [])
    vfpl = h.get("val_false_positives", [])
    vfnl = h.get("val_false_negatives", [])

    n_train_epochs = min(len(ttpl), len(ttnl), len(tfpl), len(tfnl)) if (ttpl and ttnl and tfpl and tfnl) else 0
    n_val_epochs = min(len(vtpl), len(vtnl), len(vfpl), len(vfnl)) if (vtpl and vtnl and vfpl and vfnl) else 0

    sensitivity = [
        ttnl[i] / (ttnl[i] + tfpl[i]) if (ttnl[i] + tfpl[i]) else 0.0
        for i in range(n_train_epochs)
    ]
    specificity = [
        ttpl[i] / (ttpl[i] + tfnl[i]) if (ttpl[i] + tfnl[i]) else 0.0
        for i in range(n_train_epochs)
    ]
    val_sensitivity = [
        vtnl[i] / (vtnl[i] + vfpl[i]) if (vtnl[i] + vfpl[i]) else 0.0
        for i in range(n_val_epochs)
    ]
    val_specificity = [
        vtpl[i] / (vtpl[i] + vfnl[i]) if (vtpl[i] + vfnl[i]) else 0.0
        for i in range(n_val_epochs)
    ]

    bal_acc = [(sensitivity[i] + specificity[i]) / 2 for i in range(n_train_epochs)]
    bal_acc_val = [
        (val_sensitivity[i] + val_specificity[i]) / 2 for i in range(n_val_epochs)
    ]

    plt.figure(figsize=(7, 4))
    if bal_acc:
        plt.plot(bal_acc, label="Balanced Accuracy (training)")
    if bal_acc_val:
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
    (src/train_xai.py) -- evaluated live per epoch.

    Class 0 is Hotspot (HS), Class 1 is Non-Hotspot (NHS):
      - Hotspot sensitivity (recall of class 0) = tn / (tn + fp)
      - Hotspot specificity (recall of class 1) = tp / (tp + fn)

    At `on_train_end`, the model's weights are reset to the best epoch's weights, so the
    caller's existing `model.save(...)` call persists the best checkpoint
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
        # Hotspot (class 0) sensitivity = tn / (tn + fp)
        # Hotspot (class 0) specificity = tp / (tp + fn)
        sensitivity = tn / (tn + fp) if (tn + fp) else 0.0
        specificity = tp / (tp + fn) if (tp + fn) else 0.0
        bal_acc = 0.5 * (sensitivity + specificity)
        if bal_acc > self.best_value:
            self.best_value = bal_acc
            self.best_epoch = epoch
            self.best_weights = [w.copy() for w in self.model.get_weights()]

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
