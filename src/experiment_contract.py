"""Shared Experiment Contract for OpenHSDBench Via-3 Research Pipeline.

Version: 1.0.0
Author: Integration Owner (Shared Contract)

This contract defines the unified specifications, metric definitions, threshold
calibration rules, checkpoint criteria, and leakage safeguards across Workstreams A-D.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


CONTRACT_VERSION = "1.0.0"
DATASET_NAME = "OpenHSDBench_via3"

# Exact audited manifest SHA-256 checksums
EXPECTED_MANIFEST_HASHES = {
    "via3_split_manifest.json": "17bc670452a083c882d18c0de7f100b88327571f6e3795f0fc0fce512b8e4d53",
    "via3_exclusion_manifest.json": "f4d269cace5b94d1629aeefdd9f0b55d9fb210444529aa31a0af93326518755a",
    "via3_cohort_clean_test.json": "28d45645bb2b0dcb2f9365d02ce39a6083eb2a506926417c986e817b5441dd8f",
    "via3_cohort_official_test.json": "149627ea68634a4b835fa81ef0e4785d11a5ae9b2fec9237577d45837622adbe",
}

# Audited exact counts
AUDITED_COUNTS = {
    "train": {"total": 16158, "hs": 499, "nhs": 15659},
    "val": {"total": 2846, "hs": 87, "nhs": 2759},
    "clean_test": {"total": 12803, "hs": 411, "nhs": 12392},
    "official_test": {"total": 12905, "hs": 422, "nhs": 12483},
    "excluded_test": {"total": 102, "hs": 11, "nhs": 91},
}

# Label semantics: Explicitly 1 = Hotspot (HS), 0 = Non-Hotspot (NHS)
LABEL_MAPPING = {
    0: {"name": "Non-Hotspot", "code": "NHS", "is_positive": False},
    1: {"name": "Hotspot", "code": "HS", "is_positive": True},
}

# Primary and secondary evaluation metrics
PRIMARY_METRIC = "pr_auc"
SECONDARY_METRICS = ["roc_auc", "recall_sensitivity", "precision", "f1_score", "balanced_accuracy"]

# Pre-training verification gate file path
PRETRAINING_GATE_FILE = Path("/home/peskybird/Projects/Ai_Ml_DA/.pretraining_verification_passed")


def file_sha256(path: Path | str) -> str:
    """Compute standard SHA-256 hex digest of file contents."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest_integrity(manifest_dir: Path | str) -> Dict[str, bool]:
    """Verify presence and exact SHA-256 hashes of all via-3 manifests."""
    d = Path(manifest_dir)
    results = {}
    for filename, expected_hash in EXPECTED_MANIFEST_HASHES.items():
        file_path = d / filename
        if not file_path.exists():
            results[filename] = False
            continue
        actual_hash = file_sha256(file_path)
        results[filename] = (actual_hash == expected_hash)
    return results


def check_pretraining_gate() -> Dict[str, Any]:
    """Check whether the pre-training verification gate is open.

    Full training must NOT begin unless this gate is explicitly confirmed.
    """
    if PRETRAINING_GATE_FILE.exists():
        try:
            with open(PRETRAINING_GATE_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            return {"gate_open": True, "reason": "Gate file present", "detail": content}
        except Exception as e:
            return {"gate_open": False, "reason": f"Gate file read error: {e}"}
    return {
        "gate_open": False,
        "reason": "Pre-training verification is running or gate file not found. Full training is gated.",
    }


def compute_contract_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute canonical metrics adhering to the experiment contract (1 = HS positive).

    Definitions:
      - Hotspot Recall (Sensitivity): TP / (TP + FN)
      - Hotspot Specificity: TN / (TN + FP)
      - Hotspot Precision: TP / (TP + FP)
      - Hotspot F1: 2 * Precision * Recall / (Precision + Recall)
      - Balanced Accuracy: 0.5 * (Sensitivity + Specificity)
      - PR-AUC: Exact trapz / step average precision
      - ROC-AUC: Wilcoxon-Mann-Whitney rank-sum statistic
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    n_samples = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_accuracy = float(0.5 * (recall + specificity))
    accuracy = float((tp + tn) / n_samples) if n_samples > 0 else 0.0

    # PR-AUC / Average Precision calculation
    if n_pos == 0:
        pr_auc = 0.0
    else:
        order = np.argsort(-y_prob)
        y_sorted = y_true[order]
        tp_cum = np.cumsum(y_sorted == 1)
        fp_cum = np.cumsum(y_sorted == 0)
        recalls = tp_cum / n_pos
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        recalls_prev = np.concatenate(([0.0], recalls[:-1]))
        pr_auc = float(np.sum((recalls - recalls_prev) * precisions))

    # ROC-AUC via Wilcoxon-Mann-Whitney rank sum
    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
    else:
        ranks = np.argsort(np.argsort(y_prob)) + 1
        pos_ranks = ranks[y_true == 1]
        u = np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0
        roc_auc = float(u / (n_pos * n_neg))

    return {
        "n_samples": n_samples,
        "n_pos_hs": n_pos,
        "n_neg_nhs": n_neg,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": f1,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "positive_label": 1,
        "positive_class": "Hotspot (HS)",
        "negative_class": "Non-Hotspot (NHS)",
    }


def select_threshold_from_val(
    val_labels: np.ndarray | list[int],
    val_probs: np.ndarray | list[float],
    threshold_grid: Optional[list[float]] = None,
    objective: str = "max_f1",
    min_precision: float = 0.0,
    target_recall: float = 0.70,
) -> Dict[str, Any]:
    """Calibrate decision threshold strictly on VALIDATION probabilities.

    Supported objectives:
      - 'max_f1': Maximizes Hotspot F1 score.
      - 'max_recall_constrained': Maximizes Hotspot recall subject to precision >= min_precision.
      - 'target_recall': Finds threshold achieving recall >= target_recall with highest precision.
      - 'max_balanced_accuracy': Maximizes balanced accuracy.

    RULES:
      - NEVER use test set labels or predictions during this selection.
      - The returned threshold tau* is frozen before held-out test evaluation.
    """
    if threshold_grid is None:
        threshold_grid = [round(t, 4) for t in np.linspace(0.01, 0.99, 99)]

    val_labels = np.asarray(val_labels).astype(int).ravel()
    val_probs = np.asarray(val_probs).astype(float).ravel()

    sweep_records = []
    best_threshold = 0.50
    best_score = -1.0
    best_metrics = None

    for tau in threshold_grid:
        m = compute_contract_metrics(val_labels, val_probs, threshold=tau)
        rec = {
            "threshold": tau,
            "tp": m["confusion_matrix"]["tp"],
            "fp": m["confusion_matrix"]["fp"],
            "fn": m["confusion_matrix"]["fn"],
            "tn": m["confusion_matrix"]["tn"],
            "precision": m["precision"],
            "recall": m["recall_sensitivity"],
            "f1_score": m["f1_score"],
            "balanced_accuracy": m["balanced_accuracy"],
        }
        sweep_records.append(rec)

        if objective == "max_f1":
            score = m["f1_score"]
            if score > best_score:
                best_score = score
                best_threshold = tau
                best_metrics = m
        elif objective == "max_recall_constrained":
            if m["precision"] >= min_precision:
                score = m["recall_sensitivity"]
                if score > best_score:
                    best_score = score
                    best_threshold = tau
                    best_metrics = m
        elif objective == "target_recall":
            if m["recall_sensitivity"] >= target_recall:
                score = m["precision"]
                if score > best_score:
                    best_score = score
                    best_threshold = tau
                    best_metrics = m
        elif objective == "max_balanced_accuracy":
            score = m["balanced_accuracy"]
            if score > best_score:
                best_score = score
                best_threshold = tau
                best_metrics = m

    # Fallback to default 0.50 if no threshold met constraints
    if best_metrics is None:
        best_threshold = 0.50
        best_metrics = compute_contract_metrics(val_labels, val_probs, threshold=0.50)

    return {
        "selected_threshold": best_threshold,
        "objective": objective,
        "objective_score": best_score,
        "metrics_at_selected_threshold": best_metrics,
        "sweep_table": sweep_records,
    }


def verify_leakage_safeguards(
    train_paths: list[str],
    val_paths: list[str],
    clean_test_paths: list[str],
    exclusion_manifest_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify that there is zero overlap between train, val, and clean test sets."""
    train_set = set(Path(p).name for p in train_paths)
    val_set = set(Path(p).name for p in val_paths)
    test_set = set(Path(p).name for p in clean_test_paths)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    excluded_in_clean_test = 0
    if exclusion_manifest_path and Path(exclusion_manifest_path).exists():
        with open(exclusion_manifest_path, "r", encoding="utf-8") as f:
            ex_data = json.load(f)
        excluded_files = set(ex_data.get("excluded_test_samples", []))
        excluded_in_clean_test = len(test_set & excluded_files)

    is_clean = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
        and excluded_in_clean_test == 0
    )

    return {
        "is_leakage_free": is_clean,
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "excluded_samples_in_clean_test": excluded_in_clean_test,
    }


def save_predictions_csv(
    paths: list[str],
    y_true: list[int] | np.ndarray,
    probs: np.ndarray | list[float],
    out_path: Path | str,
) -> Path:
    """Save raw per-sample prediction probabilities and decisions.

    Columns: sample_id, filename, ground_truth, hs_probability, predicted_label
    Where:
      - sample_id: ID prefix before first underscore (e.g. 'Hvia107504')
      - filename: base name of image file (e.g. 'Hvia107504_mb_mb_lccout.oas.gds.png')
      - ground_truth: integer label (0 = NHS, 1 = HS)
      - hs_probability: continuous predicted HS probability in [0.0, 1.0] rounded to 6 decimal places
      - predicted_label: binary decision at default threshold 0.5 (1 if prob >= 0.5 else 0)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_true_arr = np.asarray(y_true).astype(int).ravel()
    probs_arr = np.asarray(probs).astype(float).ravel()

    assert len(paths) == len(y_true_arr) == len(probs_arr), (
        f"Length mismatch: {len(paths)} paths, {len(y_true_arr)} labels, {len(probs_arr)} probs"
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "filename", "ground_truth", "hs_probability", "predicted_label"])
        for p, true_lbl, prob in zip(paths, y_true_arr, probs_arr):
            fn = Path(p).name
            sample_id = fn.split("_")[0]
            prob_val = float(prob)
            pred_lbl = 1 if prob_val >= 0.5 else 0
            writer.writerow([sample_id, fn, int(true_lbl), round(prob_val, 6), pred_lbl])

    return out_path
