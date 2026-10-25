#!/usr/bin/env python3
"""Workstream A: Reproducible Baseline Runner for OpenHSDBench via-3.

Integrity Safeguards:
- Enforces manifest SHA-256 verification and count validation.
- Verifies zero train/val and train/clean-test leakage.
- Enforces the Pre-Training Verification Gate: will not launch full training
  unless the pre-training verification gate is explicitly open.
- Provides --dry-run mode for zero-cost integration validation.
- Validates best checkpoint restoration before frozen held-out test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import tensorflow as tf

from src.experiment_contract import (
    AUDITED_COUNTS,
    CONTRACT_VERSION,
    check_pretraining_gate,
    compute_contract_metrics,
    file_sha256,
    save_predictions_csv,
    select_threshold_from_val,
    verify_leakage_safeguards,
    verify_manifest_integrity,
)
from src.model import build_custom_cnn
from src.openhsd_data import OpenHSDVia3Data, load_via3_manifest_datasets


class BestCheckpointCallback(tf.keras.callbacks.Callback):
    """Tracks validation PR-AUC at epoch end and saves/restores best weights."""

    def __init__(self, val_ds: tf.data.Dataset, val_labels: list[int], ckpt_path: Path):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = np.asarray(val_labels).astype(int)
        self.ckpt_path = Path(ckpt_path)
        self.best_pr_auc = -1.0
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.best_weights = None
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probs = self.model.predict(self.val_ds, verbose=0).ravel()
        metrics = compute_contract_metrics(self.val_labels, probs, threshold=0.5)
        pr_auc = metrics["pr_auc"]
        f1 = metrics["f1_score"]

        record = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "val_pr_auc": pr_auc,
            "val_f1": f1,
            "val_recall": metrics["recall_sensitivity"],
            "val_precision": metrics["precision"],
            "val_specificity": metrics["specificity"],
            "val_balanced_acc": metrics["balanced_accuracy"],
            "val_tp": metrics["confusion_matrix"]["tp"],
            "val_fp": metrics["confusion_matrix"]["fp"],
            "val_fn": metrics["confusion_matrix"]["fn"],
            "val_tn": metrics["confusion_matrix"]["tn"],
        }
        self.history.append(record)

        is_better = False
        if pr_auc > self.best_pr_auc + 1e-4:
            is_better = True
        elif abs(pr_auc - self.best_pr_auc) <= 1e-4 and f1 > self.best_f1 + 1e-4:
            is_better = True

        if is_better or self.best_weights is None:
            self.best_pr_auc = pr_auc
            self.best_f1 = f1
            self.best_epoch = epoch + 1
            self.best_weights = [w.copy() for w in self.model.get_weights()]
            self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.ckpt_path)
            print(f" -> Epoch {epoch+1:02d}: New BEST val PR-AUC: {pr_auc:.4f} (F1: {f1:.4f}) [Saved {self.ckpt_path.name}]")
        else:
            print(f" -> Epoch {epoch+1:02d}: val PR-AUC: {pr_auc:.4f} (F1: {f1:.4f}) [Best: {self.best_pr_auc:.4f} at Ep {self.best_epoch}]")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            print(f"\n[ValidationPRAUCCallback] Restoring best checkpoint weights from Epoch {self.best_epoch}")
            self.model.set_weights(self.best_weights)


def run_baseline(
    manifest_dir: Path,
    output_dir: Path,
    models_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    dry_run: bool = False,
    smoke: bool = False,
    override_gate: bool = False,
) -> dict:
    """Run baseline validation and training workflow."""
    print("=" * 80)
    print(f"WORKSTREAM A: BASELINE RUNNER (Dry-Run: {dry_run}, Smoke: {smoke}, Seed: {seed})")
    print("=" * 80)

    # 1. Manifest Integrity Validation
    manifest_status = verify_manifest_integrity(manifest_dir)
    print(f"[1/6] Manifest Integrity Check: {manifest_status}")
    if not all(manifest_status.values()):
        failed = [k for k, v in manifest_status.items() if not v]
        raise RuntimeError(f"Manifest integrity verification failed for: {failed}")

    # 2. Gate Verification Check
    gate_status = check_pretraining_gate()
    print(f"[2/6] Pre-Training Verification Gate: {gate_status['reason']}")
    if not gate_status["gate_open"] and not dry_run and not override_gate:
        print("\n[CRITICAL ERROR] Pre-training verification gate is CLOSED.")
        print("Full training is forbidden until pre-training verification explicitly passes.")
        print("Use --dry-run to test the pipeline without training.")
        sys.exit(2)

    # 3. Load Datasets and Validate Counts
    print("[3/6] Loading datasets and validating population counts...")
    data = load_via3_manifest_datasets(
        manifest_dir=manifest_dir,
        batch_size=batch_size,
        seed=seed,
        smoke=smoke or dry_run,
        smoke_samples=256 if (smoke or dry_run) else 0,
    )

    if not smoke and not dry_run:
        for split_key, expected in AUDITED_COUNTS.items():
            if split_key in data.counts:
                actual = data.counts[split_key]
                assert actual["total"] == expected["total"], f"Count mismatch in {split_key}: {actual} vs {expected}"

    # 4. Leakage Safeguard Check
    exclusion_file = manifest_dir / "via3_exclusion_manifest.json"
    leakage = verify_leakage_safeguards(
        data.train_paths,
        data.val_paths,
        data.clean_test_paths,
        exclusion_manifest_path=exclusion_file,
    )
    print(f"[4/6] Leakage Verification: {leakage}")
    if not leakage["is_leakage_free"]:
        raise RuntimeError(f"Data leakage detected! {leakage}")

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / f"via3_baseline_seed{seed}{'_smoke' if smoke else ''}.keras"

    # In Dry-Run mode, generate a valid contract-compliant experiment manifest and exit
    if dry_run:
        print("\n[5/6] DRY RUN: Model training skipped. Generating manifest fixture...")
        dry_run_manifest = {
            "mode": "dry_run",
            "contract_version": CONTRACT_VERSION,
            "manifest_integrity": manifest_status,
            "gate_status": gate_status,
            "leakage_status": leakage,
            "dataset_counts": data.counts,
            "model_architecture": "custom_cnn_18k",
            "batch_size": batch_size,
            "seed": seed,
            "simulated_test_evaluated": False,
            "status": "PASS_DRY_RUN",
        }
        out_manifest_path = output_dir / "baseline_dry_run_manifest.json"
        with open(out_manifest_path, "w", encoding="utf-8") as f:
            json.dump(dry_run_manifest, f, indent=2)
        print(f"[6/6] Dry run successfully completed. Manifest written to {out_manifest_path}")
        return dry_run_manifest

    # 5. Model Compilation & Training
    print("\n[5/6] Building model and initiating training...")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_custom_cnn(input_shape=(224, 224, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callback = BestCheckpointCallback(
        val_ds=data.val_ds,
        val_labels=data.val_labels,
        ckpt_path=ckpt_path,
    )

    t0 = time.time()
    history = model.fit(
        data.train_ds,
        epochs=epochs,
        validation_data=data.val_ds,
        callbacks=[callback],
        verbose=1,
    )
    duration = time.time() - t0

    # 6. Frozen Evaluation
    print("\n[6/6] Freezing checkpoint and threshold, evaluating cohorts...")
    # Generate validation probabilities once for threshold selection
    val_probs = model.predict(data.val_ds, verbose=0).ravel()
    threshold_analysis = select_threshold_from_val(data.val_labels, val_probs, objective="max_f1")
    tau_star = threshold_analysis["selected_threshold"]
    print(f"Operational threshold calibrated on validation: tau* = {tau_star:.4f}")

    # Evaluate held-out clean test cohort at default tau=0.50 and calibrated tau*
    clean_test_probs = model.predict(data.clean_test_ds, verbose=0).ravel()
    clean_test_metrics_default = compute_contract_metrics(data.clean_test_labels, clean_test_probs, threshold=0.50)
    clean_test_metrics_calibrated = compute_contract_metrics(data.clean_test_labels, clean_test_probs, threshold=tau_star)

    # Diagnostic official test cohort
    official_test_probs = model.predict(data.official_test_ds, verbose=0).ravel()
    official_test_metrics = compute_contract_metrics(data.official_test_labels, official_test_probs, threshold=0.50)

    # Save per-sample prediction CSVs matching analyzer contract
    val_pred_csv = save_predictions_csv(data.val_paths, data.val_labels, val_probs, output_dir / "val_predictions.csv")
    clean_pred_csv = save_predictions_csv(data.clean_test_paths, data.clean_test_labels, clean_test_probs, output_dir / "clean_test_predictions.csv")
    official_pred_csv = save_predictions_csv(data.official_test_paths, data.official_test_labels, official_test_probs, output_dir / "official_test_predictions.csv")
    print(f"Predictions saved to {val_pred_csv.name}, {clean_pred_csv.name}, {official_pred_csv.name}")

    manifest_output = {
        "contract_version": CONTRACT_VERSION,
        "experiment_name": "via3_baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "platform": platform.platform(),
        },
        "prediction_files": {
            "val_predictions_csv": str(val_pred_csv),
            "clean_test_predictions_csv": str(clean_pred_csv),
            "official_test_predictions_csv": str(official_pred_csv),
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "seed": seed,
            "smoke": smoke,
        },
        "dataset_counts": data.counts,
        "manifest_hashes": {k: file_sha256(manifest_dir / k) for k in manifest_status},
        "checkpoint": {
            "path": str(ckpt_path),
            "best_epoch": callback.best_epoch,
            "best_val_pr_auc": callback.best_pr_auc,
            "best_val_f1": callback.best_f1,
            "sha256": file_sha256(ckpt_path) if ckpt_path.exists() else None,
        },
        "threshold_calibration": {
            "calibrated_tau": tau_star,
            "objective": threshold_analysis["objective"],
            "objective_score": threshold_analysis["objective_score"],
        },
        "metrics_val_default": compute_contract_metrics(data.val_labels, val_probs, threshold=0.50),
        "metrics_clean_test_default": clean_test_metrics_default,
        "metrics_clean_test_calibrated": clean_test_metrics_calibrated,
        "metrics_official_test_default": official_test_metrics,
        "training_duration_seconds": round(duration, 2),
        "training_history": callback.history,
    }

    out_json = output_dir / f"baseline_run_manifest_seed{seed}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest_output, f, indent=2)
    print(f"\nExperiment manifest saved to {out_json}")

    return manifest_output


def main():
    parser = argparse.ArgumentParser(description="Workstream A Baseline Runner")
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/dataset_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/via3_experiments/baseline"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/via3"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Run validation and checks without model training")
    parser.add_argument("--smoke", action="store_true", help="Run 2-epoch mini cohort test")
    parser.add_argument("--override-gate", action="store_true", help="Override pre-training gate (for authorized runs only)")
    args = parser.parse_args()

    run_baseline(
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        dry_run=args.dry_run,
        smoke=args.smoke,
        override_gate=args.override_gate,
    )


if __name__ == "__main__":
    main()
