#!/usr/bin/env python3
"""Workstream B2: Controlled Imbalance Benchmark on Via-3.

Strategies:
  1. Baseline: Unweighted binary cross-entropy.
  2. Class-Weighted: Binary cross-entropy with inverse class frequency weights
     computed strictly on training split (w0 ~ 0.5159, w1 ~ 16.1904).
  3. Balanced Sampling: 50/50 balanced mini-batch sampling from HS and NHS pools.

Safeguards:
  - Architecture, splits, and optimizer fixed across all three conditions.
  - Enforces Pre-Training Verification Gate (gated unless approved).
  - Checkpoints selected on validation PR-AUC.
  - Full --dry-run mode for pipeline verification without GPU training.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from src.experiment_contract import (
    CONTRACT_VERSION,
    check_pretraining_gate,
    compute_contract_metrics,
    save_predictions_csv,
    select_threshold_from_val,
    verify_manifest_integrity,
)
from src.model import build_custom_cnn
from src.openhsd_data import OpenHSDVia3Data, load_via3_manifest_datasets


class BestPRAUCCallback(tf.keras.callbacks.Callback):
    """Tracks validation PR-AUC and saves/restores best checkpoint weights."""

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
        m = compute_contract_metrics(self.val_labels, probs, threshold=0.50)
        pr_auc = m["pr_auc"]
        f1 = m["f1_score"]

        record = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "val_pr_auc": pr_auc,
            "val_f1": f1,
            "val_recall": m["recall_sensitivity"],
            "val_precision": m["precision"],
            "val_balanced_acc": m["balanced_accuracy"],
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

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def train_single_strategy(
    strategy: str,
    data: OpenHSDVia3Data,
    output_dir: Path,
    models_dir: Path,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
    steps_per_epoch: int | None = None,
) -> dict:
    """Train single strategy: unweighted, class_weighted, or balanced_sampling."""
    print("\n" + "=" * 70)
    print(f"RUNNING IMBALANCE STRATEGY: {strategy.upper()} (Seed: {seed}, Epochs: {epochs})")
    print("=" * 70)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_custom_cnn(input_shape=(224, 224, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = models_dir / f"via3_{strategy}_seed{seed}.keras"
    callback = BestPRAUCCallback(data.val_ds, data.val_labels, ckpt_path)

    train_ds = data.train_ds
    class_weights = None
    spe = None

    if strategy == "class_weighted":
        class_weights = data.train_class_weights
        print(f"Class Weights applied: {class_weights}")
    elif strategy == "balanced_sampling":
        train_ds = data.balanced_train_ds
        # Steps per epoch to cover roughly the dataset size
        spe = steps_per_epoch or (len(data.train_paths) // 32)
        print(f"Balanced batch sampling activated with {spe} steps/epoch.")
    else:
        print("Standard unweighted loss.")

    t0 = time.time()
    history = model.fit(
        train_ds,
        epochs=epochs,
        steps_per_epoch=spe,
        validation_data=data.val_ds,
        class_weight=class_weights,
        callbacks=[callback],
        verbose=1,
    )
    duration = time.time() - t0

    # Calibration on validation
    val_probs = model.predict(data.val_ds, verbose=0).ravel()
    calibration = select_threshold_from_val(data.val_labels, val_probs, objective="max_f1")
    tau_star = calibration["selected_threshold"]

    # Final evaluation on clean test at frozen tau=0.50 and tau*
    clean_test_probs = model.predict(data.clean_test_ds, verbose=0).ravel()
    m_clean_default = compute_contract_metrics(data.clean_test_labels, clean_test_probs, threshold=0.50)
    m_clean_star = compute_contract_metrics(data.clean_test_labels, clean_test_probs, threshold=tau_star)

    # Diagnostic evaluation on official test (contaminated reference)
    official_test_probs = model.predict(data.official_test_ds, verbose=0).ravel()
    m_official_default = compute_contract_metrics(data.official_test_labels, official_test_probs, threshold=0.50)
    m_official_star = compute_contract_metrics(data.official_test_labels, official_test_probs, threshold=tau_star)

    # Save per-sample prediction CSVs
    val_pred_csv = output_dir / f"{strategy}_val_predictions.csv"
    save_predictions_csv(data.val_paths, data.val_labels, val_probs, val_pred_csv)

    clean_test_pred_csv = output_dir / f"{strategy}_clean_test_predictions.csv"
    save_predictions_csv(data.clean_test_paths, data.clean_test_labels, clean_test_probs, clean_test_pred_csv)

    official_test_pred_csv = output_dir / f"{strategy}_official_test_predictions.csv"
    save_predictions_csv(data.official_test_paths, data.official_test_labels, official_test_probs, official_test_pred_csv)

    res = {
        "strategy": strategy,
        "seed": seed,
        "best_epoch": callback.best_epoch,
        "training_duration_s": round(duration, 2),
        "calibrated_threshold": tau_star,
        "prediction_files": {
            "val_predictions_csv": str(val_pred_csv),
            "clean_test_predictions_csv": str(clean_test_pred_csv),
            "official_test_predictions_csv": str(official_test_pred_csv),
        },
        "metrics_val_default": compute_contract_metrics(data.val_labels, val_probs, threshold=0.50),
        "metrics_val_calibrated": calibration["metrics_at_selected_threshold"],
        "metrics_clean_test_default": m_clean_default,
        "metrics_clean_test_calibrated": m_clean_star,
        "metrics_official_test_diagnostic_default": m_official_default,
        "metrics_official_test_diagnostic_calibrated": m_official_star,
        "checkpoint": str(ckpt_path),
    }

    out_json = output_dir / f"{strategy}_results_seed{seed}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    # Free memory before next strategy
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return res


def run_imbalance_experiments(
    strategies: list[str],
    manifest_dir: Path,
    output_dir: Path,
    models_dir: Path,
    epochs: int = 15,
    seed: int = 42,
    dry_run: bool = False,
    smoke: bool = False,
    override_gate: bool = False,
) -> dict:
    """Run controlled comparison across imbalance strategies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Manifest Integrity
    manifest_status = verify_manifest_integrity(manifest_dir)
    if not all(manifest_status.values()):
        raise RuntimeError(f"Manifest integrity failed: {manifest_status}")

    # 2. Gate Verification Check
    gate_status = check_pretraining_gate()
    if not gate_status["gate_open"] and not dry_run and not override_gate:
        print("\n[CRITICAL ERROR] Pre-training verification gate is CLOSED.")
        print("Full training is forbidden until pre-training verification explicitly passes.")
        print("Use --dry-run to test configuration and data pipelines.")
        sys.exit(2)

    data = load_via3_manifest_datasets(
        manifest_dir=manifest_dir,
        batch_size=32,
        seed=seed,
        smoke=smoke or dry_run,
        smoke_samples=256 if (smoke or dry_run) else 0,
    )

    if dry_run:
        print("[DRY-RUN] Verifying pipelines and batch shapes for all strategies...")
        # Verify batch extraction from standard and balanced datasets
        bx, by = next(iter(data.train_ds.take(1)))
        assert bx.shape == (32, 224, 224, 3)
        b_bx, b_by = next(iter(data.balanced_train_ds.take(1)))
        assert b_bx.shape == (32, 224, 224, 3)

        dry_manifest = {
            "mode": "dry_run",
            "contract_version": CONTRACT_VERSION,
            "strategies_verified": strategies,
            "train_class_weights": data.train_class_weights,
            "balanced_dataset_batch_shape": list(b_bx.shape),
            "status": "PASS_DRY_RUN",
        }
        out_path = output_dir / "imbalance_dry_run_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dry_manifest, f, indent=2)
        print(f"Dry run complete. Manifest saved to {out_path}")
        return dry_manifest

    results = {}
    for strat in strategies:
        results[strat] = train_single_strategy(
            strategy=strat,
            data=data,
            output_dir=output_dir,
            models_dir=models_dir,
            epochs=epochs,
            seed=seed,
        )

    summary_file = output_dir / "imbalance_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nImbalance comparison summary saved to {summary_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Workstream B2 Imbalance Strategies Runner")
    parser.add_argument("--strategies", nargs="+", default=["baseline", "class_weighted", "balanced_sampling"])
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/dataset_audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/via3_experiments/imbalance"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/via3_imbalance"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--override-gate", action="store_true")
    args = parser.parse_args()

    run_imbalance_experiments(
        strategies=args.strategies,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        epochs=2 if args.smoke else args.epochs,
        seed=args.seed,
        dry_run=args.dry_run,
        smoke=args.smoke,
        override_gate=args.override_gate,
    )


if __name__ == "__main__":
    main()
