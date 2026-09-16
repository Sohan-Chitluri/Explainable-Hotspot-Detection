#!/usr/bin/env python3
"""
Train the XAI-Ready CNN on ICCAD-12 benchmarks 1-5.

Key Features:
- Preserves exact ICCAD-12 train/test splits.
- Retains 28x28 spatial feature maps at the final conv layer (`conv_final_2`).
- Implements class weighting for extreme imbalance mitigation.
- Saves deterministic seed/configuration.
- Evaluates test-set metrics (balanced accuracy, precision, recall, specificity, F1, confusion matrix).
- Measures parameter count and inference latency.
- Compares performance directly against the baseline model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data import data_extractor, dataset_analysis
from src.metrics import (
    binary_metrics,
    plot_balanced_accuracy_curves,
    save_confusion_matrix_png,
    save_metrics_json,
)
from src.model_xai import build_xai_cnn

BENCHMARK_EPOCHS = {
    1: 5,
    2: 5,
    3: 10,
    4: 10,
    5: 10,
}

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
OPTIMIZER = "nadam"
LOSS = "binary_crossentropy"
TRAIN_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "TruePositives",
    "TrueNegatives",
    "FalsePositives",
    "FalseNegatives",
]


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Train XAI-Ready Lithography Hotspot CNN")
    p.add_argument(
        "--data-root",
        type=Path,
        default=root / "iccad-official",
        help="Root folder containing iccad1..iccad5",
    )
    p.add_argument(
        "--benchmarks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Which ICCAD benchmarks to train",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "xai_training",
        help="Directory for XAI training metrics and comparison plots",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=root / "models" / "xai",
        help="Directory for saved XAI Keras models",
    )
    p.add_argument(
        "--baseline-results",
        type=Path,
        default=root / "results" / "baseline" / "metrics.json",
        help="Path to baseline metrics.json for comparison",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def compute_class_weights(counts: dict) -> dict[int, float]:
    """Compute balanced class weights for training: n_samples / (2 * n_samples_class)."""
    n_hs = counts["train_hs"]
    n_nhs = counts["train_nhs"]
    total = n_hs + n_nhs
    if n_hs == 0 or n_nhs == 0:
        return {0: 1.0, 1: 1.0}
    # Class 0: HS (positive), Class 1: NHS (negative)
    w0 = total / (2.0 * n_hs)
    w1 = total / (2.0 * n_nhs)
    return {0: float(w0), 1: float(w1)}


def measure_inference_speed(model: tf.keras.Model, val_data_gen, n_warmup: int = 5, n_eval_batches: int = 20) -> dict:
    """Measure inference latency (ms per sample) on the current hardware."""
    val_data_gen.reset()
    # Warm-up
    for i, (x_batch, _) in enumerate(val_data_gen):
        if i >= n_warmup:
            break
        _ = model(x_batch, training=False)

    val_data_gen.reset()
    total_samples = 0
    t0 = time.perf_counter()
    for i, (x_batch, _) in enumerate(val_data_gen):
        if i >= n_eval_batches:
            break
        _ = model(x_batch, training=False)
        total_samples += len(x_batch)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    ms_per_sample = elapsed_ms / max(1, total_samples)
    return {
        "ms_per_sample": round(ms_per_sample, 3),
        "fps": round(1000.0 / ms_per_sample, 1) if ms_per_sample > 0 else 0.0,
        "eval_samples": total_samples,
    }


def evaluate_xai_model(model: tf.keras.Model, val_data_gen, positive_class_name="test_hs"):
    val_data_gen.reset()
    probs = model.predict(val_data_gen, verbose=1)
    y_true = val_data_gen.classes
    y_pred = (probs.ravel() >= 0.5).astype(int)

    class_indices = dict(val_data_gen.class_indices)
    hs_label = class_indices.get(positive_class_name, 0)

    metrics = binary_metrics(y_true, y_pred, positive_label=hs_label)
    metrics["class_indices"] = class_indices
    metrics["n_samples"] = int(len(y_true))
    metrics["n_predicted_positive"] = int(np.sum(y_pred == hs_label))
    return metrics, y_true, y_pred, probs.ravel()


def train_one_benchmark(
    benchmark: int,
    data_root: Path,
    results_dir: Path,
    models_dir: Path,
    seed: int = 42,
) -> dict:
    epochs = BENCHMARK_EPOCHS[benchmark]
    folder = str(data_root / f"iccad{benchmark}") + "/"
    print("\n" + "=" * 76)
    print(f"Training XAI CNN | Benchmark iccad{benchmark} | epochs={epochs} | seed={seed}")
    print("=" * 76)

    train_dir, val_dir, counts = dataset_analysis(folder)
    train_data_gen, val_data_gen = data_extractor(
        train_dir, val_dir, batch_size=BATCH_SIZE
    )

    class_weights = compute_class_weights(counts)
    print(f"Class distribution: {counts}")
    print(f"Computed Class Weights: HS (0)={class_weights[0]:.3f}, NHS (1)={class_weights[1]:.3f}")

    model = build_xai_cnn(input_shape=(*IMG_SIZE, 3))
    model.summary()
    param_count = int(model.count_params())

    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=TRAIN_METRICS)

    # Train with class weights
    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        class_weight=class_weights,
    )

    bench_dir = results_dir / f"iccad{benchmark}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    bal_train, bal_val = plot_balanced_accuracy_curves(
        history,
        bench_dir / "balanced_accuracy_vs_epoch.png",
        title=f"iccad{benchmark} (XAI CNN): Balanced Accuracy vs Epoch",
    )

    # Evaluate test split
    final_metrics, y_true, y_pred, probs = evaluate_xai_model(model, val_data_gen)
    latency_info = measure_inference_speed(model, val_data_gen)

    final_metrics.update(
        {
            "benchmark": f"iccad{benchmark}",
            "model_type": "xai_cnn",
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "loss": LOSS,
            "image_size": list(IMG_SIZE),
            "final_conv_layer": "conv_final_2",
            "final_conv_shape": [28, 28, 32],
            "param_count": param_count,
            "latency": latency_info,
            "class_weights": class_weights,
            "dataset_counts": counts,
            "seed": seed,
            "history_val_balanced_accuracy": bal_val,
            "best_history_val_balanced_accuracy": float(max(bal_val)) if bal_val else None,
        }
    )

    save_metrics_json(final_metrics, bench_dir / "metrics.json")
    save_confusion_matrix_png(
        final_metrics,
        bench_dir / "confusion_matrix.png",
        title=f"iccad{benchmark} (XAI CNN): Confusion Matrix (HS positive)",
    )

    model_path = models_dir / f"xai_cnn_iccad{benchmark}.keras"
    model.save(model_path)
    print(f"Saved XAI model -> {model_path}")
    print(
        f"iccad{benchmark} (XAI) Balanced Acc: {final_metrics['balanced_accuracy']:.4f} | "
        f"F1: {final_metrics['f1_score']:.4f} | Recall: {final_metrics['recall_sensitivity']:.4f} | "
        f"Spec: {final_metrics['specificity']:.4f} | Latency: {latency_info['ms_per_sample']} ms/sample"
    )

    # Clean up graph
    del model
    tf.keras.backend.clear_session()

    return final_metrics


def generate_comparison_plots_and_summary(
    all_xai_metrics: dict,
    baseline_metrics_path: Path,
    results_dir: Path,
):
    """Compare XAI CNN performance directly against baseline CNN."""
    baseline_data = {}
    if baseline_metrics_path.is_file():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f).get("per_benchmark", {})

    benchmarks = list(all_xai_metrics.keys())
    xai_bal = [all_xai_metrics[b]["balanced_accuracy"] for b in benchmarks]
    xai_f1 = [all_xai_metrics[b]["f1_score"] for b in benchmarks]
    xai_rec = [all_xai_metrics[b]["recall_sensitivity"] for b in benchmarks]
    xai_spec = [all_xai_metrics[b]["specificity"] for b in benchmarks]

    base_bal = [baseline_data.get(b, {}).get("balanced_accuracy", 0.0) for b in benchmarks]
    base_f1 = [baseline_data.get(b, {}).get("f1_score", 0.0) for b in benchmarks]
    base_rec = [baseline_data.get(b, {}).get("recall_sensitivity", 0.0) for b in benchmarks]
    base_spec = [baseline_data.get(b, {}).get("specificity", 0.0) for b in benchmarks]

    x = np.arange(len(benchmarks))
    width = 0.35

    # 1. Balanced Accuracy Bar Chart Comparison
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.bar(x - width / 2, base_bal, width, label="Baseline CNN (15x15)", color="#8b0000", alpha=0.85)
    plt.bar(x + width / 2, xai_bal, width, label="XAI-Ready CNN (28x28)", color="#1f77b4", alpha=0.9)
    plt.xlabel("ICCAD-12 Benchmark", fontweight="bold", fontsize=12)
    plt.ylabel("Test Balanced Accuracy", fontweight="bold", fontsize=12)
    plt.title("Classifier Performance: Baseline vs XAI-Ready CNN", fontweight="bold", fontsize=13)
    plt.xticks(x, [b.replace("iccad", "ICCAD-") for b in benchmarks])
    plt.ylim(0.7, 1.02)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "baseline_vs_xai_balanced_accuracy.png")
    plt.close()

    # Build summary record
    summary = {
        "benchmarks": benchmarks,
        "xai_average_balanced_accuracy": float(np.mean(xai_bal)),
        "baseline_average_balanced_accuracy": float(np.mean(base_bal)) if base_bal else None,
        "xai_average_f1": float(np.mean(xai_f1)),
        "baseline_average_f1": float(np.mean(base_f1)) if base_f1 else None,
        "per_benchmark_xai": all_xai_metrics,
        "per_benchmark_baseline": baseline_data,
        "comparison_table": {
            b: {
                "baseline_bal_acc": base_bal[i],
                "xai_bal_acc": xai_bal[i],
                "baseline_f1": base_f1[i],
                "xai_f1": xai_f1[i],
                "baseline_recall": base_rec[i],
                "xai_recall": xai_rec[i],
                "baseline_specificity": base_spec[i],
                "xai_specificity": xai_spec[i],
                "xai_param_count": all_xai_metrics[b]["param_count"],
                "xai_latency_ms": all_xai_metrics[b]["latency"]["ms_per_sample"],
            }
            for i, b in enumerate(benchmarks)
        },
    }

    save_metrics_json(summary, results_dir / "metrics.json")
    print(f"\nWrote combined summary metrics -> {results_dir / 'metrics.json'}")
    print(f"XAI CNN Average Balanced Accuracy = {summary['xai_average_balanced_accuracy']:.4f}")
    if summary["baseline_average_balanced_accuracy"]:
        print(f"Baseline Average Balanced Accuracy = {summary['baseline_average_balanced_accuracy']:.4f}")


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    print("Executing XAI CNN Training Pipeline:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  Device: {'GPU:0' if gpus else 'CPU'}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for b in args.benchmarks:
        all_metrics[f"iccad{b}"] = train_one_benchmark(
            b, args.data_root, args.results_dir, args.models_dir, seed=args.seed
        )

    generate_comparison_plots_and_summary(
        all_metrics, args.baseline_results, args.results_dir
    )


if __name__ == "__main__":
    main()
