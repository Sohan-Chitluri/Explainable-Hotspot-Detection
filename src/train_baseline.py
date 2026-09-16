#!/usr/bin/env python3
"""
Train the repository custom CNN baseline on ICCAD-12 benchmarks.

Faithful reproduction of LHD_CustomModel.ipynb (Approach 2), adapted for local
execution (CPU allowed; Colab drive/unrar/GPU hard-fail removed).

Usage (from repo root, with .venv activated):
  python -m src.train_baseline
  python -m src.train_baseline --benchmarks 1 2
  python -m src.train_baseline --data-root ./iccad-official
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Headless-friendly plotting
import matplotlib

matplotlib.use("Agg")

import numpy as np
import tensorflow as tf
from tensorflow.keras import models as keras_models

from src.data import data_extractor, dataset_analysis
from src.metrics import (
    binary_metrics,
    plot_balanced_accuracy_curves,
    save_confusion_matrix_png,
    save_metrics_json,
)
from src.model import build_custom_cnn

# Epochs per benchmark, matching the notebook.
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
    p = argparse.ArgumentParser(description="Custom CNN lithography hotspot baseline")
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
        default=root / "results" / "baseline",
        help="Directory for metrics and plots",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=root / "models",
        help="Directory for saved Keras models",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for numpy/TF (reproducibility aid)",
    )
    return p.parse_args()


def ensure_dataset(data_root: Path, benchmarks: list[int]):
    missing = []
    for b in benchmarks:
        base = data_root / f"iccad{b}"
        required = [
            base / "train" / "train_hs",
            base / "train" / "train_nhs",
            base / "test" / "test_hs",
            base / "test" / "test_nhs",
        ]
        for path in required:
            if not path.is_dir():
                missing.append(str(path))
            elif len(os.listdir(path)) == 0:
                missing.append(f"{path} (empty)")
    if missing:
        raise FileNotFoundError(
            "ICCAD-12 dataset directories missing or empty:\n  - "
            + "\n  - ".join(missing)
            + "\n\nExpected layout under data root:\n"
            "  iccad-official/iccad{N}/train/train_hs|train_nhs\n"
            "  iccad-official/iccad{N}/test/test_hs|test_nhs\n"
            "Obtain iccad_official.rar from the README Google Drive link and extract."
        )


def evaluate_generator(model, val_data_gen, positive_class_name="test_hs"):
    """Run inference on the validation generator and compute HS-positive metrics."""
    val_data_gen.reset()
    probs = model.predict(val_data_gen, verbose=1)
    y_true = val_data_gen.classes
    y_pred = (probs.ravel() >= 0.5).astype(int)

    class_indices = dict(val_data_gen.class_indices)
    # flow_from_directory: alphabetical -> test_hs=0, test_nhs=1
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
):
    epochs = BENCHMARK_EPOCHS[benchmark]
    folder = str(data_root / f"iccad{benchmark}") + "/"
    print("=" * 72)
    print(f"Benchmark iccad{benchmark} | epochs={epochs} | folder={folder}")
    print("=" * 72)

    train_dir, val_dir, counts = dataset_analysis(folder)
    train_data_gen, val_data_gen = data_extractor(
        train_dir, val_dir, batch_size=BATCH_SIZE
    )
    print("class_indices (train):", train_data_gen.class_indices)
    print("class_indices (val):", val_data_gen.class_indices)

    model = build_custom_cnn(input_shape=(*IMG_SIZE, 3))
    model.summary()
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=TRAIN_METRICS)

    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
    )

    bench_dir = results_dir / f"iccad{benchmark}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    bal_train, bal_val = plot_balanced_accuracy_curves(
        history,
        bench_dir / "balanced_accuracy_vs_epoch.png",
        title=f"iccad{benchmark}: Balanced Accuracy vs Epoch",
    )

    # Final evaluation on the held-out test split (notebook's validation_dir).
    final_metrics, y_true, y_pred, probs = evaluate_generator(model, val_data_gen)
    final_metrics.update(
        {
            "benchmark": f"iccad{benchmark}",
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "optimizer": OPTIMIZER,
            "loss": LOSS,
            "image_size": list(IMG_SIZE),
            "rescale": "1/255",
            "dataset_counts": counts,
            "history_val_balanced_accuracy": bal_val,
            "best_history_val_balanced_accuracy": float(max(bal_val)) if bal_val else None,
        }
    )

    save_metrics_json(final_metrics, bench_dir / "metrics.json")
    save_confusion_matrix_png(
        final_metrics,
        bench_dir / "confusion_matrix.png",
        title=f"iccad{benchmark}: Confusion Matrix (HS positive)",
    )

    # Save a small prediction sample table for inspection.
    pred_path = bench_dir / "predictions_summary.csv"
    # Store aggregate only (full prediction dumps can be huge for iccad2/3).
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"n_samples,{final_metrics['n_samples']}\n")
        f.write(f"n_pred_hs,{final_metrics['n_predicted_positive']}\n")
        f.write(f"balanced_accuracy,{final_metrics['balanced_accuracy']}\n")
        f.write(f"f1_score,{final_metrics['f1_score']}\n")

    model_path = models_dir / f"custom_cnn_iccad{benchmark}.keras"
    model.save(model_path)
    print(f"Saved model -> {model_path}")
    print(
        f"iccad{benchmark} balanced_accuracy={final_metrics['balanced_accuracy']:.4f} "
        f"f1={final_metrics['f1_score']:.4f} "
        f"recall={final_metrics['recall_sensitivity']:.4f} "
        f"specificity={final_metrics['specificity']:.4f}"
    )

    # Free graph memory between benchmarks
    del model
    keras_models  # silence linters
    tf.keras.backend.clear_session()

    return final_metrics


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    print("Using:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  Device: {'GPU' if gpus else 'CPU'}")

    ensure_dataset(args.data_root, args.benchmarks)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    started = datetime.now(timezone.utc).isoformat()

    for b in args.benchmarks:
        all_metrics[f"iccad{b}"] = train_one_benchmark(
            b, args.data_root, args.results_dir, args.models_dir
        )

    bal_values = [all_metrics[k]["balanced_accuracy"] for k in all_metrics]
    summary = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__ if hasattr(tf, "__version__") else "unknown",
        "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
        "benchmarks": list(all_metrics.keys()),
        "balanced_accuracy_per_benchmark": {
            k: all_metrics[k]["balanced_accuracy"] for k in all_metrics
        },
        "average_balanced_accuracy": float(sum(bal_values) / len(bal_values))
        if bal_values
        else None,
        "per_benchmark": all_metrics,
        "training_config": {
            "optimizer": OPTIMIZER,
            "loss": LOSS,
            "batch_size": BATCH_SIZE,
            "image_size": list(IMG_SIZE),
            "epochs_by_benchmark": {str(k): v for k, v in BENCHMARK_EPOCHS.items()},
            "positive_class_for_reported_metrics": "HS (folder *_hs, label 0)",
        },
    }
    save_metrics_json(summary, args.results_dir / "metrics.json")

    # Summary bar plot like the notebook
    import matplotlib.pyplot as plt

    labels = [k.replace("iccad", "") for k in all_metrics.keys()]
    values = [all_metrics[k]["balanced_accuracy"] for k in all_metrics.keys()]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="maroon", width=0.4)
    plt.xlabel("ICCAD-12 benchmarks", fontweight="bold", fontsize=12)
    plt.ylabel("Validation Balanced Accuracy", fontweight="bold", fontsize=12)
    plt.title("Custom CNN baseline balanced accuracy")
    plt.tight_layout()
    plt.savefig(args.results_dir / "balanced_accuracy_bar.png", dpi=150)
    plt.close()

    print("\n" + "=" * 72)
    print("BASELINE COMPLETE")
    print(f"Average validation balanced accuracy = {summary['average_balanced_accuracy']:.4f}")
    print(f"Summary metrics: {args.results_dir / 'metrics.json'}")
    print(f"Models directory: {args.models_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
