#!/usr/bin/env python3
"""
Seed Variance Study for iccad1, iccad2, iccad5
==============================================

Runs train_xai.py 3 times each for the three benchmarks with the largest
old-vs-clean deltas, using different random seeds.

Evaluates all 3 seed runs on the official untouched test set and reports:
- Mean test balanced accuracy
- Std dev, min, max
- Individual values for each seed
- Comparison to original leaky checkpoint accuracy

Output:
- models/xai/retrained_clean/seed_variance/{benchmark}_seed{N}.keras
- results/xai_training/seed_variance/report.txt
- results/xai_training/seed_variance/results.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from src.data import make_three_way_split
from src.metrics import binary_metrics, BestBalancedAccuracyCheckpoint
from src.model_xai import build_xai_cnn

BENCHMARKS_TO_TEST = [1, 2, 5]  # iccad1, iccad2, iccad5
SEEDS = [101, 202, 303]  # Different from the original 42

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

# Original leaky checkpoint accuracies (from the JSON comparison)
ORIGINAL_LEAKY_ACCURACIES = {
    "iccad1": 0.8935,
    "iccad2": 0.9314,
    "iccad5": 0.9501,
}


def compute_class_weights(counts: dict) -> dict[int, float]:
    """Compute balanced class weights for training."""
    n_hs = counts["train_hs"]
    n_nhs = counts["train_nhs"]
    total = n_hs + n_nhs
    if n_hs == 0 or n_nhs == 0:
        return {0: 1.0, 1: 1.0}
    w0 = total / (2.0 * n_hs)
    w1 = total / (2.0 * n_nhs)
    return {0: float(w0), 1: float(w1)}


def evaluate_model_on_test(model: tf.keras.Model, test_gen) -> float:
    """Evaluate model on test set and return balanced accuracy."""
    test_gen.reset()
    probs = model.predict(test_gen, verbose=0)
    y_true = test_gen.classes
    y_pred = (probs.ravel() >= 0.5).astype(int)

    class_indices = dict(test_gen.class_indices)
    hs_label = class_indices.get("test_hs", 0)

    metrics = binary_metrics(y_true, y_pred, positive_label=hs_label)
    return metrics["balanced_accuracy"]


def train_benchmark_with_seed(
    benchmark: int,
    seed: int,
    data_root: Path,
    results_dir: Path,
    models_dir: Path,
) -> float:
    """
    Train one benchmark with a specific seed.
    Returns the test set balanced accuracy.
    """
    tf.keras.utils.set_random_seed(seed)

    benchmark_name = f"iccad{benchmark}"
    benchmark_dir = data_root / benchmark_name
    epochs = BENCHMARK_EPOCHS[benchmark]

    print(f"  Training {benchmark_name} with seed {seed}...", flush=True)

    # Three-way split with the given seed
    split = make_three_way_split(
        benchmark_dir,
        benchmark_name,
        val_frac=0.15,
        seed=seed,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )

    train_gen = split.train_gen
    val_gen = split.val_gen
    test_gen = split.test_gen
    counts = split.counts

    class_weights = compute_class_weights(counts)

    # Build and train model
    model = build_xai_cnn(input_shape=(*IMG_SIZE, 3))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=TRAIN_METRICS)

    best_ckpt_cb = BestBalancedAccuracyCheckpoint()
    model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[best_ckpt_cb],
        verbose=0,
    )

    # Evaluate on test set
    test_balanced_acc = evaluate_model_on_test(model, test_gen)

    # Save checkpoint
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{benchmark_name}_seed{seed}.keras"
    model.save(model_path)
    print(f"    → Saved to {model_path}")
    print(f"    → Test balanced accuracy: {test_balanced_acc:.4f}")

    # Clean up
    del model
    tf.keras.backend.clear_session()

    return test_balanced_acc


def main():
    root = Path(__file__).resolve().parents[1]
    data_root = root / "iccad-official"
    models_dir = root / "models" / "xai" / "retrained_clean" / "seed_variance"
    results_dir = root / "results" / "xai_training" / "seed_variance"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Seed Variance Study: iccad1, iccad2, iccad5")
    print("=" * 80)
    print(f"Data root: {data_root}")
    print(f"Models dir: {models_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Seeds: {SEEDS}")
    print(f"Benchmarks: {BENCHMARKS_TO_TEST}")
    print()

    all_results = {}

    for benchmark in BENCHMARKS_TO_TEST:
        benchmark_name = f"iccad{benchmark}"
        print(f"\n{benchmark_name.upper()}")
        print("-" * 80)

        accuracies = []
        for seed in SEEDS:
            acc = train_benchmark_with_seed(
                benchmark,
                seed,
                data_root,
                results_dir,
                models_dir,
            )
            accuracies.append(acc)

        # Compute statistics
        accuracies = np.array(accuracies)
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        min_acc = float(np.min(accuracies))
        max_acc = float(np.max(accuracies))
        original_leaky = ORIGINAL_LEAKY_ACCURACIES[benchmark_name]

        # Determine if original leaky is inside or outside the new seed range
        inside_range = min_acc <= original_leaky <= max_acc

        all_results[benchmark_name] = {
            "seeds": SEEDS,
            "accuracies": accuracies.tolist(),
            "mean": mean_acc,
            "std": std_acc,
            "min": min_acc,
            "max": max_acc,
            "original_leaky_accuracy": original_leaky,
            "original_inside_range": inside_range,
        }

        print(f"\nResults Summary:")
        print(f"  Seed 101: {accuracies[0]:.4f}")
        print(f"  Seed 202: {accuracies[1]:.4f}")
        print(f"  Seed 303: {accuracies[2]:.4f}")
        print(f"  Mean ± Std: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Range: [{min_acc:.4f}, {max_acc:.4f}]")
        print(f"\n  Original leaky checkpoint: {original_leaky:.4f}")
        if inside_range:
            print(f"  ✓ Inside range: plausibly just variance")
        else:
            print(f"  ✗ Outside range: likely a real leakage effect")

    # Write detailed report
    report_path = results_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Seed Variance Study Report: iccad1, iccad2, iccad5\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Data root: {data_root}\n")
        f.write(f"Models dir: {models_dir}\n")
        f.write("\n")

        for benchmark_name in sorted(all_results.keys()):
            res = all_results[benchmark_name]
            f.write(f"\n{benchmark_name.upper()}\n")
            f.write("-" * 80 + "\n")
            f.write("| Seed | Test Balanced Accuracy |\n")
            f.write("|------|------------------------|\n")
            for seed, acc in zip(res["seeds"], res["accuracies"]):
                f.write(f"| {seed:4d} | {acc:22.4f} |\n")
            f.write(f"| Mean |       {res['mean']:.4f} ± {res['std']:.4f}      |\n")
            f.write(f"| Min  |       {res['min']:.4f}                |\n")
            f.write(f"| Max  |       {res['max']:.4f}                |\n")
            f.write(f"\nOriginal leaky checkpoint: {res['original_leaky_accuracy']:.4f}\n")
            if res["original_inside_range"]:
                f.write(f"Status: ✓ Inside range [{res['min']:.4f}, {res['max']:.4f}] "
                       f"→ plausibly just variance\n")
            else:
                f.write(f"Status: ✗ Outside range [{res['min']:.4f}, {res['max']:.4f}] "
                       f"→ likely a real leakage effect\n")
            f.write("\n")

    print(f"\n\nReport written to: {report_path}")

    # Write JSON results
    json_path = results_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON results written to: {json_path}")


if __name__ == "__main__":
    main()
