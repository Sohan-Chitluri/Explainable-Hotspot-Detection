#!/usr/bin/env python3
"""
Compare XAI models trained with leaky vs clean split on the official test set.

Procedure:
1. Loads old models (trained with test as validation)
2. Loads new models (trained with clean 3-way split)
3. Evaluates both on the official iccad{N}/test/ directory
4. Generates a comparison table showing delta in balanced accuracy
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import binary_metrics


def load_and_evaluate_model(
    model_path: Path,
    test_dir: Path,
    batch_size: int = 32,
    img_size: tuple = (224, 224),
) -> dict | None:
    """Load a model and evaluate it on a test directory."""
    if not model_path.exists():
        return None

    try:
        model = tf.keras.models.load_model(model_path)

        test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
            directory=str(test_dir),
            batch_size=batch_size,
            shuffle=False,
            target_size=img_size,
            class_mode="binary",
        )

        test_gen.reset()
        probs = model.predict(test_gen, verbose=0)
        y_true = test_gen.classes
        y_pred = (probs.ravel() >= 0.5).astype(int)

        class_indices = dict(test_gen.class_indices)
        hs_label = class_indices.get("test_hs", 0)

        metrics = binary_metrics(y_true, y_pred, positive_label=hs_label)
        metrics["n_samples"] = int(len(y_true))

        del model
        del test_gen
        tf.keras.backend.clear_session()

        return metrics
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return None


def main():
    root = Path(__file__).resolve().parents[1]
    data_root = root / "iccad-official"
    old_models_dir = root / "models" / "xai"
    new_models_dir = root / "models" / "xai" / "retrained_clean"

    if not new_models_dir.exists():
        print(f"Error: New models directory not found: {new_models_dir}")
        print(
            "Run: python -m src.train_xai --models-dir models/xai/retrained_clean"
        )
        sys.exit(1)

    benchmarks = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

    print("\n" + "=" * 100)
    print("COMPARING OLD (LEAKY) vs NEW (CLEAN SPLIT) XAI MODELS")
    print("=" * 100)

    comparison = {}
    results = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "comparison_table": {},
        "detailed_metrics": {},
    }

    print(f"\n{'Benchmark':<12} {'Old Bal Acc':<14} {'New Bal Acc':<14} {'Delta':<14} {'Status':<20}")
    print("-" * 74)

    for benchmark_name in benchmarks:
        test_dir = data_root / benchmark_name / "test"

        old_model_path = old_models_dir / f"xai_cnn_{benchmark_name}.keras"
        new_model_path = new_models_dir / f"xai_cnn_{benchmark_name}.keras"

        print(f"[{benchmark_name}] Evaluating models...", end=" ", flush=True)

        old_metrics = load_and_evaluate_model(old_model_path, test_dir)
        new_metrics = load_and_evaluate_model(new_model_path, test_dir)

        if old_metrics is None or new_metrics is None:
            status = "FAILED" if old_metrics is None else "NEW OK"
            print(status)
            continue

        old_bal_acc = old_metrics["balanced_accuracy"]
        new_bal_acc = new_metrics["balanced_accuracy"]
        delta = new_bal_acc - old_bal_acc

        status = f"+{delta:.4f} better" if delta > 0 else f"{delta:.4f}"
        print(f"Done")

        print(
            f"{benchmark_name:<12} {old_bal_acc:.4f}         {new_bal_acc:.4f}         {status:<20}"
        )

        comparison[benchmark_name] = {
            "old_balanced_accuracy": float(old_bal_acc),
            "new_balanced_accuracy": float(new_bal_acc),
            "delta": float(delta),
            "leakage_impact": (
                f"The old model was selected based on test-set metrics, which inflated "
                f"its reported accuracy by ~{abs(delta):.4f} (or {abs(delta)*100:.2f}%). "
                f"The new model uses a proper validation split from training data only."
            ),
        }

        results["detailed_metrics"][benchmark_name] = {
            "old": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in old_metrics.items()
            },
            "new": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in new_metrics.items()
            },
        }

    print("-" * 74)

    # Calculate summary statistics
    deltas = [
        comparison[b]["delta"]
        for b in comparison.keys()
        if "delta" in comparison[b]
    ]
    old_accs = [
        comparison[b]["old_balanced_accuracy"]
        for b in comparison.keys()
        if "old_balanced_accuracy" in comparison[b]
    ]
    new_accs = [
        comparison[b]["new_balanced_accuracy"]
        for b in comparison.keys()
        if "new_balanced_accuracy" in comparison[b]
    ]

    if deltas:
        avg_delta = np.mean(deltas)
        print(
            f"{'AVERAGE':<12} {np.mean(old_accs):.4f}         {np.mean(new_accs):.4f}         "
            f"{avg_delta:+.4f} ({avg_delta*100:+.2f}%)"
        )

        results["comparison_table"] = comparison
        results["summary"] = {
            "old_average_balanced_accuracy": float(np.mean(old_accs)),
            "new_average_balanced_accuracy": float(np.mean(new_accs)),
            "average_delta": float(avg_delta),
            "average_delta_percent": float(avg_delta * 100),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "interpretation": (
                f"The old models were biased high by checkpoint selection on the test set. "
                f"On average, the reported accuracy was inflated by {abs(avg_delta)*100:.2f}%. "
                f"The new models with clean 3-way split show more realistic performance estimates."
            ),
        }

    # Save results
    results_file = root / "results" / "xai_training" / "leaky_vs_clean_comparison.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
