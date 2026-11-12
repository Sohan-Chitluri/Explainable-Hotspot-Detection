#!/usr/bin/env python3
"""
Factorial Ablation Training and Evaluation for ICCAD-5 Penultimate Representation Collapse.

Designs a 2x2 Factorial Experiment across independent random seeds (101, 202, 303):
- Factor 1: Batch sampling in {natural, balanced}
- Factor 2: Loss weighting in {unweighted, class_weighted}

Configurations:
  Config A: Natural class distribution + Unweighted loss
  Config B: Natural class distribution + Class-weighted loss
  Config C: Balanced HS/NHS sampling   + Unweighted loss
  Config D: Balanced HS/NHS sampling   + Class-weighted loss

Strict experimental integrity:
- Uses make_three_way_split (15% train carve-out for validation, seed-matched).
- BestBalancedAccuracyCheckpoint restores weights from val carve-out only.
- Official ICCAD-5 test partition (41 HS, 19,327 NHS) evaluated once post-training.
- Exposure counts tracked for all training HS instances.
- Penultimate representations, norms, dead units, and CAM readiness analyzed per model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from src.data import make_three_way_split
from src.metrics import BestBalancedAccuracyCheckpoint, binary_metrics
from src.model_xai import build_xai_cnn
from src.train_xai import compute_class_weights, TRAIN_METRICS, OPTIMIZER, LOSS, IMG_SIZE, BATCH_SIZE

EPOCHS = 10
DATA_ROOT = ROOT / "iccad-official" / "iccad5"
RESULTS_DIR = ROOT / "results" / "iccad5_factorial"
MODELS_DIR = RESULTS_DIR / "models"
EVALS_DIR = RESULTS_DIR / "evaluations"


def get_config_details(config_id: str) -> tuple[str, bool, bool]:
    cid = config_id.upper()
    if cid == "A":
        return "natural_unweighted", False, False
    elif cid == "B":
        return "natural_class_weighted", False, True
    elif cid == "C":
        return "balanced_unweighted", True, False
    elif cid == "D":
        return "balanced_class_weighted", True, True
    else:
        raise ValueError(f"Unknown config: {config_id}")


def train_and_eval_single_run(config_id: str, seed: int, data_cache: dict | None = None) -> dict:
    config_name, use_balanced, use_class_weight = get_config_details(config_id)
    print("\n" + "=" * 78)
    print(f"RUN: Config {config_id} ({config_name}) | Seed {seed} | Balanced={use_balanced} | Weighted={use_class_weight}")
    print("=" * 78)

    # 1. Deterministic seeding
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    # 2. Dataset split (15% val carve-out)
    split = make_three_way_split(
        DATA_ROOT,
        "iccad5",
        val_frac=0.15,
        seed=seed,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
    )
    train_gen, val_gen, test_gen = split.train_gen, split.val_gen, split.test_gen
    counts = split.counts
    split_manifest = split.split_manifest
    train_subset_total = split_manifest["train_subset"]["total"]
    steps_per_epoch = int(np.ceil(train_subset_total / BATCH_SIZE))

    # Preload or use cached images to avoid disk I/O bottleneck
    if data_cache is not None and data_cache.get("seed") == seed:
        hs_train_imgs = data_cache["hs_train_imgs"]
        hs_train_paths = data_cache["hs_train_paths"]
        nhs_train_imgs = data_cache["nhs_train_imgs"]
        nhs_train_paths = data_cache["nhs_train_paths"]
        X_val = data_cache["X_val"]
        y_val = data_cache["y_val"]
        all_train_imgs = data_cache["all_train_imgs"]
        all_train_labels = data_cache["all_train_labels"]
    else:
        print("Caching train & val subsets in RAM for seed", seed)
        hs_label = train_gen.class_indices["train_hs"]
        hs_train_paths = sorted([p for p, c in zip(train_gen.filepaths, train_gen.classes) if c == hs_label])
        nhs_train_paths = sorted([p for p, c in zip(train_gen.filepaths, train_gen.classes) if c != hs_label])

        hs_train_imgs = np.stack([img_to_array(load_img(p, target_size=IMG_SIZE, interpolation="nearest")) / 255.0 for p in hs_train_paths])
        nhs_train_imgs = np.stack([img_to_array(load_img(p, target_size=IMG_SIZE, interpolation="nearest")) / 255.0 for p in nhs_train_paths])

        # All training data for natural generator
        all_train_paths = list(train_gen.filepaths)
        all_train_labels = np.array(train_gen.classes, dtype=np.float32)
        all_train_imgs = np.stack([img_to_array(load_img(p, target_size=IMG_SIZE, interpolation="nearest")) / 255.0 for p in all_train_paths])

        # Validation data
        val_paths = list(val_gen.filepaths)
        y_val = np.array(val_gen.classes, dtype=np.float32)
        X_val = np.stack([img_to_array(load_img(p, target_size=IMG_SIZE, interpolation="nearest")) / 255.0 for p in val_paths])

        data_cache = {
            "seed": seed,
            "hs_train_imgs": hs_train_imgs,
            "hs_train_paths": hs_train_paths,
            "nhs_train_imgs": nhs_train_imgs,
            "nhs_train_paths": nhs_train_paths,
            "X_val": X_val,
            "y_val": y_val,
            "all_train_imgs": all_train_imgs,
            "all_train_labels": all_train_labels,
        }

    n_hs_pool = len(hs_train_paths)
    n_nhs_pool = len(nhs_train_paths)
    print(f"Train subset: {n_hs_pool} HS, {n_nhs_pool} NHS. Steps per epoch: {steps_per_epoch}")

    # Track HS exposures
    hs_exposure_counts = {os.path.basename(p): 0 for p in hs_train_paths}

    # 3. Model construction & compilation
    model = build_xai_cnn(input_shape=(*IMG_SIZE, 3))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=TRAIN_METRICS)
    init_checksum = float(np.sum([np.abs(w).sum() for w in model.get_weights()]))

    class_weights = compute_class_weights(counts) if use_class_weight else None
    best_ckpt_cb = BestBalancedAccuracyCheckpoint()

    t_start = time.time()
    if use_balanced:
        # Balanced batch generator with replacement and exposure logging
        rng = np.random.default_rng(seed)
        half = BATCH_SIZE // 2

        def balanced_gen():
            while True:
                idx_hs = rng.choice(n_hs_pool, size=half, replace=True)
                idx_nhs = rng.choice(n_nhs_pool, size=BATCH_SIZE - half, replace=True)
                for i in idx_hs:
                    hs_exposure_counts[os.path.basename(hs_train_paths[i])] += 1
                
                batch_x = np.concatenate([hs_train_imgs[idx_hs], nhs_train_imgs[idx_nhs]], axis=0)
                batch_y = np.array([0.0] * half + [1.0] * (BATCH_SIZE - half), dtype=np.float32)
                perm = rng.permutation(BATCH_SIZE)
                yield batch_x[perm], batch_y[perm]

        train_ds = tf.data.Dataset.from_generator(
            balanced_gen,
            output_signature=(
                tf.TensorSpec(shape=(BATCH_SIZE, *IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE,), dtype=tf.float32),
            ),
        ).take(steps_per_epoch * EPOCHS)

        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[best_ckpt_cb],
            verbose=1,
        )
    else:
        # Natural distribution generator with exposure logging
        # In natural sampling, every training image is seen exactly 1 time per epoch
        for p in hs_train_paths:
            hs_exposure_counts[os.path.basename(p)] += EPOCHS

        # Create shuffled dataset from in-memory arrays
        train_ds = tf.data.Dataset.from_tensor_slices((all_train_imgs, all_train_labels))
        train_ds = train_ds.shuffle(buffer_size=len(all_train_imgs), seed=seed).batch(BATCH_SIZE)

        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[best_ckpt_cb],
            verbose=1,
        )

    t_train = time.time() - t_start
    print(f"Training completed in {t_train:.1f}s. Selected epoch: {best_ckpt_cb.best_epoch} (val bal_acc: {best_ckpt_cb.best_value:.4f})")

    # Save model checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"xai_cnn_iccad5_{config_id}_seed{seed}.keras"
    model.save(str(model_path))

    # 4. Evaluation on Official Test Set (Untouched)
    print("Evaluating official test set (19,368 images)...")
    test_gen.reset()
    probs = model.predict(test_gen, verbose=0).ravel()
    y_true = np.array(test_gen.classes, dtype=int)
    y_pred = (probs >= 0.5).astype(int)

    cls_metrics = binary_metrics(y_true, y_pred, positive_label=0)

    # 5. Extract Penultimate Representations & Dead Units
    pen_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer("dense_penultimate").output, model.output])
    
    # Evaluate 41 HS specifically
    hs_test_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(DATA_ROOT / "test"),
        classes=["test_hs"],
        batch_size=41,
        shuffle=False,
        target_size=IMG_SIZE,
        class_mode=None,
    )
    hs_test_imgs = next(hs_test_gen)
    hs_pen_acts, hs_probs = pen_model.predict(hs_test_imgs, verbose=0)
    
    # Sample 500 NHS for representation statistics
    nhs_test_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
        str(DATA_ROOT / "test"),
        classes=["test_nhs"],
        batch_size=500,
        shuffle=False,
        target_size=IMG_SIZE,
        class_mode=None,
    )
    nhs_test_imgs = next(nhs_test_gen)
    nhs_pen_acts, nhs_probs = pen_model.predict(nhs_test_imgs, verbose=0)

    # Weights and biases
    out_w, out_b = model.get_layer("dense_output").get_weights()
    b_out = float(out_b[0])

    # Collapse metrics
    hs_exact_zeros = int(np.sum(np.all(hs_pen_acts == 0.0, axis=1)))
    hs_near_zeros = int(np.sum(np.linalg.norm(hs_pen_acts, axis=1) < 1e-3))
    nhs_exact_zeros = int(np.sum(np.all(nhs_pen_acts == 0.0, axis=1)))
    nhs_near_zeros = int(np.sum(np.linalg.norm(nhs_pen_acts, axis=1) < 1e-3))

    hs_norms = np.linalg.norm(hs_pen_acts, axis=1)
    nhs_norms = np.linalg.norm(nhs_pen_acts, axis=1)

    # Dead units across the test subset (HS + NHS)
    comb_acts = np.concatenate([hs_pen_acts, nhs_pen_acts], axis=0)
    dead_units = [int(u) for u in range(16) if np.all(comb_acts[:, u] == 0.0)]

    # Output contributions
    col_mask = np.all(hs_pen_acts == 0.0, axis=1)
    feat_contrib = float(np.mean(np.dot(hs_pen_acts[col_mask], out_w))) if np.any(col_mask) else 0.0

    # HS exposure summary
    exp_vals = list(hs_exposure_counts.values())

    result_record = {
        "config_id": config_id,
        "config_name": config_name,
        "batch_sampling": "balanced" if use_balanced else "natural",
        "loss_weighting": "class_weighted" if use_class_weight else "unweighted",
        "seed": seed,
        "selected_epoch": int(best_ckpt_cb.best_epoch),
        "selected_val_balanced_accuracy": float(best_ckpt_cb.best_value),
        "train_time_sec": round(t_train, 2),
        "init_weight_checksum": init_checksum,
        "b_out": b_out,
        "sigma_b_out": float(1.0 / (1.0 + np.exp(-b_out))),
        # Test performance
        "tp": int(cls_metrics["confusion_matrix"]["tp"]),
        "tn": int(cls_metrics["confusion_matrix"]["tn"]),
        "fp": int(cls_metrics["confusion_matrix"]["fp"]),
        "fn": int(cls_metrics["confusion_matrix"]["fn"]),
        "recall_hs": float(cls_metrics["recall_sensitivity"]),
        "specificity_nhs": float(cls_metrics["specificity"]),
        "precision": float(cls_metrics["precision"]),
        "f1": float(cls_metrics["f1_score"]),
        "balanced_accuracy": float(cls_metrics["balanced_accuracy"]),
        # Collapse statistics
        "n_hs_test": 41,
        "hs_exact_zero_count": hs_exact_zeros,
        "hs_exact_zero_pct": round(hs_exact_zeros / 41.0 * 100.0, 3),
        "hs_near_zero_count": hs_near_zeros,
        "hs_near_zero_pct": round(hs_near_zeros / 41.0 * 100.0, 3),
        "nhs_exact_zero_count_500": nhs_exact_zeros,
        "nhs_exact_zero_pct_500": round(nhs_exact_zeros / 500.0 * 100.0, 3),
        "hs_norm_mean": float(np.mean(hs_norms)),
        "hs_norm_std": float(np.std(hs_norms)),
        "hs_norm_median": float(np.median(hs_norms)),
        "nhs_norm_mean": float(np.mean(nhs_norms)),
        "dead_unit_count": len(dead_units),
        "dead_unit_indices": dead_units,
        "dead_unit_fraction": round(len(dead_units) / 16.0, 4),
        "collapsed_feature_contrib_mean": feat_contrib,
        # Exposure metrics
        "unique_hs_train_images": n_hs_pool,
        "unique_hs_seen": int(np.sum(np.array(exp_vals) > 0)),
        "hs_exposure_total": int(np.sum(exp_vals)),
        "hs_exposure_min": int(np.min(exp_vals)),
        "hs_exposure_max": int(np.max(exp_vals)),
        "hs_exposure_mean": round(float(np.mean(exp_vals)), 2),
        "hs_exposure_std": round(float(np.std(exp_vals)), 2),
        "checkpoint_path": str(model_path.relative_to(ROOT)),
    }

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    eval_json = EVALS_DIR / f"{config_id}_seed{seed}_metrics.json"
    with open(eval_json, "w") as f:
        json.dump(result_record, f, indent=2)

    return result_record, data_cache


def main():
    parser = argparse.ArgumentParser(description="Run 2x2 Factorial Ablation on ICCAD-5.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--configs", type=str, nargs="+", default=["A", "B", "C", "D"])
    args = parser.parse_args()

    all_records = []
    for seed in args.seeds:
        data_cache = None
        for config_id in args.configs:
            record, data_cache = train_and_eval_single_run(config_id, seed, data_cache)
            all_records.append(record)

    df = pd.DataFrame(all_records)
    out_csv = RESULTS_DIR / "factorial_matrix_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 78)
    print(f"All {len(all_records)} factorial runs completed! Summary saved to {out_csv}")
    print("=" * 78)
    print(df[["config_id", "batch_sampling", "loss_weighting", "seed", "hs_exact_zero_pct", "recall_hs", "specificity_nhs", "balanced_accuracy"]].to_string())


if __name__ == "__main__":
    main()
