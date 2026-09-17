#!/usr/bin/env python3
"""
ICCAD5 Class-Imbalance Ablation: trains the two NEW conditions needed to isolate the
effect of class-imbalance handling on the dense_penultimate bottleneck (see
docs/ICCAD5_IMBALANCE_ABLATION.md for full design rationale):

  A. BASELINE           -- best-checkpoint selection, NO class weighting, standard
                            (natural class-distribution) batches.
  C. BALANCED SAMPLING   -- best-checkpoint selection, NO class weighting, batches drawn
                            50/50 HS/NHS via a balanced sampler.

Condition B (CLASS-WEIGHTED) is NOT retrained here -- it is exactly the already-existing
models/xai/retrained/xai_cnn_iccad5_best.keras from docs/ICCAD5_CHECKPOINT_FIX.md (same
architecture, same seed, same best-checkpoint callback, class_weight applied) and is
reused as-is, per the instruction to reuse existing utilities/results rather than
duplicate work.

Architecture, optimizer, loss, learning rate, epochs, batch size, preprocessing, and the
train/test split are identical to the existing production training pipeline
(src/train_xai.py). Neither the production checkpoint nor any existing XAI/occlusion/
geometry result is modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.data import data_extractor, dataset_analysis
from src.metrics import BestBalancedAccuracyCheckpoint, save_metrics_json, save_confusion_matrix_png
from src.model_xai import build_xai_cnn
from src.train_xai import evaluate_xai_model, measure_inference_speed, TRAIN_METRICS, OPTIMIZER, LOSS, IMG_SIZE, BATCH_SIZE

SEED = 42
EPOCHS = 10  # BENCHMARK_EPOCHS[5] in src/train_xai.py
DATA_ROOT = ROOT / "iccad-official" / "iccad5"
MODELS_OUT = ROOT / "models" / "xai" / "experimental"
RESULTS_OUT = ROOT / "results" / "xai" / "iccad5_imbalance_ablation" / "training"


def load_and_preprocess(path: str) -> np.ndarray:
    """Identical to ImageDataGenerator(rescale=1/255).flow_from_directory's per-image
    pipeline: load_img default interpolation='nearest', img_to_array, rescale by 1/255."""
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return arr.astype(np.float32)


def balanced_batch_generator(hs_paths, nhs_paths, batch_size, seed):
    """Yields batches with exactly batch_size//2 HS and batch_size//2 NHS images, drawn
    with replacement (documented: HS pool has only 26 images, far fewer than
    batch_size//2 * steps_per_epoch draws per epoch, so with-replacement sampling is
    required for HS; NHS also sampled with replacement for a uniform, simple procedure
    across both classes -- see docs/ICCAD5_IMBALANCE_ABLATION.md Sec 3 for the exact
    per-epoch coverage implication of this choice)."""
    rng = np.random.default_rng(seed)
    half = batch_size // 2
    while True:
        hs_batch = rng.choice(hs_paths, size=half, replace=True)
        nhs_batch = rng.choice(nhs_paths, size=batch_size - half, replace=True)
        paths = list(hs_batch) + list(nhs_batch)
        labels = np.array([0.0] * half + [1.0] * (batch_size - half), dtype=np.float32)  # 0=HS,1=NHS
        order = rng.permutation(batch_size)
        X = np.stack([load_and_preprocess(paths[i]) for i in order])
        y = labels[order]
        yield X, y


def build_balanced_dataset(hs_paths, nhs_paths, batch_size, seed, steps_per_epoch):
    gen = lambda: balanced_batch_generator(hs_paths, nhs_paths, batch_size, seed)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, *IMG_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
        ),
    )
    return ds.take(steps_per_epoch * EPOCHS)  # bounded so .fit()'s own epoch bookkeeping is exact


def train_condition(name: str, use_balanced_sampling: bool):
    print("\n" + "=" * 76)
    print(f"ICCAD5 imbalance-ablation condition: {name}")
    print("=" * 76)

    tf.keras.utils.set_random_seed(SEED)

    train_dir, val_dir, counts = dataset_analysis(str(DATA_ROOT) + "/")
    print(f"Dataset counts: {counts}")

    _, val_data_gen = data_extractor(train_dir, val_dir, batch_size=BATCH_SIZE)

    model = build_xai_cnn(input_shape=(*IMG_SIZE, 3))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=TRAIN_METRICS)

    best_ckpt_cb = BestBalancedAccuracyCheckpoint()

    if use_balanced_sampling:
        hs_paths = sorted(str(p) for p in (Path(train_dir) / "train_hs").glob("*"))
        nhs_paths = sorted(str(p) for p in (Path(train_dir) / "train_nhs").glob("*"))
        steps_per_epoch = -(-counts["total_train"] // BATCH_SIZE)  # ceil, matches flow_from_directory's default step count
        print(f"Balanced sampling: {len(hs_paths)} HS / {len(nhs_paths)} NHS train images available; "
              f"steps_per_epoch={steps_per_epoch} (matched to the standard-generator step count), "
              f"batch composition = {BATCH_SIZE // 2} HS + {BATCH_SIZE - BATCH_SIZE // 2} NHS per batch (with replacement).")
        train_ds = build_balanced_dataset(hs_paths, nhs_paths, BATCH_SIZE, SEED, steps_per_epoch)
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_data_gen,
            callbacks=[best_ckpt_cb],
        )
    else:
        train_data_gen, _ = data_extractor(train_dir, val_dir, batch_size=BATCH_SIZE)
        history = model.fit(
            train_data_gen,
            epochs=EPOCHS,
            validation_data=val_data_gen,
            callbacks=[best_ckpt_cb],  # NOTE: no class_weight -- isolates the sampling/weighting variable
        )

    final_metrics, y_true, y_pred, probs = evaluate_xai_model(model, val_data_gen)
    latency_info = measure_inference_speed(model, val_data_gen)

    final_metrics.update({
        "condition": name,
        "benchmark": "iccad5",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "seed": SEED,
        "class_weight_applied": False,
        "balanced_sampling_applied": use_balanced_sampling,
        "dataset_counts": counts,
        "latency": latency_info,
        "checkpoint_selection": {
            "criterion": "best validation balanced accuracy (same BestBalancedAccuracyCheckpoint as production)",
            "selected_epoch": best_ckpt_cb.best_epoch,
            "selected_epoch_val_balanced_accuracy": best_ckpt_cb.best_value,
        },
        "history_val_true_positives": history.history.get("val_true_positives"),
        "history_val_true_negatives": history.history.get("val_true_negatives"),
        "history_val_false_positives": history.history.get("val_false_positives"),
        "history_val_false_negatives": history.history.get("val_false_negatives"),
    })

    bench_dir = RESULTS_OUT / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    save_metrics_json(final_metrics, bench_dir / "metrics.json")
    save_confusion_matrix_png(final_metrics, bench_dir / "confusion_matrix.png",
                               title=f"iccad5 ({name}): Confusion Matrix (HS positive)")

    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_OUT / f"xai_cnn_iccad5_{name}.keras"
    model.save(model_path)
    print(f"Saved -> {model_path}")
    print(f"[{name}] Balanced Acc: {final_metrics['balanced_accuracy']:.4f} | Selected epoch: {best_ckpt_cb.best_epoch} "
          f"| TP={final_metrics['confusion_matrix']['tp']} TN={final_metrics['confusion_matrix']['tn']} "
          f"FP={final_metrics['confusion_matrix']['fp']} FN={final_metrics['confusion_matrix']['fn']}")

    del model
    tf.keras.backend.clear_session()
    return final_metrics


def main():
    train_condition("baseline_noweight", use_balanced_sampling=False)
    train_condition("balanced_sampling", use_balanced_sampling=True)


if __name__ == "__main__":
    main()
