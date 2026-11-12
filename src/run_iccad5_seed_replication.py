#!/usr/bin/env python3
"""
ICCAD5 Class-Imbalance Ablation -- SEED REPLICATION.

Replicates the two non-class-weighted conditions from docs/ICCAD5_IMBALANCE_ABLATION.md
(baseline_noweight, balanced_sampling) across independent random seeds, to test whether
the collapse-elimination / HS-elevated-unit findings are reproducible rather than a
one-off artifact of a single training run.

This script is a direct generalization of src/run_iccad5_imbalance_ablation.py: the ONLY
difference is that `seed` is now a parameter (previously hardcoded to 42) threaded through
BOTH (a) tf.keras.utils.set_random_seed (weight init + ImageDataGenerator shuffling, which
draws on the global numpy RNG state) and (b) the balanced-sampling generator's own
np.random.default_rng(seed) (draw order of HS/NHS images per batch). Architecture,
optimizer, loss, learning rate, batch size, epoch budget, preprocessing, and the
train/test split directory structure are all identical and untouched.

Seeds used (all != 42, the original imbalance-ablation seed):
    balanced_sampling: 101, 202, 303
    baseline_noweight: 101, 202          (secondary control, compute permitting)

The original condition B (class_weighted) is not retrained -- it is not part of this
replication (see docs/ICCAD5_SEED_REPLICATION.md Sec 2-3 for why).

MIGRATED (leakage fix): This script previously used ``dataset_analysis`` + ``data_extractor``,
    which map ``validation_dir`` to ``iccad5/test/`` — so ``BestBalancedAccuracyCheckpoint``
    was selecting checkpoints using test-set metrics across ALL seeds. See
    ``docs/TEST_VALIDATION_LEAKAGE_AUDIT.md`` (HIGH severity, §3.2) for the full trace.

    It now uses ``src/data.py::make_three_way_split``, called with the SAME ``seed`` used for
    ``tf.keras.utils.set_random_seed`` at each run, so the train/val carve-out split moves
    together with the model's own seed instead of being fixed across replications.
    ``BestBalancedAccuracyCheckpoint`` and ``model.fit(validation_data=...)`` monitor the
    resulting stratified 15% carve-out of ``iccad5/train/``, and ``iccad5/test/`` is read
    exactly once, after training, for the reported final metrics. The balanced-sampling
    condition's HS/NHS path pools are drawn from the same train-subset generator (via
    ``_train_subset_paths``), so they no longer include the val carve-out or test images.

    Only the data source has changed — the per-seed replication design, architecture,
    optimizer, loss, epochs, and batch size are untouched.

    See also: ``src/run_iccad5_imbalance_ablation.py`` (same migration).
"""


from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.data import make_three_way_split
from src.metrics import BestBalancedAccuracyCheckpoint, save_metrics_json, save_confusion_matrix_png
from src.model_xai import build_xai_cnn
from src.train_xai import evaluate_xai_model, measure_inference_speed, TRAIN_METRICS, OPTIMIZER, LOSS, IMG_SIZE, BATCH_SIZE

EPOCHS = 10  # BENCHMARK_EPOCHS[5] in src/train_xai.py -- unchanged
DATA_ROOT = ROOT / "iccad-official" / "iccad5"
MODELS_OUT = ROOT / "models" / "xai" / "experimental"
RESULTS_OUT = ROOT / "results" / "xai" / "iccad5_seed_replication" / "training"

ORIGINAL_SEED = 42  # the seed used for the original baseline_noweight / balanced_sampling runs

RUNS = [
    # (run_name, use_balanced_sampling, seed)
    ("balanced_sampling_seed1", True, 101),
    ("balanced_sampling_seed2", True, 202),
    ("balanced_sampling_seed3", True, 303),
    ("baseline_noweight_seed1", False, 101),
    ("baseline_noweight_seed2", False, 202),
]


def load_and_preprocess(path: str) -> np.ndarray:
    """Identical to ImageDataGenerator(rescale=1/255).flow_from_directory's per-image
    pipeline: load_img default interpolation='nearest', img_to_array, rescale by 1/255."""
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return arr.astype(np.float32)


def balanced_batch_generator(hs_paths, nhs_paths, batch_size, seed):
    """Yields batches with exactly batch_size//2 HS and batch_size//2 NHS images, drawn
    with replacement. Identical logic to run_iccad5_imbalance_ablation.py; `seed` now
    varies per replication run instead of being fixed at 42."""
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
    return ds.take(steps_per_epoch * EPOCHS)


def _train_subset_paths(train_gen):
    """Return (hs_paths, nhs_paths) drawn only from the 85% train subset behind `train_gen`.

    Reads flow_from_dataframe's `.filepaths` / `.classes` / `.class_indices` so the
    balanced sampler never sees the val carve-out or test/ images that
    make_three_way_split set aside for `train_gen`'s sibling generators.
    """
    hs_label = train_gen.class_indices["train_hs"]
    hs_paths = [p for p, c in zip(train_gen.filepaths, train_gen.classes) if c == hs_label]
    nhs_paths = [p for p, c in zip(train_gen.filepaths, train_gen.classes) if c != hs_label]
    return sorted(hs_paths), sorted(nhs_paths)


def train_condition(name: str, use_balanced_sampling: bool, seed: int):
    print("\n" + "=" * 76)
    print(f"ICCAD5 seed-replication run: {name}  (seed={seed}, balanced_sampling={use_balanced_sampling})")
    print("=" * 76)

    assert seed != ORIGINAL_SEED, f"seed {seed} must differ from the original run's seed {ORIGINAL_SEED}"

    tf.keras.utils.set_random_seed(seed)

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
    print(f"Dataset counts: {counts}")

    model = build_xai_cnn(input_shape=(*IMG_SIZE, 3))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=TRAIN_METRICS)
    init_weight_checksum = float(np.sum([np.abs(w).sum() for w in model.get_weights()]))

    best_ckpt_cb = BestBalancedAccuracyCheckpoint()

    if use_balanced_sampling:
        hs_paths, nhs_paths = _train_subset_paths(train_gen)
        steps_per_epoch = -(-train_subset_total // BATCH_SIZE)
        print(f"Balanced sampling: {len(hs_paths)} HS / {len(nhs_paths)} NHS train-subset images; "
              f"steps_per_epoch={steps_per_epoch}, seed={seed}")
        train_ds = build_balanced_dataset(hs_paths, nhs_paths, BATCH_SIZE, seed, steps_per_epoch)
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[best_ckpt_cb],
        )
    else:
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[best_ckpt_cb],
        )

    final_metrics, y_true, y_pred, probs = evaluate_xai_model(model, test_gen)
    latency_info = measure_inference_speed(model, test_gen)

    final_metrics.update({
        "run_name": name,
        "condition_family": "balanced_sampling" if use_balanced_sampling else "baseline_noweight",
        "benchmark": "iccad5",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "seed": seed,
        "init_weight_abs_sum_checksum": init_weight_checksum,
        "class_weight_applied": False,
        "balanced_sampling_applied": use_balanced_sampling,
        "dataset_counts": counts,
        "latency": latency_info,
        "checkpoint_selection": {
            "criterion": (
                "best validation balanced accuracy, computed on the val CARVE-OUT "
                "(15% of iccad5/train/, stratified per class, seeded with this run's "
                "own seed) — NOT on iccad5/test/. See BestBalancedAccuracyCheckpoint "
                "in src/metrics.py and docs/THREE_WAY_SPLIT.md."
            ),
            "val_set": "carve-out (iccad5/train/ subset)",
            "val_set_n_hs": split_manifest["val_subset"]["n_hs"],
            "val_set_n_nhs": split_manifest["val_subset"]["n_nhs"],
            "val_frac": split_manifest["split_design"]["val_frac"],
            "split_seed": split_manifest["split_design"]["seed"],
            "selected_epoch": best_ckpt_cb.best_epoch,
            "selected_epoch_val_balanced_accuracy": best_ckpt_cb.best_value,
            "split_manifest_file": "split_manifest.json",
        },
        "history_val_true_positives": history.history.get("val_true_positives"),
        "history_val_true_negatives": history.history.get("val_true_negatives"),
        "history_val_false_positives": history.history.get("val_false_positives"),
        "history_val_false_negatives": history.history.get("val_false_negatives"),
    })

    bench_dir = RESULTS_OUT / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    with open(bench_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)
    save_metrics_json(final_metrics, bench_dir / "metrics.json")
    save_confusion_matrix_png(final_metrics, bench_dir / "confusion_matrix.png",
                               title=f"iccad5 ({name}, seed={seed}): Confusion Matrix (HS positive)")

    MODELS_OUT.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_OUT / f"xai_cnn_iccad5_{name}.keras"
    model.save(model_path)
    print(f"Saved -> {model_path}")
    print(f"[{name}] seed={seed} init_checksum={init_weight_checksum:.4f} "
          f"Balanced Acc: {final_metrics['balanced_accuracy']:.4f} | Selected epoch: {best_ckpt_cb.best_epoch} "
          f"| TP={final_metrics['confusion_matrix']['tp']} TN={final_metrics['confusion_matrix']['tn']} "
          f"FP={final_metrics['confusion_matrix']['fp']} FN={final_metrics['confusion_matrix']['fn']}")

    del model
    tf.keras.backend.clear_session()
    return final_metrics


def main():
    checksums = {}
    for name, use_balanced, seed in RUNS:
        m = train_condition(name, use_balanced, seed)
        checksums[name] = m["init_weight_abs_sum_checksum"]
    print("\n" + "=" * 76)
    print("Initial-weight checksums (must all differ from each other AND from the original")
    print("seed=42 run to confirm seeds actually changed initialization):")
    for k, v in checksums.items():
        print(f"  {k}: {v:.6f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
