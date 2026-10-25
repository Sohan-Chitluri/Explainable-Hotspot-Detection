"""OpenHSDBench Dataset Adapter for Via-3.

Provides three-way stratified split (train, validation carve-out, and held-out test)
with explicit semantic label mappings for OpenHSDBench:
  0 -> Non-Hotspot (NHS)
  1 -> Hotspot (HS)

Never touches test sets for validation, model selection, or threshold tuning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image
import tensorflow as tf


class OpenHSDSplit(NamedTuple):
    """Return value of `make_openhsd_split`."""
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    test_ds: tf.data.Dataset
    counts: dict[str, dict[str, int]]
    manifest: dict


class OpenHSDVia3Data(NamedTuple):
    """Container for via-3 datasets, splits, paths, labels, and class weights."""
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset
    clean_test_ds: tf.data.Dataset
    official_test_ds: tf.data.Dataset
    train_paths: list[str]
    train_labels: list[int]
    val_paths: list[str]
    val_labels: list[int]
    clean_test_paths: list[str]
    clean_test_labels: list[int]
    official_test_paths: list[str]
    official_test_labels: list[int]
    counts: dict[str, dict[str, int]]
    train_class_weights: dict[int, float]
    train_class_weights_softened: dict[int, float]
    manifest_info: dict
    balanced_train_ds: Optional[tf.data.Dataset] = None


def _load_and_preprocess_image(path_str: str, img_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Read grayscale image, resize using nearest neighbor, expand to 3 channels normalized to [0, 1]."""
    try:
        with Image.open(path_str) as im:
            im_gray = im.convert("L")
            im_resized = im_gray.resize(img_size, Image.Resampling.NEAREST)
            arr = np.array(im_resized, dtype=np.float32) / 255.0
            rgb = np.repeat(arr[:, :, np.newaxis], 3, axis=-1)
            return rgb
    except Exception as e:
        raise FileNotFoundError(f"Failed to read image {path_str}: {e}") from e


def _parse_image_and_label(path, label, img_size=(224, 224), augment=False):
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=1)
    img = tf.image.resize(img, img_size, method="nearest")
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.repeat(img, 3, axis=-1)
    img.set_shape((*img_size, 3))
    label = tf.cast(label, tf.float32)
    label.set_shape(())
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
    return img, label


def _create_tf_dataset_native(
    file_paths: list[str],
    labels: list[int],
    batch_size: int = 32,
    shuffle: bool = False,
    img_size: tuple[int, int] = (224, 224),
    seed: int = 42,
    augment: bool = False,
) -> tf.data.Dataset:
    """Build a high-performance tf.data pipeline using native TF operations."""
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(file_paths), 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: _parse_image_and_label(p, l, img_size=img_size, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _create_balanced_tf_dataset(
    file_paths: list[str],
    labels: list[int],
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    seed: int = 42,
    augment: bool = False,
) -> tf.data.Dataset:
    """Build a balanced 50/50 batch sampling dataset (equal HS and NHS per batch)."""
    hs_paths = [p for p, l in zip(file_paths, labels) if l == 1]
    hs_labels = [1 for _ in hs_paths]
    nhs_paths = [p for p, l in zip(file_paths, labels) if l == 0]
    nhs_labels = [0 for _ in nhs_paths]

    ds_hs = tf.data.Dataset.from_tensor_slices((hs_paths, hs_labels)).repeat().shuffle(
        buffer_size=min(len(hs_paths), 2000), seed=seed
    ).map(
        lambda p, l: _parse_image_and_label(p, l, img_size=img_size, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds_nhs = tf.data.Dataset.from_tensor_slices((nhs_paths, nhs_labels)).repeat().shuffle(
        buffer_size=min(len(nhs_paths), 10000), seed=seed + 1
    ).map(
        lambda p, l: _parse_image_and_label(p, l, img_size=img_size, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # 50/50 balance
    balanced_ds = tf.data.Dataset.sample_from_datasets([ds_hs, ds_nhs], weights=[0.5, 0.5], seed=seed)
    balanced_ds = balanced_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return balanced_ds


def _stratified_split(
    paths: list[str],
    labels: list[int],
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Perform deterministic stratified split per class using NumPy."""
    rng = np.random.default_rng(seed)

    hs_indices = np.array([i for i, lbl in enumerate(labels) if lbl == 1])
    nhs_indices = np.array([i for i, lbl in enumerate(labels) if lbl == 0])

    rng.shuffle(hs_indices)
    rng.shuffle(nhs_indices)

    n_val_hs = int(round(len(hs_indices) * val_frac))
    n_val_nhs = int(round(len(nhs_indices) * val_frac))

    val_hs_idx = set(hs_indices[:n_val_hs])
    val_nhs_idx = set(nhs_indices[:n_val_nhs])
    val_set = val_hs_idx | val_nhs_idx

    tr_paths = [p for i, p in enumerate(paths) if i not in val_set]
    tr_labels = [l for i, l in enumerate(labels) if i not in val_set]
    val_paths = [p for i, p in enumerate(paths) if i in val_set]
    val_labels = [l for i, l in enumerate(labels) if i in val_set]

    return tr_paths, val_paths, tr_labels, val_labels


def make_openhsd_split(
    subset_dir: str | Path,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
) -> OpenHSDSplit:
    """Create stratified train / validation / test splits for an OpenHSDBench subset."""
    subset_dir = Path(subset_dir)
    train_json_path = subset_dir / "train.json"
    test_json_path = subset_dir / "test.json"

    if not train_json_path.exists() or not test_json_path.exists():
        raise FileNotFoundError(f"Missing train.json or test.json in {subset_dir}")

    with open(train_json_path, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    train_paths = [str((subset_dir / item["image"]).resolve()) for item in train_raw]
    train_labels = [int(item["label"]) for item in train_raw]

    test_paths = [str((subset_dir / item["image"]).resolve()) for item in test_raw]
    test_labels = [int(item["label"]) for item in test_raw]

    tr_paths, val_paths, tr_labels, val_labels = _stratified_split(
        train_paths,
        train_labels,
        val_frac=val_frac,
        seed=seed,
    )

    counts = {
        "train": {"total": len(tr_paths), "hs": int(np.sum(tr_labels)), "nhs": len(tr_paths) - int(np.sum(tr_labels))},
        "val": {"total": len(val_paths), "hs": int(np.sum(val_labels)), "nhs": len(val_paths) - int(np.sum(val_labels))},
        "test": {"total": len(test_paths), "hs": int(np.sum(test_labels)), "nhs": len(test_paths) - int(np.sum(test_labels))},
    }

    manifest = {
        "subset_dir": str(subset_dir),
        "seed": seed,
        "val_frac": val_frac,
        "counts": counts,
        "val_files": [Path(p).name for p in val_paths],
    }

    train_ds = _create_tf_dataset_native(tr_paths, tr_labels, batch_size=batch_size, shuffle=True, img_size=img_size, seed=seed)
    val_ds = _create_tf_dataset_native(val_paths, val_labels, batch_size=batch_size, shuffle=False, img_size=img_size)
    test_ds = _create_tf_dataset_native(test_paths, test_labels, batch_size=batch_size, shuffle=False, img_size=img_size)

    return OpenHSDSplit(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        counts=counts,
        manifest=manifest,
    )


def load_via3_manifest_datasets(
    manifest_dir: Optional[str | Path] = None,
    data_dir: str | Path = "/home/peskybird/Projects/data/openhsd/via-3",
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
    seed: int = 42,
    smoke: bool = False,
    smoke_samples: int = 256,
    augment_train: bool = False,
) -> OpenHSDVia3Data:
    """Load group-aware train/val partitions and clean held-out test cohorts for via-3.

    Explicit label semantics:
      0 = Non-Hotspot (NHS)
      1 = Hotspot (HS)
    """
    if manifest_dir is None:
        local_dir = Path("results/dataset_audit")
        if (local_dir / "via3_split_manifest.json").exists():
            manifest_dir = local_dir
        else:
            manifest_dir = Path("/home/peskybird/Projects/openhsd-audit-training/results/dataset_audit")
    else:
        manifest_dir = Path(manifest_dir)

    data_dir = Path(data_dir)

    split_manifest_file = manifest_dir / "via3_split_manifest.json"
    clean_test_file = manifest_dir / "via3_cohort_clean_test.json"
    official_test_file = manifest_dir / "via3_cohort_official_test.json"

    for p in [split_manifest_file, clean_test_file, official_test_file]:
        if not p.exists():
            raise FileNotFoundError(f"Required via-3 manifest not found: {p}")

    with open(split_manifest_file, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    with open(clean_test_file, "r", encoding="utf-8") as f:
        clean_test_manifest = json.load(f)
    with open(official_test_file, "r", encoding="utf-8") as f:
        official_test_manifest = json.load(f)

    def extract_paths_and_labels(rel_paths: list[str]) -> tuple[list[str], list[int]]:
        paths = []
        labels = []
        for rel in rel_paths:
            full = str((data_dir / rel).resolve())
            fn = Path(rel).name
            if fn.startswith("H"):
                lbl = 1
            elif fn.startswith("N"):
                lbl = 0
            else:
                raise ValueError(f"Filename {fn} does not start with expected prefix H or N")
            paths.append(full)
            labels.append(lbl)
        return paths, labels

    tr_rel = split_manifest["train_partition"]["sample_files"]
    val_rel = split_manifest["val_partition"]["sample_files"]
    clean_rel = [x["image"] for x in clean_test_manifest["samples"]]
    off_rel = [x["image"] for x in official_test_manifest["samples"]]

    tr_paths, tr_labels = extract_paths_and_labels(tr_rel)
    val_paths, val_labels = extract_paths_and_labels(val_rel)
    clean_paths, clean_labels = extract_paths_and_labels(clean_rel)
    off_paths, off_labels = extract_paths_and_labels(off_rel)

    # Validate against manifest numbers
    assert len(tr_paths) == split_manifest["train_partition"]["total_samples"], (
        f"Train mismatch: {len(tr_paths)} vs {split_manifest['train_partition']['total_samples']}"
    )
    assert sum(tr_labels) == split_manifest["train_partition"]["hs_samples"], (
        f"Train HS mismatch: {sum(tr_labels)} vs {split_manifest['train_partition']['hs_samples']}"
    )
    assert len(val_paths) == split_manifest["val_partition"]["total_samples"], (
        f"Val mismatch: {len(val_paths)} vs {split_manifest['val_partition']['total_samples']}"
    )
    assert sum(val_labels) == split_manifest["val_partition"]["hs_samples"], (
        f"Val HS mismatch: {sum(val_labels)} vs {split_manifest['val_partition']['hs_samples']}"
    )
    assert len(clean_paths) == clean_test_manifest["total"], "Clean test mismatch"
    assert sum(clean_labels) == clean_test_manifest["hs"], "Clean test HS mismatch"
    assert len(off_paths) == official_test_manifest["total"], "Official test mismatch"
    assert sum(off_labels) == official_test_manifest["hs"], "Official test HS mismatch"

    if smoke:
        def select_smoke_subset(paths: list[str], labels: list[int], max_samples: int):
            hs_idx = [i for i, l in enumerate(labels) if l == 1]
            nhs_idx = [i for i, l in enumerate(labels) if l == 0]
            n_hs = min(len(hs_idx), max(4, max_samples // 8))
            n_nhs = min(len(nhs_idx), max_samples - n_hs)
            sel = hs_idx[:n_hs] + nhs_idx[:n_nhs]
            return [paths[i] for i in sel], [labels[i] for i in sel]

        tr_paths, tr_labels = select_smoke_subset(tr_paths, tr_labels, max_samples=smoke_samples)
        val_paths, val_labels = select_smoke_subset(val_paths, val_labels, max_samples=smoke_samples // 2)
        clean_paths, clean_labels = select_smoke_subset(clean_paths, clean_labels, max_samples=smoke_samples // 2)
        off_paths, off_labels = select_smoke_subset(off_paths, off_labels, max_samples=smoke_samples // 2)

    counts = {
        "train": {"total": len(tr_paths), "hs": int(np.sum(tr_labels)), "nhs": len(tr_paths) - int(np.sum(tr_labels))},
        "val": {"total": len(val_paths), "hs": int(np.sum(val_labels)), "nhs": len(val_paths) - int(np.sum(val_labels))},
        "clean_test": {"total": len(clean_paths), "hs": int(np.sum(clean_labels)), "nhs": len(clean_paths) - int(np.sum(clean_labels))},
        "official_test": {"total": len(off_paths), "hs": int(np.sum(off_labels)), "nhs": len(off_paths) - int(np.sum(off_labels))},
    }

    # Class weights strictly from training partition
    n_total_tr = len(tr_labels)
    n_hs_tr = int(np.sum(tr_labels))
    n_nhs_tr = n_total_tr - n_hs_tr
    w0 = n_total_tr / (2.0 * max(1, n_nhs_tr))
    w1 = n_total_tr / (2.0 * max(1, n_hs_tr))
    train_class_weights = {0: float(w0), 1: float(w1)}
    train_class_weights_softened = {0: float(np.sqrt(w0)), 1: float(np.sqrt(w1))}

    train_ds = _create_tf_dataset_native(
        tr_paths, tr_labels, batch_size=batch_size, shuffle=True, img_size=img_size, seed=seed, augment=augment_train
    )
    val_ds = _create_tf_dataset_native(
        val_paths, val_labels, batch_size=batch_size, shuffle=False, img_size=img_size, augment=False
    )
    clean_test_ds = _create_tf_dataset_native(
        clean_paths, clean_labels, batch_size=batch_size, shuffle=False, img_size=img_size, augment=False
    )
    official_test_ds = _create_tf_dataset_native(
        off_paths, off_labels, batch_size=batch_size, shuffle=False, img_size=img_size, augment=False
    )

    balanced_train_ds = _create_balanced_tf_dataset(
        tr_paths, tr_labels, batch_size=batch_size, img_size=img_size, seed=seed, augment=augment_train
    )

    return OpenHSDVia3Data(
        train_ds=train_ds,
        val_ds=val_ds,
        clean_test_ds=clean_test_ds,
        official_test_ds=official_test_ds,
        train_paths=tr_paths,
        train_labels=tr_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        clean_test_paths=clean_paths,
        clean_test_labels=clean_labels,
        official_test_paths=off_paths,
        official_test_labels=off_labels,
        counts=counts,
        train_class_weights=train_class_weights,
        train_class_weights_softened=train_class_weights_softened,
        manifest_info={
            "manifest_dir": str(manifest_dir),
            "data_dir": str(data_dir),
            "seed": seed,
            "smoke": smoke,
            "batch_size": batch_size,
            "img_size": list(img_size),
            "augment_train": augment_train,
        },
        balanced_train_ds=balanced_train_ds,
    )
