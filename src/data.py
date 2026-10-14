"""Dataset helpers faithful to LHD_CustomModel.ipynb.

Three-way split
---------------
``make_three_way_split`` is the canonical entry point for any training script
that needs a genuine train / validation / test separation:

- **train** : 85 % of ``iccad{N}/train/train_hs`` + ``train_nhs``, shuffled.
- **val**   : 15 % carve-out from ``iccad{N}/train/``, stratified per class,
              selected once with a fixed seed.  Used ONLY for callback / checkpoint
              monitoring and per-epoch curves — never for final reported metrics.
- **test**  : ``iccad{N}/test/`` — read exactly once, after training, for the
              single final evaluation that is reported.

The split is implemented via ``flow_from_dataframe`` with explicit file path
lists so every file assigned to each split is recorded in the returned
``split_manifest`` dict (serialisable to JSON).  No files are moved on disk.

Legacy helpers
--------------
``dataset_analysis`` and ``data_extractor`` are preserved unchanged so that
scripts not yet migrated (``run_iccad5_seed_replication.py``,
``run_iccad5_imbalance_ablation.py``) continue to work.  Those scripts still
point ``validation_data`` at the official test set — see the
``TODO(leakage-fix)`` headers in each of them for the pending migration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------------------------------------------------------
# Legacy helpers (preserved verbatim — do NOT change without updating callers)
# ---------------------------------------------------------------------------

def dataset_analysis(folder: str):
    """Return train/validation dirs and print class counts (notebook logic).

    .. warning::
        ``validation_dir`` returned here is ``iccad{N}/test/``, NOT a
        held-out carve-out of the training data.  This is intentional for
        backward compatibility with scripts not yet migrated to
        ``make_three_way_split``.  See ``docs/THREE_WAY_SPLIT.md``.
    """
    # Notebook uses os.path.dirname(folder). With a trailing slash this keeps
    # the benchmark folder itself (e.g. .../iccad1/), which is intentional.
    base_dir = os.path.join(os.path.dirname(folder))
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "test")

    train_hotspot_dir = os.path.join(train_dir, "train_hs")
    train_not_hotspot_dir = os.path.join(train_dir, "train_nhs")

    validation_hotspot_dir = os.path.join(validation_dir, "test_hs")
    validation_not_hotspot_dir = os.path.join(validation_dir, "test_nhs")

    num_hs_tr = len(os.listdir(train_hotspot_dir))
    num_nhs_tr = len(os.listdir(train_not_hotspot_dir))
    num_hs_val = len(os.listdir(validation_hotspot_dir))
    num_nhs_val = len(os.listdir(validation_not_hotspot_dir))

    total_train = num_hs_tr + num_nhs_tr
    total_val = num_hs_val + num_nhs_val

    print("The dataset contains:")
    print(total_train, "Training images")
    print(total_val, "Validation images")
    print("\nThe training set contains:")
    print(num_hs_tr, "Images with hotspot")
    print(num_nhs_tr, "Images without hotspot")
    print("\nThe validation set contains:")
    print(num_hs_val, "Images with hotspot")
    print(num_nhs_val, "Images without hotspot")
    print()

    counts = {
        "train_hs": num_hs_tr,
        "train_nhs": num_nhs_tr,
        "test_hs": num_hs_val,
        "test_nhs": num_nhs_val,
        "total_train": total_train,
        "total_val": total_val,
    }
    return train_dir, validation_dir, counts


def data_extractor(train_dir: str, validation_dir: str, batch_size: int = 32):
    """ImageDataGenerators matching the notebook (rescale 1/255, 224x224, binary)."""
    image_gen_train = ImageDataGenerator(rescale=1.0 / 255)
    image_gen_val = ImageDataGenerator(rescale=1.0 / 255)

    train_data_gen = image_gen_train.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        shuffle=True,
        target_size=(224, 224),
        class_mode="binary",
    )

    val_data_gen = image_gen_val.flow_from_directory(
        directory=validation_dir,
        batch_size=batch_size,
        shuffle=False,
        target_size=(224, 224),
        class_mode="binary",
    )

    return train_data_gen, val_data_gen


# ---------------------------------------------------------------------------
# Three-way split — canonical entry point for all migrated training scripts
# ---------------------------------------------------------------------------

class ThreeWaySplit(NamedTuple):
    """Return value of ``make_three_way_split``."""
    train_gen: object       # DirectoryIterator: 85 % of iccad{N}/train/
    val_gen: object         # DirectoryIterator: 15 % carve-out of iccad{N}/train/
    test_gen: object        # DirectoryIterator: iccad{N}/test/ (UNTOUCHED)
    counts: dict            # raw file counts (all four original dirs)
    split_manifest: dict    # exact filenames in each split, reproducibility record


def make_three_way_split(
    benchmark_dir: str | Path,
    benchmark_name: str,
    *,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_size: int = 32,
    img_size: tuple[int, int] = (224, 224),
) -> ThreeWaySplit:
    """Create stratified train / val-carve-out / test generators for one benchmark.

    Parameters
    ----------
    benchmark_dir:
        Path to ``iccad-official/iccad{N}/`` (the parent of ``train/`` and ``test/``).
    benchmark_name:
        Human-readable label stored in the manifest (e.g. ``"iccad5"``).
    val_frac:
        Fraction of *training* images to carve out per class for validation.
        Default 0.15 (15 %).  Applied independently to HS and NHS so class
        balance is preserved.  At 15 % iccad5 yields 4 val HS samples, which
        is the documented minimum for a non-degenerate sensitivity estimate.
    seed:
        Numpy seed controlling both the per-class shuffle and the
        ``flow_from_dataframe`` shuffle of the training subset.  Use the same
        seed as ``tf.keras.utils.set_random_seed`` at the call site for a
        fully reproducible run.
    batch_size:
        Batch size for all three generators.
    img_size:
        ``(height, width)`` passed to ``target_size``.

    Returns
    -------
    ThreeWaySplit
        Named tuple with ``train_gen``, ``val_gen``, ``test_gen``,
        ``counts`` (raw directory counts), and ``split_manifest``
        (dict — JSON-serialisable — recording every filename in val).

    Notes
    -----
    *   The split is performed in **memory only** — no files are moved or
        copied on disk.
    *   ``val_gen`` and ``test_gen`` use ``shuffle=False`` so ``classes``
        attributes are stable and can be used by ``evaluate_*`` helpers.
    *   Class labels in ``train_gen`` and ``val_gen`` are ``"train_hs"`` /
        ``"train_nhs"`` (alphabetical → 0 / 1), matching the ordering that
        ``flow_from_directory`` on ``train/`` produces and that the existing
        ``evaluate_xai_model`` / ``evaluate_generator`` helpers expect via
        ``class_indices.get("test_hs", 0)`` (the HS label is always 0).
    *   ``test_gen`` uses ``flow_from_directory`` on ``iccad{N}/test/`` with
        ``test_hs`` / ``test_nhs`` subdirectories — identical to the old
        ``data_extractor`` behaviour.
    """
    benchmark_dir = Path(benchmark_dir)
    train_root = benchmark_dir / "train"
    test_root = benchmark_dir / "test"

    hs_dir = train_root / "train_hs"
    nhs_dir = train_root / "train_nhs"
    test_hs_dir = test_root / "test_hs"
    test_nhs_dir = test_root / "test_nhs"

    for d in (hs_dir, nhs_dir, test_hs_dir, test_nhs_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Expected directory not found: {d}\n"
                "Check that benchmark_dir points to iccad-official/iccad{N}/"
            )

    # ------------------------------------------------------------------ #
    # 1. List and sort files (deterministic baseline before shuffling)    #
    # ------------------------------------------------------------------ #
    hs_files = sorted(f for f in os.listdir(hs_dir)
                      if not f.startswith("."))
    nhs_files = sorted(f for f in os.listdir(nhs_dir)
                       if not f.startswith("."))

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 2. Stratified split — each class independently                      #
    # ------------------------------------------------------------------ #
    def _split_class(files: list[str]) -> tuple[list[str], list[str]]:
        """Return (train_files, val_files) for one class."""
        n = len(files)
        n_val = max(1, round(n * val_frac))   # always keep ≥1 val sample
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_subset = [files[i] for i in sorted(val_idx)]
        train_subset = [files[i] for i in sorted(train_idx)]
        return train_subset, val_subset

    hs_train_files, hs_val_files = _split_class(hs_files)
    nhs_train_files, nhs_val_files = _split_class(nhs_files)

    # ------------------------------------------------------------------ #
    # 3. Build DataFrames for flow_from_dataframe                         #
    # ------------------------------------------------------------------ #
    # Paths are relative to train_root: "train_hs/basename" and
    # "train_nhs/basename".  This lets Keras resolve and validate the
    # actual file paths (directory=train_root, filename=subdir/basename).
    # Labels use the same string values as flow_from_directory alphabetical
    # ordering: "train_hs" → 0 (HS, positive class),
    #            "train_nhs" → 1 (NHS, negative class).
    hs_train_paths = [f"train_hs/{f}" for f in hs_train_files]
    hs_val_paths   = [f"train_hs/{f}" for f in hs_val_files]
    nhs_train_paths = [f"train_nhs/{f}" for f in nhs_train_files]
    nhs_val_paths   = [f"train_nhs/{f}" for f in nhs_val_files]

    train_df = pd.DataFrame(
        {
            "filename": hs_train_paths + nhs_train_paths,
            "class": ["train_hs"] * len(hs_train_files)
                   + ["train_nhs"] * len(nhs_train_files),
        }
    )
    val_df = pd.DataFrame(
        {
            "filename": hs_val_paths + nhs_val_paths,
            "class": ["train_hs"] * len(hs_val_files)
                   + ["train_nhs"] * len(nhs_val_files),
        }
    )

    # ------------------------------------------------------------------ #
    # 4. Build generators                                                  #
    # ------------------------------------------------------------------ #
    rescale_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = rescale_gen.flow_from_dataframe(
        train_df,
        directory=str(train_root),
        x_col="filename",
        y_col="class",
        class_mode="binary",
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
        val_df,
        directory=str(train_root),
        x_col="filename",
        y_col="class",
        class_mode="binary",
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # test_gen: official ICCAD test split, untouched, used once post-training
    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        directory=str(test_root),
        batch_size=batch_size,
        shuffle=False,
        target_size=img_size,
        class_mode="binary",
    )

    # ------------------------------------------------------------------ #
    # 5. Raw counts (full dirs, before split)                             #
    # ------------------------------------------------------------------ #
    counts = {
        "train_hs": len(hs_files),
        "train_nhs": len(nhs_files),
        "test_hs": len(os.listdir(test_hs_dir)),
        "test_nhs": len(os.listdir(test_nhs_dir)),
        "total_train_original": len(hs_files) + len(nhs_files),
        "total_test": len(os.listdir(test_hs_dir)) + len(os.listdir(test_nhs_dir)),
    }

    # ------------------------------------------------------------------ #
    # 6. Split manifest — JSON-serialisable reproducibility record        #
    # ------------------------------------------------------------------ #
    split_manifest = {
        "benchmark": benchmark_name,
        "split_design": {
            "val_frac": val_frac,
            "seed": seed,
            "stratified_per_class": True,
            "method": "flow_from_dataframe with numpy-seeded index split",
        },
        "train_subset": {
            "n_hs": len(hs_train_files),
            "n_nhs": len(nhs_train_files),
            "total": len(hs_train_files) + len(nhs_train_files),
        },
        "val_subset": {
            "n_hs": len(hs_val_files),
            "n_nhs": len(nhs_val_files),
            "total": len(hs_val_files) + len(nhs_val_files),
            # Filenames stored so the split can be audited and reproduced
            "hs_filenames": hs_val_files,
            "nhs_filenames": nhs_val_files,
        },
        "test_set": {
            "n_hs": counts["test_hs"],
            "n_nhs": counts["test_nhs"],
            "total": counts["total_test"],
            "note": (
                "Official ICCAD test split; read exactly once after training "
                "for final reported metrics only."
            ),
        },
        "iccad5_warning": (
            "iccad5 has only 26 HS training images. At val_frac=0.15, "
            "val_hs=4. Validation sensitivity will be coarse-grained "
            "(~0.25 step resolution). This is a dataset limitation."
        ) if benchmark_name == "iccad5" else None,
    }

    # Print summary matching dataset_analysis style
    print(f"\n[make_three_way_split] {benchmark_name} | val_frac={val_frac} | seed={seed}")
    print(f"  Original train: {counts['train_hs']} HS + {counts['train_nhs']} NHS "
          f"= {counts['total_train_original']}")
    print(f"  Train subset:   {split_manifest['train_subset']['n_hs']} HS + "
          f"{split_manifest['train_subset']['n_nhs']} NHS "
          f"= {split_manifest['train_subset']['total']}")
    print(f"  Val carve-out:  {split_manifest['val_subset']['n_hs']} HS + "
          f"{split_manifest['val_subset']['n_nhs']} NHS "
          f"= {split_manifest['val_subset']['total']}")
    print(f"  Test (untouched): {counts['test_hs']} HS + {counts['test_nhs']} NHS "
          f"= {counts['total_test']}")
    if benchmark_name == "iccad5":
        print(f"  WARNING: iccad5 val_hs={split_manifest['val_subset']['n_hs']} — "
              "see docs/THREE_WAY_SPLIT.md §iccad5")
    print()

    return ThreeWaySplit(
        train_gen=train_gen,
        val_gen=val_gen,
        test_gen=test_gen,
        counts=counts,
        split_manifest=split_manifest,
    )
