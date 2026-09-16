"""Dataset helpers faithful to LHD_CustomModel.ipynb."""

from __future__ import annotations

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def dataset_analysis(folder: str):
    """Return train/validation dirs and print class counts (notebook logic)."""
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
