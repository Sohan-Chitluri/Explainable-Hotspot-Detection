# Custom CNN Baseline (ICCAD-12 Hotspot Detection)

Faithful local reproduction of Approach 2 from `LHD_CustomModel.ipynb`  
(repository: [Intelectron6/Lithography-Hotspot-Detection](https://github.com/Intelectron6/Lithography-Hotspot-Detection)).  
No Grad-CAM / XAI changes in this stage.

## Environment

| Item | Value |
|------|--------|
| OS | NixOS |
| Python | 3.11.16 (`.venv`) |
| TensorFlow | 2.15.1 |
| Keras | 2.15.0 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| Seaborn | 0.13.2 |
| Pandas | 3.0.5 |
| Pillow | 12.3.0 |
| Device | iccad1–2: CPU; iccad3–5: NVIDIA RTX 3050 (GPU via `scripts/tf_gpu.sh`) |

GPU stack (optional): pip NVIDIA CUDA 12.2 / cuDNN 8.9 wheels + system `libcuda` from `/run/opengl-driver/lib`. See `requirements.txt`.

## Dataset

Source: README Google Drive link → `iccad_official.rar` (local archive extracted to `iccad-official/`).

Expected layout (matches notebook `/content/iccad-official/iccad{N}/`):

```text
iccad-official/
  iccad{1..5}/
    train/train_hs/
    train/train_nhs/
    test/test_hs/
    test/test_nhs/
```

`ImageDataGenerator.flow_from_directory` class indices (alphabetical):

- `*_hs` → label **0** (hotspot)
- `*_nhs` → label **1** (non-hotspot)

Reported metrics treat **HS (label 0)** as the positive class.

## Preprocessing

- Rescale: `1/255`
- Resize: `224 × 224`
- Batch size: `32`
- `class_mode='binary'`
- Train shuffle: `True`; test shuffle: `False`
- No augmentation (same as notebook)

## CNN architecture

Sequential custom CNN from the notebook (~12.8k params):

1. Conv2D(12, 3×3, ELU) → Conv2D(12, 3×3, ELU) → Conv2D(12, 3×3) → BatchNorm → ELU → MaxPool(2×2)
2. MaxPool(5×5)
3. Conv2D(12, 3×3, ELU) → Conv2D(12, 3×3, ELU) → Conv2D(12, 3×3) → BatchNorm → ELU → MaxPool(2×2)
4. Flatten → Dropout(0.3) → Dense(10, ReLU) → Dense(1, sigmoid)

Implementation: `src/model.py`

## Training configuration

| Setting | Value |
|---------|--------|
| Optimizer | Nadam |
| Loss | binary cross-entropy |
| Epochs | iccad1–2: **5**; iccad3–5: **10** (notebook) |
| Batch size | 32 |
| Seed | 42 (`tf.keras.utils.set_random_seed`) |

## Evaluation metrics (test split)

Primary: **balanced accuracy** (imbalanced ICCAD-12). Also: confusion matrix, precision, recall/sensitivity, specificity, F1. Plain accuracy is recorded but not used as the main score.

Metrics are computed on the **final-epoch** model with HS as positive. The notebook’s summary bar chart used **best-epoch** Keras-history balanced accuracy (NHS as Keras default positive for TP/FP counters; balanced accuracy is class-symmetric).

## Baseline results

| Benchmark | Balanced Acc. | Precision | Recall (HS) | Specificity | F1 |
|-----------|---------------|-----------|-------------|-------------|-----|
| iccad1 | 0.8851 | 0.1804 | 0.9867 | 0.7835 | 0.3051 |
| iccad2 | 0.9893 | 0.5119 | 0.9900 | 0.9886 | 0.6749 |
| iccad3 | 0.9672 | 0.4707 | 0.9773 | 0.9571 | 0.6354 |
| iccad4 | 0.8578 | 0.6720 | 0.7175 | 0.9981 | 0.6940 |
| iccad5 | 0.9822 | 0.1556 | 0.9756 | 0.9888 | 0.2685 |
| **Average** | **0.9363** | | | | |

Notebook reported average validation balanced accuracy ≈ **0.953** (best-epoch history). Local average final-epoch HS-positive balanced accuracy = **0.9363**.

## Outputs

```text
models/
  custom_cnn_iccad{1..5}.keras
results/baseline/
  metrics.json
  balanced_accuracy_bar.png
  iccad{N}/metrics.json
  iccad{N}/confusion_matrix.png
  iccad{N}/balanced_accuracy_vs_epoch.png
```

## Reproduce

```bash
cd /home/peskybird/Projects/Ai_Ml_DA
source .venv/bin/activate   # or use paths below

# CPU (all benchmarks)
.venv/bin/python -m src.train_baseline

# GPU on NixOS (recommended for iccad3–5)
bash scripts/tf_gpu.sh -m src.train_baseline --benchmarks 3 4 5

# Single benchmark
bash scripts/tf_gpu.sh -m src.train_baseline --benchmarks 1
```

Requires extracted `iccad-official/` (from `iccad_official.rar`). Do not commit `.venv/`, the RAR, or `iccad-official/` (see `.gitignore`).
