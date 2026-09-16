# Baseline Model & Pipeline Audit Report (Phase 1)

This audit documents the technical specifications, data pipeline, CNN architecture, and explainability limitations of the existing custom CNN baseline for ICCAD-12 lithography hotspot detection.

---

## 1. Environment & Library Specifications

| Component | Version | How It Is Called / Used |
|:---|:---|:---|
| **Python** | `3.11.16` | Main execution runtime in `.venv` |
| **TensorFlow** | `2.15.1` | Deep learning backend (`tf.keras`) |
| **Keras** | `2.15.0` | High-level API for model construction, training, and gradients |
| **NumPy** | `1.26.4` | Matrix computations, score evaluation, array reshaping |
| **SciPy** | `1.17.1` | Advanced scientific calculations |
| **Matplotlib** | `3.11.2` | Heatmap generation, colormaps, confusion matrices, summary panels |
| **Seaborn** | `0.13.2` | Confusion matrix heatmap formatting |
| **Pandas** | `3.0.5` | Metrics and evaluation CSV management |
| **Pillow** | `12.3.0` | Image loading, resizing, and raw layout RGB conversion |
| **GPU Execution** | NVIDIA CUDA 12.2 / cuDNN 8.9 | Invoked via `scripts/tf_gpu.sh` setting `LD_LIBRARY_PATH` to pip nvidia wheels + `/run/opengl-driver/lib` |

---

## 2. Dataset & Preprocessing Audit

### 2.1 Dataset Structure & Splits
- **Source**: ICCAD-12 Contest benchmarks extracted under `iccad-official/iccad{1..5}/`.
- **Splits**: Predefined benchmark directories:
  - `iccad{N}/train/`: `train_hs` (Hotspot) and `train_nhs` (Non-Hotspot)
  - `iccad{N}/test/`: `test_hs` (Hotspot) and `test_nhs` (Non-Hotspot)
- **Class Label Assignment** (via `ImageDataGenerator.flow_from_directory` alphabetical order):
  - Label `0`: `*_hs` (Hotspot) $\implies$ Defined as the **Positive Class**
  - Label `1`: `*_nhs` (Non-Hotspot) $\implies$ Defined as the **Negative Class**

### 2.2 Class Distribution
| Benchmark | Train HS | Train NHS | Train Total | Train HS% | Test HS | Test NHS | Test Total | Test HS% | Imbalance Ratio (NHS:HS in Test) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **iccad1** | 99 | 340 | 439 | 22.55% | 226 | 4,679 | 4,905 | 4.61% | 20.7 : 1 |
| **iccad2** | 174 | 5,285 | 5,459 | 3.19% | 498 | 41,298 | 41,796 | 1.19% | 82.9 : 1 |
| **iccad3** | 909 | 4,643 | 5,552 | 16.37% | 1,808 | 46,333 | 48,141 | 3.76% | 25.6 : 1 |
| **iccad4** | 95 | 4,452 | 4,547 | 2.09% | 177 | 31,890 | 32,067 | 0.55% | 180.2 : 1 |
| **iccad5** | 26 | 2,716 | 2,742 | 0.95% | 41 | 19,327 | 19,368 | 0.21% | 471.4 : 1 |

### 2.3 Image Preprocessing Specifications
- **Input Resolution**: $224 \times 224 \times 3$ (RGB)
- **Resize / Interpolation**: Bilinear interpolation (PIL / Keras `ImageDataGenerator(target_size=(224, 224))`)
- **Normalization**: Rescaled by $1.0 / 255.0$ mapping integer pixels $[0, 255] \to \text{float32 } [0.0, 1.0]$
- **Data Augmentation**: **None** (baseline uses raw images without rotation, flipping, or jitter)
- **Class Weighting in Training**: **None** (unweighted standard binary cross-entropy loss)

---

## 3. Baseline CNN Architecture Audit

The baseline custom CNN reproduces Approach 2 from `LHD_CustomModel.ipynb` (`src/model.py`):

```text
==================================================================================================
Layer (type)                     Output Shape          Param #     Activation    Padding
==================================================================================================
conv2d (Conv2D)                  (None, 222, 222, 12)  336         ELU           valid
conv2d_1 (Conv2D)                (None, 220, 220, 12)  1,308       ELU           valid
conv2d_2 (Conv2D)                (None, 218, 218, 12)  1,308       Linear        valid
batch_normalization (BatchNorm)  (None, 218, 218, 12)  48          —             —
activation (Activation)          (None, 218, 218, 12)  0           ELU           —
max_pooling2d (MaxPooling2D 2x2) (None, 109, 109, 12)  0           —             valid
--------------------------------------------------------------------------------------------------
max_pooling2d_1 (MaxPool 5x5)    (None, 21, 21, 12)    0           —             valid
--------------------------------------------------------------------------------------------------
conv2d_3 (Conv2D)                (None, 19, 19, 12)    1,308       ELU           valid
conv2d_4 (Conv2D)                (None, 17, 17, 12)    1,308       ELU           valid
conv2d_5 (Conv2D) [TARGET LAYER] (None, 15, 15, 12)    1,308       Linear        valid
batch_normalization_1 (BatchNorm)(None, 15, 15, 12)    48          —             —
activation_1 (Activation)        (None, 15, 15, 12)    0           ELU           —
max_pooling2d_2 (MaxPool 2x2)    (None, 7, 7, 12)      0           —             valid
--------------------------------------------------------------------------------------------------
flatten (Flatten)                (None, 588)           0           —             —
dropout (Dropout 0.3)            (None, 588)           0           —             —
dense (Dense 10)                 (None, 10)            5,890       ReLU          —
dense_1 (Dense 1)                (None, 1)             11          Sigmoid       —
==================================================================================================
Total Parameters: 12,873 (Trainable: 12,825, Non-trainable: 48)
```

### Key Dimensions
- **Final Convolutional Layer**: `conv2d_5` with spatial shape **$(15, 15, 12)$**
- **Spatial Reduction Factor**: $224 / 15 \approx 14.93\times$ reduction along each axis ($\sim 223\times$ reduction in spatial pixel area).

---

## 4. Baseline Evaluation Results Summary

| Benchmark | Epochs | Balanced Acc. | Precision | Recall (HS) | Specificity | F1 Score | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **iccad1** | 5 | 0.8851 | 0.1804 | 0.9867 | 0.7835 | 0.3051 | 0.7929 |
| **iccad2** | 5 | 0.9893 | 0.5119 | 0.9900 | 0.9886 | 0.6749 | 0.9886 |
| **iccad3** | 10 | 0.9672 | 0.4707 | 0.9773 | 0.9571 | 0.6354 | 0.9579 |
| **iccad4** | 10 | 0.8578 | 0.6720 | 0.7175 | 0.9981 | 0.6940 | 0.9965 |
| **iccad5** | 10 | 0.9822 | 0.1556 | 0.9756 | 0.9888 | 0.2685 | 0.9887 |
| **Average** | — | **0.9363** | **0.3981** | **0.9294** | **0.9432** | **0.5156** | **0.9449** |

---

## 5. Root Cause Analysis: Why Baseline Grad-CAM Is Poor

1. **Aggressive Spatial Pooling & Valid Padding**:
   The combination of three valid convolutions, a $2\times 2$ MaxPool, an aggressive $5\times 5$ MaxPool, and three subsequent valid convolutions collapses the layout from $224 \times 224$ to a mere $15 \times 15$ grid. Each activation bin represents approximately a $15 \times 15$ patch in input space.
2. **Coarse Receptive Field & Blurred Localization**:
   When a $15 \times 15$ activation map is upsampled via bilinear interpolation to $224 \times 224$, the resulting heatmap forms a large, diffuse Gaussian-like blob that covers wide tracts of unrelated layout geometry (such as power rails, neighboring standard cells, or empty space) rather than isolating critical sub-100nm pitch gaps or line ends.
3. **Lack of Multi-Method Comparison**:
   The baseline only computes vanilla Grad-CAM with global average pooling of gradients, which can underestimate fine-grained sub-structure contributions compared to Grad-CAM++ (higher-order gradient weighting) or LayerCAM (spatial pixel-wise weighting).
4. **Single-Layer Inspection**:
   Relying solely on `conv2d_5` overlooks richer spatial details preserved in earlier convolutional layers (e.g. `conv2d_2` at $218 \times 218$).

---

## 6. Action Plan for Phase 2–7

1. **Keep Baseline Untouched**: Preserve all baseline models in `models/` and metrics in `results/baseline/`.
2. **Build XAI-Ready Architecture (`src/model_xai.py`)**:
   - Use `'same'` padding and modular $2\times 2$ pooling to maintain a $28 \times 28$ final convolutional feature map.
   - Maintain lightweight footprint ($\approx 25\text{k} - 35\text{k}$ parameters).
3. **Train & Validate XAI-Ready Classifier (`src/train_xai.py`)**:
   - Train on all 5 benchmarks with deterministic seed (`seed=42`) and balanced class weighting.
   - Save models to `models/xai/` and benchmark comparison to `results/xai_training/`.
4. **Develop Multi-Method XAI Engine (`src/xai_engine.py`)**:
   - Implement Grad-CAM, Grad-CAM++, and LayerCAM.
   - Support both True-Class and Predicted-Class attributions using pre-sigmoid logits.
5. **Multi-Layer Comparison & Diagnostics (`src/xai_diagnostics.py` & `src/run_xai.py`)**:
   - Extract early ($112 \times 112$), intermediate ($56 \times 56$), and final ($28 \times 28$) layer attributions.
   - Compute strict quantitative diagnostics (centroids, spread, threshold area fractions, connected components).
6. **Compile Comprehensive Documentation**:
   - `docs/XAI_MODEL.md`, `docs/XAI_METHODS.md`, `results/xai/README.md`.
