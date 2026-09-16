# Grad-CAM Visual Explanations for Lithography Hotspot Detection

This document provides a comprehensive technical guide to the Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for explaining Hotspot (HS) and Non-Hotspot (NHS) predictions in the ICCAD-12 custom CNN baseline models.

---

## 1. What is Grad-CAM?

**Gradient-weighted Class Activation Mapping (Grad-CAM)** is an explainable AI (XAI) technique designed to provide visual explanations for decisions made by Convolutional Neural Networks (CNNs).

Unlike standard Class Activation Mapping (CAM), which requires global average pooling immediately before the softmax classifier and necessitates retraining or architectural constraints, Grad-CAM uses the gradient information flowing into the final convolutional layer of any CNN to calculate importance weights for each feature map.

Reference:
> Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).  
> *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*.  
> IEEE International Conference on Computer Vision (ICCV 2017), pp. 618–626.

---

## 2. Why is Grad-CAM Used for Lithography Hotspot Detection?

Lithography hotspot detection identifies geometric layout configurations in integrated circuits (VLSI design) that are susceptible to critical manufacturing defects during optical photolithography (e.g., pinching/necking leading to open circuits, or bridging leading to short circuits).

Applying Grad-CAM to hotspot detection enables us to:
1. **Localize Critical Features**: Determine which spatial regions and geometric topologies (e.g. dense pitches, line ends, corner rounding, jogs) drive the CNN's classification.
2. **Verify Learned Representations**: Confirm whether the CNN focuses on realistic layout defect zones or relies on spurious background/edge artifacts.
3. **Analyze Misclassifications**: Diagnose why False Positives (over-sensitive alarms on complex DRC-clean patterns) and False Negatives (missed subtle defects) occur.

---

## 3. Discovered CNN Architecture & Target Layer

The baseline custom CNN reproduces Approach 2 from `LHD_CustomModel.ipynb` (~12,873 total parameters):

```text
Input (224 × 224 × 3)
│
├── [Block 1]
│   ├── Conv2D(12, 3×3, activation='elu')      -> (222, 222, 12)
│   ├── Conv2D(12, 3×3, activation='elu')      -> (220, 220, 12)
│   ├── Conv2D(12, 3×3, activation=None)       -> (218, 218, 12)
│   ├── BatchNormalization                     -> (218, 218, 12)
│   ├── Activation('elu')                      -> (218, 218, 12)
│   └── MaxPooling2D(2×2)                      -> (109, 109, 12)
│
├── MaxPooling2D(5×5)                          -> (21, 21, 12)
│
├── [Block 2]
│   ├── Conv2D(12, 3×3, activation='elu')      -> (19, 19, 12)
│   ├── Conv2D(12, 3×3, activation='elu')      -> (17, 17, 12)
│   ├── Conv2D(12, 3×3, activation=None)       -> (15, 15, 12)  <-- Target Layer: conv2d_5
│   ├── BatchNormalization                     -> (15, 15, 12)
│   ├── Activation('elu')                      -> (15, 15, 12)
│   └── MaxPooling2D(2×2)                      -> (7, 7, 12)
│
├── Flatten                                    -> (588)
├── Dropout(0.3)                               -> (588)
├── Dense(10, activation='relu')               -> (10)
└── Dense(1, activation='sigmoid')             -> (1)           <-- Output: p = P(NHS)
```

### Target Convolutional Layer Verification
- **Layer Name**: `conv2d_5`
- **Output Shape**: `(None, 15, 15, 12)`
- **Rationale**: `conv2d_5` is the final convolutional layer in the network before dense classification. It captures the highest-level spatial-semantic features with a large receptive field across the layout.

---

## 4. Mathematical Formulation & Gradient Calculation

### Binary Sigmoid Classifier Target Score Formulation

The output layer is a single neuron with sigmoid activation:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$
where $z = W^T x_{\text{dense}} + b$ is the pre-sigmoid logit.

In the ICCAD-12 data pipeline:
- `*_hs` (Hotspot) $\to$ Label **0** $\implies P(\text{HS}) = 1 - p$
- `*_nhs` (Non-Hotspot) $\to$ Label **1** $\implies P(\text{NHS}) = p$

### Avoiding Sigmoid Gradient Saturation
When evaluating highly confident predictions ($p \approx 1.0$ or $p \approx 0.0$), taking gradients of the post-sigmoid probability $\sigma(z)$ leads to vanishing gradients because $\sigma'(z) = \sigma(z)(1-\sigma(z)) \to 0$.

Following the original formulation (Selvaraju et al., ICCV 2017), we compute gradients of the **pre-sigmoid logit score** $y^c$:
$$y^{\text{NHS}} = +z$$
$$y^{\text{HS}} = -z$$

### Grad-CAM Weight Calculation
For feature maps $A^k \in \mathbb{R}^{15 \times 15}$ of layer `conv2d_5` ($k = 1, \dots, 12$ channels):

1. **Global Average Pooling of Gradients**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{15} \sum_{j=1}^{15} \frac{\partial y^c}{\partial A_{i,j}^k}$$
   where $Z = 15 \times 15 = 225$.

2. **Weighted Linear Combination & Rectification**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{12} \alpha_k^c A^k \right)$$
   The $\text{ReLU}$ operator retains only features with positive attribution towards the target class $c$.

3. **Normalization**:
   $$H(i, j) = \frac{L_{\text{Grad-CAM}}^c(i, j)}{\max_{i', j'} L_{\text{Grad-CAM}}^c(i', j') + \epsilon}$$

4. **Bilinear Upsampling**:
   The $15 \times 15$ heatmap is upsampled to the original layout dimensions ($224 \times 224$).

5. **Colormap Overlay**:
   The normalized heatmap is mapped using the `'jet'` colormap and blended with the RGB layout:
   $$I_{\text{overlay}} = (1 - \alpha) \cdot I_{\text{RGB}} + \alpha \cdot \text{Colormap}(H)$$
   with $\alpha = 0.45$.

---

## 5. Representative Sample Selection Strategy

Test samples are evaluated across all 5 ICCAD-12 benchmarks and categorized into four quadrants:

| Case | Actual Label | Predicted Label | Selection Criterion |
|:---|:---:|:---:|:---|
| **TP** (True Positive) | HS | HS | Highest $P(\text{HS}) = 1 - p$ (high confidence) |
| **TN** (True Negative) | NHS | NHS | Highest $P(\text{NHS}) = p$ (high confidence) |
| **FP** (False Positive) | NHS | HS | Highest $P(\text{HS}) = 1 - p$ (strongest false alarm) |
| **FN** (False Negative) | HS | NHS | Highest $P(\text{NHS}) = p$ (most severe miss) |

For each benchmark, up to 4 representative samples per category (16 total per benchmark when available) are extracted and saved.

---

## 6. Output Directory Structure

```text
results/gradcam/
├── gradcam_samples.csv              # Summary table with all metadata & quantitative metrics
├── GRADCAM_ANALYSIS.md              # Detailed technical interpretation report
├── summary_TP.png                   # Multi-sample comparison grid for True Positives
├── summary_TN.png                   # Multi-sample comparison grid for True Negatives
├── summary_FP.png                   # Multi-sample comparison grid for False Positives
├── summary_FN.png                   # Multi-sample comparison grid for False Negatives
├── iccad1/
│   ├── iccad1_TP_HS_predHS_sampleHS73_original.png
│   ├── iccad1_TP_HS_predHS_sampleHS73_heatmap.png
│   ├── iccad1_TP_HS_predHS_sampleHS73_overlay.png
│   ├── iccad1_TP_HS_predHS_sampleHS73_comparison.png
│   └── ... (all samples for iccad1)
├── iccad2/
├── iccad3/
├── iccad4/
└── iccad5/
```

---

## 7. Quantitative Spatial Metrics Computed

For every sample, the following spatial metrics are recorded in `gradcam_samples.csv`:
- `total_activation`: $\sum_{i,j} H(i, j)$
- `mean_activation`: Mean intensity across the $224 \times 224$ layout domain
- `peak_activation`: Maximum activation value (1.0 for non-zero heatmaps)
- `top10_threshold`: Intensity threshold at the 90th percentile of pixels
- `top10_activation_fraction`: Fraction of total activation mass located in the top 10% brightest pixels
- `centroid_x`, `centroid_y`: Center of mass $(\bar{x}, \bar{y})$ of the activation distribution
- `spatial_spread`: Weighted radial standard deviation $\sigma_r = \sqrt{\frac{\sum ((x-\bar{x})^2 + (y-\bar{y})^2) H(x,y)}{\sum H(x,y)}}$
- `activated_area_fraction_gt05`: Fraction of layout pixels with $H(x, y) \ge 0.5$

---

## 8. Reproducibility Commands

To rerun the full Grad-CAM extraction pipeline across all 5 ICCAD-12 benchmarks using CUDA/GPU:

```bash
cd /home/peskybird/Projects/Ai_Ml_DA

# GPU execution via environment wrapper (NixOS / CUDA 12.2)
bash scripts/tf_gpu.sh -m src.run_gradcam

# Custom benchmark subset or sample count:
bash scripts/tf_gpu.sh -m src.run_gradcam --benchmarks 1 2 3 --samples-per-case 5
```

---

## 9. Known Limitations & Technical Scope

1. **Spatial Attribution vs. Physical Causality**:
   Grad-CAM identifies spatial regions that positively increase the network's classification score. It does not model optical diffraction, Abbe illumination, or photoresist dissolution chemistry.
2. **Coarse Spatial Resolution**:
   Because `conv2d_5` outputs a $15 \times 15$ grid (downsampled by two MaxPool stages: $2\times 2$ and $5\times 5$), the spatial attribution maps have an effective patch resolution of approximately $15 \times 15$ pixels before bilinear upsampling.
3. **Single Convolutional Layer Focus**:
   Grad-CAM visualizes features from `conv2d_5`. Intermediate multi-scale features in Block 1 (`conv2d_2`) are integrated only through their downstream forward influence.
