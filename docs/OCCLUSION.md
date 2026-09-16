# Occlusion-Based Sensitivity Analysis for Lithography Hotspot Detection

This document provides the complete theoretical formulation, implementation architecture, and empirical findings for the **Perturbation-Based Occlusion Sensitivity** framework on ICCAD-12 benchmarks 1–5.

---

## 1. Motivation & Methodological Overview

While Gradient-Weighted Class Activation Mapping (Grad-CAM, Grad-CAM++, LayerCAM) calculates visual explanations via backpropagation gradients $\frac{\partial S}{\partial A^k}$, **Occlusion Sensitivity** provides an independent, perturbation-based validation mechanism:

$$\text{Core Question: If a specific spatial region is occluded (removed), how much does the model's decision change?}$$

By systematically replacing sliding windows of the input layout with a neutral baseline and evaluating the forward prediction drop, we directly measure the empirical sensitivity of the neural network without relying on gradient approximations or backpropagation assumptions.

---

## 2. Mathematical Formulation

### 2.1 Target Score & Pre-Sigmoid Logit
Matching the gradient-based XAI pipeline, occlusion importance is formulated using the **pre-sigmoid logit score** $z$:
- For Hotspot class (Positive Class, index 0): $S_{\text{target}}(\mathbf{x}) = -z_{\text{NHS}}(\mathbf{x})$
- For Non-Hotspot class (Negative Class, index 1): $S_{\text{target}}(\mathbf{x}) = +z_{\text{NHS}}(\mathbf{x})$

Using pre-sigmoid logits eliminates gradient and score compression artifacts caused by sigmoid saturation ($\sigma(z) \to 1.0$ or $0.0$) on high-confidence layout patterns.

### 2.2 Occlusion Importance Definition
For an input layout $\mathbf{x} \in \mathbb{R}^{224 \times 224 \times 3}$ and a spatial occlusion patch $W_k = [y_1:y_2, x_1:x_2]$, the occluded image $\mathbf{x}^{(k)}$ is formed by setting:

$$\mathbf{x}^{(k)}_{i, j, :} = \begin{cases} v_{\text{baseline}} & \text{if } (i, j) \in W_k \\ \mathbf{x}_{i, j, :} & \text{otherwise} \end{cases}$$

The sensitivity importance for window $W_k$ w.r.t the target class $c$ is defined as the **drop in target class score**:

$$\Delta S_k = S_{\text{target}}(\mathbf{x}) - S_{\text{target}}(\mathbf{x}^{(k)})$$

* **$\Delta S_k > 0$ (Positive Sensitivity)**: Occluding the patch *reduces* the target score $\implies$ The features inside $W_k$ actively **support** the target prediction.
* **$\Delta S_k < 0$ (Negative Sensitivity)**: Occluding the patch *increases* the target score $\implies$ The features inside $W_k$ **suppress** the target prediction.

### 2.3 2D Spatial Aggregation & Coverage Normalization
Because sliding windows overlap across the $224 \times 224$ grid, each pixel $(i, j)$ is covered by multiple windows. Contributions are accumulated and divided by the exact coverage count:

$$M_{\text{raw}}(i, j) = \frac{\sum_{k: (i, j) \in W_k} \Delta S_k}{\sum_{k: (i, j) \in W_k} 1}$$

### 2.4 Positive Attribution & Visualization Normalization
To isolate features providing evidence *in favor* of the model's decision:
1. **Positive Attribution Map**: $M_{\text{pos}}(i, j) = \max(M_{\text{raw}}(i, j), 0)$
2. **Normalized Heatmap**: $M_{\text{norm}}(i, j) = \frac{M_{\text{pos}}(i, j)}{\max_{u,v} M_{\text{pos}}(u,v) + 10^{-10}}$
3. **Overlay Blending**: $\mathbf{I}_{\text{overlay}} = (1 - \alpha)\mathbf{I}_{\text{orig}} + \alpha \cdot \text{Colormap}(M_{\text{norm}})$ with $\alpha = 0.45$.

---

## 3. Parameter Specifications

| Parameter | Value | Technical Rationale |
|:---|:---:|:---|
| **Window Size ($W \times H$)** | $32 \times 32$ px | Approximately $14.3\%$ of the $224\times 224$ input. Sufficiently large to perturb multi-track layout patterns while small enough to isolate localized defect regions. |
| **Stride ($S_y, S_x$)** | $16 \times 16$ px | $50\%$ window overlap ($4\times$ coverage across interior pixels), ensuring smooth spatial gradients without boundary banding. |
| **Grid Dimensions** | $13 \times 13 = 169$ windows | $169$ forward passes per sample, evaluated in GPU batches of 64 ($\approx 0.25$ seconds per image). |
| **Baseline Value ($v_{\text{baseline}}$)** | $0.0$ | Corresponds to empty layout background (absence of metal geometries/features in binary CAD data). |
| **Sample Set** | 57 representative samples | Identical 57 samples from `xai_diagnostics.csv` across ICCAD benchmarks 1–5 (15 TP, 15 TN, 15 FP, 12 FN). |

---

## 4. Quantitative Comparison: Occlusion vs CAM Methods

Cross-correlation and spatial similarity metrics between Occlusion sensitivity maps and Gradient-based CAM methods across all 57 representative samples:

### Mean Pearson Correlation ($r$) by Category
| Category | Occlusion vs Grad-CAM | Occlusion vs Grad-CAM++ | Occlusion vs LayerCAM |
|:---|:---:|:---:|:---:|
| **True Positive (TP)** | $+0.2721$ | $+0.0965$ | **$+0.4891$** |
| **True Negative (TN)** | $-0.1924$ | $+0.0997$ | **$+0.2118$** |
| **False Positive (FP)** | $+0.3045$ | $+0.1263$ | **$+0.4403$** |
| **False Negative (FN)** | $-0.0985$ | $+0.0749$ | **$+0.1711$** |
| **Overall Mean** | **$+0.0803$** | **$+0.1006$** | **$+0.3363$** |

### Mean Cosine Similarity by Category
| Category | Occlusion vs Grad-CAM | Occlusion vs Grad-CAM++ | Occlusion vs LayerCAM |
|:---|:---:|:---:|:---:|
| **True Positive (TP)** | $0.4867$ | $0.2629$ | **$0.5915$** |
| **True Negative (TN)** | $0.0643$ | $0.2898$ | **$0.4488$** |
| **False Positive (FP)** | $0.4986$ | $0.2970$ | **$0.5657$** |
| **False Negative (FN)** | $0.0226$ | $0.1605$ | **$0.3706$** |

### Key Observations
1. **LayerCAM Shows Strongest Agreement with Occlusion**:
   LayerCAM achieves the highest Pearson correlation ($+0.4891$ for TP, $+0.4403$ for FP) and Cosine similarity ($0.5915$ for TP) with Occlusion sensitivity. Because LayerCAM retains spatial pixel-wise gradient weighting rather than collapsing spatial maps via Global Average Pooling, its attribution aligns most closely with localized physical perturbations.
2. **Grad-CAM exhibits Divergence on Negatives**:
   While Grad-CAM correlates positively with Occlusion on positive predictions (TP $+0.27$, FP $+0.30$), it diverges on negative classifications (TN $-0.19$, FN $-0.10$) due to diffuse pooling across background regions.
3. **Focal Attribution on Hotspots**:
   For True Positives, the spatial spread radius of Occlusion sensitivity averages **$54.10$ px**, representing the tightest localization among all confusion categories.

---

## 5. Directory Structure & Generated Artifacts

```text
results/xai/occlusion/
├── occlusion_diagnostics.csv         # Complete 57-sample quantitative metrics table (37 columns)
├── summaries/                        # 5-Method Comparative Summary Grids
│   ├── summary_TP.png                # TP Comparison (Original | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion)
│   ├── summary_TN.png                # TN Comparison
│   ├── summary_FP.png                # FP Comparison
│   └── summary_FN.png                # FN Comparison
├── iccad1/                           # Per-sample folders with 5 explanation artifacts each
│   ├── iccad1_TP_HS_predHS_HS32/
│   │   ├── original.png
│   │   ├── occlusion.png             # Normalized colormapped heatmap
│   │   ├── occlusion_overlay.png     # Alpha-blended overlay (alpha=0.45)
│   │   ├── raw_occlusion_map.npy     # Raw signed numerical sensitivity array
│   │   └── methods_comparison_5panel.png # 5-panel side-by-side comparison
│   └── ...
├── iccad2/
├── iccad3/
├── iccad4/
└── iccad5/
```

---

## 6. Reproducibility

To rerun the full occlusion pipeline across all 57 representative samples:

```bash
# GPU Execution (via CUDA launcher)
bash scripts/tf_gpu.sh -m src.run_occlusion --benchmarks 1 2 3 4 5

# Direct Python Execution
python -m src.run_occlusion --window-size 32 --stride 16 --baseline-value 0.0 --batch-size 64
```

---

## 7. Scientific Scope & Interpretability Boundaries

* **Sensitivity vs Physical Causality**: Occlusion sensitivity identifies regions whose removal causes a measurable drop in the neural network's internal classification score. It does **not** directly simulate physical photolithography optical diffraction or etch processes.
* **Non-Destructive Execution**: The occlusion pipeline is completely read-only with respect to existing Grad-CAM, Grad-CAM++, and LayerCAM outputs in `results/xai/`.
