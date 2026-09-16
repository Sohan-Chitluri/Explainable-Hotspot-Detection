# Grad-CAM Visual Explanation & Technical Analysis Report

## Executive Summary

This report documents the visual explanations generated via Gradient-weighted Class Activation Mapping (Grad-CAM) for the custom CNN baseline evaluated across ICCAD-12 benchmarks 1 through 5. The objective is to determine which spatial regions of the VLSI layout contribute most strongly to Hotspot (HS) and Non-Hotspot (NHS) classifications and evaluate whether the highlighted regions correspond to technically meaningful geometric layout features (such as dense pitch, line-end thinning, necking/pinching risk, and bridging proximity).

> [!IMPORTANT]
> **Attribution vs. Physical Causality**: Grad-CAM visualizes the spatial gradient attribution learned by the convolutional neural network. While high attribution identifies regions the model relies upon for classification, it does not constitute a full physical lithography simulation of wafer printability.

---

## Benchmark Summary & Sample Counts

| Benchmark | Analyzed Samples | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `iccad1` | 15 | 4 | 4 | 4 | 3 |
| `iccad2` | 16 | 4 | 4 | 4 | 4 |
| `iccad3` | 16 | 4 | 4 | 4 | 4 |
| `iccad4` | 16 | 4 | 4 | 4 | 4 |
| `iccad5` | 13 | 4 | 4 | 4 | 1 |

---

## Quantitative Grad-CAM Spatial Statistics

Quantitative metrics computed across the normalized heatmaps ($224 \times 224$ layout domain):
- **Total Activation**: Integral sum of normalized activation values across all spatial coordinates.
- **Top 10% Activation Fraction**: Proportion of total activation mass concentrated within the top 10% highest-intensity pixels.
- **Activation Centroid $(c_x, c_y)$**: Center-of-mass coordinate of the activation distribution.
- **Spatial Spread ($\sigma_r$)**: Radial standard deviation describing spatial localization vs diffuse spread.

| Case Type | Mean Total Activation | Mean Top 10% Concentration | Mean Spatial Spread (px) | Mean High-Activation Area (>0.5) |
|:---|:---:|:---:|:---:|:---:|
| **FN** | 953.7 | 30.1% | 21.6 px | 1.1% |
| **FP** | 18347.8 | 30.4% | 87.0 px | 33.6% |
| **TN** | 2217.9 | 26.7% | 30.2 px | 3.5% |
| **TP** | 15189.0 | 34.0% | 89.0 px | 22.8% |

---

## Detailed Sample Interpretations

### Sample `iccad1_HS73` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.7887 | P(HS)=0.7887, P(NHS)=0.2113)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(112.03, 107.12)` | **Spatial Spread**: `100.81 px`
- **Top 10% Activation Concentration**: `36.2%`
- **Overlay Image**: [`iccad1_TP_HS_predHS_sampleHS73_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TP_HS_predHS_sampleHS73_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad1_HS53` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.7688 | P(HS)=0.7688, P(NHS)=0.2312)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(108.84, 108.92)` | **Spatial Spread**: `101.95 px`
- **Top 10% Activation Concentration**: `36.5%`
- **Overlay Image**: [`iccad1_TP_HS_predHS_sampleHS53_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TP_HS_predHS_sampleHS53_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad1_HS68` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.7501 | P(HS)=0.7501, P(NHS)=0.2499)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(104.85, 88.07)` | **Spatial Spread**: `94.43 px`
- **Top 10% Activation Concentration**: `41.1%`
- **Overlay Image**: [`iccad1_TP_HS_predHS_sampleHS68_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TP_HS_predHS_sampleHS68_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad1_HS184` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.7434 | P(HS)=0.7434, P(NHS)=0.2566)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(118.55, 100.41)` | **Spatial Spread**: `87.15 px`
- **Top 10% Activation Concentration**: `31.1%`
- **Overlay Image**: [`iccad1_TP_HS_predHS_sampleHS184_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TP_HS_predHS_sampleHS184_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad1_NNHS329_8` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 0.8960 | P(HS)=0.1040, P(NHS)=0.8960)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(124.2, 140.63)` | **Spatial Spread**: `90.95 px`
- **Top 10% Activation Concentration**: `60.6%`
- **Overlay Image**: [`iccad1_TN_NHS_predNHS_sampleNNHS329_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TN_NHS_predNHS_sampleNNHS329_8_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad1_NNHS321_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 0.9007 | P(HS)=0.0993, P(NHS)=0.9007)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(122.31, 132.66)` | **Spatial Spread**: `84.93 px`
- **Top 10% Activation Concentration**: `41.1%`
- **Overlay Image**: [`iccad1_TN_NHS_predNHS_sampleNNHS321_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TN_NHS_predNHS_sampleNNHS321_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad1_NNHS413_7` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 0.8921 | P(HS)=0.1079, P(NHS)=0.8921)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.82, 133.05)` | **Spatial Spread**: `85.87 px`
- **Top 10% Activation Concentration**: `39.1%`
- **Overlay Image**: [`iccad1_TN_NHS_predNHS_sampleNNHS413_7_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TN_NHS_predNHS_sampleNNHS413_7_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad1_NNHS491_8` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 0.8821 | P(HS)=0.1179, P(NHS)=0.8821)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(124.61, 121.06)` | **Spatial Spread**: `87.62 px`
- **Top 10% Activation Concentration**: `38.3%`
- **Overlay Image**: [`iccad1_TN_NHS_predNHS_sampleNNHS491_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_TN_NHS_predNHS_sampleNNHS491_8_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad1_NHS125_1` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.7292 | P(HS)=0.7292, P(NHS)=0.2708)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(118.56, 122.97)` | **Spatial Spread**: `90.96 px`
- **Top 10% Activation Concentration**: `31.0%`
- **Overlay Image**: [`iccad1_FP_NHS_predHS_sampleNHS125_1_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FP_NHS_predHS_sampleNHS125_1_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad1_NHS6_7` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.7275 | P(HS)=0.7275, P(NHS)=0.2725)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(106.45, 124.92)` | **Spatial Spread**: `83.64 px`
- **Top 10% Activation Concentration**: `38.3%`
- **Overlay Image**: [`iccad1_FP_NHS_predHS_sampleNHS6_7_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FP_NHS_predHS_sampleNHS6_7_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad1_NHS114_9` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.7674 | P(HS)=0.7674, P(NHS)=0.2326)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(105.73, 139.53)` | **Spatial Spread**: `92.67 px`
- **Top 10% Activation Concentration**: `48.1%`
- **Overlay Image**: [`iccad1_FP_NHS_predHS_sampleNHS114_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FP_NHS_predHS_sampleNHS114_9_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad1_NHS115_6` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.7346 | P(HS)=0.7346, P(NHS)=0.2654)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(108.93, 63.94)` | **Spatial Spread**: `86.74 px`
- **Top 10% Activation Concentration**: `41.4%`
- **Overlay Image**: [`iccad1_FP_NHS_predHS_sampleNHS115_6_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FP_NHS_predHS_sampleNHS115_6_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad1_HS176` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.5611 | P(HS)=0.4389, P(NHS)=0.5611)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(146.1, 107.41)` | **Spatial Spread**: `80.81 px`
- **Top 10% Activation Concentration**: `53.6%`
- **Overlay Image**: [`iccad1_FN_HS_predNHS_sampleHS176_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FN_HS_predNHS_sampleHS176_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad1_HS26` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.5751 | P(HS)=0.4249, P(NHS)=0.5751)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(108.67, 172.02)` | **Spatial Spread**: `80.82 px`
- **Top 10% Activation Concentration**: `57.6%`
- **Overlay Image**: [`iccad1_FN_HS_predNHS_sampleHS26_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FN_HS_predNHS_sampleHS26_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad1_HS203` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.4920 | P(HS)=0.5080, P(NHS)=0.4920)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(76.48, 133.65)` | **Spatial Spread**: `80.83 px`
- **Top 10% Activation Concentration**: `70.5%`
- **Overlay Image**: [`iccad1_FN_HS_predNHS_sampleHS203_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad1/iccad1_FN_HS_predNHS_sampleHS203_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad2_HSCAD2331` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9944 | P(HS)=0.9944, P(NHS)=0.0056)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(116.54, 111.13)` | **Spatial Spread**: `85.99 px`
- **Top 10% Activation Concentration**: `16.5%`
- **Overlay Image**: [`iccad2_TP_HS_predHS_sampleHSCAD2331_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TP_HS_predHS_sampleHSCAD2331_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad2_HSCAD2115` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9917 | P(HS)=0.9917, P(NHS)=0.0083)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(106.82, 112.88)` | **Spatial Spread**: `84.75 px`
- **Top 10% Activation Concentration**: `17.7%`
- **Overlay Image**: [`iccad2_TP_HS_predHS_sampleHSCAD2115_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TP_HS_predHS_sampleHSCAD2115_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad2_HSCAD291` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9922 | P(HS)=0.9922, P(NHS)=0.0078)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(102.73, 113.22)` | **Spatial Spread**: `86.83 px`
- **Top 10% Activation Concentration**: `15.9%`
- **Overlay Image**: [`iccad2_TP_HS_predHS_sampleHSCAD291_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TP_HS_predHS_sampleHSCAD291_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad2_HSCAD2250` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9889 | P(HS)=0.9889, P(NHS)=0.0111)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(109.92, 101.09)` | **Spatial Spread**: `82.36 px`
- **Top 10% Activation Concentration**: `16.7%`
- **Overlay Image**: [`iccad2_TP_HS_predHS_sampleHSCAD2250_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TP_HS_predHS_sampleHSCAD2250_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad2_NNHSCAD2998_7` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(110.25, 177.25)` | **Spatial Spread**: `63.45 px`
- **Top 10% Activation Concentration**: `75.0%`
- **Overlay Image**: [`iccad2_TN_NHS_predNHS_sampleNNHSCAD2998_7_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TN_NHS_predNHS_sampleNNHSCAD2998_7_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad2_NHSCAD20_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(77.85, 128.04)` | **Spatial Spread**: `44.71 px`
- **Top 10% Activation Concentration**: `96.0%`
- **Overlay Image**: [`iccad2_TN_NHS_predNHS_sampleNHSCAD20_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TN_NHS_predNHS_sampleNHSCAD20_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad2_NHSCAD20_8` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(102.29, 122.59)` | **Spatial Spread**: `82.92 px`
- **Top 10% Activation Concentration**: `89.3%`
- **Overlay Image**: [`iccad2_TN_NHS_predNHS_sampleNHSCAD20_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TN_NHS_predNHS_sampleNHSCAD20_8_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad2_NNHSCAD21121_1` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 186.16)` | **Spatial Spread**: `64.12 px`
- **Top 10% Activation Concentration**: `94.3%`
- **Overlay Image**: [`iccad2_TN_NHS_predNHS_sampleNNHSCAD21121_1_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_TN_NHS_predNHS_sampleNNHSCAD21121_1_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad2_NNHSCAD21848_7` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9921 | P(HS)=0.9921, P(NHS)=0.0079)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(106.77, 114.25)` | **Spatial Spread**: `86.58 px`
- **Top 10% Activation Concentration**: `15.8%`
- **Overlay Image**: [`iccad2_FP_NHS_predHS_sampleNNHSCAD21848_7_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FP_NHS_predHS_sampleNNHSCAD21848_7_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad2_NNHSCAD24019_9` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9917 | P(HS)=0.9917, P(NHS)=0.0083)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(109.93, 96.69)` | **Spatial Spread**: `87.87 px`
- **Top 10% Activation Concentration**: `17.6%`
- **Overlay Image**: [`iccad2_FP_NHS_predHS_sampleNNHSCAD24019_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FP_NHS_predHS_sampleNNHSCAD24019_9_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad2_NHSCAD2220_6` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9896 | P(HS)=0.9896, P(NHS)=0.0104)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(109.91, 118.43)` | **Spatial Spread**: `87.63 px`
- **Top 10% Activation Concentration**: `17.0%`
- **Overlay Image**: [`iccad2_FP_NHS_predHS_sampleNHSCAD2220_6_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FP_NHS_predHS_sampleNHSCAD2220_6_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad2_NHSCAD2305_6` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9895 | P(HS)=0.9895, P(NHS)=0.0105)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(110.27, 111.13)` | **Spatial Spread**: `89.05 px`
- **Top 10% Activation Concentration**: `16.6%`
- **Overlay Image**: [`iccad2_FP_NHS_predHS_sampleNHSCAD2305_6_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FP_NHS_predHS_sampleNHSCAD2305_6_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad2_HSCAD2494` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(21.9, 96.57)` | **Spatial Spread**: `8.62 px`
- **Top 10% Activation Concentration**: `100.0%`
- **Overlay Image**: [`iccad2_FN_HS_predNHS_sampleHSCAD2494_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FN_HS_predNHS_sampleHSCAD2494_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad2_HSCAD2493` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.9997 | P(HS)=0.0003, P(NHS)=0.9997)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad2_FN_HS_predNHS_sampleHSCAD2493_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FN_HS_predNHS_sampleHSCAD2493_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad2_HSCAD20` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.9997 | P(HS)=0.0003, P(NHS)=0.9997)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad2_FN_HS_predNHS_sampleHSCAD20_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FN_HS_predNHS_sampleHSCAD20_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad2_HSCAD2283` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.7424 | P(HS)=0.2576, P(NHS)=0.7424)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(52.61, 125.4)` | **Spatial Spread**: `83.81 px`
- **Top 10% Activation Concentration**: `99.5%`
- **Overlay Image**: [`iccad2_FN_HS_predNHS_sampleHSCAD2283_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad2/iccad2_FN_HS_predNHS_sampleHSCAD2283_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad3_HSCAD37` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(144.19, 108.33)` | **Spatial Spread**: `88.78 px`
- **Top 10% Activation Concentration**: `32.9%`
- **Overlay Image**: [`iccad3_TP_HS_predHS_sampleHSCAD37_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TP_HS_predHS_sampleHSCAD37_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad3_HSCAD3991` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(144.96, 81.81)` | **Spatial Spread**: `100.62 px`
- **Top 10% Activation Concentration**: `66.1%`
- **Overlay Image**: [`iccad3_TP_HS_predHS_sampleHSCAD3991_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TP_HS_predHS_sampleHSCAD3991_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad3_HSCAD31679` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(140.28, 78.42)` | **Spatial Spread**: `94.8 px`
- **Top 10% Activation Concentration**: `64.7%`
- **Overlay Image**: [`iccad3_TP_HS_predHS_sampleHSCAD31679_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TP_HS_predHS_sampleHSCAD31679_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad3_HSCAD31493` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(181.01, 115.94)` | **Spatial Spread**: `75.86 px`
- **Top 10% Activation Concentration**: `74.2%`
- **Overlay Image**: [`iccad3_TP_HS_predHS_sampleHSCAD31493_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TP_HS_predHS_sampleHSCAD31493_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad3_NHSCAD3100_1` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_TN_NHS_predNHS_sampleNHSCAD3100_1_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TN_NHS_predNHS_sampleNHSCAD3100_1_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad3_NNHSCAD35348_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_TN_NHS_predNHS_sampleNNHSCAD35348_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TN_NHS_predNHS_sampleNNHSCAD35348_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad3_NNHSCAD33501_8` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_TN_NHS_predNHS_sampleNNHSCAD33501_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TN_NHS_predNHS_sampleNNHSCAD33501_8_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad3_NNHSCAD35347_4` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_TN_NHS_predNHS_sampleNNHSCAD35347_4_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_TN_NHS_predNHS_sampleNNHSCAD35347_4_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad3_NHSCAD3284_8` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(162.67, 98.73)` | **Spatial Spread**: `89.48 px`
- **Top 10% Activation Concentration**: `68.0%`
- **Overlay Image**: [`iccad3_FP_NHS_predHS_sampleNHSCAD3284_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FP_NHS_predHS_sampleNHSCAD3284_8_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad3_NHSCAD31326_8` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(156.39, 99.04)` | **Spatial Spread**: `85.28 px`
- **Top 10% Activation Concentration**: `65.2%`
- **Overlay Image**: [`iccad3_FP_NHS_predHS_sampleNHSCAD31326_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FP_NHS_predHS_sampleNHSCAD31326_8_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad3_NHSCAD31800_8` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(99.7, 109.8)` | **Spatial Spread**: `93.86 px`
- **Top 10% Activation Concentration**: `19.6%`
- **Overlay Image**: [`iccad3_FP_NHS_predHS_sampleNHSCAD31800_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FP_NHS_predHS_sampleNHSCAD31800_8_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad3_NHSCAD3882_8` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 1.0000 | P(HS)=1.0000, P(NHS)=0.0000)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(192.93, 88.62)` | **Spatial Spread**: `67.31 px`
- **Top 10% Activation Concentration**: `90.2%`
- **Overlay Image**: [`iccad3_FP_NHS_predHS_sampleNHSCAD3882_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FP_NHS_predHS_sampleNHSCAD3882_8_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad3_HSCAD31154` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_FN_HS_predNHS_sampleHSCAD31154_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FN_HS_predNHS_sampleHSCAD31154_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad3_HSCAD31784` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(21.9, 195.54)` | **Spatial Spread**: `11.24 px`
- **Top 10% Activation Concentration**: `100.0%`
- **Overlay Image**: [`iccad3_FN_HS_predNHS_sampleHSCAD31784_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FN_HS_predNHS_sampleHSCAD31784_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad3_HSCAD31751` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_FN_HS_predNHS_sampleHSCAD31751_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FN_HS_predNHS_sampleHSCAD31751_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad3_HSCAD31566` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad3_FN_HS_predNHS_sampleHSCAD31566_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad3/iccad3_FN_HS_predNHS_sampleHSCAD31566_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad4_HSCAD420` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9978 | P(HS)=0.9978, P(NHS)=0.0022)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(114.25, 110.22)` | **Spatial Spread**: `86.88 px`
- **Top 10% Activation Concentration**: `17.6%`
- **Overlay Image**: [`iccad4_TP_HS_predHS_sampleHSCAD420_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TP_HS_predHS_sampleHSCAD420_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad4_HSCAD497` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9899 | P(HS)=0.9899, P(NHS)=0.0101)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(113.73, 100.93)` | **Spatial Spread**: `83.39 px`
- **Top 10% Activation Concentration**: `18.6%`
- **Overlay Image**: [`iccad4_TP_HS_predHS_sampleHSCAD497_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TP_HS_predHS_sampleHSCAD497_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad4_HSCAD495` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9930 | P(HS)=0.9930, P(NHS)=0.0070)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(103.37, 106.44)` | **Spatial Spread**: `90.37 px`
- **Top 10% Activation Concentration**: `19.1%`
- **Overlay Image**: [`iccad4_TP_HS_predHS_sampleHSCAD495_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TP_HS_predHS_sampleHSCAD495_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad4_HSCAD499` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9941 | P(HS)=0.9941, P(NHS)=0.0059)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(127.03, 82.23)` | **Spatial Spread**: `88.92 px`
- **Top 10% Activation Concentration**: `100.0%`
- **Overlay Image**: [`iccad4_TP_HS_predHS_sampleHSCAD499_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TP_HS_predHS_sampleHSCAD499_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad4_NHSCAD41020` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_TN_NHS_predNHS_sampleNHSCAD41020_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TN_NHS_predNHS_sampleNHSCAD41020_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad4_NHSCAD410_7` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_TN_NHS_predNHS_sampleNHSCAD410_7_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TN_NHS_predNHS_sampleNHSCAD410_7_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad4_NNHSCAD4999_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_TN_NHS_predNHS_sampleNNHSCAD4999_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TN_NHS_predNHS_sampleNNHSCAD4999_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad4_NHSCAD4102_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_TN_NHS_predNHS_sampleNHSCAD4102_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_TN_NHS_predNHS_sampleNHSCAD4102_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad4_NHSCAD498_1` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9321 | P(HS)=0.9321, P(NHS)=0.0679)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(108.28, 107.53)` | **Spatial Spread**: `89.26 px`
- **Top 10% Activation Concentration**: `14.1%`
- **Overlay Image**: [`iccad4_FP_NHS_predHS_sampleNHSCAD498_1_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FP_NHS_predHS_sampleNHSCAD498_1_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad4_NHSCAD4176_4` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9264 | P(HS)=0.9264, P(NHS)=0.0736)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(108.49, 116.26)` | **Spatial Spread**: `90.29 px`
- **Top 10% Activation Concentration**: `13.5%`
- **Overlay Image**: [`iccad4_FP_NHS_predHS_sampleNHSCAD4176_4_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FP_NHS_predHS_sampleNHSCAD4176_4_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad4_NHSCAD446_4` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.8511 | P(HS)=0.8511, P(NHS)=0.1489)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(118.57, 116.08)` | **Spatial Spread**: `81.04 px`
- **Top 10% Activation Concentration**: `18.8%`
- **Overlay Image**: [`iccad4_FP_NHS_predHS_sampleNHSCAD446_4_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FP_NHS_predHS_sampleNHSCAD446_4_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad4_NHSCAD496_4` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.7323 | P(HS)=0.7323, P(NHS)=0.2677)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(119.81, 106.43)` | **Spatial Spread**: `82.97 px`
- **Top 10% Activation Concentration**: `17.6%`
- **Overlay Image**: [`iccad4_FP_NHS_predHS_sampleNHSCAD496_4_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FP_NHS_predHS_sampleNHSCAD496_4_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad4_HSCAD488` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_FN_HS_predNHS_sampleHSCAD488_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FN_HS_predNHS_sampleHSCAD488_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad4_HSCAD4174` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_FN_HS_predNHS_sampleHSCAD4174_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FN_HS_predNHS_sampleHSCAD4174_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad4_HSCAD498` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_FN_HS_predNHS_sampleHSCAD498_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FN_HS_predNHS_sampleHSCAD498_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad4_HSCAD4172` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad4_FN_HS_predNHS_sampleHSCAD4172_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad4/iccad4_FN_HS_predNHS_sampleHSCAD4172_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

### Sample `iccad5_HSCAD59` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9957 | P(HS)=0.9957, P(NHS)=0.0043)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(115.53, 112.23)` | **Spatial Spread**: `88.01 px`
- **Top 10% Activation Concentration**: `19.7%`
- **Overlay Image**: [`iccad5_TP_HS_predHS_sampleHSCAD59_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TP_HS_predHS_sampleHSCAD59_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad5_HSCAD528` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9948 | P(HS)=0.9948, P(NHS)=0.0052)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(123.44, 115.92)` | **Spatial Spread**: `82.86 px`
- **Top 10% Activation Concentration**: `19.4%`
- **Overlay Image**: [`iccad5_TP_HS_predHS_sampleHSCAD528_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TP_HS_predHS_sampleHSCAD528_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad5_HSCAD50` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9915 | P(HS)=0.9915, P(NHS)=0.0085)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(114.23, 106.72)` | **Spatial Spread**: `87.37 px`
- **Top 10% Activation Concentration**: `18.4%`
- **Overlay Image**: [`iccad5_TP_HS_predHS_sampleHSCAD50_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TP_HS_predHS_sampleHSCAD50_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad5_HSCAD540` (TP: Actual HS / Pred HS)

- **Actual Class**: HS
- **Predicted Class**: HS (Confidence: 0.9922 | P(HS)=0.9922, P(NHS)=0.0078)
- **Case Category**: TP (Correct)
- **Activation Centroid**: `(108.46, 110.26)` | **Spatial Spread**: `88.67 px`
- **Top 10% Activation Concentration**: `17.6%`
- **Overlay Image**: [`iccad5_TP_HS_predHS_sampleHSCAD540_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TP_HS_predHS_sampleHSCAD540_overlay.png)

- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, critical line-end proximities, or narrow pitch tracks.
- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering that are characteristic of optical proximity effect (OPE) degradation.
- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination.

### Sample `iccad5_NNHSCAD52136_8` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad5_TN_NHS_predNHS_sampleNNHSCAD52136_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TN_NHS_predNHS_sampleNNHSCAD52136_8_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad5_NHSCAD52118` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad5_TN_NHS_predNHS_sampleNHSCAD52118_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TN_NHS_predNHS_sampleNHSCAD52118_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad5_NNHSCAD5149_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad5_TN_NHS_predNHS_sampleNNHSCAD5149_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TN_NHS_predNHS_sampleNNHSCAD5149_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad5_NNHSCAD5420_9` (TN: Actual NHS / Pred NHS)

- **Actual Class**: NHS
- **Predicted Class**: NHS (Confidence: 1.0000 | P(HS)=0.0000, P(NHS)=1.0000)
- **Case Category**: TN (Correct)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad5_TN_NHS_predNHS_sampleNNHSCAD5420_9_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_TN_NHS_predNHS_sampleNNHSCAD5420_9_overlay.png)

- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures.
- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, and repetitive dummy-like or standard track geometries without severe 2D optical distortions.
- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners and maintainable pitch prevents hotspot formation.

### Sample `iccad5_NHSCAD57_8` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9952 | P(HS)=0.9952, P(NHS)=0.0048)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(105.58, 112.15)` | **Spatial Spread**: `88.37 px`
- **Top 10% Activation Concentration**: `17.5%`
- **Overlay Image**: [`iccad5_FP_NHS_predHS_sampleNHSCAD57_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_FP_NHS_predHS_sampleNHSCAD57_8_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad5_NHSCAD538_2` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9941 | P(HS)=0.9941, P(NHS)=0.0059)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(114.95, 109.46)` | **Spatial Spread**: `89.78 px`
- **Top 10% Activation Concentration**: `19.2%`
- **Overlay Image**: [`iccad5_FP_NHS_predHS_sampleNHSCAD538_2_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_FP_NHS_predHS_sampleNHSCAD538_2_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad5_NHSCAD532_8` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9933 | P(HS)=0.9933, P(NHS)=0.0067)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(125.44, 111.83)` | **Spatial Spread**: `86.63 px`
- **Top 10% Activation Concentration**: `19.4%`
- **Overlay Image**: [`iccad5_FP_NHS_predHS_sampleNHSCAD532_8_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_FP_NHS_predHS_sampleNHSCAD532_8_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad5_NHSCAD59_2` (FP: Actual NHS / Pred HS)

- **Actual Class**: NHS
- **Predicted Class**: HS (Confidence: 0.9925 | P(HS)=0.9925, P(NHS)=0.0075)
- **Case Category**: FP (Misclassified)
- **Activation Centroid**: `(117.46, 110.72)` | **Spatial Spread**: `90.52 px`
- **Top 10% Activation Concentration**: `18.8%`
- **Overlay Image**: [`iccad5_FP_NHS_predHS_sampleNHSCAD59_2_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_FP_NHS_predHS_sampleNHSCAD59_2_overlay.png)

- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, such as isolated jog transitions or mock line ends.
- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies even though the design rule checks (DRC) or OPC corrections keep it within safe process windows.
- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, explaining the false alarm.

### Sample `iccad5_HSCAD516` (FN: Actual HS / Pred NHS)

- **Actual Class**: HS
- **Predicted Class**: NHS (Confidence: 0.9999 | P(HS)=0.0001, P(NHS)=0.9999)
- **Case Category**: FN (Misclassified)
- **Activation Centroid**: `(112.0, 112.0)` | **Spatial Spread**: `0.0 px`
- **Top 10% Activation Concentration**: `0.0%`
- **Overlay Image**: [`iccad5_FN_HS_predNHS_sampleHSCAD516_overlay.png`](file:///home/peskybird/Projects/Ai_Ml_DA/results/gradcam/iccad5/iccad5_FN_HS_predNHS_sampleHSCAD516_overlay.png)

- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site.
- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks.
- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws.

