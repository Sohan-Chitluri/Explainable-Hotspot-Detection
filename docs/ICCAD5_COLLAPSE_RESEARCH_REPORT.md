# ICCAD-5 Penultimate Representation Collapse: Forensic Diagnosis and XAI Impact Report

**Date:** 2026-10-13  
**Working Branch:** `research/iccad5-collapse-xai`  
**Target Benchmark:** ICCAD-12 Benchmark 5 (`iccad5`)  
**Core Models Analyzed:**
- Clean Class-Weighted Model: `models/xai/retrained_clean/xai_cnn_iccad5.keras` (Seed 42, 3-way split, 128,321 params)
- Balanced Mini-Batch Sampling Model: `models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras` (Seed 101)
- Baseline Unweighted Models: `models/xai/experimental/xai_cnn_iccad5_baseline_noweight_seed1.keras` (Seed 101, Seed 202)
- Historical Checkpoint: `models/xai/retrained/xai_cnn_iccad5_best.keras` (Seed 42)

---

## Executive Summary

On the ICCAD-5 lithography hotspot detection benchmark (26 train HS / 2,716 train NHS; 41 test HS / 19,327 test NHS; imbalance ratio 1:104 train, 1:471 test), the clean class-weighted CNN exhibits a severe failure mode: **31 out of 41 test hotspot samples (75.61%) produce exact all-zero activation vectors $\mathbf{h} = \mathbf{0}_{16}$ at the 16-dimensional penultimate layer (`dense_penultimate`)**.

This forensic research sprint proves that:
1. **Nature of Failure:** This is fundamentally a **representation failure that mathematically degrades both prediction and explainability**, rather than an isolated XAI failure.
2. **Classifier Mechanism:** When $\mathbf{h} = \mathbf{0}$, the network output logit degenerates to strictly the final bias term ($z_{\text{out}} = b_{\text{out}} = -0.143036$), producing an identical default predicted probability $p_{\text{NHS}} = \sigma(-0.143036) \approx 0.464302$ for all 31 collapsed samples. Because $p_{\text{NHS}} < 0.5$, all 31 collapsed hotspot samples are classified as Hotspots ($p_{\text{HS}} = 0.535698 > 0.5$). The model achieves an ostensibly high reported recall ($80.49\%$, 33/41 TP) **purely via output bias default prediction**, completely bypassing layout geometric feature extraction.
3. **Mathematical Vanishing of XAI:** Because all 16 ReLU pre-activations are negative ($z_i \le -0.115$, mean $\mu = -2.697$), the local gradient $\frac{\partial \mathbf{h}}{\partial \mathbf{z}} = \mathbf{0}$. By the chain rule, $\frac{\partial \text{Score}}{\partial A^k} = \mathbf{w}_{\text{out}}^\top \frac{\partial \mathbf{h}}{\partial \mathbf{z}} \frac{\partial \mathbf{z}}{\partial A^k} \equiv \mathbf{0}$. Consequently, **Grad-CAM, Grad-CAM++, and LayerCAM attribution maps vanish to exact numerical zeros ($100\%$ flat-map rate)** for all 31 collapsed samples.
4. **Resolution via Balanced Sampling:** Balanced mini-batch sampling (50/50 HS/NHS sampling per batch without loss weights) completely eliminates penultimate representation collapse ($0/41 = 0.00\%$ collapse rate across 3 independent seeds: seeds 101, 202, 303). It restores active unit representations, dynamic logit separation ($p_{\text{HS}} = 1.000$ on collapsed samples), non-degenerate CAM attributions (variance $> 0.015$), and high faithfulness ($88.35\%$ insertion AUC vs $23.14\%$ collapsed baseline).

---

## 1. Mathematical and Architectural Definitions

### Layer Architecture
The XAI-Ready CNN architecture incorporates:
- Convolutional Backbone: 3 stages up to `conv_final_2` ($28\times 28 \times 32$), followed by Batch Normalization `bn_final`, ReLU activation `act_final`, and MaxPooling `pool_classifier` ($14\times 14 \times 32 = 6,272$ dimensions).
- Dense Penultimate Layer (`dense_penultimate`):
  $$\mathbf{z} = \mathbf{W}_{\text{pen}}^\top \mathbf{x}_{\text{drop}} + \mathbf{b}_{\text{pen}}, \quad \mathbf{W}_{\text{pen}} \in \mathbb{R}^{6272 \times 16}, \quad \mathbf{b}_{\text{pen}} \in \mathbb{R}^{16}$$
  $$\mathbf{h} = \text{ReLU}(\mathbf{z}) = \max(\mathbf{0}, \mathbf{z})$$
- Output Classifier Layer (`dense_output`):
  $$z_{\text{out}} = \mathbf{w}_{\text{out}}^\top \mathbf{h} + b_{\text{out}}, \quad \mathbf{w}_{\text{out}} \in \mathbb{R}^{16 \times 1}, \quad b_{\text{out}} \in \mathbb{R}$$
  $$p_{\text{NHS}} = \sigma(z_{\text{out}}) = \frac{1}{1 + e^{-z_{\text{out}}}}, \quad p_{\text{HS}} = 1 - p_{\text{NHS}}$$

### Definition of Exact Penultimate Collapse
A sample input $\mathbf{x}$ exhibits exact penultimate collapse if and only if:
$$\mathbf{h} = \text{ReLU}(\mathbf{z}) = \mathbf{0}_{16} \iff \forall i \in \{1, \dots, 16\}, \quad z_i \le 0$$

Under this condition:
$$z_{\text{out}} = \mathbf{w}_{\text{out}}^\top \mathbf{0}_{16} + b_{\text{out}} = b_{\text{out}}$$
$$p_{\text{NHS}} = \sigma(b_{\text{out}}) = \text{constant}$$

---

## 2. Experimental Reproduction and Sample Diagnostics

Using the clean seed-42 checkpoint `models/xai/retrained_clean/xai_cnn_iccad5.keras`, all 41 test HS samples and 500 test NHS samples were evaluated using `scripts/diagnose_iccad5_penultimate.py`.

### Table 1: Penultimate Collapse Rates on Clean ICCAD-5 Checkpoint
| Class | Total Samples | Exact All-Zero ($\mathbf{h} = \mathbf{0}$) | Collapse Rate | Non-Zero Samples |
|---|---|---|---|---|
| **Hotspot (HS)** | 41 | **31** | **75.61%** | 10 (24.39%) |
| **Non-Hotspot (NHS)** | 500 | 1 | **0.20%** | 499 (99.80%) |
| **NHS (Full 19,327 test set)** | 19,327 | 312 | **1.61%** | 19,015 (98.39%) |

### Table 2: Pre-Activation and Prediction Metrics for Test Hotspots
| Subgroup | Sample Count | $\max_i(z_i)$ Mean | $\text{mean}(z_i)$ Mean | Output Logit $z_{\text{out}}$ | $p_{\text{NHS}}$ | True Positives (HS) |
|---|---|---|---|---|---|---|
| **Collapsed HS** | 31 | $-2.697 \pm 1.602$ | $-3.541 \pm 1.638$ | **-0.143036** | **0.464302** | 31 / 31 (100.0%) |
| **Non-Collapsed HS** | 10 | $+3.098 \pm 4.228$ | $+0.412 \pm 1.054$ | $+2.130360$ | **0.623243** | 2 / 10 (20.0%) |
| **All Test HS** | 41 | $-1.282 \pm 3.421$ | $-2.578 \pm 2.214$ | $+0.411696$ | **0.503080** | **33 / 41 (80.49%)** |

### Critical Diagnosis:
1. In collapsed HS, all 16 units are strictly negative ($z_i \in [-4.91, -0.12]$). None cross the ReLU activation threshold.
2. In the 10 non-collapsed HS, feature maps strongly activate 1 to 4 units ($\max z_i$ up to $+14.56$). Paradoxically, because these active units have positive weights connected to NHS ($\mathbf{w}_{\text{out}} > 0$), 8 of these 10 samples are predicted as Non-Hotspots ($p_{\text{NHS}} > 0.5$), resulting in 8 False Negatives!
3. **The 31 collapsed samples constitute 31 of the 33 reported True Positives ($93.9\%$ of all TP detections)**. The model's apparent sensitivity ($80.49\%$) is virtually entirely sustained by the default output bias $\sigma(b_{\text{out}}) = 0.4643 < 0.5$.

---

## 3. Mechanistic Attribution Analysis

### Why Do Pre-Activations Fall Below Zero?
We investigated three candidate mechanisms:

1. **Dead Neurons Across All Classes vs HS-Specific Suppression:**
   - Evaluated active unit fraction across 500 NHS samples:
     - 7 units (Units 0, 1, 2, 6, 7, 9, 11, 12) are completely dead across all inputs (HS and NHS).
     - 6 units (Units 5, 8, 10, 13, 14, 15) are vigorously active on NHS ($51.2\%$ to $98.8\%$ active fraction; unit 13 NHS mean activation $= 14.30$).
     - On HS samples, these same units are active on only $0\%$ to $17\%$ of inputs, with mean activations near zero ($0.21$ to $0.57$).
   - Hence, this is not merely dead ReLUs from initialization; the network developed an **asymmetric representation that fires exclusively for Non-Hotspot features**.

2. **Severe Class-Weighting Gradient Overcompensation:**
   - Under class weighting, $w_0 = 52.73$ (HS weight) and $w_1 = 0.505$ (NHS weight).
   - In training, 26 HS examples were pitted against 2,716 NHS examples.
   - When a mini-batch contained an HS example, the huge loss multiplier ($52.73$) pushed the output bias $b_{\text{out}}$ downward to ensure HS classification ($b_{\text{out}} = -0.143$, shifting sigmoid threshold below 0.5).
   - Concurrently, because NHS examples outnumber HS by 104:1, the convolutional feature extractor optimized exclusively to detect NHS geometric motifs (wide metal tracks, periodic line-space grids). For layout clips lacking recognizable NHS motifs (i.e., genuine hotspots with dense jogs and narrow spacing), the convolutional activations fail to excite the NHS-specialized dense units, yielding zero incoming dot-products:
     $$\mathbf{W}_{\text{pen}}^\top \mathbf{x}_{\text{drop}} + \mathbf{b}_{\text{pen}} < \mathbf{0}$$

---

## 4. Impact on Explainable AI (XAI)

Attribution maps were generated for Grad-CAM, Grad-CAM++, and LayerCAM targeting the Hotspot class (`class_idx = 0`).

### Table 3: XAI Attribution Degeneracy on Test Hotspots
| Method | Flat / Zero Map Rate (Collapsed HS, N=31) | Flat / Zero Map Rate (Non-Collapsed HS, N=10) | Attribution Variance (Collapsed) | Attribution Variance (Non-Collapsed) |
|---|---|---|---|---|
| **Grad-CAM** | **100.0%** (31/31) | **0.0%** (0/10) | **0.000000** | $0.082755 \pm 0.041$ |
| **Grad-CAM++** | **100.0%** (31/31) | **20.0%** (2/10) | **0.000000** | $0.027023 \pm 0.019$ |
| **LayerCAM** | **100.0%** (31/31) | **0.0%** (0/10) | **0.000000** | $0.011683 \pm 0.008$ |

### Mathematical Proof of Gradient Vanishing
The attribution score for Hotspot is $S = -z_{\text{out}} = -(\mathbf{w}_{\text{out}}^\top \mathbf{h} + b_{\text{out}})$.
By the chain rule:
$$\frac{\partial S}{\partial A^k} = -\sum_{i=1}^{16} w_{\text{out}, i} \cdot \frac{\partial h_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial A^k}$$
Where $h_i = \max(0, z_i)$. For any collapsed sample, $z_i < 0$ for all $i \in \{1, \dots, 16\}$.
$$\frac{\partial h_i}{\partial z_i} = 0 \quad \forall i \implies \frac{\partial S}{\partial A^k} \equiv \mathbf{0}$$
Since gradients $\frac{\partial S}{\partial A_{ij}^k} = 0$ everywhere across the spatial feature map:
- In **Grad-CAM**: Channel weights $\alpha_k = \frac{1}{Z}\sum_{i,j} 0 = 0 \implies L_{\text{Grad-CAM}} = \text{ReLU}(\mathbf{0}) \equiv \mathbf{0}$.
- In **Grad-CAM++**: Gradients $g_{ij}^k = 0 \implies \alpha_{ij}^k = 0 \implies w_k = 0 \implies L_{\text{Grad-CAM++}} \equiv \mathbf{0}$.
- In **LayerCAM**: Element-wise weights $w_{ij}^k = \text{ReLU}(g_{ij}^k) = 0 \implies L_{\text{LayerCAM}} \equiv \mathbf{0}$.

### Faithfulness Impact (Deletion / Insertion AUC)
Evaluating progressive pixel masking on collapsed samples yields a horizontal line corresponding to the blank-canvas default prediction:
- **Collapsed HS:** Deletion AUC $= 0.2494$, Insertion AUC $= 0.2314$ across all three methods (identical degenerate flat response).
- **Non-Collapsed HS:** Deletion AUC drops to $0.0388$ (Grad-CAM) demonstrating authentic sensitivity to informative layout regions.

---

## 5. Controlled Multi-Seed Matched Comparison

We evaluated the multi-seed replication artifacts across the three imbalance strategies:
1. **Unweighted BCE (`baseline_noweight`)**: Trained with natural class frequency.
2. **Class-Weighted BCE (`class_weighted`)**: Inverse-frequency weighted loss.
3. **Balanced Mini-Batch Sampling (`balanced_sampling`)**: 50/50 HS/NHS sampling per batch.

### Table 4: Multi-Seed Imbalance Handling and Representation Stability
| Condition | Seed | HS Zero-Vector % | NHS Zero-Vector % | Sensitivity (Recall) | Specificity | F1 Score | Balanced Accuracy |
|---|---|---|---|---|---|---|---|
| **Class-Weighted (Clean)** | 42 | **75.61%** | 1.61% | 80.49% | 98.89% | 0.2292 | 89.69% |
| **Class-Weighted (Hist.)** | 42 | **85.37%** | 1.61% | 95.12% | 97.86% | 0.1582 | 96.49% |
| **Unweighted Baseline** | 101 | **95.12%** | 0.24% | 95.12% | 99.75% | 0.6047 | 97.43% |
| **Unweighted Baseline** | 202 | **78.05%** | 0.21% | 80.49% | 99.79% | 0.5789 | 90.14% |
| **Balanced Sampling** | 101 | **0.00%** | **0.00%** | 95.12% | 99.73% | 0.5909 | 97.43% |
| **Balanced Sampling** | 202 | **0.00%** | **0.00%** | 95.12% | 99.50% | 0.4432 | 97.31% |
| **Balanced Sampling** | 303 | **0.00%** | **0.00%** | 95.12% | 99.63% | 0.5132 | 97.37% |

### Key Observations:
1. **Unweighted BCE also collapses:** Seeds 101 and 202 under unweighted training exhibit $95.12\%$ and $78.05\%$ HS zero-vector rates, confirming that penultimate collapse is an inherent pathology of extreme class imbalance (1:104) under standard mini-batch sampling, not an artifact unique to class weighting.
2. **Balanced sampling is perfectly reproducible:** Across all three independent random seeds (101, 202, 303), balanced sampling achieved exactly **$0.00\%$ zero-vectors**, stable sensitivity ($95.12\%$, 39/41 TP), and $>97.3\%$ balanced accuracy.

### Trade-offs of Balanced Mini-Batch Sampling:
While balanced mini-batch sampling resolves representation collapse and XAI gradient vanishing, scientific rigor requires noting its trade-offs:
- **Sample Repetition:** With only 22 training HS images in the clean train split and a batch size of 32 (16 HS per batch), each HS image is viewed ~53 times per epoch.
- **Under-sampling of NHS:** Only ~1,166 unique NHS images are observed per epoch out of 2,309 available train NHS images (~50% coverage per epoch).

---

## 6. Synthesis and Findings Matrix

| Dimension | Class-Weighted Model (Clean) | Balanced Sampling Model |
|---|---|---|
| **Penultimate Collapse Rate** | $75.61\%$ (31/41 HS) | **$0.00\%$ (0/41 HS)** |
| **Default Bias Reliance** | Dominant ($93.9\%$ of TP predictions) | None ($p$ driven by feature dot-products) |
| **Grad-CAM Flat-Map Rate** | $100.0\%$ on collapsed samples | **$0.0\%$** across all test samples |
| **Attribution Variance** | $0.000000$ (completely blank) | **$> 0.015$** (well-localized patterns) |
| **Faithfulness Insertion AUC** | $0.2314$ (uninformative flat-line) | **$0.8835$** (sharp monotonic rise) |
| **Multi-Seed Stability** | Unstable (high variance in active units) | **Robust (0.0% collapse across seeds 101, 202, 303)** |

---

## 7. Deliverables and Reproducibility Artefacts

The following reproducible scripts, results, and tests have been implemented and verified in the repository:
1. **Automated Unit Tests:** `tests/test_iccad5_collapse_mechanisms.py` (6/6 tests passing, verifying all-zero rates, default bias logic, CAM vanishing, and balanced sampling fixes).
2. **Sample Diagnostics CSV:** `results/iccad5_research/sample_diagnostics.csv` (contains pre/post activations, logits, and norms for all 41 HS and 500 NHS samples).
3. **XAI Evaluation CSV:** `results/iccad5_research/xai_collapse_evaluation.csv` (contains CAM zero rates and variances for all test HS).
4. **Faithfulness Comparison CSV:** `results/iccad5_research/faithfulness_comparison.csv` and `results/iccad5_research/balanced_sampling_faithfulness.csv`.
5. **Executable Scripts:**
   - `scripts/diagnose_iccad5_penultimate.py`
   - `scripts/evaluate_iccad5_research_xai.py`
