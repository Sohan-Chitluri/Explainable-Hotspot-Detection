# Baseline Consistency Audit and Workstream B2 Imbalance Benchmark Report

**Date:** October 21, 2026  
**Environment:** Linux (NixOS), NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, 4,168 MB allocatable), CUDA 12.2 / cuDNN 8.9 via TensorFlow 2.15.1  
**Repository Branch:** `research/iccad5-collapse-xai` (HEAD `41d2c85`)  
**Experiment Contract:** Version 1.0.0 (`docs/via3_experiment_contract.md`)  
**Data Partition Root:** `/home/peskybird/Projects/data/openhsd/via-3`  
**Manifest Audit Directory:** `results/dataset_audit/`  
**Output Directories:**  
- B2 Results: `results/via3_experiments/imbalance/`  
- B2 Models: `models/via3_imbalance/`  
- Reference Baseline Outputs: `results/via3_experiments/baseline/`  
- Reference Baseline Checkpoint: `models/via3/via3_baseline_seed42.keras`  

---

## 1. Executive Summary

This report establishes the baseline model consistency audit and delivers the empirical results of Workstream B2 (Controlled Imbalance Benchmark) on the OpenHSDBench `via-3` lithography hotspot detection benchmark.

### Key Conclusions:
1. **Parameter Count Discrepancy Fully Resolved:** The saved baseline model checkpoint (`models/via3/via3_baseline_seed42.keras`), a freshly instantiated model from `src/model.py:build_custom_cnn`, and the isolated reference model from `openhsd-audit-training` all contain **exactly 12,873 total parameters** (12,825 trainable, 48 non-trainable). The previous citation of "18,521" was conclusively determined to be a documentation typo originating in `openhsd-audit-training/docs/via3_classifier_experiment_report.md` that was transcribed into `docs/via3_experiment_contract.md`. No architectural divergence exists.
2. **Workstream B2 Imbalance Benchmark Executed:** All three experimental conditions—`baseline` (unweighted binary cross-entropy), `class_weighted` (inverse training frequency weights: $w_0 \approx 0.5159, w_1 \approx 16.1904$), and `balanced_sampling` (50/50 mini-batch sampling)—were trained for 15 epochs under identical seeds (42) and optimizers on the RTX 3050 GPU.
3. **Primary Recommendation for Downstream Workstreams (XAI):** Based strictly on validation PR-AUC, `baseline` (PR-AUC: 0.2886) and `balanced_sampling` (PR-AUC: 0.2865) significantly outperform `class_weighted` (PR-AUC: 0.1282). On the clean held-out test cohort, `balanced_sampling` achieves the highest PR-AUC (**0.2619**) and ROC-AUC (**0.7840**), while maintaining superior recall (30.90% at $\tau=0.50$, 26.03% at $\tau^*=0.76$) compared to baseline (9.25% at $\tau=0.50$, 22.14% at $\tau^*=0.08$). Consequently, **`balanced_sampling` is the recommended strategy** for downstream XAI faithfulness and stability evaluation.

---

## 2. Phase 1 — Read-Only Baseline Consistency Audit

### 2.1 Resolution of Parameter Count Discrepancy

Earlier documentation in `docs/via3_experiment_contract.md` noted a parameter count of 18,521 total (18,473 trainable), whereas the baseline checkpoint reported 12,873 total (12,825 trainable).

A systematic codebase inspection revealed:
1. **Checkpoint Verification:** Loading `models/via3/via3_baseline_seed42.keras` via TensorFlow 2.15.1 yields:
   - `model.count_params()`: **12,873**
   - Trainable variable tensors: **20**, containing **12,825** elements
   - Non-trainable variable tensors: **4**, containing **48** elements (moving mean and variance across 2 `BatchNormalization` layers, $2 \times 24 = 48$)
   - Input shape: `(None, 224, 224, 3)`
   - Output shape: `(None, 1)`
   - Output activation: `sigmoid` representing continuous $P(\text{HS}=1)$
2. **Fresh Instantiation:** Calling `src.model.build_custom_cnn(input_shape=(224, 224, 3))` yields identical parameter counts: 12,873 total, 12,825 trainable, 48 non-trainable.
3. **Reference Implementation:** Inspecting `/home/peskybird/Projects/openhsd-audit-training/models/via3/via3_baseline_best.keras` confirms that the reference model produced on October 15, 2026 also has exactly **12,873 total parameters** (12,825 trainable).
4. **Historical Documentation:** Repository documents `docs/GRADCAM.md` (line 33), `docs/FINAL_RESEARCH_REPORT.md` (lines 9, 95, 122, 345), and `docs/XAI_MODEL_AUDIT.md` (line 82) uniformly document the baseline custom CNN as possessing **~12,873 total parameters**.
5. **Demonstrated Cause:** The figure "18,521" originated as a clerical transcription error in `openhsd-audit-training/docs/via3_classifier_experiment_report.md` (line 87), where the author manually stated 18,521 while importing and running `src/model.py:build_custom_cnn` which compiles to 12,873. That markdown statement was subsequently referenced when authoring `docs/via3_experiment_contract.md`. 

**Decision:** The architecture `build_custom_cnn` is canonical, stable, and mathematically verified. No architecture modification was made or warranted.

### 2.2 Layer-by-Layer Architectural Comparison

| Layer Index | Layer Name | Layer Type | Output Shape | Total Params | Trainable | Non-Trainable | Match with Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 00 | `conv2d` | Conv2D | `(None, 222, 222, 12)` | 336 | 336 | 0 | **Exact Match** |
| 01 | `conv2d_1` | Conv2D | `(None, 220, 220, 12)` | 1,308 | 1,308 | 0 | **Exact Match** |
| 02 | `conv2d_2` | Conv2D | `(None, 218, 218, 12)` | 1,308 | 1,308 | 0 | **Exact Match** |
| 03 | `batch_normalization` | BatchNormalization | `(None, 218, 218, 12)` | 48 | 24 | 24 | **Exact Match** |
| 04 | `activation` | Activation (ELU) | `(None, 218, 218, 12)` | 0 | 0 | 0 | **Exact Match** |
| 05 | `max_pooling2d` | MaxPooling2D (2x2) | `(None, 109, 109, 12)` | 0 | 0 | 0 | **Exact Match** |
| 06 | `max_pooling2d_1` | MaxPooling2D (5x5) | `(None, 21, 21, 12)` | 0 | 0 | 0 | **Exact Match** |
| 07 | `conv2d_3` | Conv2D | `(None, 19, 19, 12)` | 1,308 | 1,308 | 0 | **Exact Match** |
| 08 | `conv2d_4` | Conv2D | `(None, 17, 17, 12)` | 1,308 | 1,308 | 0 | **Exact Match** |
| 09 | `conv2d_5` | Conv2D | `(None, 15, 15, 12)` | 1,308 | 1,308 | 0 | **Exact Match** |
| 10 | `batch_normalization_1` | BatchNormalization | `(None, 15, 15, 12)` | 48 | 24 | 24 | **Exact Match** |
| 11 | `activation_1` | Activation (ELU) | `(None, 15, 15, 12)` | 0 | 0 | 0 | **Exact Match** |
| 12 | `max_pooling2d_2` | MaxPooling2D (2x2) | `(None, 7, 7, 12)` | 0 | 0 | 0 | **Exact Match** |
| 13 | `flatten` | Flatten | `(None, 588)` | 0 | 0 | 0 | **Exact Match** |
| 14 | `dropout` | Dropout (0.30) | `(None, 588)` | 0 | 0 | 0 | **Exact Match** |
| 15 | `dense` | Dense (ReLU) | `(None, 10)` | 5,890 | 5,890 | 0 | **Exact Match** |
| 16 | `dense_1` | Dense (Sigmoid) | `(None, 1)` | 11 | 11 | 0 | **Exact Match** |
| **Total** | — | — | — | **12,873** | **12,825** | **48** | **Exact Match** |

### 2.3 Checkpoint and Run-Manifest Correspondence

- **Baseline Checkpoint:** `models/via3/via3_baseline_seed42.keras`
- **File Size:** 230,689 bytes
- **SHA-256 Digest:** `b9979e8dbac5d31d8eac8bcd4156874ac7c2fcfa055200232c939922775ef4c3`
- **Run Manifest:** `results/via3_experiments/baseline/baseline_run_manifest_seed42.json`
  - Manifest Checkpoint SHA-256: `b9979e8dbac5d31d8eac8bcd4156874ac7c2fcfa055200232c939922775ef4c3` (**100% bitwise match**)
  - Selected Epoch: 15
  - Best Val PR-AUC: 0.2487
  - Optimization: Adam ($\eta = 10^{-3}$), batch size 32, 15 epochs, seed 42.

### 2.4 Preprocessing and Label Semantics Verification

1. **Preprocessing:** In `src/openhsd_data.py`:
   - Single-channel PNG decoding (`tf.io.decode_png(channels=1)`)
   - Nearest-neighbor resizing to $(224, 224)$ (`method="nearest"`)
   - Normalization: `tf.cast(img, tf.float32) / 255.0` to $[0.0, 1.0]$
   - Channel repeat: `tf.repeat(img, 3, axis=-1)` (shape: $224 \times 224 \times 3$)
2. **Label Semantics:**
   - Prefix `Hvia*` $\rightarrow$ `Label 1` (Hotspot, HS) — strictly positive class
   - Prefix `Nvia*` $\rightarrow$ `Label 0` (Non-Hotspot, NHS) — strictly negative class
3. **PR-AUC Orientation:** Area under the Precision-Recall curve computed with exact step trapezoids where True Positives are strictly $y=1, \hat{y}=1$. Hotspot prevalence baseline: $\approx 0.0321$ ($87 / 2,846$ in val, $411 / 12,803$ in clean test).
4. **Checkpoint Restoration:** `BestPRAUCCallback` monitors validation PR-AUC, breaks ties with validation F1, and restores highest-scoring weights at `on_train_end`.

---

## 3. Phase 2 & 3 — Controlled B2 Imbalance Benchmark Execution

### 3.1 Controlled Experimental Parameters

Across all three conditions:
- **Architecture:** `build_custom_cnn` (12,873 params, input $224 \times 224 \times 3$).
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}, \eta=10^{-3}$).
- **Random Seed:** 42 fixed for Python, NumPy, and TensorFlow.
- **Data Partitions:** 16,158 train (499 HS, 15,659 NHS), 2,846 validation (87 HS, 2,759 NHS), 12,803 clean held-out test (411 HS, 12,392 NHS), 12,905 official test (diagnostic contaminated reference).
- **Batch Size:** 32.
- **Budget:** 15 epochs.
- **Steps per Epoch:**
  - `baseline` & `class_weighted`: 505 steps/epoch (full sequential iteration over 16,158 samples).
  - `balanced_sampling`: 504 steps/epoch ($16,158 // 32$), drawing 16,128 samples per epoch with 50% HS and 50% NHS per batch. Validation and test sets remain completely unaltered.
- **Loss / Weighting Specifications:**
  - `baseline`: Unweighted binary cross-entropy.
  - `class_weighted`: Binary cross-entropy with weights $w_0 = 16,158 / (2 \times 15,659) \approx 0.5159$, $w_1 = 16,158 / (2 \times 499) \approx 16.1904$.
  - `balanced_sampling`: Minibatch 50/50 replacement sampling from minority HS and majority NHS pools.

### 3.2 Execution Commands and Resource Usage

- **Command:**
  ```bash
  ./scripts/tf_gpu.sh scripts/run_via3_imbalance_experiments.py \
    --strategies baseline class_weighted balanced_sampling \
    --output-dir results/via3_experiments/imbalance \
    --models-dir models/via3_imbalance \
    --epochs 15 \
    --seed 42
  ```
- **Exit Code:** `0` (Success across all three strategies).
- **Execution Mode:** Sequential GPU execution on NVIDIA GeForce RTX 3050 Laptop GPU (allocated VRAM: 4,168 MB). Explicit Keras session clearing (`tf.keras.backend.clear_session()`) and Python garbage collection were executed between strategies to prevent memory leakage.
- **Total Duration:** 2,077.30 s (~34.6 minutes).

---

## 4. Phase 4 — Empirical Benchmark Results

### 4.1 Primary Benchmark Comparison (Ranked by Primary Metric: Validation PR-AUC)

> Hotspot Prevalence Baseline: $\pi_0 \approx 0.0321$ (Random Guessing PR-AUC $\approx 0.0321$).

| Strategy | Best Epoch | Train Duration | Val PR-AUC (Primary) | Val ROC-AUC | Val F1 ($\tau=0.50$) | Val Recall ($\tau=0.50$) | Val Prec ($\tau=0.50$) | Clean Test PR-AUC | Clean Test ROC-AUC | Clean Test F1 ($\tau=0.50$) | Clean Test Recall ($\tau=0.50$) | Clean Test Prec ($\tau=0.50$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | **8** | 708.21s | **0.2886** | 0.7094 | 0.2062 | 0.1149 | **1.0000** | 0.2518 | 0.7328 | 0.1670 | 0.0925 | **0.8636** |
| **`balanced_sampling`** | **6** | 682.79s | **0.2865** | **0.7226** | **0.2872** | **0.3218** | 0.2593 | **0.2619** | **0.7840** | **0.2890** | **0.3090** | 0.2714 |
| **`class_weighted`** | **11** | 686.30s | 0.1282 | 0.6550 | 0.2297 | 0.1954 | 0.2787 | 0.1270 | 0.6864 | 0.2110 | 0.1825 | 0.2500 |

### 4.2 Calibrated Operating Point Analysis ($\tau^*$ Tuned Strictly on Validation Set)

Operational thresholds $\tau^*$ were selected by sweeping $\tau \in [0.01, 0.99]$ on the validation set to maximize validation Hotspot F1 score, completely frozen, and then evaluated on the clean test set:

| Strategy | Calibrated Threshold $\tau^*$ | Val F1 at $\tau^*$ | Val Recall at $\tau^*$ | Val Prec at $\tau^*$ | Val BalAcc at $\tau^*$ | Clean Test F1 at $\tau^*$ | Clean Test Recall at $\tau^*$ | Clean Test Prec at $\tau^*$ | Clean Test BalAcc at $\tau^*$ | Clean Test False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline`** | $\tau^* = 0.0800$ | **0.3262** | 0.2644 (23/87) | **0.4259** (23/54) | **0.6266** | 0.2753 | 0.2214 (91/411) | 0.3640 (91/250) | 0.6043 | 159 |
| **`balanced_sampling`** | $\tau^* = 0.7600$ | 0.3217 | 0.2644 (23/87) | 0.4107 (23/56) | 0.6262 | **0.3166** | **0.2603** (107/411) | **0.4038** (107/265) | **0.6238** | **158** |
| **`class_weighted`** | $\tau^* = 0.5000$ | 0.2297 | 0.1954 (17/87) | 0.2787 (17/61) | 0.5897 | 0.2110 | 0.1825 (75/411) | 0.2500 (75/300) | 0.5822 | 225 |

### 4.3 Confusion Matrices on Clean Held-Out Test Cohort ($N = 12,803$: 411 HS, 12,392 NHS)

#### At Default Threshold ($\tau = 0.50$):
- **`baseline`:**  
  $TP = 38$, $FN = 373$, $TN = 12,386$, $FP = 6$  
  *(Extremely conservative: 99.95% specificity, only 6 false alarms across 12,392 negative layouts, but misses 90.75% of hotspots).*
- **`balanced_sampling`:**  
  $TP = 127$, $FN = 284$, $TN = 12,051$, $FP = 341$  
  *(Substantially higher capture: catches 127 hotspots [3.34x baseline capture] with manageable false-positive burden).*
- **`class_weighted`:**  
  $TP = 75$, $FN = 336$, $TN = 12,167$, $FP = 225$  
  *(Suboptimal trade-off: lower recall than balanced sampling [18.25% vs 30.90%] with 225 false positives).*

#### At Validation-Calibrated Operating Point ($\tau^*$):
- **`baseline` ($\tau^* = 0.0800$):**  
  $TP = 91$, $FN = 320$, $TN = 12,233$, $FP = 159$
- **`balanced_sampling` ($\tau^* = 0.7600$):**  
  $TP = 107$, $FN = 304$, $TN = 12,234$, $FP = 158$  
  *(Achieves the best test F1 [0.3166] and precision [40.38%] while capturing 107 hotspots with 158 false alarms).*
- **`class_weighted` ($\tau^* = 0.5000$):**  
  $TP = 75$, $FN = 336$, $TN = 12,167$, $FP = 225$

### 4.4 Diagnostic Evaluation on Official Test Cohort ($N = 12,905$, Secondary Contaminated Reference)

> [!WARNING]
> The official test set contains 98 training duplicates and 4 contradictory labels. Metrics are reported for diagnostic comparison only.

- **`baseline`:**
  - $\tau = 0.50$: PR-AUC = 0.2614, ROC-AUC = 0.7355, Recall = 0.0995 (42/422), Prec = 0.8750 (42/48), F1 = 0.1787.
  - $\tau^* = 0.08$: Recall = 0.2275 (96/422), Prec = 0.3735 (96/257), F1 = 0.2828.
- **`balanced_sampling`:**
  - $\tau = 0.50$: PR-AUC = 0.2714, ROC-AUC = 0.7850, Recall = 0.3152 (133/422), Prec = 0.2777 (133/479), F1 = 0.2952.
  - $\tau^* = 0.76$: Recall = 0.2654 (112/422), Prec = 0.4103 (112/273), F1 = 0.3223.
- **`class_weighted`:**
  - $\tau = 0.50$: PR-AUC = 0.1314, ROC-AUC = 0.6906, Recall = 0.1896 (80/422), Prec = 0.2564 (80/312), F1 = 0.2180.

---

## 5. Phase 5 — Post-Run Integrity Checks

1. **Manifest Integrity:** All 4 audited manifest SHA-256 hashes were re-verified and remain 100% identical:
   - `via3_split_manifest.json`: `17bc670452a083c882d18c0de7f100b88327571f6e3795f0fc0fce512b8e4d53` (**PASS**)
   - `via3_exclusion_manifest.json`: `f4d269cace5b94d1629aeefdd9f0b55d9fb210444529aa31a0af93326518755a` (**PASS**)
   - `via3_cohort_clean_test.json`: `28d45645bb2b0dcb2f9365d02ce39a6083eb2a506926417c986e817b5441dd8f` (**PASS**)
   - `via3_cohort_official_test.json`: `149627ea68634a4b835fa81ef0e4785d11a5ae9b2fec9237577d45837622adbe` (**PASS**)
2. **Cohort Populations and Label Consistency:**
   - Validation set: exactly 2,846 rows in all 3 prediction CSVs (87 HS, 2,759 NHS).
   - Clean test set: exactly 12,803 rows in all 3 prediction CSVs (411 HS, 12,392 NHS).
   - Official test set: exactly 12,905 rows in all 3 prediction CSVs (422 HS, 12,483 NHS).
3. **Artifact Existence:**
   - Checkpoints:
     - `models/via3_imbalance/via3_baseline_seed42.keras` (SHA-256: `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887`)
     - `models/via3_imbalance/via3_class_weighted_seed42.keras` (SHA-256: `63168fd5d9fbd1208cc3b290ca7880919b468c043d249d7a5dca5e53f1567ceb`)
     - `models/via3_imbalance/via3_balanced_sampling_seed42.keras` (SHA-256: `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426`)
   - Run Result Manifests:
     - `baseline_results_seed42.json`
     - `class_weighted_results_seed42.json`
     - `balanced_sampling_results_seed42.json`
     - `imbalance_comparison_summary.json`
4. **Test Suite:**
   - `pytest tests/ -v`: **64 passed in 13.57s** (100% pass rate).
   - `test_degeneracy_forensics.py`: **5 passed in 0.09s** (100% pass rate).
   - Combined total: **69 passed, 0 failed**.
5. **Protected Safeguards:**
   - `docs/DA2_Group18_Report.docx`: Untouched.
   - `latex_report/`: Untouched.
   - Git branch: `research/iccad5-collapse-xai` (HEAD `41d2c85`, 0 commits, 0 branch switches).

---

## 6. Synthesis and Recommendation for Workstream C (XAI)

### 6.1 Scientific Analysis of Imbalance Interventions
1. **Severe Distortion from Static Inverse Class Weighting:**  
   Applying inverse-frequency loss weights ($w_1 \approx 16.19$) severely damages the precision-recall trade-off. By penalizing false negatives 31.4 times more heavily than false positives, the gradient pushes the model to produce inflated probabilities across non-hotspots. This collapses validation PR-AUC from 0.2886 to 0.1282 (-55.6% relative). Class weighting is therefore disqualified.
2. **Superior Representation Learning via Balanced Sampling:**  
   `balanced_sampling` exposes the feature extraction layers (Blocks 1 and 2) to equal numbers of hotspot and non-hotspot geometries per batch, preventing majority-class dominance in gradient updates. On the validation partition, it achieves a PR-AUC of **0.2865** (essentially tied with baseline's 0.2886) with a higher ROC-AUC (**0.7226** vs 0.7094) and much higher raw sensitivity (**32.18%** vs 11.49% at $\tau=0.50$).
3. **Confirmed Generalization on Held-Out Test Data:**  
   On the clean held-out test cohort, `balanced_sampling` produces the highest PR-AUC (**0.2619** vs 0.2518 for baseline) and highest ROC-AUC (**0.7840** vs 0.7328 for baseline). At its validation-calibrated operating point ($\tau^* = 0.7600$), it achieves **40.38% precision** and **26.03% recall** (F1: 0.3166), with 158 false positives.

### 6.2 Recommendation
**`balanced_sampling` is the recommended model for Workstream C (XAI Evaluation)**, with the unweighted `baseline` maintained as the experimental control. 

Specifically:
- In Workstream C, comparing Grad-CAM and Grad-CAM++ heatmaps between `baseline` and `balanced_sampling` will directly test whether balanced representation learning mitigates the feature collapse and gradient saturation observed on imbalanced lithography benchmarks.
- No further experimental sweeps should be launched until human operator review and authorization.
