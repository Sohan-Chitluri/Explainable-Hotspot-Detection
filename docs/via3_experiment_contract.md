# Shared Experiment Contract: OpenHSDBench Via-3 Hotspot Pipeline

**Contract Version:** `1.0.0`  
**Target Architecture:** Custom CNN (18,521 parameters; `src/model.py:build_custom_cnn`)  
**Hardware Profile:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  
**Primary Integration Branch:** `research/iccad5-collapse-xai`  
**Data Partition Root:** `/home/peskybird/Projects/data/openhsd/via-3`  
**Manifest Audit Directory:** `results/dataset_audit`  

---

## 1. Dataset Identity and Manifests

All experiments on OpenHSDBench `via-3` must consume the exact audited partitions defined by SHA-256 verified manifests.

### 1.1 Manifest Hashes & Cohort Specifications

| Manifest Name | SHA-256 Checksum | Purpose / Description |
| :--- | :--- | :--- |
| `via3_split_manifest.json` | `17bc670452a083c882d18c0de7f100b88327571f6e3795f0fc0fce512b8e4d53` | Group-aware 85/15 train/val partition (0 pattern overlap) |
| `via3_exclusion_manifest.json` | `f4d269cace5b94d1629aeefdd9f0b55d9fb210444529aa31a0af93326518755a` | 102 contaminated test samples (98 train duplicates + 4 contradictory labels) |
| `via3_cohort_clean_test.json` | `28d45645bb2b0dcb2f9365d02ce39a6083eb2a506926417c986e817b5441dd8f` | Primary held-out clean test cohort ($N=12,803$) |
| `via3_cohort_official_test.json` | `149627ea68634a4b835fa81ef0e4785d11a5ae9b2fec9237577d45837622adbe` | Diagnostic untouched official test cohort ($N=12,905$) |

### 1.2 Audited Cohort Populations

- **Training Partition:** 16,158 samples (499 Hotspots, 15,659 Non-Hotspots; ratio 31.4:1).
- **Validation Partition:** 2,846 samples (87 Hotspots, 2,759 Non-Hotspots; ratio 31.7:1).
- **Clean Held-Out Test Cohort:** 12,803 samples (411 Hotspots, 12,392 Non-Hotspots; ratio 30.1:1).
- **Official Test Set (Diagnostic):** 12,905 samples (422 Hotspots, 12,483 Non-Hotspots).

---

## 2. Label Semantics & Metric Definitions

### 2.1 Label Mapping
- **`Label 1`**: Hotspot (`HS`) — **Positive Class**
- **`Label 0`**: Non-Hotspot (`NHS`) — **Negative Class**

> [!IMPORTANT]
> This reverses the legacy ICCAD-12 alphabetical directory indexing where `train_hs` was class 0. In this contract, `1` is strictly the positive class for all metrics, loss weighting, and attribution scoring.

### 2.2 Canonical Metric Formulas
Given predicted probabilities $\hat{p} \in [0, 1]$ and threshold $\tau \in (0, 1)$:
- $\text{Predicted Hotspot} \iff \hat{p} \ge \tau$
- **True Positives ($TP$):** Actual HS ($y=1$), Predicted HS ($\hat{y}=1$)
- **False Negatives ($FN$):** Actual HS ($y=1$), Predicted NHS ($\hat{y}=0$)
- **False Positives ($FP$):** Actual NHS ($y=0$), Predicted HS ($\hat{y}=1$)
- **True Negatives ($TN$):** Actual NHS ($y=0$), Predicted NHS ($\hat{y}=0$)

- **Hotspot Recall (Sensitivity):** $\text{Recall}_{\text{HS}} = \frac{TP}{TP + FN}$
- **Hotspot Specificity:** $\text{Specificity}_{\text{HS}} = \frac{TN}{TN + FP}$
- **Hotspot Precision:** $\text{Precision}_{\text{HS}} = \frac{TP}{TP + FP}$
- **Hotspot F1-Score:** $\text{F1}_{\text{HS}} = \frac{2 \cdot \text{Precision}_{\text{HS}} \cdot \text{Recall}_{\text{HS}}}{\text{Precision}_{\text{HS}} + \text{Recall}_{\text{HS}}}$
- **Balanced Accuracy:** $\text{BalAcc} = \frac{\text{Recall}_{\text{HS}} + \text{Specificity}_{\text{HS}}}{2}$
- **Primary Metric — PR-AUC (Average Precision):** Area under the Precision-Recall curve computed via exact trapezoidal steps over sorted probabilities.
- **ROC-AUC:** Area under the ROC curve computed via the Wilcoxon-Mann-Whitney rank-sum statistic.

---

## 3. Checkpoint Selection & Early Stopping

1. **Monitored Metric:** Validation PR-AUC evaluated on the full validation partition at the conclusion of every epoch.
2. **Tie-Breaking Rule:** In the event of a PR-AUC tie ($|\Delta| \le 10^{-4}$), validation Hotspot F1 score breaks the tie.
3. **Weight Restoration:** At training completion, the callback strictly restores the weights from the highest-scoring validation epoch before any final evaluation.
4. **Never Touch Test Set:** Neither the clean held-out test cohort nor the official test set is ever evaluated during training or checkpoint selection.

---

## 4. Decision Threshold Calibration Protocol

1. **Default Threshold:** $\tau = 0.50$ is reported as the standard baseline reference.
2. **Operational Threshold Tuning ($\tau^*$):**
   - Validation set predicted probabilities are generated **once** from the selected checkpoint.
   - A grid sweep over $\tau \in [0.01, 0.99]$ evaluates validation precision, recall, and F1.
   - The operational threshold $\tau^*$ is selected based on a predeclared objective (e.g., maximize validation F1, or maximize recall subject to precision $\ge 0.50$).
   - $\tau^*$ is completely frozen before touching the held-out test cohort.
3. **Strict Ban on Test Tuning:** Threshold selection using test set metrics is strictly prohibited.

---

## 5. Model Architecture, Preprocessing, and Training Budget

- **Input Preprocessing:** Nearest-neighbor resizing to $(224, 224)$, normalized to float32 $[0.0, 1.0]$, 3 identical channels.
- **Model Topology:** Custom CNN with 18,521 total parameters (18,473 trainable):
  - Block 1: Conv(12, 3x3, ELU) -> Conv(12, 3x3, ELU) -> Conv(12, 3x3) -> BatchNorm -> ELU -> MaxPool(2x2)
  - Mid: MaxPool(5x5)
  - Block 2: Conv(12, 3x3, ELU) -> Conv(12, 3x3, ELU) -> Conv(12, 3x3) -> BatchNorm -> ELU -> MaxPool(2x2)
  - Head: Flatten -> Dropout(0.30) -> Dense(10, ReLU) -> Dense(1, Sigmoid)
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-7}$), initial learning rate $\eta = 10^{-3}$.
- **Batch Size:** 32.
- **Budget:** 15 epochs standard (2 epochs for smoke verification mode).
- **Reproducibility Seeds:** Canonical seed 42, with multi-seed variance orchestration across seeds {42, 100, 2024}.

---

## 6. Pre-Training Verification Gate

Full model training, expensive GPU sweeps, or multi-seed runs MUST NOT commence until the pre-training verification gate explicitly passes.
- **Status Check:** Evaluated via `src.experiment_contract.check_pretraining_gate()`.
- **Default State:** Gated (**NO-GO** for training until approved).
- **Allowed Pre-Gate Actions:** Manifest integrity audits, code tests, dry-run simulations, fixture reporting tests, and schema validation.
