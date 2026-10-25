# Workstream E4.1 — Multi-Seed XAI Robustness Preflight and Cohort Freeze

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3`  
**Evaluated Scope:** Multi-Seed Explainability (XAI) Robustness Evaluation Planning  
**Target Models:** Baseline (Unweighted) vs. Balanced Sampling across Seeds `42`, `100`, and `2024` (6 checkpoints total)  
**Target Cohort:** Fixed 50-layout diagnostic cohort from [`xai_sample_cohort_manifest.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json)  
**Preflight Audit Date:** 2026-10-23  
**Auditor:** Independent Automated Research Integrity Auditor  

---

## 1. Executive Verdict

### **Verdict: READY FOR AUTHORIZATION**

The multi-seed explainability robustness evaluation is mathematically specified, resource-safe, and fully verified across all required code, model, and data dependencies. No blocking data defects exist.

- **Diagnostic Cohort Verified & Frozen:** All 50 physical layout images in [`results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json) exist on disk under `/home/peskybird/Projects/data/openhsd/via-3/test/`. The manifest hash is frozen at SHA-256 `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`.
- **Six Model Checkpoints Verified:** All 6 target custom CNN models (Baseline and Balanced Sampling across Seeds 42, 100, 2024) exist in [`models/via3_multiseed/`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/), share the identical 12,873-parameter architecture, and possess verified SHA-256 checksums.
- **Prediction & Threshold Provenance Complete:** Operating thresholds are frozen strictly from validation PR-curve optimization ($\tau^* \in [0.01, 0.08]$ for Baseline, $\tau^* \in [0.53, 0.76]$ for Balanced Sampling). All 50 cohort layouts match prediction records in every clean test CSV.
- **Methodology Fully Reconciled:** The attribution engine targets the pre-sigmoid logit $+z_{\text{HS}}$ (eliminating gradient saturation), convolutional layer `conv2d_5` ($15 \times 15 \times 12$), sliding-window occlusion ($32 \times 32$, stride $16 \times 16$), and local Gaussian stability ($\sigma = 0.05, T=3$).
- **Output Isolation Guaranteed:** New execution will write strictly to `results/via3_experiments/xai_multiseed/`, leaving the audited Seed 42 reference (`results/via3_experiments/xai_comparison/`), Workstream D drafts, and coursework files completely untouched.
- **Hard Safety Adherence:** **Zero GPU computation, attribution mapping, model inference, or file mutations were performed during this preflight.**

---

## 2. Scope and Verification Limitations

1. **Inspection and Planning Only:** This preflight audited the codebase, verified checkpoint architectures and hashes, resolved file paths, and established the statistical analysis plan. No attributions were generated.
2. **Diagnostic Cohort Scope:** The 50-layout diagnostic cohort is an error-stratified diagnostic benchmark (oversampling rare discordances such as `balanced_won` and `baseline_won`), not an i.i.d. population sample. Inferences apply strictly to this diagnostic cohort.
3. **Small Sample Size ($N=3$ Seeds):** Evaluating explanations across 3 random seeds provides an initial check of initialization sensitivity, not a broad characterization of seed population variance. Formal inferential hypothesis testing across seeds has low statistical power.

---

## 3. Cohort Identity and Verification (Audit Section 3)

The diagnostic cohort manifest was inspected at:  
[`results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json)

- **File Size:** 9,719 bytes
- **SHA-256 Checksum:** `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`
- **Total Diagnostic Layouts:** Exactly 50 unique layout filenames.
- **Physical Image Existence:** All 50 layout images were resolved and verified on disk in `/home/peskybird/Projects/data/openhsd/via-3/test/` (50 found, 0 missing).

### 3.1 Stratum Composition

| Cohort Stratum | Ground Truth | Target Characteristic | Sample Count | Representative Filename |
| :--- | :---: | :--- | :---: | :--- |
| `shared_tp` | Hotspot ($1$) | Concordant true positive detection by both models | 10 | `Hvia87665_mb_mb_lccout.oas.gds.png` |
| `shared_fn` | Hotspot ($1$) | Concordant false negative miss by both models | 10 | `Hvia97282_mb_mb_lccout.oas.gds.png` |
| `balanced_won` | Hotspot ($1$) | Discordant: captured by Balanced, missed by Baseline | 10 | `Hvia64627_mb_mb_lccout.oas.gds.png` |
| `baseline_won` | Hotspot ($1$) | Discordant: captured by Baseline, missed by Balanced | 10 | `Hvia80436_mb_mb_lccout.oas.gds.png` |
| `shared_tn` | Non-Hotspot ($0$) | Concordant true negative clean layout | 5 | `Nvia44933_mb_mb_lccout.oas.gds.png` |
| `shared_fp` | Non-Hotspot ($0$) | Concordant false alarm by both models | 5 | `Nvia25674_mb_mb_lccout.oas.gds.png` |
| **Total Cohort** | **40 HS / 10 NHS** | **Stratified Error-Mode Diagnostic Sample** | **50** | — |

*Invariant Rule:* This exact 50-layout cohort is frozen. No resampling, replacement, or label modification is permitted.

---

## 4. Six-Checkpoint Provenance Matrix (Audit Section 4)

All six checkpoints located in [`models/via3_multiseed/`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/) were inspected for architecture, parameter counts, and checksums:

| Checkpoint Identifier | Strategy | Seed | File Path | File Size | SHA-256 Checksum | Trainable Params | Origin / Status |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `base_42` | Baseline | 42 | `models/via3_multiseed/via3_baseline_seed42.keras` | 230,689 B | `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887` | 12,873 | Audited Reference (Verified against `models/via3_imbalance/`) |
| `base_100` | Baseline | 100 | `models/via3_multiseed/via3_baseline_seed100.keras` | 230,689 B | `f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455` | 12,873 | Trained in Workstream E2 (Epoch 15 best val PR-AUC) |
| `base_2024` | Baseline | 2024 | `models/via3_multiseed/via3_baseline_seed2024.keras` | 230,689 B | `9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d` | 12,873 | Trained in Workstream E2 (Epoch 15 best val PR-AUC) |
| `bal_42` | Balanced | 42 | `models/via3_multiseed/via3_balanced_sampling_seed42.keras` | 230,689 B | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` | 12,873 | Audited Reference (Verified against `models/via3_imbalance/`) |
| `bal_100` | Balanced | 100 | `models/via3_multiseed/via3_balanced_sampling_seed100.keras` | 230,689 B | `c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815` | 12,873 | Trained in Workstream E2 (Epoch 4 best val PR-AUC) |
| `bal_2024` | Balanced | 2024 | `models/via3_multiseed/via3_balanced_sampling_seed2024.keras` | 230,689 B | `297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9` | 12,873 | Trained in Workstream E2 (Epoch 12 best val PR-AUC) |

*Alternative Baseline Reference Checkpoint:*  
In Workstream C, the comparative XAI script loaded `models/via3/via3_baseline_seed42.keras` (SHA-256: `b9979e8dbac5d31d8eac8bcd4156874ac7c2fcfa055200232c939922775ef4c3`). In Workstream E2/E3, all models were unified under the audited `models/via3_multiseed/` directory, using `via3_baseline_seed42.keras` (SHA-256: `97a9cb8b...`) which corresponds directly to the Workstream B2 imbalance suite and shares the exact training pipeline of seeds 100 and 2024. Both checkpoints remain intact and available.

---

## 5. Prediction and Threshold Provenance Matrix (Audit Section 5)

All 6 models have complete prediction records across validation, clean test, and official test splits. All 50 cohort layouts were cross-referenced against the clean test prediction CSVs:

| Model | Calibrated Threshold $\tau^*$ | Threshold Objective | Clean Test PR-AUC | Clean Test F1 | Clean Test Recall | Cohort HS Predicted ($/40$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Seed 42** | $\tau^* = 0.08$ | Validation `max_f1` | 0.2518 | 0.2753 | 0.2214 | 20 / 40 |
| **Baseline Seed 100** | $\tau^* = 0.07$ | Validation `max_f1` | 0.2983 | 0.3146 | 0.2993 | 21 / 40 |
| **Baseline Seed 2024** | $\tau^* = 0.01$ | Validation `max_f1` | 0.2745 | 0.2904 | 0.1946 | 13 / 40 |
| **Balanced Seed 42** | $\tau^* = 0.76$ | Validation `max_f1` | 0.2619 | 0.3166 | 0.2603 | 20 / 40 |
| **Balanced Seed 100** | $\tau^* = 0.74$ | Validation `max_f1` | 0.2168 | 0.2665 | 0.2798 | 21 / 40 |
| **Balanced Seed 2024** | $\tau^* = 0.53$ | Validation `max_f1` | 0.3120 | 0.3438 | 0.2798 | 20 / 40 |

### Invariants Verified:
1. **No Data Leakage:** All operating thresholds were selected exclusively from validation sets (`val_predictions.csv`, 2,846 samples). Test data was not accessed during threshold selection.
2. **Attribution Target:** In [`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py#L74-L94), the attribution target score for Hotspot ($1$) is explicitly defined as the pre-sigmoid logit $+z_{\text{HS}}$:
   $$\text{Score}(\text{HS}) = z_{\text{HS}} = \mathbf{w}_{\text{dense\_1}}^T \mathbf{a}_{\text{dense}} + b$$
   This target is strictly identical across all 6 models, completely bypassing sigmoid saturation.
3. **Threshold Orthogonality:** Operating thresholds are applied to binary decision outputs, but do **not** modify or weight continuous attribution maps or pre-sigmoid gradients.

---

## 6. Verified Attribution and Perturbation Methodology (Audit Section 6)

The methodology implemented in [`scripts/evaluate_via3_xai.py`](file:///home/peskybird/Projects/Ai_Ml_DA/scripts/evaluate_via3_xai.py), [`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py), [`src/faithfulness.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/faithfulness.py), and [`src/occlusion.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/occlusion.py) was verified:

### 6.1 Methods and Configuration

| Method | Source Implementation | Mathematical Definition | Target Layer & Resolution | Normalization |
| :--- | :--- | :--- | :---: | :---: |
| **Grad-CAM** | `src/xai_engine.py:explain_gradcam` | $\text{ReLU}\left(\sum_k \alpha_k A^k\right), \; \alpha_k = \frac{1}{Z}\sum_{i,j} \frac{\partial z_{\text{HS}}}{\partial A_{i,j}^k}$ | `conv2d_5` ($15 \times 15 \times 12$) | Min-max scaled to $[0, 1]$, resized to $(224, 224)$ |
| **Grad-CAM++** | `src/xai_engine.py:explain_gradcam_plus_plus` | Second/third-order gradient weighting for co-occurring features | `conv2d_5` ($15 \times 15 \times 12$) | Min-max scaled to $[0, 1]$, resized to $(224, 224)$ |
| **LayerCAM** | `src/xai_engine.py:explain_layercam` | Spatially localized pixel-wise positive gradient weighting: $\text{ReLU}(w_{i,j}^k A_{i,j}^k)$ | `conv2d_5` ($15 \times 15 \times 12$) | Min-max scaled to $[0, 1]$, resized to $(224, 224)$ |
| **Gradient Saliency** | `src/xai_engine.py:explain_gradient_saliency` | Input gradient magnitude: $\max_c |\nabla_{\mathbf{x}} z_{\text{HS}}|$ | Input image $\mathbf{x}$ ($224 \times 224 \times 3$) | Min-max scaled to $[0, 1]$ |
| **Occlusion** | `src/occlusion.py:OcclusionSensitivity` | Perturbation drop: $\Delta z_{\text{HS}} = z_{\text{orig}} - z_{\text{masked}}$ with sliding patch | Window $32 \times 32$, Stride $16 \times 16$, fill $0.0$ | Normalized by window coverage count, scaled to $[0, 1]$ |
| **Random Baseline** | `src/faithfulness.py:compute_deletion_insertion_curves` | Fixed pseudo-random pixel permutation (`seed=42`) | Input resolution ($224 \times 224$) | Uniform random ordering |

### 6.2 Faithfulness Perturbation Protocol
- **Number of Steps:** Exactly 10 steps ($10\%$ of pixels per step = $5,017$ pixels/step out of $50,176$).
- **Deletion Protocol:** Starts at original layout image $x_{\text{orig}}$. At each step, top attribution pixels are set to bare substrate background ($0.0$).
- **Insertion Protocol:** Starts at blank substrate canvas $\mathbf{0}^{224 \times 224 \times 3}$. At each step, top attribution pixels from $x_{\text{orig}}$ are restored.
- **AUC Calculation:** Trapezoidal numerical integration over normalized perturbation intervals $[0, 1]$ with $\Delta x = 0.10$.

### 6.3 Stability Perturbation Protocol
- **Noise Type:** Additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ with $\sigma = 0.05$.
- **Tensor Clipping:** Hard-clipped to valid image bounds: `np.clip(img + noise, 0.0, 1.0)`.
- **Number of Trials:** Exactly $T = 3$ trials per layout image.
- **Metric:** Mean pairwise Spearman rank correlation $\bar{\rho}$ between flattened unperturbed heatmap and perturbed heatmaps.

### 6.4 Established Methodological Qualifications (Preserved from Audit)
1. **Attribution Faithfulness vs Causal Sufficiency:** Pixel insertion/deletion measures model sensitivity under zero-fill masking, not formal Pearlian causal interventions.
2. **Out-of-Distribution (OOD) Blank Canvas:** An all-zero layout canvas yields high sigmoid probabilities ($P \approx 0.94 - 0.99$) due to convolutional biases without negative feature activations, causing U-shaped deletion tails.
3. **Random Attribution as Empirical Reference:** Random permutation is an empirical unguided control, not a theoretical universal lower bound.
4. **Spatial Contrast vs Defect Localization:** Spatial heatmap standard deviation measures contrast and dynamic range, not semantic defect boundary alignment.

---

## 7. Unresolved Methodological Discrepancies

The preflight audit identified two minor implementation details from Workstream C that require explicit harmonization in Workstream E4:
1. **Random Baseline Determinism:** In `src/faithfulness.py:81`, `np.random.default_rng(seed=42)` is re-instantiated within the loop for `method == 'random'`. This applies the exact same pseudo-random pixel permutation to every layout image and model. In E4, this will be maintained for deterministic comparability with the Seed 42 benchmark, but documented explicitly as a fixed permutation control.
2. **Baseline Seed 42 Checkpoint Origin:** Workstream C evaluated `models/via3/via3_baseline_seed42.keras` (hash `b9979e8d...`). Workstream E2 evaluated `models/via3_multiseed/via3_baseline_seed42.keras` (hash `97a9cb8b...`), which matches the imbalance benchmark suite. In E4, the primary comparison will evaluate the 6 unified models in `models/via3_multiseed/`, while retaining the original Workstream C records as a verified reference.

---

## 8. Proposed Six-Checkpoint Evaluation Matrix (Audit Section 7)

The proposed Workstream E4 evaluation matrix encompasses:
- **6 Checkpoints:** 3 Seeds ($42, 100, 2024$) $\times$ 2 Strategies (Baseline, Balanced Sampling).
- **50 Fixed Diagnostic Layouts:** Matched across all 6 models.
- **6 Attribution Methods:** Grad-CAM, Grad-CAM++, LayerCAM, Saliency, Occlusion, Random.
- **Evaluation Points:** $6 \text{ models} \times 50 \text{ layouts} \times 6 \text{ methods} = \mathbf{1,800} \text{ total attribution evaluations}$.

| Model Key | Strategy | Seed | Diagnostic Layouts | Methods Evaluated | Perturbation Steps | Stability Trials | Checkpoint Path |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| `base_42` | Baseline | 42 | 50 (Frozen) | 6 methods | 10 steps | $T=3$ | `models/via3_multiseed/via3_baseline_seed42.keras` |
| `base_100` | Baseline | 100 | 50 (Frozen) | 6 methods | 10 steps | $T=3$ | `models/via3_multiseed/via3_baseline_seed100.keras` |
| `base_2024`| Baseline | 2024 | 50 (Frozen) | 6 methods | 10 steps | $T=3$ | `models/via3_multiseed/via3_baseline_seed2024.keras` |
| `bal_42` | Balanced | 42 | 50 (Frozen) | 6 methods | 10 steps | $T=3$ | `models/via3_multiseed/via3_balanced_sampling_seed42.keras` |
| `bal_100` | Balanced | 100 | 50 (Frozen) | 6 methods | 10 steps | $T=3$ | `models/via3_multiseed/via3_balanced_sampling_seed100.keras` |
| `bal_2024` | Balanced | 2024 | 50 (Frozen) | 6 methods | 10 steps | $T=3$ | `models/via3_multiseed/via3_balanced_sampling_seed2024.keras` |

---

## 9. Statistical Analysis Plan (Audit Section 8)

1. **Within-Layout Pairing:**  
   For every physical layout image $i \in \{1, \dots, 50\}$, the paired difference between Balanced Sampling and Baseline under seed $S \in \{42, 100, 2024\}$ is computed as:
   $$\Delta_S(i) = \text{Metric}_{\text{bal}, S}(i) - \text{Metric}_{\text{base}, S}(i)$$
2. **Primary Endpoint:**  
   Grad-CAM Insertion AUC and Deletion AUC paired difference within the 50 diagnostic layouts.
3. **Secondary Endpoints:**  
   Grad-CAM Stability $\rho$; secondary methods (Grad-CAM++, LayerCAM, Saliency, Occlusion); Spatial Contrast (heatmap standard deviation); Flat-CAM collapse rate.
4. **Multiple Comparisons Strategy:**  
   Apply Holm-Bonferroni step-down correction within each seed's 17-test family to control family-wise error rate at $\alpha = 0.05$.
5. **Seed Variance vs. Layout Variance Decomposition:**  
   Compute mean paired difference for each seed:
   $$\bar{\Delta}(S) = \frac{1}{50}\sum_{i=1}^{50} \Delta_S(i)$$
   Evaluate whether the direction of advantage is consistent ($\bar{\Delta}(S) > 0$) across all three independent seeds.
6. **Explicit Sample Size Warning:**  
   $N=3$ training seeds do not permit precise estimation of the population distribution of training seed variance. The 50 layout-level evaluations must not be pooled as independent degrees of freedom for seed-level claims.

---

## 10. Output Isolation and Resource Assessment (Audit Section 9)

### 10.1 Isolated Output Directory Structure
To strictly avoid collisions with prior workstreams, all outputs for Workstream E4 will be quarantined under `results/via3_experiments/xai_multiseed/`:

```
results/via3_experiments/xai_multiseed/
├── cohort/
│   └── xai_sample_cohort_manifest.json (frozen copy)
├── baseline/
│   ├── seed_42/
│   │   ├── faithfulness_metrics.csv
│   │   ├── stability_metrics.csv
│   │   ├── collapse_diagnostics.csv
│   │   └── xai_summary_seed42.json
│   ├── seed_100/
│   │   └── ...
│   └── seed_2024/
│       └── ...
├── balanced_sampling/
│   ├── seed_42/
│   │   └── ...
│   ├── seed_100/
│   │   └── ...
│   └── seed_2024/
│       └── ...
└── multiseed_xai_comparison_summary.json
```

### 10.2 Overwrite and Resume Protection
- Pre-execution checks will verify whether an output directory already contains valid, non-empty CSV files.
- If present, training/evaluation will skip existing runs to allow safe resume on interruption.

### 10.3 Resource & Runtime Assessment
- **GPU Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- **Forward Passes per Model:** ~1,200 forward passes per model on $224 \times 224 \times 3$ tensors.
- **Estimated Runtime:** ~50 seconds per model $\times$ 6 models $\approx$ **5.0 to 7.0 minutes total**.
- **Peak VRAM:** Forward pass batch size $\le 64$ patches requires $< 1.5$ GB VRAM (well within the 6.0 GB total).
- **Execution Mode:** Strictly sequential (one model at a time).

---

## 11. Required Remediation Ranked by Severity

| Priority | Issue / Risk | Remediation in E4 Runner Implementation |
| :---: | :--- | :--- |
| **HIGH** | Avoid overwriting Seed 42 Workstream C artifacts. | Quarantine all multi-seed XAI outputs to `results/via3_experiments/xai_multiseed/`. |
| **MEDIUM** | Cohort drift across seeds. | Explicitly load the frozen `xai_sample_cohort_manifest.json` by path and verify its SHA-256 hash before running any model. |
| **MEDIUM** | Pseudo-replication in hypothesis testing. | Structure statistical tests so that layout-level paired tests are reported separately from across-seed variance summaries. |
| **LOW** | Logging overhead during Occlusion sliding window. | Silence verbose batch printouts to prevent oversized log files. |

---

## 12. Actions Authorized Only After Explicit Human Approval

Upon explicit human authorization, the execution phase (Workstream E4.2) will:
1. Implement the isolated multi-seed XAI runner script (`scripts/run_via3_xai_multiseed.py`).
2. Add focused unit tests verifying output isolation, cohort hash validation, and safe resume.
3. Execute sequential XAI evaluation across the 6 checkpoints on the frozen 50 diagnostic layouts (~6 minutes total GPU time).
4. Save the 18 per-seed XAI metric CSVs (`faithfulness_metrics.csv`, `stability_metrics.csv`, `collapse_diagnostics.csv`) and JSON summaries.
5. Compile the comprehensive Workstream E4.2 Multi-Seed XAI Robustness Report.
