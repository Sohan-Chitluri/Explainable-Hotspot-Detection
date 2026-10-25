# Workstream E1 — Multi-Seed Robustness: Experiment Contract and Preflight Audit

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3`  
**Target Seeds:** Reference Seed `42` (audited, read-only), Exploration Seeds `100` and `2024`  
**Audit Date:** 2026-10-22  
**Auditor:** Independent Automated Research Integrity Auditor  

---

## 1. Executive Verdict

### **Verdict: READY WITH BLOCKERS**

The dataset, split manifests, model architecture, preprocessing pipeline, and GPU execution environment are fully verified, stable, and ready for multi-seed robustness training. The reference Seed 42 artifacts are preserved and verified by SHA-256 checksums.

However, execution must **NOT** be launched until two concrete code blockers in the existing orchestration script [`scripts/run_via3_multiseed.py`](file:///home/peskybird/Projects/Ai_Ml_DA/scripts/run_via3_multiseed.py) are remediated:
1. **Strategy Limitation in Multi-Seed Script (`BLOCKER-01`):** In `scripts/run_via3_multiseed.py` line 138, the execution loop unconditionally invokes `run_baseline`, completely ignoring the candidate `balanced_sampling` strategy and the `class_weighted` control. The script cannot currently execute the primary winning model (`balanced_sampling`) across seeds.
2. **Missing Reference-Seed Immutability & Overwrite Guards (`BLOCKER-02`):** In `scripts/run_via3_multiseed.py`, passing default seeds `[42, 100, 2024]` attempts to retrain Seed 42 from scratch rather than importing the audited Seed 42 reference artifacts as read-only inputs. Furthermore, the script lacks file-existence detection to prevent silent overwrites of completed seed runs.

Once these two code remediations are authorized and applied in Workstream E2, the repository will transition immediately to `READY FOR TRAINING`. **No GPU training, model retraining, or evaluation was executed during this audit.**

---

## 2. Repository State and Working-Tree Summary

- **Current Git Branch:** `research/iccad5-collapse-xai`
- **Current Git HEAD:** `41d2c854604a576f01b3fddb72c2c09d43fbcb8e`
- **Working Tree State:** Clean of untracked empirical mutations. All primary files match the completed Workstream D release-gate state.
- **Protected Coursework Paths:**
  - `docs/DA2_Group18_Report.docx` — **Untouched & Preserved**
  - `latex_report/` — **Untouched & Preserved**
- **Canonical Workstream D Drafts:**
  - `draft_reports/via3/draft_via3_experimental_report.md` (SHA-256: `bf2916e49ec27f6687fdb48a0ba51bfcdcdb608d9ca5e11924dc761960e1f89e`) — **Preserved**
  - `draft_reports/via3/draft_via3_table_summary.tex` (SHA-256: `9d8369dc9f31f3c39376738fb91b5758017a1993cf26050d92fe25816d5495d4`) — **Preserved**

### 2.1 Verified Reference Seed 42 Artifacts
| Strategy / Checkpoint | File Path | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :---: |
| **Baseline (Seed 42)** | `models/via3/via3_baseline_seed42.keras` | 230,689 B | `b9979e8dbac5d31d8eac8bcd4156874ac7c2fcfa055200232c939922775ef4c3` |
| **Balanced Sampling (Seed 42)** | `models/via3_imbalance/via3_balanced_sampling_seed42.keras` | 230,689 B | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` |
| **Class-Weighted (Seed 42)** | `models/via3_imbalance/via3_class_weighted_seed42.keras` | 230,689 B | `63168fd5d9fbd1208cc3b290ca7880919b468c043d249d7a5dca5e53f1567ceb` |
| **B2 Imbalance Summary** | `results/via3_experiments/imbalance/imbalance_comparison_summary.json` | 14,222 B | `7040963af5a7810fd1a16a5b12d873456ff6f265aec984efe27021793fba3dfb` |
| **XAI Comparison Summary** | `results/via3_experiments/xai_comparison/xai_comparison_summary.json` | 4,482 B | `af709518eda3d152abf02e622d288feb69195886d6bdf9d9dde00274d32a0279` |
| **50-Layout Diagnostic Manifest** | `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json` | 9,719 B | `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb` |

### 2.2 Audited Dataset Manifests (`results/dataset_audit/`)
All 16 audited dataset manifests are intact. Key partition manifests:
- `via3_split_manifest.json` (SHA-256: `17bc670452a083c882d18c0de7f100b88327571f6e3795f0fc0fce512b8e4d53`)
- `via3_cohort_clean_test.json` (SHA-256: `28d45645bb2b0dcb2f9365d02ce39a6083eb2a506926417c986e817b5441dd8f`)
- `via3_cohort_official_test.json` (SHA-256: `149627ea68634a4b835fa81ef0e4785d11a5ae9b2fec9237577d45837622adbe`)
- `via3_exclusion_manifest.json` (SHA-256: `f4d269cace5b94d1629aeefdd9f0b55d9fb210444529aa31a0af93326518755a`)

---

## 3. Verified Experiment Configuration and Protocol Trace

The experimental protocol that generated the audited Seed 42 benchmark was traced through `src/experiment_contract.py`, `src/openhsd_data.py`, `src/model.py`, and `scripts/run_via3_imbalance_experiments.py`:

| Parameter / Protocol Element | Specification | Verification Source | Classification |
| :--- | :--- | :--- | :---: |
| **Dataset Splits** | Train: 22,763 (729 HS / 22,034 NHS)<br>Val: 2,845 (91 HS / 2,754 NHS)<br>Clean Test: 12,803 (411 HS / 12,392 NHS)<br>Official Test: 12,905 (411 HS / 12,494 NHS) | `src/experiment_contract.py:AUDITED_COUNTS` | **Verified from implementation & artifacts** |
| **Label Mapping** | `1 = Hotspot (HS)`, `0 = Non-Hotspot (NHS)` | `src/openhsd_data.py:load_via3_manifest_datasets` | **Verified from implementation** |
| **Image Preprocessing** | Grayscale resize to $(224, 224)$ via nearest neighbor, normalized to $[0, 1]$, repeated across 3 channels. | `src/openhsd_data.py:_load_and_preprocess_image` | **Verified from implementation** |
| **Model Architecture** | Custom CNN: 2 basic blocks (Conv2D(12)-ELU-BatchNorm-MaxPool), Flatten, Dropout(0.3), Dense(10, ReLU), Dense(1, Sigmoid). Exactly 12,873 trainable parameters. | `src/model.py:build_custom_cnn` | **Verified from implementation & checkpoint inspection** |
| **Optimizer & Loss** | Adam ($\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-7}$), Binary Cross-Entropy (`binary_crossentropy`). | `scripts/run_via3_imbalance_experiments.py:108` | **Verified from implementation** |
| **Batch Size & Epochs** | Batch size = 32, Epochs = 15. | `scripts/run_via3_imbalance_experiments.py:100` | **Verified from implementation** |
| **Checkpoint Callback** | `BestPRAUCCallback`: monitors validation set every epoch, tracks validation PR-AUC with secondary tie-breaker on F1. Restores best weights at train end and saves `ckpt_path`. | `scripts/run_via3_imbalance_experiments.py:48-92` | **Verified from implementation** |
| **Threshold Calibration** | Sweep on validation probabilities $\tau \in [0.01, 0.99]$ with step $0.01$ targeting `max_f1`. Selected threshold $\tau^*$ is frozen before test inference. | `src/experiment_contract.py:select_threshold_from_val` | **Verified from implementation** |
| **Baseline Strategy** | Standard unweighted loss on natural training distribution. | `scripts/run_via3_imbalance_experiments.py:127` | **Verified from implementation** |
| **Balanced Sampling** | `balanced_train_ds`: balanced batch sampling interleaving 16 HS and 16 NHS per batch with `steps_per_epoch = 22,763 // 32 = 711`. Minority class samples are repeated with replacement. | `src/openhsd_data.py:create_balanced_via3_dataset` | **Verified from implementation** |
| **Class-Weighted Control** | `class_weights = {0: 1.0, 1: 30.2249}` reflecting inverse class frequencies in training split ($22,034 / 729$). | `src/openhsd_data.py:157` | **Verified from implementation** |
| **XAI Cohort** | Fixed 50-layout diagnostic cohort stratified across 6 strata (`shared_tp`: 10, `shared_fn`: 10, `balanced_won`: 10, `baseline_won`: 10, `shared_tn`: 5, `shared_fp`: 5). | `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json` | **Verified from artifact** |

---

## 4. Multi-Seed Experiment Contract (Seeds 42, 100, 2024)

### 4.1 Core Multi-Seed Principles
1. **Reference Seed 42 is Frozen:** Seed 42 will **not** be retrained. All metrics, checkpoints, and predictions already verified will serve as the immutable reference for Seed 42.
2. **Deterministic Seed Setting:** For each run with seed $S \in \{100, 2024\}$, the random environment must be initialized before dataset pipeline creation and model building via:
   ```python
   tf.keras.backend.clear_session()
   tf.keras.utils.set_random_seed(S)
   np.random.seed(S)
   ```
3. **Identical Splitting:** Dataset split membership is strictly frozen. Train, validation, clean test, and official test partitions remain identical across all seeds.
4. **Independent Checkpoint Selection:** Each seed selects its best checkpoint independently using the identical validation rule (`best_val_pr_auc`).
5. **Independent Validation Calibration:** Each seed selects its operational threshold $\tau^*_S$ exclusively from its own validation predictions using `max_f1`. No test data is used for threshold calibration.
6. **Double Operating Point Reporting:** For every seed, test metrics will be computed and reported at both:
   - Default threshold $\tau = 0.50$
   - Validation-calibrated threshold $\tau^*_S$
7. **Threshold-Independent Metrics:** PR-AUC and ROC-AUC are computed over the full continuous probability spectrum.
8. **Sample Size Caveat:** $N=3$ seeds provide an initial empirical assessment of optimization and initialization sensitivity, not a definitive characterization of population variance.

### 4.2 Proposed Seed-by-Seed Training Matrix

| Run ID | Strategy | Seed | Status | Expected Runtime | Output Directory | Checkpoint Path |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `base_42` | Baseline (Unweighted) | 42 | **COMPLETED (Reference)** | 0s (Frozen) | `results/via3_experiments/imbalance/` | `models/via3/via3_baseline_seed42.keras` |
| `base_100` | Baseline (Unweighted) | 100 | **PENDING AUTHORIZATION** | ~70s | `results/via3_experiments/multiseed/baseline/seed_100/` | `models/via3_multiseed/via3_baseline_seed100.keras` |
| `base_2024`| Baseline (Unweighted) | 2024 | **PENDING AUTHORIZATION** | ~70s | `results/via3_experiments/multiseed/baseline/seed_2024/` | `models/via3_multiseed/via3_baseline_seed2024.keras` |
| `bal_42` | Balanced Sampling | 42 | **COMPLETED (Reference)** | 0s (Frozen) | `results/via3_experiments/imbalance/` | `models/via3_imbalance/via3_balanced_sampling_seed42.keras` |
| `bal_100` | Balanced Sampling | 100 | **PENDING AUTHORIZATION** | ~70s | `results/via3_experiments/multiseed/balanced_sampling/seed_100/` | `models/via3_multiseed/via3_balanced_sampling_seed100.keras` |
| `bal_2024` | Balanced Sampling | 2024 | **PENDING AUTHORIZATION** | ~70s | `results/via3_experiments/multiseed/balanced_sampling/seed_2024/` | `models/via3_multiseed/via3_balanced_sampling_seed2024.keras` |
| `cw_42` | Class-Weighted (Control)| 42 | **COMPLETED (Reference)** | 0s (Frozen) | `results/via3_experiments/imbalance/` | `models/via3_imbalance/via3_class_weighted_seed42.keras` |
| `cw_100` | Class-Weighted (Control)| 100 | **OPTIONAL CONTROL** | ~75s | `results/via3_experiments/multiseed/class_weighted/seed_100/` | `models/via3_multiseed/via3_class_weighted_seed100.keras` |
| `cw_2024` | Class-Weighted (Control)| 2024 | **OPTIONAL CONTROL** | ~75s | `results/via3_experiments/multiseed/class_weighted/seed_2024/` | `models/via3_multiseed/via3_class_weighted_seed2024.keras` |

*Note on Class-Weighted Control:* Class-weighted training is included in the matrix as an optional control. If omitted during the initial run to conserve GPU cycles, the primary comparison between Baseline and Balanced Sampling remains fully valid and self-contained.

---

## 5. Output Isolation and Overwrite Protection Audit

### 5.1 Code Deficiencies in Existing Runner (`scripts/run_via3_multiseed.py`)

Inspection of `scripts/run_via3_multiseed.py` revealed four concrete deficiencies that must be remediated in Workstream E2 before execution:

1. **Hardcoded Baseline Invocations (Line 132 & 138):**
   ```python
   # Existing code in scripts/run_via3_multiseed.py:
   from scripts.run_via3_baseline import run_baseline
   ...
   for s in seeds:
       res = run_baseline(...)
   ```
   *Defect:* The script does not support `strategy="balanced_sampling"`. Calling it with `--strategy balanced_sampling` executes the unweighted baseline instead.
2. **No Seed 42 Reference Bypass (Lines 135–148):**
   *Defect:* The loop iterates over all passed seeds without checking if Seed 42 is already complete. Running with `--seeds 42 100 2024` would re-train Seed 42 and overwrite the reference checkpoint.
3. **Directory Path Collisions:**
   *Defect:* The existing script writes to `output_dir / f"seed_{s}"` without namespacing by strategy. If baseline and balanced sampling are run to the same output directory, prediction files (`clean_test_predictions.csv`) will overwrite each other.
4. **Missing Overwrite Safeguards:**
   *Defect:* The script lacks a pre-check to confirm whether an output directory already contains valid predictions and checkpoints before initiating training.

### 5.2 Proposed Architecture for Workstream E2 Runner Patch
In E2, `scripts/run_via3_multiseed.py` will be updated to:
- Import `train_single_strategy` from `scripts/run_via3_imbalance_experiments.py` to support all three strategies with identical data pipelines.
- Partition outputs by strategy: `results/via3_experiments/multiseed/{strategy}/seed_{seed}/`.
- When `seed == 42`: automatically load the audited results from `results/via3_experiments/imbalance/` and checkpoint from `models/via3_imbalance/` without invoking training.
- Check if `clean_test_predictions.csv` and checkpoint file already exist. If present and valid, log and skip training to allow safe resume on interruption.

---

## 6. GPU Resource and Runtime Preflight

### 6.1 Hardware and Environment Verification
- **GPU Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (Bus ID: `0000:01:00.0`).
- **Total VRAM:** 6,144 MiB (6 GB).
- **Available VRAM:** ~6,132 MiB (12 MiB currently in use by desktop environment, 0 active GPU compute jobs).
- **GPU Temperature & Power:** 50°C, 8W / 40W (Idle).
- **CUDA / Driver Stack:** NVIDIA Driver 595.99.02, CUDA 13.2.
- **Execution Wrapper:** `./scripts/tf_gpu.sh` sets `LD_LIBRARY_PATH` to `/run/opengl-driver/lib` and NixOS venv CUDA/cuDNN wheels, configuring `XLA_FLAGS`.

### 6.2 Prior Runtime Evidence and Capacity Planning
From the audited Seed 42 runs in `results/via3_experiments/imbalance/imbalance_comparison_summary.json`:
- Baseline training duration: **66.53 seconds** (15 epochs).
- Balanced sampling training duration: **70.18 seconds** (15 epochs).
- Class-weighted training duration: **78.96 seconds** (15 epochs).

**Projected Runtime for 4 New Runs (Seeds 100 & 2024 for Baseline & Balanced):**
- Baseline (seeds 100, 2024): $2 \times 70\text{s} = 140\text{s}$ (~2.3 min).
- Balanced Sampling (seeds 100, 2024): $2 \times 70\text{s} = 140\text{s}$ (~2.3 min).
- Total training runtime: **~4.7 minutes**.
- Sequential execution recommendation: Strongly recommend executing sequentially (one seed and strategy at a time) to ensure VRAM remains below 2 GB and avoid memory fragmentation.

---

## 7. Multi-Seed XAI Robustness Plan

### 7.1 Evaluated Checkpoints
- Baseline: Seeds `42`, `100`, `2024`.
- Balanced Sampling: Seeds `42`, `100`, `2024`.

### 7.2 Fixed 50-Layout Diagnostic Cohort
To allow direct within-layout paired analysis across seeds and models:
- All models across all seeds will be evaluated on the **exact same 50 physical layout images** recorded in `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json`.
- Layouts will **not** be re-sampled per seed. This eliminates layout sampling variance as a confounding factor when measuring seed variance.

### 7.3 Multi-Seed XAI Analysis Protocol
1. **Methods Evaluated:** Grad-CAM, Grad-CAM++, LayerCAM, Gradient Saliency, Occlusion Sensitivity (32x32), and Random Baseline.
2. **Within-Layout Pairing:** Each layout image $i \in \{1, \dots, 50\}$ has a matched explanation under Baseline and Balanced Sampling for Seed $S$.
3. **Separating Seed Variance from Layout Variance:**
   - Compute mean Insertion AUC, Deletion AUC, and Stability $\rho$ across the 50 layouts for each seed: $\mu_{\text{ins}}(S)$, $\mu_{\text{del}}(S)$, $\mu_{\text{stab}}(S)$.
   - Report inter-seed mean and standard deviation:
     $$\bar{\mu} = \frac{1}{3}\sum_{S \in \{42, 100, 2024\}} \mu(S), \quad s_{\text{seed}} = \sqrt{\frac{1}{2}\sum_{S} (\mu(S) - \bar{\mu})^2}$$
   - Compute paired seed differences: $\Delta(S) = \mu_{\text{bal}}(S) - \mu_{\text{base}}(S)$.
   - Verify whether $\Delta(S) > 0$ holds across all three independent seeds.
4. **Multiplicity Strategy:** Apply Holm-Bonferroni correction within each seed's 17-test family, and evaluate consistency of adjusted significance across seeds.

---

## 8. Aggregation and Statistical Reporting Plan

Post-training aggregation will generate:
1. **Individual Seed Metrics Table:**
   - Display PR-AUC, ROC-AUC, HS Recall, HS Precision, HS F1, Balanced Accuracy, and calibrated threshold $\tau^*_S$ for each seed $S \in \{42, 100, 2024\}$.
2. **Across-Seed Statistical Summary Table:**
   - Mean, sample standard deviation ($ddof=1$), and 95% confidence intervals based on Student's $t$-distribution ($t_{\text{crit}} = 4.303$ for $df = 2$):
     $$\text{CI}_{95\%} = \bar{x} \pm 4.303 \times \frac{s}{\sqrt{3}}$$
3. **Paired Seed Comparison:**
   - Compute paired difference $\Delta_S = \text{Metric}_{\text{bal}, S} - \text{Metric}_{\text{base}, S}$ for each seed.
   - Confirm whether the PR-AUC and F1 advantages of Balanced Sampling are consistently positive across all seeds.
4. **Threshold Variance Analysis:**
   - Report the range and standard deviation of calibrated thresholds $\tau^*_S$ across seeds to measure threshold sensitivity to weight initialization.

---

## 9. Required Remediations Before Training Launch

| Severity | Issue ID | Description | Remediation Plan (To apply in Workstream E2) |
| :---: | :---: | :--- | :--- |
| **CRITICAL** | `BLOCKER-01` | `scripts/run_via3_multiseed.py` only invokes `run_baseline`, ignoring `balanced_sampling`. | Refactor runner to import `train_single_strategy` from `scripts/run_via3_imbalance_experiments.py`, enabling execution of `baseline`, `balanced_sampling`, and `class_weighted`. |
| **CRITICAL** | `BLOCKER-02` | `scripts/run_via3_multiseed.py` retrains Seed 42 and lacks overwrite guards. | Add reference-seed bypass (auto-load audited Seed 42 outputs without training) and file-existence checks before model training. |
| **MEDIUM** | `BLOCKER-03` | Output paths do not namespace by strategy. | Update output structure to `results/via3_experiments/multiseed/{strategy}/seed_{seed}/` to eliminate filename collisions. |

---

## 10. Scope of Work Authorized for Workstream E2

Once human authorization is granted:
1. Apply the narrow code fixes for `BLOCKER-01`, `BLOCKER-02`, and `BLOCKER-03` to `scripts/run_via3_multiseed.py`.
2. Verify output isolation via a 10-sample dry-run or unit test.
3. Launch sequential GPU training via `./scripts/tf_gpu.sh`:
   - Baseline: Seeds `100`, `2024` (~2.3 min total).
   - Balanced Sampling: Seeds `100`, `2024` (~2.3 min total).
   - (Optional) Class-Weighted: Seeds `100`, `2024` (~2.5 min total).
4. Run frozen test evaluation and save per-seed prediction CSVs.
5. Execute multi-seed XAI evaluation on the fixed 50 diagnostic layouts.
6. Aggregate multi-seed statistics and update `draft_reports/via3/` tables.

---

## 11. Human-Approval Checklist

- [ ] Acknowledge that Reference Seed 42 is frozen and will not be retrained.
- [ ] Authorize code refactoring of `scripts/run_via3_multiseed.py` to fix `BLOCKER-01`, `BLOCKER-02`, and `BLOCKER-03`.
- [ ] Authorize sequential GPU training of Seeds 100 and 2024 for Baseline and Balanced Sampling (projected ~5 minutes total runtime on RTX 3050).
- [ ] Specify whether Class-Weighted control should also be trained for Seeds 100 and 2024 (+2.5 minutes).
- [ ] Confirm permission to proceed with Workstream E2.
