# Workstream E4.3-A — Independent Artifact and Provenance Audit

**Audit Date:** 2026-10-23  
**Auditor:** Independent Read-Only Audit Assistant  
**Repository:** `/home/peskybird/Projects/Ai_Ml_DA/`  
**Target Evaluation:** Workstream E4.2 Multi-Seed XAI Robustness Evaluation  
**Audited Artifact Directory:** `results/via3_experiments/xai_multiseed/`  
**Overall Verdict:** **PASS WITH QUALIFICATIONS**  

---

## 1. Executive Summary

This report provides a comprehensive, read-only audit of the completed Workstream E4.2 multi-seed explainability (XAI) robustness evaluation on OpenHSDBench `via-3`.

All empirical metrics, row counts, grouping dimensions, and summary tables were independently reconstructed directly from the machine-readable CSV artifacts (`master_faithfulness_metrics.csv`, `master_stability_metrics.csv`, `master_collapse_diagnostics.csv`) and JSON summary documents, without relying on claims made in the narrative reports.

### Key Audit Findings:
1. **Cryptographic Provenance Verified:** The frozen 50-layout diagnostic cohort manifest matches SHA-256 hash `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb` byte-for-byte. All six model checkpoints in `models/via3_multiseed/` match their expected SHA-256 hashes exactly.
2. **Structural Integrity 100% Complete:**
   - `master_faithfulness_metrics.csv`: Exactly 1,800 rows ($6\text{ models} \times 50\text{ layouts} \times 6\text{ methods}$), zero nulls, zero non-finite values, zero duplicate keys.
   - `master_stability_metrics.csv`: Exactly 1,500 rows ($6\text{ models} \times 50\text{ layouts} \times 5\text{ deterministic methods}$), zero nulls, zero non-finite values, zero duplicate keys.
   - `master_collapse_diagnostics.csv`: Exactly 300 rows ($6\text{ models} \times 50\text{ layouts} \times 1\text{ CAM evaluation}$), zero nulls, zero non-finite values, zero duplicate keys.
   - 100% of rows report status `SUCCESS`.
3. **Headline Scientific Results Fully Reconstructed:**
   - **Balanced Sampling eliminates Flat-CAM collapse:** 0 / 150 Flat-CAMs (0.0%) in Balanced Sampling vs. 3 / 150 (2.0%) in Baseline; Balanced Sampling preserves spatial standard deviation across seeds at $0.1552 \pm 0.0067$ vs. Baseline dropping to $0.0589 \pm 0.0132$ in Seed 2024.
   - **Occlusion is the most faithful explainer:** Balanced Sampling Insertion AUC is $0.8422 \pm 0.1034$ with a consistently positive paired boost over Baseline across all 3 seeds ($\Delta = +0.4060 \pm 0.1229$).
   - **Grad-CAM++ fails stability testing:** Mean stability Spearman rank correlation is $\bar{\rho} \le 0.31$ across all six models and seeds (Baseline: $0.2692 \pm 0.0401$, Balanced: $0.1717 \pm 0.0799$).
   - **Grad-CAM & LayerCAM are highly stable in Balanced models:** Grad-CAM $\bar{\rho} = 0.9505 \pm 0.0179$; LayerCAM $\bar{\rho} = 0.9554 \pm 0.0075$.
   - **Models pass Adebayo parameter randomization:** $|\bar{\rho}| \le 0.33$ across all six models.
4. **Discrepancies Identified:**
   - **Zero discrepancies** between reconstructed CSV metrics and `multiseed_xai_comparison_summary.json`.
   - **One minor reporting discrepancy (ISSUE-MIN-01)** in Table 4 of `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md`: In the `Baseline Pooled` row, `Spatial Std SD` is reported as `0.0589` (which is a transcription duplicate of the `Spatial Std Mean` of `Baseline Seed 2024` immediately above it), whereas the true sample standard deviation across all 150 layout observations is `0.0571` (or $0.0615$ across the 3 seed means).
   - **Reporting clarification needed (ISSUE-MIN-02):** The $\pm$ notation in Tables 1, 2, 3, and 5 represents the sample standard deviation across $K=3$ training seeds (`ddof=1`), while in Table 4 per-model rows it represents standard deviation across $N=50$ layout observations.

---

## 2. Read-Only Boundary and Audit Inventory

This audit operated strictly under read-only constraints:
- No GPU execution or neural network inference was initiated.
- No model checkpoints were trained, modified, or overwritten.
- No existing source code, test code, or prior report was modified.
- No git commits, resets, or branch changes were made.
- Protected files (`docs/DA2_Group18_Report.docx`, `latex_report/`, `draft_reports/via3/`, `results/via3_experiments/xai_comparison/`) remain completely untouched.

### Primary Artifacts Inspected:

| Artifact Path | File Size | SHA-256 Hash | Purpose |
| :--- | :--- | :--- | :--- |
| `results/via3_experiments/xai_multiseed/WORKSTREAM_E4_2_EXECUTION_REPORT.md` | 5.5 KB | `a793a54b38d3...` | Hardware telemetry, execution runtime logs |
| `results/via3_experiments/xai_multiseed/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` | 13.2 KB | `871a9bc28f41...` | Primary scientific report with summary tables |
| `results/via3_experiments/xai_multiseed/multiseed_xai_comparison_summary.json` | 29.8 KB | `7648d65b407e...` | Machine-readable summary & aggregations |
| `results/via3_experiments/xai_multiseed/reproducibility_manifest.json` | 2.0 KB | `75f28469ad93...` | Provenance manifest with artifact hashes |
| `results/via3_experiments/xai_multiseed/master_faithfulness_metrics.csv` | 214.7 KB | `0b49d2c414e5...` | Layout-level deletion & insertion AUC metrics |
| `results/via3_experiments/xai_multiseed/master_stability_metrics.csv` | 169.6 KB | `fabbaee2eaee...` | Layout-level noise perturbation stability metrics |
| `results/via3_experiments/xai_multiseed/master_collapse_diagnostics.csv` | 35.8 KB | `8db96f2daffb...` | Layout-level CAM contrast & Flat-CAM metrics |
| `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json` | 14.5 KB | `1c547c162c32...` | Canonical frozen 50-layout diagnostic cohort |
| `results/via3_experiments/multiseed/WORKSTREAM_E4_1_XAI_PREFLIGHT.md` | 24.3 KB | `4f388bbd0fa4...` | Frozen preflight experiment contract |

---

## 3. Cohort and Checkpoint Provenance Audit

### 3.1 Cohort Manifest Integrity
The diagnostic cohort manifest was independently verified against the contract frozen in Workstream E4.1:
- Canonical Path: `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json`
- Quarantined Copy: `results/via3_experiments/xai_multiseed/cohort/xai_sample_cohort_manifest.json`
- Expected Hash: `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`
- Computed Hash: `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`
- **Verdict:** **MATCH (VERIFIED)**. Exactly 50 unique layout PNG filenames across four confusion quadrants (15 TP, 10 FN, 15 TN, 10 FP).

### 3.2 Model Checkpoint Integrity
The six evaluated model checkpoints in `models/via3_multiseed/` were independently hashed without loading model weights into memory:

| Strategy | Seed | File Path | Expected Hash (E1/E2 Preflight) | Actual Audited SHA-256 | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 42 | `models/via3_multiseed/via3_baseline_seed42.keras` | `97a9cb8bf776b76b...` | `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887` | **MATCH (VERIFIED)** |
| **Baseline** | 100 | `models/via3_multiseed/via3_baseline_seed100.keras` | `f28e4277738794a6...` | `f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455` | **MATCH (VERIFIED)** |
| **Baseline** | 2024 | `models/via3_multiseed/via3_baseline_seed2024.keras` | `9347e568cbe59672...` | `9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d` | **MATCH (VERIFIED)** |
| **Balanced** | 42 | `models/via3_multiseed/via3_balanced_sampling_seed42.keras` | `551a861f1e3537b9...` | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` | **MATCH (VERIFIED)** |
| **Balanced** | 100 | `models/via3_multiseed/via3_balanced_sampling_seed100.keras` | `c957304bf07a8aab...` | `c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815` | **MATCH (VERIFIED)** |
| **Balanced** | 2024 | `models/via3_multiseed/via3_balanced_sampling_seed2024.keras` | `297fa95748adc7a1...` | `297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9` | **MATCH (VERIFIED)** |

### 3.3 Reproducibility Manifest Completeness
`reproducibility_manifest.json` was parsed and checked against actual file properties:
- Records Python version (`3.11.16`), TensorFlow version (`2.15.1`), NumPy version (`1.26.4`).
- Records evaluation seeds (`[42, 100, 2024]`), random baseline seed (`42`), and perturbation noise seed (`42`).
- All recorded SHA-256 hashes for output files (`master_faithfulness_metrics_csv`, `master_stability_metrics_csv`, `master_collapse_diagnostics_csv`, `multiseed_xai_comparison_summary_json`) match the on-disk files byte-for-byte.

---

## 4. CSV Structural Integrity and Schema Audit

### 4.1 Master Faithfulness Metrics (`master_faithfulness_metrics.csv`)
- **Row Count:** Exactly 1,800 rows.
- **Evaluation Dimension Reconciled:**
  $$2\text{ Strategies} \times 3\text{ Seeds} \times 50\text{ Layouts} \times 6\text{ Methods} = 1,800\text{ records}$$
- **Schema & Data Types:**
  - `model`: `str` (6 unique values: `{baseline,balanced_sampling}_seed{42,100,2024}`)
  - `strategy`: `str` (`baseline`, `balanced_sampling`)
  - `seed`: `int64` (`42`, `100`, `2024`)
  - `filename`: `str` (50 unique filenames matching cohort manifest)
  - `cohort`: `str` (`test`)
  - `ground_truth`: `int64` (`0` or `1`, exactly 25 of each per model)
  - `method`: `str` (`gradcam`, `gradcam_plus_plus`, `layercam`, `saliency`, `occlusion`, `random`)
  - `deletion_auc`: `float64`, Range: $[0.0003, 0.9830]$, Mean: $0.3423$
  - `insertion_auc`: `float64`, Range: $[0.0033, 0.9956]$, Mean: $0.4447$
  - `status`: `str` (`SUCCESS` for all 1,800 rows)
- **Null Values:** 0
- **Non-Finite Values:** 0
- **Primary Key Uniqueness (`strategy, seed, filename, method`):** 1,800 unique rows (0 duplicates).

### 4.2 Master Stability Metrics (`master_stability_metrics.csv`)
- **Row Count:** Exactly 1,500 rows.
- **Evaluation Dimension Reconciled:**
  $$2\text{ Strategies} \times 3\text{ Seeds} \times 50\text{ Layouts} \times 5\text{ Methods} = 1,500\text{ records}$$
- **Why 1,500 rows instead of 1,800:** Random Baseline is intentionally excluded from stability perturbation testing. Stability measures whether the deterministic gradient or feature-attribution maps produced by a model are robust against small input noise ($\sigma=0.05$). Random Baseline is an arbitrary layout permutation without deterministic weights or activations; testing its stability under input noise is scientifically meaningless.
- **Schema & Data Types:**
  - `model`, `strategy`, `seed`, `filename`, `cohort`, `ground_truth`, `status`: Identical schema to faithfulness.
  - `method`: 5 methods (`gradcam`, `gradcam_plus_plus`, `layercam`, `saliency`, `occlusion`).
  - `stability_spearman_rho`: `float64`, Range: $[-0.4107, 1.0000]$, Mean: $0.6698$
  - `status`: `SUCCESS` for all 1,500 rows.
- **Null Values:** 0
- **Non-Finite Values:** 0
- **Primary Key Uniqueness (`strategy, seed, filename, method`):** 1,500 unique rows (0 duplicates).

### 4.3 Master Collapse Diagnostics (`master_collapse_diagnostics.csv`)
- **Row Count:** Exactly 300 rows.
- **Evaluation Dimension Reconciled:**
  $$2\text{ Strategies} \times 3\text{ Seeds} \times 50\text{ Layouts} = 300\text{ records}$$
- **Why 300 rows instead of 1,800:** Representation collapse (Flat-CAM pathology) evaluates whether the target convolutional layer (`conv2d_5`) feature maps and gradients collapse into uniform constant values under Grad-CAM. Because collapse is a model-level representation property assessed on the target layer, exactly one diagnostic evaluation is performed per model-layout pair.
- **Schema & Data Types:**
  - `model`, `strategy`, `seed`, `filename`, `cohort`, `ground_truth`, `status`: Identical schema.
  - `is_flat_cam`: `bool` (`True` for 3 records, `False` for 297 records).
  - `cam_std`: `float64`, Range: $[0.0000, 0.2269]$, Mean: $0.1423$
  - `cam_max`: `float64`, Range: $[0.0000, 1.0000]$, Mean: $0.9900$
  - `cam_mean`: `float64`, Range: $[0.0000, 0.2995]$, Mean: $0.1424$
  - `status`: `SUCCESS` for all 300 rows.
- **Null Values:** 0
- **Non-Finite Values:** 0
- **Primary Key Uniqueness (`strategy, seed, filename`):** 300 unique rows (0 duplicates).

---

## 5. Independent Metric Reconstruction and Comparison

Using independent, in-memory data analysis on the raw CSV records, all per-seed and cross-seed statistics were recalculated.

### 5.1 Per-Seed Metric Reconstruction vs. JSON Summary
Each per-seed mean was calculated across the 50 layout observations and compared to `multiseed_xai_comparison_summary.json["per_model_summaries"]`:

| Model Identifier | Method | CSV Rec. Del AUC | JSON Del AUC | CSV Rec. Ins AUC | JSON Ins AUC | CSV Rec. Stab $\bar{\rho}$ | JSON Stab $\bar{\rho}$ | Discrepancy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_seed42` | `gradcam` | 0.2706 | 0.2706 | 0.4369 | 0.4369 | 0.9712 | 0.9712 | **None (Exact)** |
| `baseline_seed42` | `gradcam_plus_plus`| 0.3712 | 0.3712 | 0.3663 | 0.3663 | 0.2365 | 0.2365 | **None (Exact)** |
| `baseline_seed42` | `layercam` | 0.4248 | 0.4248 | 0.3016 | 0.3016 | 0.9679 | 0.9679 | **None (Exact)** |
| `baseline_seed42` | `saliency` | 0.8053 | 0.8053 | 0.2169 | 0.2169 | 0.4480 | 0.4480 | **None (Exact)** |
| `baseline_seed42` | `occlusion` | 0.2713 | 0.2713 | 0.4542 | 0.4542 | 0.6761 | 0.6761 | **None (Exact)** |
| `baseline_seed42` | `random` | 0.3271 | 0.3271 | 0.3277 | 0.3277 | N/A | N/A | **None (Exact)** |
| `baseline_seed100` | `gradcam` | 0.2044 | 0.2044 | 0.2054 | 0.2054 | 0.9538 | 0.9538 | **None (Exact)** |
| `baseline_seed100` | `occlusion` | 0.1070 | 0.1070 | 0.3940 | 0.3940 | 0.7594 | 0.7594 | **None (Exact)** |
| `baseline_seed2024`| `gradcam` | 0.1293 | 0.1293 | 0.3309 | 0.3309 | 0.6363 | 0.6363 | **None (Exact)** |
| `baseline_seed2024`| `occlusion` | 0.1115 | 0.1115 | 0.4605 | 0.4605 | 0.6600 | 0.6600 | **None (Exact)** |
| `balanced_sampling_seed42` | `gradcam` | 0.3734 | 0.3734 | 0.6118 | 0.6118 | 0.9710 | 0.9710 | **None (Exact)** |
| `balanced_sampling_seed42` | `occlusion` | 0.1470 | 0.1470 | 0.9208 | 0.9208 | 0.7444 | 0.7444 | **None (Exact)** |
| `balanced_sampling_seed100` | `gradcam` | 0.5732 | 0.5732 | 0.4977 | 0.4977 | 0.9424 | 0.9424 | **None (Exact)** |
| `balanced_sampling_seed100` | `occlusion` | 0.2297 | 0.2297 | 0.8808 | 0.8808 | 0.7524 | 0.7524 | **None (Exact)** |
| `balanced_sampling_seed2024`| `gradcam` | 0.1649 | 0.1649 | 0.3272 | 0.3272 | 0.9380 | 0.9380 | **None (Exact)** |
| `balanced_sampling_seed2024`| `occlusion` | 0.0328 | 0.0328 | 0.7251 | 0.7251 | 0.7243 | 0.7243 | **None (Exact)** |

*Audit Result:* Across all 36 model-method combinations (6 models $\times$ 6 methods), **zero numerical discrepancies** exist between the CSV records and the JSON summary.

### 5.2 Cross-Seed Metric Aggregation Audit
Cross-seed aggregation computes the summary across the $K=3$ training seeds.
- **Mean:** $\bar{X} = \frac{1}{3} \sum_{k=1}^3 \bar{x}_k$
- **Standard Deviation:** Sample standard deviation with Bessel's correction ($ddof=1$):
  $$s = \sqrt{\frac{1}{2} \sum_{k=1}^3 (\bar{x}_k - \bar{X})^2}$$

| Strategy | Method | Metric | Recomputed Mean $\pm$ SD | JSON Recorded Mean $\pm$ SD | Report Table 1/2/3 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `gradcam` | Insertion AUC | $0.3244 \pm 0.1159$ | $0.3244 \pm 0.1159$ | $0.3244 \pm 0.1159$ | **VERIFIED** |
| | `gradcam` | Deletion AUC | $0.2014 \pm 0.0707$ | $0.2014 \pm 0.0707$ | $0.2014 \pm 0.0707$ | **VERIFIED** |
| | `gradcam` | Stability $\bar{\rho}$ | $0.8538 \pm 0.1885$ | $0.8538 \pm 0.1885$ | $0.8538 \pm 0.1885$ | **VERIFIED** |
| | `occlusion` | Insertion AUC | $0.4362 \pm 0.0367$ | $0.4362 \pm 0.0367$ | $0.4362 \pm 0.0367$ | **VERIFIED** |
| | `occlusion` | Deletion AUC | $0.1633 \pm 0.0936$ | $0.1633 \pm 0.0936$ | $0.1633 \pm 0.0936$ | **VERIFIED** |
| | `occlusion` | Stability $\bar{\rho}$ | $0.6985 \pm 0.0534$ | $0.6985 \pm 0.0534$ | $0.6985 \pm 0.0534$ | **VERIFIED** |
| | `gradcam_plus_plus`| Stability $\bar{\rho}$| $0.2692 \pm 0.0401$ | $0.2692 \pm 0.0401$ | $0.2692 \pm 0.0401$ | **VERIFIED** |
| | `layercam` | Stability $\bar{\rho}$ | $0.9556 \pm 0.0185$ | $0.9556 \pm 0.0185$ | $0.9556 \pm 0.0185$ | **VERIFIED** |
| **Balanced** | `gradcam` | Insertion AUC | $0.4789 \pm 0.1432$ | $0.4789 \pm 0.1432$ | $0.4789 \pm 0.1432$ | **VERIFIED** |
| | `gradcam` | Deletion AUC | $0.3705 \pm 0.2042$ | $0.3705 \pm 0.2042$ | $0.3705 \pm 0.2042$ | **VERIFIED** |
| | `gradcam` | Stability $\bar{\rho}$ | $0.9505 \pm 0.0179$ | $0.9505 \pm 0.0179$ | $0.9505 \pm 0.0179$ | **VERIFIED** |
| | `occlusion` | Insertion AUC | $0.8422 \pm 0.1034$ | $0.8422 \pm 0.1034$ | $0.8422 \pm 0.1034$ | **VERIFIED** |
| | `occlusion` | Deletion AUC | $0.1365 \pm 0.0989$ | $0.1365 \pm 0.0989$ | $0.1365 \pm 0.0989$ | **VERIFIED** |
| | `occlusion` | Stability $\bar{\rho}$ | $0.7404 \pm 0.0145$ | $0.7404 \pm 0.0145$ | $0.7404 \pm 0.0145$ | **VERIFIED** |
| | `gradcam_plus_plus`| Stability $\bar{\rho}$| $0.1717 \pm 0.0799$ | $0.1717 \pm 0.0799$ | $0.1717 \pm 0.0799$ | **VERIFIED** |
| | `layercam` | Stability $\bar{\rho}$ | $0.9554 \pm 0.0075$ | $0.9554 \pm 0.0075$ | $0.9554 \pm 0.0075$ | **VERIFIED** |

*Audit Result:* The headline figures reported across Tables 1, 2, 3, and 5 reconcile with $100\%$ precision against independently reconstructed statistics from the underlying CSVs.

---

## 6. Discrepancy Analysis

### Finding ISSUE-MIN-01: Table 4 Markdown Typo in Pooled Baseline Spatial Std SD
- **Location:** `results/via3_experiments/xai_multiseed/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md`, Section 4, Table 4, row 126:
  ```markdown
  | *Baseline Pooled* | *All* | *150* | *3* | *2.0%* | *0.1294* | *0.0589* | *0.9800* |
  ```
- **Observed Value in Report:** `0.0589` in column `Spatial Std SD`.
- **Independently Computed Values:**
  - Standard deviation of `cam_std` across all 150 layout observations ($ddof=1$): $\mathbf{0.0571}$.
  - Standard deviation of the 3 seed means ($[0.1568, 0.1724, 0.0589]$, $ddof=1$): $\mathbf{0.0615}$.
- **Source of Error:** Row 125 immediately preceding is `Baseline Seed 2024`, whose `Spatial Std Mean` is `0.0589`. During manual table formatting, the value `0.0589` was mistakenly transcribed into the `Spatial Std SD` column of the pooled row.
- **Classification:** **MINOR**. Does not alter any empirical artifact, CSV, or scientific conclusion.

### Finding ISSUE-MIN-02: Aggregation Unit Ambiguity in Report Table Headers
- **Location:** Table headers in `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` (Tables 1, 2, 3, 4, 5).
- **Issue:** The header reads `Aggregate Mean $\pm$ SD`. The text does not explicitly define whether the standard deviation is computed across individual layout observations ($N=50$ or $150$) or across the $K=3$ random training seeds.
- **Audit Verification:**
  - In Tables 1, 2, 3, and 5: The SD is computed **across the $K=3$ training seeds** ($ddof=1$).
  - In Table 4 (per-seed rows): The SD is computed **across the $N=50$ layout observations** within that model run ($ddof=1$).
- **Classification:** **MINOR**. The report text correctly notes that $K=3$ seeds were evaluated, but future paper draft revisions should make the aggregation unit explicit in table footnotes.

---

## 7. Claim-by-Claim Verification Matrix

| Claim in E4.2 Narrative | Evidence Source | Audited Value | Status |
| :--- | :--- | :--- | :--- |
| **All 1,800 planned evaluations completed** | `master_faithfulness_metrics.csv` | Exactly 1,800 rows with status `SUCCESS` | **VERIFIED** |
| **Balanced Sampling eliminates Flat-CAMs** | `master_collapse_diagnostics.csv` | Balanced: 0 / 150 (0.0%); Baseline: 3 / 150 (2.0%) | **VERIFIED** |
| **Baseline Seed 2024 suffered contrast collapse** | `master_collapse_diagnostics.csv` | Mean spatial std dropped to $0.0589 \pm 0.0132$ | **VERIFIED** |
| **Occlusion has highest insertion AUC** | `master_faithfulness_metrics.csv` | Balanced Occlusion Insertion AUC: $0.8422 \pm 0.1034$ | **VERIFIED** |
| **Occlusion boost over Baseline is consistently positive** | Paired per-seed differences | Seed 42: $+0.4666$, Seed 100: $+0.4868$, Seed 2024: $+0.2646$ | **VERIFIED** |
| **Grad-CAM++ fails stability across all seeds** | `master_stability_metrics.csv` | Baseline: $0.2692 \pm 0.0401$; Balanced: $0.1717 \pm 0.0799$ | **VERIFIED** |
| **Grad-CAM and LayerCAM maintain high stability** | `master_stability_metrics.csv` | Balanced Grad-CAM: $0.9505 \pm 0.0179$; LayerCAM: $0.9554 \pm 0.0075$ | **VERIFIED** |
| **Explanations pass Adebayo parameter randomization** | `multiseed_xai_comparison_summary.json` | Mean $\bar{\rho} \in [-0.3321, +0.0553]$ against randomized CNN | **VERIFIED** |
| **Checkpoints remained unaltered during inference** | Checkpoint SHA-256 direct hash | Matches canonical hashes from Workstream B2 / E2 | **VERIFIED** |

---

## 8. Classification of Findings

| Finding ID | Level | Summary | Impact |
| :--- | :--- | :--- | :--- |
| **ISSUE-BLK-01** | `NONE` | No blocking issues detected. | None. Execution and outputs are reproducible. |
| **ISSUE-MAJ-01** | `NONE` | No major integrity or scientific flaws. | None. Core hypotheses hold empirically. |
| **ISSUE-MIN-01** | `MINOR` | Table 4 Markdown report lists `0.0589` instead of `0.0571` for pooled baseline spatial std. | Cosmetic typographical error in Markdown report. |
| **ISSUE-MIN-02** | `MINOR` | Table header `Aggregate Mean $\pm$ SD` lacks explicit indication of $K=3$ seed units. | Requires clarifying footnote in paper draft. |

---

## 9. Recommendations for Future Reporting Workstreams

In accordance with safety constraints, no edits were made to `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` during this audit. When a subsequent paper drafting workstream (e.g., Workstream E4.3-B or Workstream D update) is authorized:

1. **Correct Table 4 Typo:**
   In `results/via3_experiments/xai_multiseed/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` (Table 4, row 126), update the pooled baseline row from:
   ```markdown
   | *Baseline Pooled* | *All* | *150* | *3* | *2.0%* | *0.1294* | *0.0589* | *0.9800* |
   ```
   to:
   ```markdown
   | *Baseline Pooled* | *All* | *150* | *3* | *2.0%* | *0.1294* | *0.0571* | *0.9800* |
   ```
   (and footnote that the cross-seed standard deviation across the 3 seed means is $0.0615$).
2. **Standardize Table Footnotes:**
   Add explicit table notes to future LaTeX and Markdown tables:
   *"Values represent Mean $\pm$ SD across $K=3$ independent training seeds (Seeds 42, 100, 2024), evaluated on the frozen 50-layout diagnostic cohort."*

---

## 10. Final Audit Verdict

**OVERALL AUDIT VERDICT: PASS WITH QUALIFICATIONS**

The Workstream E4.2 multi-seed explainability artifacts are verified to be structurally sound, cryptographically intact, internally consistent, and free from blocking or major defects. All headline metrics and conclusions are reproducible from the underlying raw data.
