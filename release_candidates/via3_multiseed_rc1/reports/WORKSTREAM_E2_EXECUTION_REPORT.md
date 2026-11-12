# Workstream E2 — Safe Multi-Seed Runner Remediation and Controlled Execution Report

**Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3`  
**Executed Seeds:** Reference Seed `42` (audited, read-only), Exploration Seeds `100` and `2024`  
**Strategies:** `baseline` (unweighted) and `balanced_sampling` (50/50 mini-batch)  
**Execution Date:** 2026-10-22 / 2026-10-23  
**Auditor / Runner:** Independent Automated Research Integrity Auditor  

---

## 1. Executive Verdict

### **Verdict: SUCCESSFUL CONTROLLED EXECUTION**

The two critical blockers identified in [`WORKSTREAM_E1_PREFLIGHT.md`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/WORKSTREAM_E1_PREFLIGHT.md) (`BLOCKER-01` strategy dispatch and `BLOCKER-02` Seed 42 immutability & overwrite guards), along with `BLOCKER-03` (path collision risk), have been fully remediated in [`scripts/run_via3_multiseed.py`](file:///home/peskybird/Projects/Ai_Ml_DA/scripts/run_via3_multiseed.py).

Sequential multi-seed GPU training for Seeds `100` and `2024` was executed on the NVIDIA GeForce RTX 3050 Laptop GPU under strictly isolated namespaces:
- **Baseline (Unweighted):** Seeds `42`, `100`, `2024` completed.
- **Balanced Sampling:** Seeds `42`, `100`, `2024` completed.
- **Reference Seed 42:** Verified by SHA-256 and imported without retraining.
- **Protected Paths:** All coursework artifacts (`docs/DA2_Group18_Report.docx`, `latex_report/`) and canonical Workstream D drafts (`draft_reports/via3/`) remained strictly untouched.
- **Test Integrity:** All 70 repository tests in `pytest tests/` pass cleanly.

---

## 2. Remediation of Blockers

### 2.1 Strategy Dispatch (`BLOCKER-01`)
- **Defect:** Line 138 of `scripts/run_via3_multiseed.py` originally hardcoded `run_baseline`, failing to execute `balanced_sampling` or `class_weighted`.
- **Fix:** Refactored `run_multiseed_orchestration` to validate against `SUPPORTED_STRATEGIES = ["baseline", "balanced_sampling", "class_weighted"]` and dynamically invoke `train_single_strategy` from `scripts/run_via3_imbalance_experiments.py`.
- **Verification:** Tested via `test_multiseed_strategy_validation` and confirmed by successful end-to-end training of both `baseline` and `balanced_sampling`.

### 2.2 Seed 42 Reference Bypass & Overwrite Guards (`BLOCKER-02`)
- **Defect:** Calling the script with `--seeds 42 100 2024` would re-train Seed 42 from scratch, overwriting audited checkpoints.
- **Fix:** Implemented `load_reference_seed42`:
  - When `seed == 42`, training is strictly bypassed.
  - Checkpoint SHA-256 checksums are verified against `KNOWN_REFERENCE_HASHES` (`97a9cb8b...` or `b9979e8d...` for baseline, `551a861f...` for balanced sampling).
  - Copies reference prediction CSVs (`clean_test`, `val`, `official_test`) and results JSON to the namespaced seed directory.
  - Original reference artifacts in `results/via3_experiments/imbalance/` and `models/via3_imbalance/` remain read-only.
- **Verification:** Unit tests `test_multiseed_seed42_reference_loading` and `test_multiseed_seed42_hash_mismatch` pass.

### 2.3 Output Namespacing & Safe Resume (`BLOCKER-03`)
- **Defect:** Output paths originally wrote directly to `output_dir / f"seed_{s}"`, causing filename collisions between strategies.
- **Fix:** Updated destination paths to `results/via3_experiments/multiseed/{strategy}/seed_{seed}/` and checkpoints to `models/via3_multiseed/via3_{strategy}_seed{seed}.keras`.
- **Safe Resume Guard:** Implemented `is_seed_run_completed`. If all 3 prediction CSVs, the result JSON, and the non-empty checkpoint exist, training is skipped and existing results are loaded. This was empirically validated during execution when a server restart occurred: Seed 42 loaded from reference, Seed 100 was resumed safely without retraining, and Seed 2024 completed cleanly.

---

## 3. Test Suite Verification

Five new unit tests were added to [`tests/test_via3_workstreams.py`](file:///home/peskybird/Projects/Ai_Ml_DA/tests/test_via3_workstreams.py):
1. `test_multiseed_strategy_validation`: Rejects invalid strategy strings with `ValueError`.
2. `test_multiseed_seed42_reference_loading`: Confirms Seed 42 loads without retraining, copies files, and matches reference hashes.
3. `test_multiseed_seed42_hash_mismatch`: Confirms that corrupted checkpoint files trigger `RuntimeError`.
4. `test_multiseed_strategy_namespacing`: Confirms isolated directory generation for baseline and balanced sampling without collisions.
5. `test_multiseed_is_completed_guard`: Confirms file-existence, completion detection, and size validation.

**Full Test Suite Execution:**
```bash
./scripts/tf_gpu.sh -m pytest tests/
# Result: 70 passed in 26.50s (100% pass rate)
```

---

## 4. Hardware, Environment, and Checkpoint Manifest

### 4.1 Hardware Environment
- **GPU:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Bus ID `0000:01:00.0`, Driver 595.99.02, CUDA 13.2)
- **VRAM Utilization:** Peak memory per training run: ~4.17 GB / 6.00 GB (No OOM, sequential execution)
- **Epoch Training Duration:** Full training split (22,763 samples, 711 batches/epoch for baseline, 504 batches/epoch for balanced sampling) took ~42–49 seconds/epoch, averaging ~11.5 minutes per 15-epoch seed run.

### 4.2 Checkpoint Manifest (`models/via3_multiseed/`)

| Model Checkpoint | Strategy | Seed | File Size | SHA-256 Checksum | Origin |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `via3_baseline_seed42.keras` | Baseline | 42 | 230,689 B | `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887` | Audited Reference (Verified) |
| `via3_baseline_seed100.keras` | Baseline | 100 | 230,689 B | `f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455` | Trained in E2 |
| `via3_baseline_seed2024.keras` | Baseline | 2024 | 230,689 B | `9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d` | Trained in E2 |
| `via3_balanced_sampling_seed42.keras` | Balanced | 42 | 230,689 B | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` | Audited Reference (Verified) |
| `via3_balanced_sampling_seed100.keras` | Balanced | 100 | 230,689 B | `c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815` | Trained in E2 |
| `via3_balanced_sampling_seed2024.keras` | Balanced | 2024 | 230,689 B | `297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9` | Trained in E2 |

All 6 checkpoints have identical architecture (12,873 trainable parameters).

---

## 5. Multi-Seed Empirical Results (Clean Test Partition, $N=12,803$)

### 5.1 Baseline (Unweighted Binary Cross-Entropy)
Evaluated on Clean Test ($N=12,803$, 411 HS / 12,392 NHS) at validation-calibrated threshold $\tau^*$:

| Metric | Seed 42 (Ref) | Seed 100 | Seed 2024 | Mean $\pm$ Sample Std ($ddof=1$) | 95% Confidence Interval ($t=4.303$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PR-AUC** | 0.2518 | 0.2983 | 0.2745 | **0.2749 $\pm$ 0.0233** | $[0.2171, 0.3327]$ |
| **ROC-AUC** | 0.7328 | 0.7739 | 0.7728 | **0.7598 $\pm$ 0.0234** | $[0.7017, 0.8180]$ |
| **HS Recall** | 0.2214 (91/411) | 0.2993 (123/411) | 0.1946 (80/411) | **0.2384 $\pm$ 0.0544** | $[0.1034, 0.3735]$ |
| **HS Precision** | 0.3640 | 0.3315 | 0.5714 | **0.4223 $\pm$ 0.1301** | $[0.0990, 0.7456]$ |
| **HS F1-Score** | 0.2753 | 0.3146 | 0.2904 | **0.2934 $\pm$ 0.0198** | $[0.2443, 0.3426]$ |
| **Balanced Accuracy** | 0.6043 | 0.6396 | 0.5949 | **0.6129 $\pm$ 0.0236** | $[0.5543, 0.6715]$ |
| **Calibrated Threshold $\tau^*$** | 0.08 | 0.07 | 0.01 | **0.0533 $\pm$ 0.0379** | $[-0.0407, 0.1474]$ |
| **Best Val PR-AUC Epoch** | 8 | 15 | 15 | — | — |

*Summary File:* [`results/via3_experiments/multiseed/baseline/multiseed_baseline_summary.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/baseline/multiseed_baseline_summary.json)

---

### 5.2 Balanced Sampling (50/50 Mini-Batch Sampling)
Evaluated on Clean Test ($N=12,803$, 411 HS / 12,392 NHS) at validation-calibrated threshold $\tau^*$:

| Metric | Seed 42 (Ref) | Seed 100 | Seed 2024 | Mean $\pm$ Sample Std ($ddof=1$) | 95% Confidence Interval ($t=4.303$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PR-AUC** | 0.2619 | 0.2168 | 0.3120 | **0.2635 $\pm$ 0.0476** | $[0.1452, 0.3819]$ |
| **ROC-AUC** | 0.7840 | 0.7456 | 0.7985 | **0.7760 $\pm$ 0.0273** | $[0.7082, 0.8439]$ |
| **HS Recall** | 0.2603 (107/411) | 0.2798 (115/411) | 0.2798 (115/411) | **0.2733 $\pm$ 0.0112** | $[0.2454, 0.3012]$ |
| **HS Precision** | 0.4038 | 0.2544 | 0.4457 | **0.3680 $\pm$ 0.1006** | $[0.1182, 0.6178]$ |
| **HS F1-Score** | 0.3166 | 0.2665 | 0.3438 | **0.3090 $\pm$ 0.0392** | $[0.2116, 0.4063]$ |
| **Balanced Accuracy** | 0.6238 | 0.6263 | 0.6341 | **0.6281 $\pm$ 0.0054** | $[0.6147, 0.6415]$ |
| **Calibrated Threshold $\tau^*$** | 0.76 | 0.74 | 0.53 | **0.6767 $\pm$ 0.1274** | $[0.3601, 0.9932]$ |
| **Best Val PR-AUC Epoch** | 6 | 4 | 12 | — | — |

*Summary File:* [`results/via3_experiments/multiseed/balanced_sampling/multiseed_balanced_sampling_summary.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/balanced_sampling/multiseed_balanced_sampling_summary.json)

---

### 5.3 Within-Seed Paired Differences ($\Delta_S = \text{Balanced}_S - \text{Baseline}_S$)

| Metric | Seed 42 Difference | Seed 100 Difference | Seed 2024 Difference | Mean Difference $\bar{\Delta}$ |
| :--- | :---: | :---: | :---: | :---: |
| **$\Delta$ PR-AUC** | $+0.0101$ | $-0.0815$ | $+0.0375$ | **$-0.0114$** |
| **$\Delta$ ROC-AUC** | $+0.0512$ | $-0.0283$ | $+0.0257$ | **$+0.0162$** |
| **$\Delta$ HS Recall** | $+0.0389$ | $-0.0195$ | $+0.0852$ | **$+0.0349$** (+3.5 pp) |
| **$\Delta$ HS F1-Score** | $+0.0413$ | $-0.0481$ | $+0.0534$ | **$+0.0156$** |
| **$\Delta$ Balanced Accuracy** | $+0.0195$ | $-0.0133$ | $+0.0392$ | **$+0.0152$** |
| **$\Delta \tau^*$ (Operating Point)** | $+0.6800$ | $+0.6700$ | $+0.5200$ | **$+0.6233$** |

---

## 6. Key Scientific and Methodological Findings

1. **Seed Variance Refutes Unambiguous PR-AUC Superiority:**
   In Seed 42, Balanced Sampling held a slight PR-AUC lead ($0.2619$ vs $0.2518$). In Seed 2024, Balanced Sampling held an even larger lead ($0.3120$ vs $0.2745$). However, in Seed 100, Baseline attained an unusually high PR-AUC ($0.2983$ vs $0.2168$). Across all 3 seeds, the 95% confidence intervals overlap substantially ($[0.2171, 0.3327]$ for Baseline vs $[0.1452, 0.3819]$ for Balanced Sampling). Single-seed comparisons were indeed vulnerable to initialization variance, fully vindicating the audit qualifications from Workstream D.

2. **Balanced Sampling Stabilizes Minority Class Recall:**
   While PR-AUC fluctuates across seeds, Balanced Sampling exhibits remarkable stability in Hotspot Recall:
   - Baseline HS Recall: $0.2384 \pm 0.0544$ (range $0.1946$ to $0.2993$).
   - Balanced Sampling HS Recall: $0.2733 \pm 0.0112$ (range $0.2603$ to $0.2798$).
   Balanced Sampling achieves a **~5x reduction in recall standard deviation** across seeds ($s=0.0112$ vs $s=0.0544$) while improving average recall by $+3.5$ percentage points.

3. **Operating Threshold Stability:**
   The unweighted baseline remains heavily biased toward the negative class, requiring extreme threshold down-shifting ($\bar{\tau}^* = 0.0533$, reaching as low as $\tau^* = 0.01$ in Seed 2024) to capture hotspots. In contrast, Balanced Sampling operates comfortably in the range $\tau^* \in [0.53, 0.76]$ ($\bar{\tau}^* = 0.6767$).

4. **Balanced Accuracy Superiority Persists on Average:**
   Balanced sampling achieves a higher mean Balanced Accuracy ($0.6281 \pm 0.0054$) with tighter across-seed bounds ($[0.6147, 0.6415]$) than Baseline ($0.6129 \pm 0.0236$, bounds $[0.5543, 0.6715]$).

---

## 7. Protected Paths and Audit Invariant Verification

| Protected Target | Expected State | Actual State | Verification Source |
| :--- | :--- | :--- | :---: |
| `docs/DA2_Group18_Report.docx` | Untouched | Untouched | Checksum / Git Status |
| `latex_report/` | Untouched | Untouched | Git Status |
| `draft_reports/via3/draft_via3_experimental_report.md` | SHA-256 `bf2916e4...` | SHA-256 `bf2916e4...` | Checksum Verified |
| `draft_reports/via3/draft_via3_table_summary.tex` | SHA-256 `9d8369dc...` | SHA-256 `9d8369dc...` | Checksum Verified |
| `models/via3_imbalance/via3_balanced_sampling_seed42.keras` | SHA-256 `551a861f...` | SHA-256 `551a861f...` | Checksum Verified |
| `models/via3_imbalance/via3_baseline_seed42.keras` | SHA-256 `97a9cb8b...` | SHA-256 `97a9cb8b...` | Checksum Verified |
| `results/via3_experiments/imbalance/imbalance_comparison_summary.json` | SHA-256 `7040963a...` | SHA-256 `7040963a...` | Checksum Verified |
| `results/dataset_audit/` (All 16 manifests) | 16 manifests valid | 16 manifests valid | `verify_manifest_integrity` |

---

## 8. Summary of Created Workstream E2 Artifacts

1. **Runner Script:** [`scripts/run_via3_multiseed.py`](file:///home/peskybird/Projects/Ai_Ml_DA/scripts/run_via3_multiseed.py)
2. **Focused Tests:** Added 5 unit tests in [`tests/test_via3_workstreams.py`](file:///home/peskybird/Projects/Ai_Ml_DA/tests/test_via3_workstreams.py)
3. **Checkpoints:**
   - [`models/via3_multiseed/via3_baseline_seed100.keras`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/via3_baseline_seed100.keras)
   - [`models/via3_multiseed/via3_baseline_seed2024.keras`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/via3_baseline_seed2024.keras)
   - [`models/via3_multiseed/via3_baseline_seed42.keras`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/via3_baseline_seed42.keras)
   - [`models/via3_multiseed/via3_balanced_sampling_seed100.keras`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/via3_balanced_sampling_seed100.keras)
   - [`models/via3_multiseed/via3_balanced_sampling_seed2024.keras`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/via3_balanced_sampling_seed2024.keras)
   - [`models/via3_multiseed/via3_balanced_sampling_seed42.keras`](file:///home/peskybird/Projects/Ai_Ml_DA/models/via3_multiseed/via3_balanced_sampling_seed42.keras)
4. **Per-Seed Prediction CSVs (18 files across 6 seed directories):**
   - `results/via3_experiments/multiseed/baseline/seed_{42,100,2024}/`
   - `results/via3_experiments/multiseed/balanced_sampling/seed_{42,100,2024}/`
5. **Summaries:**
   - [`results/via3_experiments/multiseed/baseline/multiseed_baseline_summary.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/baseline/multiseed_baseline_summary.json)
   - [`results/via3_experiments/multiseed/balanced_sampling/multiseed_balanced_sampling_summary.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/balanced_sampling/multiseed_balanced_sampling_summary.json)
6. **Execution Report:** [`results/via3_experiments/multiseed/WORKSTREAM_E2_EXECUTION_REPORT.md`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/WORKSTREAM_E2_EXECUTION_REPORT.md)
