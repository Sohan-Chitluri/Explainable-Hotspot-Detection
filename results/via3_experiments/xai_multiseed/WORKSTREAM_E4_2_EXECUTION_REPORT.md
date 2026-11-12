# Workstream E4.2 — Multi-Seed XAI Robustness Execution Report

**Date:** 2026-10-23  
**Status:** PASS  
**Execution Environment:** NixOS Linux 6.12.35, NVIDIA GeForce RTX 3050 6GB Laptop GPU (Driver: 595.99.02, CUDA: 13.2 / cuDNN: 8.9.4), TensorFlow 2.15.1, Python 3.11.16  
**Runner Script:** `scripts/run_via3_xai_multiseed.py`  
**Output Directory:** `results/via3_experiments/xai_multiseed/`  

---

## 1. Executive Summary

This report documents the controlled execution of **Workstream E4.2: Multi-Seed Explainability (XAI) Robustness Evaluation** on OpenHSDBench `via-3`.

Following the frozen protocol and preflight specification approved in `results/via3_experiments/multiseed/WORKSTREAM_E4_1_XAI_PREFLIGHT.md`, all **1,800 planned attribution evaluations** were executed sequentially across the six authorized checkpoints and the 50-layout diagnostic cohort without failure, truncation, or memory overflow.

### Key Execution Highlights:
- **Models Evaluated:** 6 checkpoints (Baseline Seeds 42, 100, 2024; Balanced Sampling Seeds 42, 100, 2024).
- **Cohort:** Exact 50-layout diagnostic cohort frozen in `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json` (SHA-256: `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`).
- **Attribution Methods:** 6 methods (Grad-CAM, Grad-CAM++, LayerCAM, Gradient Saliency, Occlusion Sensitivity [$32 \times 32$, stride $16 \times 16$, fill $0.0$], Random Baseline).
- **Faithfulness Evaluations:** 1,800 10-step progressive deletion and insertion curves ($6 \text{ models} \times 50 \text{ layouts} \times 6 \text{ methods}$).
- **Stability Trials:** 1,500 trials with additive Gaussian noise ($\sigma=0.05, T=3$ trials, Spearman rank correlation) across the 5 deterministic attribution methods.
- **Collapse Diagnostics:** 300 evaluations of CAM spatial standard deviation and Flat-CAM detection ($1 \text{ CAM evaluation} \times 50 \text{ layouts} \times 6 \text{ models}$).
- **Adebayo Randomization Checks:** 6 models evaluated on 5 curated layouts against randomized-weight CNNs.
- **Failures / Errors:** 0 failures, 0 dropped rows. Total suite duration: 617.20 seconds (~10.3 minutes).

---

## 2. Checkpoint Provenance & Verification

Prior to inference, all 6 model checkpoints were cryptographically verified against their canonical hashes recorded in Workstreams A, B2, and E2.

| Strategy | Seed | Checkpoint Path | Expected & Verified SHA-256 | Architecture Parameters | Preflight Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 42 | `models/via3_multiseed/via3_baseline_seed42.keras` | `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887` | 12,873 params | VERIFIED |
| **Baseline** | 100 | `models/via3_multiseed/via3_baseline_seed100.keras` | `f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455` | 12,873 params | VERIFIED |
| **Baseline** | 2024 | `models/via3_multiseed/via3_baseline_seed2024.keras` | `9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d` | 12,873 params | VERIFIED |
| **Balanced Sampling** | 42 | `models/via3_multiseed/via3_balanced_sampling_seed42.keras` | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` | 12,873 params | VERIFIED |
| **Balanced Sampling** | 100 | `models/via3_multiseed/via3_balanced_sampling_seed100.keras` | `c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815` | 12,873 params | VERIFIED |
| **Balanced Sampling** | 2024 | `models/via3_multiseed/via3_balanced_sampling_seed2024.keras` | `297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9` | 12,873 params | VERIFIED |

---

## 3. Diagnostic Cohort Verification

The diagnostic cohort used in this evaluation was loaded from `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json` and mirrored to `results/via3_experiments/xai_multiseed/cohort/xai_sample_cohort_manifest.json`.

- **Cohort Manifest SHA-256:** `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`
- **Total Diagnostic Layouts:** 50
- **Partition:** Test split of OpenHSDBench `via-3` (`/home/peskybird/Projects/data/openhsd/via-3/test/`)
- **Quadrant Representation:**
  - True Positives (TP): 15 layouts
  - False Negatives (FN): 10 layouts
  - True Negatives (TN): 15 layouts
  - False Positives (FP): 10 layouts
- **Ground Truth Distribution:** 25 Hotspot (label 1), 25 Non-Hotspot (label 0)
- **Image Integrity:** All 50 PNG layout files verified present, uncorrupted, and standardized to shape `(224, 224, 3)` with pixel values in `[0.0, 1.0]`.

---

## 4. Execution Timing & Hardware Telemetry

Sequential model-by-model execution was enforced to maintain safe GPU thermals and memory limits on the 6 GB laptop GPU.

| Checkpoint Name | Evaluated Cohort Size | Methods | Total Sub-Evals | Duration (s) | GPU Memory Allocated | Adebayo Check Status ($\bar{\rho}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_seed42` | 50 | 6 | 300 | 114.71s | <1.5 GB | PASSED ($\rho = 0.0000$) |
| `baseline_seed100` | 50 | 6 | 300 | 99.93s | <1.5 GB | PASSED ($\rho = -0.3321$) |
| `baseline_seed2024` | 50 | 6 | 300 | 100.28s | <1.5 GB | PASSED ($\rho = 0.0000$) |
| `balanced_sampling_seed42` | 50 | 6 | 300 | 101.02s | <1.5 GB | PASSED ($\rho = -0.1331$) |
| `balanced_sampling_seed100` | 50 | 6 | 300 | 96.82s | <1.5 GB | PASSED ($\rho = -0.0571$) |
| `balanced_sampling_seed2024` | 50 | 6 | 300 | 104.49s | <1.5 GB | PASSED ($\rho = +0.0553$) |
| **Total / Suite Summary** | **50** | **6** | **1,800** | **617.20s** | **Peak <1.5 GB** | **6 / 6 PASSED** |

---

## 5. Generated Artifact Accounting

All generated artifacts are strictly quarantined under `results/via3_experiments/xai_multiseed/`. No canonical Workstream D drafts, Workstream C outputs, or coursework artifacts were modified.

### Master Deliverables:

| Filename | Type | Rows / Entries | SHA-256 Hash | Integrity Notes |
| :--- | :--- | :--- | :--- | :--- |
| `master_faithfulness_metrics.csv` | CSV | 1,800 rows | `0b49d2c414e54a7965b4d6c6d09e008f41385e40ef5d4cbbe810ad3228519ad0` | Complete matrix: 6 models $\times$ 50 samples $\times$ 6 methods |
| `master_stability_metrics.csv` | CSV | 1,500 rows | `fabbaee2eaeef5c4ea3f51fd64a40332492f5e94d6bb78158b9a2288135124aa` | Complete matrix: 6 models $\times$ 50 samples $\times$ 5 methods |
| `master_collapse_diagnostics.csv` | CSV | 300 rows | `8db96f2daffb25c20ad36895468ead267c1e99a8b89ee112e9bb62892bd5bfc3` | Complete matrix: 6 models $\times$ 50 samples |
| `multiseed_xai_comparison_summary.json` | JSON | 29.8 KB | `7648d65b407ed6e54b14f455b2a761d999f390d8401b36bde2ad30d680cf95a5` | Full cross-seed aggregations and paired deltas |
| `reproducibility_manifest.json` | JSON | 2.0 KB | `75f28469ad9373cf58bb06cb38ae59d5f7b8faec7bf2987a0774a88ae8210373` | Provenance manifest with SHA-256 for all outputs |

### Subdirectory Structure:
```
results/via3_experiments/xai_multiseed/
├── cohort/
│   └── xai_sample_cohort_manifest.json
├── baseline/
│   ├── seed_42/
│   │   ├── faithfulness_metrics.csv (300 rows)
│   │   ├── stability_metrics.csv (250 rows)
│   │   ├── collapse_diagnostics.csv (50 rows)
│   │   └── xai_summary.json
│   ├── seed_100/
│   │   ├── faithfulness_metrics.csv (300 rows)
│   │   ├── stability_metrics.csv (250 rows)
│   │   ├── collapse_diagnostics.csv (50 rows)
│   │   └── xai_summary.json
│   └── seed_2024/
│       ├── faithfulness_metrics.csv (300 rows)
│       ├── stability_metrics.csv (250 rows)
│       ├── collapse_diagnostics.csv (50 rows)
│       └── xai_summary.json
├── balanced_sampling/
│   ├── seed_42/
│   │   ├── faithfulness_metrics.csv (300 rows)
│   │   ├── stability_metrics.csv (250 rows)
│   │   ├── collapse_diagnostics.csv (50 rows)
│   │   └── xai_summary.json
│   ├── seed_100/
│   │   ├── faithfulness_metrics.csv (300 rows)
│   │   ├── stability_metrics.csv (250 rows)
│   │   ├── collapse_diagnostics.csv (50 rows)
│   │   └── xai_summary.json
│   └── seed_2024/
│       ├── faithfulness_metrics.csv (300 rows)
│       ├── stability_metrics.csv (250 rows)
│       ├── collapse_diagnostics.csv (50 rows)
│       └── xai_summary.json
├── master_faithfulness_metrics.csv
├── master_stability_metrics.csv
├── master_collapse_diagnostics.csv
├── multiseed_xai_comparison_summary.json
├── reproducibility_manifest.json
├── WORKSTREAM_E4_2_EXECUTION_REPORT.md
└── WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md
```

---

## 6. Execution Safety & Boundary Compliance

1. **Retraining Prohibition:** No models were trained or fine-tuned. All checkpoints were loaded strictly in inference/eval mode.
2. **Threshold Isolation:** Classification decision thresholds were not altered or recalibrated during this XAI evaluation.
3. **Protected File Safeguard:** Confirmed that `docs/DA2_Group18_Report.docx`, `latex_report/`, `draft_reports/via3/`, and `results/via3_experiments/xai_comparison/` remain completely untouched.
4. **Independent Execution:** All outputs were isolated in a fresh, dedicated directory `results/via3_experiments/xai_multiseed/`.

---

## 7. Execution Verdict

**VERDICT: PASS**

The Workstream E4.2 GPU execution executed all 1,800 planned attribution evaluations, 1,500 stability trials, and 300 collapse diagnostic checks across the 6 authorized models with 100% completion rate. All output artifacts have been compiled, verified for schema compliance, and cryptographically hashed in the reproducibility manifest.
