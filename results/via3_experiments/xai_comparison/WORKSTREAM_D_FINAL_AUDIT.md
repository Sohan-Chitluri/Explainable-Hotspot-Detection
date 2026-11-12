# Workstream D — Final Statistical Integrity, Reproducibility, and Cross-Artifact Audit

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Audited Artifacts:**
- Report Generator: `scripts/generate_paper_reports.py`
- Test Harness: `tests/test_via3_workstreams.py`
- Markdown Synthesis: `draft_reports/via3/draft_via3_experimental_report.md`
- LaTeX Tables: `draft_reports/via3/draft_via3_table_summary.tex`
- Upstream Summaries:
  - `results/via3_experiments/imbalance/imbalance_comparison_summary.json`
  - `results/via3_experiments/xai_comparison/xai_comparison_summary.json`
  - `results/via3_experiments/threshold_analysis/via3_baseline_threshold_analysis.json`
  - `results/via3_experiments/xai_comparison/XAI_METHODOLOGY_AUDIT.md`
- Raw Prediction & Metric Records:
  - `results/via3_experiments/imbalance/baseline_clean_test_predictions.csv`
  - `results/via3_experiments/imbalance/balanced_sampling_clean_test_predictions.csv`
  - `results/via3_experiments/imbalance/class_weighted_clean_test_predictions.csv`
  - `results/via3_experiments/xai_comparison/faithfulness_metrics.csv`
  - `results/via3_experiments/xai_comparison/stability_metrics.csv`
  - `results/via3_experiments/xai_comparison/collapse_diagnostics.csv`
  - `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json`

**Audit Date:** 2026-10-22  
**Auditor:** Independent Automated Research Integrity & Statistical Auditor  

---

## 1. Executive Verdict

### **Verdict: PASS WITH QUALIFICATIONS**

The Workstream D reporting pipeline successfully transitions from mock/synthetic fixtures to consuming verified empirical data from Workstreams B1, B2, and C. Every numerical cell in Table 1 (Classification Performance on $N=12,803$), Table 2 (Paired XAI Faithfulness & Stability on $N=50$), and Table 3 (Representation Collapse & Spatial Std Dev) was independently verified and recomputed from raw prediction and attribution files.

The verdict is qualified based on five primary audit findings:
1. **Unadjusted Nominal $p$-Values in Tables:** The reported Wilcoxon signed-rank and paired $t$-test $p$-values in Table 2 are unadjusted nominal values across 17 hypothesis tests. While primary effects (Grad-CAM, LayerCAM, Saliency, Occlusion insertion; Grad-CAM stability) remain significant under Holm-Bonferroni correction ($p_{\text{adj}} < 0.005$), secondary comparisons (Occlusion stability $p=0.0196$, Random baseline insertion $p=0.0149$, Random deletion $p=0.0242$) lose statistical significance at the family-wise $\alpha = 0.05$ level. Table headers and captions must explicitly designate these as unadjusted nominal $p$-values.
2. **Random-Baseline Score Disparity Explained:** The observed difference in Random Baseline Insertion AUC ($0.4183$ baseline vs. $0.5135$ balanced sampling) is a mathematical artifact of the models' disparate baseline probability regimes (mean prediction probability on the cohort is $0.1795$ for baseline vs. $0.5744$ for balanced sampling), rather than an explanation quality difference. Random baseline performance should be presented as an unguided control, not as an inter-model comparative metric.
3. **LaTeX String Slicing Truncation Discrepancy:** In `draft_via3_table_summary.tex` (Table 3), the spatial standard deviation dispersion is formatted using string slicing `std_s[:5]`, which truncates `0.0229` to `0.022`, `0.0265` to `0.026`, and `0.0246` to `0.024`, creating a minor formatting mismatch with the Markdown report (`0.0229`, `0.0265`, `0.0246`).
4. **Multi-Seed Variance Status:** Multi-seed study across seeds `[42, 100, 2024]` remains in verified dry-run standby. The report correctly labels Table 2 entries as reflecting the audited Seed 42 reference models; this distinction must be preserved in paper drafts.
5. **Codification of Audit Qualifications:** Section 6 of `draft_via3_experimental_report.md` correctly codifies the five mandatory qualifications from `XAI_METHODOLOGY_AUDIT.md` (no causal sufficiency claims, OOD blank-canvas rebound, probability baseline asymmetry, diagnostic cohort scope, parameter-only randomization).

**No empirical re-runs or GPU workloads are required.** The outputs are sound for internal review and pre-print drafting once the reporting-level qualifications are noted.

---

## 2. Scope and Limitations

### 2.1 What Was Executed and Independently Recomputed
- Recomputed SHA-256 checksums for all 17 primary code, report, summary, and model checkpoint artifacts.
- Reconstructed confusion matrices (TP, FP, TN, FN), precision, recall, F1, balanced accuracy, PR-AUC, and ROC-AUC for all 5 rows of Table 1 directly from the 12,803 raw test prediction records (`clean_test_predictions.csv`).
- Recomputed validation threshold search curves (`select_threshold_from_val`) on 2,845 validation records to verify that $\tau^*_{\text{base}} = 0.08$, $\tau^*_{\text{bal}} = 0.76$, and $\tau^*_{\text{cw}} = 0.50$ were derived strictly from validation data with zero test leakage.
- Recomputed paired Wilcoxon signed-rank tests and paired Student's $t$-tests across all $N=50$ layout samples for 6 attribution methods across insertion AUC, deletion AUC, and stability $\rho$.
- Conducted full Holm-Bonferroni step-down multiplicity correction across all 17 paired hypothesis tests.
- Recomputed mean spatial standard deviations, sample standard deviations, and flat-CAM counts across all 150 layout rows in `collapse_diagnostics.csv`.
- Cross-verified every cell between `draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`.

### 2.2 What Was Inspected
- Code paths in `scripts/generate_paper_reports.py`, `src/experiment_contract.py`, `src/results_schema.py`, and `tests/test_via3_workstreams.py`.
- Adebayo model parameter randomization records and code protocol in `scripts/evaluate_via3_xai.py`.
- Workstream B1, B2, and C provenance manifests.

### 2.3 Limitations & Unavailable Data
- Multi-seed training runs (seeds 100 and 2024) were intentionally not executed to comply with safety constraints; only the audited Seed 42 empirical models exist.
- Per-sample Adebayo correlation records for the 5 individual samples exist as an aggregate scalar ($\bar{\rho} = -0.0486$) in `xai_comparison_summary.json` rather than a separate CSV file.

---

## 3. Artifact and Provenance Matrix

| Artifact Path | Source / Generating Script | Check Performed | Outcome | Evidence (SHA-256 / Checksum) |
| :--- | :--- | :---: | :---: | :--- |
| `scripts/generate_paper_reports.py` | Workstream D Pipeline | Static Code Audit & Execution | **PASS** | `aa719ea5f009bd58411a389384732bcb4eac165148d7e4010e66d964eb92c5e7` |
| `tests/test_via3_workstreams.py` | Pytest Harness | Test Execution (8/8 Passed) | **PASS** | `68d3e2fb89129e7f82a4d2f8650bb41ac7c8ccc849b066a3a66b7acb18b22511` |
| `draft_via3_experimental_report.md` | `scripts/generate_paper_reports.py` | Markdown Syntax & Cell Audit | **PASS** | `115aa5804c65b1352d6469ec5fac14129dd6ff2ba4eeccdc808025fcd46d84fb` |
| `draft_via3_table_summary.tex` | `scripts/generate_paper_reports.py` | LaTeX Syntax & Cell Audit | **PASS W/ QUAL** | `e03fb0ab5f7c94fc576967232ed734ecee9b3af32d98309272ce064e35b911f0` |
| `imbalance_comparison_summary.json` | Workstream B2 Runner | JSON Schema & Value Audit | **PASS** | `7040963af5a7810fd1a16a5b12d873456ff6f265aec984efe27021793fba3dfb` |
| `via3_baseline_threshold_analysis.json` | Workstream B1 Runner | Threshold Curve Verification | **PASS** | `601d11c1630d6a1027966e4582f0c9adaa90a88180871bc53e3025895eceb301` |
| `xai_comparison_summary.json` | Workstream C Runner | Summary Metric Verification | **PASS** | `af709518eda3d152abf02e622d288feb69195886d6bdf9d9dde00274d32a0279` |
| `XAI_METHODOLOGY_AUDIT.md` | Workstream C Audit | Content & Constraint Audit | **PASS** | `ae3965ff81c2c5a82816482ef1a82819cead65fc02dbd81d9185fd150c4edf44` |
| `xai_sample_cohort_manifest.json` | `scripts/evaluate_via3_xai.py` | Cohort Integrity Verification | **PASS** | `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb` |
| `faithfulness_metrics.csv` | `scripts/evaluate_via3_xai.py` | 900-Row Independent Recalculation | **PASS** | `3027effc303eb9e30e4be7c75b8f2fe8cdc0c8ba66bba32c732ab460b7aaabaf` |
| `stability_metrics.csv` | `scripts/evaluate_via3_xai.py` | 750-Row Independent Recalculation | **PASS** | `4bec8df0a115a2ef4455ae48126d92112449985291fe6a19bcd90113ff041440` |
| `collapse_diagnostics.csv` | `scripts/evaluate_via3_xai.py` | 150-Row Spatial Std Recalculation | **PASS** | `c4a845793d2e7b5b031cae10998c3ba75f9cfc6ae4a0881d267836c634ca69da` |
| `via3_baseline_seed42.keras` | Workstream B1 Training | Checkpoint Integrity Check | **PASS** | `b9979e8dbac5d31d8eac8bcd4156874ac7c2fcfa055200232c939922775ef4c3` |
| `via3_balanced_sampling_seed42.keras` | Workstream B2 Training | Checkpoint Integrity Check | **PASS** | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` |
| `via3_class_weighted_seed42.keras` | Workstream B2 Training | Checkpoint Integrity Check | **PASS** | `63168fd5d9fbd1208cc3b290ca7880919b468c043d249d7a5dca5e53f1567ceb` |

---

## 4. Classification Metric Verification (Table 1)

Every cell of Table 1 in `draft_via3_experimental_report.md` and `draft_via3_table_summary.tex` was independently reconstructed from the raw prediction CSV files using `src/experiment_contract.py:compute_contract_metrics`.

### 4.1 Confusion Matrix and Metric Verification Matrix ($N=12,803$; $411$ HS, $12,392$ NHS)

| Strategy | $\tau$ | TP | FP | TN | FN | Reported PR-AUC | Recalc PR-AUC | Reported ROC-AUC | Recalc ROC-AUC | Reported Recall | Recalc Recall | Reported Precision | Recalc Precision | Reported F1 | Recalc F1 | Reported BalAcc | Recalc BalAcc | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Unweighted)** | 0.50 | 38 | 6 | 12386 | 373 | 0.2518 | 0.2518 | 0.7328 | 0.7328 | 0.0925 | 0.0925 | 0.8636 | 0.8636 | 0.1670 | 0.1670 | 0.5460 | 0.5460 | **VERIFIED** |
| **Baseline (Calibrated)** | 0.08 | 91 | 159 | 12233 | 320 | 0.2518 | 0.2518 | 0.7328 | 0.7328 | 0.2214 | 0.2214 | 0.3640 | 0.3640 | 0.2753 | 0.2753 | 0.6043 | 0.6043 | **VERIFIED** |
| **Class-Weighted (Calibrated)** | 0.50 | 75 | 225 | 12167 | 336 | 0.1270 | 0.1270 | 0.6864 | 0.6864 | 0.1825 | 0.1825 | 0.2500 | 0.2500 | 0.2110 | 0.2110 | 0.5822 | 0.5822 | **VERIFIED** |
| **Balanced Sampling (Default)** | 0.50 | 127 | 341 | 12051 | 284 | 0.2619 | 0.2619 | 0.7840 | 0.7840 | 0.3090 | 0.3090 | 0.2714 | 0.2714 | 0.2890 | 0.2890 | 0.6407 | 0.6407 | **VERIFIED** |
| **Balanced Sampling (Calibrated)** | 0.76 | 107 | 158 | 12234 | 304 | 0.2619 | 0.2619 | 0.7840 | 0.7840 | 0.2603 | 0.2603 | 0.4038 | 0.4038 | 0.3166 | 0.3166 | 0.6238 | 0.6238 | **VERIFIED** |

### 4.2 Threshold Selection Integrity Check
- **Validation Dataset:** $N = 2,845$ ($91$ HS, $2,754$ NHS).
- Re-running `select_threshold_from_val` with `objective="max_f1"` yields:
  - Baseline validation maximum F1: $0.3262$ at $\tau^* = 0.08$.
  - Balanced sampling validation maximum F1: $0.3217$ at $\tau^* = 0.76$.
  - Class-weighted validation maximum F1: $0.2297$ at $\tau^* = 0.50$.
- **Zero Test Leakage Verified:** All calibrated thresholds were selected strictly from validation data and frozen prior to clean test evaluation. No test predictions or test ground-truth labels influenced threshold selection.

---

## 5. XAI Metric Verification (Table 2)

All 18 rows of Table 2 in `draft_via3_experimental_report.md` and `draft_via3_table_summary.tex` were independently recalculated from `faithfulness_metrics.csv` (900 rows) and `stability_metrics.csv` (750 rows).

### 5.1 Per-Method Recalculation Summary ($N=50$ Layouts)

| Method | Metric | Reported Baseline | Recomputed Baseline | Reported Balanced | Recomputed Balanced | Reported $\Delta$ | Recomputed $\Delta$ | Exact Match? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Grad-CAM** | Insertion AUC $\uparrow$ | 0.3393 | 0.3393 | 0.6118 | 0.6118 | +0.2726 | +0.2726 | **YES** |
| | Deletion AUC $\downarrow$ | 0.2744 | 0.2744 | 0.3734 | 0.3734 | +0.0991 | +0.0991 | **YES** |
| | Stability $\rho$ $\uparrow$ | 0.9328 | 0.9328 | 0.9710 | 0.9710 | +0.0383 | +0.0383 | **YES** |
| **Grad-CAM++** | Insertion AUC $\uparrow$ | 0.2854 | 0.2854 | 0.5166 | 0.5166 | +0.2312 | +0.2312 | **YES** |
| | Deletion AUC $\downarrow$ | 0.2714 | 0.2714 | 0.4686 | 0.4686 | +0.1972 | +0.1972 | **YES** |
| | Stability $\rho$ $\uparrow$ | 0.2760 | 0.2760 | 0.2453 | 0.2453 | -0.0307 | -0.0307 | **YES** |
| **LayerCAM** | Insertion AUC $\uparrow$ | 0.2322 | 0.2322 | 0.5528 | 0.5528 | +0.3206 | +0.3206 | **YES** |
| | Deletion AUC $\downarrow$ | 0.3703 | 0.3703 | 0.4506 | 0.4506 | +0.0803 | +0.0803 | **YES** |
| | Stability $\rho$ $\uparrow$ | 0.9688 | 0.9688 | 0.9638 | 0.9638 | -0.0049 | -0.0049 | **YES** |
| **Gradient Saliency** | Insertion AUC $\uparrow$ | 0.2007 | 0.2007 | 0.5758 | 0.5758 | +0.3752 | +0.3752 | **YES** |
| | Deletion AUC $\downarrow$ | 0.8061 | 0.8061 | 0.7315 | 0.7315 | -0.0746 | -0.0746 | **YES** |
| | Stability $\rho$ $\uparrow$ | 0.4313 | 0.4313 | 0.4526 | 0.4526 | +0.0213 | +0.0213 | **YES** |
| **Occlusion (32x32)**| Insertion AUC $\uparrow$ | 0.4027 | 0.4027 | 0.9208 | 0.9208 | +0.5182 | +0.5182 | **YES** |
| | Deletion AUC $\downarrow$ | 0.2444 | 0.2444 | 0.1470 | 0.1470 | -0.0974 | -0.0974 | **YES** |
| | Stability $\rho$ $\uparrow$ | 0.7045 | 0.7045 | 0.7444 | 0.7444 | +0.0400 | +0.0400 | **YES** |
| **Random Baseline** | Insertion AUC | 0.4183 | 0.4183 | 0.5135 | 0.5135 | +0.0952 | +0.0952 | **YES** |
| | Deletion AUC | 0.4170 | 0.4170 | 0.5119 | 0.5119 | +0.0949 | +0.0949 | **YES** |

### 5.2 Mechanistic Audit of the Random-Baseline Discrepancy
The audit specifically investigated why Random Baseline Insertion AUC is higher for Balanced Sampling ($0.5135$) than Baseline ($0.4183$), with a nominal $p = 0.0149$:
1. **Model Calibration & Average Probability Bias:** Across the 50 cohort layouts, the mean unperturbed hotspot prediction probability is $0.1795$ for Baseline versus $0.5744$ for Balanced Sampling.
2. **Intermediate Random Curve Regimes:** Because Balanced Sampling operates with higher baseline sensitivity, arbitrary random sub-samplings of layout pixels produce higher intermediate sigmoid activations than in the Baseline model.
3. **Audit Finding:** The difference between $0.4183$ and $0.5135$ does *not* reflect an explanation quality defect. It is a mathematical consequence of comparing raw, unnormalized prediction probabilities across models with different prediction baselines. Downstream manuscripts must explicitly caution against treating random-baseline differences as model superiority metrics.

---

## 6. Statistical Integrity and Multiplicity Audit

### 6.1 Multiplicity Analysis: Holm-Bonferroni Correction ($M = 17$ Tests)
The table presents 17 distinct hypothesis tests on paired layout samples. Evaluating family-wise error rate control under Holm-Bonferroni step-down correction ($\alpha = 0.05$):

| Rank $i$ | Comparison Test | Nominal $p$-value | Holm Threshold $\frac{\alpha}{M - i + 1}$ | Significant after Holm? | Effect Size / Direction ($\Delta$) |
| :-: | :--- | :---: | :---: | :---: | :---: |
| **1** | Grad-CAM Insertion AUC | $< 10^{-8}$ | $0.0029$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.2726$ |
| **2** | Grad-CAM Stability $\rho$ | $3.92 \times 10^{-8}$ | $0.0031$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.0383$ |
| **3** | Grad-CAM++ Insertion AUC | $7.55 \times 10^{-9}$ | $0.0033$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.2312$ |
| **4** | Grad-CAM++ Deletion AUC | $2.00 \times 10^{-7}$ | $0.0036$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.1972$ |
| **5** | LayerCAM Insertion AUC | $1.81 \times 10^{-10}$ | $0.0038$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.3206$ |
| **6** | Gradient Saliency Insertion AUC | $2.81 \times 10^{-9}$ | $0.0042$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.3752$ |
| **7** | Gradient Saliency Deletion AUC | $2.89 \times 10^{-7}$ | $0.0045$ | **YES** ($p_{\text{adj}} < 0.0001$) | $-0.0746$ |
| **8** | Gradient Saliency Stability $\rho$ | $9.06 \times 10^{-10}$ | $0.0050$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.0213$ |
| **9** | Occlusion Insertion AUC | $5.33 \times 10^{-15}$ | $0.0056$ | **YES** ($p_{\text{adj}} < 0.0001$) | $+0.5182$ |
| **10** | Occlusion Deletion AUC | $4.79 \times 10^{-6}$ | $0.0062$ | **YES** ($p_{\text{adj}} < 0.0001$) | $-0.0974$ |
| **11** | Grad-CAM Deletion AUC | $1.43 \times 10^{-3}$ | $0.0071$ | **YES** ($p_{\text{adj}} = 0.0100$) | $+0.0991$ |
| **12** | LayerCAM Deletion AUC | $2.28 \times 10^{-3}$ | $0.0083$ | **YES** ($p_{\text{adj}} = 0.0137$) | $+0.0803$ |
| **13** | Random Baseline Insertion AUC | $0.0149$ | $0.0100$ | **NO** (Not significant) | $+0.0952$ |
| **14** | Occlusion Stability $\rho$ | $0.0196$ | $0.0125$ | **NO** (Not significant) | $+0.0400$ |
| **15** | Random Baseline Deletion AUC | $0.0242$ | $0.0167$ | **NO** (Not significant) | $+0.0949$ |
| **16** | Grad-CAM++ Stability $\rho$ | $0.5210$ | $0.0250$ | **NO** (Not significant) | $-0.0307$ |
| **17** | LayerCAM Stability $\rho$ | $0.9771$ | $0.0500$ | **NO** (Not significant) | $-0.0049$ |

### 6.2 Key Statistical Findings
1. **Primary Claims are Multiplicity-Robust:** The primary insertion faithfulness improvements across all five attribution methods (Grad-CAM, Grad-CAM++, LayerCAM, Saliency, Occlusion) and Grad-CAM stability retain extreme significance ($p_{\text{adj}} < 0.0001$) after rigorous family-wise error control.
2. **Nominal Status Required:** In Table 2, the column header currently labels the $p$-values as `Wilcoxon $p$` without designating them as nominal/unadjusted. Secondary findings such as Occlusion stability ($p = 0.0196$) must not be reported as confirmed significant without noting that they fall above the adjusted family-wise threshold ($0.0125$).

---

## 7. Heatmap and Randomization Checks (Table 3 & Section 5)

### 7.1 Representation Collapse Diagnostics Verification ($N=50$)
Reconstructed directly from `collapse_diagnostics.csv`:
- **Baseline (Unweighted):** $0 / 50$ flat CAMs ($0.0\%$), Mean Spatial Std Dev: $0.1166 \pm 0.0229$, Max Spatial Std Dev: $0.2009$.
- **Balanced Sampling:** $0 / 50$ flat CAMs ($0.0\%$), Mean Spatial Std Dev: $0.1587 \pm 0.0265$, Max Spatial Std Dev: $0.2269$.
- **Class-Weighted (Negative Control):** $2 / 50$ flat CAMs ($4.0\%$), Mean Spatial Std Dev: $0.0865 \pm 0.0246$, Max Spatial Std Dev: $0.1307$.
- **Relative Spatial Dynamic Range Difference:** $+36.1\%$ ($p < 10^{-11}$).

### 7.2 Adebayo Sanity Check Protocol Review
- **Executed Protocol:** Full model parameter randomization re-initializing Conv2D and Dense layer weights with $\mathcal{N}(0, 0.05)$ Gaussian noise on 5 baseline layouts.
- **Observed Result:** Mean Spearman rank correlation with original heatmap is $\bar{\rho} = -0.0486$, passing the sanity threshold ($|\rho| < 0.20$).
- **Mandatory Disclosure Verified:** Section 6 of the report explicitly discloses that label randomization and cascading layer-wise ablations were omitted, satisfying the audit scope constraints.

---

## 8. Scientific Claim Review

| # | Manuscript Claim | Classification | Evidence & Context |
| :-: | :--- | :---: | :--- |
| **1** | **Balanced sampling achieves highest PR-AUC and F1 on clean test cohort** | **SUPPORTED** | Balanced sampling achieves PR-AUC $0.2619$ (vs. $0.2518$ baseline) and calibrated F1 $0.3166$ (vs. $0.2753$ baseline). |
| **2** | **Balanced sampling improves insertion-based attribution faithfulness** | **SUPPORTED WITH QUALIFICATIONS** | Insertion AUC improvements are large and statistically significant across all 5 methods ($p_{\text{adj}} < 0.0001$). *Qualification:* Unnormalized curves evaluate recovery from an elevated OOD blank canvas prediction ($P \approx 0.94 - 0.99$). |
| **3** | **Balanced sampling improves attribution stability** | **SUPPORTED WITH QUALIFICATIONS** | Confirmed for Grad-CAM ($\Delta = +0.0383, p_{\text{adj}} < 0.0001$) and Saliency ($\Delta = +0.0213, p_{\text{adj}} < 0.0001$). *Qualification:* Not observed for LayerCAM or Grad-CAM++; Occlusion stability ($p = 0.0196$) loses significance under Holm-Bonferroni correction. |
| **4** | **No representation collapse occurs on via-3** | **SUPPORTED** | $0.0\%$ flat CAMs on both baseline and balanced sampling across all 50 cohort layouts. |
| **5** | **Heatmaps isolate causally sufficient defect features** | **CONTRADICTED BY IMPLEMENTATION** | The perturbation operator zero-masks pixels and suffers from OOD boundary effects; it does not perform causal interventions. *Verified Status:* Replaced with *"attribution faithfulness under progressive pixel replacement"* in Section 6. |
| **6** | **Random baseline proves balanced sampling has higher attribution quality** | **CONTRADICTED BY IMPLEMENTATION** | Random baseline insertion AUC difference ($0.5135$ vs $0.4183$) is an artifact of the balanced model's higher average prediction score trajectory, not explanation quality. |
| **7** | **XAI performance differences generalize unconditionally to the whole population** | **INSUFFICIENT EVIDENCE** | The $N=50$ cohort is a stratified diagnostic sample designed to expose error modes, not an i.i.d. population sample. Claims must remain qualified to the diagnostic benchmark. |

---

## 9. Markdown and LaTeX Cross-Artifact Consistency

A comprehensive cell-by-cell cross-check was conducted between `draft_reports/via3/draft_via3_experimental_report.md` and `draft_reports/via3/draft_via3_table_summary.tex`.

### 9.1 Verification Findings
1. **Classification Performance (Table 1):** 100% exact numerical agreement across all 25 values (PR-AUC, ROC-AUC, Recall, Precision, F1, BalAcc, Decision $\tau$).
2. **Paired XAI Comparison (Table 2):** 100% exact numerical agreement across all 54 values (Baseline means, Balanced means, Paired differences, Wilcoxon $p$, Paired $t$ $p$).
3. **Table 3 Formatting Discrepancy Found:**
   - In Markdown Table 4: Dispersion is rendered as `0.1166 ± 0.0229`, `0.1587 ± 0.0265`, `0.0865 ± 0.0246`.
   - In LaTeX Table 3: Rendered as `$0.1166 \pm 0.022$`, `$0.1587 \pm 0.026$`, `$0.0865 \pm 0.024$`.
   - **Root Cause:** In `scripts/generate_paper_reports.py` line 256, the LaTeX generator used string slicing `std_s[:5]` (e.g., `"0.0229"[:5] -> "0.022"`), which truncated rather than rounding to 4 decimal places.
   - **Severity:** Low (presentation formatting only; does not affect means or statistical tests).

---

## 10. Findings Ranked by Severity

| Severity | Finding ID | Description | Remediation Type |
| :---: | :---: | :--- | :---: |
| **HIGH** | `FIND-01` | **Unadjusted Multiple Testing:** Table 2 lists nominal $p$-values without designating them as unadjusted across 17 tests. Secondary findings (Occlusion stability, Random baseline) do not survive Holm-Bonferroni correction. | Reporting / Caption Update |
| **HIGH** | `FIND-02` | **Random Baseline Interpretation:** Random baseline insertion AUC difference is driven by baseline prediction confidence asymmetry, not explanation quality. | Reporting / Text Note |
| **MEDIUM** | `FIND-03` | **Multi-Seed Clarification:** Multi-seed study is in verified dry-run standby; Table 2 entries must be explicitly labeled as single-seed reference (Seed 42) in external drafts. | Reporting / Table Note |
| **LOW** | `FIND-04` | **LaTeX String Slicing Truncation:** In Table 3, spatial std dev dispersion was truncated to 3 decimal places via `std_s[:5]` instead of standard rounding. | Code / LaTeX Formatting |

---

## 11. Recommended Remediation Plan

All remediations are strictly reporting and textual updates. **No model retraining, GPU execution, or experimental re-runs are required.**

1. **Table 2 Header & Caption Clarification:**
   - Update Table 2 in both Markdown and LaTeX to specify: *"Nominal Wilcoxon $p$-value (Unadjusted for multiple testing; primary insertion results remain significant at $p_{\text{adj}} < 0.001$ under Holm-Bonferroni correction)."*
2. **Random Baseline Explanatory Footnote:**
   - Add a footnote to Table 2 explaining that Random Baseline Insertion AUC differences reflect disparate average prediction baselines on the cohort rather than explanation quality differences.
3. **LaTeX Formatting Polish:**
   - In `scripts/generate_paper_reports.py`, format the spatial standard deviation as `\pm {float(std_s):.4f}` instead of slicing `std_s[:5]`.
4. **Preserve Repository Safeguards:**
   - Continue strictly preserving `docs/DA2_Group18_Report.docx` and `latex_report/`.

---

## 12. Final Readiness Decision

### **Decision: SUITABLE FOR INTERNAL REVIEW AND PREPRINT DRAFTING**

The experimental outputs and statistical tables synthesized by Workstream D accurately represent the underlying data. All code and artifact provenance chains are intact, zero test leakage occurred in threshold calibration, and the primary conclusions regarding Balanced Sampling superiority on clean-test PR-AUC and XAI insertion faithfulness are statistically robust and multiplicity-proof. Incorporating the four textual remediations above will ensure full compliance with peer-review standards.
