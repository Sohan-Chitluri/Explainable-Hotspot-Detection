# Workstream E3 — Multi-Seed Statistical Audit and Aggregation Verification

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3`  
**Target Workstream:** Workstream E2 Multi-Seed Robustness Evaluation  
**Evaluated Seeds:** Reference Seed `42`, Exploration Seeds `100` and `2024`  
**Evaluated Strategies:** Baseline (Unweighted) vs Balanced Sampling (50/50 Mini-Batch)  
**Primary Report Audited:** [`results/via3_experiments/multiseed/WORKSTREAM_E2_EXECUTION_REPORT.md`](file:///home/peskybird/Projects/Ai_Ml_DA/results/via3_experiments/multiseed/WORKSTREAM_E2_EXECUTION_REPORT.md)  
**Audit Date:** 2026-10-23  
**Auditor:** Independent Automated Research Integrity Auditor  

---

## 1. Executive Verdict

### **Verdict: PASS WITH QUALIFICATIONS**

All empirical data, model checkpoints, per-sample prediction files, threshold calibrations, and summary statistics reported in Workstream E2 have been independently verified and reconstructed directly from raw disk artifacts. 

- **Data Integrity & Traceability:** All 18 prediction CSV files (covering 2 strategies $\times$ 3 seeds $\times$ 3 splits) exist, have valid non-NaN probability distributions, match audited partition sample counts exactly (Clean Test: 12,803 samples; Official Test: 12,905 samples; Validation: 2,846 samples), and possess complete row/column schemas.
- **Metric Reproducibility:** Every single point metric (PR-AUC, ROC-AUC, HS Recall, HS Precision, HS F1, Balanced Accuracy, Accuracy, TP, FP, TN, FN) across all 6 seed models was independently recalculated from raw prediction probabilities and ground-truth labels. The recalculated values match the reported metrics within numerical float precision ($< 0.0003$ maximum difference).
- **Audit Invariants & Protected Paths:** Reference Seed 42 checkpoints match original SHA-256 hashes (`97a9cb8b...` and `551a861f...`). Coursework paths (`docs/DA2_Group18_Report.docx`, `latex_report/`) and canonical Workstream D drafts (`draft_reports/via3/`) remained strictly untouched.

### Mandatory Methodological Qualifications Before Paper Incorporation
While the numerical pipeline passes audit without defects, downstream reporting in scientific manuscripts must incorporate four vital statistical qualifications:
1. **Refutation of Unambiguous PR-AUC Superiority:** In Seed 42, Balanced Sampling led Baseline in PR-AUC ($0.2619$ vs $0.2518$). However, across 3 seeds, Baseline achieved higher average PR-AUC ($0.2749 \pm 0.0233$) than Balanced Sampling ($0.2635 \pm 0.0476$), driven by Baseline Seed 100 ($0.2983$). The 95% confidence intervals overlap substantially ($[0.2171, 0.3327]$ vs $[0.1452, 0.3819]$). Claims of universal PR-AUC dominance by Balanced Sampling are contradicted by multi-seed data.
2. **Student's $t$ Confidence Interval Caveat:** The reported 95% confidence intervals rely on normal-theory Student's $t$ multipliers ($t_{\text{crit}} = 4.303$ for $df = 2$). Because bounded evaluation metrics and thresholds on $N=3$ seeds violate Gaussian assumptions, the resulting interval for Baseline threshold extends into negative values ($[-0.0407, 0.1474]$). In paper drafts, results should be reported as `mean ± SD` with individual seed points $[s_{42}, s_{100}, s_{2024}]$, rather than inferential normal CIs.
3. **Recall Stabilization as Primary Benefit:** Balanced Sampling's primary empirical advantage across seeds is **minority class sensitivity stabilization**: HS Recall standard deviation drops from $0.0544$ (Baseline) to $0.0112$ (Balanced Sampling), a ~5x variance reduction across seeds, while improving average recall from $23.8\%$ to $27.3\%$.
4. **Operating Threshold vs Probability Calibration:** The upward shift in operating thresholds ($\bar{\tau}^* = 0.6767$ for Balanced Sampling vs $0.0533$ for Baseline) reflects training under an artificial 50/50 prior, NOT true probability calibration. Uncalibrated posteriors must not be claimed as calibrated probabilities.

---

## 2. Artifact Inventory and Provenance (Audit A)

### 2.1 Model Checkpoint Inventory (`models/via3_multiseed/`)

| Model Checkpoint | Strategy | Seed | File Size | SHA-256 Checksum | Provenance / Verification Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `via3_baseline_seed42.keras` | Baseline | 42 | 230,689 B | `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887` | **Verified:** Identical to audited reference in `models/via3_imbalance/` |
| `via3_baseline_seed100.keras` | Baseline | 100 | 230,689 B | `f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455` | **Verified:** Newly trained in E2 (15 epochs) |
| `via3_baseline_seed2024.keras` | Baseline | 2024 | 230,689 B | `9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d` | **Verified:** Newly trained in E2 (15 epochs) |
| `via3_balanced_sampling_seed42.keras` | Balanced | 42 | 230,689 B | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` | **Verified:** Identical to audited reference in `models/via3_imbalance/` |
| `via3_balanced_sampling_seed100.keras` | Balanced | 100 | 230,689 B | `c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815` | **Verified:** Newly trained in E2 (15 epochs) |
| `via3_balanced_sampling_seed2024.keras` | Balanced | 2024 | 230,689 B | `297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9` | **Verified:** Newly trained in E2 (15 epochs) |

*Architecture Verification:* All 6 checkpoints were inspected via TensorFlow. Every checkpoint instantiates a custom CNN with exactly 12,873 trainable parameters.

---

### 2.2 Reconciled Prediction CSV Inventory (18 Total Files)

Every strategy $\times$ seed combination produced 3 CSV files under `results/via3_experiments/multiseed/{strategy}/seed_{seed}/`.  
All 18 files share the exact header schema: `sample_id,filename,ground_truth,hs_probability,predicted_label`.

| Strategy | Seed | Split | Total Lines | Header Lines | Data Rows | HS (1) | NHS (0) | Probability Range | NaNs |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 42 | Clean Test | 12,804 | 1 | 12,803 | 411 | 12,392 | $[0.0001, 0.9836]$ | 0 |
| **Baseline** | 42 | Official Test | 12,906 | 1 | 12,905 | 422 | 12,483 | $[0.0001, 0.9836]$ | 0 |
| **Baseline** | 42 | Validation | 2,847 | 1 | 2,846 | 87 | 2,759 | $[0.0002, 0.9639]$ | 0 |
| **Baseline** | 100 | Clean Test | 12,804 | 1 | 12,803 | 411 | 12,392 | $[0.0000, 0.9889]$ | 0 |
| **Baseline** | 100 | Official Test | 12,906 | 1 | 12,905 | 422 | 12,483 | $[0.0000, 0.9889]$ | 0 |
| **Baseline** | 100 | Validation | 2,847 | 1 | 2,846 | 87 | 2,759 | $[0.0000, 0.9838]$ | 0 |
| **Baseline** | 2024 | Clean Test | 12,804 | 1 | 12,803 | 411 | 12,392 | $[0.0000, 0.4687]$ | 0 |
| **Baseline** | 2024 | Official Test | 12,906 | 1 | 12,905 | 422 | 12,483 | $[0.0000, 0.5770]$ | 0 |
| **Baseline** | 2024 | Validation | 2,847 | 1 | 2,846 | 87 | 2,759 | $[0.0000, 0.3268]$ | 0 |
| **Balanced** | 42 | Clean Test | 12,804 | 1 | 12,803 | 411 | 12,392 | $[0.0000, 0.9998]$ | 0 |
| **Balanced** | 42 | Official Test | 12,906 | 1 | 12,905 | 422 | 12,483 | $[0.0000, 0.9998]$ | 0 |
| **Balanced** | 42 | Validation | 2,847 | 1 | 2,846 | 87 | 2,759 | $[0.0000, 0.9996]$ | 0 |
| **Balanced** | 100 | Clean Test | 12,804 | 1 | 12,803 | 411 | 12,392 | $[0.0001, 0.9974]$ | 0 |
| **Balanced** | 100 | Official Test | 12,906 | 1 | 12,905 | 422 | 12,483 | $[0.0001, 0.9974]$ | 0 |
| **Balanced** | 100 | Validation | 2,847 | 1 | 2,846 | 87 | 2,759 | $[0.0001, 0.9974]$ | 0 |
| **Balanced** | 2024 | Clean Test | 12,804 | 1 | 12,803 | 411 | 12,392 | $[0.0000, 0.9986]$ | 0 |
| **Balanced** | 2024 | Official Test | 12,906 | 1 | 12,905 | 422 | 12,483 | $[0.0000, 0.9986]$ | 0 |
| **Balanced** | 2024 | Validation | 2,847 | 1 | 2,846 | 87 | 2,759 | $[0.0000, 0.9975]$ | 0 |

*Confirmation:* Exactly 18 files reconciled. Total lines in each clean test file are $12,804 = 1 \text{ header} + 12,803 \text{ rows}$. Ground truth sums to exactly 411 HS and 12,392 NHS across all clean test files.

---

## 3. Independent Metric Recalculation (Audit B)

Using the raw prediction probabilities $p_i$ and true labels $y_i \in \{0, 1\}$ from the 18 CSV files, metrics were independently recalculated from first principles without relying on stored JSON records.

### 3.1 Clean Test Metrics at Validation-Calibrated Operating Threshold $\tau^*$

| Strategy | Seed | Threshold $\tau^*$ | Metric | Stored JSON Value | Independently Recalculated | Discrepancy |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Baseline** | 42 | 0.08 | PR-AUC | 0.251826 | 0.251826 | $< 10^{-6}$ |
| | | | ROC-AUC | 0.732811 | 0.732811 | $< 10^{-6}$ |
| | | | HS Recall | 0.221411 (91/411) | 0.221411 (91/411) | $0.0$ |
| | | | HS Precision | 0.364000 (91/250) | 0.364000 (91/250) | $0.0$ |
| | | | HS F1-Score | 0.275340 | 0.275340 | $< 10^{-6}$ |
| | | | Balanced Acc | 0.604290 | 0.604290 | $< 10^{-6}$ |
| | | | TP / FP / TN / FN | 91 / 159 / 12233 / 320 | 91 / 159 / 12233 / 320 | Exact Match |
| **Baseline** | 100 | 0.07 | PR-AUC | 0.298336 | 0.298382 | $4.6 \times 10^{-5}$ |
| | | | ROC-AUC | 0.773918 | 0.773918 | $< 10^{-6}$ |
| | | | HS Recall | 0.299270 (123/411) | 0.299270 (123/411) | $0.0$ |
| | | | HS Precision | 0.331536 (123/371) | 0.331536 (123/371) | $0.0$ |
| | | | HS F1-Score | 0.314578 | 0.314578 | $< 10^{-6}$ |
| | | | Balanced Acc | 0.639629 | 0.639629 | $< 10^{-6}$ |
| | | | TP / FP / TN / FN | 123 / 248 / 12144 / 288 | 123 / 248 / 12144 / 288 | Exact Match |
| **Baseline** | 2024 | 0.01 | PR-AUC | 0.274547 | 0.274548 | $< 10^{-6}$ |
| | | | ROC-AUC | 0.772754 | 0.772664 | $9.0 \times 10^{-5}$ |
| | | | HS Recall | 0.194647 (80/411) | 0.194647 (80/411) | $0.0$ |
| | | | HS Precision | 0.571429 (80/140) | 0.571429 (80/140) | $0.0$ |
| | | | HS F1-Score | 0.290381 | 0.290381 | $< 10^{-6}$ |
| | | | Balanced Acc | 0.594903 | 0.594903 | $< 10^{-6}$ |
| | | | TP / FP / TN / FN | 80 / 60 / 12332 / 331 | 80 / 60 / 12332 / 331 | Exact Match |
| **Balanced** | 42 | 0.76 | PR-AUC | 0.261857 | 0.261858 | $< 10^{-6}$ |
| | | | ROC-AUC | 0.783983 | 0.783983 | $< 10^{-6}$ |
| | | | HS Recall | 0.260341 (107/411) | 0.260341 (107/411) | $0.0$ |
| | | | HS Precision | 0.403774 (107/265) | 0.403774 (107/265) | $0.0$ |
| | | | HS F1-Score | 0.316568 | 0.316568 | $< 10^{-6}$ |
| | | | Balanced Acc | 0.623795 | 0.623795 | $< 10^{-6}$ |
| | | | TP / FP / TN / FN | 107 / 158 / 12234 / 304 | 107 / 158 / 12234 / 304 | Exact Match |
| **Balanced** | 100 | 0.74 | PR-AUC | 0.216772 | 0.216772 | $< 10^{-6}$ |
| | | | ROC-AUC | 0.745609 | 0.745609 | $< 10^{-6}$ |
| | | | HS Recall | 0.279805 (115/411) | 0.279805 (115/411) | $0.0$ |
| | | | HS Precision | 0.254425 (115/452) | 0.254425 (115/452) | $0.0$ |
| | | | HS F1-Score | 0.266512 | 0.266512 | $< 10^{-6}$ |
| | | | Balanced Acc | 0.626305 | 0.626305 | $< 10^{-6}$ |
| | | | TP / FP / TN / FN | 115 / 337 / 12055 / 296 | 115 / 337 / 12055 / 296 | Exact Match |
| **Balanced** | 2024 | 0.53 | PR-AUC | 0.311993 | 0.311993 | $< 10^{-6}$ |
| | | | ROC-AUC | 0.798479 | 0.798220 | $2.6 \times 10^{-4}$ |
| | | | HS Recall | 0.279805 (115/411) | 0.279805 (115/411) | $0.0$ |
| | | | HS Precision | 0.445736 (115/258) | 0.445736 (115/258) | $0.0$ |
| | | | HS F1-Score | 0.343797 | 0.343797 | $< 10^{-6}$ |
| | | | Balanced Acc | 0.634133 | 0.634133 | $< 10^{-6}$ |
| | | | TP / FP / TN / FN | 115 / 143 / 12249 / 296 | 115 / 143 / 12249 / 296 | Exact Match |

*Verdict:* Recalculation completely reproduces stored metrics. Confusion matrix cells match perfectly across all 6 models.

---

### 3.2 Clean Test Metrics at Default Uncalibrated Threshold $\tau = 0.50$

To evaluate model readiness without threshold calibration, metrics were independently calculated at standard $\tau = 0.50$:

| Strategy | Seed | Threshold $\tau$ | HS Recall | HS Precision | HS F1-Score | Balanced Acc | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 42 | 0.50 | 0.0925 (38/411) | 0.8636 | 0.1670 | 0.5460 | 38 / 6 / 12386 / 373 |
| **Baseline** | 100 | 0.50 | 0.1095 (45/411) | 0.8491 | 0.1940 | 0.5544 | 45 / 8 / 12384 / 366 |
| **Baseline** | 2024 | 0.50 | **0.0000 (0/411)** | **0.0000** | **0.0000** | **0.5000** | **0 / 0 / 12392 / 411** |
| **Balanced** | 42 | 0.50 | 0.3090 (127/411) | 0.2714 | 0.2890 | 0.6407 | 127 / 341 / 12051 / 284 |
| **Balanced** | 100 | 0.50 | 0.3869 (159/411) | 0.1559 | 0.2222 | 0.6587 | 159 / 861 / 11531 / 252 |
| **Balanced** | 2024 | 0.50 | 0.2798 (115/411) | 0.4291 | 0.3387 | 0.6337 | 115 / 153 / 12239 / 296 |

**Crucial Finding (Complete Class Collapse in Baseline Seed 2024 at $\tau = 0.50$):**  
In Baseline Seed 2024, the maximum probability predicted on any clean test sample was $0.4687$. Consequently, at default threshold $\tau = 0.50$, the model flagged **0 hotspots out of 411** ($\text{Recall} = 0.0\%$, $\text{Precision} = 0.0\%$, $\text{F1} = 0.0\%$, $\text{TP} = 0$). This demonstrates that without explicit validation threshold tuning, unweighted baseline models can suffer catastrophic failure at standard operational thresholds. Balanced Sampling avoids this failure mode across all seeds.

---

## 4. Aggregate Mean, Standard Deviation, and Interval Audit (Audit C)

### 4.1 Comparison of Independently Recomputed Aggregates vs Stored Summaries

The sample standard deviation is computed with Bessel's correction ($ddof=1$):
$$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2}, \quad n=3$$

| Strategy | Metric | Independent Mean | Stored Mean | Independent $s$ ($ddof=1$) | Stored $s$ | Stored 95% Interval | Independent Formula Recomputation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **PR-AUC** | 0.2749 | 0.2749 | 0.0233 | 0.0233 | $[0.2171, 0.3327]$ | $[0.2171, 0.3328]$ |
| | **ROC-AUC** | 0.7598 | 0.7598 | 0.0234 | 0.0234 | $[0.7017, 0.8180]$ | $[0.7017, 0.8179]$ |
| | **HS Recall** | 0.2384 | 0.2384 | 0.0544 | 0.0544 | $[0.1034, 0.3735]$ | $[0.1034, 0.3735]$ |
| | **HS Precision** | 0.4223 | 0.4223 | 0.1301 | 0.1301 | $[0.0990, 0.7456]$ | $[0.0990, 0.7456]$ |
| | **HS F1-Score** | 0.2934 | 0.2934 | 0.0198 | 0.0198 | $[0.2443, 0.3426]$ | $[0.2443, 0.3426]$ |
| | **Balanced Acc** | 0.6129 | 0.6129 | 0.0236 | 0.0236 | $[0.5543, 0.6715]$ | $[0.5543, 0.6715]$ |
| | **Threshold $\tau^*$** | 0.0533 | 0.0533 | 0.0379 | 0.0379 | $[-0.0407, 0.1474]$ | $[-0.0407, 0.1474]$ |
| **Balanced** | **PR-AUC** | 0.2635 | 0.2635 | 0.0476 | 0.0476 | $[0.1452, 0.3819]$ | $[0.1452, 0.3818]$ |
| | **ROC-AUC** | 0.7759 | 0.7760 | 0.0272 | 0.0273 | $[0.7082, 0.8439]$ | $[0.7083, 0.8435]$ |
| | **HS Recall** | 0.2733 | 0.2733 | 0.0112 | 0.0112 | $[0.2454, 0.3012]$ | $[0.2454, 0.3012]$ |
| | **HS Precision** | 0.3680 | 0.3680 | 0.1006 | 0.1006 | $[0.1182, 0.6178]$ | $[0.1182, 0.6178]$ |
| | **HS F1-Score** | 0.3090 | 0.3090 | 0.0392 | 0.0392 | $[0.2116, 0.4063]$ | $[0.2116, 0.4063]$ |
| | **Balanced Acc** | 0.6281 | 0.6281 | 0.0054 | 0.0054 | $[0.6147, 0.6415]$ | $[0.6147, 0.6415]$ |
| | **Threshold $\tau^*$** | 0.6767 | 0.6767 | 0.1274 | 0.1274 | $[0.3601, 0.9932]$ | $[0.3601, 0.9932]$ |

---

### 4.2 Mathematical Evaluation of the "95% Confidence Interval"

1. **Formula Audit:**  
   The stored interval was reconstructed exactly as:
   $$\text{CI}_{95\%} = \bar{x} \pm t_{0.975, \, df=2} \times \frac{s}{\sqrt{3}} = \bar{x} \pm 4.303 \times \frac{s}{1.73205} = \bar{x} \pm 2.4843 \times s$$
2. **Support Boundary Violation:**  
   For the Baseline calibrated threshold ($\bar{\tau}^* = 0.0533, s = 0.0379$), the formula yields:
   $$0.0533 \pm 4.303 \times \frac{0.0379}{\sqrt{3}} = [-0.0407, 0.1474]$$
   The lower bound falls below zero ($-0.0407$), even though decision thresholds are strictly bounded by $\tau \in [0, 1]$.
3. **Statistical Validity:**  
   Student's $t$-intervals assume that the underlying population of seed outcomes is normally distributed. With only $N=3$ samples, metric bounded support $[0, 1]$, and significant skewness, calling this interval an exact "95% confidence interval" in scientific reporting is statistically inappropriate.
4. **Recommendation for Manuscript:**  
   Do not report normal-theory Student's $t$ confidence intervals for $N=3$. Instead, report:
   $$\text{Mean} \pm \text{SD} \quad [\text{Seed 42}, \text{Seed 100}, \text{Seed 2024}]$$
   This presents the exact empirical dispersion without making ungrounded distributional assumptions.

---

## 5. Paired Strategy Differences by Seed (Audit D)

For each seed $S \in \{42, 100, 2024\}$, the paired difference on the identical 12,803 test layouts was computed as:
$$\Delta_S = \text{Metric}_{\text{bal}, S} - \text{Metric}_{\text{base}, S}$$

| Metric | Seed 42 Difference | Seed 100 Difference | Seed 2024 Difference | Mean Paired $\bar{\Delta}$ | Sample SD $s_\Delta$ | Sign Consistency Across 3 Seeds |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PR-AUC** | $+0.0101$ | $-0.0815$ | $+0.0375$ | **$-0.0114$** | $0.0623$ | **Inconsistent** (2 positive, 1 negative) |
| **ROC-AUC** | $+0.0512$ | $-0.0283$ | $+0.0256$ | **$+0.0162$** | $0.0406$ | **Inconsistent** (2 positive, 1 negative) |
| **HS Recall** | $+0.0389$ | $-0.0195$ | $+0.0852$ | **$+0.0349$** | $0.0524$ | **Inconsistent** (2 positive, 1 negative) |
| **HS Precision** | $+0.0398$ | $-0.0771$ | $-0.1257$ | **$-0.0543$** | $0.0851$ | **Inconsistent** (1 positive, 2 negative) |
| **HS F1-Score** | $+0.0413$ | $-0.0481$ | $+0.0534$ | **$+0.0156$** | $0.0554$ | **Inconsistent** (2 positive, 1 negative) |
| **Balanced Accuracy** | $+0.0195$ | $-0.0133$ | $+0.0392$ | **$+0.0152$** | $0.0265$ | **Inconsistent** (2 positive, 1 negative) |
| **Threshold $\tau^*$** | $+0.6800$ | $+0.6700$ | $+0.5200$ | **$+0.6233$** | $0.0896$ | **Consistent** (3 positive, 0 negative) |

### Key Methodological Clarifications on Paired Comparisons
- **Arithmetic Mean Verification:** The E2 report's mean paired differences match the exact arithmetic mean of the per-seed differences: $\bar{\Delta} = \frac{1}{3}\sum_S \Delta_S$.
- **Independence Fallacy Warning:** The 12,803 test layout images must **not** be treated as independent degrees of freedom for training-seed claims. Testing $N=12,803$ samples under 3 training seeds yields exactly $N=3$ independent training replicates, not $N=38,409$.
- **Statistical Significance from 3 Seeds:** With $N=3$, non-parametric paired tests (such as Wilcoxon signed-rank) cannot achieve statistical significance at $\alpha = 0.05$ (minimum two-tailed $p$-value for $N=3$ is $p=0.25$). A paired $t$-test on $\Delta \text{PR-AUC}$ yields $t = -0.0114 / (0.0623 / \sqrt{3}) = -0.317, p = 0.78$. Therefore, no claim of statistically significant performance difference between strategies across seeds can be justified from 3 seeds.

---

## 6. Threshold Selection and Score-Scale Findings (Audit E)

### 6.1 Validation Selection Protocol Trace
The threshold selection protocol in `select_threshold_from_val` was audited:
1. **Search Grid:** Line 199 defines $\tau \in [0.01, 0.99]$ with step $0.01$ (99 candidate operating points).
2. **Objective Function:** Lines 224–229 maximize Hotspot F1 score on validation data: $\tau^* = \arg\max_\tau \text{F1}_{\text{val}}(\tau)$.
3. **Deterministic Tie-Breaking:** Ties are broken strictly by first encounter in ascending search order (`score > best_score`).
4. **Data Leakage Immunity:** Audit confirmed that `val_predictions.csv` was generated from partition `val` (2,846 samples). The test set ($N=12,803$) was never accessed during threshold search.

### 6.2 Score Scale vs Probability Calibration
- **Baseline Behavior:** Trained on 729 HS vs 22,034 NHS (class ratio ~1:30.2). The model outputs probabilities severely compressed into the lower end of the range. Calibrated thresholds are very low: $\tau^* \in \{0.08, 0.07, 0.01\}$.
- **Balanced Sampling Behavior:** Trained on 50/50 mini-batches (HS repeated with replacement). The network's effective prior during training is 0.50, shifting the continuous output distribution upward. Calibrated thresholds are substantially higher: $\tau^* \in \{0.76, 0.74, 0.53\}$.
- **Audit Warning on Score Scale:** Moving the operating threshold from $0.05$ to $0.68$ **does not constitute evidence of calibrated probabilities**. The balanced network's output is an uncalibrated posterior under an artificial training distribution. No Platt scaling, isotonic regression, or Brier score minimization was performed. Scientific reporting must explicitly distinguish an operational decision threshold from true Bayesian probability calibration.

---

## 7. Scientific Claim Audit (Audit F)

| Claim Statement | Audit Classification | Evidence & Rationale | Confidence |
| :--- | :---: | :--- | :---: |
| **"Balanced sampling improves Hotspot Recall"** | **SUPPORTED WITH QUALIFICATIONS** | Mean recall increased from $23.8\%$ to $27.3\%$ ($+3.5$ pp). Supported in 2 of 3 seeds (Seed 42: $+3.9$ pp; Seed 2024: $+8.5$ pp), but Seed 100 Baseline had slightly higher recall ($29.9\%$ vs $28.0\%$). | **High** |
| **"Balanced sampling reduces recall variability across seeds"** | **SUPPORTED** | Sample standard deviation for HS Recall dropped from $s=0.0544$ (Baseline) to $s=0.0112$ (Balanced Sampling), representing a ~5x variance reduction across seeds. | **High** |
| **"Balanced sampling improves PR-AUC"** | **CONTRADICTED BY DATA** | Across 3 seeds, Baseline achieved higher mean PR-AUC ($0.2749$) than Balanced Sampling ($0.2635$). In Seed 100, Baseline was substantially higher ($0.2983$ vs $0.2168$). CIs overlap completely ($[0.2171, 0.3327]$ vs $[0.1452, 0.3819]$). | **High** |
| **"Balanced sampling improves ROC-AUC"** | **SUPPORTED WITH QUALIFICATIONS** | Mean ROC-AUC increased marginally ($0.7760$ vs $0.7598$, $+0.0162$). Supported in Seeds 42 and 2024, but reversed in Seed 100. | **High** |
| **"Balanced sampling improves F1-Score"** | **SUPPORTED WITH QUALIFICATIONS** | Mean F1 increased from $0.2934$ to $0.3090$ ($+0.0156$). Seed 42 and 2024 showed gains ($+0.0413$ and $+0.0534$), but Seed 100 favored Baseline ($-0.0481$). | **High** |
| **"Balanced sampling operates at more natural/stable thresholds"** | **SUPPORTED WITH QUALIFICATIONS** | Thresholds remain safely above zero ($\tau^* \in [0.53, 0.76]$) and avoid class collapse at $\tau=0.50$, whereas Baseline collapses to 0 hotspots in Seed 2024 at $\tau=0.50$. However, threshold std across seeds is higher for Balanced ($0.1274$) than Baseline ($0.0379$). | **High** |
| **"Results establish broad multi-seed robustness / generalization"** | **INSUFFICIENT EVIDENCE** | $N=3$ seeds provide an initial sensitivity check against weight initialization, not a comprehensive characterization of population variance or out-of-distribution generalization. | **High** |

---

## 8. Discrepancies and Limitations (Audit G)

1. **Baseline Seed 2024 Extreme Low Threshold:**  
   In Baseline Seed 2024, maximum test probability was $0.4687$, and validation F1 was maximized at $\tau^* = 0.01$. At this operating point, only 80 hotspots were detected ($\text{Recall} = 19.5\%$, $\text{Precision} = 57.1\%$). This highlights that unweighted baseline training can produce highly compressed probability scales that require extreme threshold tuning.
2. **Confidence Interval Truncation Anomaly:**  
   The baseline threshold 95% CI $[-0.0407, 0.1474]$ crosses zero. While mathematically faithful to the unconstrained Student's $t$ formula, reporting negative probabilities or thresholds is scientifically unphysical.
3. **Small Sample Size ($N=3$):**  
   All inferential statistical tests across seeds have very low power. Qualitative observations (e.g., recall stabilization) are strong descriptively, but formal hypothesis testing across seeds cannot establish statistical significance at $N=3$.

---

## 9. Recommended Corrections Ranked by Severity

| Priority | Issue / Risk | Location | Remediation Plan |
| :---: | :--- | :--- | :--- |
| **HIGH** | Overclaiming PR-AUC superiority for Balanced Sampling. | Future Workstream E reporting / Paper Drafts | Explicitly state that multi-seed evaluation reveals overlapping PR-AUC distributions ($0.2749 \pm 0.0233$ for Baseline vs $0.2635 \pm 0.0476$ for Balanced Sampling), and that Seed 42 was not representative of universal PR-AUC dominance. |
| **MEDIUM** | Reporting normal-theory 95% CIs with $N=3$ and negative lower bounds. | Table captions and summary tables | Replace 95% CI columns with `mean ± SD` accompanied by individual seed values $[s_{42}, s_{100}, s_{2024}]$. |
| **MEDIUM** | Conflating operating threshold shift with probability calibration. | Narrative in methodology sections | Clarify that balanced batch sampling changes the empirical prior and shifts decision thresholds toward $\approx 0.68$, but does not constitute true posterior calibration. |
| **LOW** | Minor rounding differences ($< 0.0003$) in secondary ranking metrics. | Summary JSON files | Document that PR-AUC and ROC-AUC numerical integration step sizes cause sub-$10^{-4}$ variations between implementations. |

---

## 10. Release Decision for Paper Integration

### **Decision: CONDITIONALLY APPROVED FOR DRAFT REPORTING**

The multi-seed classification results from Workstream E2 are **empirically verified, reproducible from disk, and safe to incorporate into research drafts**, provided that the four mandatory qualifications in Section 1 and Section 9 are strictly integrated into the reporting text.

**Next Workstream Scope:**  
- Proceed to multi-seed XAI evaluation on the frozen 50 diagnostic layouts only when authorized.
- Update paper draft tables using `mean ± SD` and individual seed columns rather than normal-theory CIs.
