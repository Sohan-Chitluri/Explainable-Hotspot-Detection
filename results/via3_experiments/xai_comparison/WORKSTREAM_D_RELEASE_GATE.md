# Workstream D — Final Release Gate Report

**Target Package:** Via-3 Lithography Hotspot Classification & XAI Research Package  
**Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Audited Release Candidates:**
- Markdown Synthesis: `draft_reports/via3/draft_via3_experimental_report.md` (SHA-256: `bf2916e49ec27f6687fdb48a0ba51bfcdcdb608d9ca5e11924dc761960e1f89e`)
- LaTeX Table Bundle: `draft_reports/via3/draft_via3_table_summary.tex` (SHA-256: `9d8369dc9f31f3c39376738fb91b5758017a1993cf26050d92fe25816d5495d4`)
- Report Generator: `scripts/generate_paper_reports.py` (SHA-256: `999ec0c2c5f53aa681766af790d3f3370a0f5a96529eae881f6271fa175ede7e`)
- Release Verification Harness: `tests/test_via3_workstreams.py` (SHA-256: `c8485e4aa0a97ab57775f8136995207782b8445117c21d935968f0486fad7812`)

**Gate Date:** 2026-10-22  
**Auditor:** Independent Automated Research Integrity & Release Gate Auditor  

---

## 1. Executive Verdict

### **Verdict: PASS WITH QUALIFICATIONS**

The Workstream D research artifacts are internally consistent, statistically sound, and fully reproducible from primary empirical observations. All four audit remediations (`FIND-01` through `FIND-04`) were independently confirmed in both `draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`. Running the generator against an isolated temporary directory confirmed 100% byte-for-byte reproducibility of the release candidates without side effects.

### Summary of Authorizing Qualifications:
1. **Multiplicity Boundary:** The 17 paired hypothesis tests in Table 2 are correctly labeled as unadjusted nominal $p$-values. The primary insertion faithfulness advantages across all 5 attribution methods and Grad-CAM stability remain robustly significant ($p_{\text{adj}} < 0.0001$) under family-wise Holm-Bonferroni correction ($\alpha = 0.05$). Secondary comparisons (Occlusion stability nominal $p = 0.0196$, Random baseline insertion nominal $p = 0.0149$, Random baseline deletion nominal $p = 0.0242$) do not survive family-wise correction.
2. **Prediction Trajectory Baseline Effect:** The difference in Random Baseline Insertion AUC ($0.5135$ balanced vs. $0.4183$ baseline) is mathematically driven by the difference in model prediction baselines across the 50 cohort layouts (mean $P(\text{HS}) = 0.5744$ for balanced sampling vs. $0.1795$ for baseline), rather than attribution quality.
3. **Single-Seed Reference Scope:** All reported empirical figures reflect the audited Seed 42 models. Multi-seed variance study across seeds `[42, 100, 2024]` remains in verified dry-run standby mode.
4. **Diagnostic Cohort Scope:** The 50-layout evaluation set is a stratified diagnostic sample designed to probe error modes (`shared_tp`, `shared_fn`, `balanced_won`, `baseline_won`), not an i.i.d. population estimate.

---

## 2. Scope and Limitations

### 2.1 What Was Executed and Verified
- **Remediation Verification:** Verified presence, line locations, and exact text of `FIND-01`, `FIND-02`, `FIND-03`, and `FIND-04` across Markdown, LaTeX, and Python code.
- **Holm-Bonferroni Re-execution:** Executed the exact step-down multiplicity procedure on all 17 paired comparisons, confirming sequential decision thresholds and adjusted $p$-values.
- **Isolated Generator Reproducibility:** Executed `generate_draft_reports` inside a sandboxed temporary directory (`tempfile.TemporaryDirectory`) using real empirical inputs, confirming exact byte-for-byte matching with canonical draft files.
- **Test Harness Execution:** Executed `pytest tests/test_via3_workstreams.py` (8/8 passed) and full repository regression suite (65/65 passed).
- **Coursework Safeguard Verification:** Confirmed that `docs/DA2_Group18_Report.docx` and `latex_report/` remain untouched.

### 2.2 Scope Limitations
- Multi-seed training runs for seeds 100 and 2024 were intentionally omitted to prevent unauthorized GPU expenditures.
- Verification is limited to OpenHSDBench `via-3`.

---

## 3. FIND-01 through FIND-04 Verification Matrix

| Finding ID | Remediation Requirement | Verification Outcome | Exact Document Locations & Evidence |
| :---: | :--- | :---: | :--- |
| **`FIND-01`** | Relabel table headers as nominal $p$-values; specify $M=17$ test family; state Holm correction status. | **VERIFIED (PASS)** | - **Markdown Draft:** Table 2 header line 29: `Nominal Wilcoxon p* \| Nominal Paired t p`. Callout lines 50–51: explicitly documents $M=17$ family, $\alpha=0.05$ family-wise control, survival of primary insertion tests ($p_{\text{adj}} < 0.0001$), and non-survival of secondary tests (Occlusion stability $p=0.0196$, threshold $0.0125$; Random baseline $p=0.0149, 0.0242$, thresholds $0.0100, 0.0167$). Section 6 Point 6 lines 79–80 codifies this.<br>- **LaTeX Draft:** Table 2 header line 33: `Nominal Wilcoxon p* & Nominal Paired t p`. Caption line 28 & footnote line 53: explicitly states unadjusted nominal status across $M=17$ tests and details Holm survival. |
| **`FIND-02`** | Contextualize random-baseline insertion AUC disparity using cohort mean prediction probabilities ($0.1795$ vs. $0.5744$). | **VERIFIED (PASS)** | - **Markdown Draft:** Callout lines 53–55 & Section 6 Point 7 line 81: explicitly documents cohort mean $P(\text{HS}) = 0.5744$ (balanced) vs. $0.1795$ (baseline), explaining that higher intermediate sigmoid activations account for the curve differences.<br>- **LaTeX Draft:** Table 2 footnote line 53: explicitly includes `Random Baseline Note (FIND-02)` with cohort mean probabilities $0.5744$ vs. $0.1795$. |
| **`FIND-03`** | Explicitly label XAI results as reflecting the audited Seed 42 reference models; avoid implying multi-seed completion. | **VERIFIED (PASS)** | - **Markdown Draft:** Section 3 title line 27: `## 3. XAI Faithfulness and Stability Comparison ($N=50$ Paired Layouts, Seed 42 Reference)`. Section 2 table lines 16–21 & note line 24: states Seed 42 reference run status and notes multi-seed is in dry-run standby.<br>- **LaTeX Draft:** Table 2 caption line 28: explicitly adds `(N=50 Layouts, Seed 42 Reference)`. Table 3 caption line 60: explicitly adds `(N=50, Seed 42 Reference)`. |
| **`FIND-04`** | Eliminate LaTeX string slicing `std_s[:5]`; apply standard 4-decimal place rounding consistent with Markdown. | **VERIFIED (PASS)** | - **Generator Code:** `scripts/generate_paper_reports.py` line 262: replaced `std_s[:5]` with floating-point formatted `${std_num:.4f}`.<br>- **LaTeX Output:** Table 3 line 67: renders `$0.1166 \pm 0.0229$`, `$0.1587 \pm 0.0265$`, and `$0.0865 \pm 0.0246$`, achieving 100% agreement with Markdown Table 4 line 59. |

---

## 4. Holm Correction Verification and Statistical Conclusions

### 4.1 Specification of the Multiple-Testing Family
- **Family Size:** $M = 17$ paired comparisons between Baseline (Seed 42) and Balanced Sampling (Seed 42) on the 50 matched layout samples.
- **Hypothesis Structure:** 6 Insertion AUC comparisons, 6 Deletion AUC comparisons, and 5 Stability $\rho$ comparisons (Random baseline has no perturbation stability metric).
- **Test Type:** Primary non-parametric two-sided paired Wilcoxon signed-rank tests.

### 4.2 Sequential Step-Down Execution Matrix ($\alpha = 0.05$)

| Rank $k$ | Comparison Hypothesis | Nominal Wilcoxon $p_{(k)}$ | Sequential Threshold $\frac{0.05}{17 - k + 1}$ | Holm Decision Rule | Step-Down Adjusted $\tilde{p}_{(k)}$ | Statistical Interpretation |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | Grad-CAM Insertion AUC | $< 10^{-8}$ | $0.0029$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.2726$) |
| **2** | Grad-CAM Stability $\rho$ | $3.92 \times 10^{-8}$ | $0.0031$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.0383$) |
| **3** | Grad-CAM++ Insertion AUC | $7.55 \times 10^{-9}$ | $0.0033$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.2312$) |
| **4** | Grad-CAM++ Deletion AUC | $2.00 \times 10^{-7}$ | $0.0036$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.1972$) |
| **5** | LayerCAM Insertion AUC | $1.81 \times 10^{-10}$ | $0.0038$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.3206$) |
| **6** | Gradient Saliency Insertion AUC | $2.81 \times 10^{-9}$ | $0.0042$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.3752$) |
| **7** | Gradient Saliency Deletion AUC | $2.89 \times 10^{-7}$ | $0.0045$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = -0.0746$) |
| **8** | Gradient Saliency Stability $\rho$ | $9.06 \times 10^{-10}$ | $0.0050$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.0213$) |
| **9** | Occlusion Insertion AUC | $5.33 \times 10^{-15}$ | $0.0056$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = +0.5182$) |
| **10** | Occlusion Deletion AUC | $4.79 \times 10^{-6}$ | $0.0062$ | **Reject $H_0$** | $< 0.0001$ | Statistically significant ($\Delta = -0.0974$) |
| **11** | Grad-CAM Deletion AUC | $0.0014$ | $0.0071$ | **Reject $H_0$** | $0.0098$ | Statistically significant ($\Delta = +0.0991$) |
| **12** | LayerCAM Deletion AUC | $0.0023$ | $0.0083$ | **Reject $H_0$** | $0.0138$ | Statistically significant ($\Delta = +0.0803$) |
| **13** | Random Baseline Insertion AUC | $0.0149$ | $0.0100$ | **Fail to Reject** | $0.0745$ | **Not significant** ($p_{(13)} > 0.0100$) |
| **14** | Occlusion Stability $\rho$ | $0.0196$ | $0.0125$ | **Fail to Reject** | $0.0784$ | **Not significant** ($p_{(14)} > 0.0125$) |
| **15** | Random Baseline Deletion AUC | $0.0242$ | $0.0167$ | **Fail to Reject** | $0.0784$ | **Not significant** ($p_{(15)} > 0.0167$) |
| **16** | Grad-CAM++ Stability $\rho$ | $0.5210$ | $0.0250$ | **Fail to Reject** | $1.0000$ | **Not significant** ($\Delta = -0.0307$) |
| **17** | LayerCAM Stability $\rho$ | $0.9771$ | $0.0500$ | **Fail to Reject** | $1.0000$ | **Not significant** ($\Delta = -0.0049$) |

### 4.3 Methodological Conclusions Supported:
1. **Multiplicity Robustness:** The claim that Balanced Sampling improves insertion-based faithfulness is supported across all 5 attribution methods under strict family-wise error control ($p_{\text{adj}} < 0.0001$).
2. **Attribution Stability Qualification:** Grad-CAM stability gain ($\Delta = +0.0383, p_{\text{adj}} < 0.0001$) and Saliency stability gain ($\Delta = +0.0213, p_{\text{adj}} < 0.0001$) survive multiplicity correction. However, Occlusion stability ($\Delta = +0.0400$, nominal $p = 0.0196$) does not survive Holm correction ($\tilde{p} = 0.0784 > 0.05$). The drafts correctly acknowledge this limitation.

---

## 5. Numerical Cross-Artifact Verification

Every numerical value across all tables was verified for exact concordance:

### 5.1 Classification Metrics ($N=12,803$; $411$ HS, $12,392$ NHS)
- **Baseline ($\tau=0.50$):** PR-AUC `0.2518`, ROC-AUC `0.7328`, Recall `0.0925`, Precision `0.8636`, F1 `0.1670`, BalAcc `0.5460` (**Concordant**).
- **Baseline ($\tau^*=0.08$ Calibrated):** PR-AUC `0.2518`, ROC-AUC `0.7328`, Recall `0.2214`, Precision `0.3640`, F1 `0.2753`, BalAcc `0.6043` (**Concordant**).
- **Class-Weighted ($\tau^*=0.50$ Calibrated):** PR-AUC `0.1270`, ROC-AUC `0.6864`, Recall `0.1825`, Precision `0.2500`, F1 `0.2110`, BalAcc `0.5822` (**Concordant**).
- **Balanced Sampling ($\tau=0.50$):** PR-AUC `0.2619`, ROC-AUC `0.7840`, Recall `0.3090`, Precision `0.2714`, F1 `0.2890`, BalAcc `0.6407` (**Concordant**).
- **Balanced Sampling ($\tau^*=0.76$ Calibrated):** PR-AUC `0.2619`, ROC-AUC `0.7840`, Recall `0.2603`, Precision `0.4038`, F1 `0.3166`, BalAcc `0.6238` (**Concordant**).

### 5.2 Representation Collapse & Spatial Standard Deviation ($N=50$)
- **Baseline:** $0/50$ flat CAMs (`0.0%`), Spatial Std: `$0.1166 \pm 0.0229$`, Max: `0.2009` (**Concordant in both MD and LaTeX**).
- **Balanced Sampling:** $0/50$ flat CAMs (`0.0%`), Spatial Std: `$0.1587 \pm 0.0265$`, Max: `0.2269` (**Concordant in both MD and LaTeX**).
- **Class-Weighted:** $2/50$ flat CAMs (`4.0%`), Spatial Std: `$0.0865 \pm 0.0246$`, Max: `0.1307` (**Concordant in both MD and LaTeX**).

### 5.3 Terminology Audit
- No instances of conflicting abbreviations (`AUI` / `AUD`) were detected in either draft. The outputs consistently employ the standard terms "Insertion AUC" and "Deletion AUC".

---

## 6. Isolated Generator Reproducibility Findings

An isolated reproducibility check was performed by executing `scripts/generate_paper_reports.py:generate_draft_reports` with output destination redirected to a temporary scratch directory:

1. **Output Isolation:** Verified that the generator writes strictly to the provided `output_dir` path and performs zero in-place mutations of repository files.
2. **Byte-for-Byte Output Verification:**
   - Generated Markdown file was compared against `draft_reports/via3/draft_via3_experimental_report.md`: **Exact byte-for-byte match (0 differences)**.
   - Generated LaTeX table file was compared against `draft_reports/via3/draft_via3_table_summary.tex`: **Exact byte-for-byte match (0 differences)**.
3. **Data Source Verification:** Verified that the generator directly loads `results/via3_experiments/imbalance/imbalance_comparison_summary.json` and `results/via3_experiments/xai_comparison/xai_comparison_summary.json`. No hardcoded synthetic mocks were loaded when real empirical data was present.

---

## 7. Test Results and Repository Integrity

| Test Execution | Scope | Command | Outcome |
| :--- | :--- | :--- | :---: |
| **Workstream D Automated Tests** | Focused Release Check | `./.venv/bin/pytest tests/test_via3_workstreams.py` | **8 Passed in 10.53s** |
| **Full Regression Suite** | Full Repository Regression | `./.venv/bin/pytest tests/` | **65 Passed in 13.88s** |

### Repository Working Tree Integrity:
- `git status` confirms zero unexpected unstaged or staged modifications.
- Protected files verified untouched:
  - `docs/DA2_Group18_Report.docx` — **Unmodified**
  - `latex_report/` — **Unmodified**
- All 16 audited dataset manifests in `results/dataset_audit/` match original SHA-256 checksums.

---

## 8. Remaining Limitations and Scientific Scope

1. **Single-Seed Reference Scope:**
   - The primary empirical results are derived from the audited reference models trained under Seed 42. While multi-seed execution scripts and test harnesses are prepared and validated in dry-run mode, full multi-seed GPU training (e.g. seeds 100, 2024) has not been performed. Downstream manuscripts must explicitly present findings as single-seed reference results.
2. **Attribution Faithfulness vs. Physical Defect Geometry:**
   - Insertion and deletion curves evaluate statistical model score sensitivity under progressive pixel zero-masking. They evaluate attribution faithfulness under zero-fill perturbations, not physical defect boundary necessity or formal Pearlian causal sufficiency.
3. **Diagnostic Cohort Representation:**
   - The 50-layout cohort deliberately enriches for error and discrepancy cases (`balanced_won`, `baseline_won`). The reported statistical significance demonstrates meaningful differences across these diagnostic regimes, but should not be generalized as unconditional population averages over the 12,803-sample clean test set.

---

## 9. Release Recommendation

| Dissemination Level | Release Status | Authorization Rationale |
| :--- | :---: | :--- |
| **Internal Research Review** | **APPROVED** | Artifacts are fully reproducible, mathematically verified, and accompanied by complete provenance records. |
| **Preprint Draft (arXiv / TechReport)** | **APPROVED** | Draft tables and report incorporate all necessary statistical qualifications, multiplicity corrections, and baseline contextualizations. |
| **External Journal / Conference Camera-Ready** | **CONDITIONAL** | Requires executing the prepared multi-seed training suite (seeds 100 and 2024) to satisfy camera-ready multi-seed variance reporting requirements. Single-seed methodology is otherwise sound. |

---

## 10. Prioritized Blocker List

- **Blocker Count:** **0**
- No critical or high-severity blockers remain. The research package is formally frozen and verified in `draft_reports/via3/`.
