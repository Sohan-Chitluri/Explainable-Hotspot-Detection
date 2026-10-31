# Release Notes: OpenHSDBench `via-3` Multi-Seed Classification & XAI (RC1)

**Release Candidate:** `via3_multiseed_rc1`  
**Dataset:** OpenHSDBench `via-3` Lithography Hotspot Benchmark  
**Date:** 2026-10-25  
**Final Release Gate Verdict:** **PASS WITH QUALIFICATIONS**  

---

## 1. Scope of the Release

This release candidate synthesizes the results of an extensive multi-seed experimental campaign on the OpenHSDBench `via-3` benchmark, spanning binary hotspot classification and explainable artificial intelligence (XAI) robustness:

- **Classification Scope:** Custom 18-layer CNNs (12,873 parameters) evaluated across three independent random seeds (`42`, `100`, `2024`) and two primary training strategies—**Baseline (unweighted empirical distribution)** and **Balanced Sampling (50/50 mini-batch balancing)**—plus a single-seed **Class-Weighted** model (`seed 42`) serving as an auxiliary reference. Performance is benchmarked on the clean held-out test split ($N=12,803$ layouts: 411 Hotspot [HS], 12,392 Non-Hotspot [NHS]) using validation-calibrated operating thresholds $\tau^*$ derived on $N=2,846$ validation layouts.
- **Explainability Scope:** Systematic evaluation of six attribution methods (**Grad-CAM**, **Grad-CAM++**, **LayerCAM**, **Gradient Saliency**, **Occlusion Sensitivity**, and a **Random Baseline**) across all six trained models on a frozen 50-layout diagnostic cohort. The study includes 1,800 pixel insertion/deletion faithfulness curves, 1,500 input-noise stability trials ($\sigma=0.05, T=3$), 300 Grad-CAM representation collapse diagnostics, and 6 model parameter randomization sanity checks.

---

## 2. Key Scientific Findings Supported by Reconciled Evidence

### 2.1 Multi-Seed Hotspot Classification
1. **Refutation of Universal PR-AUC Dominance:** While Balanced Sampling achieved higher PR-AUC in Seed 42 ($0.2619$ vs. $0.2518$), cross-seed evaluation refutes any claim of universal PR-AUC superiority. Baseline achieved a higher multi-seed average PR-AUC ($0.2749 \pm 0.0233$) than Balanced Sampling ($0.2635 \pm 0.0476$), driven by Baseline Seed 100 ($0.2983$). The paired strategy difference is slightly negative ($-0.0114 \pm 0.0623$) and sign-inconsistent across seeds ($+0.0101, -0.0815, +0.0375$).
2. **Minority-Class Sensitivity Stabilization:** Balanced Sampling's definitive operational advantage is stabilizing hotspot recall. The cross-seed standard deviation for HS Recall dropped ~5-fold from $0.0544$ (Baseline, range $19.46\% - 29.93\%$) to $0.0112$ (Balanced Sampling, range $26.03\% - 27.98\%$), while raising average recall from $23.84\%$ to $27.33\%$.
3. **Catastrophic Default-Threshold Failure in Baseline:** At standard uncalibrated $\tau = 0.50$, Baseline Seed 2024 suffered complete detection collapse, identifying **0 hotspots out of 411** ($\text{Recall} = 0.0\%$, max test probability $0.4687$). Balanced Sampling maintained stable recall across all seeds at default $\tau = 0.50$ (Seed 42: $30.90\%$, Seed 100: $38.69\%$, Seed 2024: $27.98\%$).
4. **Operating Threshold Shift vs. True Bayesian Calibration:** Calibrated decision thresholds shifted upward from $\bar{\tau}^* = 0.0533 \pm 0.0379$ (Baseline) to $\bar{\tau}^* = 0.6767 \pm 0.1274$ (Balanced Sampling). This shift is a mechanical consequence of the 50/50 artificial sampling prior during training and does not constitute true Bayesian posterior probability calibration.

### 2.2 Explainability (XAI) Robustness & Representation Dynamics
1. **Elimination of Grad-CAM Representation Collapse:** In Baseline models, 3 of 150 layout heatmaps suffered Flat-CAM collapse (spatial standard deviation $\text{std}(M) < 10^{-4}$), and Baseline Seed 2024 exhibited severe spatial dynamic range attenuation (spatial std $0.0589 \pm 0.0132$). Balanced Sampling completely eliminated Flat-CAMs ($0/150$, $0.0\%$) and preserved robust contrast ($0.1552 \pm 0.0275$).
2. **LayerCAM Noise Invariance:** LayerCAM exhibited the highest and most uniform stability under input Gaussian noise ($\sigma=0.05$) across both training strategies and all seeds ($\bar{\rho} = 0.9556 \pm 0.0185$ Baseline, $0.9554 \pm 0.0075$ Balanced), confirming that localized element-wise gradient gating provides superior resistance to high-frequency background noise.
3. **Pathological Instability of Grad-CAM++:** Grad-CAM++ exhibited severe noise instability across all six models ($\bar{\rho} = 0.2692 \pm 0.0401$ Baseline, $\bar{\rho} = 0.1717 \pm 0.0799$ Balanced, dropping to $\bar{\rho} = 0.0867$ in Balanced Seed 100), demonstrating its unsuitability for precision lithographic hotspot inspection.
4. **Protocol-Specific Interpretation of Occlusion Sensitivity:** Occlusion Sensitivity achieved the highest Insertion AUC ($0.8422 \pm 0.1034$ in Balanced Sampling) and lowest Deletion AUC ($0.1365 \pm 0.0989$). **Crucially, this cannot be claimed as universal or causal superiority.** The insertion/deletion benchmark evaluates models using progressive square patch zero-masking, identical to Occlusion Sensitivity's $32\times32$ zero-fill sliding window. This structural protocol alignment mechanically favors Occlusion over gradient-based methods. Occlusion Sensitivity is faithful *under progressive substrate replacement protocols*, but operates at coarse spatial resolution ($32\times32$) and requires ~100x higher computational time.

---

## 3. Disclosures and Statistical Discipline

### 3.1 Aggregation Units: Seeds ($K=3$) vs. Layout Observations ($N=12,803$ / $N=50$)
The study strictly separates layout evaluation observations from independent training replicates:
- The 12,803 clean test layouts and 50 diagnostic layouts represent repeated measurements on individual models, not independent training runs.
- Cross-seed statistics reflect $K=3$ independent training runs (`seeds 42, 100, 2024`).
- With $K=3$, non-parametric paired tests cannot achieve significance at $\alpha=0.05$ (minimum two-tailed $p=0.25$), and paired $t$-tests on strategy differences yield non-significant results ($p=0.78$ for PR-AUC). No inferential claims of statistically significant superiority are made.
- Normal-theory Student's $t$ confidence intervals are omitted because bounded metrics violate normality assumptions and yield unphysical negative lower bounds for baseline thresholds ($[-0.0407, 0.1474]$). Metrics are reported as empirical dispersion: `Mean ± SD [Seed 42, Seed 100, Seed 2024]`.

### 3.2 Model Parameter Randomization Sanity Check (Adebayo et al., 2018)
Convolutional and dense weights were randomized with Gaussian noise ($\mathcal{N}(0, 0.05)$) to verify whether attributions depend on learned weights:
- **Prespecified Programmatic Criterion:** `"passed_sanity_check": bool(abs(adebayo_mean_rho) < 0.20)` (`scripts/run_via3_xai_multiseed.py:436`).
- **Observed Programmatic Result:** **Five of six models passed.**
  - Baseline Seed 42: $\bar{\rho} = 0.0000$ (PASSED)
  - Baseline Seed 100: $\bar{\rho} = -0.3321$ (**FAILED — FLAGGED**)
  - Baseline Seed 2024: $\bar{\rho} = 0.0000$ (PASSED)
  - Balanced Seed 42: $\bar{\rho} = -0.1331$ (PASSED)
  - Balanced Seed 100: $\bar{\rho} = -0.0571$ (PASSED)
  - Balanced Seed 2024: $\bar{\rho} = +0.0553$ (PASSED)
- **Substantive Interpretation:** A correlation of $\bar{\rho} = -0.3321$ indicates that explanations changed substantially under randomized weights (attributions did not act as invariant edge detectors). However, because $|-0.3321| > 0.20$, claiming 6/6 passed would represent unprespecified post-hoc threshold relaxation. The release notes transparently record the programmatic flag alongside the substantive interpretation.

### 3.3 Cohort Composition Reconciliation
The 50-layout diagnostic cohort (`xai_sample_cohort_manifest.json`) is composed of **40 Hotspot (label 1)** and **10 Non-Hotspot (label 0)** layouts across six stratified error modes:
- `shared_tp`: 10 layouts (HS)
- `shared_fn`: 10 layouts (HS)
- `balanced_won`: 10 layouts (HS)
- `baseline_won`: 10 layouts (HS)
- `shared_tn`: 5 layouts (NHS)
- `shared_fp`: 5 layouts (NHS)

Informal narrative phrasing in historical report `WORKSTREAM_E4_2_EXECUTION_REPORT.md` citing "25 HS / 25 NHS" was a typographical error describing quadrants rather than ground-truth labels. The true on-disk cohort has been verified as 40 HS / 10 NHS.

### 3.4 Table 4 Typographical Reconciliation
Earlier draft versions cited `0.0589` as the pooled Baseline spatial standard deviation. Forensic audit confirmed this was a typographical transcription duplicate of the Seed 2024 mean ($0.0589$). The verified sample standard deviation across all 150 layout observations ($ddof=1$) is $\mathbf{0.0571}$ (population $ddof=0$ is $0.0569$). The cross-seed standard deviation across the three seed means is $\mathbf{0.0615}$ ($K=3, ddof=1$). Both numbers are correctly reported in the reconciled drafts.

---

## 4. Preserved Historical Audit Errata

To preserve immutable forensic audit integrity, historical reports have not been altered retroactively:
1. **Manifest Checksum in E4.2:** Table 5 of `WORKSTREAM_E4_2_EXECUTION_REPORT.md` lists `75f28469ad...` for `reproducibility_manifest.json`. This digest was captured prior to runtime environment injection. The final on-disk manifest hashes to `9655c3480db1...`. All artifact hashes *contained within* the manifest match on-disk files byte-for-byte.
2. **Cohort Narrative in E4.2:** Section 3 of `WORKSTREAM_E4_2_EXECUTION_REPORT.md` states "25 HS / 25 NHS", which is refuted by the frozen manifest and corrected in the release drafts.

---

## 5. Summary of Release Qualifications

In accordance with Workstream E5, this release candidate carries the status **PASS WITH QUALIFICATIONS**. The empirical data is fully verified and structurally sound, but the candidate cannot be certified as an unconditional "push-button" reproducible benchmark due to runner resume limitations, uncompiled local PDF tables, and prespecified threshold flags. See [`KNOWN_LIMITATIONS.md`](./KNOWN_LIMITATIONS.md) for full remediation guidance.
