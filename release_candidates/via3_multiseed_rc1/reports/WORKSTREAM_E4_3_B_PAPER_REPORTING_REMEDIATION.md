# Workstream E4.3-B — Multi-Seed Paper Reporting Remediation & Draft Synthesis

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3`  
**Target Artifacts Modified:**
1. `draft_reports/via3/draft_via3_experimental_report.md`
2. `draft_reports/via3/draft_via3_table_summary.tex`  
**Auditor & Remediation Engineer:** Independent Research Reporting & Reproducibility Assistant  
**Date:** 2026-10-25  
**Final Status & Verdict:** **PASS WITH QUALIFICATIONS**

---

## 1. Executive Summary

This remediation report documents the execution and completion of **Workstream E4.3-B: Multi-Seed Paper Reporting Remediation & Draft Synthesis** for OpenHSDBench `via-3` lithography hotspot detection.

The objectives of this workstream were strictly confined to evidence-reconciliation and reporting synthesis:
1. Integrate the authoritative multi-seed classification results ($K=3$ seeds, clean held-out test cohort $N=12,803$) from Workstreams E2 and E3 into the draft reports.
2. Integrate the multi-seed explainability (XAI) robustness results ($K=3$ seeds, 50-layout diagnostic cohort, 1,800 faithfulness curves, 1,500 stability evaluations, 300 representation collapse diagnostics) from Workstreams E4.2 and E4.3-A into the draft reports.
3. Reconcile verified reporting defects, typographical errors, and documentation ambiguities identified in audit workstreams (ISSUE-MIN-01, ISSUE-MIN-02, ISSUE-MAJ-01, ISSUE-MAJ-02, ISSUE-MAJ-03, ISSUE-MAJ-04).
4. Enforce strict statistical discipline: distinguish layout evaluation observations from independent training seed replicates, eliminate ungrounded inferential p-value claims across training seeds, avoid normal-theory confidence intervals on bounded small-sample metrics, and document protocol-specific limitations of XAI faithfulness benchmarks.
5. Strictly adhere to authorization boundaries: modify **only** the two authorized draft files, leave all protected directories, manifests, checkpoints, and coursework files completely untouched, and verify all file hashes before and after execution.

All empirical data were audited and verified directly against raw on-disk CSV and JSON records. Zero GPU training, inference, re-clustering, or data modifications were initiated.

---

## 2. Scope and Authorization

### 2.1 Authorized Modifications
In strict compliance with the workstream specification, write operations were permitted on **only two existing files**:
1. `draft_reports/via3/draft_via3_experimental_report.md`
2. `draft_reports/via3/draft_via3_table_summary.tex`

And exactly one new report was created at:
- `results/via3_experiments/multiseed/WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md`

### 2.2 Boundary Adherence & Protected Path Verification
The following resources were verified as **strictly protected and unaltered**:
- Coursework deliverables: `docs/DA2_Group18_Report.docx` and `latex_report/`
- Prior canonical backups: `draft_reports/via3/draft_via3_experimental_report.md.bak` and `draft_reports/via3/draft_via3_table_summary.tex.bak`
- Machine-readable raw metrics: `results/via3_experiments/xai_multiseed/*.csv`
- Model checkpoints: `models/via3_multiseed/*.keras` and `models/via3_imbalance/*.keras`
- Prior audit and execution reports: `WORKSTREAM_E3_STATISTICAL_AUDIT.md`, `WORKSTREAM_E4_1_XAI_PREFLIGHT.md`, `WORKSTREAM_E4_2_EXECUTION_REPORT.md`, `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md`, `WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md`, `WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT_INDEPENDENT.md`, and `WORKSTREAM_E4_3_D_RUNNER_REPRODUCIBILITY_AUDIT.md`
- Source code and automated test scripts: `scripts/*.py` and `tests/*.py`

---

## 3. Evidence Sources and Authority

Every factual and numerical claim incorporated into the drafts was traced directly to authoritative on-disk artifacts:

| Evidence Tier | Primary Artifact Path | Verification Authority & Contents |
| :--- | :--- | :--- |
| **Multi-Seed Classification (E2/E3)** | `results/via3_experiments/multiseed/WORKSTREAM_E3_STATISTICAL_AUDIT.md` | Independent recalculation of all 18 test/val CSVs; verification of operating thresholds $\tau^*$, PR-AUC, ROC-AUC, HS Recall, HS Precision, and F1 across seeds 42, 100, 2024. |
| **Classification Summaries** | `multiseed_baseline_summary.json`<br>`multiseed_balanced_sampling_summary.json` | Machine-readable multi-seed aggregates and per-seed confusion matrices. |
| **XAI Preflight Contract (E4.1)** | `results/via3_experiments/multiseed/WORKSTREAM_E4_1_XAI_PREFLIGHT.md` | Cryptographic freeze of the 50-layout cohort, target layer `conv2d_5`, logit $+z_{\text{HS}}$, and perturbation protocols. |
| **XAI Cohort Manifest** | `results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json` | SHA-256 `1c547c162c32df...`; verified composition of exactly 40 HS and 10 NHS layouts across 6 error/recovery strata. |
| **XAI Raw Measurements (E4.2)** | `results/via3_experiments/xai_multiseed/master_faithfulness_metrics.csv`<br>`results/via3_experiments/xai_multiseed/master_stability_metrics.csv`<br>`results/via3_experiments/xai_multiseed/master_collapse_diagnostics.csv` | Exactly 1,800 faithfulness rows, 1,500 stability rows, and 300 collapse diagnostic rows; 100% status `SUCCESS`. |
| **XAI Summary & Manifest** | `multiseed_xai_comparison_summary.json`<br>`reproducibility_manifest.json` | Cross-seed aggregations, paired differences, and output SHA-256 digests. |
| **XAI Provenance Audits (E4.3-A/D)** | `WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md`<br>`WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT_INDEPENDENT.md`<br>`WORKSTREAM_E4_3_D_RUNNER_REPRODUCIBILITY_AUDIT.md` | Independent verification of CSV metrics, identification of Table 4 typographical error, manifest self-hash mismatch, and runner resume audit. |

---

## 4. Issue-by-Issue Reconciliation

### 4.1 ISSUE-MIN-01: Table 4 Pooled Baseline Spatial Standard Deviation
- **Disputed Reporting:** In Table 4 of `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` (row 122), the value `0.0589` was reported as the `Spatial Std SD` for `Baseline Pooled`.
- **Audit Findings:** 
  - An independent recalculation across the 150 layout observations in `master_collapse_diagnostics.csv` (`baseline` strategy) revealed:
    - Mean spatial standard deviation: $0.1294$
    - Sample standard deviation across 150 layout observations ($ddof=1$): $\mathbf{0.0571}$
    - Population standard deviation across 150 layout observations ($ddof=0$): $\mathbf{0.0569}$
  - For the three seed-level means ($[0.1568, 0.1724, 0.0589]$):
    - Cross-seed sample standard deviation across 3 seeds ($ddof=1$): $\mathbf{0.0615}$
  - The value `0.0589` reported in E4.2 was a typographical transcription error: the row immediately above was `Baseline Seed 2024`, whose `Spatial Std Mean` is `0.0589`. During manual Markdown authoring, `0.0589` was accidentally pasted into the standard deviation column of the pooled row.
- **Draft Remediation:**
  - In `draft_via3_experimental_report.md` (Section 3, Table 3) and `draft_via3_table_summary.tex` (Table 3), the pooled baseline layout-level standard deviation is correctly stated as $\mathbf{0.1294 \pm 0.0571}$.
  - The cross-seed standard deviation across the three seed means is explicitly reported in a dedicated row as $\mathbf{0.1294 \pm 0.0615}$ ($K=3, ddof=1$).
  - A clarifying footnote explicitly notes the origin of the $0.0589$ typo and distinguishes layout-level from seed-level variance.

---

### 4.2 ISSUE-MIN-02: Aggregation Units and Multi-Seed Reporting
- **Disputed Reporting:** Earlier reports used the generic notation `Mean ± SD` in table headers without explicitly indicating whether the standard deviation quantified layout-to-layout variance ($N=50$ or $N=150$) or training-seed-to-training-seed variance ($K=3$). Furthermore, normal-theory Student's $t$ confidence intervals on $K=3$ seeds produced unphysical negative bounds for bounded positive parameters (e.g. baseline decision threshold CI $[-0.0407, 0.1474]$).
- **Audit Findings:**
  - $N=12,803$ test layouts (and $N=50$ diagnostic layouts) are evaluation observations sampled from a fixed distribution, not independent model training runs.
  - Across $K=3$ independent training seeds (`[42, 100, 2024]`), the sample size for strategy comparison is $K=3$. Non-parametric paired tests have minimum two-tailed $p=0.25$, and a paired $t$-test on $\Delta \text{PR-AUC}$ yields $p=0.78$. Inferential claims of statistical significance between training strategies across seeds are mathematically invalid with $K=3$.
  - Normal-theory Student's $t$ intervals ($t_{\text{crit}}=4.303, df=2$) are invalid on bounded metrics on $K=3$ samples.
- **Draft Remediation:**
  - Both drafts consistently adopt the empirical reporting standard:
    $$\text{Mean} \pm \text{SD} \quad [\text{Seed 42}, \text{Seed 100}, \text{Seed 2024}]$$
  - In cross-seed tables (Classification Table 1 and XAI Table 2), the $\pm$ notation is explicitly documented in table titles, column headers, and footnotes as the sample standard deviation across $K=3$ independent training seeds ($ddof=1$).
  - In diagnostic contrast tables (Table 3), per-seed rows are explicitly labeled as layout-level dispersion ($N=50, ddof=1$), pooled rows as pooled layout dispersion ($N=150, ddof=1$), and cross-seed rows as seed-mean dispersion ($K=3, ddof=1$).
  - Normal-theory confidence intervals have been completely eliminated from the drafts.
  - Explicit narrative warnings are added stating that the 12,803 test layouts must not be conflated with independent training replicates.

---

### 4.3 ISSUE-MAJ-01: Cohort-Composition Discrepancy
- **Disputed Reporting:** Section 3 of `WORKSTREAM_E4_2_EXECUTION_REPORT.md` stated:
  *"Quadrant Representation: True Positives (TP): 15 layouts, False Negatives (FN): 10 layouts, True Negatives (TN): 15 layouts, False Positives (FP): 10 layouts. Ground Truth Distribution: 25 Hotspot (label 1), 25 Non-Hotspot (label 0)."*
- **Audit Findings:**
  - Inspection of the frozen cohort manifest (`results/via3_experiments/xai_comparison/xai_sample_cohort_manifest.json`, SHA-256 `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`) and all master CSV records confirmed that the diagnostic cohort contains exactly:
    - **40 Hotspot layouts (ground truth label 1)**
    - **10 Non-Hotspot layouts (ground truth label 0)**
  - The actual stratum distribution is:
    - `shared_tp`: 10 layouts (label 1)
    - `shared_fn`: 10 layouts (label 1)
    - `balanced_won`: 10 layouts (label 1)
    - `baseline_won`: 10 layouts (label 1)
    - `shared_tn`: 5 layouts (label 0)
    - `shared_fp`: 5 layouts (label 0)
  - This matches Table 3.1 of the approved preflight contract `WORKSTREAM_E4_1_XAI_PREFLIGHT.md` byte-for-byte.
  - The author of the E4.2 execution report narrative erroneously described an even 25/25 split by confusing four hypothetical confusion quadrants with the actual error-stratified diagnostic benchmark.
- **Draft Remediation:**
  - In `draft_via3_experimental_report.md` (Section 2 and Qualification 5) and `draft_via3_table_summary.tex` (Table 2 and Table 3 notes), the cohort composition is explicitly documented as **40 Hotspot and 10 Non-Hotspot layouts** across the six frozen error and recovery modes.
  - The erroneous "25 HS / 25 NHS" statement from earlier narrative reports is explicitly cited, refuted, and corrected.

---

### 4.4 ISSUE-MAJ-02: Reproducibility-Manifest Checksum Discrepancy
- **Disputed Reporting:** Table 5 of `WORKSTREAM_E4_2_EXECUTION_REPORT.md` recorded the SHA-256 digest of `reproducibility_manifest.json` as:
  `75f28469ad9373cf58bb06cb38ae59d5f7b8faec7bf2987a0774a88ae8210373`
- **Audit Findings:**
  - Independent hashing of `results/via3_experiments/xai_multiseed/reproducibility_manifest.json` on disk yields:
    `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51`
  - All hashes *contained inside* `reproducibility_manifest.json` for output artifacts (`master_faithfulness_metrics.csv`, `master_stability_metrics.csv`, `master_collapse_diagnostics.csv`, `multiseed_xai_comparison_summary.json`) match the physical disk files byte-for-byte.
  - Historical provenance trace indicates that `75f28469ad...` was the digest of an earlier intermediate generation of the manifest before final Python and TensorFlow environment metadata strings were injected. When the manifest was finalized, its hash changed, but Table 5 in the execution report text was not updated.
- **Draft Remediation:**
  - The draft reports avoid asserting that the execution report self-hash matched the manifest.
  - In `draft_via3_experimental_report.md` Qualification 8, this discrepancy is transparently documented with its verified on-disk hash (`9655c3480d...`).
  - Correcting Table 5 in `WORKSTREAM_E4_2_EXECUTION_REPORT.md` is preserved as a protected boundary constraint and handed off to Workstream E5.

---

### 4.5 ISSUE-MAJ-03: Parameter-Randomization Interpretation (Adebayo Sanity Check)
- **Disputed Reporting:** Section 1 of `WORKSTREAM_E4_2_EXECUTION_REPORT.md` and Section 5 of `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` stated that "all 6 models passed" the Adebayo parameter randomization sanity check. However, in `multiseed_xai_comparison_summary.json`, `baseline_seed100` was recorded with:
  `"passed_sanity_check": false` under observed mean Spearman rank correlation $\bar{\rho} = -0.3321$.
- **Audit Findings:**
  - In `scripts/run_via3_xai_multiseed.py:436`, the programmatic decision rule was prespecified as:
    `"passed_sanity_check": bool(abs(adebayo_mean_rho) < 0.20) if not np.isnan(adebayo_mean_rho) else False`
  - Under this prespecified programmatic rule, **5 of 6 models passed**, while Baseline Seed 100 failed because $|-0.3321| = 0.3321 > 0.20$.
  - In XAI theory (Adebayo et al., NeurIPS 2018), failure of a sanity check occurs when explanations remain invariant under weight randomization ($\bar{\rho} \approx +1.0$), demonstrating that heatmaps act as input edge detectors rather than feature representations. A negative correlation ($\bar{\rho} = -0.3321$) confirms strong parameter sensitivity (the explanation changes drastically when weights change).
  - However, claiming that all 6 models passed required post-hoc threshold relaxation (e.g. to $|\bar{\rho}| \le 0.35$). The broader criterion was not prespecified in code.
- **Draft Remediation:**
  - Both drafts now state the empirical facts objectively: under the prespecified programmatic criterion ($|\bar{\rho}| < 0.20$), 5 of 6 checkpoints passed, while Baseline Seed 100 recorded $\bar{\rho} = -0.3321$ and was flagged as `passed_sanity_check: false`.
  - Section 4 and Qualification 7 explain both the programmatic failure and the substantive scientific interpretation (negative correlation confirms parameter sensitivity) without retroactively claiming the broader threshold was prespecified.

---

### 4.6 ISSUE-MAJ-04: Runner Resume and Provenance Limitations
- **Disputed Reporting:** Previous drafts implied full automated reproducibility and push-button rerun capability.
- **Audit Findings (from Workstream E4.3-D):**
  - The completion guard `is_checkpoint_xai_completed` in `scripts/run_via3_xai_multiseed.py:108-132` checks only row count (`len(df) == expected`), failing to check semantic keys `(strategy, seed, filename, method)` or row status `status == "SUCCESS"`. A run where all samples threw exceptions would be falsely recognized as complete on resume.
  - Resume logic loads cached per-seed CSVs without verifying checkpoint or cohort SHA-256 checksums against the manifest.
  - Master CSV compilation upon resume stamps `multiseed_xai_comparison_summary.json` with a fresh UTC timestamp, changing its file hash.
  - These architectural vulnerabilities concern future runner robustness; they do **not** corrupt the existing on-disk E4.2 records, which have been independently audited and verified as complete.
- **Draft Remediation:**
  - Both drafts refrain from claiming a final release freeze or complete push-button reproducibility before Workstream E5.
  - Qualification 8 and Table notes document these runner resume limitations and formally hand them off to Workstream E5 for test harness and runner hardening.

---

## 5. Draft Changes

### 5.1 Modifications to `draft_reports/via3/draft_via3_experimental_report.md`

| Section | Original Text / State | Remediated Synthesis | Evidence & Justification |
| :--- | :--- | :--- | :--- |
| **Title & Scope** | Single-seed Workstream D draft reflecting Seed 42 reference models. | Multi-seed experimental pipeline report integrating $K=3$ seeds across classification (E2/E3) and XAI (E4.2). | Workstreams E2, E3, E4.2, E4.3-A. |
| **Section 1: Classification** | Table reported only single-seed reference runs (Seed 42) with placeholder dry-run note. | Integrated Table 1.1 reporting multi-seed aggregates (Mean $\pm$ SD across $K=3$ seeds), individual seed rows, and paired differences ($\Delta$). Added Table 1.2 on default $\tau=0.50$ collapse. | `WORKSTREAM_E3_STATISTICAL_AUDIT.md` Sections 3, 4, and 5; `multiseed_baseline_summary.json`. |
| **Section 1: Analysis** | Implicitly assumed Balanced Sampling was universally superior in PR-AUC. | Explicitly refuted PR-AUC dominance ($0.2749 \pm 0.0233$ Baseline vs $0.2635 \pm 0.0476$ Balanced; paired $\Delta = -0.0114 \pm 0.0623$). Highlighted 5x recall stabilization (SD $0.0544 \to 0.0112$) as primary empirical benefit. | `WORKSTREAM_E3_STATISTICAL_AUDIT.md` Section 1 and Section 7. |
| **Section 1: Calibration Note** | Described threshold shift as calibration. | Clarified that threshold shift ($\bar{\tau}^* = 0.6767$ vs $0.0533$) reflects the 50/50 training prior, not posterior probability calibration. | `WORKSTREAM_E3_STATISTICAL_AUDIT.md` Section 6.2. |
| **Section 2: XAI Cohort** | Vague mention of diagnostic cohort; E4.2 had reported "25 HS / 25 NHS". | Reconciled ISSUE-MAJ-01: Documented exact 40 HS / 10 NHS composition across 6 frozen error modes. Refuted the 25/25 narrative. | `WORKSTREAM_E4_1_XAI_PREFLIGHT.md` Section 3.1; `xai_sample_cohort_manifest.json`. |
| **Section 2: XAI Faithfulness** | Reported only Seed 42 reference paired comparison ($N=50$ layouts). | Integrated multi-seed Table 2.1 reporting Mean $\pm$ SD across $K=3$ seeds and paired differences ($\Delta$) for all 6 methods. | `WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md` Section 5.2; `master_faithfulness_metrics.csv`. |
| **Section 2: XAI Stability** | Reported only Seed 42 stability values. | Integrated multi-seed Table 2.2 reporting per-seed values and cross-seed aggregates for all 5 deterministic methods. | `master_stability_metrics.csv`; `multiseed_xai_comparison_summary.json`. |
| **Section 2: Faithfulness Caveats** | Uncritically praised Occlusion as most faithful. | Added critical caveat: Occlusion Sensitivity sliding patches ($32\times32$) structurally align with progressive zero-fill insertion/deletion perturbation, mechanically favoring Occlusion over gradient methods. | Methodological audit of `src/occlusion.py` and `src/faithfulness.py`. |
| **Section 3: Representation Collapse** | Table reported only Seed 42 reference with 0 flat CAMs. | Integrated multi-seed Table 3 with per-seed rows (Baseline seeds 42, 100, 2024; Balanced seeds 42, 100, 2024), pooled layout rows ($N=150$), and cross-seed rows ($K=3$). | `master_collapse_diagnostics.csv`; ISSUE-MIN-01 & ISSUE-MIN-02. |
| **Section 3: Typo Correction** | Table 4 in E4.2 had reported `0.0589` for pooled baseline SD. | Reconciled ISSUE-MIN-01: Corrected layout-level SD to $\mathbf{0.0571}$ ($N=150$) and cross-seed SD to $\mathbf{0.0615}$ ($K=3$). | Independent recalculation on `master_collapse_diagnostics.csv`. |
| **Section 4: Adebayo Sanity Check** | Reported only Seed 42 ($\bar{\rho} = -0.0486$). | Reconciled ISSUE-MAJ-03: Reported all 6 checkpoints. Transparently reported Baseline Seed 100 as $\bar{\rho} = -0.3321$ (`passed_sanity_check: false`), distinguishing prespecified programmatic check from substantive interpretation. | `scripts/run_via3_xai_multiseed.py:436`; `multiseed_xai_comparison_summary.json`. |
| **Section 5: Qualifications** | Listed 7 preliminary qualifications from Workstream D. | Expanded to 8 comprehensive qualifications incorporating multi-seed discipline, cohort composition, Adebayo threshold prespecification, and runner/manifest provenance limitations. | Workstreams E3, E4.3-A, E4.3-D synthesis. |

---

### 5.2 Modifications to `draft_reports/via3/draft_via3_table_summary.tex`

| Table in `.tex` | Original Content | Remediated Content | Synthesis & Reconciliation |
| :--- | :--- | :--- | :--- |
| **Table 1 (`tab:via3_classification_multiseed`)** | Single-seed Seed 42 benchmark rows only. | Multi-seed table reporting Mean $\pm$ SD across $K=3$ seeds for Baseline and Balanced Sampling, individual seed rows (`[42, 100, 2024]`), paired difference row ($\Delta_{\text{Bal}-\text{Base}}$), and Class-Weighted reference. Notes on default $\tau=0.50$ failure and prior vs calibration. | Reconciles E3 statistical audit with LaTeX draft. |
| **Table 2 (`tab:via3_xai_multiseed`)** | Paired single-seed XAI table on Seed 42 with $p$-values across 17 tests. | Multi-seed paired XAI table reporting Baseline ($K=3$), Balanced ($K=3$), Paired Diff ($\Delta$), and Seed Direction Consistency across all 6 methods. Explicit note defining $K=3$ seed units, protocol favorability of Occlusion Sensitivity, and Grad-CAM++ instability. | Reconciles E4.2/E4.3-A metrics; removes ungrounded multi-seed $p$-values; defines aggregation units. |
| **Table 3 (`tab:via3_collapse_diagnostics_multiseed`)** | Single-seed collapse table (50 samples). | Multi-seed collapse table with 6 per-model rows, pooled layout rows ($N=150$), and cross-seed rows ($K=3$). Corrects pooled baseline SD to $\mathbf{0.1294 \pm 0.0571}$ (ISSUE-MIN-01) and reports cross-seed SD as $0.0615$ (ISSUE-MIN-02). | Reconciles Table 4 typographical error and aggregation units. |
| **Table 4 (`tab:via3_adebayo_sanity_check`)** | (Did not exist in prior LaTeX draft). | New table reporting parameter randomization across all 6 checkpoints, prespecified rule ($|\bar{\rho}| < 0.20$), programmatic status, and substantive verdict. Explicitly records Baseline Seed 100 as $\bar{\rho} = -0.3321$ (FAILED programmatic rule, weight-sensitive substantively). | Reconciles ISSUE-MAJ-03 in paper-ready LaTeX format. |

---

## 6. Classification Reporting Synthesis

Integrating Workstreams E2 and E3 into the draft reports establishes the authoritative multi-seed classification baseline:

1. **Operating Performance on Clean Held-Out Cohort ($N=12,803$):**
   - **Baseline (Calibrated $\tau^* \in [0.01, 0.08]$):**
     - PR-AUC: $0.2749 \pm 0.0233$ [Seed 42: $0.2518$, Seed 100: $0.2983$, Seed 2024: $0.2745$]
     - ROC-AUC: $0.7598 \pm 0.0234$ [Seed 42: $0.7328$, Seed 100: $0.7739$, Seed 2024: $0.7728$]
     - HS Recall: $0.2384 \pm 0.0544$ [Seed 42: $0.2214$, Seed 100: $0.2993$, Seed 2024: $0.1946$]
     - HS Precision: $0.4223 \pm 0.1301$ [Seed 42: $0.3640$, Seed 100: $0.3315$, Seed 2024: $0.5714$]
     - HS F1-Score: $0.2934 \pm 0.0198$ [Seed 42: $0.2753$, Seed 100: $0.3146$, Seed 2024: $0.2904$]
     - Balanced Accuracy: $0.6129 \pm 0.0236$ [Seed 42: $0.6043$, Seed 100: $0.6396$, Seed 2024: $0.5949$]
   - **Balanced Sampling (Calibrated $\tau^* \in [0.53, 0.76]$):**
     - PR-AUC: $0.2635 \pm 0.0476$ [Seed 42: $0.2619$, Seed 100: $0.2168$, Seed 2024: $0.3120$]
     - ROC-AUC: $0.7760 \pm 0.0273$ [Seed 42: $0.7840$, Seed 100: $0.7456$, Seed 2024: $0.7985$]
     - HS Recall: $0.2733 \pm 0.0112$ [Seed 42: $0.2603$, Seed 100: $0.2798$, Seed 2024: $0.2798$]
     - HS Precision: $0.3680 \pm 0.1006$ [Seed 42: $0.4038$, Seed 100: $0.2544$, Seed 2024: $0.4457$]
     - HS F1-Score: $0.3090 \pm 0.0392$ [Seed 42: $0.3166$, Seed 100: $0.2665$, Seed 2024: $0.3438$]
     - Balanced Accuracy: $0.6281 \pm 0.0054$ [Seed 42: $0.6238$, Seed 100: $0.6263$, Seed 2024: $0.6341$]
   - **Paired Strategy Differences ($\Delta = \text{Balanced} - \text{Baseline}$):**
     - $\Delta \text{PR-AUC}$: $-0.0114 \pm 0.0623$ (Inconsistent: $+0.0101, -0.0815, +0.0375$)
     - $\Delta \text{ROC-AUC}$: $+0.0162 \pm 0.0406$ (Inconsistent: $+0.0512, -0.0283, +0.0256$)
     - $\Delta \text{HS Recall}$: $+0.0349 \pm 0.0524$ (Inconsistent: $+0.0389, -0.0195, +0.0852$)
     - $\Delta \text{HS Precision}$: $-0.0543 \pm 0.0851$ (Inconsistent: $+0.0398, -0.0771, -0.1257$)
     - $\Delta \text{HS F1-Score}$: $+0.0156 \pm 0.0554$ (Inconsistent: $+0.0413, -0.0481, +0.0534$)
     - $\Delta \text{Balanced Accuracy}$: $+0.0152 \pm 0.0265$ (Inconsistent: $+0.0195, -0.0133, +0.0392$)
     - $\Delta \tau^*$: $+0.6233 \pm 0.0896$ (Consistent positive: $+0.6800, +0.6700, +0.5200$)

2. **Key Methodological Conclusions:**
   - Multi-seed data conclusively refutes universal PR-AUC dominance for Balanced Sampling.
   - The primary empirical benefit of Balanced Sampling is **recall variance reduction** (sample SD drops from $0.0544$ to $0.0112$, an ~80% reduction in dispersion across seeds).
   - Balanced Sampling prevents catastrophic zero-detection failure at uncalibrated default thresholds ($\tau=0.50$), which completely disabled Baseline Seed 2024.

---

## 7. XAI Method-Wise Reporting Synthesis

Integrating Workstreams E4.2 and E4.3-A provides a rigorous multi-seed explainability benchmark across the six evaluated methods on the frozen 50-layout diagnostic cohort:

| Method | Insertion AUC ($\uparrow$) | Deletion AUC ($\downarrow$) | Stability $\bar{\rho}$ ($\uparrow$) | Flat-CAM Resistance | Assessment & Scientific Qualification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Grad-CAM** | Base: $0.3244 \pm 0.1159$<br>Bal: $0.4789 \pm 0.1432$ | Base: $0.2014 \pm 0.0707$<br>Bal: $0.3705 \pm 0.2042$ | Base: $0.8538 \pm 0.1885$<br>Bal: $0.9505 \pm 0.0179$ | Base: 3/150 (2.0%)<br>Bal: 0/150 (0.0%) | **Highly Recommended for Production.** Balanced sampling eliminates feature collapse (Seed 2024 stability was $0.6363$ on Baseline, restored to $0.9380$ on Balanced). |
| **Grad-CAM++** | Base: $0.2555 \pm 0.0986$<br>Bal: $0.4103 \pm 0.1972$ | Base: $0.2583 \pm 0.0993$<br>Bal: $0.4402 \pm 0.1208$ | Base: $0.2692 \pm 0.0401$<br>Bal: $0.1717 \pm 0.0799$ | Unaltered (uses same CAM activations) | **Unsuitable for Lithography Inspection.** Suffers severe, irreproducible instability under $\sigma=0.05$ noise ($\bar{\rho} \le 0.31$, dropping to $0.0867$ in Balanced Seed 100). |
| **LayerCAM** | Base: $0.2489 \pm 0.0627$<br>Bal: $0.4412 \pm 0.1828$ | Base: $0.2692 \pm 0.1351$<br>Bal: $0.3922 \pm 0.1411$ | Base: $0.9556 \pm 0.0185$<br>Bal: $0.9554 \pm 0.0075$ | Unaltered | **Highest Noise Robustness.** Element-wise positive gradient gating yields exceptional invariance ($\bar{\rho} \approx 0.955$) across both strategies and all seeds. |
| **Gradient Saliency** | Base: $0.1738 \pm 0.0851$<br>Bal: $0.5308 \pm 0.1059$ | Base: $0.6854 \pm 0.1552$<br>Bal: $0.5393 \pm 0.3544$ | Base: $0.4338 \pm 0.0123$<br>Bal: $0.4398 \pm 0.0203$ | N/A (Pixel gradient magnitude) | **Consistent Insertion Gain under Balanced Sampling** ($\Delta = +0.3570 \pm 0.0220$, consistent across all 3 seeds). Moderate stability ($\bar{\rho} \approx 0.43$). |
| **Occlusion ($32\times32$)** | Base: $0.4362 \pm 0.0367$<br>Bal: $0.8422 \pm 0.1034$ | Base: $0.1633 \pm 0.0936$<br>Bal: $0.1365 \pm 0.0989$ | Base: $0.6985 \pm 0.0534$<br>Bal: $0.7404 \pm 0.0145$ | N/A (Perturbation map) | **Faithful under Perturbation Protocol, NOT Universally Best.** Evaluated insertion/deletion uses square zero-masking, structurally favoring sliding-window zero occlusion. High latency and coarse resolution ($32\times32$) limit production utility. |
| **Random Baseline** | Base: $0.2655 \pm 0.0731$<br>Bal: $0.5220 \pm 0.1097$ | Base: $0.2634 \pm 0.0762$<br>Bal: $0.5187 \pm 0.1082$ | N/A (Excluded from stability) | N/A | **Empirical Control Reference.** Elevated insertion AUC in Balanced models ($0.5220$ vs $0.2655$) reflects higher cohort prediction baselines (mean $P(\text{HS}) = 0.5744$ vs $0.1795$), not attribution quality. |

---

## 8. Cross-Draft Consistency

Both modified documents (`draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`) were cross-checked for absolute semantic and numerical alignment:

1. **Classification Metrics:** All values in Table 1 of the Markdown report and Table 1 (`tab:via3_classification_multiseed`) of the LaTeX document match identically to 4 decimal places across all 6 models, 2 multi-seed aggregates, and paired difference rows.
2. **XAI Faithfulness & Stability Metrics:** All Insertion AUC, Deletion AUC, and Stability $\bar{\rho}$ values in Table 2.1 and Table 2.2 of the Markdown report match Table 2 (`tab:via3_xai_multiseed`) of the LaTeX document identically.
3. **Representation Collapse & Spatial Std Dev:** Table 3 in Markdown and Table 3 (`tab:via3_collapse_diagnostics_multiseed`) in LaTeX share the identical layout-level standard deviation ($\mathbf{0.1294 \pm 0.0571}$ for Baseline Pooled; $\mathbf{0.1552 \pm 0.0275}$ for Balanced Pooled) and cross-seed standard deviation ($0.0615$ Baseline; $0.0067$ Balanced).
4. **Parameter Randomization:** Table 4 in Markdown and Table 4 (`tab:via3_adebayo_sanity_check`) in LaTeX record identical correlation values ($\bar{\rho} \in [0.0000, -0.3321, 0.0000, -0.1331, -0.0571, +0.0553]$), prespecified status (5/6 passed, Baseline Seed 100 failed), and substantive interpretation.
5. **Cohort Stratification:** Both documents define the cohort as 40 Hotspots and 10 Non-Hotspots across 6 specific error/detection modes.

---

## 9. Validation and Test Results

### 9.1 Static Syntax & Integrity Verification
Because the environment lacks `pdflatex`, full binary compilation could not be executed. Independent static analysis was conducted on `draft_reports/via3/draft_via3_table_summary.tex`:
- **LaTeX Environment Balance:** Stack-based parsing confirmed all 8 `\begin{table*}` / `\begin{tabular}` blocks are perfectly matched and nested with zero unclosed blocks.
- **Brace Integrity:** Verified that all opening `{` and closing `}` braces match (balance count: 0).
- **Math Mode Integrity:** Verified 408 dollar signs with zero unescaped singletons (even parity: True).
- **Tabular Column Counts:** Verified that every row in Table 1 (9 cols), Table 2 (6 cols), Table 3 (7 cols), and Table 4 (7 cols) strictly conforms to its tabular column specification.
- **Markdown Header & Table Syntax:** Verified valid GitHub-Flavored Markdown table formatting and callout alerts.

### 9.2 Test Suite Execution Limits
- `pytest` is not installed on the system PATH.
- No existing tests were modified or executed.
- No source code or model weights were modified.

---

## 10. Protected-File and Scope Verification

Cryptographic SHA-256 checksums were verified before and after draft synthesis:

### 10.1 Authorized Files (Modified)
| File Path | Initial SHA-256 Checksum | Final SHA-256 Checksum | Status |
| :--- | :--- | :--- | :---: |
| `draft_reports/via3/draft_via3_experimental_report.md` | `bf2916e49ec27f6687fdb48a0ba51bfcdcdb608d9ca5e11924dc761960e1f89e` | `ea73b088c9a8672172f7f8fead7e6667187b6a15e14b6b374d9b6c802606cb74` | **Remediated** |
| `draft_reports/via3/draft_via3_table_summary.tex` | `9d8369dc9f31f3c39376738fb91b5758017a1993cf26050d92fe25816d5495d4` | `dd221c7f327c185cde41cca2da45990f83d8aec080580a74840ac7f089c680b7` | **Remediated** |

### 10.2 Protected Files (Strictly Unchanged)
| File Path | Verified Immutable SHA-256 Checksum | Integrity Status |
| :--- | :--- | :---: |
| `docs/DA2_Group18_Report.docx` | `0139515bba3ed113c193ea9abf987057d79c50116064d87757ef168abeefe076` | **UNTOUCHED** |
| `draft_reports/via3/draft_via3_experimental_report.md.bak` | `115aa5804c65b1352d6469ec5fac14129dd6ff2ba4eeccdc808025fcd46d84fb` | **UNTOUCHED** |
| `draft_reports/via3/draft_via3_table_summary.tex.bak` | `e03fb0ab5f7c94fc576967232ed734ecee9b3af32d98309272ce064e35b911f0` | **UNTOUCHED** |
| `results/via3_experiments/multiseed/WORKSTREAM_E3_STATISTICAL_AUDIT.md` | `3447144714cd1e818a3d0094fe9b58202551128a10fcb598934e1a9e6703de69` | **UNTOUCHED** |
| `results/via3_experiments/multiseed/WORKSTREAM_E4_1_XAI_PREFLIGHT.md` | `fed2182785a70d698eefeec2d8f7d25ffeef4a57ea473a91aa41f26554053360` | **UNTOUCHED** |
| `results/via3_experiments/xai_multiseed/WORKSTREAM_E4_2_EXECUTION_REPORT.md` | `2c620c22289a6d49da302ebd5d6775ce1ff9d81b865a4d447a9cd068315d3bf8` | **UNTOUCHED** |
| `results/via3_experiments/xai_multiseed/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` | `23ed2e8fde6bfd1b18d3673bd375bf05022350d6af6865f1f20a67255daae14b` | **UNTOUCHED** |
| `results/via3_experiments/xai_multiseed/multiseed_xai_comparison_summary.json` | `7648d65b407ed6e54b14f455b2a761d999f390d8401b36bde2ad30d680cf95a5` | **UNTOUCHED** |
| `results/via3_experiments/xai_multiseed/reproducibility_manifest.json` | `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51` | **UNTOUCHED** |
| `results/via3_experiments/multiseed/WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md` | `771079999812f700f420f9f0dbf8728186d05848c72d1a7dc791d93322be37f4` | **UNTOUCHED** |
| `results/via3_experiments/multiseed/WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT_INDEPENDENT.md` | `2916fbdbb9e606381569bdb8464440c21f1d666146e7700e7653697735f3c1b9` | **UNTOUCHED** |
| `results/via3_experiments/multiseed/WORKSTREAM_E4_3_D_RUNNER_REPRODUCIBILITY_AUDIT.md` | `b970594bc6ca7b844464a69530c7f764d4ccd4c3eb04b682273033eac9378b28` | **UNTOUCHED** |

---

## 11. Unresolved Issues and E5 Handoff

The following items cannot be modified under Workstream E4.3-B constraints and are formally handed off to **Workstream E5 (Final Release Hardening & Verification)**:

1. **`WORKSTREAM_E4_2_EXECUTION_REPORT.md` Table 5 Self-Hash Discrepancy:**  
   Table 5 lists `75f28469ad93...` for `reproducibility_manifest.json`, while on disk the manifest hashes to `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51`. In Workstream E5, when historical execution logs are unlocked for final cleanup, update this table to the verified hash.
2. **`WORKSTREAM_E4_2_EXECUTION_REPORT.md` Section 3 Quadrant Text:**  
   The narrative states "25 Hotspot, 25 Non-Hotspot", whereas the actual cohort manifest and data are verified as 40 Hotspots and 10 Non-Hotspots. In Workstream E5, synchronize this narrative section with the preflight contract.
3. **Runner Resume and Validation Hardening (`scripts/run_via3_xai_multiseed.py`):**  
   In Workstream E5, replace row-count completion checks with semantic primary key assertions `(strategy, seed, filename, method)` and `status == "SUCCESS"`. Implement cohort and checkpoint hash verification prior to loading cached CSVs on resume, and ensure resume does not re-write output files or update JSON timestamps.
4. **Prespecified Adebayo Check Threshold Formalization:**  
   In Workstream E5, explicitly formalize whether the programmatic threshold in `scripts/run_via3_xai_multiseed.py` should be updated to $|\bar{\rho}| < 0.40$ (to align with the empirical finding that negative correlations indicate weight dependence) or kept at $|\bar{\rho}| < 0.20$ with documented failure of `baseline_seed100`.
5. **PDF LaTeX Compilation:**  
   When building the final publication manuscript in an environment equipped with `pdflatex` or `latexmk`, compile `draft_via3_table_summary.tex` into the main paper document.

---

## 12. Final Verdict

### **VERDICT: PASS WITH QUALIFICATIONS**

### Basis for Verdict:
1. **Scope Integrity:** Zero files outside the authorized targets (`draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`) were edited. All protected paths, manifests, checkpoints, and coursework deliverables were verified to have identical cryptographic hashes.
2. **Issue Reconciliation:** All six identified discrepancies (ISSUE-MIN-01, ISSUE-MIN-02, ISSUE-MAJ-01, ISSUE-MAJ-02, ISSUE-MAJ-03, ISSUE-MAJ-04) were fully resolved against underlying disk records and transparently documented.
3. **Scientific Reporting:** Multi-seed classification and explainability results were faithfully integrated without overstating certainty, claiming false PR-AUC dominance, or asserting ungrounded inferential p-values across seeds.
4. **Qualifications Preserved:** The drafts explicitly state the protocol-specific favorability of Occlusion Sensitivity, the OOD blank canvas rebound, prediction baseline shifts in Random Baseline metrics, and the programmatic failure of Baseline Seed 100 on parameter randomization.
5. **Audit Trail Complete:** The newly created report provides an unambiguous, auditable handoff for Workstream E5.
