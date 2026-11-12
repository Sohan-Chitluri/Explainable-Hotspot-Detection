# Workstream E5 — Final Release Gate & Multi-Seed Freeze Audit

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3`  
**Evaluated Scope:** Multi-Seed Classification, Explainability (XAI), Reproducibility Manifest, Runner Infrastructure, and Paper Drafts  
**Auditor:** Independent Release & Reproducibility Auditor  
**Audit Date:** 2026-10-25  
**Audit Status:** Complete  
**Final Release Gate Verdict:** **PASS WITH QUALIFICATIONS**

---

## 1. Executive Summary and Final Verdict

### Final Verdict: **PASS WITH QUALIFICATIONS**

This report delivers the comprehensive, evidence-based final release audit for the OpenHSDBench `via-3` multi-seed classification and explainability research package.

The multi-seed experimental campaign spanning Workstreams E1 through E4.3-B was subjected to an exhaustive read-only inspection. The empirical data, trained neural network checkpoints, per-sample predictions, attribution curves, representation collapse diagnostics, and paper-reporting drafts are **empirically valid, internally consistent, cryptographically traceable, and structurally sound**.

### Summary of Core Release Findings:
1. **Empirical Data & Model Checkpoint Integrity (PASS):** All six custom CNN checkpoints (`via3_baseline_seed{42,100,2024}.keras` and `via3_balanced_sampling_seed{42,100,2024}.keras`) exist on disk, have identical 18-layer architectures with 12,873 trainable parameters, and match their canonical SHA-256 checksums byte-for-byte.
2. **Master Measurement Matrix Completeness (PASS):** All 1,800 faithfulness evaluations, 1,500 input-noise stability evaluations, and 300 representation collapse diagnostics in `results/via3_experiments/xai_multiseed/` are 100% complete, have 0 duplicate keys, contain 0 missing/NaN values, and report 100% status `SUCCESS`.
3. **Draft Synthesis & Defect Remediation (PASS):** Both authorized drafts (`draft_reports/via3/draft_via3_experimental_report.md` and `draft_reports/via3/draft_via3_table_summary.tex`) accurately integrate multi-seed classification and XAI results, distinguish layout evaluation observations from independent training seed replicates, correct the Table 4 typographical error ($0.0571$ layout SD vs. $0.0615$ seed SD), document the verified 40 HS / 10 NHS diagnostic cohort composition, and acknowledge the protocol-specific favorability of Occlusion Sensitivity.
4. **Test Suite Verification (PASS):** All 73 automated tests in the repository test suite pass cleanly (100% pass rate in 13.13 seconds) under Python 3.11.16.
5. **Release Qualifications (QUALIFICATIONS):** While the empirical data is uncorrupted, the package cannot be awarded an unqualified "push-button" reproducibility freeze due to three specific non-blocking limitations:
   - *Runner Resume & Timestamp Mechanics:* `scripts/run_via3_xai_multiseed.py` checks completion via total row counts rather than semantic primary keys and generates fresh UTC timestamps upon resume, mutating JSON hashes.
   - *Historical Execution Report Typo:* Table 5 in historical `WORKSTREAM_E4_2_EXECUTION_REPORT.md` lists `75f28469ad...` for `reproducibility_manifest.json` (captured prior to environment finalization), whereas the actual on-disk manifest hashes to `9655c3480db1...`.
   - *Sanity Check Prespecification:* Under the prespecified programmatic threshold ($|\bar{\rho}| < 0.20$), 5 of 6 models passed, while Baseline Seed 100 recorded $\bar{\rho} = -0.3321$ (`passed_sanity_check: false`), requiring transparent scientific qualification rather than assertions of post-hoc perfection.
   - *TeX Compilation:* Binary `pdflatex` is not installed on the system PATH; LaTeX tables were validated via complete static syntax and stack parsing rather than PDF binary compilation.

The package is **fully suitable for a qualified research release and paper submission**, provided the documented limitations and qualifications accompany the release notes.

---

## 2. Scope and Environment

### 2.1 Audit Boundaries
- **Operating Mode:** Pure read-only release audit. Zero files were modified, renamed, or deleted during the evaluation. Zero GPU training or neural network inference was initiated.
- **Protected Paths Inspected:** Coursework deliverables (`docs/DA2_Group18_Report.docx`, `latex_report/`), canonical drafts (`draft_reports/via3/`), checkpoints (`models/`), raw experiment metrics (`results/`), and source code (`scripts/`, `src/`, `tests/`).

### 2.2 Host & Tooling Telemetry
- **Operating System:** Linux 6.12.35 (x86_64 NixOS)
- **Host Time:** 2026-10-25T21:48:00+05:30
- **GPU Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (PCI Bus `0000:01:00.0`, Driver 595.99.02, CUDA 13.2)
- **Virtual Environment:** Python 3.11.16 located at `/home/peskybird/Projects/Ai_Ml_DA/.venv/bin/python`
  - TensorFlow: `2.15.1`
  - NumPy: `1.26.4`
  - Pytest: `9.1.1`
  - Pandas: `2.2.3`
  - Scipy: `1.15.2`

---

## 3. Artifact Inventory and Hashes

Every core artifact in the multi-seed release package was verified for physical existence, file size, and SHA-256 cryptographic digest:

### 3.1 Model Checkpoints (`models/via3_multiseed/`)
All six checkpoints instantiate a custom CNN with exactly 12,873 trainable parameters.

| Checkpoint Identifier | Strategy | Seed | File Size | SHA-256 Cryptographic Checksum | Provenance Status |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `via3_baseline_seed42.keras` | Baseline | 42 | 230,689 B | `97a9cb8bf776b76b6ed1c0fe6a38b4de31138c97cfb64a1b29ff2f228a852887` | Audited Reference (Verified) |
| `via3_baseline_seed100.keras` | Baseline | 100 | 230,689 B | `f28e4277738794a64d557fd5a5c3eeead32d32f8cbf08642ad135554746d3455` | Trained in E2 (Epoch 15) |
| `via3_baseline_seed2024.keras` | Baseline | 2024 | 230,689 B | `9347e568cbe59672f295bd3865bc9708d0a279ecd5c4e94ddf54a1b96cf5ae0d` | Trained in E2 (Epoch 15) |
| `via3_balanced_sampling_seed42.keras` | Balanced | 42 | 230,689 B | `551a861f1e3537b93144edf22b5d2bc71456548ca576aea5a3781021fa8ac426` | Audited Reference (Verified) |
| `via3_balanced_sampling_seed100.keras` | Balanced | 100 | 230,689 B | `c957304bf07a8aabf9a7861c4b2d5bf1c050c87bba0454ae4cbed9d086ef5815` | Trained in E2 (Epoch 4) |
| `via3_balanced_sampling_seed2024.keras` | Balanced | 2024 | 230,689 B | `297fa95748adc7a1234467cee898f14872942d5d4b1484e539bd1308f39b89e9` | Trained in E2 (Epoch 12) |

### 3.2 Master XAI Records (`results/via3_experiments/xai_multiseed/`)
| Artifact File | Record Dimension | File Size | SHA-256 Checksum | Schema & Integrity |
| :--- | :---: | :---: | :--- | :---: |
| `master_faithfulness_metrics.csv` | 1,800 rows | 214,693 B | `0b49d2c414e54a7965b4d6c6d09e008f41385e40ef5d4cbbe810ad3228519ad0` | 100% SUCCESS, 0 duplicates |
| `master_stability_metrics.csv` | 1,500 rows | 169,554 B | `fabbaee2eaeef5c4ea3f51fd64a40332492f5e94d6bb78158b9a2288135124aa` | 100% SUCCESS, 0 duplicates |
| `master_collapse_diagnostics.csv` | 300 rows | 35,799 B | `8db96f2daffb25c20ad36895468ead267c1e99a8b89ee112e9bb62892bd5bfc3` | 100% SUCCESS, 0 duplicates |
| `multiseed_xai_comparison_summary.json` | 1,117 lines | 29,847 B | `7648d65b407ed6e54b14f455b2a761d999f390d8401b36bde2ad30d680cf95a5` | Exact match with master CSVs |
| `reproducibility_manifest.json` | 45 lines | 1,996 B | `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51` | Complete output file digests |

### 3.3 Cohort Manifests & Authoritative Drafts
| Artifact File | Size | SHA-256 Checksum | Role in Release |
| :--- | :---: | :--- | :--- |
| `xai_sample_cohort_manifest.json` (canonical) | 9,719 B | `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb` | Frozen 50-layout cohort |
| `xai_sample_cohort_manifest.json` (mirrored) | 9,719 B | `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb` | Mirror copy in `xai_multiseed/cohort/` |
| `draft_via3_experimental_report.md` | 15,225 B | `ea73b088c9a8672172f7f8fead7e6667187b6a15e14b6b374d9b6c802606cb74` | Remediated Markdown report |
| `draft_via3_table_summary.tex` | 8,971 B | `dd221c7f327c185cde41cca2da45990f83d8aec080580a74840ac7f089c680b7` | Remediated LaTeX tables |
| `WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md` | 36,651 B | `d0f3d06b93088cd01b7a6b91000e90ccba0c7b8e89ddea305ac23cac3940b5a6` | Authoritative remediation audit |

---

## 4. Classification and XAI Record Validation

### 4.1 Classification Records (Clean Held-Out Cohort, $N=12,803$)
All 18 prediction CSV files across 2 strategies, 3 seeds, and 3 partitions exist and match expected row counts (Clean Test: 12,803; Official Test: 12,905; Validation: 2,846).

The independently recalculated clean test metrics at validation-calibrated operating thresholds $\tau^*$ confirm:
- **Baseline (Calibrated $\tau^* \in [0.01, 0.08]$):**
  - Operating Threshold $\tau^*$: $0.0533 \pm 0.0379$ [$s_{42}: 0.08, s_{100}: 0.07, s_{2024}: 0.01$]
  - PR-AUC: $0.2749 \pm 0.0233$ [$s_{42}: 0.2518, s_{100}: 0.2983, s_{2024}: 0.2745$]
  - ROC-AUC: $0.7598 \pm 0.0234$ [$s_{42}: 0.7328, s_{100}: 0.7739, s_{2024}: 0.7728$]
  - HS Recall: $0.2384 \pm 0.0544$ [$s_{42}: 0.2214, s_{100}: 0.2993, s_{2024}: 0.1946$]
  - HS Precision: $0.4223 \pm 0.1301$ [$s_{42}: 0.3640, s_{100}: 0.3315, s_{2024}: 0.5714$]
  - HS F1 Score: $0.2934 \pm 0.0198$ [$s_{42}: 0.2753, s_{100}: 0.3146, s_{2024}: 0.2904$]
  - Balanced Accuracy: $0.6129 \pm 0.0236$ [$s_{42}: 0.6043, s_{100}: 0.6396, s_{2024}: 0.5949$]
- **Balanced Sampling (Calibrated $\tau^* \in [0.53, 0.76]$):**
  - Operating Threshold $\tau^*$: $0.6767 \pm 0.1274$ [$s_{42}: 0.76, s_{100}: 0.74, s_{2024}: 0.53$]
  - PR-AUC: $0.2635 \pm 0.0476$ [$s_{42}: 0.2619, s_{100}: 0.2168, s_{2024}: 0.3120$]
  - ROC-AUC: $0.7760 \pm 0.0273$ [$s_{42}: 0.7840, s_{100}: 0.7456, s_{2024}: 0.7985$]
  - HS Recall: $0.2733 \pm 0.0112$ [$s_{42}: 0.2603, s_{100}: 0.2798, s_{2024}: 0.2798$]
  - HS Precision: $0.3680 \pm 0.1006$ [$s_{42}: 0.4038, s_{100}: 0.2544, s_{2024}: 0.4457$]
  - HS F1 Score: $0.3090 \pm 0.0392$ [$s_{42}: 0.3166, s_{100}: 0.2665, s_{2024}: 0.3438$]
  - Balanced Accuracy: $0.6281 \pm 0.0054$ [$s_{42}: 0.6238, s_{100}: 0.6263, s_{2024}: 0.6341$]
- **Paired Strategy Differences ($\Delta = \text{Balanced} - \text{Baseline}$):**
  - $\Delta \text{PR-AUC}$: $-0.0114 \pm 0.0623$ (Inconsistent direction: $+0.0101, -0.0815, +0.0375$)
  - $\Delta \text{ROC-AUC}$: $+0.0162 \pm 0.0406$ (Inconsistent direction: $+0.0512, -0.0283, +0.0256$)
  - $\Delta \text{HS Recall}$: $+0.0349 \pm 0.0524$ (Inconsistent direction: $+0.0389, -0.0195, +0.0852$)
  - $\Delta \tau^*$: $+0.6233 \pm 0.0896$ (Consistent positive: $+0.6800, +0.6700, +0.5200$)
- **Default-Threshold Catastrophic Failure in Baseline:** At standard uncalibrated $\tau = 0.50$, Baseline Seed 2024 output probabilities were bounded below $0.4687$, resulting in **0 hotspots detected out of 411** ($\text{Recall} = 0.0\%$). Balanced Sampling maintained Recall $\ge 27.98\%$ across all seeds at default $\tau = 0.50$.

### 4.2 XAI Master Records Audit
Direct automated parsing of the three master CSV files verified:
- **`master_faithfulness_metrics.csv`:** Exactly 1,800 rows ($2\text{ strategies} \times 3\text{ seeds} \times 50\text{ layouts} \times 6\text{ methods}$). Primary key `(strategy, seed, filename, method)` is 100% unique (0 duplicates). Status is 100% `SUCCESS`. Range: Deletion AUC $[0.0003, 0.9830]$, Insertion AUC $[0.0033, 0.9956]$. Zero nulls, zero NaNs.
- **`master_stability_metrics.csv`:** Exactly 1,500 rows ($2\text{ strategies} \times 3\text{ seeds} \times 50\text{ layouts} \times 5\text{ deterministic methods}$). Random baseline is intentionally omitted. Primary key is 100% unique (0 duplicates). Status is 100% `SUCCESS`. Range: Spearman $\bar{\rho} \in [-0.4107, 1.0000]$. Zero nulls, zero NaNs.
- **`master_collapse_diagnostics.csv`:** Exactly 300 rows ($2\text{ strategies} \times 3\text{ seeds} \times 50\text{ layouts}$). Primary key `(strategy, seed, filename)` is 100% unique (0 duplicates). Status is 100% `SUCCESS`. Spatial standard deviation range $[0.0000, 0.2269]$.

---

## 5. Scientific Consistency and Reporting Verification

The two remediated draft reports (`draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`) were audited against the underlying data records:

1. **Reconciliation of Table 4 Typographical Error (ISSUE-MIN-01):**
   - In earlier reports, `0.0589` was mistakenly reported as the pooled baseline spatial standard deviation.
   - The drafts now correctly state $\mathbf{0.1294 \pm 0.0571}$ as the sample standard deviation across all 150 layout observations ($ddof=1$), and $\mathbf{0.1294 \pm 0.0615}$ as the cross-seed standard deviation across the three seed means ($K=3, ddof=1$).
2. **Aggregation Unit Discipline (ISSUE-MIN-02):**
   - The drafts explicitly distinguish layout-level observations ($N=12,803$ or $N=50$) from independent model training replicates ($K=3$).
   - Table headers, columns, and footnotes explicitly label whether dispersion reflects layout variance ($ddof=1, N=50$ or $150$) or training-seed variance ($ddof=1, K=3$).
   - Normal-theory Student's $t$ confidence intervals have been eliminated.
3. **Cohort Ground-Truth Composition (ISSUE-MAJ-01):**
   - Both drafts define the 50-layout diagnostic cohort as **40 Hotspot and 10 Non-Hotspot layouts** across six specific error and recovery modes (10 `shared_tp`, 10 `shared_fn`, 10 `balanced_won`, 10 `baseline_won`, 5 `shared_tn`, 5 `shared_fp`).
   - The narrative typo in earlier execution reports ("25 HS / 25 NHS") is explicitly cited, refuted, and corrected.
4. **Parameter Randomization Prespecified Status (ISSUE-MAJ-03):**
   - The drafts transparently state that under the prespecified programmatic threshold ($|\bar{\rho}| < 0.20$), 5 of 6 models passed, while Baseline Seed 100 recorded $\bar{\rho} = -0.3321$ (`passed_sanity_check: false`).
   - The substantive interpretation (negative correlation confirms weight sensitivity rather than edge invariance) is clearly distinguished from post-hoc threshold relaxation.
5. **Occlusion Sensitivity Protocol Qualification:**
   - Both drafts acknowledge that Occlusion Sensitivity ($32\times32$ sliding zero-fill patch) structurally matches the perturbation mechanics of the insertion/deletion benchmark (progressive zero-fill), mechanically favoring Occlusion over gradient methods. It is reported as faithful *under this perturbation protocol*, not universally best.
6. **Cross-Draft Numerical Consistency:**
   - Every metric, standard deviation, and paired difference matches with 100% precision between the Markdown report and LaTeX tables.

---

## 6. Manifest and Provenance Audit

Audit of `results/via3_experiments/xai_multiseed/reproducibility_manifest.json`:
- **Current On-Disk Hash:** `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51`
- **Execution Report Discrepancy (ISSUE-MAJ-02):** Table 5 in `WORKSTREAM_E4_2_EXECUTION_REPORT.md` lists `75f28469ad9373cf58bb06cb38ae59d5f7b8faec7bf2987a0774a88ae8210373`. This historical report digest reflects an earlier intermediate generation prior to environment metadata injection. All output artifact hashes *recorded inside* the manifest match on-disk files byte-for-byte.

### Manifest Scope Audit:
- **Adequately Recorded:**
  - Checkpoint SHA-256 hashes (all 6 match on disk).
  - Diagnostic cohort manifest SHA-256 (`1c547c162c32df...`).
  - Runtime environment versions: Python (`3.11.16`), TensorFlow (`2.15.1`), NumPy (`1.26.4`).
  - Master output file paths, row counts, and SHA-256 hashes.
  - Evaluation random seeds (`[42, 100, 2024]`), perturbation seed (`42`), and baseline seed (`42`).
- **Omitted from Manifest (Documented for Release Notes):**
  - Exact Git commit SHA and branch dirty status.
  - Attribution hyperparameters (perturbation steps: 10, target layer: `conv2d_5`, target logit: $+z_{\text{HS}}$, occlusion window: $32\times32$, stride: $16\times16$, noise magnitude: $\sigma=0.05$).
  - Versions of Pandas (`2.2.3`) and Scipy (`1.15.2`).
  - Image canvas resolution ($224\times224\times3$) and normalization range ($[0.0, 1.0]$).

---

## 7. Runner and Resume-Safety Audit

Inspection of `scripts/run_via3_xai_multiseed.py` revealed three specific implementation vulnerabilities affecting automated rerun safety:

1. **Blind Row-Count Completion Guard (`lines 108–132`):**
   ```python
   # scripts/run_via3_xai_multiseed.py:127-129
   if len(df_faith) != expected_faith or len(df_stab) != expected_stab or len(df_coll) != expected_coll:
       return False
   return True
   ```
   *Vulnerability:* The guard validates only `len(df) == expected`, ignoring semantic primary keys `(strategy, seed, filename, method)` and status flags (`status == "SUCCESS"`). If an evaluation crashed on all samples and wrote 300 error rows, the resume logic would falsely treat the run as complete.
2. **Unvalidated Cache Reuse on Resume (`lines 512–535`):**
   When resuming, the script loads cached CSVs without asserting that the on-disk model checkpoint or cohort manifest matches expected SHA-256 hashes.
3. **Unconditional Master Overwrite & Timestamp Mutation (`lines 630 & 647–686`):**
   Resuming a fully completed run re-writes master CSVs and stamps `multiseed_xai_comparison_summary.json` with a fresh UTC timestamp (`time.strftime(...)`), mutating its SHA-256 hash and invalidating the reproducibility manifest.

*Classification:* These vulnerabilities represent future runner and automation limitations; they do **not** corrupt the existing E4.2 master records on disk, which were independently audited and verified as complete.

---

## 8. Test and LaTeX Verification

### 8.1 Automated Test Suite Execution
The complete repository test suite was executed in the project virtual environment:
```bash
/home/peskybird/Projects/Ai_Ml_DA/.venv/bin/pytest tests/
```
**Result:** **73 passed in 13.13s (100% pass rate)**.

All test modules passed without errors or warnings:
- `tests/test_experiment_contract.py`: 6 passed
- `tests/test_faithfulness_and_xai.py`: 6 passed
- `tests/test_iccad5_collapse_mechanisms.py`: 8 passed
- `tests/test_manuscript_numerical_consistency.py`: 7 passed
- `tests/test_metrics_and_occlusion.py`: 11 passed
- `tests/test_preprocessing_legacy_diff.py`: 1 passed
- `tests/test_safe_eval_output.py`: 2 passed
- `tests/test_three_way_split.py`: 15 passed
- `tests/test_via3_workstreams.py`: 16 passed
- `tests/test_xai_preprocessing.py`: 1 passed

### 8.2 LaTeX Structural Verification
Because binary `pdflatex` is not installed on the system PATH, full PDF compilation could not be executed. Static syntax analysis was executed via independent Python stack parsing:
- **Environment Nesting:** All 8 `\begin{table*}` / `\begin{tabular}` blocks are perfectly matched and nested.
- **Brace Parity:** All opening `{` and closing `}` braces are balanced (0 open braces).
- **Math Parity:** All 408 dollar signs are paired (even parity: True).
- **Column Counts:** All rows in Table 1 (9 cols), Table 2 (6 cols), Table 3 (7 cols), and Table 4 (7 cols) strictly match their column definitions.

---

## 9. Protected-File and Scope Verification

All protected files were audited before and after execution:
- `docs/DA2_Group18_Report.docx`: SHA-256 `0139515bba3ed113c193ea9abf987057d79c50116064d87757ef168abeefe076` (**Immutable**)
- `draft_reports/via3/draft_via3_experimental_report.md.bak`: SHA-256 `115aa5804c65b1352d6469ec5fac14129dd6ff2ba4eeccdc808025fcd46d84fb` (**Immutable**)
- `draft_reports/via3/draft_via3_table_summary.tex.bak`: SHA-256 `e03fb0ab5f7c94fc576967232ed734ecee9b3af32d98309272ce064e35b911f0` (**Immutable**)
- `WORKSTREAM_E3_STATISTICAL_AUDIT.md`: SHA-256 `3447144714cd1e818a3d0094fe9b58202551128a10fcb598934e1a9e6703de69` (**Immutable**)
- `WORKSTREAM_E4_2_EXECUTION_REPORT.md`: SHA-256 `2c620c22289a6d49da302ebd5d6775ce1ff9d81b865a4d447a9cd068315d3bf8` (**Immutable**)
- `WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md`: SHA-256 `23ed2e8fde6bfd1b18d3673bd375bf05022350d6af6865f1f20a67255daae14b` (**Immutable**)
- `WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md`: SHA-256 `d0f3d06b93088cd01b7a6b91000e90ccba0c7b8e89ddea305ac23cac3940b5a6` (**Immutable**)

---

## 10. Release-Readiness Matrix

| Category | Status | Evidence Source | Unresolved Issues | Minimum Required Action |
| :--- | :---: | :--- | :--- | :--- |
| **1. Artifact Integrity** | **PASS** | Checkpoint SHA-256 hashes, file sizes, parameter counts (12,873). | None. | Freeze model weights and directory permissions. |
| **2. Classification Reproducibility** | **PASS** | All 18 prediction CSVs, validation PR sweeps, clean test recalculation. | None. | Cite multi-seed dispersion (`Mean ± SD [42, 100, 2024]`). |
| **3. XAI Record Integrity** | **PASS** | Master CSVs: 1,800 faithfulness, 1,500 stability, 300 collapse rows. | None. | Retain master CSVs as immutable release deliverables. |
| **4. Scientific Consistency** | **PASS** | Reconciled drafts (`draft_via3_*`), refuted 25/25 typo, corrected Table 4 SD ($0.0571$). | None. | Ensure downstream paper camera-ready matches remediated drafts. |
| **5. Manifest and Provenance** | **PASS WITH QUALIFICATIONS** | `reproducibility_manifest.json` matches output CSVs byte-for-byte. | Manifest hash mismatch with historical E4.2 report Table 5 (`75f284...` vs `9655c3...`); omitted hyperparameters. | Accompany release with manifest errata note documenting hash update. |
| **6. Runner and Resume Safety** | **PASS WITH QUALIFICATIONS** | Trace of `run_via3_xai_multiseed.py`. | Row-count completion checks, unverified cache reuse, timestamp mutation upon resume. | Document that automated push-button rerun requires future runner hardening. |
| **7. Test Verification** | **PASS** | Full test execution: 73 passed in 13.13s. | None. | Maintain virtualenv pytest configuration. |
| **8. Draft/Report Consistency** | **PASS** | Exact numerical and semantic match across Markdown and LaTeX. | PDF binary compilation unverified due to lack of local `pdflatex`. | Compile final camera-ready PDF on a build system with TeX Live. |
| **9. Protected-File Integrity** | **PASS** | Pre- and post-audit SHA-256 hashes match across all protected paths. | None. | Maintain read-only boundaries. |

---

## 11. Release Blockers and Qualifications

### 11.1 Release Blockers: **NONE**
There are **zero blocking data defects, corrupted records, missing checkpoints, or invalid test results**. The empirical evidence supports the scientific conclusions of the research project.

### 11.2 Release Qualifications
1. **Runner Resume Fragility:** Re-running `scripts/run_via3_xai_multiseed.py` with resume relies on row counts rather than primary keys and re-stamps output JSONs with fresh wall-clock timestamps. Automated zero-intervention rerun guarantees cannot be claimed without runner hardening.
2. **Historical Report Typographical Errata:** Historical report `WORKSTREAM_E4_2_EXECUTION_REPORT.md` retains two known documentation defects: Table 5 cites an intermediate manifest hash (`75f28469ad...`), and Section 3 states "25 HS / 25 NHS". These remain preserved to maintain historical audit immutability, but are fully corrected in the release drafts.
3. **Parameter Randomization Prespecification:** Baseline Seed 100 recorded $\bar{\rho} = -0.3321$, failing the prespecified programmatic check ($|\bar{\rho}| < 0.20$). While negative correlation demonstrates genuine parameter sensitivity, 5 of 6 models passed programmatically.
4. **Occlusion Protocol Alignment:** Occlusion Sensitivity's high insertion AUC ($0.8422$) is protocol-favored by square zero-masking and must be qualified as protocol-specific rather than universally best.
5. **LaTeX PDF Build Dependency:** Final PDF generation was not executed locally due to the absence of `pdflatex`.

---

## 12. Minimal Remediation Plan (Post-E5 Engineering Roadmap)

To transition from a qualified research release to an automated, push-button benchmark:

1. **Harden Runner Completion and Resume (`scripts/run_via3_xai_multiseed.py`):**
   - Update `is_checkpoint_xai_completed` to assert:
     ```python
     len(df[df["status"] == "SUCCESS"]) == expected_rows
     set(df["filename"]) == set(expected_filenames)
     ```
   - Validate checkpoint SHA-256 against `EXPECTED_CHECKPOINTS` before reusing cached outputs on resume.
2. **Deterministic Output Serialization:**
   - In `master_summary`, accept an optional `--timestamp` argument or reuse the cohort manifest timestamp to ensure identical SHA-256 hashes when rebuilding master summaries from cached seed outputs.
3. **Enrich Reproducibility Manifest:**
   - Inject Git commit SHA, attribution method list, target layer name, occlusion window/stride, and noise magnitude $\sigma$ into `reproducibility_manifest.json`.
4. **Camera-Ready PDF Build:**
   - Execute `pdflatex draft_via3_table_summary.tex` on a CI/CD server with TeX Live installed to generate final camera-ready vector tables.

---

## 13. Reproducible Release Checklist

For independent verification by downstream researchers:
- [x] Checkpoint weights verified against SHA-256 digests in Section 3.1.
- [x] Clean test evaluation matches PR-AUC ($0.2749$ Baseline vs $0.2635$ Balanced) and Recall ($0.2384$ vs $0.2733$).
- [x] Diagnostic cohort frozen at 40 HS / 10 NHS (`xai_sample_cohort_manifest.json`).
- [x] Master faithfulness CSV contains exactly 1,800 rows with status `SUCCESS`.
- [x] Master stability CSV contains exactly 1,500 rows with status `SUCCESS`.
- [x] Master collapse CSV contains exactly 300 rows with status `SUCCESS`.
- [x] Full test suite executes cleanly (`pytest tests/` -> 73 passed).
- [x] LaTeX table definitions validated for nesting, braces, and column parity.
- [x] Reporting drafts accurately state multi-seed dispersion, spatial standard deviation ($0.0571$), and Adebayo check status.
- [x] All coursework deliverables (`docs/DA2_Group18_Report.docx`, `latex_report/`) remain unaltered.
