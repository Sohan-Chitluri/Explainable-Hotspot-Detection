# OpenHSDBench `via-3` Multi-Seed Classification & XAI Research Release Candidate (RC1)

**Package Identifier:** `via3_multiseed_rc1`  
**Dataset Identifier:** OpenHSDBench `via-3` Lithography Hotspot Benchmark  
**Release Engineer:** Independent Research Release Engineer (Workstream E6)  
**Baseline Release Gate:** Workstream E5 (`PASS WITH QUALIFICATIONS`)  
**Freeze Status:** **PASS WITH QUALIFICATIONS**  
**Date:** 2026-10-25  

---

## 1. Executive Overview and Scope

This release candidate (`release_candidates/via3_multiseed_rc1/`) packages the complete empirical and audit evidence base for the OpenHSDBench `via-3` lithography hotspot detection project. It captures:

1. **Multi-Seed Binary Classification:** Multi-seed evaluation of custom convolutional neural networks (12,873 parameters) trained under two distinct data-handling strategies—**Baseline (unweighted)** and **Balanced Sampling (50/50 mini-batch balancing)**—across three independent training seeds (**`42`**, **`100`**, and **`2024`**), with a single-seed **Class-Weighted** model evaluated as an auxiliary control. Performance was benchmarked on the clean held-out test cohort ($N=12,803$) using decision thresholds calibrated strictly on validation data ($N=2,846$).
2. **Multi-Seed Explainability (XAI) Robustness:** Explainability evaluations across all six neural network checkpoints using six attribution methods (**Grad-CAM**, **Grad-CAM++**, **LayerCAM**, **Gradient Saliency**, **Occlusion Sensitivity**, and a **Random Baseline**) evaluated on a frozen, stratified 50-layout diagnostic cohort (`xai_sample_cohort_manifest.json`). Evaluations encompass 1,800 pixel insertion/deletion faithfulness curves, 1,500 input-noise stability trials ($\sigma=0.05, T=3$), 300 representation collapse diagnostics, and model parameter randomization checks (Adebayo et al., NeurIPS 2018).
3. **Reconciled Scholarly Drafts:** Two paper-ready reporting drafts (`draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`) incorporating reconciled metrics, explicit separation of layout observations ($N=12,803$ / $N=50$) from seed replicates ($K=3$), corrected table standard deviations, and disclosed methodological caveats.
4. **Audit Trail E1–E5:** The unbroken chain of preflight, execution, statistical, provenance, remediation, and release-gate audits conducted across Workstreams E1 through E5.

> [!IMPORTANT]
> **Operational Scope Disclaimer:** Workstream E6 is strictly an additive, non-destructive packaging and freeze-preparation workstream. Zero experiments were rerun, zero model weights were retrained or modified, and zero underlying empirical data or historical audit reports were altered.

---

## 2. Release Status and Verification Verdict

### Current Verdict: **PASS WITH QUALIFICATIONS**

As determined by the Workstream E5 Final Release Gate audit (`WORKSTREAM_E5_FINAL_RELEASE_GATE.md`) and verified by Workstream E6:

- **Empirical Artifacts (PASS):** All 6 model checkpoints, 18 classification prediction files, 3 master XAI evaluation CSVs (1,800 faithfulness rows, 1,500 stability rows, 300 collapse rows), and the frozen cohort manifest exist on disk, are uncorrupted, and match their canonical cryptographic checksums.
- **Reporting Consistency (PASS):** The Markdown report and LaTeX table summary present mutually consistent metrics and incorporate all defect reconciliations identified in prior audits.
- **Test Suite Pass (PASS):** All 73 automated tests in `tests/` pass cleanly under Python 3.11.16.
- **Qualifications Maintained (QUALIFICATIONS):** This package is released with five explicit scientific and tooling qualifications documented in [`KNOWN_LIMITATIONS.md`](./KNOWN_LIMITATIONS.md) and [`RELEASE_NOTES.md`](./RELEASE_NOTES.md):
  1. *Runner Automation & Resume Mechanics:* `scripts/run_via3_xai_multiseed.py` validates cache completion using row counts rather than semantic composite keys and emits dynamic UTC timestamps upon resume, mutating JSON hashes.
  2. *Historical Audit Report Errata:* Historical report `WORKSTREAM_E4_2_EXECUTION_REPORT.md` retains two immutable historical typos (a pre-final manifest hash and an informal "25 HS / 25 NHS" phrase), which are superseded by the reconciled drafts and frozen manifest.
  3. *Parameter Randomization Sanity Check:* Under the prespecified programmatic rule ($|\bar{\rho}| < 0.20$), 5 of 6 models passed, while Baseline Seed 100 recorded $\bar{\rho} = -0.3321$ (`passed_sanity_check: false`), requiring transparent disclosure.
  4. *Occlusion Sensitivity Protocol Alignment:* Occlusion Sensitivity ($32\times32$ zero-fill) structurally matches the deletion/insertion benchmark primitive, mechanically inflating its metric scores. It is faithful under progressive square zero-fill protocols, not universally best.
  5. *LaTeX PDF Binary Compilation:* System environment lacks binary `pdflatex`; LaTeX syntax was validated via complete static stack parsing rather than PDF rendering.

---

## 3. Navigation of Reconciled Reporting Drafts and Audits

### 3.1 Authoritative Reconciled Drafts
The canonical reporting deliverables for publication are located in `draft_reports/via3/` (with byte-for-byte identical reference copies preserved in `./reports/`):
- **Remediated Markdown Experimental Report:**  
  In-repo path: `draft_reports/via3/draft_via3_experimental_report.md`  
  Candidate copy: [`reports/draft_via3_experimental_report.md`](./reports/draft_via3_experimental_report.md)  
  SHA-256: `ea73b088c9a8672172f7f8fead7e6667187b6a15e14b6b374d9b6c802606cb74`
- **Remediated LaTeX Table Summary:**  
  In-repo path: `draft_reports/via3/draft_via3_table_summary.tex`  
  Candidate copy: [`reports/draft_via3_table_summary.tex`](./reports/draft_via3_table_summary.tex)  
  SHA-256: `dd221c7f327c185cde41cca2da45990f83d8aec080580a74840ac7f089c680b7`

### 3.2 Audit Trail Reports (Workstreams E1 through E5)
All audit reports are preserved in repository paths, with identical copies preserved in `./reports/`:
1. **E1 Preflight:** [`reports/WORKSTREAM_E1_PREFLIGHT.md`](./reports/WORKSTREAM_E1_PREFLIGHT.md) (`b457e88efa31...`) — Multi-seed training pipeline contract and split verification.
2. **E2 Execution:** [`reports/WORKSTREAM_E2_EXECUTION_REPORT.md`](./reports/WORKSTREAM_E2_EXECUTION_REPORT.md) (`e15c59a87046...`) — Training dynamics, convergence, and checkpoint generation for seeds 100 and 2024.
3. **E3 Statistical Audit:** [`reports/WORKSTREAM_E3_STATISTICAL_AUDIT.md`](./reports/WORKSTREAM_E3_STATISTICAL_AUDIT.md) (`3447144714cd...`) — Statistical inference, paired test limits ($K=3$), and threshold sweep analysis.
4. **E4.1 XAI Preflight:** [`reports/WORKSTREAM_E4_1_XAI_PREFLIGHT.md`](./reports/WORKSTREAM_E4_1_XAI_PREFLIGHT.md) (`fed2182785a7...`) — Multi-seed XAI evaluation specification and cohort freezing.
5. **E4.2 Execution:** [`reports/WORKSTREAM_E4_2_EXECUTION_REPORT.md`](./reports/WORKSTREAM_E4_2_EXECUTION_REPORT.md) (`2c620c22289a...`) — Execution log for 1,800 faithfulness and 1,500 stability evaluations.
6. **E4.2 XAI Robustness:** [`reports/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md`](./reports/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md) (`23ed2e8fde6bf...`) — Cross-seed attribution robustness, collapse analysis, and noise stability.
7. **E4.3-A Provenance Audit:** [`reports/WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md`](./reports/WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md) (`771079999812...`) — Forensic provenance and integrity audit of E4.2 records.
8. **E4.3-B Remediation:** [`reports/WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md`](./reports/WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md) (`d0f3d06b9308...`) — Resolution of manuscript defects (ISSUE-MAJ-01 to 04, ISSUE-MIN-01 to 02).
9. **E5 Release Gate:** [`reports/WORKSTREAM_E5_FINAL_RELEASE_GATE.md`](./reports/WORKSTREAM_E5_FINAL_RELEASE_GATE.md) (`a66e40af33ed...`) — Comprehensive final audit awarding PASS WITH QUALIFICATIONS.

---

## 4. Empirical Artifact Overview

### 4.1 Checkpoints (`models/via3_multiseed/`)
Six custom CNN models with 18 layers and 12,873 trainable parameters (230,689 bytes each):
- Baseline Seed 42: `via3_baseline_seed42.keras` (`97a9cb8bf776...`)
- Baseline Seed 100: `via3_baseline_seed100.keras` (`f28e42777387...`)
- Baseline Seed 2024: `via3_baseline_seed2024.keras` (`9347e568cbe5...`)
- Balanced Sampling Seed 42: `via3_balanced_sampling_seed42.keras` (`551a861f1e35...`)
- Balanced Sampling Seed 100: `via3_balanced_sampling_seed100.keras` (`c957304bf07a...`)
- Balanced Sampling Seed 2024: `via3_balanced_sampling_seed2024.keras` (`297fa95748ad...`)

### 4.2 Classification Prediction Records (`results/via3_experiments/multiseed/`)
18 CSV prediction files covering 2 strategies $\times$ 3 seeds $\times$ 3 partitions:
- Clean Test Split ($N=12,803$ rows each): 6 files
- Official Test Split ($N=12,905$ rows each): 6 files
- Validation Split ($N=2,846$ rows each): 6 files

### 4.3 Master XAI Records (`results/via3_experiments/xai_multiseed/`)
- `master_faithfulness_metrics.csv` (1,800 rows, 100% SUCCESS, 0 nulls, 0 duplicates): `0b49d2c414e5...`
- `master_stability_metrics.csv` (1,500 rows, 100% SUCCESS, 0 nulls, 0 duplicates): `fabbaee2eae...`
- `master_collapse_diagnostics.csv` (300 rows, 100% SUCCESS, 0 nulls, 0 duplicates): `8db96f2daff...`
- `multiseed_xai_comparison_summary.json` (3-seed summary, 1,117 lines): `7648d65b407...`
- `reproducibility_manifest.json` (provenance manifest, 45 lines): `9655c3480db...`

### 4.4 Frozen Diagnostic Cohort Manifest
- `results/via3_experiments/xai_multiseed/cohort/xai_sample_cohort_manifest.json` (50 layouts: 40 HS / 10 NHS, SHA-256: `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb`).

---

## 5. Candidate Structure vs. In-Repository References

To adhere to reproducible packaging principles without duplicating large binary datasets or model weights, this release candidate differentiates between:

1. **Included in Candidate (`included_in_candidate`):**
   - Candidate documentation: `README.md`, `RELEASE_NOTES.md`, `REPRODUCIBILITY.md`, `KNOWN_LIMITATIONS.md`, `RELEASE_CHECKLIST.md`.
   - Manifests & checksums: `ARTIFACT_INVENTORY.csv`, `SHA256SUMS`.
   - Document copies: `reports/*.md` and `reports/*.tex` verified as byte-for-byte identical to authoritative drafts and audits.
2. **Referenced in Repository (`referenced_in_repository`):**
   - Neural network checkpoints (`models/via3_multiseed/*.keras`).
   - Master XAI CSVs and JSON summaries (`results/via3_experiments/xai_multiseed/*`).
   - Classification prediction CSVs (`results/via3_experiments/multiseed/**/*.csv`).
   - Reproducibility manifests (`reproducibility_manifest.json`).
   - Runner scripts (`scripts/*.py`).
   - Repository test suite (`pytest.ini`, `tests/*.py`).

---

## 6. How to Navigate and Verify the Release

1. **Inspect Artifact Inventory:** Open [`ARTIFACT_INVENTORY.csv`](./ARTIFACT_INVENTORY.csv) for a machine-readable catalog of all 62 repository and release artifacts, their categories, availability, and SHA-256 digests.
2. **Verify Candidate Checksums:** From the `release_candidates/via3_multiseed_rc1/` directory, execute:
   ```bash
   sha256sum -c SHA256SUMS
   ```
   All entries must report `OK`.
3. **Review Reproducibility Workflow:** Consult [`REPRODUCIBILITY.md`](./REPRODUCIBILITY.md) for details on recomputing metrics, inspecting static CSVs, and executing the automated test suite.
4. **Consult Limitations & Disclosures:** Review [`KNOWN_LIMITATIONS.md`](./KNOWN_LIMITATIONS.md) before publishing or citing metrics in external manuscripts.
5. **Inspect the Release Checklist:** Check [`RELEASE_CHECKLIST.md`](./RELEASE_CHECKLIST.md) for the complete sign-off matrix.
