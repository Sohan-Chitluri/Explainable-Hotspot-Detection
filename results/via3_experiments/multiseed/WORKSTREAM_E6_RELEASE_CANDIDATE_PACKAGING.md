# Workstream E6 — Research Release Candidate Packaging & Freeze Preparation Report

**Target Repository:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Dataset:** OpenHSDBench `via-3` Lithography Hotspot Benchmark  
**Role:** Independent Research Release Engineer  
**Workstream:** E6 — Packaging & Freeze Preparation  
**Baseline Gate Verdict:** Workstream E5 (`PASS WITH QUALIFICATIONS`)  
**E6 Release Candidate Verdict:** **PASS WITH QUALIFICATIONS**  
**Date:** 2026-10-25  

---

## 1. Executive Summary

This report documents the execution and verification of Workstream E6, establishing an auditable, self-contained research release candidate package for the OpenHSDBench `via-3` multi-seed classification and explainability research campaign.

The release package brings together:
1. Multi-seed classification records across two strategies (**Baseline** and **Balanced Sampling**) and three training seeds (**`42`**, **`100`**, and **`2024`**), plus single-seed class-weighted control;
2. Master explainability metrics matrices across six attribution methods (Grad-CAM, Grad-CAM++, LayerCAM, Saliency, Occlusion, and Random baseline) on the frozen 50-layout diagnostic cohort;
3. Reconciled, publication-ready reporting drafts (`draft_via3_experimental_report.md` and `draft_via3_table_summary.tex`);
4. The historical audit trail spanning Workstreams E1 through E5;
5. Comprehensive candidate documentation, machine-readable inventory (`ARTIFACT_INVENTORY.csv`), and cryptographic checksum verification (`SHA256SUMS`).

### Key Governance Declarations:
- **Zero Experiment Reruns:** Workstream E6 was conducted in pure non-destructive packaging mode. Zero neural network retraining, GPU execution, or new inference were performed.
- **Underlying Evidence Preserved:** The empirical data, checkpoints, and historical reports remain 100% unaltered.
- **Preservation of E5 Release Gate Qualifications:** The E5 release gate verdict of **`PASS WITH QUALIFICATIONS`** is strictly maintained. The package is usable, structurally verified, and ready for human release review, but carries five documented scientific and software qualifications.

---

## 2. Release Candidate Directory and Structure

A new directory was created at:

`/home/peskybird/Projects/Ai_Ml_DA/release_candidates/via3_multiseed_rc1/`

### 2.1 Candidate Manifest Structure
```
release_candidates/via3_multiseed_rc1/
|-- README.md                                  # Entry guide, scope, and navigation
|-- RELEASE_NOTES.md                           # Scientific findings, scope, and disclosures
|-- ARTIFACT_INVENTORY.csv                     # 91-item machine-readable catalog
|-- SHA256SUMS                                 # Cryptographic checksums of candidate files
|-- REPRODUCIBILITY.md                         # Taxonomy, inspection, and rerun guidance
|-- KNOWN_LIMITATIONS.md                       # Comprehensive 8-point limitation disclosures
|-- RELEASE_CHECKLIST.md                       # Sign-off matrix across governance & integrity
`-- reports/                                   # Byte-identical copies of canonical drafts & audits
    |-- draft_via3_experimental_report.md      # Authoritative Markdown report
    |-- draft_via3_table_summary.tex           # Authoritative LaTeX tables
    |-- WORKSTREAM_E1_PREFLIGHT.md             # E1 Preflight audit
    |-- WORKSTREAM_E2_EXECUTION_REPORT.md      # E2 Multi-seed training execution report
    |-- WORKSTREAM_E3_STATISTICAL_AUDIT.md     # E3 Statistical inference audit
    |-- WORKSTREAM_E4_1_XAI_PREFLIGHT.md       # E4.1 XAI preflight audit
    |-- WORKSTREAM_E4_2_EXECUTION_REPORT.md    # E4.2 XAI execution report
    |-- WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md # E4.2 XAI robustness audit
    |-- WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md # E4.3-A Provenance audit
    |-- WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md # E4.3-B Reporting remediation
    `-- WORKSTREAM_E5_FINAL_RELEASE_GATE.md    # E5 Final release gate audit
```

---

## 3. Exact Files Created and Cryptographic Checksums

All files physically included in the release candidate directory were hashed directly from disk:

| Candidate Relative Path | File Size | SHA-256 Digest | Status |
| :--- | :---: | :--- | :---: |
| `README.md` | 7,617 B | `68e9562d2fd72a6f4a89351f4c18f2cf1653b1a5f63cb265c59f8a5eeacb0e35` | Verified |
| `RELEASE_NOTES.md` | 8,432 B | `1f2dc241367c71e9ec66eb6a66878320439d25b0ed8d0f1be91264a86a9e4354` | Verified |
| `ARTIFACT_INVENTORY.csv` | 13,873 B | `18cb765a4253662e07a51438bd097526f10cbf9c8ca4d2d000400c97c0f761f0` | Verified |
| `SHA256SUMS` | 1,407 B | `21f649495b5fe1eb6aa52f0858e6e5898858e72ddba8a3a0eeb9bb42e39958fa` | Excluded from self |
| `REPRODUCIBILITY.md` | 8,369 B | `686d8da82dd275acef63d4d0e9550d32c96f5711636ab813a6085c7bef5854dd` | Verified |
| `KNOWN_LIMITATIONS.md` | 7,624 B | `e931d1b87e896132b73d8d3e1e63d0dd7971ac8567a8fecf4c62b982014ad23a` | Verified |
| `RELEASE_CHECKLIST.md` | 6,373 B | `66bec0c1972e3f55d8d67eaa7befc17ced0511896da112f8e5ddc7148edef753` | Verified |
| `reports/draft_via3_experimental_report.md` | 24,308 B | `ea73b088c9a8672172f7f8fead7e6667187b6a15e14b6b374d9b6c802606cb74` | Byte-identical |
| `reports/draft_via3_table_summary.tex` | 13,114 B | `dd221c7f327c185cde41cca2da45990f83d8aec080580a74840ac7f089c680b7` | Byte-identical |
| `reports/WORKSTREAM_E1_PREFLIGHT.md` | 20,019 B | `b457e88efa3110cc269bcd28782710d77cd143a2bc88a5ed958a23e55ce2528b` | Byte-identical |
| `reports/WORKSTREAM_E2_EXECUTION_REPORT.md` | 15,002 B | `e15c59a870469f2cb6ebb9ce5e4d42edd1854984972d260f40c9bb6fe1bb1b31` | Byte-identical |
| `reports/WORKSTREAM_E3_STATISTICAL_AUDIT.md` | 25,737 B | `3447144714cd1e818a3d0094fe9b58202551128a10fcb598934e1a9e6703de69` | Byte-identical |
| `reports/WORKSTREAM_E4_1_XAI_PREFLIGHT.md` | 20,805 B | `fed2182785a70d698eefeec2d8f7d25ffeef4a57ea473a91aa41f26554053360` | Byte-identical |
| `reports/WORKSTREAM_E4_2_EXECUTION_REPORT.md` | 9,768 B | `2c620c22289a6d49da302ebd5d6775ce1ff9d81b865a4d447a9cd068315d3bf8` | Byte-identical |
| `reports/WORKSTREAM_E4_2_XAI_ROBUSTNESS_REPORT.md` | 16,143 B | `23ed2e8fde6bfd1b18d3673bd375bf05022350d6af6865f1f20a67255daae14b` | Byte-identical |
| `reports/WORKSTREAM_E4_3_A_ARTIFACT_PROVENANCE_AUDIT.md` | 22,919 B | `771079999812f700f420f9f0dbf8728186d05848c72d1a7dc791d93322be37f4` | Byte-identical |
| `reports/WORKSTREAM_E4_3_B_PAPER_REPORTING_REMEDIATION.md` | 36,651 B | `d0f3d06b93088cd01b7a6b91000e90ccba0c7b8e89ddea305ac23cac3940b5a6` | Byte-identical |
| `reports/WORKSTREAM_E5_FINAL_RELEASE_GATE.md` | 25,895 B | `a66e40af33ed4eadeefafc2c3bb57df7b7d227c6a976f89c277d820961d334ce` | Byte-identical |

---

## 4. Artifact Inventory Summary

The machine-readable catalog `ARTIFACT_INVENTORY.csv` contains **91 entries** spanning 17 functional categories.

### 4.1 Breakdown by Availability Status
- **`referenced_in_repository`:** 73 artifacts (checkpoints, prediction CSVs, master XAI matrices, runner scripts, test modules, and authoritative drafts).
- **`included_in_candidate`:** 17 artifacts (candidate guide documents, inventory, and byte-identical report copies).
- **`missing`:** 1 artifact (`draft_reports/via3/draft_via3_table_summary.pdf`, documented as uncompiled due to local absence of `pdflatex`).
- **`not_checked`:** 0 artifacts.

### 4.2 Breakdown by Category
| Functional Category | Count | Primary Storage Location | Availability |
| :--- | :---: | :--- | :--- |
| `classification_prediction` | 18 | `results/via3_experiments/multiseed/` | `referenced_in_repository` |
| `candidate_report_copy` | 11 | `release_candidates/via3_multiseed_rc1/reports/` | `included_in_candidate` |
| `audit_report` | 11 | `results/via3_experiments/` | `referenced_in_repository` |
| `test_suite` | 11 | `tests/`, `pytest.ini` | `referenced_in_repository` |
| `runner_script` | 7 | `scripts/` | `referenced_in_repository` |
| `checkpoint` | 6 | `models/via3_multiseed/` | `referenced_in_repository` |
| `classification_result` | 6 | `results/via3_experiments/multiseed/` | `referenced_in_repository` |
| `candidate_doc` | 5 | `release_candidates/via3_multiseed_rc1/` | `included_in_candidate` |
| `manifest` | 3 | `results/via3_experiments/` | `referenced_in_repository` |
| `xai_master_record` | 3 | `results/via3_experiments/xai_multiseed/` | `referenced_in_repository` |
| `reconciled_draft` | 2 | `draft_reports/via3/` | `referenced_in_repository` |
| `draft_backup` | 2 | `draft_reports/via3/` | `referenced_in_repository` |
| `classification_summary` | 2 | `results/via3_experiments/multiseed/` | `referenced_in_repository` |
| `candidate_catalog` | 1 | `release_candidates/via3_multiseed_rc1/` | `included_in_candidate` |
| `xai_summary` | 1 | `results/via3_experiments/xai_multiseed/` | `referenced_in_repository` |
| `coursework_protected` | 1 | `docs/` | `referenced_in_repository` |
| `compiled_deliverable` | 1 | `draft_reports/via3/` | `missing` |
| **Total** | **91** | | |

---

## 5. Checksum and Integrity Verification

1. **`SHA256SUMS` Verification:** Re-executing `sha256sum -c SHA256SUMS` inside `release_candidates/via3_multiseed_rc1/` succeeded with **17 of 17 entries reporting `OK` (100% verification rate)**.
2. **Report Byte-for-Byte Verification:** Direct byte-level comparison (`cmp`) between the 11 files in `./reports/` and their respective repository sources confirmed identical content (0 byte discrepancies).
3. **Automated Test Suite Execution:** Re-executing `.venv/bin/pytest tests/` passed **73 of 73 tests (100% pass rate in 16.75 seconds)**.

---

## 6. Protected-File and Working-Tree Preservation

Pre-audit and post-audit cryptographic verification confirmed that **zero protected files were modified**:

- `docs/DA2_Group18_Report.docx`: SHA-256 `0139515bba3ed113c193ea9abf987057d79c50116064d87757ef168abeefe076` (**Immutable**)
- `latex_report/`: All 23 files unchanged (including `document.tex`, `document.pdf`, and figures).
- `draft_reports/via3/`: Canonical drafts and backups unchanged:
  - `draft_via3_experimental_report.md`: `ea73b088c9a8672172f7f8fead7e6667187b6a15e14b6b374d9b6c802606cb74` (**Immutable**)
  - `draft_via3_table_summary.tex`: `dd221c7f327c185cde41cca2da45990f83d8aec080580a74840ac7f089c680b7` (**Immutable**)
- Checkpoints in `models/via3_multiseed/`: All 6 checkpoints unchanged.
- Historical audit reports E1–E5: All unchanged.
- Working tree: Git status confirmed branch `research/iccad5-collapse-xai` with pre-existing working tree modifications strictly preserved; only the new directory `release_candidates/` was added as untracked content.

---

## 7. Discrepancies and Missing Artifacts

One non-blocking missing artifact was recorded:
- **`draft_reports/via3/draft_via3_table_summary.pdf`:** Documented as `missing` in `ARTIFACT_INVENTORY.csv` (empty SHA-256) because binary `pdflatex` is not installed on the system PATH. The underlying LaTeX table source (`draft_via3_table_summary.tex`) is structurally verified via Python stack parsing.

---

## 8. Summary of Release Qualifications

This release candidate is governed by five explicit qualifications:
1. **Runner Resume Row-Count Fragility:** The execution runner checks row count rather than composite primary keys and success status flags.
2. **Timestamp Mutation on Resume:** Rebuilding master summaries updates UTC timestamps, changing JSON digests.
3. **Adebayo Randomization Check (5/6 Passed):** Baseline Seed 100 recorded $\bar{\rho} = -0.3321$, exceeding the prespecified $|\bar{\rho}| < 0.20$ threshold (`passed_sanity_check: false`), though demonstrating substantive weight sensitivity.
4. **Occlusion Protocol Alignment:** Occlusion Sensitivity's high Insertion AUC ($0.8422$) is mechanically protocol-favored by square zero-masking and is not universally superior.
5. **LaTeX PDF Compilation:** Local compilation requires a TeX Live build host.

---

## 9. Readiness for Human Release Approval

- **Candidate Readiness:** The package `via3_multiseed_rc1` is complete, internally consistent, auditable, and self-contained for review.
- **Approval Gate:** In accordance with release-engineering standards, this release candidate packaging does not itself publish or permanently freeze the release. **Formal human release approval and sign-off remain required** prior to manuscript submission or external archiving.

---

## 10. Final Workstream E6 Verdict

### **PASS WITH QUALIFICATIONS**

The research release candidate package `release_candidates/via3_multiseed_rc1/` successfully fulfills all requirements of Workstream E6. Checksums verify, all 91 inventory entries are cataloged, all 73 tests pass, protected coursework files remain untouched, and all scientific and tooling qualifications are fully disclosed.
