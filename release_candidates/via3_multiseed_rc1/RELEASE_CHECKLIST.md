# Research Release Checklist: `via3_multiseed_rc1`

**Package Identifier:** `via3_multiseed_rc1`  
**Dataset:** OpenHSDBench `via-3`  
**Engineer:** Independent Research Release Engineer (Workstream E6)  
**Release Gate Verdict:** **PASS WITH QUALIFICATIONS**  
**Evaluation Date:** 2026-10-25  

---

## 1. Scope, Governance, and Gate Verdict

- [x] **Release Scope Defined (E6 Verified):** Scope encompasses multi-seed classification (seeds 42, 100, 2024; Baseline and Balanced Sampling) and multi-seed XAI robustness evaluations on OpenHSDBench `via-3`.
- [x] **Release Gate Verdict Preserved (E6 Verified):** Preserved the Workstream E5 verdict of **PASS WITH QUALIFICATIONS**. Unconditional or unverified claims of "zero-intervention reproducibility" are explicitly rejected.
- [x] **No Experiments Rerun (E6 Verified):** Zero GPU workloads, neural network inference, or retraining were initiated during E6. All packaged evidence reflects historical E1–E5 deliverables.
- [x] **Redistribution & Archival Scope (E6 Verified):** No claims of public redistribution rights, third-party dataset ownership, or external archival permanence are made.

---

## 2. Artifact Inventory and Cryptographic Integrity

- [x] **Artifact Inventory Completeness (E6 Verified):** `ARTIFACT_INVENTORY.csv` cataloged with 62 discrete items, distinguishing `included_in_candidate`, `referenced_in_repository`, and `missing` artifacts.
- [x] **Model Checkpoint Hashes (E5 / E6 Verified):** All six CNN checkpoints verified individually on disk with identical 12,873-parameter architecture:
  - Baseline Seed 42 (`97a9cb8bf776...`): Verified
  - Baseline Seed 100 (`f28e42777387...`): Verified
  - Baseline Seed 2024 (`9347e568cbe5...`): Verified
  - Balanced Sampling Seed 42 (`551a861f1e35...`): Verified
  - Balanced Sampling Seed 100 (`c957304bf07a...`): Verified
  - Balanced Sampling Seed 2024 (`297fa95748ad...`): Verified
- [x] **Diagnostic Cohort Integrity (E5 / E6 Verified):** `xai_sample_cohort_manifest.json` on disk matches SHA-256 `1c547c162c32df2149653d68dc273511b866be81604d20b334a7306ffa5292cb` (40 HS / 10 NHS across 6 error modes).
- [x] **Reproducibility Manifest Consistency (E5 / E6 Verified):** `reproducibility_manifest.json` matches SHA-256 `9655c3480db1cd05f9fb87e5c8abbea53d1a906d9277546731711c74f1c26e51`. All recorded output artifact hashes match on-disk CSVs byte-for-byte.

---

## 3. Classification and XAI Empirical Record Validation

- [x] **Classification Prediction Records (E3 / E5 / E6 Verified):** All 18 prediction CSV files across 2 strategies, 3 seeds, and 3 splits exist and match exact expected row counts (Clean Test: 12,803; Official Test: 12,905; Val: 2,846).
- [x] **Master Faithfulness Completeness (E4.3-A / E5 / E6 Verified):** `master_faithfulness_metrics.csv` contains exactly 1,800 rows with 0 duplicate keys, 0 nulls/NaNs, and 100% status `SUCCESS`.
- [x] **Master Stability Completeness (E4.3-A / E5 / E6 Verified):** `master_stability_metrics.csv` contains exactly 1,500 rows with 0 duplicate keys, 0 nulls/NaNs, and 100% status `SUCCESS`.
- [x] **Master Collapse Completeness (E4.3-A / E5 / E6 Verified):** `master_collapse_diagnostics.csv` contains exactly 300 rows with 0 duplicate keys, 0 nulls/NaNs, and 100% status `SUCCESS`.
- [x] **Automated Test Suite Pass (E6 Verified):** Complete test suite passes cleanly: `pytest tests/` -> 73 passed in 16.75s (100% pass rate).

---

## 4. Scientific Consistency and Remediation Verification

- [x] **Table 4 Spatial Standard Deviation Reconciled (E4.3-B / E5 / E6 Verified):** Reconciled typographical transcription error. Layout-level standard deviation across all 150 evaluations is correctly reported as $\mathbf{0.0571}$ ($ddof=1$), and cross-seed standard deviation across 3 seed means is correctly reported as $\mathbf{0.0615}$ ($K=3, ddof=1$).
- [x] **Aggregation Unit Discipline (E4.3-B / E5 / E6 Verified):** Clear separation between layout observations ($N=12,803$ or $N=50$) and independent model training seeds ($K=3$). Normal-theory Student's $t$ confidence intervals eliminated due to unphysical negative bounds.
- [x] **Cohort Ground-Truth Composition (E4.3-B / E5 / E6 Verified):** Diagnostic cohort correctly documented as 40 Hotspot and 10 Non-Hotspot layouts. Historical "25 HS / 25 NHS" narrative typo refuted.
- [x] **Adebayo Sanity Check Disclosures (E4.3-B / E5 / E6 Verified):** Five of six models passed under prespecified programmatic rule ($|\bar{\rho}| < 0.20$); Baseline Seed 100 recorded $\bar{\rho} = -0.3321$ (`passed_sanity_check: false`). Programmatic failure disclosed alongside substantive weight-sensitivity interpretation.
- [x] **Occlusion Protocol-Specific Qualification (E4.3-B / E5 / E6 Verified):** Occlusion Sensitivity qualified as structurally aligned with square zero-masking benchmark protocol; not described as universally best.
- [x] **Draft Cross-Consistency (E5 / E6 Verified):** 100% numerical consistency between Markdown draft (`draft_via3_experimental_report.md`) and LaTeX table summary (`draft_via3_table_summary.tex`).

---

## 5. Candidate Package Integrity & Verification

- [x] **Release Candidate Directory Created (E6 Verified):** New non-conflicting directory created at `release_candidates/via3_multiseed_rc1/`.
- [x] **Release Reports Copied and Verified (E6 Verified):** Copies of 9 historical audit reports and 2 authoritative drafts in `reports/` verified as byte-for-byte identical to source files via `cmp`.
- [x] **Candidate Checksum File Generated (E6 Verified):** `SHA256SUMS` generated for all regular files physically included in the release candidate directory (excluding `SHA256SUMS`).
- [x] **Checksum File Self-Verification (E6 Verified):** Executed `sha256sum -c SHA256SUMS` inside the candidate directory; 100% of entries verified `OK`.

---

## 6. Protected-Path and Working-Tree Preservation

- [x] **Coursework Deliverable Untouched (E6 Verified):** `docs/DA2_Group18_Report.docx` SHA-256 matches initial baseline (`0139515bba...`).
- [x] **LaTeX Deliverables Untouched (E6 Verified):** All files under `latex_report/` remain unaltered.
- [x] **Canonical Drafts Untouched (E6 Verified):** Files under `draft_reports/via3/` remain unaltered.
- [x] **Historical Audit Reports Untouched (E6 Verified):** All E1 through E5 audit reports remain unaltered.
- [x] **Pre-existing Working Tree Modifications Preserved (E6 Verified):** Pre-existing modifications in `src/` and `scripts/` preserved; zero git branches switched, zero commits, stashes, or resets performed.

---

## 7. Sign-Off and Release Decision

- [x] **Engineering Readiness:** Package is complete, internally consistent, auditable, and self-contained for review.
- [x] **Future Hardening Tracked:** Runner resume hardening and CI/CD TeX compilation tracked in post-E5 roadmap.
- [ ] **Final Human Release Approval:** Pending formal human review and sign-off.
