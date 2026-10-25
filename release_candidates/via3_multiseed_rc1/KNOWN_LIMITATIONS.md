# Known Limitations and Methodological Qualifications: `via3_multiseed_rc1`

**Release Candidate:** `via3_multiseed_rc1`  
**Dataset:** OpenHSDBench `via-3`  
**Date:** 2026-10-25  
**Audit Status:** **PASS WITH QUALIFICATIONS**  

---

## 1. Overview and Engineering Distinctions

This document provides transparent disclosure of all known scientific, statistical, and software engineering limitations associated with the OpenHSDBench `via-3` multi-seed classification and explainability release candidate.

> [!NOTE]
> **Distinction Between Tooling Limitations and Data Corruption:**  
> Exhaustive audits in Workstreams E4.3-A and E5 confirmed that **zero data corruption, numerical truncation, or missing records exist in the empirical release deliverables**. The limitations documented below represent software engineering vulnerabilities, statistical scope boundaries, and protocol-specific constraints, none of which invalidate the underlying experimental findings.

---

## 2. Documented Limitations Matrix

### 2.1 Limitation 1: Blind Row-Count Completion Guard in XAI Runner
- **What is known:** In `scripts/run_via3_xai_multiseed.py:127-129`, the function `is_checkpoint_xai_completed` evaluates whether cached evaluations exist by checking only total row counts (`len(df) == expected_rows`).
- **What is not established:** It is not established that any corrupted or failed runs were incorrectly accepted during E4.2 execution. Direct forensic inspection verified that all 1,800 faithfulness rows, 1,500 stability rows, and 300 collapse rows have status `SUCCESS` and unique composite primary keys `(strategy, seed, filename, method)`.
- **Impact on current release:** Low. The historical dataset is valid, but future rerun attempts on incomplete or crashing runs could falsely bypass execution.
- **Recommended future action:** Refactor the completion guard to validate `(df["status"] == "SUCCESS").sum() == expected` and assert filename set equivalence: `set(df["filename"]) == set(expected_filenames)`.

### 2.2 Limitation 2: Unvalidated Cache Reuse on Resume
- **What is known:** When resuming a partially completed run, `scripts/run_via3_xai_multiseed.py` loads existing CSVs from disk without verifying whether the on-disk model checkpoint or cohort manifest matches expected SHA-256 digests.
- **What is not established:** It is not established that disparate checkpoints or cohorts were used; audit confirmed that all six checkpoints match their canonical SHA-256 hashes and identical 12,873-parameter architecture.
- **Impact on current release:** Low. Existing records are provenanced, but rerun tooling lacks cryptographic checkpoint anchoring.
- **Recommended future action:** Add an explicit pre-load cryptographic guard checking `hashlib.sha256(ckpt_bytes).hexdigest() == EXPECTED_CHECKPOINTS[model_key]`.

### 2.3 Limitation 3: Output Serialization Timestamp Mutation on Resume
- **What is known:** When `scripts/run_via3_xai_multiseed.py` is executed with `--resume`, re-serializing the final `multiseed_xai_comparison_summary.json` injects a dynamic UTC timestamp via `time.strftime(...)`. This mutates the file's SHA-256 digest, invalidating `reproducibility_manifest.json` even when empirical metrics are identical.
- **What is not established:** It is not established that metrics change upon resume. All numerical values remain identical.
- **Impact on current release:** Moderate. Prevents zero-intervention push-button reruns from achieving deterministic bitwise parity without manual manifest regeneration.
- **Recommended future action:** Support a `--freeze-timestamp` command-line argument or reuse the cohort manifest timestamp to ensure bit-level JSON reproducibility.

### 2.4 Limitation 4: Historical Execution Report Errata Preserved
- **What is known:** Historical report `WORKSTREAM_E4_2_EXECUTION_REPORT.md` retains two known documentation discrepancies:
  1. Table 5 lists an intermediate digest `75f28469ad...` for `reproducibility_manifest.json` captured prior to runtime metadata injection (actual hash: `9655c3480db1...`).
  2. Section 3 contains an informal phrasing describing the cohort as "25 HS / 25 NHS", reflecting outcome quadrants rather than ground-truth labels.
- **What is not established:** It is not established that the on-disk manifest or cohort was corrupted. The frozen cohort has 40 HS / 10 NHS, and all artifact hashes recorded *inside* the manifest match on-disk CSVs byte-for-byte.
- **Impact on current release:** Low. Preserved deliberately to maintain historical audit immutability; fully reconciled and corrected in the release candidate drafts.
- **Recommended future action:** Downstream citations must reference `draft_via3_experimental_report.md` and `WORKSTREAM_E5_FINAL_RELEASE_GATE.md` rather than uncorrected historical drafts.

### 2.5 Limitation 5: Model Parameter Randomization Threshold Exceeded (Adebayo Sanity Check)
- **What is known:** In `scripts/run_via3_xai_multiseed.py:436`, the Adebayo weight randomization check was prespecified as `"passed_sanity_check": bool(abs(adebayo_mean_rho) < 0.20)`. Under this rule, **5 of 6 models passed**, while Baseline Seed 100 recorded $\bar{\rho} = -0.3321$ and was flagged as `passed_sanity_check: false`.
- **What is not established:** It is not established that Baseline Seed 100 explanations act as invariant edge detectors. In explainability theory, weight invariance produces $\bar{\rho} \approx +1.0$. A negative correlation of $-0.3321$ confirms genuine sensitivity to model weights.
- **Impact on current release:** Moderate. Precludes claiming that 6 of 6 models passed under the prespecified programmatic threshold without post-hoc threshold relaxation.
- **Recommended future action:** Transparently document the 5/6 programmatic pass rate alongside the substantive interpretation, avoiding post-hoc threshold changes.

### 2.6 Limitation 6: Protocol-Specific Alignment of Occlusion Sensitivity
- **What is known:** Occlusion Sensitivity achieved the highest Insertion AUC ($0.8422 \pm 0.1034$ in Balanced Sampling) and lowest Deletion AUC ($0.1365 \pm 0.0989$). The evaluation benchmark employs progressive $32\times32$ square zero-masking, identical to Occlusion Sensitivity's sliding window perturbation primitive.
- **What is not established:** It is not established that Occlusion Sensitivity provides universally or causally superior explanations outside of square zero-fill perturbation benchmarks.
- **Impact on current release:** Moderate. Prevents claiming Occlusion Sensitivity as universally best.
- **Recommended future action:** Document Occlusion Sensitivity as faithful *under progressive substrate replacement protocols*, noting its coarse spatial resolution and ~100x latency penalty.

### 2.7 Limitation 7: LaTeX PDF Binary Compilation Unverified
- **What is known:** System PATH in the audit environment does not contain binary `pdflatex`. LaTeX tables in `draft_via3_table_summary.tex` were validated via static Python stack parsing for brace parity, environment nesting, column counts, and math delimiter matching.
- **What is not established:** It is not established that any visual rendering defects, hyphenation errors, or layout overflows exist in the compiled PDF.
- **Impact on current release:** Low for data integrity; moderate for publication readiness.
- **Recommended future action:** Compile the LaTeX table source to PDF on a CI/CD build machine equipped with TeX Live prior to final camera-ready publication.

### 2.8 Limitation 8: Provenance Manifest Scope Omissions
- **What is known:** While `reproducibility_manifest.json` captures checkpoints, cohort digest, core library versions (Python, TF, NumPy), and output CSV hashes, it omits the exact Git commit SHA, branch dirty status, attribution hyperparameters (target layer `conv2d_5`, occlusion stride $16\times16$), and secondary library versions (Pandas 2.2.3, Scipy 1.15.2).
- **What is not established:** It is not established that missing provenance fields obscure reproducibility, as hyperparameters and environments are fully specified in the E1–E5 reports.
- **Impact on current release:** Low.
- **Recommended future action:** Inject Git commit SHA, git status, attribution hyperparameters, and full virtual environment `pip freeze` into future manifest revisions.
