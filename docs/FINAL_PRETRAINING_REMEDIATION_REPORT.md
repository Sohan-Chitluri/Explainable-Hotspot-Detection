# Final Pre-Training Remediation Report — AUD-01, End-to-End Handoff, and Gate Closure

**Report Date:** 2026-10-21T15:15:00Z (20:45:00 IST)  
**Workspace:** `/home/peskybird/Projects/Ai_Ml_DA`  
**Git Branch:** `research/iccad5-collapse-xai`  
**Git Commit HEAD:** `41d2c854604a576f01b3fddb72c2c09d43fbcb8e`  
**Execution Environment:** NixOS Linux x86_64, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  
**Execution Wrapper:** `./scripts/tf_gpu.sh` (Python 3.11.16, TensorFlow 2.15.1, CUDA 12.2 / cuDNN 8.9.4)  

---

## 1. Executive Summary & Gate Status

This remediation sprint successfully resolved issue **AUD-01** (prediction CSV export), validated the end-to-end handoff from baseline training to threshold analysis using dedicated smoke-test cohorts, reconciled the test exclusion accounting with verifiable arithmetic, and verified that all repository safeguards and background process gates remain strictly closed.

### Gate Evaluation Matrix

| Gate / Criterion | Status | Summary / Primary Evidence |
| :--- | :---: | :--- |
| **1. AUD-01 Resolution** | **PASS** | `save_predictions_csv` implemented in contract and integrated into baseline runner; exports `val_predictions.csv`, `clean_test_predictions.csv`, and `official_test_predictions.csv`. |
| **2. Prediction CSV Contract & Schema** | **PASS** | Exact column match (`sample_id`, `filename`, `ground_truth`, `hs_probability`, `predicted_label`); unique IDs; valid binary labels; continuous finite probabilities in $[0, 1]$. |
| **3. End-to-End Smoke Handoff** | **PASS** | GPU smoke run completed with exit code `0`; threshold analysis consumed validation predictions directly and evaluated clean test at frozen $\tau^* = 0.0100$. |
| **4. Threshold Validation Isolation** | **PASS** | `select_threshold_from_val` accepts strictly validation predictions; clean test evaluated only post-calibration; zero test leakage. |
| **5. Exclusion Accounting Reconciliation** | **PASS** | Arithmetic proved: $98 \text{ duplicates} + 8 \text{ contradictory} - 4 \text{ intersection} = 102 \text{ unique exclusions}$. |
| **6. Background Process Guard** | **PASS** | PID `537431` (`agy` on `pts/2`) confirmed active (accumulated 02:55 CPU time); training gate remains **CLOSED**. |
| **7. Regression & Forensic Tests** | **PASS** | **69/69 executed tests passed** (64 unit tests in `tests/`, 5 forensics tests in `degeneracy_forensics`). |
| **8. Protected Artifact Safeguards** | **PASS** | 4/4 manifest SHA-256 hashes intact; `docs/DA2_Group18_Report.docx` and `latex_report/` 100% unmodified. |
| **9. Full-Cohort Training Gate** | **PASS** | Full-cohort training was **NOT started**. |
| **10. Approval Token State** | **PASS** | Token `.pretraining_verification_passed` has **NOT been created or issued**. |

---

## 2. AUD-01 Implementation Details and Changed Files

### Root Cause & Scope
`scripts/run_via3_baseline.py` previously persisted summary JSON manifests and Keras model checkpoints, but did not export the per-sample CSV predictions required by downstream consumers, specifically `scripts/analyze_via3_thresholds.py`.

### Minimal Compatible Remediation
Two files were modified to resolve AUD-01 without changing model architecture, data splits, or metric definitions:

1. **[`src/experiment_contract.py#L302-L338`](file:///home/peskybird/Projects/Ai_Ml_DA/src/experiment_contract.py#L302-L338):**
   Added the canonical `save_predictions_csv` helper:
   ```python
   def save_predictions_csv(
       paths: list[str],
       y_true: list[int] | np.ndarray,
       probs: np.ndarray | list[float],
       out_path: Path | str,
   ) -> Path:
       """Save raw per-sample prediction probabilities and decisions."""
       out_path = Path(out_path)
       out_path.parent.mkdir(parents=True, exist_ok=True)
       y_true_arr = np.asarray(y_true).astype(int).ravel()
       probs_arr = np.asarray(probs).astype(float).ravel()
       assert len(paths) == len(y_true_arr) == len(probs_arr)

       with open(out_path, "w", newline="", encoding="utf-8") as f:
           writer = csv.writer(f)
           writer.writerow(["sample_id", "filename", "ground_truth", "hs_probability", "predicted_label"])
           for p, true_lbl, prob in zip(paths, y_true_arr, probs_arr):
               fn = Path(p).name
               sample_id = fn.split("_")[0]
               prob_val = float(prob)
               pred_lbl = 1 if prob_val >= 0.5 else 0
               writer.writerow([sample_id, fn, int(true_lbl), round(prob_val, 6), pred_lbl])
       return out_path
   ```

2. **[`scripts/run_via3_baseline.py#L243-L256`](file:///home/peskybird/Projects/Ai_Ml_DA/scripts/run_via3_baseline.py#L243-L256):**
   Imported and called `save_predictions_csv` immediately after model prediction:
   ```python
   val_pred_csv = save_predictions_csv(data.val_paths, data.val_labels, val_probs, output_dir / "val_predictions.csv")
   clean_pred_csv = save_predictions_csv(data.clean_test_paths, data.clean_test_labels, clean_test_probs, output_dir / "clean_test_predictions.csv")
   official_pred_csv = save_predictions_csv(data.official_test_paths, data.official_test_labels, official_test_probs, output_dir / "official_test_predictions.csv")
   ```
   Prediction file paths were recorded inside `manifest_output["prediction_files"]`.

3. **[`tests/test_experiment_contract.py#L96-L137`](file:///home/peskybird/Projects/Ai_Ml_DA/tests/test_experiment_contract.py#L96-L137):**
   Added unit test `test_save_predictions_csv_schema_and_counts` covering column order, value ranges, ID uniqueness, and binary classification decisions.

---

## 3. CSV Schema and Row-Count Verification

During the smoke-test handoff in dedicated directory `results/via3_smoke_handoff_remediation/`, the exported CSVs were audited with fresh automated assertions:

| File Name | Row Count | Expected Count | Column Schema | Sample IDs Unique? | Label Values | Probability Range |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `val_predictions.csv` | **128** | 128 (16 HS, 112 NHS) | `sample_id, filename, ground_truth, hs_probability, predicted_label` | **Yes (128/128)** | `{0, 1}` | $[0.000329, 0.443194]$ |
| `clean_test_predictions.csv` | **128** | 128 (16 HS, 112 NHS) | `sample_id, filename, ground_truth, hs_probability, predicted_label` | **Yes (128/128)** | `{0, 1}` | $[0.000301, 0.457812]$ |
| `official_test_predictions.csv` | **128** | 128 (16 HS, 112 NHS) | `sample_id, filename, ground_truth, hs_probability, predicted_label` | **Yes (128/128)** | `{0, 1}` | $[0.000301, 0.457812]$ |

- All probabilities are continuous, finite IEEE float values bounded strictly in $[0.0, 1.0]$.
- Every sample ID matches the exact prefix (e.g., `Hvia107504`, `Nvia118233`).
- In full-cohort production runs, row counts will expand to exactly 2,846 (`val`), 12,803 (`clean_test`), and 12,905 (`official_test`).

---

## 4. End-to-End Smoke Test & Threshold Handoff Results

The complete baseline $\rightarrow$ threshold analysis handoff was executed end-to-end:

### Step A: Baseline Smoke Execution
```bash
./scripts/tf_gpu.sh scripts/run_via3_baseline.py \
  --smoke \
  --override-gate \
  --output-dir results/via3_smoke_handoff_remediation \
  --models-dir models/via3_smoke_handoff_remediation
```
- **Exit Code:** `0`
- **Device Utilized:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (Memory allocated: 4,168 MB)
- **Duration:** 15.2 seconds
- **Checkpoint Generated:** `models/via3_smoke_handoff_remediation/via3_baseline_seed42_smoke.keras` (230,689 bytes)
- **Weight Restoration:** Best weights restored from Epoch 1 (val PR-AUC = 0.1163) before cohort evaluation.

### Step B: Threshold Analysis Execution
```bash
./.venv/bin/python scripts/analyze_via3_thresholds.py \
  --val-predictions results/via3_smoke_handoff_remediation/val_predictions.csv \
  --test-predictions results/via3_smoke_handoff_remediation/clean_test_predictions.csv \
  --output-dir results/via3_smoke_handoff_remediation/threshold_analysis \
  --objective max_f1
```
- **Exit Code:** `0`
- **Threshold Swept:** 99 intervals in $[0.01, 0.99]$ on validation probabilities.
- **Calibrated Operational Threshold:** $\tau^* = 0.0100$ (maximizing validation F1).
- **Generated Visualizations & Data Tables:**
  - `via3_baseline_threshold_tradeoff.png` (70,715 bytes)
  - `via3_baseline_val_pr_curve.png` (46,963 bytes)
  - `via3_baseline_val_cm_calibrated.png` (28,222 bytes)
  - `via3_baseline_clean_test_cm_calibrated.png` (28,927 bytes)
  - `via3_baseline_threshold_sweep.csv` (4,390 bytes)
  - `via3_baseline_threshold_analysis.json` (2,574 bytes)

---

## 5. Explicit Evidence of Threshold Validation Isolation

The control flow of threshold calibration was audited to prove that clean-test data cannot influence the chosen threshold $\tau^*$:

1. **Input Signature:** In [`src/experiment_contract.py#L142-L149`](file:///home/peskybird/Projects/Ai_Ml_DA/src/experiment_contract.py#L142-L149), `select_threshold_from_val` accepts only:
   ```python
   def select_threshold_from_val(
       val_labels: np.ndarray | list[int],
       val_probs: np.ndarray | list[float],
       threshold_grid: Optional[list[float]] = None,
       objective: str = "max_f1",
       min_precision: float = 0.0,
       target_recall: float = 0.70,
   ) -> Dict[str, Any]:
   ```
   Test labels and test probabilities are neither accepted nor accessible in this scope.

2. **Sequential Execution Guard:** In [`scripts/analyze_via3_thresholds.py#L127-L175`](file:///home/peskybird/Projects/Ai_Ml_DA/scripts/analyze_via3_thresholds.py#L127-L175):
   - `calibration = select_threshold_from_val(val_labels, val_probs, ...)` is evaluated at line 127.
   - `tau_star = calibration["selected_threshold"]` is extracted and frozen at line 134.
   - Validation plots, sweep tables, and validation confusion matrices are serialized at lines 143–166.
   - Test predictions are inspected only at line 170 via `compute_contract_metrics(test_labels, test_probs, threshold=tau_star)`.
   - If `--test-predictions` is omitted, the analyzer executes and completes normally without test data.

---

## 6. Exclusion Accounting Reconciliation

Auditing [`results/dataset_audit/via3_exclusion_manifest.json`](file:///home/peskybird/Projects/Ai_Ml_DA/results/dataset_audit/via3_exclusion_manifest.json) resolved the relationship between the 98 duplicate samples and 8 contradiction records:

### Category Analysis
1. **Exact Train Duplicates:** **98 samples**
   - 89 NHS samples that duplicate training NHS images.
   - 5 HS samples that duplicate training HS images (`Hvia120726`, `Hvia5453`, `Hvia13962`, `Hvia11539`, `Hvia2571`).
   - 4 HS samples that duplicate training NHS images (`Hvia87136`, `Hvia106850`, `Hvia103556`, `Hvia109958`).

2. **Contradictory Label Instances:** **8 samples**
   - 4 cross-split contradictions (the 4 HS samples matching training NHS images).
   - 4 test-internal contradictions (2 pairs of identical images carrying opposing labels):
     - Pair 1: `Nvia118233` (0) and `Hvia30568` (1) [Canonical hash: `52965724136e...`]
     - Pair 2: `Nvia56056` (0) and `Hvia112551` (1) [Canonical hash: `0fa41217ff96...`]

3. **Intersection ($\text{Duplicates} \cap \text{Contradictory}$):** **4 samples**
   - Exactly the 4 cross-split samples: `Hvia87136`, `Hvia106850`, `Hvia103556`, `Hvia109958`.
   - These 4 samples satisfy both definitions: they are exact pixel duplicates of training images, and their labels contradict the training ground truth.

### Arithmetic Reconciliation
$$\begin{aligned}
|\text{Total Unique Excluded Samples}| &= |\text{Exact Train Duplicates}| + |\text{Contradictory Labels}| - |\text{Intersection}| \\
&= 98 + 8 - 4 \\
&= \mathbf{102}
\end{aligned}$$

- **Class breakdown of 102 exclusions:** 11 Hotspots ($4 \text{ cross-split} + 2 \text{ test-internal} + 5 \text{ train HS dups}$) and 91 Non-Hotspots ($89 \text{ train NHS dups} + 2 \text{ test-internal}$).
- **Clean Test Cohort:** $12,905 - 102 = \mathbf{12,803}$ samples (411 HS, 12,392 NHS). Zero ambiguity remains.

---

## 7. Background Process Status (PID 537431)

- **Process Query:** `ps -fp 537431`
- **Observed Command:** `agy` on terminal `pts/2` (parent PID: `537419`)
- **Execution Time:** Accumulated 02:55 CPU time since 19:47:00 IST.
- **Process State:** Actively running (`Sl+`).
- **Action Taken:** The process was left completely untouched. No signals were delivered.
- **Gate Implication:** Because the pre-verification process has not concluded and has not written the approval token, the baseline training gate remains **CLOSED**.

---

## 8. Regression and Protected-Artifact Verification

### 8.1 Automated Test Execution
- **Command:** `./.venv/bin/pytest tests/ --durations=10`
  - Result: **64 passed, 0 failed, 0 skipped** (14.14s)
- **Command:** `./.venv/bin/pytest results/xai/final_iccad5/degeneracy_forensics/test_degeneracy_forensics.py`
  - Result: **5 passed, 0 failed, 0 skipped** (0.16s)
- **Combined Test Count:** **69 passed, 0 failed**.

### 8.2 Audited Manifest Hashes (SHA-256)
- `via3_split_manifest.json`: `17bc670452a083c882d18c0de7f100b88327571f6e3795f0fc0fce512b8e4d53` (**VERIFIED**)
- `via3_exclusion_manifest.json`: `f4d269cace5b94d1629aeefdd9f0b55d9fb210444529aa31a0af93326518755a` (**VERIFIED**)
- `via3_cohort_clean_test.json`: `28d45645bb2b0dcb2f9365d02ce39a6083eb2a506926417c986e817b5441dd8f` (**VERIFIED**)
- `via3_cohort_official_test.json`: `149627ea68634a4b835fa81ef0e4785d11a5ae9b2fec9237577d45837622adbe` (**VERIFIED**)

### 8.3 Protected Artifact Integrity
- `git diff --stat docs/DA2_Group18_Report.docx latex_report/`: **0 files changed, 0 insertions, 0 deletions**.

---

## 9. Remaining Issues and Pre-Launch Production Sequence

### Remaining Issues
There are **zero remaining blocking issues**. The pipeline is fully remediated and verified.

### Proposed Production Command Sequence
Once the user confirms PID `537431` has concluded and issues the approval token (`touch .pretraining_verification_passed`), the production runs should be launched sequentially on the RTX 3050 GPU:

```bash
# 1. Authorize Gate (Human Lead Action)
touch .pretraining_verification_passed

# 2. Baseline Training (Workstream A)
./scripts/tf_gpu.sh scripts/run_via3_baseline.py \
  --manifest-dir results/dataset_audit \
  --output-dir results/via3_experiments/baseline \
  --models-dir models/via3 \
  --epochs 15 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42

# 3. Decision Threshold Analysis (Workstream B1)
./.venv/bin/python scripts/analyze_via3_thresholds.py \
  --val-predictions results/via3_experiments/baseline/val_predictions.csv \
  --test-predictions results/via3_experiments/baseline/clean_test_predictions.csv \
  --output-dir results/via3_experiments/threshold_analysis \
  --objective max_f1

# 4. Imbalance Intervention Experiments (Workstream B2)
./scripts/tf_gpu.sh scripts/run_via3_imbalance_experiments.py \
  --strategies baseline class_weighted balanced_sampling \
  --output-dir results/via3_experiments/imbalance \
  --models-dir models/via3_imbalance \
  --epochs 15 \
  --seed 42

# 5. Multi-Seed Variance Orchestration (Workstream B3)
./scripts/tf_gpu.sh scripts/run_via3_multiseed.py \
  --seeds 42 100 2024 \
  --strategy baseline \
  --output-dir results/via3_experiments/multiseed \
  --models-dir models/via3_multiseed \
  --epochs 15

# 6. XAI Faithfulness and Stability Evaluation (Workstream C)
./scripts/tf_gpu.sh scripts/evaluate_via3_xai.py \
  --checkpoint models/via3/via3_baseline_seed42.keras \
  --output-dir results/via3_experiments/xai_faithfulness \
  --steps 10

# 7. Paper Reports and Statistical Output Synthesis (Workstream D)
./.venv/bin/python scripts/generate_paper_reports.py \
  --input-dir results/via3_experiments \
  --output-dir draft_reports/via3
```

---

## 10. Final Compliance Statement

- Full-cohort training was **NOT started**.
- Approval token `.pretraining_verification_passed` was **NOT issued**.
- All preparation criteria are **PASS**. The pipeline is in a verified standby state awaiting human confirmation of background PID 537431.
