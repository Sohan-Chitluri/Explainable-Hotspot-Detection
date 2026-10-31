# Reproducibility Guide: OpenHSDBench `via-3` Multi-Seed Classification & XAI (RC1)

**Package:** `via3_multiseed_rc1`  
**Dataset:** OpenHSDBench `via-3` Lithography Hotspot Benchmark  
**Date:** 2026-10-25  
**Reproducibility Assessment:** **PASS WITH QUALIFICATIONS**  

---

## 1. Reproducibility Taxonomy & Release Boundaries

To ensure complete scientific transparency and avoid misleading claims, this release package distinguishes among three distinct levels of reproducibility:

```
+-------------------------------------------------------------------------------+
| Level 1: Artifact-Level Verification (FULLY VERIFIED & DETERMINISTIC)         |
| Existing checkpoints, prediction CSVs, XAI master matrices, and manifests    |
| exist on disk, match canonical SHA-256 digests, and allow independent        |
| metric recalculation directly from static records.                            |
+-------------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------------+
| Level 2: Procedure-Level Reproducibility (DOCUMENTED & AUDITED)               |
| Source code, training configurations, hyperparameter contracts, random seeds, |
| and cohort manifests are fully articulated and pass all 73 automated tests.  |
+-------------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------------+
| Level 3: Push-Button Rerun Reproducibility (QUALIFIED / NOT GUARANTEED)       |
| Complete, zero-intervention, deterministic regeneration from scratch via a    |
| single shell command without manual intervention is NOT guaranteed due to    |
| runner cache mechanics, timestamp mutation, and local build dependencies.     |
+-------------------------------------------------------------------------------+
```

> [!CAUTION]
> **Preservation of Workstream E5 Conclusion:**  
> This package is **not unconditionally push-button reproducible**. Downstream researchers must not represent this release as a turnkey, zero-intervention rerun benchmark until post-E5 runner hardening is implemented.

---

## 2. Environment and Tooling Telemetry

The empirical outputs in this release candidate were generated and audited under the following verified runtime environment:

- **Host Operating System:** Linux 6.12.35 (x86_64, NixOS)
- **Virtual Environment:** Python 3.11.16 located at `/home/peskybird/Projects/Ai_Ml_DA/.venv/bin/python`
- **Installed Core Libraries:**
  - `tensorflow`: `2.15.1`
  - `numpy`: `1.26.4`
  - `pandas`: `2.2.3`
  - `scipy`: `1.15.2`
  - `pytest`: `9.1.1`
- **Acceleration Hardware:**
  - GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (PCI `0000:01:00.0`)
  - NVIDIA Driver: `595.99.02`
  - CUDA Runtime: `13.2`

---

## 3. Codebase Architecture and Artifact Locations

### 3.1 Source Code and Test Suite
- `scripts/run_via3_multiseed.py`: Multi-seed classification training and evaluation pipeline across seeds 42, 100, and 2024.
- `scripts/run_via3_xai_multiseed.py`: Multi-seed explainability runner evaluating Grad-CAM, Grad-CAM++, LayerCAM, Saliency, Occlusion, and Random baseline.
- `scripts/analyze_via3_thresholds.py`: PR-curve sweep and F1-optimal validation threshold ($\tau^*$) calibration.
- `scripts/generate_paper_reports.py`: Remediated table and draft report synthesizer.
- `src/experiment_contract.py`: Programmatic contract defining seeds, paths, architectures, and evaluation thresholds.
- `src/openhsd_data.py`: Data ingestion, 3-way split management, and balanced batch generator.
- `src/results_schema.py`: Schema validation for prediction files and metrics matrices.
- `src/faithfulness.py`: Progressive pixel insertion/deletion AUC calculation engine.
- `src/gradcam.py`: LayerCAM, Grad-CAM, and Grad-CAM++ implementation targeting pre-sigmoid logits.
- `src/occlusion.py`: Sliding-window occlusion sensitivity implementation.
- `pytest.ini`: Root pytest configuration.
- `tests/`: 73 unit and integration tests (100% pass rate).

### 3.2 Checkpoint and Prediction Locations
- Checkpoints: `models/via3_multiseed/via3_{strategy}_seed{seed}.keras`
- Classification Predictions: `results/via3_experiments/multiseed/{strategy}/seed_{seed}/{strategy}_{split}_predictions.csv`
- Master XAI Metrics: `results/via3_experiments/xai_multiseed/master_{faithfulness,stability,collapse}_metrics.csv`
- Diagnostic Cohort: `results/via3_experiments/xai_multiseed/cohort/xai_sample_cohort_manifest.json`

---

## 4. How to Inspect and Recompute Metrics from Static Artifacts

Researchers can independently audit and recompute all empirical findings without initiating GPU execution or mutating repository state.

### 4.1 Running the Automated Test Suite
Verify that the codebase contract, split integrity, schema, and numerical consistency checks pass:
```bash
/home/peskybird/Projects/Ai_Ml_DA/.venv/bin/pytest tests/
```
*Expected Result:* `73 passed in ~13-17s`.

### 4.2 Verifying Model Checkpoints & Parameter Counts
Inspect model architectures and verify parameter counts ($12,873$ trainable parameters) using the project Python environment:
```python
import tensorflow as tf

ckpt_path = "models/via3_multiseed/via3_baseline_seed42.keras"
model = tf.keras.models.load_model(ckpt_path)
print(f"Layers: {len(model.layers)}, Trainable parameters: {model.count_params()}")
assert model.count_params() == 12873
```

### 4.3 Auditing Master XAI Records
Verify completeness, status flags, and key uniqueness across the three master XAI CSVs:
```python
import pandas as pd

# 1. Faithfulness (1,800 rows: 2 strategies x 3 seeds x 50 layouts x 6 methods)
df_faith = pd.read_csv("results/via3_experiments/xai_multiseed/master_faithfulness_metrics.csv")
assert len(df_faith) == 1800
assert (df_faith["status"] == "SUCCESS").all()
assert not df_faith.duplicated(subset=["strategy", "seed", "filename", "method"]).any()

# 2. Stability (1,500 rows: 2 strategies x 3 seeds x 50 layouts x 5 deterministic methods)
df_stab = pd.read_csv("results/via3_experiments/xai_multiseed/master_stability_metrics.csv")
assert len(df_stab) == 1500
assert (df_stab["status"] == "SUCCESS").all()
assert not df_stab.duplicated(subset=["strategy", "seed", "filename", "method"]).any()

# 3. Collapse (300 rows: 2 strategies x 3 seeds x 50 layouts)
df_coll = pd.read_csv("results/via3_experiments/xai_multiseed/master_collapse_diagnostics.csv")
assert len(df_coll) == 300
assert (df_coll["status"] == "SUCCESS").all()
assert not df_coll.duplicated(subset=["strategy", "seed", "filename"]).any()
print("All master XAI CSV records verified successfully.")
```

### 4.4 Recomputing Clean Test Classification Metrics
Recompute PR-AUC, ROC-AUC, HS Recall, HS Precision, and F1 at validation-calibrated thresholds $\tau^*$ directly from prediction CSVs:
```python
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, f1_score, recall_score, precision_score

# Example: Baseline Seed 42 at tau* = 0.08
df = pd.read_csv("results/via3_experiments/multiseed/baseline/seed_42/baseline_clean_test_predictions.csv")
y_true = df["label"].values
y_prob = df["prob"].values
y_pred = (y_prob >= 0.08).astype(int)

prec, rec, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(rec, prec)
roc_auc = roc_auc_score(y_true, y_prob)
hs_recall = recall_score(y_true, y_pred)
hs_prec = precision_score(y_true, y_pred)
hs_f1 = f1_score(y_true, y_pred)

print(f"PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}, Recall: {hs_recall:.4f}, Prec: {hs_prec:.4f}, F1: {hs_f1:.4f}")
# Expected: PR-AUC ~0.2518, ROC-AUC ~0.7328, Recall ~0.2214, Precision ~0.3640, F1 ~0.2753
```

---

## 5. Known Runner Resume and Execution Limitations

The following implementation limitations in `scripts/run_via3_xai_multiseed.py` currently preclude unconditional push-button rerun guarantees:

1. **Row-Count Completion Guards:**
   ```python
   # scripts/run_via3_xai_multiseed.py:127-129
   if len(df_faith) != expected_faith or len(df_stab) != expected_stab or len(df_coll) != expected_coll:
       return False
   return True
   ```
   The resume check tests only total row count (`len(df) == expected`), omitting checks for required primary keys `(strategy, seed, filename, method)` or verification that `status == "SUCCESS"`.
2. **Unvalidated Cache Reuse on Resume:**
   When resuming, the script loads cached per-seed CSVs without verifying that on-disk model checkpoints or cohort manifests match expected SHA-256 digests.
3. **Unconditional Master Overwrite & Timestamp Mutation:**
   Resuming a fully completed run re-writes master CSVs and stamps `multiseed_xai_comparison_summary.json` with a fresh UTC timestamp (`time.strftime(...)`), mutating its SHA-256 hash and invalidating `reproducibility_manifest.json`.
4. **LaTeX Compilation Dependency:**
   Binary `pdflatex` was not present on PATH during E5/E6 audits. Static syntax validation was executed via an independent Python parser, but automated compilation of `draft_via3_table_summary.pdf` was not verified locally.

---

## 6. Recommended Future Hardening Roadmap

To elevate this release from `PASS WITH QUALIFICATIONS` to an unconditional push-button benchmark:

1. **Semantic Primary-Key Validation:** Update `is_checkpoint_xai_completed` to assert:
   ```python
   (df_faith["status"] == "SUCCESS").sum() == expected_faith
   set(df_faith["filename"]) == set(expected_filenames)
   ```
2. **Deterministic Checkpoint Assertion:** Validate checkpoint SHA-256 against `EXPECTED_CHECKPOINTS` before loading cached CSVs.
3. **Deterministic Serialization:** Accept an optional `--freeze-timestamp` argument in `scripts/run_via3_xai_multiseed.py` to preserve summary JSON hashes.
4. **Automated TeX Compilation Pipeline:** Integrate a containerized TeX Live step into CI/CD for push-button PDF generation.
