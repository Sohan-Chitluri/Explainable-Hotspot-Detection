# XAI Model Retraining with Clean 3-Way Split

**Date:** 2026-09-20  
**Status:** In progress  
**Purpose:** Retrain all XAI CNN models to eliminate test/validation leakage identified in the audit

## Summary

The audit (`docs/TEST_VALIDATION_LEAKAGE_AUDIT.md`) identified that the original XAI models (xai_cnn_iccad1-5.keras) were trained and checkpoint-selected using the official ICCAD test set as validation data, creating biased performance estimates.

This document tracks the retraining effort using a proper 3-way split:

- **Train:** 85% of `iccad{N}/train/`, stratified per class
- **Validation:** 15% carve-out from `iccad{N}/train/`, used only for checkpoint selection
- **Test:** `iccad{N}/test/` (untouched until final evaluation)

## Implementation Status

### 1. Clean Split Function ✅
- Implemented in `src/data.py::make_three_way_split()`
- Stratified split preserves class balance
- Configurable val_frac (default 15%)
- Deterministic with seed=42
- Returns split manifest for reproducibility

### 2. Training Script Update ✅
- `src/train_xai.py` updated to use `make_three_way_split()`
- Validation generator points to carve-out, not official test
- Test generator loaded separately for post-training evaluation only
- Detailed docstring explains the split architecture

### 3. Model Retraining 🟡 IN PROGRESS
- Command: `python -m src.train_xai --models-dir models/xai/retrained_clean --results-dir results/xai_training/retrained_clean`
- Trains all 5 benchmarks (iccad1-5)
- Epochs per benchmark: {1:5, 2:5, 3:10, 4:10, 5:10}
- Saves new models to `models/xai/retrained_clean/` (preserves originals in `models/xai/`)
- Split manifests saved alongside metrics for audit trail

### 4. Comparison Analysis 📋 PENDING
- Script: `scripts/compare_old_vs_clean.py`
- Evaluates both old and new models on official test set
- Generates comparison table: old balanced accuracy vs new, with delta
- Output: `results/xai_training/leaky_vs_clean_comparison.json`

## Expected Outcomes

The retraining should produce one of these outcomes:

1. **New models perform worse on official test set** (most likely)
   - Confirms that checkpoint selection inflated old model performance
   - Delta (new - old) will be negative
   - Magnitude of delta quantifies the leakage impact

2. **New models perform similarly**
   - Suggests leakage impact was small (unlikely given the severity of the mechanism)
   - Possible if the test set distribution matches training distribution closely

3. **New models perform better** (unlikely)
   - Would suggest training hyperparameters or architecture benefit from the cleaner split
   - Would reduce confidence in the leakage hypothesis

## File Structure

```
models/xai/
├── xai_cnn_iccad1.keras          (original, checkpoint-biased)
├── xai_cnn_iccad2.keras          (original, checkpoint-biased)
├── xai_cnn_iccad3.keras          (original, checkpoint-biased)
├── xai_cnn_iccad4.keras          (original, checkpoint-biased)
├── xai_cnn_iccad5.keras          (original, checkpoint-biased)
└── retrained_clean/
    ├── xai_cnn_iccad1.keras      (new, clean split)
    ├── xai_cnn_iccad2.keras      (new, clean split)
    ├── xai_cnn_iccad3.keras      (new, clean split)
    ├── xai_cnn_iccad4.keras      (new, clean split)
    └── xai_cnn_iccad5.keras      (new, clean split)

results/xai_training/
├── retrained_clean/              (training curves and metrics for new models)
│   ├── iccad1/
│   │   ├── metrics.json
│   │   ├── split_manifest.json
│   │   └── balanced_accuracy_vs_epoch.png
│   ├── iccad2/, iccad3/, iccad4/, iccad5/
│   └── metrics.json              (summary across all 5)
└── leaky_vs_clean_comparison.json (side-by-side comparison)
```

## Next Steps

1. Wait for training to complete (ETA: 1-2 hours on CPU)
2. Run comparison script: `python scripts/compare_old_vs_clean.py`
3. Review `results/xai_training/leaky_vs_clean_comparison.json`
4. Document findings in a new audit report

## References

- `docs/TEST_VALIDATION_LEAKAGE_AUDIT.md` — identifies the problem
- `docs/THREE_WAY_SPLIT.md` — implementation details of the split
- `docs/ICCAD5_CHECKPOINT_FIX.md` — earlier incomplete attempt at a fix
- `src/data.py::make_three_way_split()` — canonical entry point
- `src/train_xai.py` — updated training script
