# Training Protocol Audit — Agent A
**Audit Date:** 2026-09-18 | **Scope:** Read-only source/document/result audit at baseline. No training, neural inference, installs, driver changes, checkpoint edits, or historical-result edits. GPU unavailable (nvidia-smi exit 6, TF GPUS []).

---

## Executive Summary

**CRITICAL FINDING:** The training protocol has **test-set leakage** — the official ICCAD test split is used as "validation" during training, the `BestBalancedAccuracyCheckpoint` callback selects the best epoch by querying this test split, and final evaluation queries the same test split again. This is not a naming issue; it is a verified selection bias that invalidates unbiased generalization claims for all checkpoints produced by this pipeline (production XAI, checkpoint-fix, imbalance-ablation, seed-replication).

**Protocol Correction Required:** Keep official test files immutable; create a versioned, disjoint development validation split from official training (grouped by parent layout/near-duplicate family with stratification and overlap audit). Use training-only grouped inner folds or a training-only fixed-epoch policy chosen before new evaluation. Predeclare one protocol. Never silently rename the old test data "validation" and claim a fresh test.

---

## 1. Exact Train/Validation/Test Splitting Implementation

### 1.1 Directory Structure (Immutable Official Splits)
```
iccad-official/
  iccad{1..5}/
    train/
      train_hs/   # Hotspot training images
      train_nhs/  # Non-hotspot training images
    test/
      test_hs/    # Hotspot test images (OFFICIAL TEST SET)
      test_nhs/   # Non-hotspot test images (OFFICIAL TEST SET)
```

**Evidence — `src/data.py:10–22`:**
```python
def dataset_analysis(folder: str):
    base_dir = os.path.join(os.path.dirname(folder))
    train_dir = os.path.join(base_dir, "train")
    validation_dir = os.path.join(base_dir, "test")  # LINE 16: TEST DIR NAMED "validation"

    train_hotspot_dir = os.path.join(train_dir, "train_hs")
    train_not_hotspot_dir = os.path.join(train_dir, "train_nhs")
    validation_hotspot_dir = os.path.join(validation_dir, "test_hs")   # LINE 21
    validation_not_hotspot_dir = os.path.join(validation_dir, "test_nhs")  # LINE 22
```

**Counts Verified (all benchmarks):**
| Benchmark | Train HS | Train NHS | Test HS | Test NHS | Test Total |
|-----------|----------|-----------|---------|----------|------------|
| iccad1    | 99       | 340       | 226     | 4,679    | 4,905      |
| iccad2    | 174      | 5,285     | 498     | 41,298   | 41,796     |
| iccad3    | 909      | 4,643     | 1,808   | 46,333   | 48,141     |
| iccad4    | 95       | 4,452     | 177     | 31,890   | 32,067     |
| iccad5    | 26       | 2,716     | 41      | 19,327   | 19,368     |

### 1.2 Data Generators — Only Two Splits Exist
**Evidence — `src/data.py:54–75`:**
```python
def data_extractor(train_dir: str, validation_dir: str, batch_size: int = 32):
    train_data_gen = image_gen_train.flow_from_directory(
        directory=train_dir, shuffle=True, ...)
    val_data_gen = image_gen_val.flow_from_directory(
        directory=validation_dir, shuffle=False, ...)  # validation_dir = test dir
    return train_data_gen, val_data_gen
```
**No third split exists.** The function signature and implementation only produce train and validation generators.

---

## 2. Checkpoint Selection Mechanism

### 2.1 BestBalancedAccuracyCheckpoint Callback
**Evidence — `src/metrics.py:116–157`:**
```python
class BestBalancedAccuracyCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tp = logs.get("val_true_positives")
        tn = logs.get("val_true_negatives")
        fp = logs.get("val_false_positives")
        fn = logs.get("val_false_negatives")
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = 0.5 * (sensitivity + specificity)
        if bal_acc > self.best_value:  # Strict > keeps first tied epoch
            self.best_value = bal_acc
            self.best_epoch = epoch
            self.best_weights = [w.copy() for w in self.model.get_weights()]

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)  # Restores best epoch weights
```

### 2.2 Training Loop Wiring
**Evidence — `src/train_xai.py:178–217`:**
```python
train_dir, val_dir, counts = dataset_analysis(folder)  # val_dir = test dir
train_data_gen, val_data_gen = data_extractor(train_dir, val_dir, ...)

best_ckpt_cb = BestBalancedAccuracyCheckpoint()
history = model.fit(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen,        # TEST SET fed as validation_data
    class_weight=class_weights,
    callbacks=[best_ckpt_cb],            # Selects best epoch on TEST SET
)

# FINAL EVALUATION ON SAME TEST SET
final_metrics, y_true, y_pred, probs = evaluate_xai_model(model, val_data_gen)
```

**Key Lines:**
- Line 178: `val_dir` = test directory (from `dataset_analysis`)
- Line 201: `validation_data=val_data_gen` — test set used during training
- Line 197: `BestBalancedAccuracyCheckpoint` monitors `val_*` metrics from test set
- Line 217: `evaluate_xai_model(model, val_data_gen)` — final eval on same test set

---

## 3. Verified Test-Set Leakage Evidence

### 3.1 Seed Replication Artifacts (Smoking Gun)
**File:** `results/xai/iccad5_seed_replication/training/balanced_sampling_seed1/metrics.json`
```json
{
  "class_indices": {"test_hs": 0, "test_nhs": 1},  // Explicit test labels
  "total_val": 19368,                               // Test set size
  "checkpoint_selection": {
    "selected_epoch": 7,
    "selected_epoch_val_balanced_accuracy": 0.9742644878200217
  },
  "balanced_accuracy": 0.9742644878200217          // Exactly equals selected score
}
```
**File:** `results/xai/iccad5_checkpoint_fix/training/iccad5/metrics.json`
```json
{
  "checkpoint_selection": {
    "selected_epoch": 8,
    "selected_epoch_val_balanced_accuracy": 0.9649252215086439
  },
  "balanced_accuracy": 0.9649252215086439
}
```

**CPU Verification (read-only):** Recomputed balanced accuracy from saved confusion matrix counts matches the stored "selected_epoch_val_balanced_accuracy" **exactly** for both checkpoints.

### 3.2 Imbalance Ablation Repeats Same Cycle
**Evidence — `src/run_iccad5_imbalance_ablation.py:96–130`:**
```python
train_dir, val_dir, counts = dataset_analysis(str(DATA_ROOT) + "/")
_, val_data_gen = data_extractor(train_dir, val_dir, batch_size=BATCH_SIZE)

history = model.fit(
    train_ds,  # or train_data_gen
    validation_data=val_data_gen,      // TEST SET
    callbacks=[best_ckpt_cb],          // Selects on TEST SET
)

final_metrics, y_true, y_pred, probs = evaluate_xai_model(model, val_data_gen)  // TEST SET
```

**Evidence — `docs/ICCAD5_IMBALANCE_ABLATION.md:91–92`** explicitly acknowledges: "validation/test is the unmodified test split."

### 3.3 Seed Replication Repeats Same Cycle
**Evidence — `src/run_iccad5_seed_replication.py:107–141`:** Identical pattern — test set used as validation_data, BestBalancedAccuracyCheckpoint selects on it, final evaluation on it.

---

## 4. Baseline Distinction (Not Accused of Same Callback Behavior)

**Evidence — `src/train_baseline.py:159–187,220–221`:**
- Monitors test each epoch (exposure)
- **No best-epoch callback** — saves final weights
- Asymmetric selection: baseline = final-epoch; XAI = best-test-epoch

---

## 5. Seed Handling & Sampler Behavior

### 5.1 Production Training (`src/train_xai.py`)
- Line 346: `tf.keras.utils.set_random_seed(args.seed)` — **once before benchmark loop**
- Line 359: `train_one_benchmark(..., seed=args.seed)` — **parameter recorded but NOT reseeded inside function**
- Training ICCAD5 alone vs after iccad1–4 consumes different RNG state; both record seed=42

### 5.2 Balanced Sampling Generator (`src/run_iccad5_imbalance_ablation.py:57–86`)
```python
def balanced_batch_generator(hs_paths, nhs_paths, batch_size, seed):
    rng = np.random.default_rng(seed)  # Local RNG per condition
    half = batch_size // 2  # 16
    while True:
        hs_batch = rng.choice(hs_paths, size=half, replace=True)      # 26 HS images
        nhs_batch = rng.choice(nhs_paths, size=batch_size - half, replace=True)  # 2716 NHS
        ...
```
- **With replacement** for both classes (HS pool only 26 images)
- 86 steps/epoch × 16 HS = 1,376 HS draws/epoch → **~52.9 draws per HS image/epoch**
- 86 steps/epoch × 16 NHS = 1,376 NHS draws/epoch → **~0.51 draws per NHS image/epoch**
- Natural iteration: 2,742 observations/epoch; Balanced: 2,752 observations/epoch
- BatchNorm present at `src/model_xai.py:30,37,45` → batch statistics differ

### 5.3 Initialization Checksum (Not Cryptographic)
**Evidence — `src/run_iccad5_seed_replication.py:114,153`:**
```python
init_weight_checksum = float(np.sum([np.abs(w).sum() for w in model.get_weights()]))
```
- Absolute-sum checksum, not SHA256
- "Must all differ" text contradicts intentional same-seed cross-condition pairing

---

## 6. Class Weighting Implementation

### 6.1 Compute Function
**Evidence — `src/train_xai.py:108–118`:**
```python
def compute_class_weights(counts: dict) -> dict[int, float]:
    n_hs = counts["train_hs"]
    n_nhs = counts["train_nhs"]
    total = n_hs + n_nhs
    w0 = total / (2.0 * n_hs)   # Class 0 = HS (positive)
    w1 = total / (2.0 * n_nhs)  # Class 1 = NHS (negative)
    return {0: float(w0), 1: float(w1)}
```

### 6.2 Application
**Evidence — `src/train_xai.py:183,202`:**
```python
class_weights = compute_class_weights(counts)
history = model.fit(..., class_weight=class_weights, ...)
```

### 6.3 Ablation Conditions (Isolation Failure)
| Condition | Class Weight | Balanced Sampling | Checkpoint Selection |
|-----------|--------------|-------------------|---------------------|
| A. Baseline (noweight) | ❌ | ❌ | ✅ (test) |
| B. Class Weighted | ✅ | ❌ | ✅ (test) — **reused existing checkpoint, not retrained** |
| C. Balanced Sampling | ❌ | ✅ | ✅ (test) |

**Critical:** Condition B was **not retrained** in ablation — existing checkpoint reused (`docs/ICCAD5_CHECKPOINT_FIX.md:98–105` acknowledges "changed realized trajectory despite same nominal seed and CPU-only retrain"). Seed equality ≠ equal initialization/stochastic trajectory/hardware.

---

## 7. Label Convention Consistency

**Evidence — `src/metrics.py:15–20`:**
```python
def binary_metrics(y_true, y_pred, positive_label=0):
    # positive_label=0 treats HS (alphabetically first folder) as positive
```

**Evidence — `src/train_xai.py:53–61`:**
```python
TRAIN_METRICS = ["Accuracy", "Precision", "Recall", "TruePositives", ...]
```
Keras native positive label = 1 (NHS), so epoch TP/TN meanings are **swapped** relative to reported HS-positive counts. Balanced accuracy is invariant under class swap — **not a BA computation bug**, but epoch-level TP/TN labels in logs are inverted vs. reported metrics.

---

## 8. Epoch Count Discrepancy

**Evidence — `src/train_xai.py:41–47`:**
```python
BENCHMARK_EPOCHS = {1: 5, 2: 5, 3: 10, 4: 10, 5: 10}
```

**Evidence — `docs/FINAL_RESEARCH_REPORT.md:135–136`:** Claims "XAI epochs=10 for all benchmarks" — **FALSE for iccad1, iccad2 (5 epochs each)**.

**Verified — `results/xai_training/iccad1/metrics.json:23`:** `"epochs": 5`

---

## 9. Summary of Protocol Defects

| # | Defect | Severity | File:Line Evidence |
|---|--------|----------|-------------------|
| 1 | Test set used as validation during training | CRITICAL | `data.py:16`, `train_xai.py:201` |
| 2 | Best-epoch selection on test set via callback | CRITICAL | `metrics.py:139–157`, `train_xai.py:197` |
| 3 | Final evaluation on same test set | CRITICAL | `train_xai.py:217` |
| 4 | No independent test set exists in pipeline | CRITICAL | `data.py:54–75` (only 2 generators) |
| 5 | Ablation/replication repeat test-as-validation cycle | CRITICAL | `run_iccad5_imbalance_ablation.py:118`, `run_iccad5_seed_replication.py:130` |
| 6 | Condition B (class-weighted) not retrained in ablation | HIGH | `run_iccad5_imbalance_ablation.py:12–16` |
| 7 | Seed not reseeded per benchmark in production | HIGH | `train_xai.py:346,359` |
| 8 | Balanced sampling uses with-replacement (52× HS oversampling) | HIGH | `run_iccad5_imbalance_ablation.py:57–86` |
| 9 | Initialization checksum not cryptographic | MEDIUM | `run_iccad5_seed_replication.py:114` |
| 10 | Epoch count misreported in final report | MEDIUM | `train_xai.py:41–47` vs `FINAL_RESEARCH_REPORT.md:135` |
| 11 | Keras native TP/TN inverted vs HS-positive reporting | MEDIUM | `metrics.py:19` vs `train_xai.py:53–61` |

---

## 10. Corrected Protocol Design

### 10.1 Split Architecture (Required)
```
iccad-official/
  iccad{1..5}/
    train/
      train_hs/
      train_nhs/
    test/                    # OFFICIAL TEST — FROZEN, NEVER TOUCHED DURING TRAINING
      test_hs/
      test_nhs/
    dev_validation/          # NEW: versioned development validation split
      dev_val_hs/            # Disjoint from train & test, grouped by layout family
      dev_val_nhs/
```

### 10.2 Split Creation Rules
1. **Source:** Draw dev_validation **only from official train directories** (train_hs, train_nhs)
2. **Grouping:** Group by parent layout/near-duplicate family (audit for overlap first)
3. **Stratification:** Preserve class proportions within groups
4. **Size:** Minimum viable HS count per benchmark (iccad5: 26 HS → max ~5 for dev_val)
5. **Versioning:** `dev_validation_v1`, `dev_validation_v2`, ... with manifest (file lists, hashes, group IDs)
6. **Immutability:** Once created, dev_validation split is frozen for that protocol version

### 10.3 Training Protocol (Corrected)
```python
# CORRECTED PSEUDO-CODE
def train_one_benchmark_corrected(benchmark, data_root, seed, protocol_version):
    # 1. Load official splits
    train_dir = data_root / f"iccad{benchmark}" / "train"
    test_dir = data_root / f"iccad{benchmark}" / "test"
    dev_val_dir = data_root / f"iccad{benchmark}" / f"dev_validation_{protocol_version}"
    
    # 2. Create generators
    train_gen = flow_from_directory(train_dir, shuffle=True, seed=seed)
    dev_val_gen = flow_from_directory(dev_val_dir, shuffle=False)
    test_gen = flow_from_directory(test_dir, shuffle=False)  # NEVER passed to fit()
    
    # 3. Train with dev_validation only
    best_ckpt = BestBalancedAccuracyCheckpoint()  # Monitors dev_val_* metrics
    history = model.fit(
        train_gen,
        validation_data=dev_val_gen,    # DEV VALIDATION ONLY
        callbacks=[best_ckpt],
        class_weight=compute_class_weights(train_counts),
    )
    
    # 4. Final evaluation ONCE on official test (after training complete)
    test_metrics = evaluate(model, test_gen)  # Single query, no selection
    
    return {
        "dev_val_selection": best_ckpt.best_epoch,  # Transparent
        "test_metrics": test_metrics,                # Unbiased
        "protocol_version": protocol_version,
        "dev_val_manifest_hash": hash(dev_val_manifest),
    }
```

### 10.4 Checkpoint Selection Policy (Predeclared)
**Option A — Fixed Epoch (Simplest, Recommended for Tiny HS):**
- Choose epoch budget *before* any training (e.g., epoch 10 for all)
- No selection callback; save final weights
- Eliminates selection bias entirely

**Option B — Dev-Validation Selection (If Dev Split Adequate):**
- Use `BestBalancedAccuracyCheckpoint` on `dev_val_*` metrics only
- Report selected epoch and dev_val score transparently
- Test set queried **exactly once** post-training

**Option C — Nested CV (If Compute Permits):**
- Outer loop: test set (final report)
- Inner loop: dev_validation selection
- Requires adequate HS counts — **not viable for iccad5 (26 HS)**

### 10.5 Seed & Reproducibility Requirements
1. `tf.keras.utils.set_random_seed(seed)` **inside** `train_one_benchmark` per benchmark
2. `ImageDataGenerator` — pass explicit `seed` to `flow_from_directory`
3. Balanced sampler: pass seed to `np.random.default_rng(seed)` per condition
4. Record: **SHA256 of initial weights**, full optimizer config, TF/NumPy/CUDA/cuDNN versions, commit hash, exact command
5. Equal seeds across conditions → verify equal initialization (not just "differs")

### 10.6 Ablation/Replication Protocol (Corrected)
| Condition | Class Weight | Balanced Sampling | Selection Split | Test Query |
|-----------|--------------|-------------------|-----------------|------------|
| A. Baseline | ❌ | ❌ | dev_validation | 1× post-training |
| B. Class Weighted | ✅ | ❌ | dev_validation | 1× post-training |
| C. Balanced Sampling | ❌ | ✅ | dev_validation | 1× post-training |

- **All three conditions fully retrained** from scratch per seed
- Matched seeds across conditions (e.g., 101, 202, 303 for A/B/C)
- Initial-weight SHA256 equality verified for same-seed cross-condition pairs
- No reuse of historical checkpoints

---

## 11. Impact on Existing Results

| Artifact | Status | Interpretation |
|----------|--------|----------------|
| Production XAI checkpoints (`models/xai/`) | **Test-selected** | Descriptive scores on explored test set; unbiased generalization NOT established |
| Checkpoint-fix (`models/xai/retrained/`) | **Test-selected** | Same; "improves held-out performance" not established |
| Imbalance ablation conditions A/C | **Test-selected** | Balanced sampling avoids collapse in these runs — descriptive only |
| Condition B (class-weighted) | **Historical, not retrained** | Different trajectory acknowledged; not a controlled comparison |
| Seed replication (3 seeds) | **Test-selected** | 0% collapse for balanced sampling across 3 seeds — descriptive, not proof |
| Baseline CNN (`src/train_baseline.py`) | **Final-epoch, test-monitored** | Asymmetric vs XAI best-test-epoch; not directly comparable |

**No existing artifact is invalidated as a record of its historical computation.** All remain valid descriptive records. The defect is in the *interpretation* of these scores as unbiased generalization estimates or as evidence for causal claims about class weighting vs. sampling.

---

## 12. Acceptance Gates for Future Training

Before any training/inference resumption:
1. [ ] Dev validation split created, manifest hashed, disjointness audited (group overlap check)
2. [ ] Protocol version declared (e.g., `protocol_v1_devval_fixedepoch10`)
3. [ ] Selection policy predeclared (fixed-epoch vs dev-val-selection)
4. [ ] All seeds/endpoints/failure criteria registered
5. [ ] Immutable outputs + initialization SHA256 specified
6. [ ] GPU availability confirmed; jobs serialized by lead
7. [ ] No best-seed selection after inspecting test performance
8. [ ] Predeclare representative checkpoint choice independently of evaluation values

---

## 13. Files Created/Modified by This Audit
- **Only output:** `docs/TRAINING_PROTOCOL_AUDIT.md` (this file)

---

*End of Audit. All findings anchored to source lines. No neural results regenerated.*