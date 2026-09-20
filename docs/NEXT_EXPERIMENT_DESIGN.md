# Next Experiment Design — Corrected Minimum Experiment Matrix

**Status**: Design only (GPU unavailable). Authorized for review/approval; no training execution.

**Scope**: ICCAD5 (benchmark 5) only. Synthesizes audits A–G (A = methodology audit; B–H = workstreams in IMPROVEMENT_TASK_REGISTRY). Purpose: define the minimum statistically defensible experiment matrix that corrects the critical defects in A-01 (test-selection leakage), A-02 (preprocessing inconsistency), A-03 (class-weighting causal isolation), and A-09 (seed/reproducibility gaps) while holding all other factors constant.

---

## 1. Fixed Factors (Identical Across ALL Conditions)

| Factor | Value | Source / Rationale |
|--------|-------|---------------------|
| **Architecture** | `src/model_xai.py` XAI-ready CNN: `conv_early_2` (112×112), `conv_mid_2` (56×56), `conv_final_2` (28×28×32), `dense_penultimate` Dense(16, ReLU), `dense_output` Dense(1, sigmoid) | Final ICCAD5 synthesis; ~128k params |
| **Training data** | Official ICCAD5 `train/train_hs` (26 images), `train/train_nhs` (2,716 images) | Immutable source rasters |
| **Validation data** | **New disjoint split from official training** (see §3) | Fixes A-01: test is no longer used for selection |
| **Test data** | Official ICCAD5 `test/test_hs` (41), `test/test_nhs` (19,327) — **held out completely until final evaluation** | Immutable; never seen during fit/callbacks |
| **Preprocessing** | RGB conversion → PIL `Image.NEAREST` resize to 224×224 → float32 / 255.0 | Fixes A-02: matches Keras `load_img` default; no bicubic |
| **Optimizer** | `tf.keras.optimizers.Nadam()` (defaults: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7) | `src/train_xai.py:51,191` |
| **Loss** | Binary cross-entropy (`from_logits=False`) | Standard |
| **Batch size** | 32 | `src/train_xai.py:41` |
| **Epochs** | 10 | `src/train_xai.py:41–47` |
| **Callbacks** | `BestBalancedAccuracyCheckpoint` (monitor=`val_balanced_accuracy`, mode=`max`, save_weights_only=True, restore_best_weights=True) + `CSVLogger` | `src/metrics.py:139–157`; selects on **validation** BA |
| **Checkpoint selection metric** | Validation balanced accuracy (BA = 0.5 × (HS_recall + NHS_specificity)) | `src/metrics.py:15–53` |
| **Test evaluation protocol** | **Exactly once**, after training finishes and best weights are restored/frozen. No test-time selection, no re-evaluation. | Fixes A-01 |
| **Random seeds** | Per-condition seeds: **101, 202, 303** (3 seeds × 3 conditions = 9 runs) | Minimum for replication; matches seed-replication precedent |
| **RNG control** | `tf.keras.utils.set_random_seed(seed)` at run start; `numpy.random.seed(seed)`; `random.seed(seed)`; balanced sampler uses `np.random.default_rng(seed)` | Fixes A-09: explicit per-run reseeding |
| **Initialization verification** | Record SHA256 of `dense_penultimate` + `dense_output` weights **before epoch 1** for every run; assert equal across conditions at same seed | Fixes A-09: matched-seed cross-condition pairing |
| **Data manifests** | JSON manifest per run: every training/validation image path, SHA256, class label, split assignment, duplicate-group ID (if any) | Fixes A-09: content-level disjointness proof |
| **Hardware/environment manifest** | TF version, CUDA/cuDNN if GPU, CPU fallback flag, `requirements.txt` hash, git commit | Reproducibility |

---

## 2. Experimental Conditions (Only Training Condition Differs)

| Condition ID | Name | Sampler | Class Weighting | Rationale |
|--------------|------|---------|-----------------|-----------|
| **C1** | `balanced_nowt` | **Balanced 50/50** with-replacement: each epoch draws 16 HS + 16 NHS per step × 86 steps = 1,376 HS draws (≈53×/HS) and 1,376 NHS draws (≈0.5×/NHS) | **None** (`class_weight=None`) | Priority 2: best replication evidence (0% collapse across seeds 101/202/303); HS-elevated units present |
| **C2** | `class_weighted` | **Natural** (Keras directory iterator, shuffle=True) | **Inverse-frequency**: `w_0 = N/(2·N_NHS) ≈ 0.505`, `w_1 = N/(2·N_HS) ≈ 52.73` (computed from **training split only**) | Priority 3: never replicated (only seed 42 in ablation); isolates loss-weighting effect |
| **C3** | `balanced_wt` | **Balanced 50/50** (same as C1) | **Same inverse-frequency weights as C2** (applied per-sample in balanced batches) | Priority 4: tests interaction; only if justified by C1 vs C2 divergence |

**Notes**:
- C1 and C3 share the **exact same sampler** (same per-epoch HS/NHS draw counts, same replacement logic, same RNG stream at same seed).
- Class weights in C2/C3 are computed **once** from the training-split class counts (after validation holdout), not from the full 26/2716.
- No other condition (e.g., `baseline_noweight` natural sampler no-weight) is included: seed replication showed it is **seed-dependent and unreliable** (95.1% and 78.0% HS collapse at seeds 101/202). Including it would waste budget on a known-unstable recipe.

---

## 3. Validation Split Design (Fixes A-01)

**Constraint**: 26 HS training images total. Validation must be disjoint from training, stratified, and documented.

### 3.1 Split Specification

| Split | HS count | NHS count | Total | HS% |
|-------|----------|-----------|-------|-----|
| **Training** | 20 | 2,173 | 2,193 | 0.91% |
| **Validation** | 6 | 543 | 549 | 1.09% |
| **Test (official, held out)** | 41 | 19,327 | 19,368 | 0.21% |

- **Stratification**: 77% / 23% train/validation split applied per class (20/26 ≈ 76.9%, 2173/2716 ≈ 80.0% — rounded to preserve integer HS counts).
- **Grouped disjointness**: If parent-layout or near-duplicate metadata exists (e.g., same clip ID with different noise), assign **entire groups** to one split. If no metadata, use path-level split and **audit for pixel-level duplicates** (exact SHA256 match) — any duplicate pair forces both into the same split.
- **Seed for split**: Fixed seed **999** (separate from training seeds). Split is **identical across all 9 runs**.
- **Validation BA noise disclosure**: With only 6 HS validation images, a single HS misclassification changes BA by ~8.3 percentage points. This is documented; the protocol is predeclared and not changed post-hoc.

### 3.2 Duplicate Handling Protocol

1. Compute SHA256 of every source raster (1200×1200 PNG) in `train/`.
2. Build equivalence classes of exact duplicates.
3. Assign each equivalence class wholly to train or validation (never split).
4. If duplicates exist across HS/NHS (label conflict), **flag and exclude** from both splits; document count.
5. Record manifest: `data_manifest_iccad5_seed999.json` with fields: `path`, `sha256`, `class`, `split`, `duplicate_group_id`.

---

## 4. Minimum Seed Plan

| Condition | Seeds | Runs |
|-----------|-------|------|
| C1 `balanced_nowt` | 101, 202, 303 | 3 |
| C2 `class_weighted` | 101, 202, 303 | 3 |
| C3 `balanced_wt` | 101, 202, 303 | 3 |
| **Total** | | **9** |

- Seeds 101/202/303 are **predeclared**; no seed selection after seeing results.
- Each run is **independent** (fresh process, fresh RNG state initialized from the run seed).
- Failure criterion: if a run crashes or produces NaN loss, it is recorded as a failure (0/3 for that condition) — no substitution.

---

## 5. Per-Run Outputs (Required for Downstream Analysis)

Each of the 9 runs produces:

| Artifact | Path Pattern | Description |
|----------|--------------|-------------|
| Best checkpoint (weights) | `results/xai/improvements/next_exp/iccad5/{cond}_seed{seed}/best_weights.keras` | Restored best-epoch weights (weights-only) |
| Training log | `.../training_log.csv` | Per-epoch: train_loss, train_BA, val_loss, val_BA, lr |
| Checkpoint metadata | `.../checkpoint_meta.json` | `best_epoch` (0-indexed), `val_BA_at_best`, `train_BA_at_best`, `init_weights_sha256`, `best_weights_sha256`, `seed`, `condition`, `split_seed` |
| Test evaluation (once) | `.../test_eval.json` | Confusion matrix (HS=positive), BA, precision, recall, specificity, F1, AUC-ROC, AUC-PR — **computed once on official test set after freeze** |
| Full test predictions | `.../test_predictions.npz` | `y_true`, `y_pred_prob`, `y_pred_class` (threshold 0.5) for all 19,368 test images — for downstream XAI/geometry if authorized |
| Data manifest | `.../data_manifest.json` | Copy of the split manifest used |

---

## 6. Representation Audit (Priority 5 — Post-Training Analysis)

**Not a training condition**. Performed **after** all 9 runs complete, on the **frozen best checkpoints**.

| Audit | Method | Scope |
|-------|--------|-------|
| **Bottleneck collapse rate** | Forward pass on full test set (19,368); record `dense_penultimate` activation vector per sample; flag exact-zero (L2 < 1e-12) | All 9 checkpoints × full test set |
| **HS/NHS collapse stratification** | Collapse rate by true label (HS vs NHS) and case type (TP/TN/FP/FN) | All 9 checkpoints |
| **Dead unit count** | Units with zero activation on **all** 19,368 test samples | All 9 checkpoints |
| **HS-elevated units** | Per-unit AUC (HS vs NHS) on test set; count units with AUC ≥ 0.90 | All 9 checkpoints |
| **Gradient probe (Grad-CAM degeneracy)** | Compute `conv_final_2` gradient w.r.t. pre-sigmoid logit for the 11 final-pass samples (3 TP, 3 TN, 3 FP, 2 FN); flag `ptp < 1e-6` | 3 checkpoints (one per condition, seed 101) × 11 samples |
| **Preprocessing provenance check** | Verify test-time tensors match training-time NEAREST preprocessing (compare to saved training-batch tensors) | All 9 checkpoints |

**Output**: `docs/NEXT_EXPERIMENT_REPRESENTATION_AUDIT.md` with tables per condition/seed, validity flags for every metric (no silent zeros).

---

## 7. Statistical Analysis Plan (Predeclared)

### 7.1 Primary Comparison (C1 vs C2 at matched seeds)

| Metric | Test | Null | Family |
|--------|------|------|--------|
| Test BA | Wilcoxon signed-rank (paired by seed) | Median difference = 0 | Primary |
| HS collapse rate | McNemar (paired per-sample) | No difference | Primary |
| Dead unit count | Wilcoxon signed-rank | Median difference = 0 | Primary |

### 7.2 Secondary (C3 vs C1, C3 vs C2)

Same tests; **Holm-Bonferroni** correction across 3 pairwise comparisons × 3 metrics = 9 tests.

### 7.3 Descriptive (No Inference)

- Per-seed stability: SD of test BA across 3 seeds within condition.
- Collapse rate by case type (TP/TN/FP/FN) — reported with exact binomial 95% CI (Clopper-Pearson).
- HS-elevated unit count distribution.

### 7.4 Reporting Rules

- Report **all 9 runs** (no best-seed selection).
- Report exact counts (n=19,368 test) not just rates.
- Distinguish **run-level** (n=3 per condition) from **image-level** (n=19,368) uncertainty.
- No production-freeze claim; no universal XAI ranking; no isolated loss-weighting causal claim.

---

## 8. Execution Gate Checklist (Before Any Training)

- [ ] GPU visibility restored (`nvidia-smi` exit 0, TF reports `GPUS [GPU:0]`)
- [ ] B preprocessing tests pass (CPU prediction equivalence: corrected tensors match official path on 20+ deterministic rasters)
- [ ] Validation split manifest approved (lead sign-off on HS counts, duplicate audit, group disjointness)
- [ ] Initialization-hash protocol implemented and unit-tested
- [ ] All 9 run configs registered (seed, condition, output paths, failure criteria)
- [ ] Lead serializes jobs (no concurrent GPU processes)
- [ ] Immutable output directories created; write-once policy enforced

---

## 9. What This Design Does NOT Include

- Architecture search (width, depth, pooling, activation)
- Hyperparameter search (lr, batch, epochs, optimizer)
- Data augmentation
- Combined `baseline_noweight` condition (known unstable)
- XAI/occlusion/geometry regeneration (gated on representation audit + lead approval)
- Any claim that the official test set is a "virgin holdout" (it has informed prior recipe choices)

---

## 10. Change Control

Any deviation from this design requires:
1. Written justification referencing a specific audit finding (A-01 through A-09)
2. Lead approval before implementation
3. Amendment appended to this document (versioned)

---

**End of Design Document**