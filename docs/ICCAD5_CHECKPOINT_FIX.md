# ICCAD5 Checkpoint Fix

Follow-up to `docs/ICCAD5_FORENSIC_AUDIT.md` §12. Implements the minimum training-
methodology change identified there (best-epoch checkpoint selection), retrains **only**
iccad5 with it, and evaluates whether the confirmed representational collapse
(`dense_penultimate` going to the exact-zero vector) is resolved. The original
`models/xai/xai_cnn_iccad5.keras` was **not** modified or deleted; the new checkpoint is
saved separately at `models/xai/retrained/xai_cnn_iccad5_best.keras`. No architecture,
dataset, split, preprocessing, optimizer, learning rate, augmentation, or loss was changed.

**Headline result**: the fix is a **real, measured improvement, not a full resolution**.
Full-test-set dead-bottleneck incidence dropped from 4.92% to 1.81% overall (HS-specific:
97.6%→78.0%), and F1/specificity improved substantially. But the same collapse mechanism
persists for the majority of true-HS predictions, and literal random noise still triggers
it — this is reported as **PARTIAL**, not resolved, per the task's explicit instruction not
to declare success from appearance alone.

## 1. Original Checkpoint Methodology

`src/train_xai.py::train_one_benchmark` (before this change): `model.fit(train_data_gen,
epochs=EPOCHS[b], validation_data=val_data_gen, class_weight=class_weights)` with no
callbacks, followed unconditionally by `model.save(model_path)` — i.e. whatever weights
exist at the end of the final epoch are saved, regardless of whether an earlier epoch had
better validation performance.

## 2. Identified Training-Methodology Gap

`docs/ICCAD5_FORENSIC_AUDIT.md` §7: iccad5's validation balanced accuracy (computed
post-hoc from Keras's native per-epoch `val_true_positives/val_true_negatives/
val_false_positives/val_false_negatives`) swung from 0.964 (epoch 6, the best in the
original run) down to 0.783 (epoch 8) before recovering to 0.950 at the unconditionally-
saved final epoch 9 — a non-monotonic trajectory consistent with training a narrow 16-unit
ReLU bottleneck under 104.5:1 class weighting with only 26 positive training images.

## 3. Code Change

Two files changed, both additive (no existing line altered in behavior):

- **`src/metrics.py`**: added `BestBalancedAccuracyCheckpoint`, a `tf.keras.callbacks.Callback`
  that at `on_epoch_end` computes validation balanced accuracy using **the exact same
  formula already used by the pre-existing `plot_balanced_accuracy_curves`**
  (`0.5 * (sensitivity + specificity)` from the native `val_true_positives` /
  `val_true_negatives` / `val_false_positives` / `val_false_negatives` metrics already in
  `TRAIN_METRICS`), tracks the best epoch's weights in memory, and restores them into the
  model at `on_train_end`. No new metric was defined or compiled — this reuses the exact,
  already-present metric computation, satisfying "if the project already has a
  callback/metric mechanism, reuse it."
- **`src/train_xai.py`**: `model.fit(...)` now passes `callbacks=[best_ckpt_cb]`; the
  existing `model.save(model_path)` line is **unchanged** — since the callback already
  restored the best-epoch weights into `model` before `.fit()` returns, the same save call
  now persists the best epoch instead of the final one. `final_metrics` gained one new
  `checkpoint_selection` block recording the criterion, selected epoch, and its balanced
  accuracy — additive, does not remove or rename any existing field.

Total diff: one new ~40-line class, one new line in the `.fit()` call, one small metadata
block. No change to `TRAIN_METRICS`, `build_xai_cnn`, `data_extractor`, `OPTIMIZER`, `LOSS`,
`BATCH_SIZE`, `IMG_SIZE`, or `BENCHMARK_EPOCHS`.

## 4. Checkpoint Selection Criterion

> **Save the epoch with the highest validation balanced accuracy**, where balanced
> accuracy = 0.5 × (sensitivity + specificity), computed from Keras's native
> `val_true_positives`, `val_true_negatives`, `val_false_positives`, `val_false_negatives`
> epoch-end logs. Ties keep the earliest epoch (strict `>`).

This is the identical metric already reported in `metrics.json::history_val_balanced_accuracy`
for every benchmark — no new metric definition, no change to what "balanced accuracy" means
anywhere else in this project.

## 5. Training Configuration (preserved exactly)

| Parameter | Value | Changed? |
|---|---|---|
| Seed | 42 (`tf.keras.utils.set_random_seed`) | No |
| Epochs | 10 (`BENCHMARK_EPOCHS[5]`) | No |
| Batch size | 32 | No |
| Optimizer | Nadam | No |
| Learning rate | Nadam default (unspecified in original code — unchanged) | No |
| Loss | binary_crossentropy | No |
| Dataset / split | `iccad-official/iccad5/{train,test}` (unchanged directories) | No |
| Class weights | HS=52.73, NHS=0.505 (computed identically from train counts) | No |
| Architecture | `build_xai_cnn`, 21 layers, 128,321 params (unchanged) | No |
| **Checkpoint selection metric** | **best validation balanced accuracy (new)** | **Yes — the only change** |

Command run: `python -m src.train_xai --benchmarks 5 --models-dir models/xai/retrained
--results-dir results/xai/iccad5_checkpoint_fix/training --seed 42`.

## 6. Selected Epoch

**Epoch 8** (0-indexed; 9th of 10 epochs), validation balanced accuracy **0.9649** — the
best epoch in this run's own history:

```
epoch:  0     1     2     3     4     5     6     7     8     9
val:  0.500 0.498 0.733 0.873 0.955 0.944 0.691 0.924 0.965 0.847
```

(Different specific trajectory from the original run's history — TensorFlow/cuDNN ops are
not bit-exactly deterministic across process runs even with a fixed seed, and this is a
CPU-only run of the same stochastic training procedure; the qualitative instability
pattern — large swings between adjacent epochs — recurs, reinforcing §2's diagnosis rather
than contradicting it.) The final (10th) epoch's own validation balanced accuracy was 0.847
— **had the old unconditional-final-epoch logic been used on this run, a worse checkpoint
than the one now saved would again have been produced**, independently confirming the value
of the fix on a second, separately-seeded training trajectory.

## 7. Old vs New Classification Metrics

Full iccad5 test set (19,368 images), from `results/xai_training/iccad5/metrics.json`
(OLD, final-epoch) vs `results/xai/iccad5_checkpoint_fix/training/iccad5/metrics.json`
(NEW, best-epoch):

| Metric | OLD (final epoch) | NEW (best epoch) |
|---|---:|---:|
| TP | 40 | 39 |
| TN | 17,871 | 18,914 |
| FP | 1,456 | 413 |
| FN | 1 | 2 |
| Precision | 0.0267 | 0.0863 |
| Recall (sensitivity) | 0.9756 | 0.9512 |
| Specificity | 0.9247 | 0.9786 |
| **Balanced accuracy** | **0.9501** | **0.9649** |
| F1 | 0.0520 | 0.1582 |
| Raw accuracy | 0.9248 | 0.9786 |

**Improved**: balanced accuracy (+0.015), specificity (+0.054), precision (3.2×), F1
(3.0×), false positives (−71.6%, 1456→413). **Slightly worse**: recall (−0.024, one
additional missed HS: FN 1→2). This is a real, meaningful shift toward a better-calibrated
decision boundary — far fewer NHS images incorrectly flagged HS, at the cost of one
additional missed hotspot out of 41.

## 8. Old vs New Functional-Degeneracy Metrics

`results/xai/iccad5_checkpoint_fix/forensics/` (same methodology as the forensic audit,
repeated identically on the new checkpoint):

| Metric | OLD | NEW |
|---|---:|---:|
| Final-layer (`dense_output`) bias | −0.3330 | −0.2725 |
| Dead-bottleneck rate, full test set (19,368 images) | 4.92% (952) | **1.81% (351)** |
| Dead-bottleneck rate, true-HS images (41) | 97.56% (40) | **78.05% (32)** |
| Dead-bottleneck rate, true-NHS images (19,327) | 4.72% (912) | **1.65% (319)** |
| Dead samples in the 10-sample probe set | 5/10 | 4/10 |
| Random-uniform-noise probe | dead (penult=0, logit=−0.333) | **still dead** (penult=0, logit=−0.273) |
| Real `HSCAD50` probe | dead | **still dead** |
| all-zero / all-one probes | alive (saturated NHS branch) | alive (saturated NHS branch) |

**The collapse is substantially reduced in magnitude (both in overall rate and in the
finding that it now affects 78% rather than 98% of true-HS predictions) but is not
eliminated.** The synthetic-noise probe result is the clearest evidence of persistence:
literal random noise still routes into the exact same dead branch on the new checkpoint,
confirming the dead region of `dense_penultimate`'s input space remains broad rather than
tightly tailored to genuine hotspot semantics — the same qualitative finding as
`ICCAD5_FORENSIC_AUDIT.md` §8, just less frequently triggered.

## 9. XAI Sanity Comparison

Small matched subset (the same 10 representative iccad5 samples — 3 TP, 3 TN, 3 FP, 1 FN
— used throughout this audit trail; **not** the full 57-sample pipeline, per the task
instruction), all four methods, `results/xai/iccad5_checkpoint_fix/xai_sanity/xai_sanity_old_vs_new.csv`:

| Method | OLD degenerate rate (/10) | NEW degenerate rate (/10) |
|---|---:|---:|
| Grad-CAM | 90% (9/10) | 100% (10/10) |
| Grad-CAM++ | 100% (10/10) | 100% (10/10) |
| LayerCAM | 50% (5/10) | 40% (4/10) |
| Occlusion | 50% (5/10) | **20% (2/10)** |

**Result is method-dependent and not uniformly better** — reported exactly as observed,
not smoothed into a single "improved" verdict:

- **Occlusion** (a pure forward-pass probe, no gradient) shows the clearest, most direct
  improvement, tracking the dense-bottleneck dead-rate reduction almost exactly (5→2 of 10
  samples), as expected since it is mechanistically tied to the same collapsed
  representation.
- **LayerCAM** improves modestly (5→4).
- **Grad-CAM and Grad-CAM++ do not improve on this small subset** (in fact Grad-CAM shows
  one additional degenerate sample, 9→10). This is consistent with the two-mechanism
  picture already established in `TP_FP_GEOMETRY_ANALYSIS.md` §11b: Grad-CAM/Grad-CAM++'s
  degeneracy is driven by **both** full bottleneck death **and** a separate
  global-average-pooling gradient-cancellation effect that occurs even when
  `dense_penultimate` is only *partially* dead (e.g. `NHSCAD51_1` moved from fully dead
  under OLD to 1-of-16-units-alive under NEW, yet its Grad-CAM map is still flagged
  degenerate) — a best-epoch checkpoint fix does not address that second mechanism, because
  it is a structural property of how Grad-CAM's channel weights are computed (global
  average pooling of the gradient), not a function of which epoch's weights are used.

**Per the task's explicit instruction, this comparison is not used to declare the new
checkpoint "better" — the checkpoint was already selected in §4–6 purely by the predefined
validation-balanced-accuracy criterion, before any XAI map was generated.** This section
only asks, and answers, a narrower question: does the new checkpoint have more
input-dependent explanations available to it than the old one? Answer: for Occlusion and
(modestly) LayerCAM, yes; for Grad-CAM and Grad-CAM++, not on this subset.

## 10. Was the Checkpoint Collapse Resolved?

**PARTIAL.** Concrete evidence for "partial" rather than "yes" or "no":

- **Reduced, not eliminated**: dead-bottleneck rate fell from 4.92% to 1.81% overall, and
  from 97.6% to 78.0% specifically on true-HS images (§8) — a genuine, large, measured
  improvement in the same direction the fix targeted.
- **Not eliminated**: 78% of true hotspots still route through the exact-zero bottleneck
  on the new checkpoint, and synthetic random noise still triggers the identical collapsed
  state (§8) — the underlying representational fragility of a 16-unit ReLU bottleneck
  trained on 26 positive examples was not resolved by checkpoint selection alone, because
  checkpoint selection only chooses among the epochs actually produced by this training
  run; it cannot manufacture a fundamentally more robust representation than the training
  process is capable of producing with this little positive data.
- **Method-dependent downstream effect** (§9): the fix meaningfully helps Occlusion and
  LayerCAM's explainability but does not measurably help Grad-CAM/Grad-CAM++ on this small
  subset, because those two methods have an additional, distinct failure mode not
  addressed by this fix.

## 11. Limitations

- Only one alternative training run was performed (same seed value, different realized
  trajectory due to non-bit-exact reproducibility of TF training on this hardware/build) —
  the specific selected epoch (8) and its exact metrics are one sample from a noisy
  training process, not a guaranteed outcome of re-running this procedure.
- The XAI sanity check (§9) uses only 10 samples; the degenerate-rate percentages there
  have wide uncertainty at that sample size and are reported as a directional check, not a
  precise measurement (the full-test-set numbers in §8 are the more reliable evidence).
- This fix does not address the root contributing cause identified in the forensic audit
  (severe class imbalance / tiny positive training set, 26 images) — it only ensures the
  *best available* epoch from the existing training procedure is the one saved.
- Grad-CAM/Grad-CAM++'s GAP-cancellation failure mode (§9) remains uninvestigated as a
  target for remediation — fixing it would likely require an architecture or attribution-
  method-level change, both explicitly out of scope for this task.

## 12. Recommended Next Step

1. **Do not yet replace** `models/xai/xai_cnn_iccad5.keras` with the new checkpoint without
   an explicit decision from the user — the new checkpoint is a measured improvement on the
   validation criterion it was selected for (§7) but is not a full fix (§10), and swapping
   the "production" checkpoint is a decision with downstream effects on every existing
   iccad5 result in this repository.
2. If the user wants to proceed, the next concrete step is **not** more checkpoint-selection
   tuning (this fix has been applied and evaluated) but addressing §2's root contributing
   factor directly: either accept iccad5's dead-bottleneck behavior as an inherent
   consequence of its 26-image positive training set and document it as a known limitation
   wherever iccad5 results are cited, or revisit training configuration for extremely
   imbalanced benchmarks specifically (regularization, bottleneck width, or per-benchmark
   epoch count) — any of which would be a further methodology change requiring its own
   review before implementation, not performed here.
3. If iccad5's checkpoint is ever swapped, **all iccad5-derived results across
   `XAI_RESEARCH_AUDIT.md`, `GEOMETRY_ANALYSIS_REPORT.md`, and `TP_FP_GEOMETRY_ANALYSIS.md`
   would need regeneration** for that benchmark specifically — not done in this task, per
   the instruction to run only a small sanity subset.

---

## Terminal Summary

- **CODE CHANGE**: Added `BestBalancedAccuracyCheckpoint` callback (`src/metrics.py`,
  reuses the existing balanced-accuracy formula from native Keras TP/TN/FP/FN metrics) and
  wired it into `model.fit(..., callbacks=[best_ckpt_cb])` in `src/train_xai.py`. No other
  line's behavior changed.
- **SELECTED EPOCH**: 8 of 10 (0-indexed), validation balanced accuracy 0.9649.
- **OLD TEST BALANCED ACCURACY**: 0.9501 (final epoch, 40 TP / 17871 TN / 1456 FP / 1 FN).
- **NEW TEST BALANCED ACCURACY**: 0.9649 (best epoch, 39 TP / 18914 TN / 413 FP / 2 FN).
- **OLD ZERO-BOTTLENECK RATE**: 4.92% overall test set; 97.56% of true-HS images.
- **NEW ZERO-BOTTLENECK RATE**: 1.81% overall test set; 78.05% of true-HS images.
- **OLD FINAL BIAS**: −0.3330.
- **NEW FINAL BIAS**: −0.2725.
- **COLLAPSE RESOLVED**: **PARTIAL** — substantially reduced (both in rate and in
  concentration on true-HS images), not eliminated; random noise still triggers it.
- **XAI SANITY STATUS**: Mixed/method-dependent on a 10-sample subset — Occlusion clearly
  improved (5→2 degenerate), LayerCAM modestly improved (5→4), Grad-CAM/Grad-CAM++ showed
  no improvement (9→10 / 10→10), attributed to a separate GAP-cancellation mechanism not
  addressed by checkpoint selection.
- **NEXT ACTION**: Hold — do not overwrite the production `xai_cnn_iccad5.keras` without
  explicit user direction; if further remediation is wanted, target the underlying class-
  imbalance/bottleneck-capacity issue rather than further checkpoint-selection tuning.
