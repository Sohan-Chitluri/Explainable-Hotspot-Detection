# ICCAD5 Forensic Audit

Forensic investigation of the input-invariance finding reported in
`docs/TP_FP_GEOMETRY_ANALYSIS.md` §11a: the `xai_cnn_iccad5.keras` checkpoint appeared to
produce identical output for visually distinct inputs. No retraining, architecture change,
or XAI-implementation change was made — this is a read-only diagnostic pass. All new
evidence is in `results/xai/iccad5_forensics/`.

**Bottom line, stated up front**: this is a **genuine trained-model degeneracy** (failure
mode **A**, with a concrete contributing methodological gap — see §12), not a
preprocessing bug, not a checkpoint-loading bug, and not an evaluation/metrics bug. A
16-unit ReLU bottleneck (`dense_penultimate`) is **completely dead** (all 16 units output
exactly zero) for 97.6% of iccad5's true-HS test images and 4.7% of its true-NHS images;
when dead, the classifier's output is mathematically just the output bias, independent of
the input. This is not random — it correlates strongly with the true label — but within
the dead region the network provides no graded, input-dependent signal at all, which is
exactly why gradient- and perturbation-based XAI methods correctly report "nothing to
attribute" for those samples.

## 1. Checkpoint Forensics

All 5 checkpoints loaded independently and compared
(`results/xai/iccad5_forensics/checkpoint_comparison.csv`,
`layer_architecture_comparison.csv`):

- **Architecture**: identical layer names, layer classes, and output shapes across all 5
  checkpoints (21 layers, 128,321 parameters each) — confirmed programmatically, not
  assumed from filenames.
- **NaN/Inf**: zero NaN and zero Inf values in any weight tensor, any benchmark.
- **Weight scale**: iccad5's overall weight std (0.081) is unremarkable, within the range
  of the other 4 checkpoints (0.046–0.137) — no evidence of exploded or collapsed weights
  globally.
- **Final-layer (`dense_output`) bias is the one clear outlier**: iccad5 = **−0.333**,
  vs. iccad1 −0.004, iccad2 +0.017, iccad3 +0.021, iccad4 +0.019 — roughly 15–90× larger
  in magnitude and the only negative-and-large value. This single number, as shown below,
  turns out to be exactly the constant output produced whenever the network collapses.

**Conclusion**: rules out failure mode **B** (checkpoint/model-loading problem) — the
checkpoint loads correctly, has the expected architecture, and has no corrupted values.

## 2. Input / Preprocessing Audit

Traced `XAIEngine.preprocess_image` (identical code path used throughout the XAI,
Occlusion, and geometry pipelines) for 10 iccad5 samples plus 2 additional real images: PIL
load → RGB convert → resize to 224×224 (bicubic default) → `/255.0` → float32 tensor.
Confirmed via direct inspection (`results/xai/iccad5_forensics/activation_dependence_iccad5.csv`):

- Raw pixel distributions differ meaningfully between images (mean 0.169–0.225, as
  expected for different binary layouts of different geometry density).
- `conv_early_2` and `conv_mid_2` activations show real, non-zero, sample-varying std
  (e.g. `conv_early_2_std` ranges 0.135–0.290 across the 10 samples; `conv_mid_2_std`
  ranges 1.07–1.46) — **the model is genuinely processing distinct input content through
  the early/mid convolutional stack.** This directly rules out failure mode **C**
  (preprocessing collapsing all inputs to the same tensor) — inputs reach the model
  distinctly and stay distinct through two full conv blocks.

## 3. Intermediate Activation Test — Where Input Dependence Disappears

10 visually distinct iccad5 samples (3 TP, 3 TN, 3 FP, 1 FN — the same 10 representative
samples used throughout this project) probed layer-by-layer
(`results/xai/iccad5_forensics/activation_dependence_iccad5.csv`,
`penultimate_pairwise_distance.csv`). Summary of the dependency chain:

```
Input                    -> differs (confirmed, §2)
conv_early_2 (224x224x16) -> differs (std 0.135-0.290, sample-dependent)
conv_mid_2   (56x56x24)   -> differs (std 1.07-1.46, sample-dependent)
conv_final_2 (28x28x32)   -> differs (std 1.37-2.70, sample-dependent)
dense_penultimate (16,)   -> COLLAPSES for a subset of samples: exactly identical
                             (pairwise L2 distance = 0.000) across 5 of the 10 samples
dense_output (logit/prob) -> IDENTICAL for the collapsed subset (logit = -0.332967,
                             exactly equal to the dense_output bias, to 6 decimal places)
```

**Exact collapse point: `dense_penultimate`**, a `Dense(16, activation="relu")` layer
immediately after `Flatten` + `Dropout(0.3)` (`src/model_xai.py`). For 5 of the 10 probed
samples (`HSCAD50`, `HSCAD51`, `HSCAD510` — all TP — and `NHSCAD50_8`, `NHSCAD51_1` — both
FP), **all 16 pre-ReLU pre-activations are negative** (`dense_penult_pre_relu_max` ranges
−3.06 to −5.56, all < 0), so the post-ReLU output is the exact zero vector for every one of
these 5 samples (`dense_penult_post_relu_sum = 0.000000`, pairwise L2 distance exactly
0.000 between all 5). Since `dense_output`'s logit is `zero_vector · W + b = b`, the logit
is **exactly the bias** (−0.332967, matching §1's checkpoint-level finding to 6 decimal
places) for every sample in this state, regardless of what the input was.

A 6th sample, `NNHSCAD5994_4` (FP), sits right at the edge of this dead region — 1 of 16
units barely positive (post-ReLU sum 1.108, a small but nonzero value) — producing a
distinct, non-degenerate logit (−0.082, p_hs=0.520). This explains why this was the *one*
FP sample not flagged as degenerate in the prior report.

The remaining 4 probed samples (3 TN, 1 FN) are **not** in this same collapsed state:
8 of 16 `dense_penultimate` units are alive, and their post-ReLU vectors are genuinely
different from each other (pairwise L2 distances 8.7–14.7, not zero) — but their logits
are very large (+15.1 to +22.5), which is a **separate phenomenon**: extreme confidence
saturation on a partially-alive (not fully dead) representation, not full collapse. This
matches the milder, partially-dead mechanism found for one iccad3 TN sample in
`TP_FP_GEOMETRY_ANALYSIS.md` §11b.

## 4. Final-Layer Audit

`dense_output` (`Dense(1, activation="sigmoid")`) itself is unremarkable: its kernel
values and bias are finite, non-extreme numbers with no NaN/Inf (§1). **The final layer is
not the source of the problem** — it is correctly and deterministically computing
`sigmoid(penultimate · W + b)`; when `penultimate` is the exact zero vector, the output is
correctly and unsurprisingly `sigmoid(b)` — this is expected linear-algebra behavior given
a dead upstream layer, not a bug in the final layer itself. **The final output is
mathematically constant precisely because its incoming representation is constant
(exactly zero)** for the affected samples — confirmed exactly as instructed in the task
brief.

## 5. Evaluation Pipeline Audit

Traced `src/train_xai.py::evaluate_xai_model` (the code that produced
`results/xai_training/iccad5/metrics.json`): Keras `ImageDataGenerator` → `flow_from_directory`
(test dir, `shuffle=False`) → `model.predict` → threshold at 0.5 → `binary_metrics`. This
pipeline was independently reproduced from scratch (different code path: direct PIL
loading + manual batching, not `ImageDataGenerator`) on the **full** iccad5 test set
(19,368 images, `results/xai/iccad5_forensics/trivial_classifier_comparison.csv`,
`full_test_dead_relu_incidence.csv`):

| Predictor | TP | TN | FP | FN | Balanced Accuracy | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A. Actual model** | 40 | 18084 | 1243 | 1 | **0.9556** | 0.976 | 0.936 |
| **B. Constant "always NHS" (majority class)** | 0 | 19327 | 0 | 41 | **0.5000** | 0.000 | 1.000 |
| **C. Constant "always HS" (collapsed-branch class)** | 41 | 0 | 19327 | 0 | **0.5000** | 1.000 | 0.000 |

**A constant predictor cannot reproduce the reported metrics — balanced accuracy would be
exactly 0.500 for either trivial baseline, vs. the model's actual 0.9556.** The originally
reported `metrics.json` balanced accuracy (0.9501) and this independent reproduction
(0.9556) differ by 0.55 percentage points; the most likely source is a resize-interpolation
difference between Keras's `ImageDataGenerator` (nearest-neighbor by default) and this
audit's PIL-bicubic reproduction (the same interpolation path used by `XAIEngine`
throughout the XAI/Occlusion/geometry pipelines) — a minor, secondary, pre-existing
preprocessing inconsistency (also noted in `XAI_RESEARCH_AUDIT.md` §2), not related to the
collapse mechanism itself, and not large enough to change any conclusion here. **This rules
out failure mode C for the evaluation pipeline as the origin of the collapse** — the
reported metrics are a genuine, reproducible measurement of a genuinely-discriminating
(if partially degenerate) classifier, not an artifact of a broken evaluation script.

## 6. Class Distribution

From `results/xai_training/iccad5/metrics.json::dataset_counts` (train split) and this
audit's direct file count (test split, confirmed identical to the metrics.json values):

| Split | HS | NHS | Total | HS % |
|---|---:|---:|---:|---:|
| Train | 26 | 2,716 | 2,742 | 0.95% |
| Test | 41 | 19,327 | 19,368 | 0.21% |

**iccad5 has, by a wide margin, the most extreme class imbalance and the smallest absolute
positive-class training set of all 5 ICCAD-12 benchmarks used in this project**:

| Benchmark | train HS | train NHS:HS ratio |
|---|---:|---:|
| iccad1 | 99 | 3.4:1 |
| iccad2 | 174 | 30.4:1 |
| iccad3 | 909 | 5.1:1 |
| iccad4 | 95 | 46.9:1 |
| **iccad5** | **26** | **104.5:1** |

A trivial majority-class ("always NHS") classifier on iccad5's test set achieves 99.79%
**raw accuracy** (19,327/19,368 correct) but exactly **0.500 balanced accuracy** — this is
reported per the task instruction, **not** characterized as a bug: this is the well-known
and expected behavior of raw accuracy under severe class imbalance, and it is exactly why
`metrics.json` reports balanced accuracy (0.950) rather than raw accuracy as its headline
number. The imbalance itself is a real property of the ICCAD-12 benchmark 5 dataset as
provided in this repository, not introduced by this project's code.

## 7. Training Artifacts

`src/train_xai.py::train_one_benchmark` (`BENCHMARK_EPOCHS[5] = 10`): plain
`model.fit(..., class_weight=class_weights)` with **no `ModelCheckpoint` callback and no
best-epoch restoration** — `model.save(model_path)` is called once, after `.fit()`
completes, saving whatever state the model is in at the *final* epoch, unconditionally.

`results/xai_training/iccad5/metrics.json::history_val_balanced_accuracy` (visualized in
`results/xai_training/iccad5/balanced_accuracy_vs_epoch.png`, re-examined here):

```
epoch:  0     1     2     3     4     5     6     7     8     9
val:  0.536 0.500 0.747 0.899 0.904 0.923 0.964 0.924 0.783 0.950
```

The validation curve is **highly non-monotonic**, dropping from 0.964 (epoch 6, the best
epoch) to 0.783 (epoch 8) before recovering to 0.950 at the final, *saved* epoch 9 — a
16-point swing in the two epochs immediately preceding the checkpoint that was actually
saved. This instability is consistent with training a narrow 16-unit ReLU bottleneck under
104.5:1 class weighting with only 26 positive training examples — a small positive-class
sample size makes the gradient signal for the rare class inherently noisy from batch to
batch. **This is a concrete, fixable methodological gap**: no incorrect dataset path,
empty training set, wrong labels, accidental early stopping, or misuse of another
benchmark's checkpoint was found (§1 confirms distinct, correctly-shaped, non-corrupted
weights per benchmark; file timestamps are sequential and consistent with one single
training run per benchmark, no evidence of checkpoint reuse) — but the **absence of
best-epoch checkpoint selection**, combined with the observed training instability, is a
plausible and evidence-consistent contributor to why the *specific* epoch that got saved
ended up with a dead bottleneck for many HS-like inputs, even though intermediate epoch 6
(bal. acc 0.964, the best in the run) might not have.

## 8. Independent Reproduction (Synthetic Probes)

A fresh script (not the XAI/Occlusion/geometry pipeline code) loaded the iccad5 checkpoint
directly and probed it with real and synthetic inputs
(`results/xai/iccad5_forensics/` — probe transcript below, not saved as CSV per the task's
instruction that synthetic inputs are diagnostic-only, not research data):

| Input | mean | conv_final_2 std | `dense_penultimate` sum | logit | p(NHS) |
|---|---:|---:|---:|---:|---:|
| all-zero image | 0.000 | 0.798 | 73.208 | +15.80 | 1.0000 |
| all-one image | 1.000 | 1.425 | 77.567 | +19.58 | 1.0000 |
| random uniform noise | 0.500 | 2.897 | **0.000** | **−0.333** | 0.4175 |
| random binary noise | 0.299 | 4.830 | **0.000** | **−0.333** | 0.4175 |
| real `HSCAD50` (TP) | 0.225 | 2.567 | **0.000** | **−0.333** | 0.4175 |
| inverted `HSCAD50` | 0.775 | 2.586 | **0.000** | **−0.333** | 0.4175 |
| real `NHSCAD510_1` (TN) | 0.169 | 2.118 | 92.060 | +22.46 | 1.0000 |
| inverted `NHSCAD510_1` | 0.831 | 2.139 | 53.135 | +14.85 | 1.0000 |

**Random noise (both uniform and binary) collapses to the exact same dead state as real
hotspot images** (`penultimate sum = 0.000`, logit = −0.333 to 6 decimal places) — this
confirms the dead region of `dense_penultimate`'s input space is **broad**, not narrowly
tailored to genuine hotspot features; a healthy, well-trained bottleneck would not produce
an identical, zero-information output for structured hotspot layouts *and* unstructured
random noise. Conversely, the extreme all-zero/all-one images land in the *other* (alive,
saturated-high) branch. Combined with §6's full-test-set finding that this same dead
branch is hit by 97.6% of true-HS images vs. only 4.7% of true-NHS images, the picture is:
**the dead branch functions as an emergent, degenerate "predict HS" decision rule that is
statistically correlated with the true label on real data, but is not discriminating on
genuine learned hotspot semantics** — it is also triggered by inputs (random noise) that
share no such semantics, meaning its correlation with the true label on real images likely
rides on some coarse, non-specific statistic (e.g. overall geometry density/global
brightness) rather than a meaningful spatial pattern.

## 9. Exact Failure Mode

**Classification: (E) Multiple interacting issues**, with the dominant one being
**(A) a genuinely collapsed representation** in the trained checkpoint:

- **Not (B)** checkpoint/model-loading: architecture, weights, and file integrity are all
  confirmed correct (§1).
- **Not (C)** preprocessing: inputs are confirmed to reach the model distinctly and to
  produce genuinely input-dependent activations through three full convolutional stages
  (§2, §3).
- **Not (D)** dataset/label issue in the sense of mislabeled or corrupted data — the class
  imbalance is real and severe (§6) and is a substantial **contributing factor**, but it is
  a property of the provided benchmark, not a code defect.
- **Not** a pure evaluation/metrics bug (§5) — the reported numbers are reproducible and
  meaningfully different from any trivial constant baseline.
- **Is (A)**: `dense_penultimate`, a 16-unit ReLU bottleneck, is completely dead (exact
  zero output) for a label-correlated but non-specific subset of inputs, confirmed by
  three independent probes (gradient, occlusion, and this direct activation trace).
- **Contributing, fixable methodological gap identified (§7)**: no best-epoch checkpoint
  selection during a training run with a highly unstable validation curve on an extremely
  imbalanced (104.5:1), tiny (26-image) positive training class — a plausible, evidence-
  consistent explanation for why this particular dead-bottleneck configuration is the one
  that got saved.

## 10. Impact on Current XAI Research

**Which conclusions are affected:**

- `XAI_RESEARCH_AUDIT.md` §4.1/§9's "Overall Pearson/cosine" table: **not invalidated**,
  but its Grad-CAM/Grad-CAM++ near-zero medians are now understood mechanistically — they
  reflect this collapse (and the related partial-saturation phenomenon, §3) rather than
  "weak but real" agreement. No numbers need correction; the interpretation is sharpened
  (already updated in `TP_FP_GEOMETRY_ANALYSIS.md` Part F).
- `GEOMETRY_ANALYSIS_REPORT.md` §9's iccad5 anomaly and the pooled Grad-CAM statistics:
  **should be read with this checkpoint-specific caveat** — iccad5's Grad-CAM/Grad-CAM++
  geometry correspondence numbers are disproportionately built from degenerate,
  arbitrary-region attributions (`TP_FP_GEOMETRY_ANALYSIS.md` §9 already flags and
  quantifies this: 31.6%/42.1% of all 57 samples degenerate for these two methods,
  concentrated in iccad5). **Not silently altered here** — the original CSVs and report
  are left as published; this audit adds the mechanistic explanation and recommends a
  flagged re-run (already recommended in the prior report's §14).
- `README.md`'s reported iccad5 balanced accuracy (95.01%): **not shown to be wrong**, but
  now understood to be achieved partly through a degenerate branch that correctly predicts
  HS for most true-HS images via a non-discriminative fallback rather than a graded,
  interpretable decision — worth a caveat wherever this number is cited as evidence of
  "understanding," though it is not evidence of a miscalculated metric (§5 rules that out).
- **No conclusion in any prior document is retracted.** The affected documents' iccad5-
  specific Grad-CAM/Grad-CAM++ findings should be treated as **checkpoint-specific and
  provisional** rather than representative of the method in general — consistent with what
  those documents already say (none of them generalized the iccad5 finding to the other
  4 benchmarks).

**Recommended treatment of iccad5 going forward**: **retain, but flag separately** — do not
exclude iccad5 from the project, and do not silently fold its numbers into cross-benchmark
averages without the degenerate-map caveat already present in
`TP_FP_GEOMETRY_ANALYSIS.md`. It should **not** be "regenerated" (i.e., XAI maps
re-rendered) until the checkpoint itself is addressed (§12) — regenerating explanations
for a checkpoint with a confirmed dead bottleneck would not fix the underlying
non-explainability of the collapsed branch.

## 11. Diagnostic Artifacts

All saved under `results/xai/iccad5_forensics/`:

- `checkpoint_comparison.csv` — per-benchmark weight/architecture statistics (§1)
- `layer_architecture_comparison.csv` — full layer-by-layer shape comparison (§1)
- `activation_dependence_iccad5.csv` — per-sample layer activation statistics, 10 samples (§3)
- `penultimate_pairwise_distance.csv` — 10×10 pairwise L2 distance matrix at the collapse layer (§3)
- `trivial_classifier_comparison.csv` — actual vs. constant-predictor metrics, full 19,368-image test set (§5)
- `full_test_dead_relu_incidence.csv` — per-image dead/alive flag and probability, full test set (§6)
- `dead_relu_and_trivial_baseline.png` — summary plot (dead-ReLU rate by label; model vs. trivial baselines)

## 12. Recommended Remediation (not performed — no retraining done in this audit)

1. **Add best-epoch checkpoint selection** (`tf.keras.callbacks.ModelCheckpoint(...,
   monitor="val_..._balanced_accuracy" or equivalent, save_best_only=True)`) to
   `src/train_xai.py` before any future retraining — directly addresses the gap in §7.
   This is a training **script** change, not an architecture change.
2. **Do not retrain iccad5 until** (1) is in place — retraining with the current
   final-epoch-only save logic risks reproducing the same or a different unstable/collapsed
   outcome.
3. Consider, for future work only: a wider or differently-regularized bottleneck, or
   dropout-rate/learning-rate adjustments specifically for benchmarks with severe positive-
   class scarcity (iccad5, and to a lesser extent iccad4) — offered as a direction, not a
   prescription, since diagnosing the *optimal* fix is outside this audit's scope.
4. Until remediated, treat iccad5's Grad-CAM/Grad-CAM++ attribution maps as unreliable for
   the ~5% of its test set (and specifically ~97.6% of its true-HS images) that fall in the
   dead branch; LayerCAM and Occlusion are less affected but not unaffected (occlusion is
   *exactly* as affected, since it is a forward-pass probe of the same collapsed decision
   surface — confirmed identical degenerate sample set in
   `TP_FP_GEOMETRY_ANALYSIS.md` §9).

---

## Terminal Summary

- **FAILURE MODE**: (E) Multiple interacting issues — dominant mechanism is (A) genuine
  trained-checkpoint collapse; contributing factor is a training-methodology gap (no
  best-epoch checkpoint selection), interacting with (D)-adjacent severe, real class
  imbalance (not a labeling bug). (B) checkpoint-loading, (C) preprocessing, and pure (C)
  evaluation/metric bugs are all ruled out with direct evidence.
- **WHERE INPUT DEPENDENCE DISAPPEARS**: exactly at `dense_penultimate`
  (`Dense(16, activation="relu")`), the layer immediately after `Flatten`+`Dropout`. Input
  → conv_early_2 → conv_mid_2 → conv_final_2 all show genuine, sample-varying activity;
  `dense_penultimate` is the exact-zero vector (all 16 ReLU units dead) for the affected
  samples, after which `dense_output` is mathematically just the bias term.
- **CHECKPOINT STATUS**: structurally valid (architecture matches all other benchmarks
  exactly, no NaN/Inf, weights in a normal range) but functionally degenerate for a subset
  of its input space; final-layer bias (−0.333) is a clear, confirmed outlier vs. the other
  4 checkpoints (−0.004 to +0.021).
- **EVALUATION STATUS**: reproducible and correct — an independent, from-scratch
  reproduction of the full 19,368-image test set matches the original reported metrics to
  within a small (0.55-point) preprocessing-interpolation discrepancy, and both are far
  from what any trivial constant predictor achieves.
- **ICCAD5 CLASS DISTRIBUTION**: train 26 HS / 2,716 NHS (104.5:1, by far the most
  imbalanced of the 5 benchmarks); test 41 HS / 19,327 NHS (0.21% positive).
- **TRIVIAL BASELINE PERFORMANCE**: both "always NHS" and "always HS" constant predictors
  achieve exactly 0.500 balanced accuracy; the actual model achieves 0.956 — the model is
  confirmed to be doing real, non-trivial, label-correlated work, not acting as a disguised
  constant classifier.
- **IMPACT ON CURRENT RESULTS**: no existing number is shown to be wrong; iccad5's
  Grad-CAM/Grad-CAM++ geometry-correspondence and correlation statistics (in
  `GEOMETRY_ANALYSIS_REPORT.md` and `XAI_RESEARCH_AUDIT.md`) should be read as
  checkpoint-specific and provisional, per the caveats already present in
  `TP_FP_GEOMETRY_ANALYSIS.md` §9/§11. Nothing has been silently altered.
- **REQUIRED FIX**: add best-epoch checkpoint selection to `src/train_xai.py` before any
  future retraining of iccad5 (or any other unstable-training benchmark).
- **WHETHER RETRAINING IS JUSTIFIED**: not yet — retraining without first fixing the
  missing best-epoch selection would not reliably resolve the collapse and could reproduce
  it. Per the task constraint, no retraining was performed in this audit.
