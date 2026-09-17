# ICCAD5 Class-Imbalance Ablation

A controlled ablation isolating whether ICCAD5's training-set class imbalance (26 HS vs.
2,716 NHS — 104.5:1) contributes to the `dense_penultimate` bottleneck collapse
documented in `docs/ICCAD5_FORENSIC_AUDIT.md` and `docs/ICCAD5_BOTTLENECK_ANALYSIS.md`.
No architecture, dataset, split, or preprocessing change; no production checkpoint
touched; no existing XAI/occlusion/geometry result modified.

**Headline result**: the ablation found a clear, large, and unexpected effect — but it
implicates **loss reweighting**, not imbalance *per se*. Two of the three conditions
(no imbalance handling at all, and balanced sampling) show **zero** exact-zero-vector
samples across the full 19,368-image test set. Only the condition matching the existing
production recipe (`class_weight`) shows the collapse (85.4% of HS test images). Balanced
sampling additionally produced four units with strong, consistent, **HS-elevated**
activation (AUC 0.97–0.99) — the first evidence in this entire investigation of anything
resembling class-selective (not just class-weighted-collapsed) structure in this
bottleneck.

## 1. Objective

Determine whether ICCAD5's extreme class imbalance is contributing to the learned
`dense_penultimate` bottleneck by testing whether three ways of handling that imbalance
change: classification performance, HS sensitivity, false-positive behavior, the
penultimate-layer representation, the number of dead units, the all-zero bottleneck rate,
and HS/NHS activation separation.

## 2. Experimental Design

Per the task's own instructions, a genuine ambiguity in the requested design was resolved
explicitly and is stated here rather than silently assumed: the task described condition
A ("BASELINE — normal training procedure with best-checkpoint selection") and condition B
("CLASS-WEIGHTED — apply class weights, don't change anything else") as distinct, but the
**existing** "normal" production procedure (`src/train_xai.py`) already applies
`class_weight` unconditionally — so a literal reading would make A and B identical. This
was resolved as follows, and is the design actually run:

- **A. BASELINE (`baseline_noweight`)** — best-checkpoint selection, **no** class
  weighting (`class_weight=None`), standard (natural class-distribution) batches. This is
  the true "imbalance handling off" control, newly trained for this ablation.
- **B. CLASS-WEIGHTED (`class_weighted`)** — **not retrained**. This is exactly the
  existing `models/xai/retrained/xai_cnn_iccad5_best.keras` from
  `docs/ICCAD5_CHECKPOINT_FIX.md`: same architecture, same seed, same best-checkpoint
  callback, `class_weight={0: 52.73, 1: 0.505}` applied. Reused per the instruction to
  reuse existing utilities/results rather than duplicate work — this checkpoint already
  *is* the literal "normal training procedure" as it currently exists in the repository.
- **C. BALANCED SAMPLING (`balanced_sampling`)** — best-checkpoint selection, **no** class
  weighting, batches drawn 50/50 HS/NHS via a custom sampler (§3). Newly trained.
- **D. COMBINED — omitted.** Per the task's own get-out clause: stacking class weighting
  and balanced sampling would make it impossible to attribute any observed effect to one
  mechanism or the other, defeating the purpose of a controlled ablation. Given that A and
  C alone already show a clean, large, consistent effect in the *same* direction (both
  eliminate the collapse), and B (the currently-weighted production recipe) is the one
  outlier, a combined condition was judged low-value for isolating mechanism and is left
  as a natural follow-up if the community wants a production-oriented "best of both"
  configuration rather than a mechanism-isolating one.

Held fixed across all three conditions: architecture (`build_xai_cnn`, 21 layers, 128,321
params), optimizer (Nadam), loss (binary cross-entropy), learning rate (framework default,
unspecified in the original code and left unspecified here), batch size (32), epochs (10),
train/val/test split (`iccad-official/iccad5`, unmodified), preprocessing (`rescale=1/255`,
224×224, `load_img` default `interpolation='nearest'` — see §11 for a preprocessing
consistency issue found and fixed *in this analysis's own extraction script*, not in
training), seed (42, `tf.keras.utils.set_random_seed`), and checkpoint-selection criterion
(`BestBalancedAccuracyCheckpoint`, reused unchanged from `src/metrics.py`).

## 3. Exact Training Differences

- **Class weighting** (`src/train_xai.py::compute_class_weights`, unchanged, reused):
  `w_c = N / (2 * N_c)`. For iccad5: `w_HS = 2742/(2*26) = 52.73`, `w_NHS =
  2742/(2*2716) = 0.505`. Applied via `model.fit(..., class_weight=...)`. Condition B
  only.
- **Balanced sampling** (new, `src/run_iccad5_imbalance_ablation.py::balanced_batch_generator`):
  each batch is built from `batch_size//2 = 16` HS images and 16 NHS images, **both
  drawn with replacement** via `numpy.random.default_rng(seed=42)`. HS pool has only 26
  images — with-replacement sampling is required (documented, not incidental). NHS pool
  (2,716 images) is also sampled with replacement for a uniform, simple procedure across
  both classes, rather than exhausting a shuffled NHS pool at a different rate than HS.
  Wrapped in `tf.data.Dataset.from_generator` and fed directly to `model.fit`. No
  `class_weight` is applied (uniform 1:1 loss weighting) — this isolates the sampling
  variable specifically.
- **Steps per epoch**: fixed at 86 (`ceil(2742/32)`) for **all three** conditions,
  identical to the standard `flow_from_directory` step count — the same number of
  gradient updates per epoch in every condition. **Documented, explicit consequence of
  balanced sampling** (per the task's own instruction): with 16 HS draws/batch × 86
  batches/epoch = 1,376 HS draws per epoch against a pool of 26 images, each HS image is
  seen **~53×/epoch** on average (with replacement); each NHS image, at 16 draws/batch ×
  86 batches = 1,376 draws against a pool of 2,716, is seen **~0.5×/epoch** on average —
  i.e. roughly half the NHS training pool is not seen in any given epoch. This is the
  defining trade-off of balanced sampling and is stated explicitly rather than left
  implicit.
- **Validation/test data**: identical `ImageDataGenerator.flow_from_directory` on the
  unmodified test split in all three conditions — never resampled, never reweighted.

## 4. Dataset Counts

Unchanged from every prior ICCAD5 document: train 26 HS / 2,716 NHS (104.5:1); test 41 HS
/ 19,327 NHS (471:1). Every table below reports HS and NHS counts explicitly.

## 5. Classification Results

`results/xai/iccad5_imbalance_ablation/classification_comparison.csv`,
`classification_comparison.png`, `confusion_matrix_<condition>.png`:

| Condition | Balanced Acc. | HS Recall | NHS Specificity | Precision | F1 | TP | TN | FP | FN | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_noweight | **0.9741** | 0.9512 | 0.9969 | 0.3939 | 0.5571 | 39 | 19267 | 60 | 2 | 7 |
| class_weighted | 0.9649 | 0.9512 | 0.9786 | 0.0863 | 0.1582 | 39 | 18914 | 413 | 2 | 8 |
| balanced_sampling | 0.9501 | 0.9024 | 0.9978 | 0.4684 | 0.6167 | 37 | 19285 | 42 | 4 | 8 |

No condition is labeled "best." Read plainly: `baseline_noweight` achieves the highest
balanced accuracy and matches `class_weighted`'s HS recall exactly (39/41) while cutting
false positives by 85.5% (413→60); `balanced_sampling` achieves the fewest false positives
of all three (42) and the highest F1 (0.617) at the cost of 2 additional missed hotspots
relative to the other two. Balanced accuracy is reported as the headline metric
throughout, per instruction — not plain accuracy.

## 6. Bottleneck Results

`results/xai/iccad5_imbalance_ablation/zero_vector_comparison.csv`,
`zero_vector_rate_by_case_and_condition.png`:

| Condition | Zero-vector rate (overall) | Zero-vector rate (HS) | Zero-vector rate (NHS) | Globally dead units |
|---|---:|---:|---:|---:|
| baseline_noweight | **0.000%** | **0.000%** | **0.000%** | 3 / 16 |
| class_weighted | 1.792% | **85.366%** | 1.614% | 6 / 16 |
| balanced_sampling | **0.000%** | **0.000%** | **0.000%** | 3 / 16 |

By case type (`zero_vector_comparison.csv`, `grouping=case_type`): `class_weighted`'s
collapse is concentrated in TP (89.7% zero-vector, 35/39) and FP (75.5%, 312/413) —
exactly the two HS-predicting outcomes, consistent with the mechanism already established
in `ICCAD5_BOTTLENECK_ANALYSIS.md` §6 (TN/FN are structurally always non-zero). **Both
`baseline_noweight` and `balanced_sampling` show 0.000% zero-vector rate in every single
case-type stratum** (TP, TN, FP, FN alike) — the collapse is not merely reduced in these
two conditions, it is **entirely absent** at the level of this specific pathology (though
each condition still has 3 permanently-dead units out of 16, §7 — a narrower but
non-degenerate bottleneck, not full restoration of all 16 units).

## 7. Per-Unit Analysis

`results/xai/iccad5_imbalance_ablation/per_unit_comparison.csv`,
`<condition>_unit_activation_distributions.png`. Units with `auc_HS_vs_NHS` outside
[0.10, 0.90] (a large, non-arbitrary separation by conventional Mann-Whitney-AUC
thresholds):

| Condition | Unit | Direction | Cohen's d | AUC (HS vs NHS) |
|---|---|---|---:|---:|
| baseline_noweight | 0 | **HS-elevated** | +4.83 | **0.982** |
| baseline_noweight | 2, 4, 9, 11, 14 | NHS-elevated | −2.26 to −4.25 | 0.018–0.026 |
| class_weighted | (none HS-elevated) | — | — | — |
| class_weighted | 8, 12 | NHS-elevated | −1.76, −2.67 | 0.025, 0.046 |
| balanced_sampling | **0, 1, 6, 15** | **HS-elevated** | **+5.00 to +6.39** | **0.979–0.987** |
| balanced_sampling | 2, 3, 7, 9, 13, 14 | NHS-elevated | −1.52 to −3.71 | 0.021–0.060 |

**This is the most novel finding of the ablation.** Every prior ICCAD5 document in this
project (`ICCAD5_FORENSIC_AUDIT.md`, `ICCAD5_BOTTLENECK_ANALYSIS.md`) found live units
were exclusively NHS-elevated — no evidence of any HS-associated unit anywhere, under
either the OLD or the class-weighted NEW checkpoint. **`baseline_noweight` produces one,
and `balanced_sampling` produces four**, strongly and consistently higher-activating
units for true-HS samples (AUC approaching 1.0, large effect sizes). Per the task's
explicit instruction, these are described as showing **"HS-associated activation"** — not
"HS detectors" and not evidence of learned physical hotspot semantics; the finding is a
statistical association in a 16-dimensional representation, evaluated against only 41 HS
test samples (§12).

No unit in any condition is exclusively active for one class with zero overlap — all
reported separations are strong statistical tendencies (large effect sizes, high AUC), not
perfect binary detectors.

## 8. Representation Comparison

`results/xai/iccad5_imbalance_ablation/representation_comparison.csv`:

| Condition | Centroid dist. (HS↔NHS) | Within-HS spread | Within-NHS spread | Centroid/within-NHS ratio | Mean penult. norm (HS) | Mean active units (HS) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_noweight | **73.79** | 10.43 | 15.36 | **4.81** | 92.56 | 1.71 |
| class_weighted | 27.27 | 1.93 | 13.55 | 2.01 | 1.10 | 0.24 |
| balanced_sampling | 31.72 | 6.32 | 6.48 | **4.89** | 33.84 | 8.29 |

`baseline_noweight` and `balanced_sampling` both show roughly **2.4× the relative class
separation** of `class_weighted` (centroid-distance-to-within-NHS-spread ratio ~4.8–4.9
vs. 2.0). `baseline_noweight`'s HS-centroid mean penultimate norm (92.6) is dramatically
higher than `class_weighted`'s (1.1, consistent with §6's near-total collapse to the
zero vector) and higher than `balanced_sampling`'s (33.8). **Classification did not
merely improve alongside these representation changes — the representation itself is
measurably different**: this conclusion is drawn from the activation statistics
(centroid distance, per-unit AUC, zero-vector rate) directly, not inferred from the
balanced-accuracy numbers in §5, per the task's explicit requirement.

## 9. PCA Analysis

`baseline_noweight_penultimate_pca.png`, `class_weighted_penultimate_pca.png`,
`balanced_sampling_penultimate_pca.png` — all 41 HS points plus a documented 1,000-point
random NHS subsample (seed=0), same method as `ICCAD5_BOTTLENECK_ANALYSIS.md` §9.

**Qualitatively distinct pattern from every previous ICCAD5 PCA plot in this project**:
under `baseline_noweight` and `balanced_sampling`, the 41 HS points form a visibly
**dispersed, non-degenerate cluster** — spread across a real range of PC1/PC2 values,
clearly separated from the NHS cloud but not collapsed onto a single overlapping point.
Under `class_weighted`, the familiar pattern from `old/new_penultimate_pca.png` recurs:
most HS points overlap at a single location (the zero vector) with 1–2 outliers scattered
elsewhere. This is descriptive only (n=41 HS, no separability test performed, per
instruction) but visually corroborates §6–§8's quantitative findings without requiring
any inferential claim.

## 10. Limitations

- HS test n=41 (as few as 2 in some FN strata) — every HS-specific number here is
  high-variance; effect sizes (Cohen's d, AUC) are emphasized over p-values throughout,
  and no claim in this report should be read as a precise population estimate.
- Only one training run per condition (same seed value, non-bit-exact reproducibility
  across runs on this hardware, already noted in `ICCAD5_CHECKPOINT_FIX.md` §6) — the
  specific selected epochs and exact metrics are one realization of a stochastic process,
  not a guaranteed outcome of re-running this procedure.
- Condition D (combined) was not run — §2 explains the reasoning; this means the ablation
  cannot speak to whether combining balanced sampling with a *reduced* (not eliminated)
  class weight might retain both the HS-associated units and stronger HS recall.
- The full 57-sample XAI pipeline was **not** rerun for any of these three checkpoints —
  per instruction, this stage answers only whether the representation changes; a small
  sanity check was also not run (deemed unnecessary given the very large,
  unambiguous representation-level effect already found, and per the instruction that a
  sanity check is optional "if useful" — the representation evidence here is sufficiently
  direct that an additional qualitative XAI check would not change the conclusion).
- Learning rate was left at the framework default in both the original code and this
  ablation (never explicitly set) — flagged for completeness, not because any evidence
  suggests it varied between conditions (it did not; identical `OPTIMIZER = "nadam"`
  string is compiled in all three).

## 11. A Preprocessing Consistency Issue Found and Fixed (this analysis's own code)

While cross-checking this ablation's per-sample zero-vector/case-type counts against the
official `classification_comparison.csv` numbers, a discrepancy was found: the first
version of the activation-extraction script (`ablation_extract.py`) used
`PIL.Image.resize` (bicubic default, the same convention used throughout the prior
`XAIEngine`-based analyses in this project) rather than `tensorflow.keras.preprocessing
.image.load_img` (nearest-neighbor default, what the **official training/evaluation**
pipeline actually uses via `ImageDataGenerator`). For most conditions this produced only
a 1-sample discrepancy, but for `balanced_sampling` it flipped 7 of 41 HS predictions
(TP 30 vs. official 37) — large enough to matter for a report whose central claims are
about per-case-type zero-vector rates. **This was caught, the extraction script was
corrected to use `load_img` (matching the official pipeline exactly), and every number in
§5–§9 above is from the corrected re-run** — confirmed to match the official
`classification_comparison.csv` TP/TN/FP/FN counts exactly for all three conditions
(§6). The core finding (zero-vector rate = 0% for baseline_noweight and balanced_sampling,
>75% for class_weighted in the HS-predicting strata) was **unchanged in direction and
magnitude** by this correction — reported as a transparency note per this session's
established practice of surfacing analysis-script bugs it finds in its own code, not as a
finding that alters any conclusion.

## 12. Interpretation

The evidence points toward **class-weighted loss reweighting**, not class imbalance in
general, as the mechanism most directly associated with the dense_penultimate collapse.
Both alternatives tested here — doing nothing about the imbalance, and addressing it
through data resampling instead of loss reweighting — produced a bottleneck with zero
exact-zero-vector samples and, in balanced sampling's case, multiple units with strong
HS-associated activation not seen anywhere else in this project's ICCAD5 investigation.
A plausible (not directly tested) mechanism: `class_weight={0: 52.73, ...}` multiplies
the loss gradient for each of the 26 HS training images by ~53×, concentrating an unusually
large gradient signal on very few examples passing through a narrow 16-unit ReLU
bottleneck — a configuration known in general to be conducive to dead-ReLU dynamics.
Balanced sampling achieves increased HS exposure through repetition instead, without
inflating the per-example gradient magnitude, which may avoid this dynamic. This is
offered as the most evidence-consistent explanation available from this ablation, not a
proven causal mechanism — establishing it directly would require inspecting
gradient magnitudes during training, which this analysis did not do.

## 13. PCA / Representation — What This Does and Does Not Support

Supports: the three conditions produce measurably different, non-equivalent
`dense_penultimate` representations, evidenced at the activation level (not merely
inferred from classification metrics). Does not support: that any specific unit is a
"hotspot detector," that either non-weighted condition has "solved" ICCAD5 (3 units are
still permanently dead in both; balanced accuracy is not higher than the current
production checkpoint's in one of the two cases), or that these findings generalize to
other benchmarks (this ablation covers ICCAD5 only, matching the task's own scope
restriction).

## 14. Recommendation for the Next XAI Stage

**Q6 requires selecting a condition to carry forward without ranking.** Based on the
predefined research criterion in this task (does the representation preserve
input-dependent, class-associated information, evidenced at the activation level) rather
than on which condition has the highest balanced accuracy: **`balanced_sampling` is
selected for continuation**, because it is the only condition combining (a) zero
exact-zero-vector collapse and (b) multiple units with strong, consistent HS-associated
activation (§7) — the two properties most directly relevant to whether a downstream XAI
method would have genuine, non-degenerate, class-relevant signal to explain.
`baseline_noweight` is a close second by the same criteria (also zero collapse, one
HS-associated unit, and in fact the highest balanced accuracy of the three) and is
recorded as the natural alternative if classification performance is weighted more
heavily in a future decision. `class_weighted` (the current production checkpoint) is
**not** selected for the next XAI stage on these grounds, though it is not deleted,
overwritten, or removed from the project.

**Q7 — should the full XAI + occlusion + geometry pipeline be regenerated for ICCAD5?**
**Not yet.** This ablation answers the representation question the task posed, but the
training methodology is not yet "settled" in the sense that would justify a full,
expensive regeneration: three real candidate configurations now exist with materially
different representations and no combined/D condition has been explored, learning-rate
and dropout were never varied, and no independent replicate run has confirmed these
results are stable across seeds. A full pipeline regeneration is recommended only after a
specific configuration is deliberately chosen (not merely "selected for continuation" as
a research default, per §14 above) as the benchmark's production checkpoint.

---

## Files Created

`results/xai/iccad5_imbalance_ablation/`: `classification_comparison.csv`,
`classification_comparison.png`, `confusion_matrix_baseline_noweight.png`,
`confusion_matrix_class_weighted.png`, `confusion_matrix_balanced_sampling.png`,
`sample_activation_comparison.csv`, `per_unit_comparison.csv`,
`zero_vector_comparison.csv`, `representation_comparison.csv`,
`baseline_noweight_penultimate_pca.png`, `class_weighted_penultimate_pca.png`,
`balanced_sampling_penultimate_pca.png`,
`baseline_noweight_unit_activation_distributions.png`,
`class_weighted_unit_activation_distributions.png`,
`balanced_sampling_unit_activation_distributions.png`,
`zero_vector_rate_by_case_and_condition.png`,
`training/baseline_noweight/{metrics.json,confusion_matrix.png}`,
`training/balanced_sampling/{metrics.json,confusion_matrix.png}`.

New checkpoints: `models/xai/experimental/xai_cnn_iccad5_baseline_noweight.keras`,
`models/xai/experimental/xai_cnn_iccad5_balanced_sampling.keras`.

## Final Decision Framework

- **Q1 (does class-imbalance handling reduce the number of dead penultimate units?)**:
  Yes — 6/16 dead under class_weighted vs. 3/16 under both baseline_noweight and
  balanced_sampling.
- **Q2 (does it reduce the all-zero bottleneck rate?)**: Yes, dramatically —
  class_weighted 1.79% overall (85.4% of HS); baseline_noweight and balanced_sampling
  both **0.00%** overall and in every case-type stratum.
- **Q3 (does it specifically reduce the HS zero-vector rate?)**: Yes — 85.4%→0.0% for
  both alternative conditions; this is the largest and most decisive of all measured
  effects.
- **Q4 (does it create stronger HS/NHS representation separation?)**: Yes, on every
  representation-level measure used (§7 per-unit AUC/effect sizes, §8 centroid-distance
  ratios, §9 PCA dispersion) — for both baseline_noweight and balanced_sampling, not
  uniformly for class_weighted.
- **Q5 (does classification improve because of a better representation, or mainly
  because of changed decision behavior?)**: Evidence supports **a genuinely different,
  more input-dependent representation** (§8's activation-level statistics, computed
  independently of the classification metrics), not merely a shifted decision threshold —
  though classification differences (§5) are modest and do not all move in the same
  direction as the representation differences (balanced_sampling has the *lowest*
  balanced accuracy of the three despite the richest per-unit structure).
- **Q6 (which condition is carried forward, without calling it "best")**: `balanced_
  sampling`, selected on the criterion of representation quality (zero collapse + multiple
  HS-associated units) rather than classification metrics; `baseline_noweight` recorded as
  the alternative if classification performance is prioritized instead. `class_weighted`
  (current production) is not selected for continuation on these specific research
  criteria.
- **Q7 (should the full XAI/occlusion/geometry pipeline be regenerated for ICCAD5 now?)**:
  No — recommended only after a specific configuration is deliberately adopted as the
  benchmark's checkpoint, not immediately following this exploratory ablation.
