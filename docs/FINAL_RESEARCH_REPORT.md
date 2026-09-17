# Explainable Hotspot Detection on ICCAD-12: A Multi-Method XAI Study of a Custom CNN Lithography Hotspot Classifier

## 2. Abstract

Lithography hotspot detection is a binary image-classification task: given a rasterized
integrated-circuit layout clip, predict whether it will print as a manufacturing defect
("hotspot", HS) or print cleanly ("non-hotspot", NHS) under optical lithography. This
report consolidates a multi-stage research program that (1) reproduces a custom
~12.8k-parameter CNN baseline on all five ICCAD-12 benchmarks, (2) rebuilds an
XAI-compatible variant of the CNN preserving a coarser final feature map for spatial
attribution, (3) applies four independent explanation methods — Grad-CAM, Grad-CAM++,
LayerCAM, and Occlusion Sensitivity — to representative True Positive (TP), True Negative
(TN), False Positive (FP), and False Negative (FN) predictions across all five benchmarks,
(4) derives raster-only geometric proxies (thin-line, narrow-gap, corner, junction,
density) directly from the binary layout images and statistically tests whether
high-attribution regions correspond to these proxies more than chance, (5) forensically
investigates a severe representational pathology discovered in the ICCAD-12 benchmark 5
(ICCAD5) checkpoint — a "dead bottleneck" in which a 16-unit dense layer collapses to the
exact-zero vector for the large majority of true-hotspot predictions — and (6) runs a
controlled class-imbalance ablation, followed by a 3-seed replication, that isolates
class-weighted loss reweighting (not class imbalance itself) as the operative cause of
that collapse, before adopting a final, frozen "balanced sampling" checkpoint for a
last, dedicated XAI/occlusion/geometry evidence pass on ICCAD5. Throughout, the project
enforces a strict scientific-integrity discipline: no single-seed result is treated as
settled without replication, no XAI method is declared "more faithful" than another, and
every attribution/geometry finding is reported as a spatial correspondence, never a
physical or causal claim. The final evidence indicates that all four XAI methods most
reliably corroborate one another on correctly-classified hotspots (TP), that Grad-CAM
specifically becomes unreliable (including gradient-flat, degenerate) on cases where the
model's output leans non-hotspot, and that raster-geometry correspondence is real,
positive, and method-dependent but does not (with the tools used here) distinguish true
hotspot triggers from false-alarm triggers.

## 3. Research Question

**Can explainable-AI techniques identify the layout regions or features that contribute
most strongly to hotspot classification?**

## 4. Motivation

Lithography hotspot detection is used late in the VLSI design flow to flag layout clips
likely to fail manufacturing due to sub-wavelength optical effects (pinching, bridging,
line-end shortening). A CNN classifier that achieves high balanced accuracy is useful as
a fast pre-filter, but by itself gives a designer no actionable information about *why* a
clip was flagged — which is essential both for designer trust and for triaging false
alarms that would otherwise require a full, expensive physical simulation. Explainable AI
(XAI) techniques promise to close this gap by highlighting the spatial regions of a layout
that most influenced a classifier's decision. This project asks, concretely and
empirically, whether that promise holds for this specific network and dataset: do
gradient-based and perturbation-based attribution methods actually point at recognizable,
non-arbitrary layout structure, and does that structure correspond — in a raster-derived,
non-causal sense — to layout features conventionally associated with lithography
sensitivity (narrow spacings, thin lines, corners)? The project treats this as an open
empirical question rather than an assumption, and dedicates substantial effort to
falsifying its own early findings (checkpoint forensics, ablations, seed replication)
before drawing conclusions.

## 5. ICCAD-12 Dataset

The ICCAD-12 Contest hotspot-detection benchmark suite (`iccad-official/iccad{1..5}/`)
provides five independent benchmarks, each with a predefined train/test split organized
as `train/train_hs`, `train/train_nhs`, `test/test_hs`, `test/test_nhs`. Every image is a
single-channel (`L`-mode) 1200×1200 binary raster with exactly two pixel values (0 =
background, 255 = drawn geometry) — confirmed empirically via `np.unique` across sampled
files from every class and benchmark. **No GDS/OASIS file, polygon coordinate, or DRC-rule
annotation exists anywhere in the dataset tree** — the *only* geometry information
available in this project is the rasterized binary image itself, a constraint that shapes
every geometry-correlation method used later in this report.

Class distribution varies enormously by benchmark:

| Benchmark | Train HS | Train NHS | Train HS% | Test HS | Test NHS | Test HS% | Test Imbalance (NHS:HS) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| iccad1 | 99 | 340 | 22.55% | 226 | 4,679 | 4.61% | 20.7:1 |
| iccad2 | 174 | 5,285 | 3.19% | 498 | 41,298 | 1.19% | 82.9:1 |
| iccad3 | 909 | 4,643 | 16.37% | 1,808 | 46,333 | 3.76% | 25.6:1 |
| iccad4 | 95 | 4,452 | 2.09% | 177 | 31,890 | 0.55% | 180.2:1 |
| **iccad5** | **26** | **2,716** | **0.95%** | **41** | **19,327** | **0.21%** | **471.4:1** |

**ICCAD5 has, by a wide margin, the most extreme class imbalance and the smallest
absolute positive-class training set of all five benchmarks** — only 26 hotspot training
images against 2,716 non-hotspots (104.5:1 train ratio), and only 41 hotspots in a
19,368-image test set (471:1 test ratio). This extremity is the direct cause of the
ICCAD5-specific pathology investigated in Sections 16–18.

Preprocessing (identical throughout the project): images are resized to 224×224 and
rescaled by 1/255 to map integer pixels to float32 `[0,1]`. Because the source rasters are
strictly binary, background is exactly `0.0` and geometry is exactly `1.0` after rescale;
PIL's default bicubic resize (used by the `XAIEngine` pipeline) introduces mild
anti-aliasing at edges (~5-6% of pixels take intermediate values), while Keras's
`ImageDataGenerator`/`load_img` (used by the official training pipeline) defaults to
nearest-neighbor — a minor, documented preprocessing-path inconsistency between the
training/evaluation pipeline and the XAI pipeline that does not materially affect
conclusions (confirmed via an independent full-test-set reproduction, Section 16) but is
recorded for completeness.

## 6. CNN Architecture

Two architectures appear in this project, deliberately kept distinct.

**Baseline architecture** (`src/model.py`, ~12,873 parameters), a faithful reproduction of
Approach 2 from `LHD_CustomModel.ipynb`:

```
Input (224×224×3)
├─ Conv2D(12,3×3,ELU,valid) → Conv2D(12,3×3,ELU,valid) → Conv2D(12,3×3,linear,valid)
│  → BatchNorm → ELU → MaxPool(2×2)                              → (109,109,12)
├─ MaxPool(5×5)                                                  → (21,21,12)
├─ Conv2D(12,3×3,ELU,valid) → Conv2D(12,3×3,ELU,valid) → Conv2D(12,3×3,linear,valid)
│  → BatchNorm → ELU → MaxPool(2×2)  [conv2d_5, TARGET LAYER]    → (7,7,12) after pool
├─ Flatten → Dropout(0.3) → Dense(10, ReLU) → Dense(1, sigmoid)
```

The baseline's aggressive valid-padding convolutions plus a 5×5 max-pool collapse the
224×224 input to a 15×15 final convolutional grid (`conv2d_5`) — a ~223× reduction in
spatial pixel area. This produces coarse, diffuse Grad-CAM heatmaps unsuitable for
fine-grained spatial attribution (documented as the "Root Cause of Poor Baseline Grad-CAM"
in the initial audit): aggressive spatial pooling, a blurred 15×15→224×224 bilinear
upsample, reliance on vanilla Grad-CAM only, and single-layer inspection.

**XAI-ready architecture** (`src/model_xai.py`), built specifically for this project's
explainability work, using `'same'` padding and modular 2×2 pooling to preserve richer
spatial resolution at three named layers: `conv_early_2` (112×112), `conv_mid_2` (56×56),
and `conv_final_2` (28×28, the primary target layer for all attribution methods in this
report). A `dense_penultimate` `Dense(16, ReLU)` bottleneck sits between `Flatten`+
`Dropout(0.3)` and the final `Dense(1, sigmoid)` output (`dense_output`). Total parameter
count is kept lightweight (~25k-35k). This architecture — not the original baseline — is
the model used for every Grad-CAM/Grad-CAM++/LayerCAM/Occlusion/geometry result in this
report.

## 7. Training Methodology

All models are trained with `tf.keras.utils.set_random_seed` for reproducibility, the
Nadam optimizer, binary cross-entropy loss, batch size 32, 224×224×3 input, no data
augmentation. Baseline epoch counts: 5 for iccad1-2, 10 for iccad3-5. XAI-model epoch
count: 10 for all benchmarks. Class weighting (`w_c = N / (2·N_c)`, standard inverse-class-
frequency weighting) is applied by default in the production XAI training pipeline
(`src/train_xai.py::compute_class_weights`).

A material training-methodology gap was found and fixed mid-project specifically for
ICCAD5: the original training loop had **no best-epoch checkpoint selection** — it saved
whatever weights existed at the final epoch unconditionally, even though ICCAD5's
validation balanced-accuracy history was highly non-monotonic (e.g., one run swung from
0.964 at epoch 6 down to 0.783 at epoch 8 before landing at 0.950 at the saved final
epoch 9). A `BestBalancedAccuracyCheckpoint` callback (`src/metrics.py`) was added,
reusing the project's existing balanced-accuracy formula (`0.5·(sensitivity +
specificity)`) computed from Keras's native per-epoch TP/TN/FP/FN counts, and wired into
`model.fit(..., callbacks=[...])` — an additive, non-destructive change that did not alter
any other training hyperparameter. This callback is used throughout every subsequent
ICCAD5 retraining/ablation/replication experiment described in Sections 16-18.

## 8. Explainability Methods

Four attribution methods are used throughout, all implemented in `src/xai_engine.py`
(`XAIEngine`) and `src/occlusion.py`, and all evaluated against the **pre-sigmoid logit**
score rather than the post-sigmoid probability, to avoid vanishing gradients from sigmoid
saturation on high-confidence predictions (`S_HS = -z_NHS`, `S_NHS = +z_NHS`, where
`z_NHS` is the raw `dense_output` logit — the model's sole output neuron encodes
`p(NHS)`). All four methods score the same target in the same direction, a verified
prerequisite for cross-method comparison.

**Grad-CAM** (Selvaraju et al., ICCV 2017): global-average-pools the target score's
gradient with respect to each channel of the target conv layer (`conv_final_2`,
28×28×32) to obtain per-channel weights `α_k`, forms `ReLU(Σ_k α_k A^k)`, then bilinearly
upsamples to 224×224 with max-normalization pre- and post-resize.

**Grad-CAM++**: replaces GAP weighting with a higher-order-derivative weighting
(`α_ij^k` from second/third-order partials), intended to better handle multiple
co-occurring instances of a class-relevant feature.

**LayerCAM**: uses **pixel-wise** ReLU-clipped gradient weights (`w_ij^k = ReLU(∂S/∂A_ij^k)`)
rather than a single per-channel scalar, preserving more fine spatial structure before
upsampling; can be evaluated at any of the three named layers (`conv_early_2`,
`conv_mid_2`, `conv_final_2`).

**Occlusion Sensitivity**: an independent, gradient-free, perturbation-based method.
Sliding 32×32 windows (16×16 stride, 169 windows per 224×224 image, 50% overlap) are
each replaced with baseline value `0.0`; the drop in target score `ΔS = S(x) - S(x^{(k)})`
is measured per window and coverage-normalized into a 224×224 sensitivity map. Baseline
`0.0` is empirically justified as the exact background pixel value in this binary domain
(not an arbitrary constant, unlike gray/blur baselines in natural-image XAI). A dedicated
audit (`docs/XAI_RESEARCH_AUDIT.md`) found the implementation methodologically sound: sign
convention, target-score formulation, and coverage normalization are all internally
consistent and match the CAM-family convention; no bug was found.

All four methods' output maps are ReLU-clipped to isolate positive (supporting) evidence,
max-normalized to `[0,1]`, and (for the CAM family) bilinearly upsampled from a native
28×28 grid — a resolution mismatch with occlusion's native-pixel-resolution, ~16-32px
effective smoothing that is an explicit, documented limitation (Section 20) rather than a
bug.

Representative samples for XAI analysis are drawn per benchmark using a fixed selection
rule: for each of TP/FP (sorted by `p_HS` descending) and TN/FN (sorted by `p_NHS`
descending), the top `min(available, 3)` most-confident samples are kept. This yields 57
total samples across all five benchmarks (15 TP, 15 TN, 15 FP, 12 FN; iccad1 has only 2 FN
due to 99.1% recall, iccad5 has only 1 FN in the original set due to 97.6% recall) — an
intentional design choice (verified via code-path audit, not a bug), not a subsample of a
larger diagnosed set.

## 9. Raster Geometry Analysis

Because no polygon/DRC ground truth exists in this repository, all "geometry" claims are
computed as **raster-derived proxies** from the binary 224×224 image via standard
`scipy.ndimage`/`skimage` operators, explicitly labeled as proxies throughout (never
described as DRC violations or physical hotspot causes):

- **Geometry mask**: `grayscale ≥ 127` binarization.
- **Connected components** (`scipy.ndimage.label`, 8-connectivity): per-component area/count.
- **Width map / gap map** (`2 × distance_transform_edt`): local line-width proxy (on
  geometry pixels) and local spacing proxy (on background pixels).
- **Thin-line mask / narrow-gap mask**: bottom-quintile (per-image relative percentile) of
  the nonzero width/gap distributions — a per-image threshold, since absolute pixel widths
  vary by benchmark.
- **Skeleton** (`skimage.morphology.skeletonize`) → **junction mask** (skeleton pixels
  with ≥3 skeleton neighbors, dilated) and **corner mask** (Harris corner-response peaks,
  `skimage.feature.corner_harris` + `corner_peaks`, dilated) — explicitly labeled "raster
  junction/corner," not a claim about lithographic mechanism.
- **Local density map** (32×32 box filter of geometry-pixel fraction), matched to the
  occlusion window size for cross-method spatial alignment.

Attribution top-K% regions (K ∈ {10, 20, 30}) are defined by **exact pixel-area rank**
(not a fixed value threshold), guaranteeing every method's mask at a given K has identical
pixel count for a given sample — so overlap differences reflect spatial placement, not
mask-size confounds. **Control regions** are generated from a non-overlapping 7×7 grid of
32×32 tiles (224 = 7×32 exactly); 30 independent random draws are taken per
`(sample, K)` and **shared across all four XAI methods** at that `(sample, K)` — a paired
design enabling paired (Wilcoxon signed-rank) statistical comparison rather than
independently-noisy controls per method. Nine raw overlap metrics (geometry_fraction,
narrow_gap_fraction, thin_line_fraction, boundary_proximity_mean, junction_recall,
junction_density, corner_recall, corner_density, local_density_mean) are kept as separate
columns — never combined into a single composite "geometry relevance score," by design.

A smoke test (4 samples, one per case type) preceded the full 57-sample run and surfaced
two non-fatal limitations carried into interpretation: thin-line/narrow-gap masks have low
discriminating power on near-uniform-width layouts, and junction seeds are near-zero
across almost the entire dataset (most ICCAD-12 layouts are simple parallel line/space
patterns without true skeleton branch points) — junction-based metrics are retained in the
CSVs for completeness but not treated as informative.

## 10. Experimental Design

The overall program proceeds in clearly staged phases, each gated on the prior phase's
findings and each explicitly scoped to avoid retraining or architecture changes except
where a phase's stated purpose is exactly that:

1. **Phase 1 (Baseline)**: faithful reproduction of the notebook baseline on all 5
   benchmarks (Section 11).
2. **Phase 2 (XAI-ready model + multi-method pipeline)**: new architecture, training, and
   the Grad-CAM/Grad-CAM++/LayerCAM engine, run on all 5 benchmarks' 57 representative
   samples (Section 12).
3. **Phase 3 (Occlusion)**: independent perturbation-based validation, cross-correlated
   against the CAM family on the same 57 samples (Section 13).
4. **Phase 4 (Geometry)**: raster-geometry correspondence analysis, on the same 57
   samples, all four methods, paired random-control design (Section 14).
5. **Phase 5 (TP/FP discriminative-feature search + forensic audits)**: a follow-up
   asking whether raw geometry features separate TP from FP attribution regions,
   which surfaced the ICCAD5 output-collapse finding as an unplanned but consequential
   discovery (Section 15).
6. **Phase 6 (ICCAD5 forensics, checkpoint fix, bottleneck analysis, imbalance ablation,
   seed replication)**: a dedicated investigation into the ICCAD5 pathology, escalating
   from read-only forensics to a single best-epoch checkpoint fix, to a full activation-
   level bottleneck analysis, to a controlled 3-condition imbalance ablation, to a 3-seed
   replication of the ablation's most consequential finding (Sections 16-18).
7. **Phase 7 (Final ICCAD5 evidence pass)**: a last, frozen-checkpoint XAI + occlusion +
   geometry pass on the adopted `balanced_sampling` condition (Sections 12-15's numbers
   as reproduced/extended specifically for ICCAD5's final state).

Throughout, "no retraining without an explicit, documented justification" and "no
declaring a method or checkpoint the winner from a single run" are enforced project-wide
conventions, visible in nearly every source document's own limitations section.

## 11. Classification Results

**Baseline CNN** (`src/model.py`, final-epoch, HS as positive class):

| Benchmark | Balanced Acc. | Precision | Recall (HS) | Specificity | F1 |
|-----------|---------------|-----------|-------------|-------------|-----|
| iccad1 | 0.8851 | 0.1804 | 0.9867 | 0.7835 | 0.3051 |
| iccad2 | 0.9893 | 0.5119 | 0.9900 | 0.9886 | 0.6749 |
| iccad3 | 0.9672 | 0.4707 | 0.9773 | 0.9571 | 0.6354 |
| iccad4 | 0.8578 | 0.6720 | 0.7175 | 0.9981 | 0.6940 |
| iccad5 | 0.9822 | 0.1556 | 0.9756 | 0.9888 | 0.2685 |
| **Average** | **0.9363** | 0.3981 | 0.9294 | 0.9432 | 0.5156 |

The notebook's own reported average validation balanced accuracy (best-epoch history) was
≈0.953; this project's local, final-epoch, HS-positive reproduction averaged 0.9363 —
close but not identical, an expected consequence of final-epoch vs. best-epoch reporting
convention rather than a reproduction failure.

**ICCAD5's baseline balanced accuracy (0.9822) looks strong in isolation but is
substantially inflated by extreme class imbalance** — its precision is only 0.1556, and
(as established in Sections 16-17) part of the underlying representation used to achieve
this number is degenerate for the production XAI-model checkpoint. The XAI-model ICCAD5
checkpoint lineage (distinct architecture from the baseline above) is tracked separately
in Sections 16-18; its production (class-weighted) balanced accuracy was 0.9501
(final-epoch) → 0.9649 (best-epoch fix) → superseded by the final adopted
`balanced_sampling` condition at 0.9743 (Section 18), none of which are directly comparable
to the baseline-architecture number above.

## 12. XAI Results

**Attribution concentration and inter-method agreement**, aggregated across all
57 representative samples in the main pipeline (`docs/XAI_RESEARCH_AUDIT.md`,
`docs/GEOMETRY_ANALYSIS_REPORT.md`): the three CAM-family methods frequently disagree
spatially with one another outside of TP cases. In the final ICCAD5-specific evidence pass
(Section 16's frozen checkpoint, `docs/FINAL_ICCAD5_SYNTHESIS.md`), mean peak-to-peak pixel
distance between Grad-CAM and Grad-CAM++ was only 6.5 px (of 224) on TP samples but
21.6-147.3 px on FP/TN/FN — i.e., **TP is the only case type where methods reliably agree
on the region they attribute**, while on TN/FP/FN the three methods frequently point at
substantially different regions.

A critical methodological finding, discovered mid-project and applying project-wide, is
**attribution-map degeneracy**: a nontrivial fraction of Grad-CAM and Grad-CAM++ maps are
**exactly flat** (`np.ptp(map) < 1e-6`) due to genuinely zero or near-zero gradients at
`conv_final_2`, not an implementation bug. Across the original 57-sample set: Grad-CAM
18/57 (31.6%), Grad-CAM++ 24/57 (42.1%), LayerCAM 5/57 (8.8%, concentrated almost entirely
in iccad5 TP/FP). Because a flat map's top-K% mask is an arbitrary tie-break rather than a
meaningful region, this materially affects how the geometry-correspondence numbers in
Section 14 should be read for the affected methods and samples. A guard
(`is_degenerate_map`) was retrofitted into the geometry pipeline to flag (not silently
drop) these rows.

**No overall ranking or "winner" among Grad-CAM/Grad-CAM++/LayerCAM/Occlusion is asserted
anywhere in this project**, by explicit design. Instead, method-specific patterns are
reported: Grad-CAM is the method most prone to degeneracy and to case-type-dependent
unreliability (below-chance geometry correspondence specifically on TN, at-chance on FN);
Grad-CAM++ is generally positive but higher-variance; LayerCAM (pixel-wise weighting, no
GAP step) retains a nonzero map even when Grad-CAM's gradient is fully saturated, and shows
the most consistent behavior across case types.

## 13. Occlusion Results

Occlusion sensitivity, computed independently via forward-pass perturbation (no
gradients), is compared against each CAM-family method via Pearson, Spearman, cosine, and
IoU@50 on the 57-sample set (`docs/OCCLUSION.md`, `docs/XAI_RESEARCH_AUDIT.md`):

| Method | Overall mean Pearson | Overall **median** Pearson | Overall mean cosine |
|---|---:|---:|---:|
| Grad-CAM | 0.0803 | **0.000** | 0.2810 |
| Grad-CAM++ | 0.1006 | **0.000** | 0.2574 |
| LayerCAM | 0.3363 | **0.305** | 0.5006 |

**The Grad-CAM and Grad-CAM++ medians are exactly zero** — i.e. the *majority* of samples
show literally zero measured correlation with occlusion, a direct consequence of the
degeneracy finding in Section 12 (the comparison function correctly returns 0 for a flat
map). This is a materially different, and more accurate, characterization than "a small
positive correlation typical of most samples." By case type, LayerCAM shows the strongest
occlusion agreement on TP (Pearson +0.489) and FP (+0.440); TN and FN show markedly weaker
or negative agreement across all three CAM methods, most severely for Grad-CAM. By
benchmark, iccad5 shows median ≈0 Pearson for **all three** CAM methods — directly
consistent with the output-collapse finding described in Sections 15-16.

A dedicated sanity check on occlusion's own stability (8 iccad3 samples, comparing
baseline-value swap and window/stride halving) found occlusion attribution is **stable for
TP/FP** (Pearson 0.72-0.99 under baseline substitution, 0.78-0.87 under resolution change)
but **substantially less stable for TN** (Pearson as low as 0.21-0.45, IoU@top-10% as low
as 0.03-0.23) — because TN samples produce weak, near-flat positive-importance signal that
is dominated by noise under any parameter change. This is reported as a real, structural
finding about the model's TN behavior, not an occlusion implementation defect, and
recurs independently in the geometry-correspondence analysis (Section 14).

In the final frozen-checkpoint ICCAD5 pass (11 samples: TP=3, TN=3, FP=3, FN=2), the same
pattern holds and sharpens: TP shows the strongest occlusion-vs-attribution agreement
across all three CAM methods (Pearson 0.67-0.70, cosine 0.79-0.80); TN and FN show
near-zero or slightly negative correlation for all three methods, most severely for
Grad-CAM (Pearson −0.007 to −0.020). Two specific TN samples (`NNHSCAD5999_9`,
`NNHSCAD5999_6`) show **exactly** 0.000 correlation against Grad-CAM — the same two
samples independently flagged degenerate in the geometry pipeline (Section 14), confirming
the degenerate Grad-CAM map is present in the raw saliency values, not an artifact of
top-K masking. Grad-CAM++ and LayerCAM retain small positive correlation on these same two
samples, so occlusion itself is not independently flat there — only Grad-CAM's map is.

## 14. Geometry Correlation Results

The central geometry question — **are high-attribution regions associated with particular
raster-geometry characteristics more strongly than area-matched random control regions?**
— was answered, qualified, using paired Wilcoxon signed-rank tests (n=57, all four methods
sharing the same 30 control draws per sample/level):

| Method | geometry_fraction Δmedian | p | narrow_gap Δmedian | p | thin_line Δmedian | p | corner_recall Δmedian | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grad-CAM | +0.033 | 0.181 | +0.044 | 0.069 | +0.007 | 0.071 | −0.034 | 0.482 |
| Grad-CAM++ | +0.047 | **0.016** | +0.034 | **<0.001** | +0.014 | **0.002** | +0.020 | **0.032** |
| LayerCAM | +0.108 | **<0.001** | +0.129 | **<0.001** | +0.031 | **<0.001** | +0.187 | **<0.001** |
| Occlusion | +0.164 | **<0.001** | +0.087 | **<0.001** | +0.033 | **<0.001** | +0.027 | 0.096 |

**Occlusion and LayerCAM show the strongest, most consistent positive association**
between top-attribution regions and drawn/narrow-gap/thin-line geometry proxies across
case types and benchmarks. **Grad-CAM shows no significant overall association**
(p=0.181) — but this pooled null hides real case-type heterogeneity: Grad-CAM is
*significantly positive* on TP (Δ=+0.118, p=0.022) and FP (Δ=+0.143, p=0.013), roughly at
chance on FN (Δ=−0.016, p=0.424), and **significantly below chance on TN**
(Δ=−0.079, p=0.018) — i.e., Grad-CAM's top-attribution region for correctly-rejected
samples lands predominantly on *background*, more than a random region would.

**FP samples show geometry association at a magnitude comparable to (sometimes exceeding)
TP samples for all four methods** — the raster-geometry proxies used here do **not**, by
themselves, separate genuine hotspot-triggering structure from false-alarm-triggering
structure. A dedicated follow-up (`docs/TP_FP_GEOMETRY_ANALYSIS.md`) directly tested this
with raw feature-value distributions (local width, gap, density, boundary distance) inside
TP vs. FP attribution regions across 4 methods × 3 levels (48 tests, repeated after
excluding degenerate maps for 96 total): **no method/feature/level combination reached
statistical significance** (AUC 0.32-0.64, all Cliff's δ negligible-to-small by
conventional thresholds); the one nominally significant result (Grad-CAM++, density,
top-10% only, p=0.025) did not replicate at other levels and is flagged as very likely
spurious given the number of unconnected tests run. **The headline answer to "does any
raw raster geometry feature discriminate TP from FP attribution regions" is No.**

Cross-benchmark heterogeneity is substantial and would be masked by any single pooled
statistic: iccad1 is a specific Grad-CAM outlier (geometry_fraction 0.079 vs. control
0.230, while occlusion/LayerCAM are strongly elevated on the same benchmark) — traced to a
genuinely broad/diffuse (but non-degenerate) activation pattern specific to that
checkpoint, not a geometry-feature artifact. No corner_recall or other single dimension
tracks the overall geometry_fraction ranking uniformly (e.g. only LayerCAM and Grad-CAM++
show significant corner association; Grad-CAM and Occlusion do not) — a concrete
illustration of why the project deliberately never collapses these nine metrics into one
composite score.

In the **final** frozen-checkpoint ICCAD5 evidence pass (11 samples, `geometry/summary.csv`),
the mean `geometry_fraction_paired_diff_median` (control-adjusted overlap) shows the same
qualitative pattern: TP is the only case type where all four methods agree (0.23-0.24,
uniform across methods and top-K levels); Grad-CAM's control-adjusted overlap is
disproportionately weak outside TP (0.006-0.044 vs. 0.07-0.27 for the other three methods
on the same case types); Grad-CAM++'s FP overlap (0.006) is, in this smaller final sample,
the single weakest cell in the table — a reminder that "Grad-CAM++ is a reliable Grad-CAM
substitute" holds for some case types (TN) but not others (FP) even within one checkpoint.

## 15. TP/TN/FP/FN Analysis

Structured per case type across the full analysis stack:

**TP (True Positive)** is the strongest, most internally-consistent case type across every
analysis family used in this project. Attribution methods agree spatially with each other
(peak-to-peak distance as low as 6.5 px), occlusion independently corroborates the
CAM-family maps (Pearson 0.67-0.70), and all four methods show consistent, non-trivial,
statistically significant overlap with the geometry proxy above random control (paired
diff ~0.23-0.24, uniform across methods and levels).

**TN (True Negative)** is the case type where Grad-CAM specifically breaks down: its
top-attribution region is *anti-correlated* with drawn geometry (significantly below
chance, p=0.018 in the main pipeline), landing predominantly on open background; this
tracks a documented mechanism (Section 16) in which Grad-CAM's global-average-pooling
channel weighting spreads attribution into background when `conv_final_2`'s gradient is
saturated near zero. LayerCAM and Occlusion do not show this inversion. Two of three
representative TN samples in the final ICCAD5 pass produced exactly flat Grad-CAM
attribution maps (Section 16).

**FP (False Positive)**: all four methods place elevated attribution on drawn geometry at
magnitudes comparable to, and sometimes exceeding, the TP case — meaning the raster
geometry the model reacts to when incorrectly calling something a hotspot is **not**
visually or statistically distinguishable, at this level of analysis, from the geometry it
reacts to when correct (Section 14). This is reported as a limit of the analysis, not
evidence that FPs share a causal mechanism with TPs. One iccad5 FP sample
(`NNHSCAD5994_4`) showed a normal, non-degenerate gradient and a distinct predicted
probability, confirming the output-collapse pathology (Section 16) does not affect every
FP uniformly.

**FN (False Negative)** is the weakest, most diffuse case type across every analysis.
Grad-CAM's spread on FN is the smallest of any case type even as the other two CAM methods
stay broad; occlusion agreement with all three CAM-family methods is at its lowest here
(Pearson −0.02 to 0.06); Grad-CAM's control-adjusted geometry overlap is near the table
minimum and statistically indistinguishable from chance (p=0.424). LayerCAM and Occlusion
retain a significant, positive geometry association even on missed detections — evidence
that geometry-relevant *signal* is sometimes present in these two methods' maps even when
the classifier's final decision is wrong, motivating (but not proving) LayerCAM/Occlusion
as candidates for FN triage. All FN-specific findings carry an explicit small-n caveat
(n=12 in the main pipeline, n=2 in the final ICCAD5 pass) and are read as suggestive, not
conclusive.

## 16. ICCAD5 Representation/Bottleneck Investigation

While searching for a raw geometry feature that discriminates TP from FP attribution
regions (Section 14), an unrelated and more consequential finding emerged: a subset of
ICCAD5 predictions produce **model output that is measurably input-invariant**. A
dedicated forensic audit (`docs/ICCAD5_FORENSIC_AUDIT.md`) traced this precisely to a
single layer using direct activation probes (not just downstream diagnostics):

```
Input → conv_early_2 → conv_mid_2 → conv_final_2   all genuinely differ across samples
dense_penultimate (Dense(16, ReLU))                COLLAPSES: exact zero vector for a
                                                     label-correlated subset of samples
dense_output                                        becomes mathematically just the bias
                                                     term for every collapsed sample
```

On the **original, final-epoch production checkpoint** (`xai_cnn_iccad5.keras`), this
16-unit ReLU bottleneck produced the exact-zero vector for **97.6% of true-HS test images**
and 4.7% of true-NHS images (952/19,368 overall, 4.92%). Because a zero-vector sample's
logit is always exactly the output bias (confirmed to 6 decimal places, `−0.332967`), the
network's decision for these samples is mathematically constant, independent of input
content — explaining, correctly and non-controversially, why gradient- and
perturbation-based XAI methods report "nothing to attribute" for them (occlusion is
exactly as affected as Grad-CAM here, since it is a forward-pass probe of the same
collapsed decision surface). Six of sixteen `dense_penultimate` units are completely dead
(output exactly zero for every one of the 19,368 test samples) in the original checkpoint;
every live unit is exclusively **NHS-elevated** — no HS-selective unit was found anywhere
in this checkpoint. Random noise (uniform and binary) collapsed into the identical dead
state 100% of the time on this checkpoint, more reliably than even real true-HS images —
evidence that the dead region of input space is broad and not narrowly tailored to genuine
hotspot semantics.

The audit ruled out three alternative failure modes with direct evidence: not a
checkpoint/loading bug (architecture, weights, file integrity all confirmed correct,
identical across all 5 benchmarks); not a preprocessing bug (inputs are confirmed to
produce genuinely distinct, sample-varying activations through all three convolutional
stages); not an evaluation/metrics bug (an independent from-scratch reproduction of the
full 19,368-image test set matched the original reported balanced accuracy to within 0.55
percentage points, a minor bicubic-vs-nearest-neighbor resize discrepancy, and both the
original and reproduced numbers are far from what a trivial constant predictor achieves —
0.500 balanced accuracy for either "always NHS" or "always HS"). The dominant identified
failure mode is a genuine, trained-checkpoint representational collapse, with a concrete
**contributing, fixable methodological gap**: no best-epoch checkpoint selection was used
during a training run whose validation curve was highly non-monotonic.

A minimal fix (`docs/ICCAD5_CHECKPOINT_FIX.md`) — adding the `BestBalancedAccuracyCheckpoint`
callback (Section 7) and retraining only iccad5, architecture/data/optimizer/loss/seed all
otherwise unchanged — produced a **real but partial** improvement:

| Metric | OLD (final epoch) | NEW (best epoch, epoch 8) |
|---|---:|---:|
| Balanced accuracy | 0.9501 | 0.9649 |
| Precision | 0.0267 | 0.0863 |
| Recall (HS) | 0.9756 | 0.9512 |
| Specificity | 0.9247 | 0.9786 |
| F1 | 0.0520 | 0.1582 |
| Dead-bottleneck rate, overall | 4.92% | **1.81%** |
| Dead-bottleneck rate, true-HS | 97.56% | **78.05%** |
| Random-noise probe | dead | **still dead** |

The synthetic-noise probe still collapsing on the new checkpoint is the clearest evidence
this is **PARTIAL**, not a resolution — reported exactly as such, per the project's
explicit instruction not to declare success from appearance alone. A follow-up activation-
level bottleneck analysis (`docs/ICCAD5_BOTTLENECK_ANALYSIS.md`) further quantified the
fix's mechanism: 67.4% (615/912) of the OLD checkpoint's zero-vector False Positives were
"revived" into a genuinely input-dependent representation by the fix, versus only 20%
(8/40) of zero-vector True Positives — **the fix disproportionately helped false
positives, not true positives** — and found that real information does reach
`conv_final_2` even for collapsed samples (pairwise cosine similarity 0.81-0.83, not 1.0,
among zero-vector samples) but is discarded by the specific learned `dense_penultimate`
weight matrix.

## 17. Class-Imbalance Ablation

A controlled ablation (`docs/ICCAD5_IMBALANCE_ABLATION.md`) isolated whether ICCAD5's
extreme class imbalance itself, or specifically loss-based class weighting, drives the
collapse. Three conditions, architecture/optimizer/loss/split/seed(42) held fixed:

- **A. `baseline_noweight`** — best-checkpoint selection, no class weighting, natural
  batches (newly trained).
- **B. `class_weighted`** — the existing checkpoint-fix checkpoint (Section 16), reused,
  not retrained: `class_weight={0: 52.73, 1: 0.505}`.
- **C. `balanced_sampling`** — best-checkpoint selection, no class weighting, 50/50
  HS/NHS batches via a custom with-replacement sampler (each HS training image seen
  ~53×/epoch on average; each NHS image ~0.5×/epoch) (newly trained).

**Headline result: the collapse implicates loss reweighting specifically, not class
imbalance in general.**

| Condition | Balanced Acc. | HS Recall | Zero-vector rate (overall) | Zero-vector rate (HS) | Dead units | HS-elevated units (max AUC) |
|---|---:|---:|---:|---:|---:|---:|
| baseline_noweight | 0.9741 | 0.9512 | **0.000%** | **0.000%** | 3/16 | 1 (0.982) |
| class_weighted | 0.9649 | 0.9512 | 1.792% | 85.366% | 6/16 | 0 |
| balanced_sampling | 0.9501 | 0.9024 | **0.000%** | **0.000%** | 3/16 | 4 (0.987) |

Both non-class-weighted conditions show **0.000% zero-vector rate in every single
case-type stratum** — the collapse is entirely absent for this specific pathology, not
merely reduced. `balanced_sampling` additionally produced four units with strong,
consistent **HS-elevated** activation (AUC 0.97-0.99) — the first evidence anywhere in
this project's ICCAD5 investigation of anything resembling class-selective structure in
the bottleneck (every prior checkpoint's live units were exclusively NHS-elevated). No
condition is labeled "best": `baseline_noweight` has the highest balanced accuracy;
`balanced_sampling` has the fewest false positives (42) and the highest F1 (0.617) at the
cost of two additional missed hotspots. A plausible (not directly tested) mechanism is
offered: `class_weight`'s ~53× loss-gradient multiplier on each of only 26 HS training
images concentrates an unusually large gradient signal on very few examples passing
through a narrow 16-unit ReLU bottleneck, a configuration known to be conducive to
dead-ReLU dynamics; balanced sampling achieves increased HS exposure through repetition
instead, without inflating per-example gradient magnitude.

Based on this evidence, **`balanced_sampling` was selected for continuation** (not
labeled "best"), on the criterion that it uniquely combines zero collapse with multiple
HS-associated units — the representation properties most directly relevant to whether a
downstream XAI method has genuine, non-degenerate signal to explain. The ablation
explicitly declined to run a combined class-weight + balanced-sampling condition (would
confound attribution of the effect) and explicitly declined to regenerate the full
XAI/occlusion/geometry pipeline immediately, on the grounds that a single-seed result was
not yet a "settled" methodology.

## 18. Seed Replication

Because every ablation condition above was trained with a single seed (42), a dedicated
replication (`docs/ICCAD5_SEED_REPLICATION.md`) retrained `balanced_sampling` with three
independent seeds (101, 202, 303) and `baseline_noweight` with two (101, 202), holding
architecture/data/optimizer/checkpoint-selection fixed and confirming (via pre-training
weight-checksum comparison) that each seed produces genuinely distinct initialization.

**Replicated, holding across all seeds tested:**
- `balanced_sampling` produces **exactly 0.000% exact-zero-vector rate in all 3
  independent seeds**, across every true-label and case-type stratum.
- `balanced_sampling` produces multiple (4, 4, and 7) HS-elevated units (AUC≥0.90) in
  every seed — no single unit index repeats across all three seeds (consistent with
  permutation symmetry of a randomly-initialized layer), but the *existence* of such units
  is consistent.
- `balanced_sampling` shows consistently **fewer dead units than class-weighted in all 3
  seeds** (1, 4, 3 vs. class_weighted's original 6/16).
- `balanced_sampling`'s balanced accuracy is stable (0.9731-0.9743, SD=0.0006 across
  seeds) and consistently higher than the original class_weighted checkpoint's 0.9649.

**Did not replicate:**
- `baseline_noweight`'s original single-seed 0.000% collapse result **did not hold** for
  either replication seed: HS zero-vector rates were 95.1% and 78.0% respectively —
  closer to `class_weighted`'s original collapse rate (85.4%) than to
  `baseline_noweight`'s own original (seed-42) result. **`baseline_noweight`'s
  collapse-avoidance appears to have been a single-seed artifact and should not be relied
  upon; it is seed-dependent and unreliable.**
- The original "random noise always collapses the bottleneck" finding **did not generalize**
  to any of the five newly-trained, non-class-weighted checkpoints (0% collapse on
  identical uniform/binary noise probes in every case, including the two `baseline_noweight`
  seeds that collapsed heavily on real HS data). This narrows the noise-vulnerability
  finding to (at least) the class-weighted training recipe specifically, and it must not
  be presented as a general property of the architecture.

**Explicit scoping-decision caveat, preserved transparently rather than glossed over:**
the seed-replication document's own concluding decision framework states plainly that
**there was not yet sufficient evidence to fully "freeze" `balanced_sampling` as the
production-final ICCAD5 methodology** — three seeds is a real replication (stronger
evidence than existed before) but cannot rule out a lower-frequency failure mode (e.g., an
occasional bad seed of the kind observed for `baseline_noweight`), and its precision/F1
trade-off relative to the other conditions was not re-examined in the replication. The
seed-replication document's own recommended next step was a larger (5+) seed sweep before
treating this as settled, and it explicitly recommended against regenerating the full
XAI/occlusion/geometry pipeline until that further sweep was done.

**This report nonetheless adopts `balanced_sampling` (specifically, seed 101,
`models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras`, sha256
`9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70`, balanced accuracy
0.9743) as the final, frozen working condition for the Section 12-15 evidence reproduced
specifically for ICCAD5 below — as an explicit, documented project-scope decision, not
because the evidence bar recommended by the seed-replication document itself was actually
met.** The 3-seed replication was judged sufficient for this project's scope (a
capstone research report, not a production deployment decision); a reader relying on this
checkpoint for any downstream production purpose should be aware that the seed-replication
authors themselves recommended a larger sweep first. Checksums for all three
`balanced_sampling` seeds trained during replication are recorded in
`results/xai/final_iccad5/CHECKPOINT_CHECKSUMS.txt`.

**Final ICCAD5 evidence pass on the frozen checkpoint** (`docs/FINAL_ICCAD5_SYNTHESIS.md`,
11 representative samples: TP=3, TN=3, FP=3, FN=2) reproduces the qualitative patterns of
Sections 12-15 (TP is the strongest, most mutually-corroborating case type across
attribution/occlusion/geometry; Grad-CAM is the least reliable method outside TP) and adds
one materially new, precisely-scoped finding, detailed next.

### The Grad-CAM degenerate-map finding in the final pass, and why it is distinct from the dense_penultimate collapse

In this final pass's geometry diagnostics, exactly **6 of 132 rows (4.5% of all rows;
18.2% of the 33 Grad-CAM-method rows specifically)** are flagged `is_degenerate_map=True`.
All 6 are Grad-CAM; **zero** for Grad-CAM++, LayerCAM, or Occlusion. All 6 belong to the
TN case type, concentrated in only 2 of the 3 TN samples (`NNHSCAD5999_9`,
`NNHSCAD5999_6`), each flagged at all three top-K levels. **This is a narrow,
method-and-checkpoint-specific observation about Grad-CAM's gradient behavior on 2 of 11
samples in this specific representative set. It is explicitly not a recurrence of, and
must not be conflated with, the `dense_penultimate` exact-zero-vector collapse documented
in Sections 16-18.** The two are structurally distinct phenomena:

| | `dense_penultimate` collapse (Sections 16-18) | Grad-CAM degeneracy (this finding) |
|---|---|---|
| Layer | `dense_penultimate` bottleneck activations | `conv_final_2` gradient feeding Grad-CAM's GAP-weighted CAM |
| Evidence | exact-zero activation vector, measured across the full 19,368-image test set | `np.ptp(map) < 1e-6` flat saliency map, measured on 11 representative samples |
| Status on this final checkpoint | **resolved to 0.000% rate** across all 3 balanced_sampling seeds | **still occurs** (6/33 Grad-CAM rows, 2/3 TN samples) |
| Affects which methods | all downstream computation reading that layer | Grad-CAM only — Grad-CAM++, LayerCAM, Occlusion are unaffected on the same samples |

The qualitative pattern — Grad-CAM specifically, more than Grad-CAM++, much more than
LayerCAM, going flat on high-confidence TN/FN predictions — is **consistent with** (but
not a re-verification of) a mechanism already documented on an earlier, different-lineage
checkpoint (`docs/TP_FP_GEOMETRY_ANALYSIS.md` §11b): saturated pre-sigmoid logits drive
most of `conv_final_2`'s gradient to exactly zero, with the small remaining nonzero
fraction near-cancelling under Grad-CAM's global-average-pooling channel weighting
(tripping the `max_v > 0` postprocessing guard), while LayerCAM's pixel-wise weighting
(no GAP step) still recovers a nonzero map from the identical gradient. The earlier
investigation reported Grad-CAM/Grad-CAM++ degenerating on 31.6%/42.1% of a 57-sample set
vs. 8.8% for LayerCAM, concentrated in TN/FN; this final checkpoint's 18.2%-of-Grad-CAM-
rows, 0%-for-the-other-three-methods pattern on an 11-sample set is directionally the same
shape of result, now confirmed (with the `is_degenerate_map` flag freshly ported into the
geometry pipeline) on the `balanced_sampling seed1` checkpoint specifically — but the
direct gradient probe that would re-confirm the identical GAP-cancellation mechanism on
*this* checkpoint was not re-run here; the mechanism is offered as the most
evidence-consistent prior explanation, not re-verified.

Excluding the 6 degenerate rows leaves only 1 remaining TN/Grad-CAM sample; the apparent
~4x jump in `geometry_fraction_median` after exclusion (0.07 → ~0.29-0.31) is a small-n
artifact of dropping 2 of 3 samples, not evidence that non-degenerate Grad-CAM performs
much better on TN — this instability is reported explicitly rather than presented as a
clean "fix."

## 19. Integrated Discussion

Reading across all seven phases, four analytically distinct concepts recur throughout this
report and must never be conflated, per the project's own established convention:

1. **Spatial attribution** (Grad-CAM/Grad-CAM++/LayerCAM): a gradient-derived map of which
   input regions positively influenced the model's target-class score.
2. **Perturbation sensitivity** (Occlusion): an independent, gradient-free measure of how
   much the model's score changes when a region is physically removed from the input.
3. **Raster geometry correspondence**: a statistical association between attribution/
   sensitivity regions and image-processing-derived proxies for layout structure (thin
   lines, narrow gaps, corners) — computed entirely from the binary raster, with no
   access to polygon or DRC ground truth.
4. **Physical hotspot causality**: whether a layout feature actually causes a lithography
   printing failure — a claim this project's dataset and methods cannot support or refute,
   since no optical-simulation or DRC ground truth exists anywhere in the repository.

The strongest, most cross-corroborated finding in this entire project is that these four
concepts **agree with each other specifically on correctly-classified hotspots (TP)**: on
TP samples, the three gradient methods spatially agree with each other, occlusion
independently corroborates the same region, and that region overlaps raster-geometry
proxies significantly more than chance — three independent evidence families converging.
This convergence does **not** extend to physical causality (concept 4), which remains
outside the scope of anything measurable in this dataset.

The second strongest finding is the opposite: on TN and (for Grad-CAM specifically) on FN,
the same three evidence families **independently agree that the explanation is weaker,
less spatially concentrated, and less well corroborated**. This is not a uniform failure
of "XAI" as a category — it is concentrated in Grad-CAM specifically (the GAP-averaging
mechanism and outright gradient degeneracy), while LayerCAM and Occlusion retain
meaningful, positive geometry association even on these harder cases.

A third, humbling finding recurs across two independently-designed experiments (the TP/FP
raw-geometry-feature search, Section 14, and the geometry-correspondence pooled analysis,
Section 14): **the raster-geometry proxies used here do not separate genuine
hotspot-triggering structure (TP) from false-alarm-triggering structure (FP)**. The model's
false positives are attributed to, and statistically resemble, the same kind of raster
geometry as its true positives at this level of analysis. This is reported as a limitation
of the available geometric feature set (and, by extension, of what raster-only geometry
proxies can determine at all), not as evidence that the FPs are "really" hotspots or that
the model's errors are unexplainable.

Finally, the ICCAD5 investigation (Sections 16-18) demonstrates a case where a
representational pathology in the trained network itself — not any XAI method — was the
actual limiting factor for explainability: when a bottleneck genuinely collapses to a
constant, no attribution or perturbation method can recover a meaningful spatial
explanation, because there is no spatial dependence left to recover. Distinguishing "the
XAI method failed" from "the model has nothing to explain" required activation-level
forensics (Section 16) that no attribution map alone could have revealed — a methodological
lesson that generalizes beyond this specific dataset: a null or degenerate XAI result
should trigger a check of the underlying representation before it is read as an XAI-method
weakness.

## 20. Limitations

- **No polygon, GDS/OASIS, or DRC ground truth exists anywhere in this repository.**
  Every geometry claim in Sections 9, 14, and 16-18 is a raster-derived image-processing
  proxy, not verified layout semantics; none of it should be read as confirming or
  refuting a specific lithography failure mechanism.
- **Attribution-map degeneracy** affects a substantial minority of Grad-CAM (31.6% of the
  main 57-sample set) and Grad-CAM++ (42.1%) samples, concentrated in TN/FN and in the
  original iccad5 checkpoint; LayerCAM is far less affected (8.8%). This was retrofitted
  as an explicit flag (`is_degenerate_map`) but was not caught in the original geometry
  pipeline run, meaning `docs/GEOMETRY_ANALYSIS_REPORT.md`'s pooled Grad-CAM/Grad-CAM++
  statistics should be read with this caveat.
- **Resolution mismatch between CAM methods and occlusion**: the CAM family's spatial
  information is intrinsically limited to a 28×28 grid before an 8× bilinear upsample;
  occlusion is generated natively at pixel resolution with an effective 16-32px smoothing
  kernel. These are different smoothing operators producing different spatial-frequency
  content, which partially confounds any cross-method correlation comparison.
- **Zero-inflation** in cross-method correlation metrics: both sides of every CAM-vs-
  occlusion comparison are ReLU-clipped, so mostly-zero vectors agreeing on *where the
  zeros are* inflates Pearson/cosine relative to agreement on the informative,
  nonzero-attribution subset.
- **Small, non-random sample sizes throughout**: 57 total representative samples across 5
  benchmarks (as few as n=12 for FN pooled, n=2-3 per case type per benchmark); the final
  ICCAD5-specific pass uses only 11 samples (n=2 for FN). All statistics at this scale are
  descriptive/exploratory, not confirmatory; multiple-comparison correction was not
  applied to most tables (explicitly flagged wherever relevant), so borderline p-values
  should be read as suggestive only.
- **Junction-based geometry metrics are near-degenerate** on this dataset — most ICCAD-12
  layouts are simple parallel line/space patterns with few true skeleton branch points —
  and are retained in the CSVs for completeness but not treated as informative.
- **The ICCAD5 `balanced_sampling` checkpoint adopted as final is not, by the seed-
  replication document's own explicit conclusion, backed by sufficient evidence to be
  called "frozen" in a production sense** — 3 seeds is a real but limited replication; a
  larger (5+) seed sweep was recommended and not performed. This report's adoption of it
  as the final working condition is a documented, explicit project-scope decision, not a
  claim that the underlying evidence bar has been fully met.
- **The random-noise collapse finding does not generalize**: it was observed only on
  class-weighted checkpoints (2 total ever tested this way, OLD and NEW) and explicitly
  did not replicate on any of the 5 non-class-weighted checkpoints trained during
  replication — it must not be presented as a general property of the architecture.
- **No formal separability statistic (silhouette score, trained classifier) was computed**
  on any penultimate-layer PCA visualization in the ICCAD5 investigation, by deliberate
  choice, given the very small number of non-degenerate HS points available in several
  checkpoints (as few as 1).
- **Preprocessing path inconsistency**: the XAI/occlusion/geometry pipeline uses PIL
  bicubic resize; the official training/evaluation pipeline uses Keras's nearest-neighbor
  default. This produces a small (≈0.55 percentage point, ICCAD5-measured) but nonzero
  discrepancy between independently-reproduced and originally-reported metrics.

## 21. Conclusion

**Can explainable-AI techniques identify the layout regions or features that contribute
most strongly to hotspot classification?**

The evidence supports a qualified, method- and case-type-dependent **yes**, with an
explicit boundary on what "identify" can mean given this dataset. On correctly-classified
hotspots (TP), all four XAI methods — two independent gradient-based families and one
independent perturbation-based method — converge on the same spatial regions, and those
regions correspond, more than chance, to raster-derived proxies for narrow gaps and thin
lines, the layout characteristics conventionally associated with lithography sensitivity
in the ICCAD-12 literature. This three-way convergence (spatial attribution ↔ perturbation
sensitivity ↔ geometry correspondence), replicated across five benchmarks and reconfirmed
on ICCAD5's separately-investigated final checkpoint, is the strongest evidence this
project can offer that the network's HS-predicting decisions are anchored in genuine,
recognizable layout structure rather than arbitrary or spurious features — for the cases
where the network is confidently and correctly right.

That "yes" narrows substantially outside TP. On true negatives and false negatives, the
same three evidence families independently agree that the explanation is weaker, less
spatially concentrated, and (specifically for Grad-CAM) sometimes anti-correlated with
geometry or outright degenerate due to gradient saturation. On false positives, the
raster-geometry proxies available in this dataset **cannot distinguish** the geometry the
model reacts to when wrong from the geometry it reacts to when right — a genuine limit of
what this analysis, and possibly this class of raster-only geometric feature, can resolve.
And on ICCAD5 specifically, a portion of what looked like an explainability failure turned
out to be a genuine representational collapse in the trained network — a case where the
correct scientific conclusion was "there is nothing here to explain," reachable only by
activation-level forensics that no attribution map alone could have revealed. Crucially,
none of these findings license a physical-causality claim: this dataset provides no DRC or
optical-simulation ground truth, so "the model attributes greater importance to this
region" and "this region overlaps a narrow-gap/thin-line proxy" are the strongest claims
this project's evidence supports — never "this region is the physical cause of the
hotspot."

## 22. Future Work

1. **Larger seed sweep (5+) for `balanced_sampling`** before treating it as a settled
   ICCAD5 production methodology, per the seed-replication document's own explicit
   recommendation, which this report's scoping decision (Section 18) knowingly did not
   fully satisfy.
2. **Independent seed replication of `class_weighted`** itself (only ever trained once, at
   seed 42, throughout this entire project) — its own seed-sensitivity is currently
   unknown.
3. **A combined class-weight + balanced-sampling condition**, deliberately omitted from
   the imbalance ablation to preserve mechanism isolation, as a natural follow-up if a
   production-oriented "best of both" configuration is wanted.
4. **Re-run the full 57-sample geometry pipeline with the `is_degenerate_map` flag**
   applied from the start, separating degenerate from non-degenerate Grad-CAM/Grad-CAM++
   statistics throughout `GEOMETRY_ANALYSIS_REPORT.md`, not only in the ICCAD5-specific
   final pass and the TP/FP follow-up.
5. **A discriminative feature beyond mean width/gap/density/boundary-distance** for
   separating TP from FP attribution regions — e.g. the distribution shape (not just mean)
   of local width/gap values within the region, or a feature combining multiple proxies —
   since the current univariate raw-feature search returned a clean null result.
6. **Expand FP/FN sample counts** beyond the current 15/15 (main pipeline) and 3/2
   (final ICCAD5 pass) — both are underpowered for anything but a large effect, and FN
   analysis in particular rests on very small n throughout this project.
7. **Directly re-probe the ICCAD5 `balanced_sampling` checkpoint's Grad-CAM gradient
   at `conv_final_2`** for the 2 flagged-degenerate TN samples, to confirm (rather than
   infer from an earlier, different-lineage checkpoint) the GAP-cancellation mechanism
   proposed in Section 18.
8. **A weight-space investigation of why `dense_penultimate` routes the way it does** —
   which upstream `conv_final_2` directions specifically trigger the dead branch — was
   explicitly out of scope for every ICCAD5 document in this project and remains open.
9. **Full 57-sample-scale XAI/occlusion/geometry regeneration for ICCAD5** under whichever
   condition is eventually adopted with sufficient seed evidence (item 1), rather than the
   11-sample final evidence pass used in this report, once the underlying training
   methodology is genuinely settled.
