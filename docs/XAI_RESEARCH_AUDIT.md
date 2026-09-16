# XAI Research Audit

Forensic audit of the Occlusion Sensitivity implementation, its comparability with the
existing Grad-CAM / Grad-CAM++ / LayerCAM pipeline, and a feasibility study for
connecting attribution maps to ICCAD-12 layout geometry. Scope: audit + small sanity
experiments only. No retraining, no weight changes, no new XAI methods.

## 1. Executive Summary

The occlusion implementation (`src/occlusion.py`, `src/run_occlusion.py`) is
**methodologically sound**. No implementation bug was found; nothing was changed in the
codebase. The sign convention, target-score formulation, and coverage-normalized
aggregation are all internally consistent and match the existing Grad-CAM family's
pre-sigmoid-logit convention.

The comparison between Occlusion and the three CAM methods (Pearson/Spearman/cosine/IoU
on max-normalized, ReLU-clipped maps) is **defensible as a descriptive, exploratory
signal but not as a faithfulness ranking**. The four maps differ systematically in native
spatial resolution (28×28 upsampled 8× for the CAMs vs. a ~32px effective box-filter for
occlusion), both are zero-inflated after ReLU clipping (inflating raw Pearson agreement),
and the maps being compared are spatially autocorrelated, which invalidates the implicit
assumption behind a Pearson p-value (there is no p-value reported today, which is
correct — but any future significance test on 224×224=50,176 "samples" per image would be
badly anti-conservative).

Sanity experiments (Part B, n=8 real samples, not simulated) show occlusion attribution
is **reasonably stable under 0/mean baseline swap and window/stride halving for TP/FP
cases** (Pearson 0.72–0.99), but **substantially less stable for TN cases** (Pearson
0.21–0.68, IoU@top-10% as low as 0.03), because TN samples produce weak, near-flat
positive-importance signal that is dominated by noise once the baseline or resolution
changes. This is a real, reportable finding, not a bug.

The ICCAD-12 dataset, as present in this repository, contains **only rasterized binary
PNG layouts** (0/255, background/geometry) — no GDS/OASIS, no polygon coordinates, no DRC
layers, no line-width/spacing metadata. Any geometry-correlation study must derive
geometry features from the raster image itself (connected components, skeletons, gap/
line-width via distance transform, corner/junction detection) — it cannot use ground-truth
polygon annotations because none exist in this repo.

**No retraining is justified.** The CNN weights are not implicated by anything found here;
this is entirely an explanation-methodology and downstream-geometry-analysis question.

## 2. Occlusion Implementation Audit

Verified against `src/occlusion.py` line-by-line:

- **Preprocessing** (`XAIEngine.preprocess_image`, `src/xai_engine.py:100-120`): images are
  loaded, resized with PIL's default (bicubic) resampling, cast to `float32`, divided by
  `255.0`. No mean-subtraction, no channel standardization. So the tensor that enters the
  model lives in `[0, 1]`.
- **Is 0.0 a valid semantic "background" value?** Verified empirically on the raw source
  PNGs (`iccad-official/iccad*/train/*/*.png`): every sampled image is **strictly binary**,
  pixel value `0` (background) or `255` (geometry) — confirmed via `np.unique` on 6 sample
  files across HS/NHS classes (2 unique values each). After the `/255.0` rescale, background
  is exactly `0.0` and geometry is exactly `1.0`. **Baseline replacement value `0.0` is
  therefore not an arbitrary or out-of-distribution constant — it is the literal, exact
  background class value used throughout training and inference.** This is a stronger
  justification than typical occlusion baselines in natural-image XAI (where gray/blur
  baselines are a known source of controversy) and does not need to be changed.
  - Caveat: PIL's default bicubic resize from 1200×1200 → 224×224 introduces mild
    anti-aliasing at geometry edges (empirically ~5–6% of pixels take intermediate values
    after resize; verified via direct resize test). So occluding a window sets it to *exactly*
    0.0, which is marginally "cleaner" than a naturally-occurring background patch near an
    edge, but this is a second-order effect, not a distributional violation.
- **Is zeroing equivalent to removing geometry?** Yes, given the above — it is the correct
  operation for this binary domain, unlike occlusion baselines on photographic ImageNet-style
  data where "zero" is an arbitrary color, not a semantic class.

## 3. Target Score and Attribution Semantics

- **Pre-sigmoid logit usage — confirmed.** `get_pre_sigmoid_logit` (`src/xai_engine.py:29-35`)
  computes `logit = W·penultimate + b` directly from the `dense_output` layer weights,
  bypassing the sigmoid. `OcclusionSensitivity.explain` (`src/occlusion.py:153-156, 172-177`)
  uses the identical function on both the original and every occluded forward pass.
- **Sign convention — confirmed correct and consistent.** Model output `preds[:,0] = p_NHS`
  (`XAIEngine.predict`, `src/xai_engine.py:122-128`), so the raw logit is the NHS logit
  `z_NHS`. `S_HS = -z_NHS`, `S_NHS = +z_NHS` is applied identically in `explain_gradcam`
  (`src/xai_engine.py:159-160`), `explain_gradcam_plus_plus` (`:191`), `explain_layercam`
  (`:243`), and `OcclusionSensitivity.explain` (`:156, 177`). All four methods score the
  same target in the same direction — this is a prerequisite for any cross-method
  comparison and it holds.
- **ΔS sign — confirmed correct.** `window_importance = orig_score - occ_scores`
  (`:185`). If occluding a patch *lowers* the target score, ΔS > 0 → that patch was
  supporting the prediction. Interpretation comment in the code (`:182-184`) matches the
  math.
- **FP/FN targeting — confirmed correct and consistent with the CAM comparison.**
  `run_occlusion.py:263` calls `occluder.explain(img_tensor, target_class=pred)` — the
  *predicted* class, not the ground-truth class. The CAM heatmaps generated for the same
  comparison (`run_occlusion.py:279-281`) also use `target_class=pred`. So for an FP
  (actual NHS, predicted HS), all four methods answer "why did the model output HS here,"
  not "why does this look like NHS." This is the right question for the occlusion↔CAM
  comparison as currently built, and there is **no true-class/predicted-class ambiguity
  inside this comparison** — both sides are predicted-class explanations.
  - Note for future work: the *pre-existing* XAI pipeline (`README.md:33`) also generates
    `true_*` heatmaps for the same samples but the occlusion script does not compare against
    those. If a future experiment wants to ask "what geometry would have made this a
    correct HS detection," it must explicitly regenerate occlusion maps with
    `target_class="HS"` (true label) — using the currently-saved `pred_*` CAM images for
    that comparison would silently reintroduce the ambiguity the current script avoids.

## 4. Occlusion Parameter Sensitivity

**Sliding-window geometry** (`_generate_window_coordinates`, `:112-130`): for 224×224,
32×32 window, 16×16 stride: `y_coords = 0,16,...,192` (13 values), same for x → 169
windows, matching the documented and CSV-recorded count exactly. `192 + 32 = 224`, so the
boundary-padding branch (`:121-124`) is never triggered for this config — full coverage
confirmed, no gap at the bottom/right edge.

**Coverage is not uniform.** Interior pixels are covered by up to 2×2 = 4 overlapping
windows (32px window / 16px stride ⇒ 50% overlap); pixels within 16px of any edge are
covered by fewer windows (as few as 1 at the corners). `raw_map = accum/count` correctly
*averages* by actual coverage, so there is no numerical bias — but the **variance of the
estimate is higher at the border** because it rests on fewer independent window
evaluations. This is visible in the diagnostics: `edge_border_concentration` should be
treated as noisier than interior statistics for the same reason. This is worth documenting,
not fixing — coverage-count normalization is the mathematically correct treatment of
non-uniform overlap.

**Sanity experiment performed** (`Part B`, real run, not simulated): 8 held-out iccad3
samples (2 TP / 2 TN / 2 FP / 2 FN, distinct from the 57 used in the main pipeline's
reused xai_diagnostics rows — same set, different config), comparing:

- **A**: window 32/stride 16, baseline 0.0 (production config)
- **B**: window 32/stride 16, baseline = empirical dataset mean pixel value (0.2910,
  measured over 20 train_hs images on the [0,1] scale) — a classic "gray-baseline"
  alternative from the general occlusion literature, tested precisely because it is *not*
  a plausible layout value here (no real pixel in a binary layout is 0.29) and should
  therefore look different if the baseline choice matters.
- **C**: window 16/stride 8, baseline 0.0 (4× the windows, finer grid)

| Comparison | Metric | TP mean | TN mean | FP mean | FN mean |
|---|---|---:|---:|---:|---:|
| A vs. B (baseline) | Pearson | 0.985 | 0.445 | 0.984 | 0.664 |
| A vs. B (baseline) | IoU@top-10% | 0.905 | 0.227 | 0.909 | 0.315 |
| A vs. C (resolution) | Pearson | 0.782 | 0.690 | 0.871 | 0.778 |
| A vs. C (resolution) | IoU@top-20% | 0.503 | 0.492 | 0.538 | 0.573 |
| A vs. C (resolution) | centroid shift (px, of 224) | 4.3 | 12.6 | 5.4 | 12.6 |

Raw per-sample numbers: `results/xai/occlusion/sanity_stability_check.csv` (generated by a
scratch script for this audit, not added to the pipeline).

**Interpretation:**
- For TP and FP, the top attribution region is **stable** under both baseline substitution
  and 2× finer resolution — the network has a strong, spatially concentrated driver of its
  decision, and occlusion finds the same region regardless of these nuisance parameters.
- For TN (and to a lesser extent FN), stability is **materially lower**. This is expected,
  not a bug: for a correctly-rejected NHS sample the model's NHS-favoring evidence is often
  diffuse (many weakly-supporting regions rather than one dominant one), so the *ranking*
  of near-tied regions is sensitive to small parameter changes. Any downstream geometry
  correlation for TN should either use raw signed maps rather than top-k regions, or should
  explicitly report this stability caveat rather than treating the TN peak location as
  meaningful in isolation.
- **Verdict:** the production config (32/16, baseline 0.0) is defensible and does not need
  to change. But TN/FN attribution peaks are less trustworthy for downstream geometry
  claims than TP/FP peaks, and any Part D/E geometry study should stratify by case type
  and report stability alongside the point estimate.

## 5. CAM-vs-Occlusion Comparison Validity

**What is actually being compared** (`compute_map_comparison`, `src/occlusion.py:30-79`,
called from `run_occlusion.py:288-290`): `norm_occ_map` (occlusion, ReLU'd + max-normalized,
native 224×224) against `gcam_h` / `gcam_pp_h` / `lcam_h` (CAM methods, ReLU'd inside
`XAIEngine._postprocess_cam`, max-normalized, bilinear-upsampled from 28×28 to 224×224,
re-normalized after resize). Both sides are predicted-class, both are positive-only,
both are `[0,1]`-normalized, both are 224×224. This part is apples-to-apples.

**Per-image max-normalization does not bias Pearson/Spearman/cosine.** Dividing each map
by its own positive scalar maximum is a per-sample affine (in fact pure scale) transform;
Pearson, Spearman, and cosine similarity are all invariant to independent positive
rescaling of each vector, so the normalization step itself is not a source of correlation
inflation or deflation. It **does** affect **IoU@0.5**, because that threshold is applied
relative to each map's own peak — this is a legitimate and standard way to define "high
attribution," but it means IoU@0.5 conflates *shape* similarity with *peakedness*: a map
with one sharp spike and a map with a broad plateau at the same location can have very
different IoU@0.5 even though both correctly localize the same region. This is a
comparison-metric caveat, not an implementation bug.

**Two things genuinely do affect validity, and should be stated as limitations rather than
fixed in code:**

1. **Resolution mismatch.** The CAM maps' spatial information is intrinsically limited to
   a 28×28 grid (`conv_final_2`, confirmed in `src/model_xai.py:44`) before an 8×
   bilinear upsample; occlusion's map is generated natively at pixel resolution with an
   effective smoothing kernel set by the 32px window / 16px stride (i.e., closer to a
   16–32px box filter). These are *different* smoothing operators producing *different*
   spatial-frequency content, even when both explanations agree on the true region of
   importance. Pixel-wise correlation between differently-smoothed maps will systematically
   under-estimate agreement relative to two maps smoothed identically. This partially
   explains why LayerCAM (which, being pixel-wise-weighted rather than GAP-weighted,
   retains more fine structure before upsampling) correlates more consistently with
   occlusion than Grad-CAM/Grad-CAM++ do (Section-wise Pearson 0.34 vs 0.08 / 0.10, see
   Section 9) — this could be a genuine agreement signal, a resolution-matching artifact,
   or both, and the current data cannot separate the two explanations.
2. **Zero-inflation.** Both sides of the comparison are ReLU-clipped, so both maps are
   dominated by pixels exactly at 0 in low-activation regions. Two mostly-zero vectors
   agreeing on *where the zeros are* inflates Pearson/cosine relative to what either
   metric would report on the (more informative) subset of pixels where at least one
   method claims nonzero importance. This is a known property of comparing sparsified
   attribution maps and is not specific to this codebase, but it means the reported
   correlations should be read as "agreement including trivial background agreement,"
   not as "agreement on where the interesting signal is."

**Conclusion:** the comparison as implemented is internally consistent and not buggy. It
is appropriate as an **exploratory concordance signal between independent explanation
paradigms**, but the specific numeric correlation values should not be read as a
faithfulness score or a ranking of which CAM method is "more correct" — see Section 11.

## 6. ICCAD-12 Geometry Availability

Inspected `iccad-official/iccad{1..5}/{train,test}/{train,test}_{hs,nhs}/*.png` directly.
Confirmed via `PIL`/`numpy`: every sampled image is a single-channel (`L` mode) 1200×1200
raster with exactly two pixel values, `0` and `255` (background / drawn geometry). There is
**no other data file** in the dataset tree — no `.gds`, `.oasis`, `.json`, `.csv`, `.txt`
coordinate or DRC-layer file anywhere under `iccad-official/`. `data.py`
(`dataset_analysis`, `data_extractor`) only ever reads directory structure (for HS/NHS
labels) and calls Keras `ImageDataGenerator` on the PNGs. `README.md`/`BASELINE.md`
describe the pipeline as faithfully reproducing a notebook that also only consumes the
rasterized PNGs.

**Conclusion: the only geometry information available in this repository is the binary
raster itself.** Any geometry-correlation study must be raster-derived (image-processing
based), not polygon-based. This repo cannot support claims phrased in terms of "the DRC
rule violated" or "the polygon that caused the hotspot," because no such ground-truth
annotation exists here — only a binary hotspot/non-hotspot label per image.

## 7. Geometry-Correlation Method

Given raster-only availability, a defensible geometry-extraction pipeline (skimage/
scipy-based, all computable from the binary image alone, no new dataset dependency):

1. **Binarize** the 224×224 preprocessed tensor at 0.5 (already near-binary post-resize).
2. **Connected components** (`scipy.ndimage.label`) on the geometry mask → per-component
   area, bounding box, count.
3. **Distance transform** (`scipy.ndimage.distance_transform_edt`) on the background mask →
   local **spacing** (gap width) at every background pixel; on the geometry mask → local
   **line width** at every geometry pixel (via the medial-axis / skeleton distance value).
4. **Skeletonization** (`skimage.morphology.skeletonize`) → thin centerlines for line-
   segment and junction/corner extraction.
5. **Corner/jog detection**: local curvature or Harris-corner response along the skeleton
   or the boundary contour (`skimage.measure.find_contours` + polygon simplification).
6. **Junction/intersection detection**: skeleton pixels with ≥3 neighbors (standard
   skeleton-graph degree count).
7. **Local feature density**: geometry-pixel fraction in a sliding window matched to the
   occlusion window size (32×32) for direct spatial alignment with the attribution grid.

All of these are legitimate, computable, and should be described as **image-derived
geometric proxies** (narrow gap, thin line, corner density, junction density) — not as
"the DRC violation" or "the lithography defect," since no such physical/EDA-rule ground
truth exists in this repo. This distinction should be stated explicitly in any resulting
report.

## 8. TP/FP/FN/TN Analysis Plan

Stratify every geometry-overlap statistic (Part D metrics) by case type, computed
per-sample then aggregated with a nonparametric interval (Section 9):

- **TP**: does the attribution peak/top-20% region overlap a *narrow-gap* or *thin-line*
  proxy region more than a random or geometry-density-matched control region? (Tests
  whether the model's HS-supporting evidence coincides with the raster features the
  ICCAD-12 literature associates with pinch/spacing violations — described as
  correspondence, not causal confirmation.)
- **TN**: given TN's lower attribution stability (Section 4), report geometry overlap only
  as a secondary/exploratory statistic and pair every point estimate with the stability
  metric from Part B so weak-signal cases aren't over-interpreted.
- **FP**: compare the attribution region's local geometry statistics (line width, gap,
  corner/junction density) against the *matched-benchmark TP population's* geometry
  statistics — i.e., "does this false alarm sit on geometry that numerically resembles true
  hotspot geometry, or is it triggered by something else (e.g., border/edge concentration,
  checked via the existing `edge_border_concentration` diagnostic)?"
- **FN**: check whether the true hotspot's narrow-gap/thin-line region receives *any*
  positive attribution mass (even if not the argmax) — i.e., is the evidence present but
  under-weighted, versus genuinely absent from the model's signal. This directly probes
  recall failure mode.

This plan only requires code that already exists (occlusion, CAM, diagnostics) plus the
new raster-geometry module from Section 7 — no retraining, no new samples beyond the
existing 57 (though a larger stratified sample would strengthen FP/FN in particular, see
Section 12).

## 9. Statistical Analysis Plan

**Current sample sizes are adequate for descriptive/exploratory reporting only.** Per
case-type, per-benchmark cells are as small as n≈2–3 (57 samples ÷ 5 benchmarks ÷ 4 case
types), and even pooled by case type across benchmarks, n=12–15 (confirmed:
`occlusion_diagnostics.csv` value counts — TP 15, TN 15, FP 15, FN 12). At this size:

- Report **medians/IQRs or means ± bootstrap CI**, not p-values. The existing "Overall
  Pearson" table (verified: it is exactly the unweighted mean of the 57 per-sample Pearson
  values — recomputed independently and matches the prompt's numbers to 4 decimals) already
  does the right thing by reporting a mean; it should also report the **spread**. Recomputed
  here: per-sample LayerCAM-Pearson standard deviation is 0.29 (TP), 0.13 (TN), 0.29 (FP),
  0.17 (FN) — i.e. the per-sample variability is comparable to or larger than the
  between-case-type differences in the mean. **The current summary table should not be read
  as showing a reliable ranking of CAM methods without reporting this spread.**
- **Do not compute a Pearson p-value on the 224×224 flattened pixel vectors as if they were
  50,176 independent observations.** Attribution maps are strongly spatially
  autocorrelated (smoothing radius ~16–32px on a 224px image ⇒ effective degrees of freedom
  on the order of hundreds, not tens of thousands). Any future significance test must use
  a spatially-aware method (permutation test with block-shuffled maps, or an effective-N
  correction) — the codebase does not currently attempt this, and that is correct: it is
  better to report no p-value than a misleading one.
- For Part D geometry-overlap statistics, use a **permutation/control-region baseline**
  (randomly placed windows of the same size, matched per-image) rather than a fixed
  theoretical null, since geometry density varies substantially across ICCAD benchmarks
  (Section 6/8) and a naive "25% overlap by chance" assumption is invalid when 30–45% of
  the raster is geometry to begin with (measured: sampled iccad1 images are 29–45% white
  pixels).

## 10. Current Limitations

- Occlusion and CAM maps differ in native spatial resolution/smoothing (Section 5) —
  correlation magnitudes are not directly comparable to a hypothetical "same-resolution"
  ground truth.
- TN/FN occlusion attribution is measurably less stable under baseline/resolution
  perturbation than TP/FP (Section 4) — point-estimate peak locations for these case types
  are weaker evidence.
- No geometry ground truth (polygons/DRC) exists in this repository — all geometry claims
  must be raster-derived proxies, explicitly labeled as such.
- n=57 (and smaller per-stratum) supports descriptive comparison and hypothesis
  *generation*, not hypothesis confirmation.
- Edge-of-image occlusion estimates rest on fewer overlapping windows than interior
  estimates (Section 4) — same direction of bias applies to the CAM `edge_border_concentration`
  diagnostic already in use.

## 11. Claims That Should Be Avoided

Scanned `docs/OCCLUSION.md`, `docs/GRADCAM.md`, `docs/XAI_MODEL_AUDIT.md`,
`docs/XAI_PIPELINE_AUDIT.md`, `README.md`. **The existing documentation is already
appropriately hedged** — both `docs/OCCLUSION.md:136` and `docs/GRADCAM.md:183` explicitly
state the method does *not* simulate physical lithography (diffraction/etch/resist
chemistry). No instance of "proves causality," "X% of the hotspot is explained," "found
the physical hotspot," or an unqualified "more faithful" ranking claim was found in
current docs. This audit's job is therefore to make sure **future** writing (e.g. Part D/E
analyses this audit recommends) stays within the same discipline:

| Avoid | Use instead |
|---|---|
| "Occlusion proves the model uses this feature" | "Occlusion shows this region's removal reduces the target score by ΔS, indicating it supports the prediction" |
| "LayerCAM is more faithful" | "LayerCAM shows higher pixel-wise agreement with occlusion sensitivity (mean Pearson 0.34 vs. 0.08–0.10); this may reflect genuine concordance, closer native resolution to occlusion's grid, or both — not established faithfulness" |
| "X% of the hotspot is explained by this region" | "This region accounts for X% of the total positive attribution mass" |
| "This region causes the lithography failure" | "This region overlaps a narrow-gap/thin-line raster proxy and receives high attribution; no DRC or physical-process ground truth is available in this dataset to confirm causal mechanism" |
| "The attribution map found the hotspot" | "The attribution map is spatially concentrated near [coordinates]; correspondence with layout geometry is reported in Section 8, not causal confirmation" |

## 12. Recommended Next Experiments

Minimum set to make progress on the research question, in order:

1. **Run the Part 7 geometry-extraction module on the existing 57 samples** (no new data
   needed) and compute Part 8's top-10/20/30% overlap statistics against connected
   components, gap width, and corner/junction density, stratified by case type, with the
   permutation-control baseline from Section 9. This directly tests the research question
   using only existing artifacts.
2. **Expand FP/FN sample count** for iccad benchmarks with the fewest FN (12 total FN
   across all 5 benchmarks is thin for a stratified geometry study) — pull additional FP/FN
   samples from the existing test sets (no retraining) using the same diagnostic-CSV
   selection logic already in the pipeline.
3. **Report the raw signed occlusion map (not just ReLU-positive)** for TN/FN cases
   specifically, since Section 4 showed positive-only top-k regions are unstable there;
   the *sign pattern* (which regions suppress vs. support) may be more stable and more
   informative than the peak location.
4. **If geometry correspondence in step 1 is weak or noisy**, do not conclude the XAI
   methods are wrong — first check whether it's an occlusion-window/geometry-feature
   resolution mismatch (32px window vs. much finer gap/line features) before drawing any
   conclusion; consider a second, finer occlusion pass (16/8, already validated as stable
   enough for TP/FP in Section 4) restricted to the geometry study only.

Explicitly **not recommended right now**: adding a 5th XAI method, retraining, or running
occlusion across the full 146,277-image test set — none are needed to answer the stated
research question and all are excluded by the task constraints.

## 13. Retraining Decision

**Not justified.** Every issue found in this audit is about explanation methodology,
comparison-metric interpretation, or missing geometry ground truth — none implicate the
trained CNN weights, none were caused by a training-time bug, and nothing here would be
resolved by retraining.

## 14. Reproducibility

The command documented in `docs/OCCLUSION.md:129`
(`python -m src.run_occlusion --window-size 32 --stride 16 --baseline-value 0.0
--batch-size 64`) was checked against the actual recorded output: `occlusion_diagnostics.csv`'s
`window_size`/`stride`/`baseline_value`/`num_windows` columns are uniformly `32`/`16`/`0.0`/`169`
across all 57 rows — **confirmed consistent with the documented command**, no drift found.

Sanity-experiment script for Section 4 was run from a scratch location (not committed) and
its output is saved at `results/xai/occlusion/sanity_stability_check.csv` for inspection;
it is a one-off diagnostic, not part of the reproducible pipeline, and can be regenerated
by re-running `OcclusionSensitivity` with the three configs listed in Section 4 on the
same 8 iccad3 samples (`sample_id`s: HSCAD31047, HSCAD31012, NHSCAD31_7, NNHSCAD35348_9,
NNHSCAD33591_8, NHSCAD31051_2, HSCAD31075, HSCAD3129).

---

**Filled-in overall cosine similarity table** (Section 9's "0.?" placeholders, computed
from the existing `occlusion_diagnostics.csv`, mean over 57 samples):

| Method | Overall Pearson | Overall Cosine |
|---|---:|---:|
| Grad-CAM | 0.0803 | 0.2810 |
| Grad-CAM++ | 0.1006 | 0.2574 |
| LayerCAM | 0.3363 | 0.5006 |
