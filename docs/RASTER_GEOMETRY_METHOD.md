# Raster-Geometry Correlation: Methodology

This document defines, precisely and in advance of running the full experiment, how
layout "geometry" is extracted from the ICCAD-12 binary raster PNGs and how it is
compared to XAI attribution maps. Per `docs/XAI_RESEARCH_AUDIT.md` §6, **the only data
available in this repository is the rasterized binary layout image** (background = 0,
drawn geometry = 255, confirmed empirically). There are no GDS/OASIS files, no polygon
coordinates, and no DRC-rule annotations anywhere in `iccad-official/`.

**Everything defined below is a raster-derived geometric proxy computed by standard
image-processing operators (`scipy.ndimage`, `skimage.morphology`, `skimage.feature`).
None of it is EDA/DRC ground truth, and none of it is asserted to correspond to a named
lithography failure mechanism.** "Narrow gap" means "small local distance-transform value
in the background," not "a spacing-rule violation." "Corner" means "a Harris-corner
response peak on the binary mask," not "a verified process-critical corner." This
distinction is maintained throughout the implementation and the results report.

## 0. Inputs

- **Raster**: the 224×224×3 `raw_rgb` array already produced by
  `XAIEngine.preprocess_image` (identical preprocessing used throughout the existing XAI
  and occlusion pipelines — same resize, same source PNGs). Converted to grayscale by
  taking the red channel (the source PNGs are single-channel `L` images duplicated across
  RGB by `.convert("RGB")`, so R=G=B).
- **Attribution maps**: for each of the 57 representative samples, the four **predicted-
  class**, max-normalized, `[0,1]`, 224×224 maps already used for the CAM↔Occlusion
  comparison in `run_occlusion.py` — i.e. `engine.explain_gradcam(...)`,
  `explain_gradcam_plus_plus(...)`, `explain_layercam(...)` (regenerated on the fly from
  the checkpoint, exactly as `run_occlusion.py` already does, to avoid depending on
  colormapped PNGs) and `raw_occlusion_map.npy` (already saved by the occlusion pipeline;
  ReLU-clipped and max-renormalized the same way as `OcclusionSensitivity.explain`
  already does internally).

No model is re-trained or modified. No new XAI method is added — the four maps are the
three existing CAM methods plus the existing Occlusion method.

## 1. Connected-Component Extraction

`geometry_mask = grayscale >= 127` (binarize at mid-gray; the source rasters are exactly
binary before the 1200→224 resize, and empirically ≥94% of pixels remain at the true
extremes after resize — see `XAI_RESEARCH_AUDIT.md` §2 — so a 127 threshold reproduces the
pre-resize binary layout with only edge-pixel rounding, which is the standard and only
defensible choice here).

`scipy.ndimage.label(geometry_mask, structure=np.ones((3,3)))` (8-connectivity, since
right-angle Manhattan geometry frequently touches only at a diagonal corner pixel after
rasterization — 8-connectivity avoids artificially splitting a single polygon into
multiple components at such corners). Output: per-image component count and per-component
pixel area. Reported as a raw per-sample scalar (`n_components`, `largest_component_area_frac`),
not folded into any other metric.

## 2. Distance-Transform Gap/Width Features

Using `scipy.ndimage.distance_transform_edt` (Euclidean distance transform):

- **Width map** (thin/thick geometry proxy): `width_map = 2 * EDT(geometry_mask)`,
  restricted to geometry pixels (0 elsewhere). At a geometry pixel, this is the diameter
  of the largest disk centered there that stays inside the geometry — a standard local
  line-width proxy.
- **Gap map** (narrow/wide spacing proxy): `gap_map = 2 * EDT(~geometry_mask)`, restricted
  to background pixels (0 elsewhere). At a background pixel, this is the diameter of the
  largest disk centered there that stays inside the background — a standard local spacing
  proxy.
- **Thin-line mask**: geometry pixels with `0 < width_map <= P20(nonzero width_map)` —
  the bottom quintile of *this image's own* nonzero width distribution. Percentile is
  computed **per image**, not globally, because absolute line width in raster pixels
  varies by benchmark (different original layouts, same 224×224 target size) — a
  per-image relative threshold is the only way to get a comparable "thin relative to this
  layout" definition across benchmarks.
- **Narrow-gap mask**: background pixels with `0 < gap_map <= P20(nonzero gap_map)`,
  same per-image relative-percentile logic.

## 3. Skeleton Extraction

`skimage.morphology.skeletonize(geometry_mask)` — standard topological thinning to a
1-pixel-wide centerline of the geometry mask. Used only as the input to junction detection
(§4); not compared to attribution directly.

## 4. Junction Detection

On the skeleton, compute each skeleton pixel's 8-neighbor count restricted to other
skeleton pixels (`scipy.ndimage.convolve` with a 3×3 ones kernel, minus the center pixel
itself). A skeleton pixel with **≥3** skeleton neighbors is a branch/junction point
(standard skeleton-graph degree criterion — degree 2 is a mid-line point, degree 1 an
endpoint, degree ≥3 a true branching junction). Junction seed pixels are dilated with a
disk of radius 2 (`skimage.morphology.dilation`, disk footprint) to form `junction_mask`,
since a 1-pixel point mask would make area-overlap metrics numerically unstable and a
2px-radius disk is small relative to the 32×32 attribution regions being tested.

## 5. Corner/Turn Detection

`skimage.feature.corner_harris(geometry_mask.astype(float), sigma=1)` computes a generic
Harris corner-response map on the binary mask; `skimage.feature.corner_peaks(response,
min_distance=5)` extracts discrete local-maximum corner coordinates (`min_distance=5px`
avoids redundant detections clustered on the same physical corner). Corner points are
dilated the same way as junctions (disk radius 2) to form `corner_mask`.

**This is a generic image-processing corner detector on a binary mask.** It is explicitly
*not* a claim about lithographic corner-rounding mechanisms; it detects any local
direction change in the drawn geometry's boundary (including on non-critical regions), and
is labeled "raster corner" throughout.

## 6. Local Geometry-Density Features

`density_map = scipy.ndimage.uniform_filter(geometry_mask.astype(float32), size=32)` — the
fraction of geometry pixels in a 32×32 box centered at each pixel. The window size (32)
is chosen to match the occlusion window size already validated in
`XAI_RESEARCH_AUDIT.md`, so density is measured at the same spatial granularity occlusion
sensitivity operates at. This is a continuous `[0,1]` feature, not thresholded into a
mask.

## 7. Definition of "Geometry Region"

Five raster-derived binary/continuous feature layers are defined per sample, all
computed only from the binary raster above:

| Name | Type | Definition |
|---|---|---|
| `geometry_mask` | binary | drawn layout (§1) |
| `narrow_gap_mask` | binary | bottom-quintile local background spacing (§2) |
| `thin_line_mask` | binary | bottom-quintile local geometry width (§2) |
| `junction_mask` | binary | skeleton branch points, dilated (§4) |
| `corner_mask` | binary | Harris corner peaks, dilated (§5) |
| `density_map` | continuous `[0,1]` | local 32×32 geometry density (§6) |

Additionally, `boundary_distance_map = EDT(~find_boundaries(geometry_mask))` —
Euclidean-pixel distance from every pixel in the whole canvas to the nearest
geometry/background boundary (`skimage.segmentation.find_boundaries`) — is used as a
continuous "boundary proximity" feature.

## 8. Definition of Attribution Regions (Top 10/20/30%)

For a normalized `[0,1]` attribution map, the top-K% region is defined as the **exact
top-K% of pixels by value** (area-based, via `argsort` selecting
`round(K/100 * 224 * 224)` highest-valued pixels), not a fixed value threshold (e.g.
"value ≥ 0.7"). This is a deliberate choice: the four XAI methods have very different
value-distribution shapes (Occlusion and LayerCAM tend to be more peaked/sparse after
ReLU-clipping than Grad-CAM/Grad-CAM++ — see `XAI_RESEARCH_AUDIT.md` §5), so a fixed
value threshold would select a different *area* for each method, and any overlap
difference would be confounded by region size rather than reflecting a real difference in
*where* attribution concentrates. An area-matched definition guarantees every method's
top-K% mask has exactly the same pixel count for a given K and sample, so overlap-fraction
differences are attributable to spatial placement, not mask size.

## 9. Randomized Control-Region Generation

The canvas is tiled into a **non-overlapping 7×7 grid of 32×32 tiles** (224 = 7×32 exactly,
zero remainder). This tile size intentionally matches the occlusion window size (§6, §8
rationale) and is a separate, non-overlapping grid from the occlusion pipeline's own
overlapping 169-window grid — used here purely as a source of spatially contiguous, area-
quantized random regions, not for computing occlusion importance.

For a given top-K% mask of pixel-area `A`, the number of control tiles is
`n_tiles = round(A / 1024)` (clipped to `[1, 49]`). **30 independent random draws** (without
replacement, uniform over the 49 tiles) are taken per `(sample, K)` combination. Critically,
**the same 30 draws are reused for all four XAI methods at that `(sample, K)`** — this is a
paired design: each method's attribution-region metrics and the shared control-region
metrics come from the same underlying draw set, which is what makes a paired statistical
comparison (§ Statistical Analysis Plan, `XAI_RESEARCH_AUDIT.md` §9) valid rather than
comparing against independently-noisy control samples per method.

Achieved control area (`n_tiles * 1024`) is stored alongside each draw's target area so
area-matching quality is auditable (typically within ±1 tile / ~2% of the true top-K%
target area).

## 10. Overlap / Association Metrics

For any binary region mask `M` (an attribution top-K mask, or one control draw), the
following **raw, non-aggregated** metrics are computed against the geometry layers from
§7 and stored as separate columns — no single "geometry relevance score" is computed:

1. `geometry_fraction` — `|M ∩ geometry_mask| / |M|`: fraction of the region's pixels
   that sit on drawn geometry (vs. background).
2. `narrow_gap_fraction` — `|M ∩ narrow_gap_mask| / |M|`.
3. `thin_line_fraction` — `|M ∩ thin_line_mask| / |M|`.
4. `boundary_proximity_mean` — mean of `boundary_distance_map` restricted to `M` (pixels;
   lower = closer to a geometry/background boundary on average).
5. `junction_recall` — `|M ∩ junction_mask| / |junction_mask|` (fraction of *all* junction
   pixels in the image that fall inside this region).
6. `junction_density_per_kpx` — junction pixels inside `M`, per 1,000 pixels of `M`
   (scale-normalized density, comparable across masks of different area).
7. `corner_recall` — analogous to (5) for `corner_mask`.
8. `corner_density_per_kpx` — analogous to (6) for `corner_mask`.
9. `local_density_mean` — mean of `density_map` restricted to `M`.

These nine numbers are computed once per `(sample, method, K)` for the attribution mask,
and once per `(sample, K, control draw)` for each of the 30 control masks (then summarized
as `control_<metric>_mean` / `control_<metric>_std` / `_min` / `_max` per `(sample, K)`,
shared across the four methods per the paired design in §9). All are kept as separate raw
columns in `geometry_diagnostics.csv` — never combined into a single index.

## 11. Statistical Handling (summary)

- Report **sample-level distributions** (median, IQR, std), not just means, stratified by
  case type (TP/TN/FP/FN) and benchmark.
- Because all four methods and the control set are evaluated on the **same 57 images**,
  method-vs-control and method-vs-method comparisons are **paired**. Use the
  **Wilcoxon signed-rank test** (paired, distribution-free — appropriate given the small
  n and no assumption of normality) on `(method_metric - control_mean_metric)` per sample,
  reported per `(metric, method, K)` combination, with an explicit multiple-comparisons
  caveat (9 metrics × 4 methods × 3 K-levels = 108 tests) rather than a single omnibus
  claim.
- No p-value is computed on flattened per-pixel vectors (the autocorrelation problem
  identified in `XAI_RESEARCH_AUDIT.md` §9) — all statistics here operate at the
  **one-summary-number-per-sample** level, which does not have that problem.

## 12. Smoke Test Procedure

Before running all 57 samples, the full pipeline (geometry extraction, mask/metric
computation, control generation, 4-panel visualization) is run on **4 samples**
(one TP, one TN, one FP, one FN), and the visual panels are inspected for:

- Does `geometry_mask` visibly reproduce the raster layout shapes?
- Does `thin_line_mask` / `narrow_gap_mask` land on visibly thin/narrow raster structures
  rather than looking like uniform noise?
- Does `junction_mask` / `corner_mask` land on visible branch points / direction changes
  rather than scattering arbitrarily?
- Is the top-K% attribution mask area exactly matched across methods for the same K?

If any feature looks arbitrary, unstable, or disconnected from visible raster structure,
the smoke test stops and the problem is documented rather than proceeding to the full run.
