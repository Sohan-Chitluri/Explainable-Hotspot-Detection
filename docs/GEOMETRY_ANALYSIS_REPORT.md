# Geometry Analysis Report

Results of the raster-geometry correlation experiment defined in
`docs/RASTER_GEOMETRY_METHOD.md`, run on all 57 representative ICCAD-12 samples (15 TP,
15 TN, 15 FP, 12 FN) across benchmarks 1–5, for all four existing XAI methods (Grad-CAM,
Grad-CAM++, LayerCAM, Occlusion). No retraining, no CNN/XAI code changes, no new XAI
method. Raw per-sample-per-method-per-level results:
`results/xai/geometry/geometry_diagnostics.csv` (684 rows). Aggregates:
`results/xai/geometry/summary.csv`. Per-sample visual panels:
`results/xai/geometry/iccad{1-5}/<sample>/geometry_overlap_panel.png` (57 total).
Summary plots: `results/xai/geometry/plots/`.

**Central question:** *Are high-attribution regions associated with particular
layout-geometric characteristics more strongly than comparable random regions?*
**Answer, qualified:** For most methods and most case types, yes — top-attribution
regions sit on drawn geometry, and specifically on the narrow-gap/thin-line raster
proxies, more than area-matched random controls, with the effect strongest for Occlusion
and LayerCAM. But the effect is method-dependent, case-type-dependent, and in one
documented case (Grad-CAM on TN samples) runs in the **opposite** direction. This is a
correspondence finding, not a causal or physical one — see §12–13.

## 1. Methodology

Full definitions in `docs/RASTER_GEOMETRY_METHOD.md`. Summary: binary raster → geometry
mask (§1) → distance-transform thin-line/narrow-gap proxies (§2) → skeleton → junction
(§4) and Harris-corner (§5) proxies → 32×32 local density (§6). Attribution top-K% masks
are **area-matched by exact pixel count** (not value-thresholded) for K ∈ {10, 20, 30}.
Controls are 30 random draws of non-overlapping 32×32 tiles, area-matched per `(sample,
K)`, **shared across all four methods** for that sample/K (paired design). Nine raw
overlap metrics computed per mask; none collapsed into a single index.

Smoke test (4 samples, one per case type) was run and visually inspected before the full
run — see §"Smoke test findings" below. No feature was found unstable or arbitrary; two
non-fatal limitations were found and are carried into the full-run interpretation.

## 2. Feature Definitions

See `docs/RASTER_GEOMETRY_METHOD.md` §1–§7 for exact formulas. All features are generic
image-processing proxies computed only from the 224×224 binary raster: connected
components, distance-transform width/gap, skeleton, skeleton-degree junctions, Harris
corners, box-filter local density, and boundary distance. None encode DRC rules,
polygon semantics, or lithography physics.

## 3. Control-Region Methodology

Non-overlapping 7×7 grid of 32×32 tiles; `n_tiles = round(target_area / 1024)`; 30 random
subset draws per `(sample, K)`, shared across methods. Achieved control area matches the
target top-K% area to within one tile (≤ ~2% relative area error) in all cases — confirmed
in `geometry_diagnostics.csv` (`target_area_px` vs. `control_area_px`; e.g. top-20% target
10,035 px vs. achieved 10,240 px, a 2.0% excess, constant across all samples since 224×224
and K=20% always round to the same tile count).

## 4. Results

### 4.1 Overall (pooled across case types, top-20%, paired Wilcoxon vs. own control)

| Method | geometry_fraction Δmedian | p | narrow_gap Δmedian | p | thin_line Δmedian | p | corner_recall Δmedian | p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grad-CAM | +0.033 | 0.181 | +0.044 | 0.069 | +0.007 | 0.071 | −0.034 | 0.482 |
| Grad-CAM++ | +0.047 | **0.016** | +0.034 | **<0.001** | +0.014 | **0.002** | +0.020 | **0.032** |
| LayerCAM | +0.108 | **<0.001** | +0.129 | **<0.001** | +0.031 | **<0.001** | +0.187 | **<0.001** |
| Occlusion | +0.164 | **<0.001** | +0.087 | **<0.001** | +0.033 | **<0.001** | +0.027 | 0.096 |

Δmedian = median(attribution-region metric − that sample's own control mean). n=57 per
row (paired). **Junction_recall is omitted from this table**: `n_junction_seeds` is 0 for
the median sample in every case type (see §"Smoke test findings" and §11) — the skeleton
of these layouts (mostly straight parallel lines/spaces) rarely branches, so junction-based
metrics are close to degenerate on this dataset and are not a reliable basis for comparison
here (reported in full in the CSV, not deleted, but not emphasized).

**Reading this table**: Occlusion and LayerCAM show the strongest, most consistent
positive association between top-attribution regions and drawn/narrow/thin geometry
proxies. Grad-CAM++ shows a smaller but still significant positive association.
**Grad-CAM shows no significant overall association** (geometry_fraction p=0.18) — this
overall null is explained by case-type heterogeneity, not a uniformly weak signal (§5).

### 4.2 Resolution/level sensitivity (geometry_fraction, all case types pooled)

| Level | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion | Control |
|---|---:|---:|---:|---:|---:|
| top-10% | 0.235 | 0.235 | 0.346 | 0.421 | 0.214 |
| top-20% | 0.216 | 0.218 | 0.324 | 0.374 | 0.225 |
| top-30% | 0.211 | 0.218 | 0.299 | 0.312 | 0.221 |

The attribution-vs-control gap for Occlusion/LayerCAM shrinks as K grows from 10%→30%
(expected: a larger region regresses toward the image's average geometry density), but
stays positive at all three levels for those two methods. Grad-CAM/Grad-CAM++ track the
control closely at every level.

## 5. TP Analysis

TP geometry_fraction (top-20%, median): Grad-CAM 0.412 (control 0.239, Δ=+0.118,
p=0.022), Grad-CAM++ 0.263 (Δ=+0.029, p=0.934 — not significant), LayerCAM 0.347
(Δ=+0.107, p<0.001), Occlusion 0.419 (Δ=+0.180, p<00.001). **For correctly-detected
hotspots, three of four methods put the top-attribution region on drawn geometry (and
specifically on the narrow-gap/thin-line proxies) significantly more than chance.**
Grad-CAM++'s null result here (despite a positive point estimate) is driven by high
sample-to-sample variance (IQR [0.111, 0.349], spanning below and above the control
median) — a variance finding, not evidence of "no effect," and consistent with
`XAI_RESEARCH_AUDIT.md` §5's Grad-CAM++/occlusion agreement being the weakest and
noisiest of the three CAM pairings.

## 6. TN Analysis

TN geometry_fraction (top-20%, median): Grad-CAM **0.027** (control 0.100, Δ=**−0.079**,
p=0.018 — significantly *below* chance), Grad-CAM++ 0.153 (Δ=+0.047, p=0.169, not
significant), LayerCAM 0.226 (Δ=+0.084, p<0.001), Occlusion 0.272 (Δ=+0.188, p<0.001).

**Grad-CAM's top-attribution region for correctly-rejected (TN) samples is
*anti-correlated* with geometry** — it lands predominantly on background more than a
random region would. This is consistent with, and sharpens, the `XAI_RESEARCH_AUDIT.md`
§4 finding that TN occlusion attribution is diffuse/unstable: here we see the *specific*
mechanism for Grad-CAM — its GAP-pooled channel weighting appears to spread attribution
into open background for confidently-background samples, rather than concentrating on the
sparse geometry that is present. LayerCAM and Occlusion do not show this inversion.

## 7. FP Analysis

FP geometry_fraction (top-20%, median): Grad-CAM 0.416 (Δ=+0.143, p=0.013), Grad-CAM++
0.337 (Δ=+0.071, p=0.073, borderline), LayerCAM 0.387 (Δ=+0.136, p<0.001), Occlusion 0.405
(Δ=+0.131, p<0.001). **All four methods place elevated attribution on drawn geometry for
false-positive samples**, at magnitudes comparable to (in some cases exceeding) the TP
case. This means: the geometry the model reacts to when it incorrectly calls something a
hotspot is **not visually or statistically distinguishable, at this level of analysis,
from the geometry it reacts to when it is correct.** The raster-geometry proxies used here
do not separate genuine hotspot-triggering structure from false-alarm-triggering
structure — a finding about the *limits* of this analysis, not evidence that FPs are
"caused" by the same mechanism as TPs.

## 8. FN Analysis

FN geometry_fraction (top-20%, median): Grad-CAM 0.109 (Δ=−0.016, p=0.424, not
significant — essentially at chance), Grad-CAM++ 0.205 (Δ=+0.035, p=0.042, small),
LayerCAM 0.265 (Δ=+0.112, p<0.001), Occlusion 0.275 (Δ=+0.142, p<0.001). **For missed
hotspots, Grad-CAM's top-attribution region is statistically indistinguishable from a
random region** — consistent with the TN finding (§6): Grad-CAM appears to produce
weak/non-discriminative attribution whenever the model's output leans NHS (whether
correctly, as in TN, or incorrectly, as in FN), regardless of what geometry is actually
present. LayerCAM and Occlusion retain a significant, positive geometry association even
on these missed detections — i.e. there is often geometry-relevant *signal* present in
these two methods' maps even when the classifier's final decision was wrong, which is the
kind of observation that motivates inspecting LayerCAM/Occlusion maps specifically for
FN triage, without claiming they "found the hotspot the model missed."

## 9. Cross-Benchmark Analysis

geometry_fraction (top-20%, median) by benchmark × method, vs. that benchmark's own
control median:

| Benchmark | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion | Control |
|---|---:|---:|---:|---:|---:|
| iccad1 | 0.079 | 0.115 | 0.348 | 0.433 | 0.230 |
| iccad2 | 0.214 | 0.369 | 0.340 | 0.458 | 0.252 |
| iccad3 | 0.321 | 0.410 | 0.456 | 0.399 | 0.209 |
| iccad4 | 0.375 | 0.242 | 0.275 | 0.297 | 0.204 |
| iccad5 | 0.279 | 0.262 | 0.250 | 0.277 | 0.197 |

**iccad1 is an outlier for Grad-CAM**: its attribution regions sit on geometry *far* less
than chance (0.079 vs. control 0.230), while Occlusion and LayerCAM are strongly elevated
on the same benchmark. This is not visible in the pooled table (§4.1) and would be masked
by any single "overall" summary — it is exactly the kind of benchmark-specific
heterogeneity that motivates never collapsing this analysis into one score. Benchmarks
also differ in baseline geometry density (control median ranges 0.197–0.252 across
benchmarks), confirming the per-image control (§3, §9 of the methodology doc) was the
right design choice — a single global control would have conflated benchmark density
differences with attribution behavior.

## 10. XAI-Method Comparison

No overall ranking is produced (by design). Observed, dimension-by-dimension:

- **Geometry correspondence (this experiment)**: Occlusion and LayerCAM show the
  strongest, most consistent positive association with geometry/narrow-gap/thin-line
  proxies across case types and benchmarks. Grad-CAM++ is positive but weaker and
  higher-variance. Grad-CAM is inconsistent — positive for TP/FP, at-chance for FN,
  *below* chance for TN, and behaves as an outlier on iccad1.
- **Perturbation-sensitivity agreement (from `XAI_RESEARCH_AUDIT.md` §5)**: LayerCAM had
  the highest raw Pearson/cosine agreement with Occlusion of the three CAM methods
  (0.336 / 0.501 vs. Grad-CAM 0.080/0.281 and Grad-CAM++ 0.101/0.257) — directionally
  consistent with this experiment's finding that LayerCAM and Occlusion also agree most
  with each other on *where geometry is*.
- **Corner association**: Only LayerCAM (p<0.001) and Grad-CAM++ (p=0.032) show a
  significant positive corner_recall effect; Grad-CAM and Occlusion do not (p=0.48, 0.10).
  This is a genuinely different pattern from the geometry_fraction ranking above — a
  concrete example of why a single combined "geometry relevance score" would have hidden
  a real, method-specific difference.
- **Stability under nuisance-parameter change** (`XAI_RESEARCH_AUDIT.md` §4): Occlusion
  itself was shown to be stable for TP/FP and unstable for TN/FN under baseline/resolution
  changes — the same case-type pattern (TP/FP strong, TN/FN weak) recurs here
  independently, in a completely different analysis (geometry correspondence rather than
  parameter-perturbation), which strengthens confidence that this is a real property of
  what the *model* does on TN/FN samples, not an artifact of one particular metric.
- **Computational cost**: unchanged from prior audits — CAM methods are single
  backward-pass, Occlusion requires 169 forward passes per sample; this experiment adds
  negligible cost on top of already-computed maps (geometry extraction is CPU-only,
  <1s/sample).

## 11. Statistical Limitations

- n=57 total, 12–15 per case type, paired Wilcoxon signed-rank tests used throughout
  (justified: same 57 images measured under both attribution-mask and shared-control
  conditions for every method — a paired, non-parametric test avoids assuming normality
  at this sample size and correctly uses the pairing).
- **Multiple comparisons**: §4.1 alone reports ~16 tests (4 methods × 4 metrics); the full
  CSV supports many more (9 metrics × 4 methods × 3 levels × 5 case-type-or-pooled groups
  ≈ 540 possible tests). No correction (e.g. Holm-Bonferroni) has been applied. Readers
  should treat borderline p-values (e.g. Grad-CAM++ FP p=0.073, TN p=0.169) as suggestive,
  not confirmatory, and treat the strongest results (LayerCAM/Occlusion, p<0.001 across
  nearly every stratification) as the more robust ones precisely because they survive
  every stratification, not because of the p-value magnitude alone.
- **Junction-based metrics are near-degenerate** on this dataset (median `n_junction_seeds`
  = 0 in all four case types) — reported for completeness in the CSV but not treated as
  informative in this report.
- Control draws (n=30) give a stable per-sample control mean/std, but 30 draws is modest
  for the extreme tails (`control_..._min/max` columns in the CSV) — reported means/medians
  are trustworthy, extreme-percentile claims from the control distribution would not be.
- Cross-benchmark comparisons (§9) are descriptive; with only 10–12 samples per benchmark,
  benchmark-level statistics are illustrative, not independently significance-tested here.

## 12. What the Experiment Supports

- Top-attribution regions from Occlusion and LayerCAM are associated with drawn geometry,
  and specifically with the narrow-gap/thin-line raster proxies, significantly more than
  area-matched random regions — consistently across case types (except the expected
  weaker TN/FN pattern) and across all 5 benchmarks.
- This association is **method-dependent**: Grad-CAM does not show it uniformly, and
  shows the *opposite* pattern specifically on TN samples and on the iccad1 benchmark.
- FP samples show geometry association at a similar magnitude to TP samples — the
  raster-geometry proxies used here do not, by themselves, separate correct hotspot
  triggers from false alarms.
- The case-type asymmetry found independently in the occlusion-stability sanity
  experiment (`XAI_RESEARCH_AUDIT.md` §4) is corroborated here by an independent
  geometry-correspondence analysis, strengthening confidence that TN/FN explanations are
  genuinely weaker/less geometry-anchored, not merely noisy under one specific stress test.

## 13. What It Does NOT Support

- **Not** evidence that any raster proxy (narrow gap, thin line, corner) corresponds to a
  verified lithography failure mechanism — no DRC or process ground truth exists in this
  repository (`XAI_RESEARCH_AUDIT.md` §6).
- **Not** a causal claim — "attribution overlaps geometry" is a spatial correspondence
  statistic, not evidence that the geometry *causes* the model's decision, let alone that
  it causes an actual lithography defect.
- **Not** a ranking of which XAI method is "best" or "most faithful" — the four methods
  disagree with each other in informative, dimension-specific ways (§10) that a ranking
  would erase.
- **Not** a resolution of why FPs occur — geometry association alone cannot distinguish
  FP-triggering structure from TP-triggering structure in this dataset.
- **Not** a statistically corrected, publication-ready significance claim — see §11.

## 14. Recommended Next Experiment

1. **Separate FP from TP geometry** using a feature the current proxies don't capture:
   since §7 found FP and TP geometry_fraction indistinguishable, the next step is a
   *discriminative* raster feature — e.g. compare the *distribution* of local width/gap
   values inside the attribution region (not just presence/absence via `thin_line_mask`,
   but the raw `width_map`/`gap_map` value distribution) between TP and FP attribution
   regions, since a genuinely narrower/tighter constriction might separate them even
   though both sit "on geometry."
2. **Investigate the Grad-CAM/iccad1 and Grad-CAM/TN inversions directly**: pull the
   `geometry_overlap_panel.png` images for iccad1 and for TN samples and visually confirm
   whether Grad-CAM's attribution is spreading into background due to the known coarse
   15×15-grid limitation documented in `docs/GRADCAM.md` §9, rather than a geometry-feature
   artifact — this is a cheap, existing-image-based check before any new computation.
3. **Enlarge FN sample count** (currently only 12) if further FN-specific geometry
   analysis is wanted — same rationale as `XAI_RESEARCH_AUDIT.md` §12.
4. Do not add a "combined geometry relevance score," add a fifth XAI method, or retrain —
   none of the findings above call for it.

---

### Smoke test findings (documented per task instructions)

Ran on 4 samples (iccad3, one TP/TN/FP/FN) before the full run. Visual inspection of
`geometry_overlap_panel.png` for each confirmed: `geometry_mask` correctly reproduces
visible raster shapes; `corner_mask` lands on visible line-ends and L-bends, not scattered
noise; attribution top-20% regions visibly differ by method and case type in the expected
directions (e.g. TN Grad-CAM/Grad-CAM++ regions fall mostly on background, matching the
later full-run finding in §6).

Two **non-fatal** limitations were found and are carried into interpretation rather than
blocking the run:

1. **`thin_line_mask`/`narrow_gap_mask` have low within-image discriminating power on
   layouts with near-uniform line width** (e.g. `HSCAD31047`, a set of near-parallel bars
   of similar width): because the threshold is the *within-image* bottom quintile, and
   most of the geometry in such a layout has similar width, the mask ends up marking
   almost the entire visible line/gap rather than a specific pinch point. This is expected
   behavior of a relative per-image threshold given a low-variance layout, not an
   implementation bug — but it means `thin_line_fraction`/`narrow_gap_fraction` are more
   informative on layouts with heterogeneous feature widths than on uniform-pitch ones,
   and this heterogeneity is not currently measured or reported per-sample.
2. **Junction seeds are near-zero across almost the entire dataset** (confirmed at full
   scale in §11) — most ICCAD-12 raster layouts in this sample set are simple parallel
   line/space patterns without true skeleton branch points. Junction-based metrics are
   retained in the CSV for completeness but are not treated as informative in this report.
