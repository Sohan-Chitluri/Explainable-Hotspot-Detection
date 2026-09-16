# TP vs FP Geometry Analysis

Follow-up to `docs/GEOMETRY_ANALYSIS_REPORT.md` §14's recommended next experiment: (A)
compare raw raster-geometry value distributions inside TP vs FP attribution regions to
look for a discriminating feature, and (B) visually/diagnostically audit the Grad-CAM
iccad1 and Grad-CAM/TN anomalies. No retraining, no CNN/XAI code changes were made. One
concrete implementation gap was found and fixed **in the new analysis code only** (§9);
the frozen CNN and the existing Grad-CAM/Grad-CAM++/LayerCAM/Occlusion implementations
were left untouched.

**Headline result, stated up front**: the search for a discriminating geometry feature
(Part A–D) came back **negative** — no raw geometry feature reliably separates TP from FP
attribution regions in this dataset. The audit (Part E–F) instead surfaced a more
consequential, unexpected finding: a subset of samples — concentrated in the iccad5
checkpoint — produce **model output that is measurably insensitive to the input image**
(confirmed by three independent probes: exact-zero gradient, exact-zero occlusion
sensitivity, and identical predicted probabilities across different images). This is a
property of the trained models, not of the XAI methods, and it materially affects how
Grad-CAM/Grad-CAM++ results (including in the prior two reports) should be read for the
affected samples.

## 1. Objective

Determine whether any of the four raw raster-geometry proxies (local line width, local
gap/spacing, local density, boundary distance) inside XAI top-K% attribution regions
differs systematically between correctly-detected hotspots (TP) and false alarms (FP),
for any of the four XAI methods, at any of the three attribution levels — and separately,
establish an evidence-based (not speculative) explanation for the Grad-CAM anomalies
identified in the prior report (iccad1 benchmark-wide low geometry correspondence;
inconsistent/inverted TN behavior).

## 2. Existing Geometry Definitions (reused, not reinvented)

All four raw features come directly from `src/geometry_features.py` /
`docs/RASTER_GEOMETRY_METHOD.md` §2, §6, computed once per sample and then aggregated
*inside* each attribution mask rather than reported as a binary in/out fraction:

- `mean_local_width_px` — mean of the distance-transform **width map** (§2 of the method
  doc: `2 × EDT(geometry_mask)`), restricted to the geometry pixels that fall inside the
  attribution mask. Undefined (`NaN`, excluded from stats) if the mask contains no
  geometry pixels.
- `mean_local_gap_px` — mean of the **gap map** (`2 × EDT(background_mask)`), restricted
  to background pixels inside the mask. Same NaN handling.
- `mean_local_density` — mean of the existing 32×32 `density_map` (§6) inside the mask
  (defined everywhere, no NaN).
- `mean_boundary_distance_px` — mean of the existing `boundary_distance_map` inside the
  mask (defined everywhere).

Attribution top-K% masks (K = 10, 20, 30) use the identical area-matched definition from
`docs/RASTER_GEOMETRY_METHOD.md` §8 (exact top-K% of pixels by value, same code path,
`topk_mask_by_area`). No new geometric definitions were introduced.

## 3. Sampling Procedure

All 15 TP and 15 FP samples from the existing 57-sample representative set (same samples
used throughout `XAI_RESEARCH_AUDIT.md` and `GEOMETRY_ANALYSIS_REPORT.md`), all 5
benchmarks, all 4 XAI methods, all 3 levels: 30 samples × 4 methods × 3 levels = 360 rows,
saved raw (one row per sample/method/level, the appropriate unit of observation per Part B
— **not** per-pixel) to `results/xai/geometry/tp_fp/tp_fp_geometry_statistics.csv`.

## 4. Statistical Methodology

- **Unit of analysis**: one summary value per (sample, method, level) — i.e. each of the
  15 TP and 15 FP images contributes exactly one number per feature per method per level.
  This avoids pseudoreplication; pixel-level values are aggregated (via the geometry
  feature maps' own per-pixel means, §2) before any comparison is made.
- **Test**: TP and FP are two independent (unpaired) sets of images, so a **Mann-Whitney
  U test** (two-sided, non-parametric — justified given n=15 per group and no normality
  assumption) is used per (method, level, feature) combination.
- **Effect size**: reported as both an **AUC** (probability that a randomly drawn TP value
  exceeds a randomly drawn FP value) and **Cliff's delta** (`= 2·AUC − 1`, range [−1, 1],
  0 = no separation). Because the Mann-Whitney U statistic is mathematically identical to
  this AUC (`AUC = U / (n_TP · n_FP)`), **no threshold was fit and no cross-validation was
  needed** — this is a closed-form rank statistic, not a trained classifier, so there is no
  overfitting risk to guard against for a single univariate feature.
- **Multiple comparisons**: 4 methods × 4 features × 3 levels = 48 tests in the primary
  (all-samples) pass, run again after excluding degenerate maps (§9) = 96 tests total. No
  correction was applied; results are interpreted with this in mind (§8, §12).
- All summary statistics (median, IQR, mean, std, n) are reported per group, not just
  means, in `tp_fp_sample_statistics.csv` / `tp_fp_sample_statistics_excl_degenerate.csv`.

## 5. TP Results

At top-20%, across all four methods, TP `mean_local_width_px` medians cluster tightly
(5.20–5.45 px), `mean_local_gap_px` medians range 8.3–18.9 px (method-dependent — driven
by mask placement differences between methods, not a TP-specific signature), local density
medians 0.257–0.371, boundary distance medians 2.9–7.5 px. None of these values in
isolation are diagnostic of anything — they only become informative relative to the FP
comparison group, in §7.

## 6. FP Results

FP medians at top-20% are close to the TP medians for every feature and every method
(e.g. Grad-CAM width 5.28 px FP vs. 5.30 px TP; Occlusion density 0.378 FP vs. 0.371 TP).
Visually confirmed in `results/xai/geometry/tp_fp/plots/*_top20_violin.png`: the TP and FP
violins overlap substantially for all four features and all four methods, with no
consistent separation in either direction.

## 7. XAI-Method Comparison (discriminative power)

**Full-sample results** (all 360 rows, including degenerate/flat attribution maps),
top-20%, from `tp_fp_sample_statistics.csv`:

| Method | Feature | AUC (TP vs FP) | Cliff's δ | p |
|---|---|---:|---:|---:|
| Grad-CAM | width | 0.613 | +0.227 | 0.300 |
| Grad-CAM | gap | 0.609 | +0.218 | 0.320 |
| Grad-CAM | density | 0.356 | −0.289 | 0.184 |
| Grad-CAM | boundary dist. | 0.618 | +0.236 | 0.281 |
| Grad-CAM++ | width | 0.413 | −0.173 | 0.431 |
| Grad-CAM++ | gap | 0.600 | +0.200 | 0.361 |
| Grad-CAM++ | density | 0.356 | −0.289 | 0.184 |
| Grad-CAM++ | boundary dist. | 0.618 | +0.236 | 0.281 |
| LayerCAM | width | 0.436 | −0.129 | 0.561 |
| LayerCAM | gap | 0.551 | +0.102 | 0.648 |
| LayerCAM | density | 0.316 | −0.369 | 0.089 |
| LayerCAM | boundary dist. | 0.560 | +0.120 | 0.590 |
| Occlusion | width | 0.480 | −0.040 | 0.868 |
| Occlusion | gap | 0.462 | −0.076 | 0.740 |
| Occlusion | density | 0.476 | −0.049 | 0.836 |
| Occlusion | boundary dist. | 0.476 | −0.049 | 0.836 |

**No feature reaches p<0.05 for any method at top-20%** (or at top-10% / top-30%, §8; full
table in the CSV). AUCs range 0.32–0.64 — by conventional Cliff's-delta thresholds
(negligible <0.15, small <0.33, medium <0.47, large ≥0.47) every effect is negligible-to-
small. **`mean_local_density` shows the most consistent direction** across Grad-CAM,
Grad-CAM++, and LayerCAM (δ = −0.29 to −0.37, i.e. FP regions are *slightly* denser than
TP regions) but does not reach significance at n=15 per group at any level. Occlusion
shows the weakest effects of all four methods (|δ| ≤ 0.08 for every feature) — its
attribution regions' raw geometry statistics are essentially indistinguishable between TP
and FP.

## 8. Threshold Sensitivity

Re-running at top-10% and top-30% (see CSV) shows the same qualitative picture: no
feature/method combination is significant at any level, and the `mean_local_density`
direction (FP slightly denser) is the only pattern that recurs across levels, though it
never reaches significance at any level either (best case: Grad-CAM top-30%,
δ=−0.36, p=0.101). **After excluding degenerate maps (§9)**, one nominally significant
result appears — Grad-CAM++, `mean_local_density`, **top-10% only**, δ=−0.62, p=0.025 (n=10
TP, 9 FP) — but it does **not** replicate at top-20% (p=0.055) or top-30%, and no other
method shows it at any level. Per the task's own instruction to flag single-threshold
results: **this is flagged as very likely a false positive** given (a) it appears at
exactly one of three levels, (b) it appears for exactly one of four methods, (c) ~48–96
tests were run without multiple-comparison correction (≈2–5 false positives expected by
chance at α=0.05), and (d) the sample size at that stratum (n=10/9) is small. It is
reported for completeness, not as a discriminating feature.

## 9. A Concrete Implementation Finding: Degenerate Attribution Maps

While building the TP/FP dataset, `is_degenerate_map` (flagging attribution maps with
`np.ptp(map) < 1e-6`, i.e. perfectly flat) was added to `src/run_tp_fp_geometry.py`,
because `topk_mask_by_area` (§8 of the methodology doc) has no guard for a flat input: on
a perfectly flat map, `argpartition` still returns *some* top-K set of pixel indices, but
it is an arbitrary tie-break (in practice, a spatially biased, low-index-first region), not
a meaningful attribution region. **This is a real gap in the new geometry-analysis code**
(not in the pre-existing Grad-CAM/Occlusion implementations, which already guard against
this — `compute_map_comparison` in `src/occlusion.py` correctly returns 0 correlation when
a map's standard deviation is below `1e-8`). It was fixed by flagging (not silently
dropping) degenerate rows and re-running the TP/FP group statistics both with and without
them (`tp_fp_sample_statistics.csv` vs. `_excl_degenerate.csv`); §7–8 report the
all-samples numbers, and the exclusion did not change the qualitative conclusion (no
significant discriminating feature).

**Scale of the issue**: 78/360 rows (21.7%) in the TP/FP dataset came from degenerate
maps: Grad-CAM 16.7% of rows, Grad-CAM++ 36.7%, LayerCAM 16.7%, Occlusion 16.7%. Extending
the same check to the full 57-sample set used in `xai_diagnostics.csv` (via its existing
`gcam_spread`/`gcam_pp_spread`/`layercam_spread` columns, all already computed and just
not previously cross-checked for exact-zero clustering) finds:

- **Grad-CAM**: 18/57 samples (31.6%) produce an exactly-flat map.
- **Grad-CAM++**: 24/57 (42.1%).
- **LayerCAM**: 5/57 (8.8%) — concentrated entirely in iccad5 TP/FP.

**This means roughly a third to two-fifths of the Grad-CAM/Grad-CAM++ samples in the
prior `GEOMETRY_ANALYSIS_REPORT.md` and the occlusion-correlation table in
`XAI_RESEARCH_AUDIT.md` contributed an arbitrary or trivially-zero result rather than a
genuine attribution signal.** For the correlation table specifically, this is *already*
handled correctly (§F below) — but for the geometry top-K% overlap analysis in the prior
report, it was not checked for and is a genuine, previously-undocumented caveat on those
results, most concentrated in the TN/FN case types (where confident-NHS predictions
saturate, §10) and in iccad5 (§10).

## 10. Grad-CAM ICCAD1 Investigation

Re-examined the iccad1 TP/FP samples that drove the `GEOMETRY_ANALYSIS_REPORT.md` §9
anomaly (Grad-CAM geometry_fraction 0.079 vs. control 0.230, benchmark-wide). Diagnostic
check via `xai_diagnostics.csv`'s existing `gcam_spread`/`gcam_area_gt50` columns (no
degeneracy: iccad1 TP/FP `gcam_spread` = 82.0–87.0 px, `gcam_area_gt50` = 0.13–0.26, both
comfortably nonzero and in the same range as other benchmarks' non-degenerate samples) —
**iccad1's Grad-CAM maps are not degenerate**; this is a distinct phenomenon from §9.

What was found: iccad1's Grad-CAM `gcam_spread` (a weighted standard deviation from the
attribution centroid) of ~82–87 px on a 224 px canvas is a genuinely **broad/diffuse**
map — roughly 37–39% of the image width as a one-sigma spread — for TP and FP alike, and
the centroid sits close to the geometric image center (median distance from center
4.9 px for iccad1 TP, vs. 8.3/5.3/3.8 px for iccad2/3/4 TP respectively — i.e. not more
centered than other benchmarks, so centering alone doesn't explain it). This is consistent
with (but does not conclusively prove) a **broad/global activation** pattern specific to
this checkpoint: the top-20% attribution mask, drawn from a wide, low-contrast map, ends up
sampling more of the image's open background than a sharper map would, dragging
geometry_fraction down. **No output-collapse or zero-gradient evidence was found for
iccad1** (unlike iccad5, §11) — the prediction confidence (p_hs≈0.75–0.78 for TP/FP) and
gradient are unremarkable. The evidence supports "broad/diffuse, non-degenerate
activation" as the mechanism; it does **not** establish *why* iccad1's checkpoint produces
broader activation than the others (that would require inspecting this checkpoint's
learned filters, which is out of scope here and would not be resolved without retraining,
which is excluded by the task constraints).

## 11. Grad-CAM TN Investigation

Two distinct mechanisms were found and are kept separate rather than merged into one
explanation:

**(a) Genuine output collapse (iccad5, most severe).** Direct forward/backward-pass probes
(not just the pre-computed diagnostics) on `HSCAD50`, `HSCAD51`, `HSCAD510` (TP) and
`NHSCAD50_8`, `NHSCAD51_1` (FP) — chosen because all four XAI methods flagged them as
degenerate simultaneously (§9) — found:
  - `tf.GradientTape` gradient of the pre-sigmoid target score w.r.t. `conv_final_2` is
    **exactly 0.0 at every element** (confirmed numerically, not just via the downstream
    `spread==0` diagnostic), even though the target score itself is a real, nonzero value
    (0.333) and `conv_final_2`'s own output is not degenerate (min −8.76, max 8.39, no dead
    channels at that layer itself — `conv_final_2` has `activation=None` per
    `model_xai.py`, so the cut-off is downstream of it).
  - **Occlusion sensitivity is independently, exactly flat too** for these samples (the
    reason they appear in the occlusion `is_degenerate_map` column, §9) — this is a
    forward-pass-only probe with no gradient involved, so this is not a gradient-tape
    artifact; the model's *output* genuinely does not change when 32×32 regions of these
    specific images are zeroed out.
  - **All five samples predict the exact same probability**, `p_hs = 0.5825` to 4 decimal
    places, despite having visibly different layouts and different raster geometry
    densities (0.214–0.312 geometry fraction, confirmed by direct pixel counting — these
    are not duplicate images).
  - The pattern recurs at the opposite extreme: iccad5's 3 TN samples and its 1 FN sample
    all predict `p_hs = 0.0000` exactly.
  - **Conclusion, evidence-based**: for at least 9 of iccad5's 10 representative samples,
    the model's output is measurably **input-invariant** — it lands on one of only two
    discrete values regardless of the specific layout content. This is a property of the
    trained `xai_cnn_iccad5.keras` checkpoint, not of any XAI method: when a model's
    decision genuinely does not depend on spatial input content, no gradient-based or
    perturbation-based explanation method can recover a meaningful spatial explanation for
    that decision, because there is no spatial dependence to recover. **This is reported as
    a finding about the model, not a defect in Grad-CAM, Grad-CAM++, LayerCAM, or
    Occlusion** — all four methods are behaving correctly by returning near-zero/flat
    output when there is genuinely nothing spatially informative to attribute. No
    retraining was performed per the task constraint; this is flagged for the user's
    awareness rather than acted on.
- **(b) Saturated-logit partial gradient starvation (iccad3 TN, less severe).** A separate
  direct probe on `NHSCAD31_7` (iccad3 TN, Grad-CAM-degenerate but **not**
  Occlusion-degenerate and **not** LayerCAM-degenerate) found: pre-sigmoid NHS logit =
  **21.1** (compare to iccad1 TN's 3.13) — an extremely saturated, high-confidence
  prediction — with 75% of `conv_final_2` gradient elements exactly zero and the remaining
  25% evidently near-canceling under Grad-CAM's global-average-pooling channel weighting
  (enough to push the GAP-weighted CAM to ≤0 everywhere, tripping the `max_v > 0` guard in
  `XAIEngine._postprocess_cam`), while LayerCAM's **pixel-wise** weighting (no spatial
  averaging step) still yields a nonzero map from the same 25% non-zero gradient elements.
  This is a plausible, partially-evidenced mechanism (**broad/global activation** +
  **GAP-specific cancellation**, both from the candidate-cause list) for why Grad-CAM
  specifically (more than Grad-CAM++, and much more than LayerCAM) is disproportionately
  degenerate on very-high-confidence predictions — consistent with Grad-CAM++/Grad-CAM
  both degenerating far more often (31.6%/42.1% of all 57 samples) than LayerCAM (8.8%,
  and only in iccad5's already-established output-collapse cluster). This mechanism is
  reported with the caveat that only one sample was directly probed at this level of
  detail; it is offered as the most evidence-consistent explanation available, not a
  fully exhaustive causal proof.

## 12. Limitations

- TP/FP group sizes (n=15 each) are small; §7–8's null result is a failure to detect a
  difference at this sample size, not proof that no difference exists at any sample size.
- 48–96 hypothesis tests were run without multiple-comparison correction; the one
  nominally significant result (§8) is flagged as likely spurious rather than treated as a
  finding.
- The degenerate-map issue (§9) was fixed in the TP/FP analysis but **not retroactively
  re-run for the full 57-sample geometry report** — `GEOMETRY_ANALYSIS_REPORT.md`'s Grad-
  CAM/Grad-CAM++ statistics should be read with this caveat until that pipeline is re-run
  with the same degenerate-map flag (recommended in §14).
- The iccad1 mechanism (§10) is evidence-consistent but not conclusively isolated; the
  iccad5 output-collapse mechanism (§11a) is directly confirmed by three independent
  probes and is the strongest-evidenced finding in this report; the iccad3 mechanism
  (§11b) is based on one directly-probed sample plus consistent aggregate statistics.
- All geometry features remain raster-derived proxies (per `XAI_RESEARCH_AUDIT.md` §6);
  none of this analysis touches or infers DRC/physical mechanism information.

## 13. Conclusions Supported by Evidence

- Attribution regions were associated with raster-derived geometric characteristics
  broadly (per `GEOMETRY_ANALYSIS_REPORT.md`), but TP and FP attribution regions were
  **not** found to differ in their raw width/gap/density/boundary-distance distributions
  at n=15 per group, for any of the four methods, at any of three attribution levels.
- `mean_local_density` is the only feature showing a consistent (if non-significant)
  directional trend (FP attribution regions slightly denser than TP) across Grad-CAM,
  Grad-CAM++, and LayerCAM.
- A subset of the iccad5 checkpoint's predictions (confirmed for at least 5 samples, with
  9/10 total samples falling into one of two output clusters) are input-invariant,
  confirmed independently by gradient, occlusion-sensitivity, and identical-probability
  evidence.
- Grad-CAM's disproportionate degeneracy rate (31.6% of 57 samples) relative to LayerCAM's
  (8.8%) is consistent with a GAP-averaging cancellation mechanism, directly observed in
  one high-confidence iccad3 TN sample.

## 14. Conclusions NOT Supported

- **Not** supported: that any raw geometry feature discriminates TP from FP attribution
  regions in this dataset — the evidence is a null result, reported as such.
- **Not** supported: that iccad5's output collapse "causes" false positives/negatives in
  any causal-mechanism sense — it is observed to correlate with degenerate XAI maps, not
  shown to explain classification errors generally (only 1 of iccad5's 3 FP samples showed
  this pattern; the other FP, `NNHSCAD5994_4`, had a normal, non-degenerate gradient and
  a distinct predicted probability, 0.5204).
- **Not** supported: any claim that Grad-CAM, Grad-CAM++, LayerCAM, or Occlusion is more
  "faithful" than another — the method differences documented here (degeneracy rate,
  geometry association) describe different failure/success patterns under different
  conditions, not a general quality ranking.
- **Not** supported: any lithography-mechanism or DRC-violation claim.

## Part F — Re-examination of the Occlusion↔CAM Correlation Analysis

Re-computed directly from `results/xai/occlusion/occlusion_diagnostics.csv` (57 rows),
reporting full distributions rather than only the mean:

| Method | mean | **median** | std | IQR | min | max |
|---|---:|---:|---:|---:|---:|---:|
| Grad-CAM | +0.080 | **0.000** | 0.293 | [0.000, 0.281] | −0.587 | +0.595 |
| Grad-CAM++ | +0.101 | **0.000** | 0.234 | [0.000, 0.210] | −0.263 | +0.645 |
| LayerCAM | +0.336 | **0.305** | 0.267 | [0.092, 0.614] | −0.083 | +0.817 |

**The Grad-CAM and Grad-CAM++ medians are exactly 0.000** — i.e. the *majority* of samples
have literally zero measured correlation with Occlusion, not a "weak positive" correlation.
This is the direct, expected consequence of the degenerate-map finding in §9: when either
map being compared is flat, `compute_map_comparison` (`src/occlusion.py`) correctly
returns `pearson_corr = 0.0` rather than an undefined or spurious value — this guard was
already correctly implemented in the existing code and required no fix. **This changes the
appropriate reading of the original `XAI_RESEARCH_AUDIT.md` §4.1/§9 mean-based summary**:
Grad-CAM's mean Pearson of 0.08 is not "a small positive correlation typical of most
samples" — it is "zero for the majority, and meaningfully positive (up to +0.595) for a
minority" — a materially different, and now evidence-backed, characterization.

**By benchmark** (median Pearson):

| Benchmark | Grad-CAM | Grad-CAM++ | LayerCAM |
|---|---:|---:|---:|
| iccad1 | 0.078 | −0.031 | 0.331 |
| iccad2 | 0.094 | 0.164 | 0.535 |
| iccad3 | 0.141 | 0.105 | 0.539 |
| iccad4 | 0.058 | 0.000 | 0.198 |
| iccad5 | **0.000** | **0.000** | **0.004** |

iccad5 shows median ≈ 0 for **all three** CAM methods — directly consistent with §11a's
output-collapse finding for that checkpoint (LayerCAM's median isn't quite exact-0, since
it degenerates on fewer iccad5 samples than Grad-CAM/Grad-CAM++ do, §9).

**Is the LayerCAM mean driven by a few outliers?** No: the 5 lowest and 5 highest
LayerCAM-Pearson samples span multiple benchmarks and case types (lowest:
iccad2/iccad4/iccad5 FN/TN/TP/FP mixed; highest: iccad2/iccad3 TP/FP mixed) — not
concentrated in one benchmark or case type, so the positive LayerCAM mean/median reflects a
broadly-distributed pattern, not a handful of samples pulling the average up.

**Metric appropriateness, reaffirmed**: Spearman (median 0.253, IQR [0.072, 0.375]) and
cosine (median 0.496, IQR [0.350, 0.696]) for LayerCAM tell a qualitatively consistent
story to Pearson (positive, moderate, broadly distributed) — the three metrics are not in
conflict. `fraction_above_050` (a zero-heaviness proxy for the occlusion maps themselves)
has median 0.056 (IQR [0.036, 0.097]) — i.e. the *positive* attribution maps compared here
are themselves fairly sparse (only ~5–10% of pixels above half-max), reaffirming the §5
zero-inflation caveat from `XAI_RESEARCH_AUDIT.md`: agreement partly reflects the two
sparse maps agreeing on *where nothing is happening*, not exclusively on where the
important region is. No new statistic overturns the original audit's qualitative
conclusion (LayerCAM/Occlusion agree most, Grad-CAM/Grad-CAM++ agree least) — this section
adds the degenerate-map mechanism as the concrete, evidence-based reason why, rather than
leaving it as an open question.

## Final Question

**"Do the current experiments provide evidence that XAI-attributed regions contain
systematically different raster-derived geometric characteristics in true-positive versus
false-positive cases?"**

**No.** Across four raw geometry features (local width, local gap, local density, boundary
distance), four XAI methods, and three attribution levels (48 tests, repeated after
excluding degenerate maps for 96 total), no method/feature/level combination showed a
statistically significant, threshold-consistent difference between TP and FP attribution
regions. The one nominal significance found (Grad-CAM++, density, top-10% only, p=0.025) did
not replicate at other levels and is attributed to multiple-comparisons noise (§8). The
closest thing to a consistent signal is a **non-significant** directional trend in
`mean_local_density` (FP slightly denser than TP) for Grad-CAM, Grad-CAM++, and LayerCAM
(Cliff's δ −0.29 to −0.37 at top-20%), which did not reach significance at n=15 per group at
any level for any method. **What's missing to answer this more conclusively**: a larger
TP/FP sample (current n=15 per group has limited power to detect anything smaller than a
large effect), and/or a feature not yet tested here (e.g. the *distribution shape* of
width/gap values within the region rather than their mean, or a feature combining multiple
proxies) — both flagged as candidate next steps, not run here to avoid searching for
significance post hoc.

**Should the CNN/XAI pipeline remain frozen, or has a concrete methodological problem been
found?** The CNN and the four existing XAI implementations should **remain frozen** — no
bug was found in them; the degenerate-map behavior in Grad-CAM/Grad-CAM++/LayerCAM is a
mathematically correct consequence of genuinely zero or near-zero gradients (in one
directly-probed case, confirmed to trace to a real, extreme logit-saturation /
gradient-starvation phenomenon in the trained network itself, not a coding defect), and
`compute_map_comparison`'s existing degenerate-map guard is correct and required no
change. **One concrete, fixable methodological gap *was* found and fixed**, but only in
this analysis's own new code: `topk_mask_by_area` (used by the geometry-correlation
pipeline) silently produces an arbitrary region for a flat attribution map. This has been
flagged (`is_degenerate_map`) and excluded in a parallel re-analysis here (§9); it is
**recommended, not yet performed,** that `src/run_geometry_analysis.py` (the full
57-sample pipeline behind `GEOMETRY_ANALYSIS_REPORT.md`) be re-run with the same flag
before treating its Grad-CAM/Grad-CAM++ statistics as final (§14 below).

## Recommended Next Experiment

1. **Re-run `src/run_geometry_analysis.py` (the full 57-sample pipeline) with the
   `is_degenerate_map` flag added**, reporting Grad-CAM/Grad-CAM++ geometry statistics
   separately for degenerate vs. non-degenerate samples, since §9 shows this affects 31.6%/
   42.1% of samples respectively.
2. **Directly forward/backward-probe the remaining iccad5 samples and the iccad3/iccad4 FN
   clusters** (the same technique used in §11) to determine what fraction of the *entire*
   57-sample set — not just the 5 samples checked here — shows genuine output-invariance,
   and whether iccad5's held-out accuracy numbers in `README.md` are being driven
   disproportionately by this collapsed-output cluster (a data-quality question, answerable
   without retraining, by comparing accuracy on the collapsed-output subset vs. the rest of
   iccad5's test set).
3. If a larger discriminative-feature search is wanted, **expand the TP/FP sample count**
   beyond the current 15/15 (same rationale as prior reports) before re-testing — current
   n is likely underpowered for anything but a large effect.
4. Do **not** retrain, and do not add a new XAI method — none of the findings here would be
   resolved by either.
