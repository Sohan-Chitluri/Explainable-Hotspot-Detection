# Final ICCAD5 XAI Synthesis (balanced_sampling seed1, frozen checkpoint)

## 0. Scope and provenance

This document synthesizes the three final-evidence artifacts generated against the
**frozen** checkpoint `models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras`
(sha256 `9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70`, seed 101, balanced
accuracy 0.9743):

- `results/xai/final_iccad5/xai_diagnostics.csv` — 11 representative samples (TP=3, TN=3,
  FP=3, FN=2), Grad-CAM / Grad-CAM++ / LayerCAM attribution + prediction metadata.
- `results/xai/final_iccad5/occlusion/occlusion_diagnostics.csv` — occlusion sensitivity on
  the same 11 samples, with Pearson/Spearman/cosine/IoU@50 correlation of each sample's
  occlusion map against each of the three Grad-CAM-family maps.
- `results/xai/final_iccad5/geometry/geometry_diagnostics.csv` (132 rows = 11 samples × 4
  methods × 3 top-K levels) plus `geometry/summary.csv` and
  `geometry/summary_excl_degenerate.csv` (48 rows each = 4 case types × 4 methods × 3
  levels), including the newly-ported `is_degenerate_map` flag.

All numbers below were computed directly from these five CSVs with pandas; none are
re-derived from the pipeline or restated from memory. Everything is phrased as "the model
attributes greater importance to..." / "overlaps a narrow-gap/thin-line proxy region..." —
never as proof of physical causality — per the convention already established in
`docs/TP_FP_GEOMETRY_ANALYSIS.md`.

This checkpoint is a **different lineage** from the checkpoints discussed in
`docs/ICCAD5_CHECKPOINT_FIX.md` and `docs/TP_FP_GEOMETRY_ANALYSIS.md` §9–11 (which examined
`xai_cnn_iccad5.keras` and `..._checkpoint_fix.keras`). Per `docs/ICCAD5_SEED_REPLICATION.md`,
`balanced_sampling` reproducibly drives the `dense_penultimate` exact-zero-vector rate to
**0.000%** across all 3 replication seeds (vs. up to 85.4% on the original recipe) and
produces 4–7 HS-elevated units (AUC 0.97–0.99) per seed. **That dense_penultimate pathology
is a different layer and a different phenomenon from what is reported in §4 below**, and the
two must not be conflated: dense_penultimate collapse is a post-hoc probe of an internal
bottleneck layer's activations across the full test set; the Grad-CAM degeneracy in §4 is a
gradient-flow property of `conv_final_2` observed on 11 specific representative samples on
this specific checkpoint.

## 1. Attribution concentration and inter-method agreement (`xai_diagnostics.csv`)

No direct inter-method correlation column exists in `xai_diagnostics.csv` (that only appears
in `occlusion_diagnostics.csv`, §2). As a proxy, spread/area/connectivity/edge-concentration
and peak-to-peak pixel distance were computed per method and averaged by case type (n=3 for
TP/TN/FP, n=2 for FN):

| case_type | gcam spread | gcam++ spread | layercam spread | gcam area>50% | gcam++ area>50% | layercam area>50% | gcam↔gcam++ peak dist (px) | gcam↔layercam peak dist (px) |
|---|---|---|---|---|---|---|---|---|
| TP | 77.7 | 79.0 | 68.6 | 0.135 | 0.070 | 0.012 | 6.5 | 47.2 |
| FP | 82.2 | 88.9 | 74.9 | 0.249 | 0.147 | 0.009 | 21.6 | 82.1 |
| TN | 27.0 | 96.1 | 98.0 | 0.000 | 0.074 | 0.017 | 147.3 | 88.2 |
| FN | 8.0  | 87.3 | 93.8 | 0.001 | 0.032 | 0.020 | 104.5 | 78.1 |

Key observations:

- **TP is the only case type where Grad-CAM and Grad-CAM++ peaks agree spatially**
  (mean peak-to-peak distance 6.5 px, on a 224 px canvas) — all other case types show
  40–150 px disagreement between methods, i.e. the three methods are pointing at
  substantially different regions of the layout.
- **Grad-CAM's spread collapses specifically on TN and FN** (27.0 px and 8.0 px vs. 68–98 px
  for Grad-CAM++/LayerCAM on the same cases), while Grad-CAM++ and LayerCAM stay broad and
  roughly consistent across all four case types (68–98 px throughout). This is the attribution-
  level signature of the degenerate-map issue quantified directly in §4.
- `gcam_connected_count` for TN averages 0.667 (i.e., some TN samples have **zero** connected
  attribution regions), consistent with a flat/near-empty Grad-CAM map — this foreshadows the
  `is_degenerate_map` finding.

## 2. Occlusion vs. Grad-CAM-family agreement (`occlusion_diagnostics.csv`)

Mean correlation of the occlusion sensitivity map against each Grad-CAM-family method,
by case type:

| case_type | vs Grad-CAM (Pearson / cosine / IoU@50) | vs Grad-CAM++ (Pearson / cosine / IoU@50) | vs LayerCAM (Pearson / cosine / IoU@50) |
|---|---|---|---|
| TP | 0.702 / 0.800 / 0.356 | 0.674 / 0.789 / 0.311 | 0.685 / 0.798 / 0.119 |
| FP | 0.429 / 0.699 / 0.276 | 0.400 / 0.672 / 0.268 | 0.321 / 0.600 / 0.026 |
| TN | −0.007 / 0.013 / 0.000 | 0.158 / 0.423 / 0.066 | 0.115 / 0.425 / 0.015 |
| FN | −0.020 / 0.013 / 0.000 | 0.055 / 0.327 / 0.000 | 0.069 / 0.379 / 0.011 |

- **TP shows the strongest occlusion-vs-attribution agreement across all three
  Grad-CAM-family methods** (Pearson 0.67–0.70, cosine 0.79–0.80) — occlusion sensitivity
  independently overlaps the same regions the gradient-based methods highlight.
- **TN and FN show near-zero or slightly negative correlation for all three methods**,
  most severely for Grad-CAM (Pearson −0.007 to −0.020). Per-sample detail shows the two TN
  samples `NNHSCAD5999_9` / `NNHSCAD5999_6` have **exactly** 0.000 Pearson/Spearman/cosine
  correlation against Grad-CAM specifically — these are the same two samples flagged
  `is_degenerate_map=True` in the geometry CSV (§4), confirming the degenerate Grad-CAM map
  is not an artifact of the geometry pipeline's top-K masking but is present in the raw
  saliency values used by the occlusion-correlation computation too.
- Grad-CAM++ and LayerCAM retain small positive correlation on the same two degenerate-for-
  Grad-CAM TN samples (e.g., 0.338 Pearson for Grad-CAM++ on both), so occlusion is not
  independently flat on these samples — only Grad-CAM's map is.

## 3. Geometry overlap-vs-random-control (`geometry/summary.csv`, `summary_excl_degenerate.csv`)

Mean `geometry_fraction_paired_diff_median` (attribution-region overlap with the raster
geometry proxy mask, minus the median of 30 random-control draws of matched area), averaged
over the three top-K levels, **with** degenerate rows included:

| case_type | gradcam | gradcam_plus | layercam | occlusion |
|---|---|---|---|---|
| TP | 0.241 | 0.232 | 0.240 | 0.233 |
| FP | 0.044 | 0.006 | 0.186 | 0.144 |
| TN | 0.013 | 0.270 | 0.073 | 0.121 |
| FN | 0.006 | 0.138 | 0.124 | 0.098 |

- **TP is the only case type where all four methods agree**: attribution regions overlap the
  geometry proxy roughly 0.23–0.24 more than chance, consistently, at every level and every
  method.
- **Grad-CAM's control-adjusted overlap is disproportionately weak outside TP** (0.006–0.044
  for FN/FP/TN) relative to the other three methods (0.07–0.27 in the same case types). This
  mirrors §1–2: plain Grad-CAM is the least reliable of the four methods specifically on the
  cases where the model's output is not a confident, correct HS call.
- Grad-CAM++'s FP overlap (0.006) is the one exception to "Grad-CAM++ tracks well outside
  TP" — its TN overlap (0.270) is in fact the single highest value in the table, so
  Grad-CAM++ is not uniformly better than Grad-CAM outside TP, only on TN/FN in this sample.

**Effect of excluding the 6 degenerate rows** (all Grad-CAM, all TN — `NNHSCAD5999_9` and
`NNHSCAD5999_6`, all 3 top-K levels each):

| level_pct | n_samples (with degenerate) | n_samples (excl. degenerate) | geometry_fraction_median (with) | geometry_fraction_median (excl.) | paired_diff_median (with) | paired_diff_median (excl.) |
|---|---|---|---|---|---|---|
| 10 | 3 | 1 | 0.0745 | 0.2874 | 0.0149 | 0.0149 |
| 20 | 3 | 1 | 0.0754 | 0.3115 | 0.0130 | 0.0431 |
| 30 | 3 | 1 | 0.0736 | 0.3072 | 0.0107 | 0.0257 |

Excluding the two degenerate TN samples leaves only **1** TN/Grad-CAM sample, so the
"median" collapses to that single remaining sample's own value — the apparent 4x jump in
`geometry_fraction_median` (0.07 → ~0.29–0.31) is a **small-n artifact of dropping 2 of 3
samples**, not evidence that non-degenerate Grad-CAM overlaps the geometry proxy much more
strongly on TN. `junction_recall_paired_diff_median` for this cell actually flips sign
(0.0 → −0.10 to −0.23) after exclusion — a further indication that with n=1 these summary
statistics are not stable and should be read as descriptive, not inferential, for this cell.
For every other (case_type, method) combination `n_degenerate_excluded=0`, so the two summary
files are identical elsewhere.

## 4. The `is_degenerate_map` finding, precisely scoped

`geometry_diagnostics.csv` flags exactly **6 of 132 rows** (4.5% of all rows; **18.2% of the
33 Grad-CAM-method rows specifically**) as `is_degenerate_map=True`. All 6 are Grad-CAM; zero
for Grad-CAM++, LayerCAM, or Occlusion. All 6 belong to the **TN** case type, and specifically
to only **2 of the 3** TN samples (`NNHSCAD5999_9`, `NNHSCAD5999_6`), each flagged at all
three top-K levels (10/20/30%). The third TN sample and all TP/FP/FN samples are clean for
every method.

**This is a narrow, method-and-checkpoint-specific observation about Grad-CAM's gradient
behavior on 2 of 11 samples in this representative set — it is not the same finding as, and
should not be read as a recurrence of, the `dense_penultimate` zero-vector collapse
documented in `docs/ICCAD5_SEED_REPLICATION.md` and `docs/ICCAD5_FORENSIC_AUDIT.md`.** The
two are structurally distinct:

| | dense_penultimate collapse (seed-replication docs) | Grad-CAM degeneracy (this document, §4) |
|---|---|---|
| Layer | `dense_penultimate` bottleneck activations | `conv_final_2` gradient feeding Grad-CAM's GAP-weighted CAM |
| Evidence | exact-zero activation vector, measured across the full test set | `np.ptp(map) < 1e-6` flat saliency map, measured on 11 representative samples |
| Status on this checkpoint | resolved to 0.000% rate (3/3 balanced_sampling seeds) | still occurs (6/33 Grad-CAM rows, 2/3 TN samples) |
| Affects which methods | all downstream computation reading that layer | Grad-CAM only (Grad-CAM++, LayerCAM, Occlusion unaffected on the same samples, §1–2) |

That said, the qualitative pattern — Grad-CAM specifically, more than Grad-CAM++, much more
than LayerCAM, going flat on high-confidence TN/FN predictions — **is consistent with** the
mechanism already documented (on a different, earlier checkpoint) in
`docs/TP_FP_GEOMETRY_ANALYSIS.md` §11(b): saturated pre-sigmoid logits driving most of
`conv_final_2`'s gradient to exactly zero, with the small remaining nonzero fraction
near-cancelling under Grad-CAM's global-average-pooling channel weighting (tripping the
`max_v > 0` guard), while LayerCAM's pixel-wise weighting (no GAP step) still recovers a
nonzero map from the same gradient. That prior investigation reported Grad-CAM/Grad-CAM++
degenerating on 31.6%/42.1% of a 57-sample set vs. 8.8% for LayerCAM, concentrated in TN/FN.
The new checkpoint's 18.2%-of-Grad-CAM-rows, 0%-for-the-other-three-methods pattern on this
11-sample set is directionally the same shape of result, now confirmed on the
`balanced_sampling seed1` checkpoint with the `is_degenerate_map` flag freshly ported into
the geometry pipeline — but this document does not re-run the direct gradient probe that
would confirm the identical GAP-cancellation mechanism on this specific checkpoint, so the
mechanism is offered as the most evidence-consistent prior explanation, not re-verified here.

## 5. Cross-analysis agreement: does the story hold across all three methods (attribution, occlusion, geometry)?

**Where all three agree:**

- **TP is the strongest, most internally-consistent case type across every analysis.**
  Attribution methods agree spatially (§1, 6.5 px peak distance), occlusion independently
  corroborates the CAM-family maps (§2, Pearson 0.67–0.70), and all four methods show
  consistent, non-trivial overlap with the geometry proxy above random control (§3, 0.23–0.24
  paired diff, uniform across methods and top-K levels).
- **FN is the weakest, most diffuse/inconsistent case type across every analysis.**
  Grad-CAM's spread on FN is the smallest of any case type (8.0 px, §1) even as the other two
  methods stay broad; occlusion agreement with all three CAM-family methods is at its
  lowest here (Pearson −0.02 to 0.06, §2); Grad-CAM's control-adjusted geometry overlap is
  also near the table minimum (0.006, §3). This is consistent with "the model attributes
  weak, poorly-localized importance to any region when it produces a false-negative call,"
  though n=2 FN samples is very small and this should be read as suggestive, not conclusive.
- **Grad-CAM is the least reliable of the four methods outside TP, on every analysis.** It is
  the only method with any degenerate maps (§4), the method with the lowest occlusion
  agreement on non-TP cases (§2), and the method with the weakest control-adjusted geometry
  overlap on FP/TN/FN (§3).

**Where they diverge:**

- Grad-CAM++'s picture is not uniform: it has the *strongest* geometry overlap on TN (0.270,
  §3) despite TN being where Grad-CAM is at its worst — so "Grad-CAM++ is a reliable
  Grad-CAM substitute" holds for TN specifically but not for FP, where Grad-CAM++'s
  control-adjusted overlap (0.006) is the single weakest cell in the whole table, weaker even
  than plain Grad-CAM's FP result (0.044).
- LayerCAM's peak location diverges most from Grad-CAM's on TN (88.2 px, §1) even though
  neither method's map is degenerate on the third (non-flagged) TN sample — the two methods
  are simply not attributing to the same region on that sample, and this document does not
  determine which (if either) is a better geometric match without inspecting that individual
  sample's own row.

## 6. Final results table

Per-case-type, per-method summary (top-20% level shown as the representative middle
stratum; occlusion-correlation columns from `occlusion_diagnostics.csv`, geometry columns
from `geometry/summary.csv` at `level_pct=20`, all with degenerate rows included):

| case_type | method | attribution spread (px) | occlusion-vs-method Pearson | geometry_fraction median | control median | paired diff median | degenerate rows (of n at this level) |
|---|---|---|---|---|---|---|---|
| TP | Grad-CAM | 77.7 | 0.702 | 0.459 | 0.206 | 0.263 | 0/3 |
| TP | Grad-CAM++ | 79.0 | 0.674 | 0.431 | 0.206 | 0.242 | 0/3 |
| TP | LayerCAM | 68.6 | 0.685 | 0.441 | 0.206 | 0.247 | 0/3 |
| TP | Occlusion | n/a | n/a (reference method) | 0.426 | 0.206 | 0.228 | 0/3 |
| FP | Grad-CAM | 82.2 | 0.429 | 0.326 | 0.324 | −0.002 | 0/3 |
| FP | Grad-CAM++ | 88.9 | 0.400 | 0.344 | 0.324 | 0.020 | 0/3 |
| FP | LayerCAM | 74.9 | 0.321 | 0.501 | 0.324 | 0.177 | 0/3 |
| FP | Occlusion | n/a | n/a (reference method) | 0.447 | 0.324 | 0.112 | 0/3 |
| TN | Grad-CAM | 27.0 | −0.007 | 0.075 | 0.064 | 0.013 | 2/3 |
| TN | Grad-CAM++ | 96.1 | 0.158 | 0.308 | 0.064 | 0.244 | 0/3 |
| TN | LayerCAM | 98.0 | 0.115 | 0.139 | 0.064 | 0.077 | 0/3 |
| TN | Occlusion | n/a | n/a (reference method) | 0.185 | 0.064 | 0.122 | 0/3 |
| FN | Grad-CAM | 8.0 | −0.020 | 0.213 | 0.194 | 0.019 | 0/2 |
| FN | Grad-CAM++ | 87.3 | 0.055 | 0.332 | 0.194 | 0.139 | 0/2 |
| FN | LayerCAM | 93.8 | 0.069 | 0.321 | 0.194 | 0.127 | 0/2 |
| FN | Occlusion | n/a | n/a (reference method) | 0.285 | 0.194 | 0.091 | 0/2 |

## 7. Bottom line

On this specific 11-sample, frozen-checkpoint evidence set: the model's attribution behavior
on **true positives is coherent and mutually corroborating** across all three independent
analysis families (attribution, occlusion, geometry). On **false negatives and (for
Grad-CAM specifically) true negatives**, the same three families independently agree that
attribution is weaker, less spatially concentrated, and less well corroborated by an
independent perturbation method — with Grad-CAM being the specific method most affected,
consistent with (but not a re-verification of) a previously-documented saturated-gradient/
GAP-cancellation mechanism from an earlier checkpoint's investigation. This should be read as
a checkpoint- and sample-scoped observation (n=11 total, n=2 for FN) rather than a general
claim about any of the four XAI methods, and it is explicitly distinct from the (separately
resolved, on this checkpoint) `dense_penultimate` activation-collapse pathology.
