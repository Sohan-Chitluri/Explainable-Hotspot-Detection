# XAI agreement analysis: historical artifact availability and robust metric protocol

## Outcome and scope

**No scientifically valid new inter-method agreement values can be calculated from the stored historical maps.** The CPU audit inspected 476 selected map artifacts: **408 CAM-family files are 224×224×3 RGB uint8 colormapped PNGs; only 68 occlusion maps are lossless scalar 224×224 float32 arrays.** All 68 occlusion arrays are finite. The historical `results/xai/` numeric-map scan found exactly these 68 `.npy` files and no other `.npy/.npz/.h5/.hdf5/.tif/.tiff` files (new improvement/forensic outputs excluded). There is no pair of different methods with both scalar maps retained for the same sample, target and checkpoint.

The script therefore emits **612 candidate pair records / 7,344 metric records, all unavailable with null values and explicit reasons**, not invented zeros or approximate raw attributions. These are availability records, **not 612 independent observations**. There are 216 separately identified cohort/benchmark/checkpoint/case/target-mode/target-class/method-pair strata. No universal agreement score, method ranking, p-value or confidence interval is produced.

This is an audit of **historical bicubic-preprocessing artifacts**, not a corrected-preprocessing result. No model inference, training, GPU operation or installation was performed. Workstream B prediction validation and regeneration of lossless maps are prerequisites to corrected scientific conclusions. Existing artifacts and shared source were not changed. Concentration analysis is left to workstream F.

## Cohorts and provenance

| Cohort | Benchmark | TP | TN | FP | FN | Total |
|---|---|---:|---:|---:|---:|---:|
| original57 | iccad1 | 3 | 3 | 3 | 2 | 11 |
| original57 | iccad2 | 3 | 3 | 3 | 3 | 12 |
| original57 | iccad3 | 3 | 3 | 3 | 3 | 12 |
| original57 | iccad4 | 3 | 3 | 3 | 3 | 12 |
| original57 | iccad5 | 3 | 3 | 3 | 1 | 10 |
| final11 | iccad5 | 3 | 3 | 3 | 2 | 11 |

`original57` is defined by `results/xai/xai_diagnostics.csv`; its total case counts are TP=15, TN=15, FP=15, FN=12. `final11` is defined separately by `results/xai/final_iccad5/xai_diagnostics.csv`. Both are joined one-to-one to their respective `occlusion/occlusion_diagnostics.csv`, with benchmark, sample ID, case and class labels checked. Counts and unique sample directories are asserted in executable integration tests.

Original runs used `models/xai/xai_cnn_iccad{1..5}.keras` according to `src/run_xai.py:248`. Their current SHA-256 hashes are recorded, **not misrepresented as historical generation-time attestations**. Final11 is documented against `models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras`, checksum `9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70`; the script checks that hash against the documented identity. This final checkpoint is not the original ICCAD5 checkpoint or checkpoint-fix model. See `docs/FINAL_ICCAD5_SYNTHESIS.md:3–36` and `results/xai/final_iccad5/CHECKPOINT_CHECKSUMS.txt`.

All 476 audited maps have per-file hashes. Input diagnostic CSVs and relevant runner sources have hashes in `provenance.csv`. Source hashes describe the source observed during this audit, not a claim to recover the exact historical runtime environment. Original input tensor hashes and generation-time per-map checkpoint bindings were not retained; case/target labels are historical metadata, not revalidated predictions under corrected preprocessing.

Target handling is explicit:

- `pred` uses the stored predicted class; `true` uses the actual class. CAM target class is passed from these labels in `src/run_xai.py:287–307`.
- TP pred/true both target HS; TN both target NHS. Those duplicate target-mode artifacts are **not independent replicates** and are kept in separate availability strata, never pooled.
- FP pred targets HS while true targets NHS; FN pred targets NHS while true targets HS. These must never be compared as if they explained the same score.
- Occlusion was saved for the predicted class only (`src/run_occlusion.py:262–268`). Consequently there are six four-method pairs per predicted-target sample and three CAM-only pairs per true-target sample. There are no fabricated true-target occlusion maps.
- original57 has 342 predicted-target and 171 true-target candidate pairs; final11 has 66 and 33 respectively. No cross-cohort, cross-checkpoint, cross-case, cross-sample or cross-target map comparisons are performed.

## Why RGB PNGs cannot support this calculation

`src/run_xai.py:295,301,307` saves CAM heatmaps via `uint8(cmap(normalized_map)[:,:,:3] * 255)`. Those files are **rendered color encodings**, not grayscale scalar attribution arrays, despite names such as `pred_gradcam.png`. PNG is lossless for its stored RGB bytes, not for the floating-point attribution that existed before colormapping and quantization. Overlays add blending with the layout; comparison panels add rendering, labels and possible resampling.

The audit does not invert the colormap, average RGB channels, decode overlays, or reconstruct raw attribution from any rendered artifact. A colormap inversion could at best supply quantized bins, not original values, exact nonzero support, or exact ties near a cutoff; normalization already discards original scale and clipping discards signs. Such recovered bins would not justify the requested raw-map agreement measurements. Pixel-color similarity is not an attribution agreement metric.

Occlusion `raw_occlusion_map.npy` is saved directly from a float array (`src/run_occlusion.py:264–268`). It contains **signed, overlapping-window-averaged target pre-sigmoid score drops**, not independent per-pixel causal effects (`src/occlusion.py:182–200`). The audited parameters are window 32, stride 16, baseline 0.0. Taking `maximum(raw, 0)` is an explicit positive-evidence transform for comparison to rectified CAMs; it is not recovery from an image. Negative importance must not be silently converted to positive magnitude. A single lossless method does not supply inter-method agreement by itself; comparing occlusion to its own positive transform would be tautological rather than a substitute.

## Implemented metric definitions for eligible lossless pairs

`pair_metrics(a, b)` in `scripts/improvements/agreement_analysis.py` is implemented and exercised by CPU tests. It requires nonempty, equally shaped, aligned 2D finite nonnegative scalar maps for the **same sample, target score, checkpoint and preprocessing**. The historical runner intentionally never feeds RGB images into this function. A future regeneration must establish those identities before calling it; the current script is an audit of this frozen corpus, not an automatic importer of new map formats.

### Top 1%, 5%, 10%, 20% rank-region overlap

For each requested percentage p on an N-pixel canvas, let `k=ceil(N*p/100)`. Select the greatest k **strictly positive** values, including **all positive pixels tied with the cutoff**. If positive support has fewer than k pixels, select that support only; **never pad with zero pixels**. On the 224×224 canvas requested k are 502, 2,509, 5,018 and 10,036.

Report separately:

- IoU = intersection / union;
- Dice = 2 × intersection / (selected count A + selected count B);
- requested k, actual selected count, threshold, cutoff tie count and support-limited flag for each map.

Inclusive ties avoid arbitrary row-major tie-breaking, especially important for blockwise occlusion. They mean the selected fraction can exceed p or fall short when support-limited; these metrics are explicitly **tie-inclusive top-p rank regions**, not falsely advertised as exact-area masks. Disjoint nonempty regions have a genuine defined zero overlap. No averaging over percentages or combination with correlations creates a universal score.

### Nonzero-support correlations and ranks

Compute Pearson and average-tie Spearman separately on:

1. **Positive-support union**: pixels where A>0 OR B>0; removes shared-zero background while retaining support disagreement.
2. **Positive-support intersection**: pixels where A>0 AND B>0; compares amplitude/rank conditional on both methods assigning positive importance.

Both report support pixel counts. Intersection correlations may look high despite almost disjoint support; they must be read alongside union correlations and overlaps. Spearman uses average ranks for ties, not arbitrary pixel ordering. The top-p IoU/Dice profile is the corresponding rank-threshold overlap analysis; no unqualified rank-biased-overlap score is imposed on heavily tied maps.

### Degenerate/undefined policy

- Constant maps, including both empty-positive/all-zero and spatially constant positive maps: all agreement metrics **undefined**, with null values and reason `constant_or_empty_positive_map`. A uniform map has no discriminating spatial ranking; it does not merit perfect top-p agreement.
- Nonfinite, signed without an explicit transform, mismatched-shape or non-2D inputs: null metrics with the relevant reason. No resizing, silently dropped pixels or absolute-value sign conversion.
- Correlation on fewer than three selected support pixels, or a constant vector on that support: **undefined** with a reason. Overlap can remain defined when correlation is not.
- Missing lossless historical CAM arrays: **unavailable** (distinct from a mathematically undefined metric on a supplied array), reason `missing_lossless_cam_maps_colormapped_png_only`.
- JSON uses null and strict `allow_nan=False`. CSV missing metric values are blank with mandatory `metric_status` and `metric_reason`. Zero counts of eligible pairs are factual counts, not zero-valued scientific metrics.

Exact stored zeros and exact ties are used, without an unvalidated universal epsilon. Near-constant but nonconstant maps can give numerically defined but scientifically unstable rankings. An eventual rerun should retain absolute raw scale, measure numerical sensitivity using a justified scale-aware threshold, and report that sensitivity separately rather than selectively deleting maps to improve agreement.

## Why historical scalar correlation CSVs are not reused

`src/occlusion.py:30–78` computes full-canvas Pearson/Spearman/cosine and a fixed **normalized-amplitude >=0.5** IoU. It substitutes **0.0** for constant-map correlations and empty-union IoU and rounds results to four decimals. Thus historical zero values do not distinguish undefined correlation from real disagreement. Fixed amplitude 0.5 is **not top 50% rank overlap** and cannot be converted into top 1/5/10/20% results.

Those scalar CSVs lack sufficient information to reconstruct new support-restricted or thresholded metrics. New availability records are not retroactive repairs of those numbers. No statement that two methods agree best, or one is universally more reliable, is warranted from this audit.

## Sampling, dependence and multiple comparisons

The historical sample selection is confidence-ranked within each benchmark/case (`src/run_xai.py:151–163`), not a random test-set sample. Per-checkpoint/case cells contain only 1–3 images. Selection, benchmark-specific model differences and overlap of sample identities across historical cohorts prevent interpreting these results as population estimates or an independent replication. Pixels are spatially autocorrelated; 50,176 pixels are not 50,176 independent observations. Overlapping occlusion windows and upsampled CAM resolution further invalidate naive pixelwise significance testing.

**No hypothesis tests were run (n=0), so no multiplicity-adjusted p-values exist.** This is deliberate, not an omitted correction. When eligible maps exist, first report per-image estimates, tie/support counts, valid/undefined denominators and descriptive within-stratum distributions. Do not manufacture bootstrap certainty from n=1–3 selected images.

For a future sufficiently sized, preregistered comparison, use images (or source-layout clusters when related raster variants exist), not pixels, as resampling units. Keep cohort/checkpoint/benchmark/target identity fixed. Specify the entire family before looking at results: all tested case contrasts × method pairs × four percentages × chosen overlap measures and support-correlation variants within the declared scientific claim. Holm family-wise correction is a conservative option under dependent metrics; if claiming across cohorts/checkpoints, expand the family rather than silently resetting it. Clearly label any chosen FDR procedure as exploratory. Do not cherry-pick whichever percentage or correlation yields significance. No tests are recommended for the present selected tiny strata.

## Reproduction and verification

From the repository root, using installed CPU packages only:

```bash
.venv/bin/python scripts/improvements/agreement_analysis.py --self-test
.venv/bin/python scripts/improvements/agreement_analysis.py
```

Executed successfully: **7 tests passed**, including identical maps, monotone rank invariance, cutoff tie expansion, support-limited selections, degenerate nulls, disjoint-support undefined correlation, invalid inputs, and actual repository cohort/artifact/pair counts. Strict JSON serialization and CSV row counts were checked after writing. Test arrays are synthetic unit fixtures only; they are never reported as scientific sample results.

Generated under `results/xai/improvements/agreement/`:

- `audit.json`: complete inventory, provenance, policy, pairwise availability and separate strata;
- `artifact_inventory.csv`: 476 entries (408 rendered CAMs, 68 scalar occlusion maps);
- `pairwise_metrics.csv`: 7,344 explicitly unavailable metric entries for 612 candidate pairs;
- `stratum_availability.csv`: 216 separately identified strata with sample and availability counts;
- `provenance.csv`: diagnostic CSV/source/checksum-file hashes;
- `execution_summary.json`: machine-readable executed counts and blocked status;
- `self_test.txt`: actual CPU test output.

To unblock actual agreement, preserve native and aligned floating-point CAM maps (with signed/raw and explicitly rectified variants distinguished), occlusion arrays, input-tensor hashes, checkpoint SHA-256, class/target-score identity, layer, resize/normalization metadata and method parameters. Generate them only after the corrected-preprocessing validation gate, in new isolated results—not by overwriting historical artifacts or decoding RGB PNGs.
