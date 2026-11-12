# Workstream C — historical geometry sensitivity reanalysis (lead-run)

## Status and scope

**Completed: historical-provenance sensitivity analysis of the existing 57-sample geometry
results. NOT executed: the corrected-preprocessing 57-sample regeneration.** That
regeneration remains gated on GPU availability, because (a) the 57-sample cohort itself
was selected under bicubic predictions, so corrected regeneration requires re-selecting
the cohort from corrected full-test inference (~146k forward passes across 5 benchmarks),
and (b) the cached occlusion arrays were computed on bicubic inputs; mixing corrected CAM
maps with historical occlusion NPYs would violate the provenance rule in
`docs/IMPROVEMENT_METHODOLOGY_AUDIT.md` A-02. All numbers below describe the historical
bicubic artifacts; none may be presented as corrected-preprocessing results.

No model was retrained; no checkpoint was modified; no file under `results/xai/geometry/`,
`results/xai/final_iccad5/`, or `results/xai/occlusion/` was altered.

## Method

- Input: `results/xai/geometry/geometry_diagnostics.csv` (684 rows), read-only.
- Degeneracy strata for the main set use the stored `xai_diagnostics.csv` spread
  diagnostics: `gcam_spread == 0` / `gcam_pp_spread == 0` / `layercam_spread == 0` marks
  an exactly-flat map under the pipeline's own uint8-rounded diagnostic. This is a
  **proxy** (not a raw `np.ptp` re-derivation); occlusion has no stored spread proxy, so
  its main-set degeneracy status is **unknown**, reported as its own stratum, never imputed.
- Statistical family (pre-declared before running): Wilcoxon signed-rank, two-sided, on
  per-sample `metric - control_mean` differences at top-20% only, for
  4 metrics × 4 methods × 4 case types = 64 tests, Holm step-down correction within the
  family. n<3 or zero-variance cells are reported as not computed with an explicit
  status, never as p=1 or fabricated zeros.
- The final 11-sample set's own exact `is_degenerate_map` flags (6/132 rows) are
  cross-checked as recorded, not re-derived.
- Output: `results/xai/improvements/geometry_reanalysis/` — `stratum_counts.csv`,
  `paired_tests_top20.csv`, `final11_exact_flag_counts.csv`, `report_crosscheck_top20.csv`,
  `plots/top20_geometry_fraction_by_case.png`, `summary.json`.

## Result 1 — the spread==0 proxy exactly reproduces the published flat-map counts

| Method | degenerate (spread==0) of 57 | published flat-map count | match |
|---|---:|---:|---|
| Grad-CAM | 18 | 18 (31.6%) | exact |
| Grad-CAM++ | 24 | 24 (42.1%) | exact |
| LayerCAM | 5 | 5 (8.8%) | exact |
| Occlusion | unknown (no stored proxy) | 16.7% of TP/FP rows flagged in the TP/FP study | not checkable here |

Stratum detail (top-20% rows): Grad-CAM degeneracy is concentrated in FN (7/12) and TN
(6/15) with 2/15 FP and 3/15 TP; Grad-CAM++ similarly (7/6/6/5); LayerCAM only in TP (3/15)
and FP (2/15). This stratification is consistent with the report's "concentrated in
TN/FN for Grad-CAM, iccad5 TP/FP for LayerCAM" description and pins it to stored
per-sample diagnostics rather than memory.

## Result 2 — report cross-check: all 16 reported top-20% medians reproduce exactly

Recomputed medians match `docs/GEOMETRY_ANALYSIS_REPORT.md` §5–8 for every
(case type, method) cell (e.g. TN Grad-CAM 0.027, TP Occlusion 0.419, FN Grad-CAM++ 0.205;
see `report_crosscheck_top20.csv`). No transcription error found in the published table.

## Result 3 — which conclusions survive Holm correction (top-20%, 64-test family)

24 of 64 computed cells are significant at Holm-adjusted p<0.05
(`paired_tests_top20.csv`). The pattern, against the report's claims:

- **Survive, strengthening the report:** LayerCAM and Occlusion are Holm-significant for
  geometry_fraction, narrow_gap, thin_line (and corner for LayerCAM) in TP, TN, and FP —
  the report's "LayerCAM/Occlusion strongest and most consistent" claim holds under family
  correction. TN Grad-CAM corner_recall is significantly *negative*
  (median −0.183, Holm p=0.042), consistent with the documented TN anti-correlation.
- **Does not survive:** Grad-CAM TP geometry_fraction (raw p=0.022 per report) and FP
  geometry_fraction (raw p=0.013) are not significant after Holm within this family; the
  surviving Grad-CAM FP cell is narrow_gap_fraction only. The report's per-cell
  significance statements for Grad-CAM TP/FP should therefore be read as
  uncorrected-exploratory, not family-confirmed.
- **All FN cells are null** under the family (consistent with the report's at-chance FN
  reading; n=12, smallest stratum).

## Result 4 — final 11-sample cross-check

Exact flags confirm 6/132 flagged rows, all Grad-CAM, at all three levels, for the two
historical TN IDs. Workstream D's separate artifact audit additionally established those
two "distinct" samples are **byte-identical duplicates of one source raster**
(`NNHSCAD5999.png6.png` and `NNHSCAD5999.png9.png`, identical SHA-256, decoded-equal;
see `docs/FINAL_GRADCAM_DEGENERACY_ANALYSIS.md`), so the final-set finding is one unique
image counted twice. Direct decoded-pixel hashing of all three TN source rasters confirms
**two unique TN rasters**, of which **one** has the historical Grad-CAM degeneracy flag.
Thus the filename-level count remains 2/3 TN files; the deduplicated TN count is 1/2
unique rasters. Neither is a population estimate. This TN-only check does not establish
a unique-image denominator for the entire 11-file cohort.

## Limitations

- Historical bicubic provenance throughout; the preprocessing fix (commit 9ea630d) changes
  future runs, not these artifacts.
- spread==0 is a stored-diagnostic proxy; occlusion's main-set degeneracy is unresolved.
- One level (top-20%) in the test family; other levels remain uncorrected-exploratory.
- Descriptive/exploratory, n=57 with confidence-selected samples; no causal or
  physical-lithography reading is licensed.
- Flat-map rows keep arbitrary top-K overlap values in the source CSV; they are retained
  in strata counts and excluded from no analysis silently.

## Command

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/improvements/geometry_reanalysis.py
```
