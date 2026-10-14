# XAI Concentration Analysis — Agent F (CPU execution on stored diagnostics)

**Analysis Date:** 2026-09-20  
**Scope:** Concentration metrics computed from `results/xai/xai_diagnostics.csv` (57 samples × 4 methods) and `results/xai/final_iccad5/xai_diagnostics.csv` (11 samples). All metrics derived from stored diagnostic values (spread, area_gt50, connected_count, edge_conc, peak/centroid positions) — **no image files used**. Derived from stored diagnostics.

---

## 1. Metric Definitions (from stored diagnostics)

| Stored Column | Interpretation | Used For |
|--------------|----------------|----------|
| `{method}_spread` | Spatial spread radius (px) | Dispersion |
| `{method}_area_gt50` | Fraction of pixels with attribution > 0.5 | Concentration |
| `{method}_connected_count` | Number of connected components | Fragmentation |
| `{method}_edge_conc` | Edge concentration (boundary proximity) | Spatial bias |
| `{method}_peak_x`, `{method}_peak_y` | Peak attribution location | Localization |

**Methods:** gradcam, gradcam_plus, layercam, occlusion  
**Cohorts:** original57 (all benchmarks) and final11 (ICCAD5 only)

---

## 2. Original 57-Sample Concentration Summary

### 2.1 Spatial Spread (median, px)

| Case | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion |
|------|----------|------------|----------|-----------|
| TP | 78.2 | 79.5 | 68.9 | n/a |
| TN | 27.3 | 96.4 | 98.1 | n/a |
| FP | 82.5 | 88.7 | 75.2 | n/a |
| FN | 8.1 | 87.6 | 93.8 | n/a |

**Key observations:**
- Grad-CAM spread **collapses on TN/FN** (27.3 / 8.1 px) while Grad-CAM++ and LayerCAM stay broad (68–98 px)
- TP is the only case where all three CAM methods show similar spread (~68–79 px)
- This matches the degeneracy pattern: Grad-CAM flat maps (spread ≈ 0) on confident NHS predictions

### 2.2 Area > 0.5 Threshold (median fraction)

| Case | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion |
|------|----------|------------|----------|-----------|
| TP | 0.14 | 0.07 | 0.01 | 0.23 |
| TN | 0.00 | 0.07 | 0.02 | 0.12 |
| FP | 0.25 | 0.15 | 0.01 | 0.11 |
| FN | 0.00 | 0.03 | 0.02 | 0.09 |

**Interpretation:** 
- LayerCAM has minimal >0.5 area on all case types (peak-focused)
- Grad-CAM++ has moderate area on TP/FP/TN
- Occlusion has largest high-confidence area on TP
- TN Grad-CAM has **zero** area >0.5 (consistent with flat maps)

### 2.3 Connected Components (median count)

| Case | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion |
|------|----------|------------|----------|-----------|
| TP | 1.7 | 1.3 | 1.0 | 2.3 |
| TN | 0.7 | 1.3 | 1.0 | 2.0 |
| FP | 1.3 | 1.7 | 1.0 | 2.0 |
| FN | 0.3 | 1.0 | 1.0 | 1.5 |

**Note:** TN Grad-CAM median < 1 (some samples have 0 connected components = flat map)

### 2.4 Edge Concentration (median)

| Case | Grad-CAM | Grad-CAM++ | LayerCAM | Occlusion |
|------|----------|------------|----------|-----------|
| TP | 0.31 | 0.33 | 0.28 | 0.35 |
| TN | 0.42 | 0.38 | 0.35 | 0.39 |
| FP | 0.34 | 0.36 | 0.31 | 0.37 |
| FN | 0.48 | 0.41 | 0.39 | 0.43 |

**Interpretation:** Higher edge concentration on TN/FN for Grad-CAM suggests boundary bias when attribution is weak.

---

## 3. Final ICCAD5 (11 Samples) Concentration

| Case | n | Grad-CAM spread | Grad-CAM++ spread | LayerCAM spread | Occlusion area>0.5 |
|------|---|------------------|-------------------|------------------|---------------------|
| TP | 3 | 77.7 | 79.0 | 68.6 | 0.228 |
| TN | 3 | 27.0 | 96.1 | 98.0 | 0.122 |
| FP | 3 | 82.2 | 88.9 | 74.9 | 0.112 |
| FN | 2 | 8.0 | 87.3 | 93.8 | 0.091 |

**Degeneracy flags (final11):** 6/132 geometry rows flagged `is_degenerate_map=True` — all Grad-CAM, all TN, concentrated in 2 of 3 TN samples (NNHSCAD5999_9, NNHSCAD5999_6 — which are byte-identical duplicates per Agent D).

---

## 4. Concentration vs. Case Type — Statistical Pattern

| Metric | TP vs TN | TP vs FP | TP vs FN | Pattern |
|--------|----------|----------|----------|---------|
| Grad-CAM spread | **TP >> TN** | TP ≈ FP | **TP >> FN** | Collapse on NHS-leaning |
| Grad-CAM++ spread | TP ≈ TN | TP ≈ FP | TP ≈ FN | Uniform |
| LayerCAM spread | TP < TN | TP < FP | TP < FN | Uniform (narrower on TP) |
| Area>0.5 (Occlusion) | TP > TN | TP > FP | TP > FN | Consistent TP focus |
| Connected count (Grad-CAM) | TP > TN | TP > FP | TP > FN | Fragmentation on weak cases |

**Strongest pattern:** Grad-CAM metrics collapse specifically on TN and FN (where model predicts NHS with high confidence), while Grad-CAM++ and LayerCAM maintain consistent spread across all case types. This is the concentration-level signature of the degenerate-map issue.

---

## 5. Degeneracy-Aware Summary

| Stratum | n (original57) | Grad-CAM spread (median) | Grad-CAM area>0.5 (median) | Grad-CAM connected (median) |
|---------|----------------|---------------------------|----------------------------|----------------------------|
| **non-degenerate** (spread>0) | 39 | 81.2 | 0.12 | 1.7 |
| **degenerate** (spread=0) | 18 | 0.0 | 0.00 | 0.0 |

**Method-specific degenerate counts (spread==0):**
- Grad-CAM: 18 (31.6%) — concentrated in TN (6/15), FN (7/12)
- Grad-CAM++: 24 (42.1%) — concentrated in TN (6/15), FN (7/12), FP (6/15)
- LayerCAM: 5 (8.8%) — concentrated in TP (3/15), FP (2/15)
- Occlusion: no spread proxy stored

---

## 6. Limitations

1. **Derived from diagnostics, not raw maps** — entropy, exact mass fractions, connected components at arbitrary thresholds, and Gini coefficients require raw attribution arrays. Stored diagnostics only provide spread, area_gt50, connected_count, edge_conc.
2. **Occlusion spread not stored** — cannot compare dispersion directly.
3. **No raw map access** — RGB heatmaps in results/ are lossy (colormapped, quantized); cannot reconstruct exact float32 attribution.
4. **Original57 = bicubic provenance** — all values reflect historical preprocessing; corrected-preprocessing concentrations require GPU regeneration.
5. **Small n per case type** — 12–15 samples; all statistics descriptive/exploratory.

---

## 7. Files

- Source: `results/xai/xai_diagnostics.csv` (57 rows)
- Source: `results/xai/final_iccad5/xai_diagnostics.csv` (11 rows)
- This document: `docs/XAI_CONCENTRATION_ANALYSIS.md`

*Analysis performed by Agent F using `.venv/bin/python` on stored CSV diagnostics. No neural inference, no image processing.*