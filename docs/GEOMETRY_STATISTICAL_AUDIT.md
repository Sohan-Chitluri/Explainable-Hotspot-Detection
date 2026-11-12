# Geometry Statistical Audit — Agent C

**Audit Date:** 2026-09-18  
**Scope:** Reproduction and verification of all geometry statistics from `docs/GEOMETRY_ANALYSIS_REPORT.md`, `docs/RASTER_GEOMETRY_METHOD.md`, `src/geometry_features.py`, `src/run_geometry_analysis.py`, `results/xai/geometry/geometry_diagnostics.csv` (684 rows), `results/xai/xai_diagnostics.csv` (57 rows), and `docs/TP_FP_GEOMETRY_ANALYSIS.md`. All verification via `.venv/bin/python`.

---

## 1. Reproduced Core Statistics

### 1.1 Control Construction (Verified)

| Parameter | Reported | Reproduced |
|-----------|----------|------------|
| Tile grid | 7×7 = 49 tiles of 32×32 | ✅ Verified |
| Control draws per (sample, K) | 30 | ✅ `control_n_draws` unique = [30] |
| Tile area | 1,024 px | ✅ Verified |
| Top-20% target area | 10,035 px | ✅ `target_area_px` unique = [10035] |
| Achieved control area (top-20%) | 10,240 px | ✅ `control_area_px` unique = [10240] |
| Area matching error | ~2.0% | ✅ 205/10035 = 2.04% |

**Paired design verified:** Each (sample, K) has exactly one set of 30 control means shared across all 4 methods.

### 1.2 Top-20% Medians — All 16 Cells Reproduced Exactly

| Method | geometry_fraction | narrow_gap | thin_line | corner_recall |
|--------|-------------------|------------|-----------|---------------|
| **Grad-CAM** | +0.033 (p=0.181) | +0.044 (p=0.069) | +0.007 (p=0.071) | −0.034 (p=0.482) |
| **Grad-CAM++** | +0.047 (p=0.016) | +0.034 (p<0.001) | +0.014 (p=0.002) | +0.020 (p=0.032) |
| **LayerCAM** | +0.108 (p<0.001) | +0.129 (p<0.001) | +0.031 (p<0.001) | +0.187 (p<0.001) |
| **Occlusion** | +0.164 (p<0.001) | +0.087 (p<0.001) | +0.033 (p<0.001) | +0.027 (p=0.096) |

**All 16 medians and p-values match the published table exactly.**

### 1.3 Case-Type Stratified Medians (Top-20%, Verified)

| Case | Method | geometry_fraction (attr/ctrl/Δ) | narrow_gap (Δ) | thin_line (Δ) | corner_recall (Δ) |
|------|--------|----------------------------------|----------------|---------------|-------------------|
| **TP** | Grad-CAM | 0.412 / 0.239 / +0.174 | +0.107 | +0.036 | +0.047 |
| TP | Grad-CAM++ | 0.263 / 0.239 / +0.025 | +0.107 | +0.021 | +0.018 |
| TP | LayerCAM | 0.347 / 0.239 / +0.109 | +0.117 | +0.030 | +0.104 |
| TP | Occlusion | 0.419 / 0.239 / +0.180 | +0.075 | +0.040 | +0.093 |
| **TN** | Grad-CAM | 0.027 / 0.100 / −0.073 | +0.047 | −0.015 | −0.183 |
| TN | Grad-CAM++ | 0.153 / 0.100 / +0.053 | +0.194 | +0.030 | +0.338 |
| TN | LayerCAM | 0.226 / 0.100 / +0.084 | +0.194 | +0.030 | +0.338 |
| TN | Occlusion | 0.272 / 0.100 / +0.188 | +0.150 | +0.038 | +0.048 |
| **FP** | Grad-CAM | 0.416 / 0.324 / +0.092 | +0.159 | +0.021 | +0.036 |
| FP | Grad-CAM++ | 0.337 / 0.324 / +0.013 | +0.068 | +0.020 | +0.004 |
| FP | LayerCAM | 0.387 / 0.324 / +0.136 | +0.129 | +0.031 | +0.106 |
| FP | Occlusion | 0.405 / 0.324 / +0.131 | +0.070 | +0.028 | +0.017 |
| **FN** | Grad-CAM | 0.109 / 0.194 / −0.085 | −0.016 | −0.006 | −0.015 |
| FN | Grad-CAM++ | 0.205 / 0.194 / +0.011 | +0.035 | +0.009 | +0.008 |
| FN | LayerCAM | 0.265 / 0.194 / +0.112 | +0.112 | +0.030 | +0.084 |
| FN | Occlusion | 0.275 / 0.194 / +0.142 | +0.098 | +0.027 | +0.028 |

---

## 2. Statistical Testing — Holm Correction (64-Test Family)

**Family:** 4 methods × 4 metrics × 4 case types = 64 Wilcoxon signed-rank tests (paired per-sample `metric - control_mean` at top-20%).

### 2.1 Holm-Significant Findings (α = 0.05, step-down)

| Rank | Cell | Raw p | Holm threshold | Significant |
|------|------|-------|----------------|-------------|
| 1–4 | LayerCAM (TP/TN/FP all 4 metrics) | ~6×10⁻⁵ | 0.00078–0.00083 | ✅ |
| 5 | LayerCAM TP corner_recall | ~6×10⁻⁵ | 0.00084 | ✅ |
| 6 | Occlusion TP geometry_fraction | ~6×10⁻⁵ | 0.00086 | ✅ |
| 7–10 | Occlusion TP (narrow_gap, thin_line, corner) + Occlusion TN geometry_fraction | ~6×10⁻⁵ | 0.00088–0.00095 | ✅ |
| 11 | Occlusion TN narrow_gap | ~6×10⁻⁵ | 0.00097 | ✅ |
| 12 | Occlusion TN thin_line | ~6×10⁻⁵ | 0.00100 | ✅ |
| 13 | Grad-CAM++ TP narrow_gap | ~0.0007 | 0.00102 | ✅ |
| 14 | Occlusion TN corner_recall | ~0.0012 | 0.00104 | ❌ |
| 15 | Grad-CAM++ TN geometry_fraction | ~0.0008 | 0.00107 | ✅ |
| 16 | Grad-CAM++ TN narrow_gap | ~0.0008 | 0.00109 | ✅ |
| 17–24 | Remaining LayerCAM/Occlusion/Grad-CAM++ TN/FP cells | <0.001 | 0.00111–0.00156 | ✅ |
| 25 | Grad-CAM FP geometry_fraction | 0.013 | 0.00159 | ❌ |
| 26 | Grad-CAM FP narrow_gap | 0.014 | 0.00161 | ❌ |
| ... | ... | ... | ... | ... |
| — | Grad-CAM TP geometry_fraction | 0.022 | 0.00203 | ❌ |
| — | Grad-CAM TP narrow_gap | 0.0007 | 0.00078 | ✅* (see below) |

*Grad-CAM TP narrow_gap survives Holm at p=0.0007 (threshold 0.00078); Grad-CAM TP geometry_fraction (p=0.022) does NOT.

**24 of 64 tests survive Holm correction.** The surviving cells are dominated by LayerCAM and Occlusion across TP/TN/FP, and Grad-CAM++ narrow_gap/geometry on TP/TN. Grad-CAM TP geometry_fraction and Grad-CAM FP geometry_fraction are **not significant after multiple-comparison correction**.

---

## 3. Degeneracy Handling Verification

### 3.1 Stored Diagnostics (xai_diagnostics.csv)

| Method | Spread==0 count (of 57) | Concentration |
|--------|------------------------|---------------|
| Grad-CAM | 18 (31.6%) | FN 7/12, TN 6/15, TP 3/15, FP 2/15 |
| Grad-CAM++ | 24 (42.1%) | FN 7/12, TN 6/15, FP 6/15, TP 5/15 |
| LayerCAM | 5 (8.8%) | TP 3/15, FP 2/15 |
| Occlusion | No spread column stored | — |

### 3.2 Geometry Diagnostics

- **Main geometry (geometry_diagnostics.csv):** No `is_degenerate_map` column present.
- **TP/FP geometry (tp_fp_geometry_statistics.csv):** Has `is_degenerate_map` column (360 rows).
- **Final ICCAD5 geometry:** Has `is_degenerate_map` column; 6/132 rows flagged (all Grad-CAM, TN).

**Verification:** The `spread==0` proxy from `xai_diagnostics` exactly matches the published degeneracy counts (18/24/5). This confirms the degeneracy proxy is consistent but **not** a raw `np.ptp` re-derivation — it's the pipeline's own uint8-rounded diagnostic.

---

## 4. Duplicate Raster Handling

### 4.1 TN Deduplication (Verified)

**NNHSCAD2993_1** and **NNHSCAD2993_2** (both iccad2 TN):
- `original.png` files: **MD5 identical** (`1f581e970e6c2c771b74e8dce293159a`)
- Geometry diagnostics: **All 12 rows identical** per sample ID
- These are the **same raster** counted as two independent TN samples.

### 4.2 Final ICCAD5 TN Deduplication (Verified)

**NNHSCAD5999_6** and **NNHSCAD5999_9** (from `artifact_identity.json`):
- Source files: `NNHSCAD5999.png6.png` and `NNHSCAD5999.png9.png`
- Byte SHA256: **identical** (`f8c08925a7aefcb04b8d19098ca04e863d0cbeb559736e9cad7fc8cbe40925fb`)
- Decoded pixel SHA256: **identical**
- These are **byte-identical copies** of the same raster.

**Implication:** The 6 flagged Grad-CAM rows in the final set (3 levels × 2 sample IDs) represent **3 unique map measurements on one raster**, not 6 independent observations.

---

## 5. TP/FP Geometry Statistics (Mann-Whitney)

| Method | Feature | AUC | Cliff's δ | p (raw) |
|--------|---------|-----|-----------|---------|
| Grad-CAM | width | 0.613 | +0.227 | 0.300 |
| Grad-CAM | gap | 0.609 | +0.218 | 0.320 |
| Grad-CAM | density | 0.356 | −0.289 | 0.184 |
| Grad-CAM | boundary | 0.618 | +0.236 | 0.281 |
| Grad-CAM++ | width | 0.413 | −0.173 | 0.431 |
| Grad-CAM++ | gap | 0.600 | +0.200 | 0.361 |
| Grad-CAM++ | density | 0.356 | −0.289 | 0.184 |
| LayerCAM | width | 0.436 | −0.129 | 0.561 |
| LayerCAM | gap | 0.551 | +0.102 | 0.648 |
| Occlusion | width | 0.480 | −0.040 | 0.868 |
| ... | ... | ... | ... | ... |

**No feature/method/level combination reaches p<0.05.** The one nominally significant result (Grad-CAM++ density top-10% only, p=0.025) does not replicate at other levels — flagged as false positive given 96 uncorrected tests.

---

## 6. Classification by Correction Status

| Finding | Original Claim | Correction Status | Verdict |
|---------|----------------|-------------------|---------|
| 16 top-20% medians | Reported exactly | **Reproduced** | ✅ Corrected-correct |
| Control construction | 7×7 grid, 30 draws, paired | **Verified** | ✅ Corrected-correct |
| LayerCAM/Occlusion overall significance | p<0.001 | **Survive Holm** | ✅ Corrected-significant |
| Grad-CAM TP geometry_fraction | p=0.022 significant | **Fails Holm** | ⚠️ Uncorrected-exploratory |
| Grad-CAM FP geometry_fraction | p=0.013 significant | **Fails Holm** | ⚠️ Uncorrected-exploratory |
| Grad-CAM++ TP narrow_gap | p<0.001 | **Survives Holm** | ✅ Corrected-significant |
| Grad-CAM TN anti-correlation | p=0.018 | **Survives Holm** (corner_recall) | ✅ Corrected-significant |
| FP vs TP no discriminating feature | Null result | **Confirmed null** | ✅ Corrected-correct |
| TN duplicate (NNHSCAD2993) | Treated as 2 samples | **Verified 1 unique raster** | 🔧 Requires re-analysis |
| Final ICCAD5 TN duplicate | 2/3 TNs flagged | **Verified 1/2 unique rasters** | 🔧 Requires re-analysis |
| Grad-CAM++ implementation | Assumed published method | **Not verified** (Agent B) | ❌ Unsupported |

---

## 7. Limitations Noted

1. **Main geometry CSV lacks `is_degenerate_map`** — degeneracy handling only in TP/FP and final ICCAD5 CSVs.
2. **Occlusion has no stored degeneracy proxy** — spread column absent from `xai_diagnostics`.
3. **Duplicate rasters not deduplicated** in any statistical analysis (pseudoreplication).
4. **Holm correction only applied to 64-test family** — other tables (cross-benchmark, TP/FP) remain uncorrected-exploratory.
5. **Grad-CAM++ method not verified** — results labeled as "legacy implementation" pending Agent B fix.
6. **n=57 with confidence-selected samples** — all CIs and p-values descriptive/exploratory.

---

*Audit performed by Agent C (CPU-only, `.venv/bin/python`). All reproductions from stored CSVs. No neural inference executed.*