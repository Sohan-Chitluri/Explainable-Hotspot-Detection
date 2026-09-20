# Sample Expansion Audit — Agent I

**Audit Date:** 2026-09-20  
**Scope:** CPU-only inspection of existing ICCAD5 predictions to design a deterministic expanded TP/TN/FP/FN sample beyond the current 11 (TP=3, TN=3, FP=3, FN=2). Based on `docs/IMPROVEMENT_METHODOLOGY_AUDIT.md`, `docs/DATASET_INTEGRITY_AUDIT.md`, `results/xai/iccad5_seed_replication/`, and `results/xai/final_iccad5/`. GPU unavailable — no new inference.

---

## 1. Existing Prediction Artifacts

### 1.1 Available Prediction Files

| File | Content | Rows |
|------|---------|------|
| `results/xai/final_iccad5/xai_diagnostics.csv` | 11 selected samples (TP=3, TN=3, FP=3, FN=2) with predictions | 11 |
| `results/xai/iccad5_seed_replication/seed_summary_stats.csv` | Aggregate classification metrics per seed | 5 |
| `results/xai/iccad5_seed_replication/classification_results.csv` | Per-seed confusion matrices | 5 |
| `results/xai/iccad5_seed_replication/sample_activation_comparison.csv` | Per-sample activations (zero-vector flags) | ? |

**Critical gap:** No **full per-sample prediction table** for all 19,368 test images exists in the current results. The seed-replication artifacts only store aggregate confusion matrices, not individual image predictions with probabilities and case types.

### 1.2 Population Estimates (from seed-replication confusion matrices)

| Condition | Seed | TP | TN | FP | FN | Total |
|-----------|------|----|----|----|----|-------|
| balanced_sampling | 101 | 39 | 19,275 | 52 | 2 | 19,368 |
| balanced_sampling | 202 | 39 | 19,242 | 85 | 2 | 19,368 |
| balanced_sampling | 303 | 39 | 19,258 | 69 | 2 | 19,368 |

**Population available for expansion (seed 101, adopted checkpoint):**
- **TP**: 39
- **TN**: 19,275
- **FP**: 52
- **FN**: 2

**FN is extremely limited** (only 2 in 19,368 test set).

---

## 2. Duplicate Contamination Risk

### 2.1 TN Duplicate (Critical)

From Agent D (`DATASET_INTEGRITY_AUDIT.md`):
- **NNHSCAD5999_6** and **NNHSCAD5999_9** are **byte-identical copies** of the same raster
- Both appear in the final11 TN set (3 TN total)
- Effective unique TN rasters in final11: **2** (not 3)

### 2.2 ICCAD5 test_nhs Structure

- 19,327 files, 89% follow `.pngN.png` pattern (synthetic replicates)
- Each "stem" (e.g., `NNHSCAD5999`) produces ~3 unique rasters replicated 3× each
- Cross-stem pixel duplicates exist (different stems decode to identical 1200×1200 arrays)
- **True unique TN rasters ≈ 2,000–3,000** (not 19,327)

### 2.3 TN Duplicate in Final Set

The two flagged Grad-CAM TN samples (`NNHSCAD5999_6`, `NNHSCAD5999_9`) are **the same raster**. The third TN (`NHSCAD51003`) appears unique.

---

## 3. Expansion Design

### 3.1 Constraints

| Constraint | Impact |
|------------|--------|
| **FN population = 2** | Cannot expand beyond 2 FN samples |
| **FP population = 52** | Can expand FP substantially |
| **TN population ≈ 2,000–3,000 unique** | Can expand TN substantially after deduplication |
| **TP population = 39** | Can expand TP moderately |
| **No full per-sample predictions stored** | Must regenerate predictions (GPU) or rely on seed-replication aggregates |

### 3.2 Deterministic Selection Strategy (Predeclared)

**Goal:** Expand to target n per case type with deterministic, reproducible selection.

**Deduplication pre-step:**
1. Compute SHA256 of all 19,368 test images
2. Group by pixel-hash (decoded RGB) to identify unique rasters
3. Assign each unique raster a canonical filename (lexicographically first)
4. Build deduplicated population per case type

**Selection per case type (deterministic):**

| Case | Target n | Selection Algorithm |
|------|----------|---------------------|
| **TP** | 15 | Sort by `p_hs` descending; take top 15 unique rasters; tie-break by filename |
| **FP** | 15 | Sort by `p_hs` descending; take top 15 unique rasters; tie-break by filename |
| **TN** | 15 | Sort by `p_nhs` descending; take top 15 unique rasters; tie-break by filename |
| **FN** | 2 | All available (n=2); no selection needed |

**Total expanded: 47 samples** (vs current 11).

**Selection seed:** Fixed at 20260920 (distinct from training seeds).

**Tie-breaking:** For equal probabilities, use lexicographic filename order.

### 3.3 Required Regeneration (GPU-Gated)

Since no full per-sample prediction table exists, the expansion **requires full test-set inference** on the adopted checkpoint:
1. Load `xai_cnn_iccad5_balanced_sampling_seed1.keras`
2. Run forward pass on all 19,368 test images with corrected preprocessing (NEAREST)
3. Collect `p_hs`, `p_nhs`, predicted class, true label
4. Compute case type per sample
5. Apply deduplication + deterministic selection above
5. Save expanded sample manifest with SHA256, case type, probabilities

**Output artifacts:**
- `results/xai/improvements/sample_expansion/expanded_manifest.csv` (47 rows)
- `results/xai/improvements/sample_expansion/population_counts.json`
- `results/xai/improvements/sample_expansion/deduplication_report.json`

---

## 4. Registry Status

- **Agent I** (this audit): `IMPROVEMENT_SAMPLE_EXPANSION.md` — completed
- **Prediction regeneration**: NO-GO (GPU unavailable)
- **Registry entry**: `docs/IMPROVEMENT_TASK_REGISTRY.md` — status "existing-prediction audit/CPU design"

---

## 5. Limitations

1. **FN ceiling = 2** — cannot increase without new data or different checkpoint
2. **No full predictions stored** — expansion requires GPU forward pass on 19,368 images
3. **Deduplication changes population counts** — true unique TN ≪ 19,327; must be done before selection
4. **Seed-replication aggregates only** — cannot verify individual sample predictions without regeneration
5. **Corrected preprocessing required** — any new inference must use NEAREST (Agent B fix)

---

## 6. Files

- Source: `results/xai/final_iccad5/xai_diagnostics.csv`
- Source: `results/xai/iccad5_seed_replication/`
- Source: `docs/DATASET_INTEGRITY_AUDIT.md` (Agent D)
- This document: `docs/IMPROVEMENT_SAMPLE_EXPANSION.md`

*Audit performed by Agent I (CPU-only). No neural inference executed. Design only — execution requires GPU.*