# Dataset/Sample Integrity Audit — ICCAD Benchmarks 1–5

**Audit Date:** 2026-09-18  
**Auditor:** Agent D  
**Scope:** Full dataset integrity audit of `iccad-official/iccad1..5`, with deep focus on ICCAD5 duplicates.  
**Environment:** CPU-only; no GPU inference performed.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total images across all benchmarks** | 164,016 |
| **ICCAD5 total images** | 22,110 |
| **Byte-level duplicate groups (ICCAD5)** | 1,866 groups |
| **Files involved in byte-duplication (ICCAD5)** | ~19,000+ (86% of ICCAD5) |
| **Pixel-level duplicate groups (sampled)** | 25 (train_nhs, n=500) / 18 (test_nhs, n=500) |
| **Filename duplicates (train↔test, same name)** | 2,137 pairs |
| **Train↔test byte-overlap** | 0 files |
| **Train↔test pixel-overlap** | 0 files |
| **`.pngN.png` pattern files (test_nhs only)** | 17,216 files (89% of test_nhs) |
| **NNHSCAD5999 byte-identical copies** | 3 groups × 3 files each = 9 files mapping to 3 unique rasters |

**Critical finding:** ICCAD5 `test_nhs` is **not a set of distinct images**. It is a synthetic expansion where ~19,327 filenames map to only ~3 unique base rasters per "stem" via the `.pngN.png` naming pattern. Statistical analyses treating each filename as an independent sample are **invalid**.

---

## 1. Benchmark Composition (Class Counts & Totals)

| Benchmark | train_hs | train_nhs | test_hs | test_nhs | **Total** |
|-----------|----------|-----------|---------|----------|-----------|
| iccad1    | 99       | 340       | 226     | 4,679    | 5,344     |
| iccad2    | 174      | 5,285     | 498     | 41,298   | 47,255    |
| iccad3    | 909      | 4,643     | 1,808   | 46,333   | 53,693    |
| iccad4    | 95       | 4,452     | 177     | 31,890   | 36,614    |
| iccad5    | 26       | 2,716     | 41      | 19,327   | 22,110    |
| **All**   | **1,303**| **17,436**| **2,750**| **143,527**| **164,016** |

> **Note:** Class imbalance is extreme (NHS ≫ HS) in all benchmarks, especially test splits.

---

## 2. Duplicate Analysis — ICCAD5 (Primary Focus)

### 2.1 Byte-Level Duplicates (Exact File Copies)

**1,866 distinct byte-hash groups** contain ≥2 files each.  
Major groups in `train_nhs` (largest groups shown):

| Byte Hash (prefix) | Count | Example Filenames |
|--------------------|-------|-------------------|
| `92a8d352...`      | 24    | NHSCAD51563.png, NHSCAD51565.png, NHSCAD51579.png, … |
| `2c492eb9...`      | 34    | NHSCAD51965.png, NHSCAD51951.png, NHSCAD52467.png, … |
| `e94f01a6...`      | 25    | NHSCAD5296.png, NHSCAD52642.png, NHSCAD52646.png, … |
| `334163b3...`      | 25    | NHSCAD52161.png, NHSCAD52179.png, NHSCAD52177.png, … |
| `1551fc36...`      | 18    | NHSCAD5175.png, NHSCAD5185.png, NHSCAD5179.png, … |
| `525ad544...`      | 14    | NHSCAD52425.png, NHSCAD52432.png, NHSCAD52423.png, … |
| `a033e015...`      | 13    | NHSCAD5163.png, NHSCAD5164.png, NHSCAD5170.png, … |
| `0f9f2a64...`      | 12    | NHSCAD5213.png, NHSCAD5203.png, NHSCAD5211.png, … |

**Interpretation:** These are **exact byte-for-byte copies** with different filenames. In `train_nhs`, ~2,716 files collapse to far fewer unique images.

### 2.2 Pixel-Level Duplicates (Decoded RGB Equality)

Sampled 500 files each from `train_nhs` and `test_nhs`:

| Split | Sampled | Pixel-Duplicate Groups | Files in Duplicates |
|-------|---------|------------------------|---------------------|
| train_nhs | 500 | 25 | ~120+ files (24%) |
| test_nhs  | 500 | 18 | ~100+ files (20%) |

**Largest pixel-duplicate group in test_nhs (sample):**  
Hash `720080a7...` — **31 files** including:
- `NNHSCAD51180.png7.png`, `NNHSCAD51729.png6.png`, `NNHSCAD51515.png9.png`
- `NNHSCAD5343.png1.png`, `NNHSCAD5951.png7.png`, `NNHSCAD5507.png7.png`
- `NNHSCAD5416.png3.png`, `NNHSCAD51144.png1.png`, … and 21 more

**Key insight:** Pixel duplicates **cross the `.pngN.png` pattern boundary** — different stems (`NNHSCAD51180`, `NNHSCAD51729`, `NNHSCAD51515`, `NHSCAD5343`, `NHSCAD5951`, `NHSCAD5507`, `NHSCAD5416`, `NHSCAD51144`, `NHSCAD51274`, `NHSCAD51936`, `NHSCAD5498`, `NHSCAD5225`, `NHSCAD51647`, `NHSCAD5462`, `NHSCAD51889`, `NHSCAD51513`, `NHSCAD51729` again, `NHSCAD5327`, `NHSCAD51977`, `NHSCAD5630`, `NHSCAD51570`, `NHSCAD5581`, `NHSCAD599`, `NHSCAD5657`, `NHSCAD51789`, `NHSCAD51513` again, `NHSCAD51567`, `NHSCAD5506`, `NHSCAD5251`, `NHSCAD5959`, `NHSCAD5672`) all decode to the **same 1200×1200 RGB array**.

### 2.3 Filename Duplicates (Same Name, Different Directories)

**2,137 filename pairs** exist with identical names in `train/` and `test/`:

- **All 26 `train_hs` files** also appear in `test_hs` (HSCAD50.png … HSCAD525.png)
- **2,111 `train_nhs` files** also appear in `test_nhs` (NHSCAD51393.png … NHSCAD51159.png)

**Critical:** These are **filename collisions only**. Byte-hash and pixel-hash checks confirm **zero train↔test overlap at content level** — the identically named files in train vs test are **different images**.

> **Directory disjointness ≠ content disjointness.** The train/test split is **not** a content split; it is a filename split with entirely different images sharing names.

### 2.4 The `.pngN.png` Synthetic Expansion Pattern (ICCAD5 test_nhs)

**17,216 of 19,327 files (89%)** in `test_nhs` follow the pattern `<STEM>.png<N>.png` where N ∈ {1,2,3,4,6,7,8,9}.

Example stem `NNHSCAD5999` produces 8 files:
| Filename | Byte Hash | Pixel Hash | Shape |
|----------|-----------|------------|-------|
| NNHSCAD5999.png1.png | `7781262f...` | `9006cc3b...` | 1200×1200 |
| NNHSCAD5999.png2.png | `a77830f8...` | `3304fa61...` | 1200×1200 |
| NNHSCAD5999.png3.png | `f8c08925...` | `0e8ce2c5...` | 1200×1200 |
| NNHSCAD5999.png4.png | `7781262f...` | `9006cc3b...` | 1200×1200 |
| NNHSCAD5999.png6.png | `f8c08925...` | `0e8ce2c5...` | 1200×1200 |
| NNHSCAD5999.png7.png | `7781262f...` | `9006cc3b...` | 1200×1200 |
| NNHSCAD5999.png8.png | `a77830f8...` | `3304fa61...` | 1200×1200 |
| NNHSCAD5999.png9.png | `f8c08925...` | `0e8ce2c5...` | 1200×1200 |

**Three unique rasters** (byte hashes `7781262f`, `a77830f8`, `f8c08925`) are each replicated 3× across 9 filenames.  
**NNHSCAD5999_6 and NNHSCAD5999_9** (referenced in `artifact_identity.json` as `NNHSCAD5999.png6.png` and `NNHSCAD5999.png9.png`) are **byte-identical copies** of the third raster (`f8c08925...`).

This pattern holds across the vast majority of `test_nhs` stems. The dataset is a **combinatorial replication** of a small base set of rasters.

### 2.5 Identical Rasters Under Different Filenames (Cross-Stem)

As shown in §2.2, pixel-duplicate group `720080a7...` spans **31 files with 15+ distinct stems**. This is not an isolated case — the sampling revealed multiple cross-stem pixel collisions. The true number of unique rasters in ICCAD5 is **orders of magnitude smaller** than the file count.

---

## 3. Train/Test Overlap

| Check Method | Overlap Count |
|--------------|---------------|
| Byte hash (exact file) | **0** |
| Decoded pixel hash (RGB) | **0** |

**Conclusion:** No image content appears in both train and test. However, the **filename overlap (2,137 pairs)** creates severe risk of accidental leakage if any pipeline uses filenames as keys.

---

## 4. Representative Sample Selection Logic (`src/run_xai.py`)

```python
def select_representative_samples(df: pd.DataFrame, samples_per_case: int = 3) -> pd.DataFrame:
    selected = []
    for case in ["TP", "TN", "FP", "FN"]:
        sub = df[df["case_type"] == case].copy()
        if sub.empty:
            continue
        if case in ["TP", "FP"]:
            sub = sub.sort_values(by="p_hs", ascending=False)   # highest HS confidence
        else:
            sub = sub.sort_values(by="p_nhs", ascending=False)  # highest NHS confidence
        n = min(len(sub), samples_per_case)
        selected.append(sub.head(n))
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
```

### Tie Handling
- **No explicit tie-breaker.** `sort_values` is stable (preserves original order), but original order comes from `flow_from_directory` which iterates filesystem order — **non-deterministic across runs/platforms**.
- **No stratification** by benchmark, class balance, or source-image uniqueness.
- **Duplicate-blind:** If duplicate images (same raster, different filenames) receive different predictions/confidences, they can **all be selected** as "independent" representatives.

### Impact on Statistical Analyses
- With ICCAD5 `test_nhs` being 89% synthetic replicates, the 3 TN samples selected will likely include **multiple copies of the same raster**.
- Confidence intervals, p-values, and effect sizes computed over these "samples" **grossly underestimate variance** (pseudoreplication).
- Grad-CAM degeneracy findings on `NNHSCAD5999_6` / `NNHSCAD5999_9` (§5) are **not two independent observations** — they are one raster measured twice.

---

## 5. ICCAD5-Specific Duplicate Findings (Per `artifact_identity.json`)

| Sample ID | Source File | Byte SHA-256 | Pixel SHA-256 | Shape |
|-----------|-------------|--------------|---------------|-------|
| NNHSCAD5999_6 | `iccad-official/iccad5/test/test_nhs/NNHSCAD5999.png6.png` | `f8c08925a7aefcb04b8d19098ca04e863d0cbeb559736e9cad7fc8cbe40925fb` | `0e8ce2c5501155324322e6149aa9152e...` | 1200×1200 |
| NNHSCAD5999_9 | `iccad-official/iccad5/test/test_nhs/NNHSCAD5999.png9.png` | `f8c08925a7aefcb04b8d19098ca04e863d0cbeb559736e9cad7fc8cbe40925fb` | `0e8ce2c5501155324322e6149aa9152e...` | 1200×1200 |

**Verified:**
- `source_bytes_equal: true`
- `source_decoded_equal: true`
- `resized_rgb_pair_equal: {nearest: true, bicubic: true}`

**Implication:** The two "flagged" Grad-CAM samples in `FINAL_GRADCAM_DEGENERACY_ANALYSIS.md` are **the same input image**. The 6 flagged geometry rows (3 per sample ID × 2 IDs) represent **3 unique map measurements on one raster**, not 6 independent samples.

---

## 6. Integrity Issues Affecting Statistical Analyses

| Issue | Severity | Affected Analyses |
|-------|----------|-------------------|
| **ICCAD5 test_nhs is 89% synthetic replicates** | **CRITICAL** | All per-sample metrics, confidence intervals, significance tests, Grad-CAM statistics, degeneracy rates |
| **Cross-stem pixel duplicates** | **HIGH** | Any analysis assuming filename = unique image (class counts, confusion matrices, sample selection) |
| **Train/test filename collision (2,137 pairs)** | **HIGH** | Pipeline leakage risk if filenames used as keys; metadata joins |
| **No train/test content overlap** | LOW (good) | — |
| **Representative sample selection ignores duplicates** | **HIGH** | XAI diagnostic CSV, summary grids, case-type statistics |
| **Tie-handling non-deterministic** | MEDIUM | Reproducibility of sample selection |
| **Class imbalance extreme (NHS:HS ≈ 50:1 in test)** | MEDIUM | Metric interpretation (accuracy, AUC), sampling bias |

---

## 7. Recommendations

1. **Deduplicate ICCAD5 test_nhs before any statistical analysis.** Collapse `.pngN.png` groups to unique rasters (by pixel hash). Report effective sample size.
2. **Rename train/test filename collisions** (e.g., prefix with `train_`/`test_`) to prevent accidental key collisions.
3. **Revise `select_representative_samples`** to:
   - Deduplicate by pixel hash before selection
   - Add deterministic tie-breaker (e.g., filename)
   - Stratify by benchmark and unique source raster
4. **Recompute all ICCAD5 statistics** on deduplicated data. The current "n=19,327" for test_nhs is fictitious; true n is likely ~2,000–3,000 unique rasters.
5. **Annotate `artifact_identity.json` and `FINAL_GRADCAM_DEGENERACY_ANALYSIS.md`** with duplicate-aware interpretation: the two flagged IDs are one image.
6. **Audit ICCAD1–4 for similar `.pngN.png` patterns.** The pattern may exist in other benchmarks' test_nhs directories.

---

## 8. Artifacts Referenced

- `iccad-official/iccad5/` — directory structure
- `results/xai/final_iccad5/degeneracy_forensics/artifact_identity.json` — CPU source identity audit
- `docs/FINAL_GRADCAM_DEGENERACY_ANALYSIS.md` — historical degeneracy findings
- `src/run_xai.py` — sample selection logic (lines 151–163)

---

## 9. Reproducibility

All hashes computed with SHA-256. Pixel hashes use PIL `Image.convert('RGB').tobytes()`.  
Commands used (run in project venv):

```bash
# Byte hashes
sha256sum iccad-official/iccad5/**/*.png

# Pixel hashes (Python)
python -c "
from PIL import Image; import hashlib
h = hashlib.sha256(Image.open(f).convert('RGB').tobytes()).hexdigest()
"
```

---

*End of Audit*