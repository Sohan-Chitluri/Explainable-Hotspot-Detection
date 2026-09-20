# XAI Agreement & Concentration Metrics Extension

**Purpose:** Define additional robust metrics for XAI attribution-map agreement (between two methods) and concentration (within a single map). All formulas are explicit, computational steps are specified, and degeneracy cases are handled. No composite score is produced; every metric is reported separately per case type (TP/TN/FP/FN).

---

## 1. Notation & Conventions

- Let `A, B ∈ [0,1]^{H×W}` be two attribution maps (same spatial resolution).
- Let `a = A.flatten()`, `b = B.flatten()` be their vectorized forms (length `N = H·W`).
- Pixel indices `i = 0 … N-1`.
- For any map `M`, define its **mass** `m(M) = Σ_i M_i`.
- Define **normalized mass distribution** `p(M)_i = M_i / m(M)` when `m(M) > 0`; otherwise `p(M)` is undefined (degenerate).
- **Non-zero mask**: `nz(M) = { i | M_i > 0 }`, `nz_union = nz(A) ∪ nz(B)`, `nz_intersect = nz(A) ∩ nz(B)`.
- **Top-K mask** for map `M` at fraction `K ∈ (0,1]`: sort pixels by `M_i` descending, take smallest prefix whose cumulative sum ≥ `K·m(M)`. If `m(M) = 0`, mask is empty.
- **Connected components**: 4-connectivity (von Neumann neighborhood) unless stated otherwise. Threshold `τ ∈ [0,1]` applied to the *original* `[0,1]` map (not normalized).
- **Case types**: Every metric is computed and reported separately for TP, TN, FP, FN subsets. No pooling across case types.

---

## 2. Agreement Metrics (Pairwise: A vs B)

### 2.1 Top-K Overlap (Area-Based)

For `K ∈ {0.01, 0.05, 0.10, 0.20}`:

1. Compute top-K masks `M_A(K)`, `M_B(K)` (boolean arrays of shape `H×W`).
2. Intersection over Union:
   ```
   IoU_K(A,B) = |M_A(K) ∧ M_B(K)| / |M_A(K) ∨ M_B(K)|
   ```
   where `|·|` is pixel count. If denominator = 0 (both maps have zero mass), define `IoU_K = 1.0` (trivial agreement on empty support).

3. **Symmetric overlap fraction** (alternative, more interpretable):
   ```
   Overlap_K(A,B) = |M_A(K) ∧ M_B(K)| / min(|M_A(K)|, |M_B(K)|)
   ```
   Returns `1.0` if smaller top-K region is fully contained in the larger. If `min(|M_A|,|M_B|) = 0`, define `Overlap_K = 1.0`.

4. **Centroid distance** (in pixels, for top-K regions):
   ```
   c_A(K) = centroid of M_A(K)  (mean of pixel coordinates where mask is True)
   c_B(K) = centroid of M_B(K)
   centroid_dist_K = ||c_A(K) - c_B(K)||_2
   ```
   If either mask empty, return `NaN` (record as missing, do not impute).

**Reporting:** For each `K ∈ {1%, 5%, 10%, 20%}`, report `IoU_K`, `Overlap_K`, `centroid_dist_K` per sample, then median/IQR per case type.

---

### 2.2 Non-Zero Attribution Correlation

Restrict to pixels where at least one map claims signal:

1. Index set `S = nz_union = { i | A_i > 0 or B_i > 0 }`.
2. If `|S| < 2`, correlation undefined → report `NaN`.
3. Extract sub-vectors `a_S`, `b_S` (values at indices in `S`).
4. **Pearson on non-zero union**:
   ```
   ρ_P_nz = corr(a_S, b_S)  [standard Pearson]
   ```
5. **Spearman on non-zero union**:
   ```
   ρ_S_nz = corr(rank(a_S), rank(b_S))  [Spearman on ranks of non-zero values]
   ```
   Use average ranks for ties.

**Why:** Removes inflation from shared background zeros (see Audit §5). If both maps are sparse and agree on zero locations, full-map Pearson is misleadingly high; this metric measures agreement *where signal exists*.

---

### 2.3 Rank-Based Overlap (Spearman on Full Flattened Maps)

Full-map Spearman is invariant to monotonic transforms and less sensitive to zero-inflation than Pearson.

1. Flatten both maps: `a = A.flatten()`, `b = B.flatten()`.
2. Compute ranks with average tie-handling: `r_a = rank(a)`, `r_b = rank(b)` (1-indexed, length `N`).
3. **Spearman correlation**:
   ```
   ρ_S_full = 1 - (6 · Σ_i (r_a[i] - r_b[i])^2) / (N · (N^2 - 1))
   ```
   Equivalent to Pearson on ranks. Handles ties correctly.
4. **Top-K rank agreement** (complementary to area-based overlap):
   - Let `T_A(K)` = indices of top `⌈K·N⌉` ranks in `r_a`.
   - Let `T_B(K)` = indices of top `⌈K·N⌉` ranks in `r_b`.
   - Rank-overlap `K`: `|T_A(K) ∩ T_B(K)| / |T_A(K)|`.

**Reporting:** `ρ_S_full` per sample, median/IQR per case type. Rank-overlap at same `K` as §2.1.

---

### 2.4 Cosine Similarity (Full & Non-Zero)

1. **Full-map cosine**:
   ```
   cos_full = (a · b) / (||a||_2 · ||b||_2)
   ```
   If `||a||=0` or `||b||=0`, define `cos_full = 1.0` if both zero, else `0.0`.

2. **Non-zero union cosine**: same formula on `a_S, b_S` from §2.2. If `|S| < 1`, `NaN`.

**Note:** Cosine is scale-invariant but not shift-invariant; Pearson/Spearman are preferred for pattern agreement. Included for completeness.

---

## 3. Concentration Metrics (Single Map A)

All metrics below are computed per map, then summarized per case type.

### 3.1 Shannon Entropy of Attribution Distribution

Treat normalized mass `p_i = A_i / m(A)` as a probability distribution (when `m(A) > 0`).

```
H(A) = - Σ_i p_i · log(p_i)       (natural log, units = nats)
```

- If `m(A) = 0`: `H = 0` (degenerate, no information).
- Maximum entropy for `N` pixels: `log(N)` (uniform distribution).
- **Normalized entropy**: `H_norm = H(A) / log(N)` ∈ `[0,1]`.
- **Effective support size (perplexity)**:
  ```
  perplexity = exp(H(A))         [effective number of pixels carrying mass]
  perplexity_frac = perplexity / N
  ```

**Interpretation:** Low entropy/perplexity → concentrated attribution. High → diffuse.

---

### 3.2 Effective Attribution Area (Equivalent Area)

```
A_eff = exp(H(A))                (same as perplexity)
A_eff_px = A_eff                  [in pixels]
A_eff_frac = A_eff / N            [fraction of image]
```

Equivalently, the area of a uniform distribution with the same entropy. Computationally identical to §3.1 perplexity; reported separately for clarity.

---

### 3.3 Minimum Area for Cumulative Mass Fractions

For target mass fractions `F ∈ {0.50, 0.80, 0.90}`:

1. Sort pixels by `A_i` descending: `A_(1) ≥ A_(2) ≥ … ≥ A_(N)`.
2. Compute cumulative mass: `C_k = Σ_{j=1..k} A_(j)`.
3. Find smallest `k*` such that `C_k* ≥ F · m(A)`.
4. **Minimum area**: `Area_F = k*` (in pixels), `Area_F_frac = k* / N`.

If `m(A) = 0`: `Area_F = 0`, `Area_F_frac = 0`.

**Why this over thresholding:** Thresholding at a fixed value (e.g., 0.5) is resolution- and scale-dependent. Mass-fraction area is intrinsic to the map's distribution and comparable across methods.

---

### 3.4 Connected Components (Explicit Threshold & Connectivity)

For a given threshold `τ ∈ [0,1]` (recommended defaults: `τ ∈ {0.1, 0.25, 0.5, 0.75}`):

1. Binarize: `B_τ = (A ≥ τ)`.
2. Label connected components with **4-connectivity** (von Neumann: up/down/left/right). *Rationale:* 4-connectivity prevents diagonal bridging across single-pixel gaps, matching the geometry analysis's distance-transform proxies.
   - Alternative: 8-connectivity available as option; report which used.
3. Let components be `C_1, …, C_C` where `C = count`.
4. Component areas: `area_j = |C_j|` (pixel count).
5. **Metrics**:
   - `cc_count_τ = C`
   - `cc_largest_area_px_τ = max_j area_j` (0 if `C=0`)
   - `cc_largest_area_frac_τ = cc_largest_area_px_τ / N`
   - `cc_areas_sorted_τ = sorted(area_j, reverse=True)` (for full distribution)
   - `cc_mass_fraction_τ = (Σ_{i∈C_1} A_i) / m(A)`  [mass in largest component]

**Degeneracy:** If `m(A) = 0` or `C = 0`, all component metrics = 0; `cc_mass_fraction = NaN`.

**Reporting:** For each `τ`, report `cc_count`, `cc_largest_area_frac`, `cc_mass_fraction` per sample. Also report at the **adaptive threshold** `τ* = max(A) / 2` (half-maximum) to capture the peak structure regardless of absolute scale.

---

### 3.5 Gini Coefficient of Attribution Distribution

Inequality measure: 0 = perfectly uniform, 1 = all mass in one pixel.

```
G(A) = (Σ_i Σ_j |A_i - A_j|) / (2 · N · m(A))      for m(A) > 0
     = 0                                              for m(A) = 0
```

Efficient computation (O(N log N) via sorting):
1. Sort `A` ascending: `A_(1) ≤ A_(2) ≤ … ≤ A_(N)`.
2. `G = (2 · Σ_{i=1..N} i · A_(i)) / (N · m(A)) - (N+1)/N`.

**Why:** Complementary to entropy; more sensitive to extreme concentration (single hot pixel vs. few hot pixels).

---

### 3.6 Spatial Dispersion Metrics

**3.6.1 Weighted Centroid** (already in `xai_diagnostics.py`):
```
c_x = Σ_i x_i · A_i / m(A),   c_y = Σ_i y_i · A_i / m(A)
```

**3.6.2 Weighted Covariance Matrix** (2×2):
```
Σ_xx = Σ_i (x_i - c_x)^2 · A_i / m(A)
Σ_yy = Σ_i (y_i - c_y)^2 · A_i / m(A)
Σ_xy = Σ_i (x_i - c_x)(y_i - c_y) · A_i / m(A)
```

**3.6.3 Principal Axes (Elliptical Spread):**
Eigenvalues `λ_1 ≥ λ_2 ≥ 0` of `[[Σ_xx, Σ_xy], [Σ_xy, Σ_yy]]`.
- `major_axis = 2 · sqrt(λ_1)`   [≈ 2σ length]
- `minor_axis = 2 · sqrt(λ_2)`
- `eccentricity = sqrt(1 - λ_2/λ_1)`  (0 = circular, 1 = line)
- `orientation = 0.5 · atan2(2·Σ_xy, Σ_xx - Σ_yy)`  (radians, -π/2 to π/2)

If `m(A) = 0`: all zero.

**Why:** Characterizes elongation and orientation of the attribution blob — relevant for layout geometry (lines, corners).

---

## 4. Computational Pipeline (Per Sample Pair)

```
Input: Two attribution maps A, B (H×W, [0,1]), case_type ∈ {TP,TN,FP,FN}

1. Validate shapes match; clip to [0,1]; flatten to vectors a,b.

2. Agreement metrics (§2):
   - For K in {0.01, 0.05, 0.10, 0.20}:
       * Top-K masks, IoU_K, Overlap_K, centroid_dist_K
       * Rank-overlap_K
   - Non-zero union Pearson (ρ_P_nz) & Spearman (ρ_S_nz)
   - Full-map Spearman (ρ_S_full)
   - Full & non-zero cosine

3. Concentration metrics for A (§3) and independently for B:
   - Entropy H, H_norm, perplexity, perplexity_frac
   - Area_50, Area_80, Area_90 (px and frac)
   - For τ in {0.1, 0.25, 0.5, 0.75, half-max}:
       * cc_count, cc_largest_area_frac, cc_mass_fraction
   - Gini coefficient G
   - Weighted covariance eigenvalues, eccentricity, orientation

4. Output: Structured dict with all values + case_type tag.
```

---

## 5. Degeneracy Handling Summary

| Condition | Entropy | Perplexity | Area_F | CC metrics | Gini | Correlations |
|-----------|---------|------------|--------|------------|------|--------------|
| `m(A)=0`  | 0       | 0          | 0      | 0 / NaN    | 0    | 1.0 (both zero) / 0.0 (one zero) |
| `|S|<2`   | —       | —          | —      | —          | —    | NaN |
| Single component | — | — | — | largest=total | — | — |

All `NaN` values are propagated to per-case-type summaries (median skips `NaN`; count of valid samples reported alongside).

---

## 6. Per-Case-Type Reporting Format

For each case type `C ∈ {TP, TN, FP, FN}`, report:

```
Metric_Name:
  median: X.XXXX
  iqr: [Q1, Q3]
  n_valid: N / N_total
  n_nan: M
```

No aggregation across case types. No single composite index.

---

## 7. Implementation Notes

- **Numerical stability:** Use `float64` for accumulation (entropy, covariance). Clip `p_i` to `≥ 1e-15` before `log`.
- **Tie handling in ranks:** Use `scipy.stats.rankdata(method='average')` or equivalent.
- **Connected components:** `scipy.ndimage.label(structure=[[0,1,0],[1,1,1],[0,1,0]])` for 4-connectivity.
- **Performance:** All metrics O(N log N) or O(N) per map; trivial for 224×224 on CPU.
- **No GPU required:** Pure NumPy/SciPy.

---

## 8. Integration with Existing Diagnostics

These metrics **extend**, not replace, `compute_attribution_diagnostics` in `src/xai_diagnostics.py`. Recommended new function:

```python
def compute_agreement_concentration_metrics(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
    case_type: str,
    topk_fractions: tuple = (0.01, 0.05, 0.10, 0.20),
    cc_thresholds: tuple = (0.1, 0.25, 0.5, 0.75),
) -> Dict[str, float]:
    ...
```

Returns flat dict with keys prefixed by `agreement_` or `concentration_a_` / `concentration_b_` plus `case_type`.

---

## 9. What This Design Does NOT Do

- ❌ No single "XAI quality score" or weighted composite.
- ❌ No p-values or significance tests (spatial autocorrelation invalidates i.i.d. assumptions).
- ❌ No cross-case-type pooling.
- ❌ No threshold optimization (thresholds are fixed/reported, not tuned).
- ❌ No visual outputs.

All metrics are descriptive, separable, and interpretable individually.

---

## 10. Reference Implementation Sketch (Key Formulas)

```python
import numpy as np
from scipy import ndimage, stats

def topk_mask(heatmap, k_frac):
    """Boolean mask of smallest top-K fraction by cumulative mass."""
    flat = heatmap.ravel()
    m = flat.sum()
    if m == 0:
        return np.zeros_like(flat, dtype=bool)
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    k_pixels = np.searchsorted(cum, k_frac * m, side='left') + 1
    mask = np.zeros_like(flat, dtype=bool)
    mask[order[:k_pixels]] = True
    return mask.reshape(heatmap.shape)

def entropy(heatmap):
    flat = heatmap.ravel()
    m = flat.sum()
    if m == 0:
        return 0.0, 0.0, 1.0
    p = flat / m
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    H_norm = H / np.log(len(flat))
    perp = np.exp(H)
    return H, H_norm, perp

def area_for_mass_fraction(heatmap, frac):
    flat = heatmap.ravel()
    m = flat.sum()
    if m == 0:
        return 0, 0.0
    order = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[order])
    k = np.searchsorted(cum, frac * m, side='left') + 1
    return int(k), k / len(flat)

def gini(heatmap):
    flat = heatmap.ravel()
    m = flat.sum()
    if m == 0:
        return 0.0
    sorted_vals = np.sort(flat)
    n = len(sorted_vals)
    return (2 * np.sum((np.arange(n) + 1) * sorted_vals)) / (n * m) - (n + 1) / n

def cc_metrics(heatmap, threshold, connectivity=4):
    if connectivity == 4:
        struct = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    else:
        struct = np.ones((3,3), dtype=bool)
    binary = heatmap >= threshold
    labeled, count = ndimage.label(binary, structure=struct)
    if count == 0:
        return 0, 0.0, np.nan
    areas = ndimage.sum(binary, labeled, range(1, count+1))
    largest = int(np.max(areas))
    # mass fraction in largest component
    largest_label = np.argmax(areas) + 1
    mass_in_largest = np.sum(heatmap[labeled == largest_label])
    mass_frac = mass_in_largest / heatmap.sum()
    return count, largest / heatmap.size, mass_frac

def weighted_covariance(heatmap):
    h = heatmap.astype(np.float64)
    m = h.sum()
    if m == 0:
        return np.zeros((2,2))
    H, W = h.shape
    y, x = np.indices((H, W), dtype=np.float64)
    cx = np.sum(x * h) / m
    cy = np.sum(y * h) / m
    dx = x - cx
    dy = y - cy
    Sxx = np.sum(dx*dx * h) / m
    Syy = np.sum(dy*dy * h) / m
    Sxy = np.sum(dx*dy * h) / m
    return np.array([[Sxx, Sxy], [Sxy, Syy]])
```

---

*End of specification. Implement as a new module `src/xai_agreement_concentration.py` with unit tests against synthetic maps (uniform, single-pixel, two-separated-peaks, gradient).*