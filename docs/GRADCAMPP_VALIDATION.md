# Grad-CAM++ Validation — Agent B

**Audit Date:** 2026-09-18  
**Scope:** CPU-only verification of `src/xai_engine.py:177–228` (`explain_gradcam_plus_plus`) against the published Grad-CAM++ formulation (Chattopadhay et al., WACV 2018, arXiv:1710.11063). No neural inference performed; CPU-only symbolic/array gradient tests only.

---

## 1. Published Formulation Summary (Chattopadhay et al., WACV 2018)

For a target class score $Y^c$ and convolutional activation map $A^k_{ij}$ (channel $k$, spatial position $i,j$):

**First derivative:**  
$g^1_{ij}^k = \frac{\partial Y^c}{\partial A^k_{ij}}$

**Second derivative:**  
$g^2_{ij}^k = \frac{\partial^2 Y^c}{\partial (A^k_{ij})^2}$

**Third derivative:**  
$g^3_{ij}^k = \frac{\partial^3 Y^c}{\partial (A^k_{ij})^3}$

**Alpha weighting (per-pixel, per-channel):**  
$\alpha^k_{ij} = \frac{g^2_{ij}^k}{2 g^2_{ij}^k + \sum_{a,b} A^k_{ab} g^3_{ab}^k + \epsilon}$

**Channel weights:**  
$w_k = \sum_{i,j} \alpha^k_{ij} \cdot \text{ReLU}(g^1_{ij}^k)$

**CAM:**  
$L^c_{GradCAM++} = \text{ReLU}\left(\sum_k w_k A^k\right)$

---

## 2. Implementation Audit (src/xai_engine.py:177–228)

| # | Component | Paper | Implementation | Status | Evidence |
|---|-----------|-------|----------------|--------|----------|
| 1 | Target score | $Y^c$ (class score) | `target_score = -logit_nhs` (HS) or `+logit_nhs` (NHS) via `get_pre_sigmoid_logit` | ✅ MATCH | Lines 191–192; pre-sigmoid logit used |
| 2 | First derivative $g^1$ | $\partial Y^c / \partial A^k_{ij}$ | `tape_grad1.gradient(target_score, conv_outputs)` | ✅ MATCH | Line 192; persistent tape records |
| 3 | Second derivative $g^2$ | $\partial^2 Y^c / \partial (A^k_{ij})^2$ | `tape_grad2.gradient(grads_1, conv_outputs)` | ⚠️ DEVIATION | Line 194; computes Jacobian of $g^1$ w.r.t. $A$, not diagonal Hessian |
| 4 | Third derivative $g^3$ | $\partial^3 Y^c / \partial (A^k_{ij})^3$ | `tape_grad2.gradient(grads_2, conv_outputs)` | ⚠️ DEVIATION | Line 195; `grads_2` was produced outside `tape_grad2` context |
| 5 | Alpha denominator | $2g^2 + \sum_{a,b} A^k_{ab} g^3_{ab}^k + \epsilon$ | `2.0 * g2 + tf.reduce_sum(conv_out * g3, axis=(0,1), keepdims=True)` | ⚠️ PARTIAL | Line 214; uses `g3` from flawed computation; sum over spatial dims correct |
| 6 | Alpha weighting | $g^2 / \text{denom}$ | `g2 / denom` with `tf.where(denom != 0, denom, 1e-10)` | ✅ MATCH | Lines 214–216 |
| 7 | Channel weights | $\sum_{i,j} \alpha^k_{ij} \text{ReLU}(g^1_{ij}^k)$ | `tf.reduce_sum(alphas * tf.maximum(g1, 0.0), axis=(0,1))` | ✅ MATCH | Lines 218–220 |
| 8 | CAM | $\text{ReLU}(\sum_k w_k A^k)$ | `tf.maximum(conv_out @ weights[..., tf.newaxis], 0.0)` | ✅ MATCH | Lines 222–224 |
| 9 | Post-processing | max-norm → bilinear 28→224 → max-norm | Shared `_postprocess_cam` | ✅ MATCH | Lines 226, 271–281 |

---

## 3. Critical Deviations

### 3.1 Second Derivative Computation (Line 194)
**Paper:** $g^2_{ij}^k = \frac{\partial^2 Y^c}{\partial (A^k_{ij})^2}$ — diagonal of Hessian per channel per pixel.  
**Implementation:** `tape_grad2.gradient(grads_1, conv_outputs)` computes the Jacobian of the vector $g^1$ with respect to the activation tensor $A$. This produces a tensor of shape `(1, H, W, C, H, W, C)` (or reduced), not the diagonal second derivatives.  
**Impact:** The computed `grads_2` is not the per-pixel second derivative required by the alpha formula. The subsequent `grads_3` compounds this error.

### 3.2 Third Derivative Computation (Line 195)
**Paper:** $g^3_{ij}^k = \frac{\partial^3 Y^c}{\partial (A^k_{ij})^3}$  
**Implementation:** `tape_grad2.gradient(grads_2, conv_outputs)` — but `grads_2` was computed *outside* the `tape_grad2` context (the persistent tape recorded only the `grads_1` computation). The operation producing `grads_2` was not recorded, so this gradient is either zero or undefined.  
**Implementation fallback (lines 201–205):** Missing derivatives silently replaced with `tf.zeros_like(grads_1)`.  
**Impact:** `grads_3` is effectively zero everywhere; the denominator term $\sum A \cdot g^3$ collapses to zero, making $\alpha^k_{ij} = \frac{g^2}{2g^2 + \epsilon} \approx 0.5$ where $g^2 \neq 0$. This reduces Grad-CAM++ to a scaled version of a non-standard weighting.

### 3.3 Alpha Weighting Reduction
With $g^3 \approx 0$, the alpha formula simplifies to:
$$\alpha^k_{ij} \approx \frac{g^2_{ij}^k}{2 g^2_{ij}^k + \epsilon}$$
This is **not** the published Grad-CAM++ weighting. The published method's key innovation — using third-order information to handle multiple instances — is lost.

---

## 4. CPU Numerical Sanity Check (Toy Tensor)

```python
# Verification approach (conceptual; requires TF env)
# 1. Construct a small toy CNN with known analytic gradients
# 2. Compute Grad-CAM++ via implementation
# 3. Compute expected alpha via finite differences of the target score
# 4. Compare channel weights and CAM output
```
*Note: Full numerical check requires TF environment; not executed due to GPU/venv constraints in subagent.*

---

## 5. Conclusion

| Aspect | Verdict |
|--------|---------|
| **First derivative** | ✅ Correct |
| **Target score (pre-sigmoid logit)** | ✅ Correct |
| **Second derivative** | ❌ Incorrect — computes Jacobian, not diagonal Hessian |
| **Third derivative** | ❌ Incorrect — tape context error; silently zeroed |
| **Alpha weighting** | ❌ Reduced to $\frac{g^2}{2g^2+\epsilon}$ (not published formula) |
| **Channel weights & CAM** | ✅ Correct *given* the flawed alpha |

**Overall:** The current `explain_gradcam_plus_plus` **does not implement the published Grad-CAM++ algorithm**. It computes a related but distinct attribution method whose alpha weights lack the third-order correction that is the method's defining feature.

---

## 6. Impact on Existing Results

All Grad-CAM++ results in the repository (geometry analyses, occlusion correlations, degeneracy statistics, final ICCAD5 synthesis) were produced by this implementation. They **cannot be attributed to the published Grad-CAM++ method** and should be labeled as "Grad-CAM++ (legacy implementation)" or similar until fixed.

---

## 7. Recommended Fix

1. Use `tf.hessians` or explicit double-gradient tape nesting to compute diagonal second derivatives per channel per pixel.
2. For third derivatives, nest a third persistent tape or use forward-mode autodiff.
3. Alternatively, implement the simplified "Grad-CAM++ with positive gradients only" variant (Lerma & Lucas, 2022) which avoids third derivatives and has a verified equivalence proof.
4. Add a numerical verification test (`scripts/improvements/gradcampp_validation.py`) comparing against finite differences on a toy model.
5. Re-run all Grad-CAM++ dependent analyses (geometry, occlusion, degeneracy) after fix.

---

*Audit performed by Agent B (CPU-only). No neural inference executed. Implementation source: `src/xai_engine.py` commit prior to any fixes.*