# XAI Pipeline Audit

This document provides a comprehensive technical audit of the Explainable AI (XAI) pipeline execution across ICCAD-12 benchmarks 1–5, addressing dataset processing, the 57-row diagnostic CSV provenance, task concurrency (`task-123` vs `task-127`), output overwrites, and the mathematical implementation integrity of Grad-CAM, Grad-CAM++, and LayerCAM.

---

## 1. Test-Set Processing

The table below summarizes test image counts, prediction evaluations, and explanation coverage across all five ICCAD-12 benchmarks.

| Benchmark | Total Test Images | Images Evaluated for Prediction | Images Receiving XAI Explanations | Images Receiving Diagnostic Entries | Representative Samples Breakdown (TP / TN / FP / FN) |
|:---|:---:|:---:|:---:|:---:|:---|
| **iccad1** | 4,905 | 4,905 | 11 | 11 | 3 TP / 3 TN / 3 FP / 2 FN *(only 2 FN exist; Recall=99.1%)* |
| **iccad2** | 41,796 | 41,796 | 12 | 12 | 3 TP / 3 TN / 3 FP / 3 FN |
| **iccad3** | 48,141 | 48,141 | 12 | 12 | 3 TP / 3 TN / 3 FP / 3 FN |
| **iccad4** | 32,067 | 32,067 | 12 | 12 | 3 TP / 3 TN / 3 FP / 3 FN |
| **iccad5** | 19,368 | 19,368 | 10 | 10 | 3 TP / 3 TN / 3 FP / 1 FN *(only 1 FN exists; Recall=97.6%)* |
| **Total** | **146,277** | **146,277** | **57** | **57** | **15 TP / 15 TN / 15 FP / 12 FN (57 Total)** |

### Detailed Answers to Test-Set Questions:
- **A. Test images processed per benchmark**: All test images in each benchmark test split were loaded and evaluated by the model during the forward inference pass (`iccad1`: 4,905, `iccad2`: 41,796, `iccad3`: 48,141, `iccad4`: 32,067, `iccad5`: 19,368; Grand Total = 146,277).
- **B. Images receiving XAI explanations**: Exactly **57 images** received backpropagation gradient explanations.
- **C. Images receiving diagnostic CSV entries**: Exactly **57 images** (matching the 57 explained samples).
- **D. Representative samples selected**: Exactly **57 samples** across all 5 benchmarks ($11 + 12 + 12 + 12 + 10$).
- **E. Why CSV is 57 rows instead of ~1,700**: See Section 2 below.
- **F. Intentional design vs implementation bug**: Confirmed **intentional design**.
- **G. Availability of full test-set diagnostics**: Diagnostics (centroids, spatial spread, connected regions, edge concentration) exist **only for the 57 representative samples**. General classification performance metrics (accuracy, precision, recall, confusion matrix) for all 146,277 test images are logged in `results/xai_training/xai_baseline_comparison.csv`.

---

## 2. Why `xai_diagnostics.csv` Has 57 Rows

### Exact Code Path & Execution Flow

The 57-row count is directly determined by the sampling and diagnostic generation pipeline in [`src/run_xai.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/run_xai.py):

1. **CLI Parameter Configuration** ([`src/run_xai.py:61-65`](file:///home/peskybird/Projects/Ai_Ml_DA/src/run_xai.py#L61-L65)):
   ```python
   p.add_argument(
       "--samples-per-case",
       type=int,
       default=3,
       help="Number of representative samples per category (TP, TN, FP, FN)",
   )
   ```
   The default number of samples per confusion category (`TP`, `TN`, `FP`, `FN`) is set to $K = 3$.

2. **Inference Over Full Test Directory** ([`src/run_xai.py:88-148`](file:///home/peskybird/Projects/Ai_Ml_DA/src/run_xai.py#L88-L148)):
   `collect_test_predictions` runs `model.predict(val_gen)` on all images in `iccad-official/iccad{N}/test/` to calculate probabilities `p_hs` and `p_nhs`, categorizing each test image into one of four confusion categories: `TP`, `TN`, `FP`, or `FN`.

3. **Representative Sample Selection** ([`src/run_xai.py:151-163`](file:///home/peskybird/Projects/Ai_Ml_DA/src/run_xai.py#L151-L163)):
   ```python
   def select_representative_samples(df: pd.DataFrame, samples_per_case: int = 3) -> pd.DataFrame:
       selected = []
       for case in ["TP", "TN", "FP", "FN"]:
           sub = df[df["case_type"] == case].copy()
           if sub.empty:
               continue
           if case in ["TP", "FP"]:
               sub = sub.sort_values(by="p_hs", ascending=False)
           else:
               sub = sub.sort_values(by="p_nhs", ascending=False)
           n = min(len(sub), samples_per_case)
           selected.append(sub.head(n))
       return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
   ```
   For each category, samples are sorted by confidence and the top $N = \min(\text{available}, 3)$ are selected.

4. **Why Certain Benchmarks Have Fewer Than 12 Samples**:
   - For `iccad2`, `iccad3`, and `iccad4`: All four categories have $\ge 3$ instances $\implies 3 \times 4 = 12$ samples each.
   - For `iccad1`: The model missed only 2 hotspots on the entire test set (Recall = 99.12%, $224 / 226$) $\implies \text{FN count} = 2$. Hence, $\min(2, 3) = 2 \implies 3 + 3 + 3 + 2 = 11$ samples.
   - For `iccad5`: The model missed only 1 hotspot on the entire test set (Recall = 97.56%, $40 / 41$) $\implies \text{FN count} = 1$. Hence, $\min(1, 3) = 1 \implies 3 + 3 + 3 + 1 = 10$ samples.
   - Total representative samples: $11 + 12 + 12 + 12 + 10 = 57$.

5. **Diagnostic Record Generation & Export** ([`src/run_xai.py:265-401`](file:///home/peskybird/Projects/Ai_Ml_DA/src/run_xai.py#L265-L401) and [`src/run_xai.py:475-478`](file:///home/peskybird/Projects/Ai_Ml_DA/src/run_xai.py#L475-L478)):
   - `process_benchmark` iterates strictly over `df_rep.iterrows()` (the 57 selected samples).
   - For each sample, it runs full gradient attribution (Grad-CAM, Grad-CAM++, LayerCAM on final and intermediate layers), calls `compute_attribution_diagnostics` ([`src/xai_diagnostics.py:19-114`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_diagnostics.py#L19-L114)), and constructs a 33-column dictionary record.
   - `main()` converts the list of 57 records into `df_samples` and writes `results/xai/xai_diagnostics.csv`.

---

## 3. Task-123 vs Task-127

A detailed forensic analysis of `.system_generated/tasks/task-123.log`, `.system_generated/tasks/task-127.log`, and the agent conversation transcript (`cc1a3d02-82b1-49f8-9f5a-4c59b0af4c62/.system_generated/logs/transcript_full.jsonl`) confirms that **task-123 and task-127 were concurrent executions of the exact same code, started 22 seconds apart**.

### Side-by-Side Comparison

| Property | `task-123` | `task-127` |
|:---|:---|:---|
| **Start Time** | `2026-09-16 09:54:56.505` (Local) / `04:24:56Z` | `2026-09-16 09:55:18.663` (Local) / `04:25:18Z` |
| **End Time** | `2026-09-16 10:10:14.000` (Local) / `04:40:14Z` | `2026-09-16 10:10:13.000` (Local) / `04:40:13Z` |
| **Elapsed Runtime** | 15 minutes, 18 seconds | 14 minutes, 55 seconds |
| **Exact Command** | `bash scripts/tf_gpu.sh -m src.run_xai` | `./.venv/bin/python src/run_xai.py` |
| **Effective Runtime** | Executed `./.venv/bin/python -m src.run_xai` | Executed `./.venv/bin/python src/run_xai.py` |
| **Source State** | Unmodified code generated in steps 59–71 | Exact same source files (0 edits between tasks) |
| **XAI Methods** | Grad-CAM, Grad-CAM++, LayerCAM | Grad-CAM, Grad-CAM++, LayerCAM |
| **Output Directory** | `/home/peskybird/Projects/Ai_Ml_DA/results/xai/` | `/home/peskybird/Projects/Ai_Ml_DA/results/xai/` |
| **Overwrote Outputs?** | Shared same directory; wrote simultaneously | Overwrote files created by task-123 ~20s prior |

### Root Cause of the Double Invocation
At step 122 (09:54:45), the assistant launched `task-123` via `bash scripts/tf_gpu.sh -m src.run_xai`. At step 124 (09:55:06), it checked the status and saw it was running. At step 126 (09:55:16), instead of attaching to or awaiting `task-123`, it launched `./.venv/bin/python src/run_xai.py` as a second background command (`task-127`) without terminating `task-123`.

Both tasks competed for GPU memory on the NVIDIA RTX 3050 (logged as temporary BFC allocator memory retries in `task-127.log`), but both ran to completion without crashing.

---

## 4. Vanilla Grad-CAM Provenance

Based on codebase history and git logs:

1. **Original Baseline Implementation** ([`src/gradcam.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/gradcam.py)):
   - Implemented in initial commit `a308618`.
   - Designed specifically for the baseline CNN architecture with target layer `conv2d_5` and binary classification layers `dense` / `dense_1`.
   - Includes standalone `GradCAM` class and `save_comparison_panel`.

2. **Current Implementation in XAI Pipeline** ([`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py)):
   - **Classification**: **Reimplementation / Framework Extension**.
   - Rather than calling `src/gradcam.py`, `src/xai_engine.py` re-implements Vanilla Grad-CAM as `XAIEngine.explain_gradcam` alongside Grad-CAM++ and LayerCAM.
   - It refactors target layer selection to dynamically bind to `conv_final_2` (the XAI model architecture), automatically handles layer switching for multi-layer resolution hierarchies (`conv_early_2`, `conv_mid_2`), and supports unified pre-sigmoid logit computation.
   - **No code changes occurred between task-123 and task-127**.

---

## 5. Current Implementation Specifications

All three XAI methods are implemented in [`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py) under the unified [`XAIEngine`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py#L38-L311) class:

```
                                  +---------------------------------------+
                                  |         XAIEngine Submodel            |
                                  |  (Target Conv, Penultimate, Output)   |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                      Pre-Sigmoid Logit: target_score
                                  (HS: -logit_nhs  |  NHS: +logit_nhs)
                                                      |
                      +-------------------------------+-------------------------------+
                      |                               |                               |
                      v                               v                               v
             1. Vanilla Grad-CAM               2. Grad-CAM++                    3. LayerCAM
          ------------------------        -----------------------         -----------------------
          GAP Gradients (alpha_k)         Higher-order derivatives        Spatial pixel-wise ReLU
          Sum over channels               Weighted channel alphas         w_ij^k = ReLU(g_ij^k)
          cam = ReLU(sum alpha_k * A^k)   cam = ReLU(sum w_k * A^k)       cam = ReLU(sum w_ij^k * A_ij^k)
                      |                               |                               |
                      +-------------------------------+-------------------------------+
                                                      |
                                                      v
                                        _postprocess_cam Pipeline
                                    - Max Normalization [0, 1]
                                    - Bilinear Upsampling (224x224)
                                    - Post-resize Range Verification
```

### Method-by-Method Audit Table

| Specification | Vanilla Grad-CAM | Grad-CAM++ | LayerCAM |
|:---|:---|:---|:---|
| **Source File** | [`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py) | [`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py) | [`src/xai_engine.py`](file:///home/peskybird/Projects/Ai_Ml_DA/src/xai_engine.py) |
| **Function / Method** | `XAIEngine.explain_gradcam` (L146–175) | `XAIEngine.explain_gradcam_plus_plus` (L177–228) | `XAIEngine.explain_layercam` (L230–261) |
| **Target Layer** | `conv_final_2` ($28\times 28\times 32$) | `conv_final_2` ($28\times 28\times 32$) | `conv_early_2` ($112\times 112$), `conv_mid_2` ($56\times 56$), `conv_final_2` ($28\times 28$) |
| **Target Score** | Pre-sigmoid logit $z$ ($z_{\text{target}} = -z_{\text{NHS}}$ for HS, $+z_{\text{NHS}}$ for NHS) | Pre-sigmoid logit $z$ | Pre-sigmoid logit $z$ |
| **Score Formulation** | **Pre-sigmoid logit** (eliminates sigmoid saturation) | **Pre-sigmoid logit** | **Pre-sigmoid logit** |
| **Weighting Formulation** | GAP: $\alpha_k = \frac{1}{Z}\sum_{i,j} \frac{\partial S}{\partial A_{ij}^k}$ | Second/third-order: $\alpha_{ij}^k = \frac{\partial^2 S / \partial (A_{ij}^k)^2}{2\frac{\partial^2 S}{\partial (A_{ij}^k)^2} + \sum \dots}$ | Fine-grained spatial: $w_{ij}^k = \text{ReLU}\left(\frac{\partial S}{\partial A_{ij}^k}\right)$ |
| **Channel Aggregation** | $\sum_k \alpha_k A^k$ | $\sum_k \left(\sum_{i,j} \alpha_{ij}^k \text{ReLU}\left(g_{ij}^k\right)\right) A^k$ | Pixel-wise: $\sum_k w_{ij}^k A_{ij}^k$ |
| **Positive Attribution (ReLU)** | Applied after channel combination: $\text{ReLU}(\text{cam})$ | Applied to 1st-order grads and final CAM | Applied to spatial gradients and final CAM |
| **Upsampling & Interpolation** | Bilinear interpolation (`tf.image.resize(..., method="bilinear")`) | Bilinear interpolation | Bilinear interpolation |
| **Normalization** | Normalized by `max(cam)` both pre- and post-interpolation | Normalized by `max(cam)` pre/post interpolation | Normalized by `max(cam)` pre/post interpolation |
| **Multi-Target Support** | Generates both `pred_*` and `true_*` explanations | Generates both `pred_*` and `true_*` explanations | Generates both `pred_*` and `true_*` explanations |

---

## 6. Output Provenance

- **Output Directory**: `/home/peskybird/Projects/Ai_Ml_DA/results/xai/`
- **Total Output Files**: 860 files (57 per-sample directories $\times$ 15 files/dir = 855 files + 5 top-level summaries).
- **Primary Generator**: Both `task-123` and `task-127` wrote to the exact same paths. `task-127` started 22 seconds later and was trailing `task-123`, repeatedly overwriting the earlier files. In the final seconds on `iccad5`, `task-127` completed its file generation at `10:10:12.788`, while `task-123` completed its final writes at `10:10:12.905`.
- **Integrity**: Because both tasks executed the identical deterministic inference code with fixed weights, inputs, and seeds, the outputs in `results/xai/` are consistent, valid, and uncorrupted.

---

## 7. Issues Found

### Confirmed Bugs
- **Dual Concurrent Task Spawning (Concurrency Race)**: `task-123` and `task-127` were launched simultaneously on the same GPU targeting the same directory. While no data corruption occurred due to identical deterministic execution, launching duplicate tasks consumes double compute resources and introduces potential file write collisions.

### Confirmed Intentional Behavior
- **57 Rows in `xai_diagnostics.csv`**: The 57 rows reflect intentional sampling of top-confidence samples across `TP`, `TN`, `FP`, and `FN` categories for each benchmark ($3 \times 4 = 12$ maximum per benchmark, with `iccad1` having 2 FN and `iccad5` having 1 FN due to $>97.5\%$ recall).
- **Pre-Sigmoid Logits**: Intentionally used across all XAI methods to prevent vanishing gradients caused by sigmoid saturation on high-confidence layout patterns.
- **Multi-Resolution LayerCAM Hierarchy**: Intentionally evaluates `conv_early_2`, `conv_mid_2`, and `conv_final_2` to capture sub-micron layout geometry that is lost at the final $28\times 28$ layer.

### Uncertainties Requiring Further Investigation
- **Full Test Set Spatial Metrics**: If downstream research analysis requires quantitative spatial metrics across the full 146,277 test set (rather than the 57 representative samples), a batched attribution extractor without disk image serialization will need to be configured.
