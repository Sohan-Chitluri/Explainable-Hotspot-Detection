# Final-checkpoint Grad-CAM degeneracy: evidence and blocked probe

## Status and scope (2026-09-17)

**Historical map audit completed; direct final-checkpoint gradient probe NOT executed.** No neural inference, training, checkpoint modification, or shared-source edits were performed by workstream D. Execution must await lead authorization, scheduling, and workstream B's corrected-preprocessing prediction validation.

The immutable checkpoint is `models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras`, SHA-256:

```
9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70
```

D verified this file hash by reading bytes, **not by loading the neural model**. Existing artifacts are attributed to this checkpoint by `FINAL_ICCAD5_SYNTHESIS.md`; a matching present-day checkpoint hash does not independently authenticate every historical execution.

Orientation used the existing graphify graph and queried Grad-CAM/final_iccad5 degeneracy. Its mechanism-related nodes were treated as pointers to earlier documents, not fresh measurement. This report audits the current engine, CSVs, and saved images directly. It supplements, and does not edit, the final report or synthesis.

## 1. Measured historical artifact findings

Source files:

- `results/xai/final_iccad5/xai_diagnostics.csv`
- `results/xai/final_iccad5/occlusion/occlusion_diagnostics.csv`
- `results/xai/final_iccad5/geometry/geometry_diagnostics.csv`
- Saved `iccad5/*/pred_gradcam.png` and `original.png`

The reproducible CPU command below rereads the CSVs, hashes the three sources and checkpoint, and writes `results/xai/final_iccad5/degeneracy_forensics/historical_audit.json`:

```
.venv/bin/python -B scripts/improvements/degeneracy_forensics.py historical
```

It returned **132 geometry rows, 6 flagged rows**, with IDs **NNHSCAD5999_6 and NNHSCAD5999_9**, and `inference_performed: false`. There are 11 representative samples (TP=3, TN=3, FP=3, FN=2), 44 distinct sample-method combinations, and 2 flagged combinations. All flagged rows are Grad-CAM on historical TNs, each repeated at top-10%, 20%, and 30%.

Thus 6/132 = 4.55% of rows; 6/33 = 18.18% of Grad-CAM rows; equivalently 2/11 Grad-CAM sample maps, and 2/3 TN sample maps. **The six rows are not six independent failures.** Other methods and the other sample IDs have no flat-map flag in this CSV; this is not a guarantee of their faithfulness or numerical accuracy.

| Historical TN ID | Flat Grad-CAM flag | CSV p(NHS), rounded | Grad-CAM spread (px) | Connected regions | Pearson vs occlusion: Grad-CAM / Grad-CAM++ / LayerCAM |
|---|---|---:|---:|---:|---|
| NNHSCAD5999_9 | True at 10/20/30% | 1.0 | 0.00 | 0 | 0.0000 / 0.3379 / 0.2070 |
| NNHSCAD5999_6 | True at 10/20/30% | 1.0 | 0.00 | 0 | 0.0000 / 0.3379 / 0.2070 |
| NHSCAD51003 | False at all levels | 1.0 | 80.99 | 2 | -0.0197 / -0.2006 / -0.0683 |

Historical occlusion target-score values for the flagged pair are both 17.5121 (rounded). A probability reported as 1.0 is not proof of an exactly saturated numerical sigmoid, a zero logit gradient, or any particular internal mechanism. The clean TN also has CSV p(NHS)=1.0.

### Saved-image checks and duplicate source evidence

A CPU pixel audit measured the two flagged **`pred_gradcam.png`** files as 224×224 RGB, each containing exactly one RGB color, `[0, 0, 127]`; the clean TN's Grad-CAM PNG has 83 unique colors. This is a colormapped visualization, **not an occlusion map and not an unnormalized CAM tensor**. The low-end jet color is consistent with the CSV flat-map evidence, but uint8 color quantization cannot recover exact floating-point saliency or gradients.

The two saved `original.png` files are byte-identical, both SHA-256:

```
5f940cc89ec14fa209461d50d81d5817f827f8247039caf92d847eff449e896c
```

The original dataset filenames are **not** `NNHSCAD5999_6.png` and `NNHSCAD5999_9.png`; they are:

```
iccad-official/iccad5/test/test_nhs/NNHSCAD5999.png6.png
iccad-official/iccad5/test/test_nhs/NNHSCAD5999.png9.png
```

Both source files have SHA-256 `f8c08925a7aefcb04b8d19098ca04e863d0cbeb559736e9cad7fc8cbe40925fb`, identical decoded 1200×1200 arrays, and identical 224×224 RGB outputs under separately checked PIL nearest and bicubic resizing. Therefore the flagged pair represents **two dataset IDs but one byte-distinct input layout**. Keep both IDs for historical reproduction, but do not count their identical behavior as independent replication. These CPU checks concern identity and preprocessing only; no corrected predictions were calculated. Their machine-readable record is `degeneracy_forensics/artifact_identity.json`.

### What was and was not saved

The historical final_iccad5 tree contains 11 numerical `.npy` files, all `raw_occlusion_map.npy`. It does not retain numerical Grad-CAM arrays, feature activations, GAP weights, gradients, or signed pre-ReLU CAMs. `run_xai.py:293–307` saves color-rendered CAM PNGs; `run_occlusion.py:267–268` saves numerical occlusion arrays. D did not invert a colormap to pretend to reconstruct Grad-CAM tensors.

The geometry flag is a **historical measurement of resized, normalized map range**, `np.ptp(amap) < 1e-6` (`run_geometry_analysis.py:145–163`), not a gradient measurement. The engine recomputed maps for geometry from saved `original.png` (`run_geometry_analysis.py:109–121`). The raw numerical maps from that computation were not preserved.

`src/occlusion.py:44–77` returns zero Pearson/Spearman when either map has standard deviation below `1e-8`, and uses guarded cosine/IoU calculations. Consequently the flagged pair's reported zero correlations are **historical sentinel values, not valid estimates of zero association for a constant map**. Treat these correlations as invalid/undefined with a flat-map reason in corrected analyses. Their coincidence with geometry flags is corroboration across pipeline summaries, not independent proof of a gradient cause. The nonflat Grad-CAM++ and LayerCAM maps alone also cannot identify that cause.

## 2. Preprocessing provenance is material

Historical XAI preprocessing in the audited `src/xai_engine.py:100–120` converts to RGB and calls PIL `resize(target_size)` without an explicit resampling option. For RGB this is PIL's bicubic path, followed by float32 division by 255. Geometry then reloads the already-resized saved RGB original. Official training/evaluation paths use Keras `load_img`/directory-loader nearest-neighbor defaults (see `run_iccad5_seed_replication.py:62–64`).

Every historical statistic above is therefore labeled **legacy bicubic provenance**, not a corrected nearest-neighbor result. Merely loading an old 224×224 bicubic `original.png` with a nearest policy cannot undo its earlier resizing. A corrected probe must use B-validated tensors generated from the full-size source images, not these saved originals. B must also validate prediction behavior and target-class selection before any corrected scientific interpretation.

The source identity check survives both resize policies for this duplicate pair, but **that does not establish that Grad-CAM flatness survives the policy change**. `FINAL_RESEARCH_REPORT.md:86–95` calling this inconsistency immaterial is not a conclusion justified by D's evidence. No such claim is made here.

## 3. Engine audit: which mechanisms remain distinguishable?

At `src/xai_engine.py:156–173`, the score is the reconstructed **pre-sigmoid logit**: NHS uses `+z_NHS`, HS uses `-z_NHS`. `get_pre_sigmoid_logit` (`29–35`) computes the final dense affine transformation from `dense_penultimate`. It does not differentiate the sigmoid probability. The target layer `conv_final_2` is linear, before BatchNorm and ELU (`src/model_xai.py:43–46`), so feature maps may contain either sign.

For one sample let `A[i,j,k]` be that feature tensor and `G[i,j,k] = dS/dA[i,j,k]`:

```
alpha[k] = mean_(i,j) G[i,j,k]
C[i,j] = sum_k alpha[k] * A[i,j,k]
R[i,j] = max(C[i,j], 0)
```

The engine takes this ReLU **before** checking the maximum. It normalizes by `max(R)+1e-10` when positive, bilinearly resizes, and divides the resized map by its positive maximum (`263–281`). The `max_v > 0` guard does not diagnose why a map is zero.

Distinct possibilities that a flat final map does **not** resolve:

1. **Zero or small target gradients.** Requires norm, range, variance, and nonzero fractions of G. A disconnected (`None`) gradient must be an error, not imputed zeros.
2. **Spatial cancellation inside GAP.** Positive and negative gradients can average away within each channel even when individual gradients are nonzero.
3. **Cancellation between weighted channels.** Nonzero GAP weights and nonzero features can combine to a small signed CAM.
4. **An entirely nonpositive signed CAM erased by ReLU.** Large nonzero gradients and weights can yield C<=0 everywhere. This is not equivalent to vanishing gradients or GAP cancellation.
5. **Spatially constant features or a constant positive CAM.** A nonzero constant map is degenerate under the range criterion too.
6. **Finite-precision/normalization effects or very sparse positive support.** Need pre- and post-processing measurements at native and resized resolution; low spread is not identical to the flat-map flag.

Some mechanisms may coexist. A scalar cancellation ratio is descriptive, not a causal attribution to an upstream network layer. The report does not select a mechanism for either flagged TN.

### Claims requiring correction in synthesis

`FINAL_ICCAD5_SYNTHESIS.md:35–36,144–171` describes a gradient-flow property and offers a prior saturated-gradient/GAP-cancellation mechanism from other checkpoint lineages. `FINAL_RESEARCH_REPORT.md:312–329` similarly moves from flatness to a gradient explanation. For this final checkpoint, the supportable wording is:

> Two historical TN sample IDs (one byte-distinct source image) have flat resized Grad-CAM maps according to the saved geometry flags and concordant diagnostics. Direct gradients, GAP weights, and signed pre-ReLU CAMs have not been measured here. Their mechanism is unresolved.

Sigmoid saturation is **not in the differentiated score path** of this engine; a high-magnitude logit alone does not imply its derivative is near zero. Internal gating/nonlinearities can affect gradients, but need direct evidence. Moreover, a nonzero LayerCAM from exactly zero gradients would contradict `src/xai_engine.py:250–257`: pixelwise ReLU-weighting cannot recover signal from an identically zero gradient tensor. Nonzero versus very sparse/small gradients must not be conflated.

The separate `dense_penultimate` zero-vector bottleneck finding from earlier checkpoint/seed studies is not established by these flat-map flags and should not be relabeled as recurrent on this checkpoint.

## 4. Prepared diagnostics and GPU-only probe

`scripts/improvements/degeneracy_forensics.py` provides:

- `array_stats`: float64 CPU summaries with validity/reason, L2 norm, mean, population variance, min/max/range, exact nonzero counts and zero/positive/negative fractions. Empty/nonfinite arrays return null statistics, not invented zero values.
- `cam_statistics`: matching finite HWC features and gradients, global statistics, per-channel spatial feature/gradient variance, each signed GAP weight, mean absolute gradients, pre-/post-ReLU CAM statistics, and nonexclusive algebraic diagnostics.
- `historical`: CPU-only source audit, with direct gradient status explicitly invalid/unmeasured.
- `probe`: opt-in GPU-only final-checkpoint probe; no training, no CPU fallback, no overwrite of an existing run directory. It saves full tensors as NPZ and per-sample JSON, plus completion metadata only after a successful run. Partial runs lack the completion `run.json` and must not be reported complete.

Cancellation quantities are explicitly defined:

- Per-channel spatial retention: `abs(mean G_k) / mean(abs(G_k))`.
- Aggregate GAP spatial retention: `sum_k abs(alpha_k) / sum_k mean(abs(G_k))`.
- Channel-combination retention: `sum_ij abs(C_ij) / sum_ijk abs(alpha_k A_ijk)`.

Zero retention with a nonzero denominator indicates exact cancellation in the specified reduction; one indicates no sign cancellation there. Zero denominators produce `value: null, valid: false, reason: zero_denominator`. No universal near-zero cutoff or mechanism classification is imposed. `relu_erases_negative_cam` is true only when the signed CAM has a negative value and no positive values; an all-zero signed CAM is not called negative-CAM erasure.

CPU reconstruction is explicitly float64. The GPU path separately records TensorFlow-native weights, pre-/post-ReLU statistics, normalized native/resized maps, and range flags; it must not silently substitute the float64 reconstruction for engine arithmetic. Target direction is explicit, and fresh p(NHS), signed score and raw logit are recorded.

### Deferred invocation and B manifest contract

**Do not run this until the lead approves after B validation.** This command is preparation, not an executed experiment:

```
bash scripts/tf_gpu.sh -B scripts/improvements/degeneracy_forensics.py probe \
  --manifest <B-validated-manifest.json> --run-name corrected_nearest_01 \
  --authorization LEAD_APPROVED
```

Required manifest fields:

- `b_prediction_validation_passed: true`
- `preprocessing: "official_nearest_rgb_float32_0_1"`
- `checkpoint_sha256`: the exact immutable hash above
- `validation_report`, `validation_report_sha256`: real B prediction-validation evidence, reviewed by lead
- `samples`: unique safe sample IDs; each has `sample_id`, `tensor_path`, `tensor_sha256`, `target_class` (`HS` or `NHS` explicitly). Tensors must be float32 finite NHWC `(1,224,224,3)` in `[0,1]`. Paths are repository-relative or absolute.

At minimum include both flagged IDs and NHSCAD51003 as the historical clean TN control; ideally include all 11 historical IDs with B-validated current classifications. Record source image paths/hashes and B's corrected prediction metadata in each sample record as additional provenance. The literal authorization flag is an operator acknowledgment, not a security credential; manifest assertions do not replace scientific review. No placeholder manifest claims B has passed.

The probe verifies GPU presence before loading a model, sets strict GPU placement with soft placement disabled, verifies feature and gradient devices, and hashes the checkpoint before/after execution. It does not call the shared legacy preprocessor: it consumes B's approved tensors to avoid silently reintroducing bicubic input. Outputs are confined to `results/xai/final_iccad5/degeneracy_forensics/<run-name>/`. A corrected nearest probe cannot by itself explain the historical bicubic mechanism; an explicitly authorized matched legacy-input probe would be a separate experiment, not supported by this corrected-only manifest policy.

## 5. Validation and remaining work

Executed CPU-only tests:

```
.venv/bin/python -B -W error::ResourceWarning \
  results/xai/final_iccad5/degeneracy_forensics/test_degeneracy_forensics.py
```

**5 tests passed**, covering finite/invalid statistics; synthetic spatial GAP cancellation versus negative-CAM ReLU erasure; channel cancellation, zero gradients, constant positive CAM, invalid shapes/nonfinite arrays; historical 132/6 counts and IDs; rejection before TensorFlow import without lead authorization; and rejection of missing B validation/wrong resize policy. The tests assert TensorFlow was not imported. CPU artifact CLI and help also executed successfully. Synthetic test tensors are unit-test fixtures only, not checkpoint observations.

**Unverified:** successful manifest/tensor handoff from B, live TensorFlow/GPU execution, operation-level GPU compatibility, native-engine numerical parity under corrected inputs, and every requested direct-gradient statistic on actual checkpoint samples. The prepared GPU branch is not advertised as GPU-tested. A runtime incompatibility must fail closed rather than prompt a CPU workaround. No final-checkpoint gradient values or mechanism conclusions have been fabricated to fill the blocked step.

Next step is a lead-serialized GPU probe after B validation, followed by per-sample inspection of signed gradients, GAP weights, pre-ReLU CAM and processing stages, with duplicate-aware interpretation and explicit historical-versus-corrected provenance. Until then the final scientific result is **map degeneracy confirmed historically; mechanism unresolved**.
