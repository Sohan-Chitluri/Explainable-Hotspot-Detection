#!/usr/bin/env python
"""Final-checkpoint forensics. CPU statistics do not import TensorFlow.

Historical artifacts cannot supply gradient evidence. The GPU probe is opt-in,
requires lead authorization plus a B-validated tensor manifest, and never trains.
All generated artifacts are confined to final_iccad5/degeneracy_forensics/.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/xai/final_iccad5"
OUT = BASE / "degeneracy_forensics"
CHECKPOINT = ROOT / "models/xai/experimental/xai_cnn_iccad5_balanced_sampling_seed1.keras"
EXPECTED_SHA = "9479f07781b0128e19d891b295c99b544f9e6fd2b9a4628533138a47c8085b70"


def array_stats(array):
    """Finite-array population statistics in float64, with explicit invalidity."""
    a = np.asarray(array, dtype=np.float64)
    keys = ("l2_norm", "mean", "variance", "min", "max", "ptp", "nonzero_count", "zero_fraction", "positive_fraction", "negative_fraction")
    reason = "empty_array" if not a.size else ("nonfinite_array" if not np.isfinite(a).all() else None)
    if reason:
        return dict(valid=False, reason=reason, shape=list(a.shape), **dict.fromkeys(keys))
    return dict(valid=True, reason=None, shape=list(a.shape),
                l2_norm=float(np.linalg.norm(a.ravel())), mean=float(a.mean()),
                variance=float(a.var()), min=float(a.min()), max=float(a.max()),
                ptp=float(np.ptp(a)), nonzero_count=int(np.count_nonzero(a)),
                zero_fraction=float(np.mean(a == 0)), positive_fraction=float(np.mean(a > 0)),
                negative_fraction=float(np.mean(a < 0)))


def cam_statistics(features, gradients):
    """Analyze HWC arrays before clipping; no threshold-based mechanism verdict.

    Retention 0 means exact cancellation, 1 means no sign cancellation in the
    indicated reduction. Undefined zero-denominator ratios remain null.
    Float64 reconstruction is diagnostic, not an exact TF float32 reproduction.
    """
    a, g = np.asarray(features, dtype=np.float64), np.asarray(gradients, dtype=np.float64)
    if a.shape != g.shape or a.ndim != 3 or not a.size:
        return {"valid": False, "reason": "expected_matching_nonempty_HWC"}
    if not np.isfinite(a).all() or not np.isfinite(g).all():
        return {"valid": False, "reason": "nonfinite_array"}
    weights = g.mean(axis=(0, 1))
    mean_abs_g = np.abs(g).mean(axis=(0, 1))
    terms = a * weights
    pre = terms.sum(axis=-1)
    post = np.maximum(pre, 0.)

    def ratio(num, den):
        return {"value": float(num / den) if den > 0 else None,
                "valid": bool(den > 0), "reason": None if den > 0 else "zero_denominator"}

    return {"valid": True, "reason": None, "arithmetic": "float64_cpu_reconstruction",
            "gradients": array_stats(g), "features": array_stats(a),
            "gradient_spatial_variance_per_channel": g.var(axis=(0, 1)).tolist(),
            "feature_spatial_variance_per_channel": a.var(axis=(0, 1)).tolist(),
            "gap_weights": weights.tolist(), "gap_weight_stats": array_stats(weights),
            "mean_absolute_gradient_per_channel": mean_abs_g.tolist(),
            "gap_retention_per_channel": [ratio(abs(w), d) for w, d in zip(weights, mean_abs_g)],
            "cam_pre_relu": array_stats(pre), "cam_post_relu": array_stats(post),
            "diagnostic": {
                "gradient_exactly_zero": bool(np.all(g == 0)),
                "gap_weights_exactly_zero": bool(np.all(weights == 0)),
                "pre_cam_exactly_zero": bool(np.all(pre == 0)),
                "relu_erases_negative_cam": bool(np.max(pre) <= 0 and np.min(pre) < 0),
                "gap_spatial_retention": ratio(np.abs(weights).sum(), mean_abs_g.sum()),
                "channel_cam_retention": ratio(np.abs(pre).sum(), np.abs(terms).sum()),
                "positive_cam_mass": float(post.sum()),
                "negative_cam_mass": float(np.maximum(-pre, 0.).sum()),
                "interpretation": "Algebraic diagnostics can coexist; no root-cause conclusion from flatness."}}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def historical_audit():
    """Recompute historical counts from CSVs; gradient evidence is absent."""
    with open(BASE / "xai_diagnostics.csv") as fh:
        diag = list(csv.DictReader(fh))
    with open(BASE / "geometry/geometry_diagnostics.csv") as fh:
        geom = list(csv.DictReader(fh))
    with open(BASE / "occlusion/occlusion_diagnostics.csv") as fh:
        occ = list(csv.DictReader(fh))
    flagged = [r for r in geom if r["is_degenerate_map"] == "True"]
    flagged_ids = {r["sample_id"] for r in flagged}
    samples = []
    for r in diag:
        oc = next(o for o in occ if o["sample_id"] == r["sample_id"])
        samples.append({
            "sample_id": r["sample_id"], "case_type": r["case_type"],
            "p_hs": float(r["p_hs"]), "p_nhs": float(r["p_nhs"]),
            "gcam_spread_px": float(r["gcam_spread"]),
            "gcam_connected_regions": int(r["gcam_connected_count"]),
            "gcam_pearson_vs_occlusion": float(oc["corr_gcam_pearson"]),
            "gcam_pp_pearson_vs_occlusion": float(oc["corr_gcam_pp_pearson"]),
            "layercam_pearson_vs_occlusion": float(oc["corr_layercam_pearson"]),
            "gradcam_flat_observed": r["sample_id"] in flagged_ids,
            "is_degenerate_map_flagged": r["sample_id"] in flagged_ids,
            "raw_gcam_grid_saved": False, "gradients_saved": False,
            "reason_flatness_only": "np.ptp<1e-6 flag; flatness alone cannot identify mechanism"})
    return {
        "scope": "results/xai/final_iccad5 (historical, pipeline bicubic preprocessing)",
        "checkpoint_sha256_verified": CHECKPOINT.is_file() and sha256_of(CHECKPOINT) == EXPECTED_SHA,
        "samples_total": len(samples), "samples": samples,
        "geometry_rows": len(geom), "flagged_rows": len(flagged),
        "flagged_sample_ids": sorted({r["sample_id"] for r in flagged}),
        "flagged_methods": sorted({r["method"] for r in flagged}),
        "occlusion_pearson_gradcam_for_flagged": [s["gcam_pearson_vs_occlusion"] for s in samples if s["sample_id"] in flagged_ids],
        "xai_diagnostics_gradcam_spread_px_for_flagged": [s["gcam_spread_px"] for s in samples if s["sample_id"] in flagged_ids],
        "source_sha256": {str(p.relative_to(ROOT)): sha256_of(p) for p in [BASE / "xai_diagnostics.csv", BASE / "geometry/geometry_diagnostics.csv", BASE / "occlusion/occlusion_diagnostics.csv"]},
        "flagged_levels": sorted({int(r["level_pct"]) for r in flagged}),
        "unique_sample_method_pairs": len({(r["sample_id"], r["method"]) for r in geom}),
        "flagged_sample_method_pairs": len({(r["sample_id"], r["method"]) for r in flagged}),
        "direct_gradient_probe": {
            "valid": False, "reason": "no_inference_authorized_this_host",
            "statistics": None,
            "gradient_norm_mean_variance_min_max": {"valid": False,
                "reason": "requires opt-in gpu_probe under lead authorization"}}}


def validate_manifest(manifest):
    """Validate B's explicit approval and hashed prepared inputs, without inference.

    File paths are repository-relative or absolute. This validates integrity,
    not the scientific truth of B's assertion: lead must review the referenced
    validation report before permitting the command.
    """
    if manifest.get("b_prediction_validation_passed") is not True:
        raise ValueError("B prediction validation must pass first")
    if manifest.get("preprocessing") != "official_nearest_rgb_float32_0_1":
        raise ValueError("requires official nearest preprocessing")
    if manifest.get("checkpoint_sha256") != EXPECTED_SHA:
        raise ValueError("manifest checkpoint mismatch")
    report = ROOT / manifest["validation_report"]
    if sha256_of(report) != manifest["validation_report_sha256"]:
        raise ValueError("B validation report hash mismatch")
    records = manifest.get("samples", [])
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("sample IDs must be nonempty and unique")
    if not {"NNHSCAD5999_6", "NNHSCAD5999_9", "NHSCAD51003"}.issubset(ids):
        raise ValueError("include both flagged TNs and the historical clean TN control")
    prepared = []
    for row in records:
        sid = row["sample_id"]
        if not sid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sid):
            raise ValueError("unsafe sample ID")
        path = ROOT / row["tensor_path"]
        if sha256_of(path) != row["tensor_sha256"]:
            raise ValueError("tensor hash mismatch")
        x = np.load(path, allow_pickle=False)
        if x.shape != (1, 224, 224, 3) or x.dtype != np.float32 or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("expected finite float32 NHWC tensor in [0,1]")
        if row["target_class"] not in ("HS", "NHS"):
            raise ValueError("target class must be explicit, not inferred from historical case label")
        prepared.append((row, x))
    return prepared


def gpu_probe(manifest_path, run_name, authorization=None):
    """Prepared GPU-only probe. NOT executed/validated on GPU in this workstream."""
    if authorization != "LEAD_APPROVED":
        raise RuntimeError("lead authorization required before any neural inference")
    manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text())
    prepared = validate_manifest(manifest)
    if sha256_of(CHECKPOINT) != EXPECTED_SHA:
        raise ValueError("immutable checkpoint hash mismatch")
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("run name must be a safe single directory name")
    destination = OUT / run_name
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing probe run")
    import tensorflow as tf  # deferred: CPU audit and tests never import TF
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU unavailable: no model loading or CPU inference permitted")
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_soft_device_placement(False)
    with tf.device("/GPU:0"):
        model = tf.keras.models.load_model(str(CHECKPOINT), compile=False)
        sub = tf.keras.Model(model.inputs, [model.get_layer("conv_final_2").output,
                                          model.get_layer("dense_penultimate").output, model.output])
        dense = model.get_layer("dense_output")
        destination.mkdir(parents=True, exist_ok=False)
        for row, x in prepared:
            tensor = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(tensor)
                a, penult, prediction = sub(tensor, training=False)
                logit = (tf.matmul(penult, dense.kernel) + dense.bias)[:, 0]
                score = logit if row["target_class"] == "NHS" else -logit
            g = tape.gradient(score, a)
            if g is None:
                raise RuntimeError("disconnected gradient; not a measured zero gradient")
            if "GPU:" not in a.device or "GPU:" not in g.device:
                raise RuntimeError("feature/gradient placement was not GPU")
            w = tf.reduce_mean(g, axis=(0, 1, 2))
            pre = tf.squeeze(a[0] @ w[..., tf.newaxis], axis=-1)
            post = tf.maximum(pre, 0.)
            vmax = tf.reduce_max(post)
            norm = post / (vmax + 1e-10) if float(vmax.numpy()) > 0 else post
            resized = tf.image.resize(norm[..., None], (224, 224), method="bilinear")[..., 0]
            rmax = tf.reduce_max(resized)
            resized = resized / rmax if float(rmax.numpy()) > 0 else resized
            aa, gg = a[0].numpy(), g[0].numpy()
            record = cam_statistics(aa, gg)
            record.update({"sample": row, "p_nhs": float(prediction.numpy()[0, 0]),
                           "logit_nhs": float(logit.numpy()[0]), "target_score": float(score.numpy()[0]),
                           "feature_device": a.device, "gradient_device": g.device,
                           "tf_gap_weights": w.numpy().tolist(), "tf_cam_pre_relu": array_stats(pre.numpy()),
                           "tf_cam_post_relu": array_stats(post.numpy()),
                           "tf_cam_normalized_native": array_stats(norm.numpy()),
                           "tf_cam_normalized_resized": array_stats(resized.numpy()),
                           "tf_resized_is_degenerate_map": bool(np.ptp(resized.numpy()) < 1e-6)})
            sid = row["sample_id"]
            np.savez_compressed(destination / f"{sid}.npz", features=aa, gradients=gg,
                                gap_weights=w.numpy(), pre_relu_cam=pre.numpy(), post_relu_cam=post.numpy(),
                                normalized_native_cam=norm.numpy(), normalized_resized_cam=resized.numpy())
            (destination / f"{sid}.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    if sha256_of(CHECKPOINT) != EXPECTED_SHA:
        raise RuntimeError("checkpoint integrity failure after probe")
    metadata = {"status": "completed", "samples": len(prepared), "tensorflow": tf.__version__,
                "checkpoint_sha256": EXPECTED_SHA, "manifest": manifest,
                "manifest_sha256": sha256_of(manifest_path), "script_sha256": sha256_of(__file__),
                "numpy": np.__version__, "gpu": str(gpus[0]),
                "scope": "corrected nearest; do not relabel historical bicubic findings"}
    (destination / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("historical", "probe"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--authorization", choices=("LEAD_APPROVED",))
    args = parser.parse_args()
    if args.mode == "historical":
        result = historical_audit()
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "historical_audit.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(path.relative_to(ROOT)), "rows": result["geometry_rows"],
                          "flagged_rows": result["flagged_rows"], "flagged_ids": result["flagged_sample_ids"],
                          "inference_performed": False}))
    else:
        result = gpu_probe(args.manifest, args.run_name, args.authorization)
        print(json.dumps({"status": result["status"], "samples": result["samples"]}))


if __name__ == "__main__":
    main()


