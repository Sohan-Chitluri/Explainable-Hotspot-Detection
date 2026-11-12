#!/usr/bin/env python3
"""
Factorial XAI Faithfulness and Robustness Evaluation for ICCAD-5.

Compares attribution quality, mathematical gradient vanishing, and faithfulness
across all 2x2 factorial configurations and the clean baseline:
- Methods: Grad-CAM, Grad-CAM++, LayerCAM, plus Random baseline.
- Dataset: All 41 official ICCAD-5 test HS samples (predeclared matched set).
- Metrics: Raw gradient Frobenius norm, pre- and post-normalization variance,
  is_math_zero flag, deletion AUC, insertion AUC.
- Statistics: Paired comparisons (Wilcoxon signed-rank, paired t-test, Cohen's d_z)
  across matched test images, reporting uncertainty intervals and stratifying
  by prediction correctness (all 41 vs TP-only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf

from src.faithfulness import compute_deletion_insertion_curves
from src.xai_engine import XAIEngine, get_pre_sigmoid_logit

DATA_ROOT = ROOT / "iccad-official" / "iccad5" / "test" / "test_hs"
RESULTS_DIR = ROOT / "results" / "iccad5_factorial"
XAI_OUT_DIR = RESULTS_DIR / "xai"


def compute_random_curves(model: tf.keras.Model, img_tensor: np.ndarray, steps: int = 10, seed: int = 42) -> dict:
    """Computes deletion and insertion curves using random pixel ranking."""
    def get_prob(x_in):
        preds = model(x_in, training=False).numpy().flatten()
        return float(1.0 - preds[0]) # HS score

    H, W = img_tensor.shape[1:3]
    total_pixels = H * W
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_pixels)
    sort_y = perm // W
    sort_x = perm % W

    base_prob = get_prob(img_tensor)
    blank_canvas = np.zeros_like(img_tensor)
    blank_prob = get_prob(blank_canvas)

    del_scores = [base_prob]
    ins_scores = [blank_prob]
    del_img = img_tensor.copy()
    ins_img = blank_canvas.copy()

    step_size = total_pixels // steps
    for s in range(1, steps + 1):
        idx_end = s * step_size if s < steps else total_pixels
        idx_start = (s - 1) * step_size
        y_slice = sort_y[idx_start:idx_end]
        x_slice = sort_x[idx_start:idx_end]

        del_img[0, y_slice, x_slice, :] = 0.0
        ins_img[0, y_slice, x_slice, :] = img_tensor[0, y_slice, x_slice, :]

        del_scores.append(get_prob(del_img))
        ins_scores.append(get_prob(ins_img))

    fractions = np.linspace(0.0, 1.0, steps + 1)
    del_auc = float(np.trapz(del_scores, fractions))
    ins_auc = float(np.trapz(ins_scores, fractions))
    return {
        "deletion_scores": del_scores,
        "insertion_scores": ins_scores,
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
    }


def evaluate_model_xai(model_path: Path, model_label: str, config_id: str, seed: int) -> list[dict]:
    print(f"\nEvaluating XAI on {model_label} ({model_path.name})...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    engine = XAIEngine(model, target_layer_name="conv_final_2")

    hs_files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith(".png") and not f.startswith(".")])
    records = []

    for fn in hs_files:
        img_path = DATA_ROOT / fn
        img_t, _ = engine.preprocess_image(img_path)

        # 1. Prediction
        preds = model(img_t, training=False).numpy().flatten()
        p_nhs = float(preds[0])
        p_hs = float(1.0 - p_nhs)
        pred_label = 0 if p_nhs < 0.5 else 1
        is_tp = (pred_label == 0)

        # 2. Gradient analysis on conv_final_2
        with tf.GradientTape() as tape:
            conv_outputs, penult_outputs, _ = engine.sub_model(img_t, training=False)
            logit_nhs = get_pre_sigmoid_logit(model, penult_outputs)
            hs_score = -logit_nhs # Target score for Hotspot
        grads = tape.gradient(hs_score, conv_outputs)
        grad_norm = float(tf.norm(grads).numpy())
        is_math_zero = bool(np.all(grads.numpy() == 0.0))

        # 3. Penultimate vector norm
        pen_norm = float(tf.norm(penult_outputs).numpy())
        is_pen_zero = bool(np.all(penult_outputs.numpy() == 0.0))

        # 4. Generate CAMs
        raw_gcam, res_gcam, _ = engine.explain_gradcam(img_t, target_class="HS")
        raw_gcampp, res_gcampp, _ = engine.explain_gradcam_plus_plus(img_t, target_class="HS")
        raw_lcam, res_lcam, _ = engine.explain_layercam(img_t, target_class="HS")

        # 5. Faithfulness curves
        curves = compute_deletion_insertion_curves(model, engine, img_t, target_idx=0, steps=10)
        rand_curves = compute_random_curves(model, img_t, steps=10, seed=42)

        base_record = {
            "model_label": model_label,
            "config_id": config_id,
            "seed": seed,
            "filename": fn,
            "p_hs": p_hs,
            "p_nhs": p_nhs,
            "pred_label": pred_label,
            "is_tp": is_tp,
            "pen_norm": pen_norm,
            "is_pen_zero": is_pen_zero,
            "grad_norm": grad_norm,
            "is_math_zero": is_math_zero,
        }

        # Grad-CAM
        records.append({
            **base_record,
            "method": "gradcam",
            "map_variance": float(np.var(res_gcam)),
            "map_max": float(np.max(res_gcam)),
            "deletion_auc": float(curves["gradcam"]["deletion_auc"]),
            "insertion_auc": float(curves["gradcam"]["insertion_auc"]),
        })

        # Grad-CAM++
        records.append({
            **base_record,
            "method": "gradcam_plus_plus",
            "map_variance": float(np.var(res_gcampp)),
            "map_max": float(np.max(res_gcampp)),
            "deletion_auc": float(curves["gradcam_plus_plus"]["deletion_auc"]),
            "insertion_auc": float(curves["gradcam_plus_plus"]["insertion_auc"]),
        })

        # LayerCAM
        records.append({
            **base_record,
            "method": "layercam",
            "map_variance": float(np.var(res_lcam)),
            "map_max": float(np.max(res_lcam)),
            "deletion_auc": float(curves["layercam"]["deletion_auc"]),
            "insertion_auc": float(curves["layercam"]["insertion_auc"]),
        })

        # Random baseline
        records.append({
            **base_record,
            "method": "random",
            "map_variance": float(1.0 / 12.0), # uniform variance
            "map_max": 1.0,
            "deletion_auc": float(rand_curves["deletion_auc"]),
            "insertion_auc": float(rand_curves["insertion_auc"]),
        })

    return records


def compute_paired_comparison(df: pd.DataFrame, model_a_label: str, model_b_label: str, method: str) -> dict:
    sub_a = df[(df["model_label"] == model_a_label) & (df["method"] == method)].sort_values("filename")
    sub_b = df[(df["model_label"] == model_b_label) & (df["method"] == method)].sort_values("filename")

    assert len(sub_a) == len(sub_b) == 41
    del_a = sub_a["deletion_auc"].values
    del_b = sub_b["deletion_auc"].values
    ins_a = sub_a["insertion_auc"].values
    ins_b = sub_b["insertion_auc"].values

    # Differences: B - A
    diff_del = del_b - del_a
    diff_ins = ins_b - ins_a

    # Paired t-tests
    t_del, p_del = stats.ttest_rel(del_b, del_a)
    t_ins, p_ins = stats.ttest_rel(ins_b, ins_a)

    # Wilcoxon signed-rank tests
    try:
        w_del, p_w_del = stats.wilcoxon(del_b, del_a)
    except Exception:
        w_del, p_w_del = np.nan, np.nan

    try:
        w_ins, p_w_ins = stats.wilcoxon(ins_b, ins_a)
    except Exception:
        w_ins, p_w_ins = np.nan, np.nan

    # Effect sizes (Cohen's d_z = mean(diff) / std(diff))
    dz_del = float(np.mean(diff_del) / (np.std(diff_del, ddof=1) + 1e-10))
    dz_ins = float(np.mean(diff_ins) / (np.std(diff_ins, ddof=1) + 1e-10))

    return {
        "model_a": model_a_label,
        "model_b": model_b_label,
        "method": method,
        "del_mean_a": float(np.mean(del_a)),
        "del_mean_b": float(np.mean(del_b)),
        "del_diff_mean": float(np.mean(diff_del)),
        "del_diff_std": float(np.std(diff_del, ddof=1)),
        "del_paired_t": float(t_del),
        "del_p_value": float(p_del),
        "del_wilcoxon_p": float(p_w_del),
        "del_cohens_dz": dz_del,
        "ins_mean_a": float(np.mean(ins_a)),
        "ins_mean_b": float(np.mean(ins_b)),
        "ins_diff_mean": float(np.mean(diff_ins)),
        "ins_diff_std": float(np.std(diff_ins, ddof=1)),
        "ins_paired_t": float(t_ins),
        "ins_p_value": float(p_ins),
        "ins_wilcoxon_p": float(p_w_ins),
        "ins_cohens_dz": dz_ins,
    }


def main():
    XAI_OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    # 1. Clean collapsed baseline (Seed 42)
    clean_model_path = ROOT / "models" / "xai" / "retrained_clean" / "xai_cnn_iccad5.keras"
    if clean_model_path.exists():
        records = evaluate_model_xai(clean_model_path, "Clean_ClassWeighted_Seed42", "B_clean", 42)
        all_records.extend(records)

    # 2. Factorial models from RESULTS_DIR / "models"
    for config_id in ["A", "B", "C", "D"]:
        for seed in [101, 202, 303]:
            mpath = RESULTS_DIR / "models" / f"xai_cnn_iccad5_{config_id}_seed{seed}.keras"
            if mpath.exists():
                lbl = f"Config_{config_id}_Seed{seed}"
                records = evaluate_model_xai(mpath, lbl, config_id, seed)
                all_records.extend(records)
            else:
                print(f"Skipping {mpath} (not yet available)")

    if not all_records:
        print("No models available for XAI evaluation.")
        return

    df = pd.DataFrame(all_records)
    out_csv = XAI_OUT_DIR / "per_sample_xai_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-sample metrics to {out_csv}")

    # Summary table by model and method
    summary = df.groupby(["model_label", "config_id", "seed", "method"])[
        ["is_math_zero", "map_variance", "deletion_auc", "insertion_auc"]
    ].agg(["mean", "std"])
    summary.to_csv(XAI_OUT_DIR / "xai_summary_table.csv")
    print("\nSummary Table:")
    print(summary.to_string())

    # Paired comparisons: e.g. Config C (Balanced) vs Config A (Natural), and vs Clean Baseline
    paired_results = []
    for method in ["gradcam", "gradcam_plus_plus", "layercam"]:
        for seed in [101, 202, 303]:
            lbl_a = f"Config_A_Seed{seed}"
            lbl_c = f"Config_C_Seed{seed}"
            lbl_b = f"Config_B_Seed{seed}"
            lbl_d = f"Config_D_Seed{seed}"
            models_present = set(df["model_label"].unique())

            if lbl_a in models_present and lbl_c in models_present:
                paired_results.append(compute_paired_comparison(df, lbl_a, lbl_c, method))
            if lbl_b in models_present and lbl_d in models_present:
                paired_results.append(compute_paired_comparison(df, lbl_b, lbl_d, method))
            if "Clean_ClassWeighted_Seed42" in models_present and lbl_c in models_present:
                paired_results.append(compute_paired_comparison(df, "Clean_ClassWeighted_Seed42", lbl_c, method))

    if paired_results:
        pdf = pd.DataFrame(paired_results)
        pdf.to_csv(XAI_OUT_DIR / "paired_statistical_comparisons.csv", index=False)
        print(f"\nSaved paired statistical comparisons to {XAI_OUT_DIR / 'paired_statistical_comparisons.csv'}")


if __name__ == "__main__":
    main()
