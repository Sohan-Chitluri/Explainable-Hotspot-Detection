#!/usr/bin/env python3
"""
Occlusion-Based Sensitivity Pipeline for Lithography Hotspot Detection.

Evaluates the 57 representative samples across ICCAD-12 benchmarks 1–5,
generates spatial occlusion importance maps, comparative 5-panel figures,
and exports occlusion_diagnostics.csv with correlation metrics against CAM methods.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from src.occlusion import OcclusionSensitivity, compute_map_comparison
from src.xai_engine import XAIEngine


def parse_args():
    p = argparse.ArgumentParser(description="Run Occlusion Sensitivity on ICCAD-12 Representative Samples")
    p.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models" / "xai",
        help="Directory containing trained XAI models",
    )
    p.add_argument(
        "--xai-results-dir",
        type=Path,
        default=ROOT / "results" / "xai",
        help="Directory containing existing XAI results and xai_diagnostics.csv",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "xai" / "occlusion",
        help="Output directory for occlusion explanations and diagnostics",
    )
    p.add_argument(
        "--benchmarks",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Benchmarks to analyze",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Occlusion window size in pixels (default: 32)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Sliding window stride in pixels (default: 16)",
    )
    p.add_argument(
        "--baseline-value",
        type=float,
        default=0.0,
        help="Occlusion patch baseline replacement value (0.0 = empty background)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Forward evaluation batch size",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay alpha blending factor",
    )
    p.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap name for heatmaps",
    )
    return p.parse_args()


def save_5panel_comparison(
    raw_rgb: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    overlays: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
):
    """
    Save 5-panel comparison figure:
    Original Layout | Grad-CAM Overlay | Grad-CAM++ Overlay | LayerCAM Overlay | Occlusion Overlay
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), dpi=150)

    # 1. Original
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Original Layout", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    methods = [
        ("Grad-CAM", "gradcam"),
        ("Grad-CAM++", "gradcam_plus"),
        ("LayerCAM", "layercam"),
        ("Occlusion (32x32)", "occlusion"),
    ]

    for idx, (m_name, m_key) in enumerate(methods):
        ax = axes[idx + 1]
        ax.imshow(overlays[m_key])
        ax.set_title(f"{m_name} Overlay", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_category_summary_grids(
    df_samples: pd.DataFrame,
    occlusion_dir: Path,
):
    """Generate 5-method comparison grids for representative TP, TN, FP, FN samples."""
    summaries_dir = occlusion_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    case_titles = {
        "TP": "True Positives (Actual HS -> Predicted HS)",
        "TN": "True Negatives (Actual NHS -> Predicted NHS)",
        "FP": "False Positives (Actual NHS -> Predicted HS)",
        "FN": "False Negatives (Actual HS -> Predicted NHS)",
    }

    for case in ["TP", "TN", "FP", "FN"]:
        sub = df_samples[df_samples["case_type"] == case]
        if sub.empty:
            continue

        n_samples = min(len(sub), 5)
        fig, axes = plt.subplots(n_samples, 5, figsize=(18, 3.4 * n_samples), dpi=150)
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n_samples):
            row = sub.iloc[i]
            sample_dir = ROOT / row["sample_dir"]
            occ_sample_dir = ROOT / row["occlusion_sample_dir"]

            orig_img = Image.open(sample_dir / "original.png")
            gcam_ov = Image.open(sample_dir / "pred_gradcam_overlay.png")
            gcam_pp_ov = Image.open(sample_dir / "pred_gradcam_plus_overlay.png")
            lcam_ov = Image.open(sample_dir / "pred_layercam_overlay.png")
            occ_ov = Image.open(occ_sample_dir / "occlusion_overlay.png")

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"[{row['benchmark']}] {row['sample_id']}\n(Original)", fontsize=9, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gcam_ov)
            axes[i, 1].set_title(f"Grad-CAM\n(Corr={row['corr_gcam_pearson']:.2f})", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(gcam_pp_ov)
            axes[i, 2].set_title(f"Grad-CAM++\n(Corr={row['corr_gcam_pp_pearson']:.2f})", fontsize=9)
            axes[i, 2].axis("off")

            axes[i, 3].imshow(lcam_ov)
            axes[i, 3].set_title(f"LayerCAM\n(Corr={row['corr_layercam_pearson']:.2f})", fontsize=9)
            axes[i, 3].axis("off")

            axes[i, 4].imshow(occ_ov)
            axes[i, 4].set_title(f"Occlusion (32x32)\n(Max ΔS={row['max_occlusion_importance']:.2f})", fontsize=9, fontweight="bold")
            axes[i, 4].axis("off")

        fig.suptitle(f"XAI Comparison (CAM vs Occlusion): {case_titles.get(case, case)}", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        out_f = summaries_dir / f"summary_{case}.png"
        plt.savefig(out_f, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved category summary sheet -> {out_f}")


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    diag_csv_path = args.xai_results_dir / "xai_diagnostics.csv"
    if not diag_csv_path.is_file():
        raise FileNotFoundError(f"Existing diagnostics CSV not found: {diag_csv_path}")

    df_xai = pd.read_csv(diag_csv_path)
    print(f"Loaded {len(df_xai)} representative samples from {diag_csv_path}")

    # Colormap
    cmap = matplotlib.colormaps[args.colormap]

    all_records = []
    total_start_time = time.perf_counter()

    for benchmark in args.benchmarks:
        bench_str = f"iccad{benchmark}"
        bench_samples = df_xai[df_xai["benchmark"] == bench_str]
        if bench_samples.empty:
            continue

        print("\n" + "=" * 76)
        print(f"Running Occlusion Sensitivity on {len(bench_samples)} samples for Benchmark {bench_str}")
        print("=" * 76)

        model_path = args.models_dir / f"xai_cnn_iccad{benchmark}.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        engine = XAIEngine.from_checkpoint(model_path, target_layer_name="conv_final_2")
        occluder = OcclusionSensitivity(
            engine=engine,
            window_size=(args.window_size, args.window_size),
            stride=(args.stride, args.stride),
            baseline_value=args.baseline_value,
            batch_size=args.batch_size,
        )

        bench_out_dir = args.results_dir / bench_str
        bench_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (_, row) in enumerate(bench_samples.iterrows(), 1):
            sid = row["sample_id"]
            case = row["case_type"]
            act = row["actual_label"]
            pred = row["predicted_label"]
            orig_sample_dir = ROOT / row["sample_dir"]

            sample_prefix = orig_sample_dir.name
            out_sample_dir = bench_out_dir / sample_prefix
            out_sample_dir.mkdir(parents=True, exist_ok=True)

            # Load original image
            orig_img_path = orig_sample_dir / "original.png"
            img_tensor, raw_rgb = engine.preprocess_image(orig_img_path)

            # Explain via Occlusion (predicted class)
            occ_res = occluder.explain(img_tensor, target_class=pred)
            norm_occ_map = occ_res["norm_map"]
            raw_occ_map = occ_res["raw_map"]

            # Save raw numpy map
            np.save(out_sample_dir / "raw_occlusion_map.npy", raw_occ_map)

            # Save normalized heatmap and overlay PNGs
            colored_occ = np.uint8(cmap(norm_occ_map)[:, :, :3] * 255)
            occ_overlay = occluder.overlay_heatmap(raw_rgb, norm_occ_map, alpha=args.alpha, colormap=args.colormap)

            Image.fromarray(raw_rgb).save(out_sample_dir / "original.png")
            Image.fromarray(colored_occ).save(out_sample_dir / "occlusion.png")
            Image.fromarray(occ_overlay).save(out_sample_dir / "occlusion_overlay.png")

            # Generate CAM heatmaps and overlays for direct comparison
            _, gcam_h, _ = engine.explain_gradcam(img_tensor, target_class=pred)
            _, gcam_pp_h, _ = engine.explain_gradcam_plus_plus(img_tensor, target_class=pred)
            _, lcam_h, _ = engine.explain_layercam(img_tensor, target_class=pred)

            gcam_ov = engine.overlay_heatmap(raw_rgb, gcam_h, alpha=args.alpha, colormap=args.colormap)
            gcam_pp_ov = engine.overlay_heatmap(raw_rgb, gcam_pp_h, alpha=args.alpha, colormap=args.colormap)
            lcam_ov = engine.overlay_heatmap(raw_rgb, lcam_h, alpha=args.alpha, colormap=args.colormap)

            # Comparative metrics against CAM methods
            comp_gcam = compute_map_comparison(norm_occ_map, gcam_h)
            comp_gcam_pp = compute_map_comparison(norm_occ_map, gcam_pp_h)
            comp_lcam = compute_map_comparison(norm_occ_map, lcam_h)

            # Save 5-Panel Comparison Figure
            heatmaps = {
                "gradcam": gcam_h,
                "gradcam_plus": gcam_pp_h,
                "layercam": lcam_h,
                "occlusion": norm_occ_map,
            }
            overlays = {
                "gradcam": gcam_ov,
                "gradcam_plus": gcam_pp_ov,
                "layercam": lcam_ov,
                "occlusion": occ_overlay,
            }
            panel_title = (
                f"{bench_str} [{case}] {sid} | Actual: {act} -> Pred: {pred} "
                f"(P(HS)={occ_res['p_hs']:.3f}, P(NHS)={occ_res['p_nhs']:.3f})"
            )
            save_5panel_comparison(
                raw_rgb,
                heatmaps,
                overlays,
                out_sample_dir / "methods_comparison_5panel.png",
                title=panel_title,
            )

            # Compile CSV record
            rec = {
                "benchmark": bench_str,
                "sample_id": sid,
                "case_type": case,
                "actual_label": act,
                "predicted_label": pred,
                "is_correct": bool(act == pred),
                "p_hs": round(occ_res["p_hs"], 4),
                "p_nhs": round(occ_res["p_nhs"], 4),
                "sample_dir": str(orig_sample_dir.relative_to(ROOT)),
                "occlusion_sample_dir": str(out_sample_dir.relative_to(ROOT)),
                # Occlusion Parameters
                "window_size": args.window_size,
                "stride": args.stride,
                "baseline_value": args.baseline_value,
                "num_windows": occ_res["num_windows"],
                "runtime_sec": occ_res["runtime_sec"],
                # Importance Scores
                "orig_target_score": occ_res["orig_target_score"],
                "max_occlusion_importance": occ_res["max_occlusion_importance"],
                "min_occlusion_importance": occ_res["min_occlusion_importance"],
                "mean_positive_importance": occ_res["mean_positive_importance"],
                # Spatial Diagnostics
                "peak_attribution_x": occ_res["peak_attribution_x"],
                "peak_attribution_y": occ_res["peak_attribution_y"],
                "centroid_x": occ_res["centroid_x"],
                "centroid_y": occ_res["centroid_y"],
                "spatial_spread_radius": occ_res["spatial_spread_radius"],
                "fraction_above_025": occ_res["fraction_above_025"],
                "fraction_above_050": occ_res["fraction_above_050"],
                "fraction_above_075": occ_res["fraction_above_075"],
                "connected_regions_count": occ_res["connected_regions_count"],
                "largest_region_area_px": occ_res["largest_region_area_px"],
                "edge_border_concentration": occ_res["edge_border_concentration"],
                # Comparison against Grad-CAM
                "corr_gcam_pearson": comp_gcam["pearson_corr"],
                "corr_gcam_spearman": comp_gcam["spearman_corr"],
                "corr_gcam_cosine": comp_gcam["cosine_sim"],
                "corr_gcam_iou_50": comp_gcam["iou_at_50"],
                # Comparison against Grad-CAM++
                "corr_gcam_pp_pearson": comp_gcam_pp["pearson_corr"],
                "corr_gcam_pp_spearman": comp_gcam_pp["spearman_corr"],
                "corr_gcam_pp_cosine": comp_gcam_pp["cosine_sim"],
                "corr_gcam_pp_iou_50": comp_gcam_pp["iou_at_50"],
                # Comparison against LayerCAM
                "corr_layercam_pearson": comp_lcam["pearson_corr"],
                "corr_layercam_spearman": comp_lcam["spearman_corr"],
                "corr_layercam_cosine": comp_lcam["cosine_sim"],
                "corr_layercam_iou_50": comp_lcam["iou_at_50"],
            }
            all_records.append(rec)
            print(f"[{idx}/{len(bench_samples)}] {bench_str} {case} {sid} -> Max ΔS: {rec['max_occlusion_importance']:+.3f} | LayerCAM Corr: {rec['corr_layercam_pearson']:+.3f} ({rec['runtime_sec']:.2f}s)")

    # Save CSV
    df_out = pd.DataFrame(all_records)
    out_csv = args.results_dir / "occlusion_diagnostics.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved Occlusion Diagnostics CSV -> {out_csv} ({len(df_out)} records)")

    # Save Category Summaries
    create_category_summary_grids(df_out, args.results_dir)

    total_time = time.perf_counter() - total_start_time
    print("\n" + "=" * 76)
    print(f"OCCLUSION PIPELINE COMPLETE across {len(df_out)} samples in {total_time:.1f}s")
    print(f"Results Directory: {args.results_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
