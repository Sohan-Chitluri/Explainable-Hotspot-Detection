#!/usr/bin/env python3
"""
Full XAI Pipeline Execution: Grad-CAM, Grad-CAM++, LayerCAM, Multi-Layer Comparison,
and Quantitative Diagnostics for ICCAD-12 Benchmarks 1-5.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
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

from src.xai_diagnostics import compute_attribution_diagnostics
from src.xai_engine import XAIEngine


def parse_args():
    p = argparse.ArgumentParser(description="Run Full Multi-Method XAI Pipeline on ICCAD-12")
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "iccad-official",
        help="Root directory containing iccad1..iccad5",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models" / "xai",
        help="Directory containing trained XAI .keras models",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "xai",
        help="Output directory for XAI explanations and diagnostics",
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
        "--samples-per-case",
        type=int,
        default=3,
        help="Number of representative samples per category (TP, TN, FP, FN)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha blending factor",
    )
    p.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap for heatmaps",
    )
    return p.parse_args()


def clean_sample_id(filename: str) -> str:
    stem = Path(filename).name
    stem = re.sub(r"\.(png|jpg|jpeg)$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace(".png", "_").replace(".jpg", "_").replace(".", "_")
    return stem


def collect_test_predictions(
    model: tf.keras.Model,
    data_root: Path,
    benchmark: int,
    batch_size: int = 128,
) -> pd.DataFrame:
    bench_test_dir = data_root / f"iccad{benchmark}" / "test"
    if not bench_test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {bench_test_dir}")

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    val_gen = image_gen.flow_from_directory(
        directory=str(bench_test_dir),
        batch_size=batch_size,
        shuffle=False,
        target_size=(224, 224),
        class_mode="binary",
    )

    preds = model.predict(val_gen, verbose=1).ravel()
    p_nhs = preds
    p_hs = 1.0 - p_nhs

    records = []
    for rel_path, actual_idx, p_n, p_h in zip(val_gen.filenames, val_gen.classes, p_nhs, p_hs):
        full_path = bench_test_dir / rel_path
        fname = Path(rel_path).name
        act_label = "HS" if actual_idx == 0 else "NHS"
        pred_idx = 1 if p_n >= 0.5 else 0
        pred_label = "NHS" if pred_idx == 1 else "HS"
        correct = bool(actual_idx == pred_idx)
        conf = float(p_h if pred_label == "HS" else p_n)

        if actual_idx == 0 and pred_idx == 0:
            case = "TP"
        elif actual_idx == 1 and pred_idx == 1:
            case = "TN"
        elif actual_idx == 1 and pred_idx == 0:
            case = "FP"
        elif actual_idx == 0 and pred_idx == 1:
            case = "FN"
        else:
            case = "UNKNOWN"

        records.append({
            "benchmark": f"iccad{benchmark}",
            "file_path": str(full_path),
            "filename": fname,
            "sample_id": clean_sample_id(fname),
            "actual_label": act_label,
            "actual_idx": int(actual_idx),
            "predicted_label": pred_label,
            "predicted_idx": int(pred_idx),
            "p_hs": float(p_h),
            "p_nhs": float(p_n),
            "confidence": conf,
            "is_correct": correct,
            "case_type": case,
        })

    return pd.DataFrame(records)


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


def save_multi_method_comparison_panel(
    raw_rgb: np.ndarray,
    heatmaps: Dict[str, np.ndarray],
    overlays: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
):
    """Save 7-panel figure: Original | Grad-CAM Heatmap & Overlay | Grad-CAM++ Heatmap & Overlay | LayerCAM Heatmap & Overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 7, figsize=(22, 3.4), dpi=150)

    axes[0].imshow(raw_rgb)
    axes[0].set_title("Original Layout", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    methods = [("Grad-CAM", "gradcam"), ("Grad-CAM++", "gradcam_plus"), ("LayerCAM", "layercam")]
    for idx, (m_label, m_key) in enumerate(methods):
        h_ax = axes[1 + idx * 2]
        o_ax = axes[2 + idx * 2]

        h_ax.imshow(heatmaps[m_key], cmap="jet", vmin=0.0, vmax=1.0)
        h_ax.set_title(f"{m_label} Heatmap", fontsize=10, fontweight="bold")
        h_ax.axis("off")

        o_ax.imshow(overlays[m_key])
        o_ax.set_title(f"{m_label} Overlay", fontsize=10, fontweight="bold")
        o_ax.axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_layer_comparison_panel(
    raw_rgb: np.ndarray,
    layer_heatmaps: Dict[str, np.ndarray],
    layer_overlays: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
):
    """Save multi-layer comparison panel: Original | Early Layer | Mid Layer | Final Layer."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), dpi=150)

    # Row 0: Heatmaps
    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title("Original Layout", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    layer_names = ["Early (conv_early_2: 112x112)", "Intermediate (conv_mid_2: 56x56)", "Final (conv_final_2: 28x28)"]
    layer_keys = ["early", "mid", "final"]

    for i, (lname, lkey) in enumerate(zip(layer_names, layer_keys)):
        axes[0, i + 1].imshow(layer_heatmaps[lkey], cmap="jet", vmin=0.0, vmax=1.0)
        axes[0, i + 1].set_title(f"Heatmap: {lname}", fontsize=10, fontweight="bold")
        axes[0, i + 1].axis("off")

        axes[1, i + 1].imshow(layer_overlays[lkey])
        axes[1, i + 1].set_title(f"Overlay: {lname}", fontsize=10, fontweight="bold")
        axes[1, i + 1].axis("off")

    axes[1, 0].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_benchmark(
    benchmark: int,
    data_root: Path,
    models_dir: Path,
    results_dir: Path,
    samples_per_case: int = 3,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> List[Dict]:
    print("\n" + "=" * 76)
    print(f"Executing XAI Pipeline on Benchmark iccad{benchmark}")
    print("=" * 76)

    model_path = models_dir / f"xai_cnn_iccad{benchmark}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"XAI model checkpoint not found: {model_path}")

    bench_out_dir = results_dir / f"iccad{benchmark}"
    bench_out_dir.mkdir(parents=True, exist_ok=True)

    engine = XAIEngine.from_checkpoint(model_path, target_layer_name="conv_final_2")

    # Evaluate test split
    df_all = collect_test_predictions(engine.model, data_root, benchmark)
    df_rep = select_representative_samples(df_all, samples_per_case=samples_per_case)
    print(f"Selected {len(df_rep)} representative samples for full explanation.")

    records = []
    cmap = matplotlib.colormaps[colormap]

    for _, row in df_rep.iterrows():
        case = row["case_type"]
        act = row["actual_label"]
        pred = row["predicted_label"]
        sid = row["sample_id"]
        fpath = row["file_path"]

        sample_prefix = f"iccad{benchmark}_{case}_{act}_pred{pred}_{sid}"
        sample_dir = bench_out_dir / sample_prefix
        sample_dir.mkdir(parents=True, exist_ok=True)

        img_tensor, raw_rgb = engine.preprocess_image(fpath)
        p_hs, p_nhs, pred_label, pred_idx = engine.predict(img_tensor)

        # 1. Save Original Image
        orig_img_path = sample_dir / "original.png"
        Image.fromarray(raw_rgb).save(orig_img_path)

        method_heatmaps_pred = {}
        method_overlays_pred = {}
        sample_diag_records = {}

        # 2. Extract explanations for all 3 methods (both True and Predicted classes)
        for target_mode, target_name in [("pred", pred), ("true", act)]:
            # Final Layer: Grad-CAM, Grad-CAM++, LayerCAM
            engine.set_target_layer("conv_final_2")

            # Vanilla Grad-CAM
            _, gcam_h, gcam_info = engine.explain_gradcam(img_tensor, target_class=target_name)
            gcam_overlay = engine.overlay_heatmap(raw_rgb, gcam_h, alpha=alpha, colormap=colormap)
            Image.fromarray(np.uint8(cmap(gcam_h)[:, :, :3] * 255)).save(sample_dir / f"{target_mode}_gradcam.png")
            Image.fromarray(gcam_overlay).save(sample_dir / f"{target_mode}_gradcam_overlay.png")

            # Grad-CAM++
            _, gcam_pp_h, _ = engine.explain_gradcam_plus_plus(img_tensor, target_class=target_name)
            gcam_pp_overlay = engine.overlay_heatmap(raw_rgb, gcam_pp_h, alpha=alpha, colormap=colormap)
            Image.fromarray(np.uint8(cmap(gcam_pp_h)[:, :, :3] * 255)).save(sample_dir / f"{target_mode}_gradcam_plus.png")
            Image.fromarray(gcam_pp_overlay).save(sample_dir / f"{target_mode}_gradcam_plus_overlay.png")

            # LayerCAM
            _, lcam_h, _ = engine.explain_layercam(img_tensor, target_class=target_name)
            lcam_overlay = engine.overlay_heatmap(raw_rgb, lcam_h, alpha=alpha, colormap=colormap)
            Image.fromarray(np.uint8(cmap(lcam_h)[:, :, :3] * 255)).save(sample_dir / f"{target_mode}_layercam.png")
            Image.fromarray(lcam_overlay).save(sample_dir / f"{target_mode}_layercam_overlay.png")

            # Compute Diagnostics
            diag_gcam = compute_attribution_diagnostics(gcam_h)
            diag_gcam_pp = compute_attribution_diagnostics(gcam_pp_h)
            diag_lcam = compute_attribution_diagnostics(lcam_h)

            if target_mode == "pred":
                method_heatmaps_pred["gradcam"] = gcam_h
                method_heatmaps_pred["gradcam_plus"] = gcam_pp_h
                method_heatmaps_pred["layercam"] = lcam_h
                method_overlays_pred["gradcam"] = gcam_overlay
                method_overlays_pred["gradcam_plus"] = gcam_pp_overlay
                method_overlays_pred["layercam"] = lcam_overlay
                sample_diag_records = {
                    "gradcam": diag_gcam,
                    "gradcam_pp": diag_gcam_pp,
                    "layercam": diag_lcam,
                }

        # 3. Multi-Layer Comparison (LayerCAM on early, mid, final layers for predicted class)
        layer_heatmaps = {}
        layer_overlays = {}
        layer_map = {"early": "conv_early_2", "mid": "conv_mid_2", "final": "conv_final_2"}
        for lkey, lname in layer_map.items():
            engine.set_target_layer(lname)
            _, lh, _ = engine.explain_layercam(img_tensor, target_class=pred)
            layer_heatmaps[lkey] = lh
            layer_overlays[lkey] = engine.overlay_heatmap(raw_rgb, lh, alpha=alpha, colormap=colormap)

        # 4. Save Multi-Method Comparison Panel
        title_panel = (
            f"iccad{benchmark} | Case: {case} (Actual: {act} -> Pred: {pred}) | "
            f"P(HS)={p_hs:.3f}, P(NHS)={p_nhs:.3f}"
        )
        save_multi_method_comparison_panel(
            raw_rgb,
            method_heatmaps_pred,
            method_overlays_pred,
            sample_dir / "methods_comparison.png",
            title=title_panel,
        )

        # 5. Save Multi-Layer Comparison Panel
        save_layer_comparison_panel(
            raw_rgb,
            layer_heatmaps,
            layer_overlays,
            sample_dir / "layer_comparison.png",
            title=f"iccad{benchmark} [{case} {sid}] Multi-Layer Resolution Hierarchy",
        )

        # Build Record
        rec = {
            "benchmark": f"iccad{benchmark}",
            "sample_id": sid,
            "case_type": case,
            "actual_label": act,
            "predicted_label": pred,
            "is_correct": bool(act == pred),
            "p_hs": round(p_hs, 4),
            "p_nhs": round(p_nhs, 4),
            "sample_dir": str(sample_dir.relative_to(ROOT)),
            # Grad-CAM diagnostics
            "gcam_peak_x": sample_diag_records["gradcam"]["peak_attribution_x"],
            "gcam_peak_y": sample_diag_records["gradcam"]["peak_attribution_y"],
            "gcam_centroid_x": sample_diag_records["gradcam"]["centroid_x"],
            "gcam_centroid_y": sample_diag_records["gradcam"]["centroid_y"],
            "gcam_spread": sample_diag_records["gradcam"]["spatial_spread_radius"],
            "gcam_area_gt50": sample_diag_records["gradcam"]["fraction_above_050"],
            "gcam_connected_count": sample_diag_records["gradcam"]["connected_regions_count"],
            "gcam_edge_conc": sample_diag_records["gradcam"]["edge_border_concentration"],
            # Grad-CAM++ diagnostics
            "gcam_pp_peak_x": sample_diag_records["gradcam_pp"]["peak_attribution_x"],
            "gcam_pp_peak_y": sample_diag_records["gradcam_pp"]["peak_attribution_y"],
            "gcam_pp_centroid_x": sample_diag_records["gradcam_pp"]["centroid_x"],
            "gcam_pp_centroid_y": sample_diag_records["gradcam_pp"]["centroid_y"],
            "gcam_pp_spread": sample_diag_records["gradcam_pp"]["spatial_spread_radius"],
            "gcam_pp_area_gt50": sample_diag_records["gradcam_pp"]["fraction_above_050"],
            "gcam_pp_connected_count": sample_diag_records["gradcam_pp"]["connected_regions_count"],
            "gcam_pp_edge_conc": sample_diag_records["gradcam_pp"]["edge_border_concentration"],
            # LayerCAM diagnostics
            "layercam_peak_x": sample_diag_records["layercam"]["peak_attribution_x"],
            "layercam_peak_y": sample_diag_records["layercam"]["peak_attribution_y"],
            "layercam_centroid_x": sample_diag_records["layercam"]["centroid_x"],
            "layercam_centroid_y": sample_diag_records["layercam"]["centroid_y"],
            "layercam_spread": sample_diag_records["layercam"]["spatial_spread_radius"],
            "layercam_area_gt50": sample_diag_records["layercam"]["fraction_above_050"],
            "layercam_connected_count": sample_diag_records["layercam"]["connected_regions_count"],
            "layercam_edge_conc": sample_diag_records["layercam"]["edge_border_concentration"],
        }
        records.append(rec)

    return records


def create_summary_category_grids(
    df_samples: pd.DataFrame,
    results_dir: Path,
):
    """Generate multi-method composite comparison grids for TP, TN, FP, FN."""
    for case in ["TP", "TN", "FP", "FN"]:
        sub = df_samples[df_samples["case_type"] == case]
        if sub.empty:
            continue

        n_samples = min(len(sub), 5)
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 3.4 * n_samples), dpi=150)
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n_samples):
            row = sub.iloc[i]
            sdir = ROOT / row["sample_dir"]

            orig_img = Image.open(sdir / "original.png")
            gcam_ov = Image.open(sdir / "pred_gradcam_overlay.png")
            gcam_pp_ov = Image.open(sdir / "pred_gradcam_plus_overlay.png")
            lcam_ov = Image.open(sdir / "pred_layercam_overlay.png")

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f"[{row['benchmark']}] {row['sample_id']} (Orig)", fontsize=9, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gcam_ov)
            axes[i, 1].set_title(f"Grad-CAM (Spread={row['gcam_spread']:.1f}px)", fontsize=9)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(gcam_pp_ov)
            axes[i, 2].set_title(f"Grad-CAM++ (Spread={row['gcam_pp_spread']:.1f}px)", fontsize=9)
            axes[i, 2].axis("off")

            axes[i, 3].imshow(lcam_ov)
            axes[i, 3].set_title(f"LayerCAM (Area>0.5={row['layercam_area_gt50']*100:.1f}%)", fontsize=9, fontweight="bold")
            axes[i, 3].axis("off")

        case_titles = {
            "TP": "True Positives (Actual HS -> Predicted HS)",
            "TN": "True Negatives (Actual NHS -> Predicted NHS)",
            "FP": "False Positives (Actual NHS -> Predicted HS)",
            "FN": "False Negatives (Actual HS -> Predicted NHS)",
        }
        fig.suptitle(f"XAI Multi-Method Comparison: {case_titles.get(case, case)}", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        out_f = results_dir / f"summary_{case}.png"
        plt.savefig(out_f, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved category summary sheet -> {out_f}")


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for b in args.benchmarks:
        b_records = process_benchmark(
            benchmark=b,
            data_root=args.data_root,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            samples_per_case=args.samples_per_case,
            alpha=args.alpha,
            colormap=args.colormap,
        )
        all_records.extend(b_records)

    df_samples = pd.DataFrame(all_records)
    csv_path = args.results_dir / "xai_diagnostics.csv"
    df_samples.to_csv(csv_path, index=False)
    print(f"\nSaved XAI diagnostics CSV -> {csv_path} ({len(df_samples)} records)")

    create_summary_category_grids(df_samples, args.results_dir)
    print("\n" + "=" * 76)
    print("XAI EXPLANATION PIPELINE COMPLETE")
    print(f"Results directory: {args.results_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
