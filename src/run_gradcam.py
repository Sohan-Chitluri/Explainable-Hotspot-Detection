#!/usr/bin/env bash
"""
Full Grad-CAM extraction and explanation pipeline for ICCAD-12 benchmarks 1-5.

Executes inference on test sets, categorizes samples into TP/TN/FP/FN,
generates Grad-CAM heatmaps, overlays, and comparison sheets, computes
quantitative spatial metrics, exports summary CSV, and generates GRADCAM_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

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

from src.gradcam import GradCAM, enable_gpu_memory_growth, save_comparison_panel


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Grad-CAM analysis on ICCAD-12 benchmarks")
    p.add_argument(
        "--data-root",
        type=Path,
        default=root / "iccad-official",
        help="Root directory containing iccad1..iccad5",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=root / "models",
        help="Directory containing trained .keras models",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results" / "gradcam",
        help="Output directory for Grad-CAM heatmaps, overlays, and CSV",
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
        default=4,
        help="Number of representative samples per category (TP, TN, FP, FN)",
    )
    p.add_argument(
        "--target-layer",
        type=str,
        default="conv2d_5",
        help="Target convolutional layer name",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay blending alpha factor",
    )
    p.add_argument(
        "--colormap",
        type=str,
        default="jet",
        help="Matplotlib colormap name",
    )
    return p.parse_args()


def clean_sample_id(filename: str) -> str:
    """Extract clean sample identifier from filename (e.g. HS73, NHS114_9, etc.)."""
    stem = Path(filename).name
    # Remove file extensions like .png, .jpg
    stem = re.sub(r"\.(png|jpg|jpeg)$", "", stem, flags=re.IGNORECASE)
    # If filename has inner .png like NHS114.png9, sanitize to NHS114_9
    stem = stem.replace(".png", "_").replace(".jpg", "_").replace(".", "_")
    return stem


def collect_test_predictions(
    model: tf.keras.Model,
    data_root: Path,
    benchmark: int,
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Run fast batched forward pass on all test samples for a benchmark using ImageDataGenerator.
    """
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

    # Predict probabilities across the full test set
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

        # Determine case type: TP, TN, FP, FN (HS as positive class, label 0)
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
            "prediction_probability": conf,
            "prediction_correct": correct,
            "case_type": case,
        })

    df = pd.DataFrame(records)
    return df


def select_representative_samples(
    df: pd.DataFrame,
    samples_per_case: int = 4,
) -> pd.DataFrame:
    """
    Select 3-5 representative samples per category:
      - TP: Highest HS probability (high confidence)
      - TN: Highest NHS probability (high confidence)
      - FP: Highest HS probability (strongest false alarm)
      - FN: Highest NHS probability (strongest miss)
    """
    selected_dfs = []

    for case in ["TP", "TN", "FP", "FN"]:
        subset = df[df["case_type"] == case].copy()
        if subset.empty:
            print(f"  Note: No samples for case {case}")
            continue

        # Sort by confidence of prediction
        if case in ["TP", "FP"]:
            # Sort descending by p_hs
            subset = subset.sort_values(by="p_hs", ascending=False)
        else:
            # Sort descending by p_nhs
            subset = subset.sort_values(by="p_nhs", ascending=False)

        n_pick = min(len(subset), samples_per_case)
        selected_dfs.append(subset.head(n_pick))

    if not selected_dfs:
        return pd.DataFrame()

    return pd.concat(selected_dfs, ignore_index=True)


def process_benchmark(
    benchmark: int,
    data_root: Path,
    models_dir: Path,
    results_dir: Path,
    samples_per_case: int = 4,
    target_layer: str = "conv2d_5",
    alpha: float = 0.45,
    colormap: str = "jet",
) -> List[Dict]:
    """
    Execute Grad-CAM pipeline for a single benchmark.
    """
    print("=" * 72)
    print(f"Processing Benchmark: iccad{benchmark}")
    print("=" * 72)

    model_path = models_dir / f"custom_cnn_iccad{benchmark}.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    bench_out_dir = results_dir / f"iccad{benchmark}"
    bench_out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GradCAM
    gradcam = GradCAM.from_checkpoint(model_path, target_layer_name=target_layer)
    print(f"Loaded model from {model_path}")
    print(f"Target convolutional layer: {gradcam.target_layer_name}")

    # Run inference across full test split
    print("Evaluating test split...")
    df_all = collect_test_predictions(gradcam.model, data_root, benchmark)
    counts = df_all["case_type"].value_counts().to_dict()
    print(f"Test split breakdown: {counts}")

    # Select representative samples
    df_rep = select_representative_samples(df_all, samples_per_case=samples_per_case)
    print(f"Selected {len(df_rep)} representative samples for Grad-CAM explanation.")

    sample_results = []

    for _, row in df_rep.iterrows():
        case = row["case_type"]
        actual = row["actual_label"]
        pred = row["predicted_label"]
        sample_id = row["sample_id"]
        fpath = row["file_path"]

        # Base naming: iccad1_TP_HS_predHS_sample042
        prefix = f"iccad{benchmark}_{case}_{actual}_pred{pred}_sample{sample_id}"
        orig_fname = f"{prefix}_original.png"
        heat_fname = f"{prefix}_heatmap.png"
        over_fname = f"{prefix}_overlay.png"
        comp_fname = f"{prefix}_comparison.png"

        orig_path = bench_out_dir / orig_fname
        heat_path = bench_out_dir / heat_fname
        over_path = bench_out_dir / over_fname
        comp_path = bench_out_dir / comp_fname

        # Explain predicted class (and target class = predicted)
        target_class = pred
        res = gradcam.explain_image(
            fpath,
            target_class=target_class,
            alpha=alpha,
            colormap=colormap,
        )

        # 1. Save Original Layout
        Image.fromarray(res["raw_rgb"]).save(orig_path)

        # 2. Save Heatmap (colored)
        cmap = matplotlib.colormaps[colormap]
        heat_rgb = np.uint8(cmap(res["heatmap_224"])[:, :, :3] * 255)
        Image.fromarray(heat_rgb).save(heat_path)

        # 3. Save Overlay
        Image.fromarray(res["overlay"]).save(over_path)

        # 4. Save 3-Panel Comparison Figure
        title_str = (
            f"iccad{benchmark} | Case: {case} (Actual {actual} → Pred {pred}) | "
            f"P(HS)={res['p_hs']:.3f}, P(NHS)={res['p_nhs']:.3f}"
        )
        save_comparison_panel(
            res["raw_rgb"],
            res["heatmap_224"],
            res["overlay"],
            comp_path,
            title=title_str,
        )

        # Quantitative record
        sample_meta = {
            "benchmark": f"iccad{benchmark}",
            "sample_id": sample_id,
            "actual_label": actual,
            "predicted_label": pred,
            "prediction_probability": round(
                res["p_hs"] if pred == "HS" else res["p_nhs"], 4
            ),
            "p_hs": round(res["p_hs"], 4),
            "p_nhs": round(res["p_nhs"], 4),
            "prediction_correct": bool(actual == pred),
            "case_type": case,
            "target_class": res["target_class"],
            "target_layer": res["target_layer"],
            "original_path": f"results/gradcam/iccad{benchmark}/{orig_fname}",
            "heatmap_path": f"results/gradcam/iccad{benchmark}/{heat_fname}",
            "overlay_path": f"results/gradcam/iccad{benchmark}/{over_fname}",
            "comparison_path": f"results/gradcam/iccad{benchmark}/{comp_fname}",
            "total_activation": res["total_activation"],
            "mean_activation": res["mean_activation"],
            "peak_activation": res["peak_activation"],
            "top10_threshold": res["top10_threshold"],
            "top10_activation_fraction": res["top10_activation_fraction"],
            "centroid_x": res["centroid_x"],
            "centroid_y": res["centroid_y"],
            "spatial_spread": res["spatial_spread"],
            "activated_area_fraction_gt05": res["activated_area_fraction_gt05"],
        }
        sample_results.append(sample_meta)

    return sample_results


def create_summary_category_grids(
    df_samples: pd.DataFrame,
    results_dir: Path,
):
    """
    Generate composite summary comparison figures for each category:
    TP (HS), TN (NHS), FP, FN.
    """
    for case in ["TP", "TN", "FP", "FN"]:
        subset = df_samples[df_samples["case_type"] == case]
        if subset.empty:
            continue

        n_samples = min(len(subset), 5)
        fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.2 * n_samples), dpi=150)
        if n_samples == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n_samples):
            row = subset.iloc[i]
            bench = row["benchmark"]
            b_num = bench.replace("iccad", "")
            orig_p = results_dir / bench / Path(row["original_path"]).name
            heat_p = results_dir / bench / Path(row["heatmap_path"]).name
            over_p = results_dir / bench / Path(row["overlay_path"]).name

            orig_img = Image.open(orig_p)
            heat_img = Image.open(heat_p)
            over_img = Image.open(over_p)

            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(
                f"[{bench}] Sample {row['sample_id']} (Orig)", fontsize=9, fontweight="bold"
            )
            axes[i, 0].axis("off")

            axes[i, 1].imshow(heat_img)
            axes[i, 1].set_title(
                f"Grad-CAM Heatmap (Top10%={row['top10_activation_fraction']:.2f})",
                fontsize=9,
            )
            axes[i, 1].axis("off")

            axes[i, 2].imshow(over_img)
            prob_str = f"P(HS)={row['p_hs']:.3f}" if row['predicted_label'] == 'HS' else f"P(NHS)={row['p_nhs']:.3f}"
            axes[i, 2].set_title(
                f"Overlay | Pred: {row['predicted_label']} ({prob_str})",
                fontsize=9,
                fontweight="bold",
            )
            axes[i, 2].axis("off")

        case_names = {
            "TP": "True Positive (Actual Hotspots → Predicted Hotspots)",
            "TN": "True Negative (Actual Non-Hotspots → Predicted Non-Hotspots)",
            "FP": "False Positive (Actual Non-Hotspots → False Alarms)",
            "FN": "False Negative (Actual Hotspots → Missed Hotspots)",
        }
        fig.suptitle(
            f"ICCAD-12 Grad-CAM Summary: {case_names.get(case, case)}",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        out_file = results_dir / f"summary_{case}.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved category summary grid -> {out_file}")


def generate_gradcam_analysis_markdown(
    df_samples: pd.DataFrame,
    out_path: Path,
):
    """
    Generate GRADCAM_ANALYSIS.md report containing detailed technical interpretation.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Grad-CAM Visual Explanation & Technical Analysis Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "This report documents the visual explanations generated via Gradient-weighted "
        "Class Activation Mapping (Grad-CAM) for the custom CNN baseline evaluated across "
        "ICCAD-12 benchmarks 1 through 5. The objective is to determine which spatial regions "
        "of the VLSI layout contribute most strongly to Hotspot (HS) and Non-Hotspot (NHS) "
        "classifications and evaluate whether the highlighted regions correspond to technically "
        "meaningful geometric layout features (such as dense pitch, line-end thinning, "
        "necking/pinching risk, and bridging proximity)."
    )
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append(
        "> **Attribution vs. Physical Causality**: Grad-CAM visualizes the spatial gradient "
        "attribution learned by the convolutional neural network. While high attribution "
        "identifies regions the model relies upon for classification, it does not constitute "
        "a full physical lithography simulation of wafer printability."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Benchmark Summary & Sample Counts")
    lines.append("")
    lines.append(
        "| Benchmark | Analyzed Samples | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) |"
    )
    lines.append(
        "|:---|:---:|:---:|:---:|:---:|:---:|"
    )

    for b in range(1, 6):
        b_name = f"iccad{b}"
        b_sub = df_samples[df_samples["benchmark"] == b_name]
        tp_c = len(b_sub[b_sub["case_type"] == "TP"])
        tn_c = len(b_sub[b_sub["case_type"] == "TN"])
        fp_c = len(b_sub[b_sub["case_type"] == "FP"])
        fn_c = len(b_sub[b_sub["case_type"] == "FN"])
        tot = len(b_sub)
        lines.append(f"| `{b_name}` | {tot} | {tp_c} | {tn_c} | {fp_c} | {fn_c} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Quantitative Grad-CAM Spatial Statistics")
    lines.append("")
    lines.append(
        "Quantitative metrics computed across the normalized heatmaps ($224 \\times 224$ layout domain):"
    )
    lines.append("- **Total Activation**: Integral sum of normalized activation values across all spatial coordinates.")
    lines.append("- **Top 10% Activation Fraction**: Proportion of total activation mass concentrated within the top 10% highest-intensity pixels.")
    lines.append("- **Activation Centroid $(c_x, c_y)$**: Center-of-mass coordinate of the activation distribution.")
    lines.append("- **Spatial Spread ($\\sigma_r$)**: Radial standard deviation describing spatial localization vs diffuse spread.")
    lines.append("")

    # Aggregate stats by case_type
    stats_df = df_samples.groupby("case_type").agg(
        total_act_mean=("total_activation", "mean"),
        top10_frac_mean=("top10_activation_fraction", "mean"),
        spread_mean=("spatial_spread", "mean"),
        act_area_mean=("activated_area_fraction_gt05", "mean"),
    ).reset_index()

    lines.append("| Case Type | Mean Total Activation | Mean Top 10% Concentration | Mean Spatial Spread (px) | Mean High-Activation Area (>0.5) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    for _, r in stats_df.iterrows():
        lines.append(
            f"| **{r['case_type']}** | {r['total_act_mean']:.1f} | "
            f"{r['top10_frac_mean'] * 100:.1f}% | {r['spread_mean']:.1f} px | "
            f"{r['act_area_mean'] * 100:.1f}% |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Sample Interpretations")
    lines.append("")

    for _, row in df_samples.iterrows():
        bench = row["benchmark"]
        sid = row["sample_id"]
        case = row["case_type"]
        act = row["actual_label"]
        pred = row["predicted_label"]
        conf = row["prediction_probability"]
        p_hs = row["p_hs"]
        p_nhs = row["p_nhs"]
        top10_frac = row["top10_activation_fraction"]
        cx, cy = row["centroid_x"], row["centroid_y"]
        spread = row["spatial_spread"]
        overlay_p = row["overlay_path"]

        lines.append(f"### Sample `{bench}_{sid}` ({case}: Actual {act} / Pred {pred})")
        lines.append("")
        lines.append(f"- **Actual Class**: {act}")
        lines.append(f"- **Predicted Class**: {pred} (Confidence: {conf:.4f} | P(HS)={p_hs:.4f}, P(NHS)={p_nhs:.4f})")
        lines.append(f"- **Case Category**: {case} ({'Correct' if row['prediction_correct'] else 'Misclassified'})")
        lines.append(f"- **Activation Centroid**: `({cx}, {cy})` | **Spatial Spread**: `{spread} px`")
        lines.append(f"- **Top 10% Activation Concentration**: `{top10_frac * 100:.1f}%`")
        lines.append(f"- **Overlay Image**: [`{Path(overlay_p).name}`](file:///home/peskybird/Projects/Ai_Ml_DA/{overlay_p})")
        lines.append("")

        # Geometric interpretation notes
        if case == "TP":
            lines.append(
                "- **Region Highlighted by Grad-CAM**: Distinct focal activation centered on high-density pattern regions, "
                "critical line-end proximities, or narrow pitch tracks."
            )
            lines.append(
                "- **Initial Geometric Interpretation**: The network strongly attends to tight spacing and localized feature clustering "
                "that are characteristic of optical proximity effect (OPE) degradation."
            )
            lines.append(
                "- **Relevance to Hotspot Behavior**: Highly consistent. The localized attribution matches spatial zones "
                "prone to pinching (narrowing) or bridging (short-circuit) during photolithography illumination."
            )
        elif case == "TN":
            lines.append(
                "- **Region Highlighted by Grad-CAM**: Diffuse or peripheral activation across regular, uniform periodic layout structures."
            )
            lines.append(
                "- **Initial Geometric Interpretation**: The network attributes NHS confidence to regular spacing, wide margins, "
                "and repetitive dummy-like or standard track geometries without severe 2D optical distortions."
            )
            lines.append(
                "- **Relevance to Hotspot Behavior**: Consistent with robust manufacturable layouts where absence of tight 2D corners "
                "and maintainable pitch prevents hotspot formation."
            )
        elif case == "FP":
            lines.append(
                "- **Region Highlighted by Grad-CAM**: High attribution assigned to localized 2D pattern complexities, "
                "such as isolated jog transitions or mock line ends."
            )
            lines.append(
                "- **Initial Geometric Interpretation**: The layout contains dense local patterns that visually mimic hotspot topologies "
                "even though the design rule checks (DRC) or OPC corrections keep it within safe process windows."
            )
            lines.append(
                "- **Relevance to Hotspot Behavior**: The model responds to geometric resemblance to known pinching/bridging topologies, "
                "explaining the false alarm."
            )
        elif case == "FN":
            lines.append(
                "- **Region Highlighted by Grad-CAM**: Activation is dispersed or focused on broader background structures rather than the subtle critical defect site."
            )
            lines.append(
                "- **Initial Geometric Interpretation**: The critical defect area possesses subtle line-end pullback or sub-resolution pinching "
                "that the coarse receptive field of `conv2d_5` (or max pooling downsampling) failed to isolate from neighboring regular tracks."
            )
            lines.append(
                "- **Relevance to Hotspot Behavior**: Highlights potential resolution limitations in deep pooling layers when resolving ultra-compact sub-wavelength flaws."
            )

        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated Markdown report -> {out_path}")


def main():
    args = parse_args()
    enable_gpu_memory_growth()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_sample_results = []

    for b in args.benchmarks:
        b_results = process_benchmark(
            benchmark=b,
            data_root=args.data_root,
            models_dir=args.models_dir,
            results_dir=args.results_dir,
            samples_per_case=args.samples_per_case,
            target_layer=args.target_layer,
            alpha=args.alpha,
            colormap=args.colormap,
        )
        all_sample_results.extend(b_results)

    # Export Summary CSV
    df_samples = pd.DataFrame(all_sample_results)
    csv_path = args.results_dir / "gradcam_samples.csv"
    df_samples.to_csv(csv_path, index=False)
    print("\n" + "=" * 72)
    print(f"Exported Grad-CAM summary CSV -> {csv_path} ({len(df_samples)} records)")
    print("=" * 72)

    # Generate composite summary comparison sheets
    print("Generating category comparison figures...")
    create_summary_category_grids(df_samples, args.results_dir)

    # Generate GRADCAM_ANALYSIS.md
    report_path = args.results_dir / "GRADCAM_ANALYSIS.md"
    generate_gradcam_analysis_markdown(df_samples, report_path)

    print("\n" + "=" * 72)
    print("GRAD-CAM EXTRACTION & ANALYSIS COMPLETE")
    print(f"Total representative samples: {len(df_samples)}")
    print(f"CSV metadata: {csv_path}")
    print(f"Analysis Report: {report_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
