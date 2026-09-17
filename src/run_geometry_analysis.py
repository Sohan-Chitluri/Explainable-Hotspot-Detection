#!/usr/bin/env python3
"""
Raster-Geometry Correlation Pipeline.

For the 57 representative ICCAD-12 samples already used in the XAI/Occlusion pipelines,
extracts raster-derived geometry proxies (src/geometry_features.py) and measures spatial
overlap/association between XAI attribution top-K% regions (Grad-CAM, Grad-CAM++,
LayerCAM, Occlusion) and those geometry proxies, against area-matched randomized
tile-grid controls. See docs/RASTER_GEOMETRY_METHOD.md for the full methodology.

Does NOT retrain or modify the CNN. Does NOT add a new XAI method. All geometry features
are raster-derived proxies, not EDA/DRC ground truth (docs/XAI_RESEARCH_AUDIT.md Sec 6).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.geometry_features import (
    binarize_raster, compute_geometry_features, topk_mask_by_area,
    generate_control_masks, compute_overlap_metrics, METRIC_NAMES, CONTROL_DRAWS,
)
from src.xai_engine import XAIEngine

LEVELS = [0.10, 0.20, 0.30]
METHODS = ["gradcam", "gradcam_plus", "layercam", "occlusion"]
PANEL_LEVEL = 0.20  # level visualized in the 4-panel overlap figures


def seeded_rng(*parts: str) -> np.random.Generator:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return np.random.default_rng(int(h[:8], 16))


def get_attribution_maps(engine: XAIEngine, img_tensor: np.ndarray, pred: str, occ_dir: Path) -> Dict[str, np.ndarray]:
    _, gcam_h, _ = engine.explain_gradcam(img_tensor, target_class=pred)
    _, gcam_pp_h, _ = engine.explain_gradcam_plus_plus(img_tensor, target_class=pred)
    _, lcam_h, _ = engine.explain_layercam(img_tensor, target_class=pred)

    raw_occ = np.load(occ_dir / "raw_occlusion_map.npy")
    pos_occ = np.maximum(raw_occ, 0.0)
    occ_norm = pos_occ / (float(np.max(pos_occ)) + 1e-10)

    return {"gradcam": gcam_h, "gradcam_plus": gcam_pp_h, "layercam": lcam_h, "occlusion": occ_norm}


def geometry_composite_rgb(geom: Dict[str, np.ndarray]) -> np.ndarray:
    """RGB visualization: geometry=gray, thin_line=yellow, narrow_gap=cyan, junction=red, corner=magenta."""
    H, W = geom["geometry_mask"].shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[geom["geometry_mask"]] = [0.45, 0.45, 0.45]
    rgb[geom["thin_line_mask"]] = [1.0, 0.9, 0.0]
    rgb[geom["narrow_gap_mask"]] = [0.0, 0.85, 0.95]
    rgb[geom["junction_mask"]] = [0.95, 0.1, 0.1]
    rgb[geom["corner_mask"]] = [0.9, 0.0, 0.9]
    return rgb


def save_panel(raw_rgb, geom, attr_maps, sample_prefix, title, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140)

    axes[0, 0].imshow(raw_rgb)
    axes[0, 0].set_title("Original Layout", fontsize=10, fontweight="bold")

    axes[0, 1].imshow(geometry_composite_rgb(geom))
    axes[0, 1].set_title("Geometry Proxies\n(gray=geom, yellow=thin, cyan=narrow-gap,\nred=junction, magenta=corner)", fontsize=8)

    axes[0, 2].imshow(geom["density_map"], cmap="viridis")
    axes[0, 2].set_title("Local Density (32x32)", fontsize=10, fontweight="bold")

    axes[0, 3].axis("off")

    for i, method in enumerate(METHODS):
        ax = axes[1, i]
        mask = topk_mask_by_area(attr_maps[method], PANEL_LEVEL)
        ax.imshow(raw_rgb)
        overlap = mask & geom["geometry_mask"]
        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[mask] = [0.1, 0.4, 1.0, 0.45]
        rgba[overlap] = [1.0, 0.15, 0.15, 0.55]
        ax.imshow(rgba)
        gfrac = float(overlap.sum() / max(1, mask.sum()))
        ax.set_title(f"{method}\ntop-20% (blue) vs geometry (red=overlap)\ngeometry_fraction={gfrac:.2f}", fontsize=8)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def process_sample(engine, row, occlusion_dir_root, geometry_dir_root, save_plot: bool) -> List[dict]:
    benchmark = row["benchmark"]
    sid = row["sample_id"]
    case = row["case_type"]
    pred = row["predicted_label"]
    sample_dir = ROOT / row["sample_dir"]
    sample_prefix = sample_dir.name
    occ_dir = occlusion_dir_root / benchmark / sample_prefix

    img_tensor, raw_rgb = engine.preprocess_image(sample_dir / "original.png")
    geometry_mask = binarize_raster(raw_rgb)
    geom = compute_geometry_features(geometry_mask)
    attr_maps = get_attribution_maps(engine, img_tensor, pred, occ_dir)

    H, W = geometry_mask.shape
    records = []
    for level in LEVELS:
        target_area = max(1, round(level * H * W))
        rng = seeded_rng(benchmark, sid, f"{level:.2f}")
        control_masks = generate_control_masks(H, W, target_area, CONTROL_DRAWS, rng)
        control_area_px = int(control_masks[0].sum())

        control_metric_arrays = {m: [] for m in METRIC_NAMES}
        for cmask in control_masks:
            cm = compute_overlap_metrics(cmask, geom)
            for m in METRIC_NAMES:
                control_metric_arrays[m].append(cm[m])

        control_summary = {}
        for m in METRIC_NAMES:
            arr = np.array(control_metric_arrays[m])
            control_summary[f"control_{m}_mean"] = round(float(arr.mean()), 4)
            control_summary[f"control_{m}_std"] = round(float(arr.std()), 4)
            control_summary[f"control_{m}_min"] = round(float(arr.min()), 4)
            control_summary[f"control_{m}_max"] = round(float(arr.max()), 4)

        for method in METHODS:
            amap = attr_maps[method]
            # A perfectly flat attribution map (e.g. an exactly-zero gradient CAM, see
            # docs/TP_FP_GEOMETRY_ANALYSIS.md Sec 9) has no well-defined "top-K%" region --
            # argpartition still returns *a* region, but it is an arbitrary tie-break, not
            # a meaningful attribution area. Flag it rather than silently treating it as
            # real signal (ported from src/run_tp_fp_geometry.py).
            is_degenerate_map = bool(np.ptp(amap) < 1e-6)
            mask = topk_mask_by_area(amap, level)
            metrics = compute_overlap_metrics(mask, geom)
            rec = {
                "benchmark": benchmark,
                "sample_id": sid,
                "case_type": case,
                "actual_label": row["actual_label"],
                "predicted_label": pred,
                "method": method,
                "level_pct": int(round(level * 100)),
                "is_degenerate_map": is_degenerate_map,
                "mask_area_px": int(mask.sum()),
                "target_area_px": target_area,
                "control_area_px": control_area_px,
                "control_n_draws": CONTROL_DRAWS,
                "n_components": geom["n_components"],
                "largest_component_area_frac": round(geom["largest_component_area_frac"], 4),
                "n_junction_seeds": geom["n_junction_seeds"],
                "n_corner_seeds": geom["n_corner_seeds"],
                **metrics,
                **control_summary,
            }
            records.append(rec)

    if save_plot:
        out_path = geometry_dir_root / benchmark / sample_prefix / "geometry_overlap_panel.png"
        title = f"{benchmark} [{case}] {sid} | Actual: {row['actual_label']} -> Pred: {pred}"
        save_panel(raw_rgb, geom, attr_maps, sample_prefix, title, out_path)

    return records


def main():
    p = argparse.ArgumentParser(description="Raster-Geometry Correlation Pipeline")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models" / "xai")
    p.add_argument("--xai-results-dir", type=Path, default=ROOT / "results" / "xai")
    p.add_argument("--occlusion-dir", type=Path, default=ROOT / "results" / "xai" / "occlusion")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "xai" / "geometry")
    p.add_argument("--smoke-test", action="store_true", help="Run on 4 samples (1 TP/TN/FP/FN) only")
    p.add_argument("--benchmarks", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    args = p.parse_args()

    diag_csv = args.xai_results_dir / "xai_diagnostics.csv"
    df = pd.read_csv(diag_csv)

    if args.smoke_test:
        df = df[df["benchmark"] == "iccad3"].groupby("case_type").head(1).reset_index(drop=True)
        print(f"SMOKE TEST: running on {len(df)} samples: {df['sample_id'].tolist()}")
    else:
        df = df[df["benchmark"].isin([f"iccad{b}" for b in args.benchmarks])].reset_index(drop=True)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for benchmark in sorted(df["benchmark"].unique()):
        bench_rows = df[df["benchmark"] == benchmark]
        model_path = args.models_dir / f"xai_cnn_{benchmark}.keras"
        engine = XAIEngine.from_checkpoint(model_path, target_layer_name="conv_final_2")
        print(f"\n=== {benchmark}: {len(bench_rows)} samples ===")
        for idx, (_, row) in enumerate(bench_rows.iterrows(), 1):
            recs = process_sample(engine, row, args.occlusion_dir, args.results_dir, save_plot=True)
            all_records.extend(recs)
            print(f"  [{idx}/{len(bench_rows)}] {row['sample_id']} ({row['case_type']}) -> {len(recs)} rows")

    out_df = pd.DataFrame(all_records)
    out_csv = args.results_dir / "geometry_diagnostics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out_df)} rows -> {out_csv}")
    n_deg = out_df["is_degenerate_map"].sum()
    print(f"Degenerate (flat) attribution maps: {n_deg}/{len(out_df)} rows "
          f"({out_df.groupby('method')['is_degenerate_map'].mean().round(3).to_dict()})")

    if not args.smoke_test:
        build_summary(out_df, args.results_dir, exclude_degenerate=False)
        build_summary(out_df, args.results_dir, exclude_degenerate=True)
        build_summary_plots(out_df, args.results_dir / "plots")


def build_summary(df: pd.DataFrame, results_dir: Path, exclude_degenerate: bool = False):
    from src.geometry_features import METRIC_NAMES as METRICS
    rows = []
    group_cols = ["case_type", "benchmark", "method", "level_pct"]
    src = df[~df["is_degenerate_map"]] if exclude_degenerate else df
    for keys, sub in src.groupby(group_cols):
        rec = dict(zip(group_cols, keys))
        rec["n_samples"] = len(sub)
        n_deg = int(df[
            (df["case_type"] == rec["case_type"]) & (df["benchmark"] == rec["benchmark"]) &
            (df["method"] == rec["method"]) & (df["level_pct"] == rec["level_pct"])
        ]["is_degenerate_map"].sum())
        rec["n_degenerate_excluded"] = n_deg if exclude_degenerate else 0
        for m in METRICS:
            vals = sub[m].values
            ctrl_vals = sub[f"control_{m}_mean"].values
            rec[f"{m}_median"] = round(float(np.median(vals)), 4)
            rec[f"{m}_iqr"] = round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4)
            rec[f"{m}_std"] = round(float(np.std(vals)), 4)
            rec[f"{m}_control_median"] = round(float(np.median(ctrl_vals)), 4)
            rec[f"{m}_paired_diff_median"] = round(float(np.median(vals - ctrl_vals)), 4)
        rows.append(rec)
    summary_df = pd.DataFrame(rows)
    suffix = "_excl_degenerate" if exclude_degenerate else ""
    out_csv = results_dir / f"summary{suffix}.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved summary -> {out_csv} ({len(summary_df)} rows)")


def build_summary_plots(df: pd.DataFrame, plots_dir: Path):
    plots_dir.mkdir(parents=True, exist_ok=True)
    key_metrics = ["geometry_fraction", "narrow_gap_fraction", "thin_line_fraction"]
    level = 20
    sub = df[df["level_pct"] == level]

    for metric in key_metrics:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
        case_types = ["TP", "TN", "FP", "FN"]
        x = np.arange(len(case_types))
        width = 0.18
        for i, method in enumerate(METHODS):
            means, ctrl_means = [], []
            for ct in case_types:
                s = sub[(sub["method"] == method) & (sub["case_type"] == ct)]
                means.append(s[metric].mean() if len(s) else np.nan)
                ctrl_means.append(s[f"control_{metric}_mean"].mean() if len(s) else np.nan)
            ax.bar(x + i * width, means, width, label=f"{method} (attr)")
        ax.plot(x + 1.5 * width, [sub[(sub["case_type"] == ct)][f"control_{metric}_mean"].mean() for ct in case_types],
                "k--o", label="control (mean)")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(case_types)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} at top-{level}% attribution vs. random control, by case type")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{metric}_top{level}_by_case.png")
        plt.close(fig)
    print(f"Saved summary plots -> {plots_dir}")


if __name__ == "__main__":
    main()
