#!/usr/bin/env python3
"""
TP vs FP Raw Geometry Distribution Analysis.

Extracts raw (continuous) raster-derived geometry values -- local width, local gap,
local density, boundary distance -- within XAI attribution top-K% regions for TP and FP
samples, per method, per level. Reuses the exact feature definitions in
src/geometry_features.py / docs/RASTER_GEOMETRY_METHOD.md; no new geometry definitions.
Does not retrain or modify the CNN/XAI implementations.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.geometry_features import binarize_raster, compute_geometry_features, topk_mask_by_area
from src.run_geometry_analysis import get_attribution_maps, LEVELS, METHODS
from src.xai_engine import XAIEngine

OUT_DIR = ROOT / "results" / "xai" / "geometry" / "tp_fp"
FEATURES = ["mean_local_width_px", "mean_local_gap_px", "mean_local_density", "mean_boundary_distance_px"]


def region_raw_features(mask: np.ndarray, geom: dict) -> dict:
    geom_px = mask & geom["geometry_mask"]
    bg_px = mask & (~geom["geometry_mask"])

    width_map = geom.get("_width_map")
    gap_map = geom.get("_gap_map")

    mean_width = float(width_map[geom_px].mean()) if geom_px.any() else np.nan
    mean_gap = float(gap_map[bg_px].mean()) if bg_px.any() else np.nan
    mean_density = float(geom["density_map"][mask].mean())
    mean_boundary = float(geom["boundary_distance_map"][mask].mean())

    return {
        "mean_local_width_px": round(mean_width, 3) if not np.isnan(mean_width) else np.nan,
        "mean_local_gap_px": round(mean_gap, 3) if not np.isnan(mean_gap) else np.nan,
        "mean_local_density": round(mean_density, 4),
        "mean_boundary_distance_px": round(mean_boundary, 3),
        "n_geometry_px_in_mask": int(geom_px.sum()),
        "n_background_px_in_mask": int(bg_px.sum()),
        "mask_area_px": int(mask.sum()),
    }


def compute_geometry_features_with_raw(geometry_mask: np.ndarray) -> dict:
    """Wrap compute_geometry_features and also expose raw width_map/gap_map for this analysis."""
    import scipy.ndimage as ndi
    geom = compute_geometry_features(geometry_mask)
    bg_mask = ~geometry_mask
    geom["_width_map"] = np.where(geometry_mask, 2.0 * ndi.distance_transform_edt(geometry_mask), 0.0)
    geom["_gap_map"] = np.where(bg_mask, 2.0 * ndi.distance_transform_edt(bg_mask), 0.0)
    return geom


def main():
    diag_csv = ROOT / "results" / "xai" / "xai_diagnostics.csv"
    df = pd.read_csv(diag_csv)
    df = df[df["case_type"].isin(["TP", "FP"])].reset_index(drop=True)
    occlusion_dir = ROOT / "results" / "xai" / "occlusion"

    records = []
    for benchmark in sorted(df["benchmark"].unique()):
        bench_rows = df[df["benchmark"] == benchmark]
        model_path = ROOT / "models" / "xai" / f"xai_cnn_{benchmark}.keras"
        engine = XAIEngine.from_checkpoint(model_path, target_layer_name="conv_final_2")
        print(f"=== {benchmark}: {len(bench_rows)} TP/FP samples ===")
        for _, row in bench_rows.iterrows():
            sample_dir = ROOT / row["sample_dir"]
            sample_prefix = sample_dir.name
            occ_dir = occlusion_dir / benchmark / sample_prefix

            img_tensor, raw_rgb = engine.preprocess_image(sample_dir / "original.png")
            geometry_mask = binarize_raster(raw_rgb)
            geom = compute_geometry_features_with_raw(geometry_mask)
            attr_maps = get_attribution_maps(engine, img_tensor, row["predicted_label"], occ_dir)

            for level in LEVELS:
                for method in METHODS:
                    amap = attr_maps[method]
                    # A perfectly flat attribution map (e.g. an exactly-zero gradient CAM,
                    # see docs/TP_FP_GEOMETRY_ANALYSIS.md Sec 9) has no well-defined "top-K%"
                    # region -- argpartition still returns *a* region, but it is an arbitrary
                    # tie-break, not a meaningful attribution area. Flag it rather than silently
                    # treating it as real signal.
                    is_degenerate = bool(np.ptp(amap) < 1e-6)
                    mask = topk_mask_by_area(amap, level)
                    feats = region_raw_features(mask, geom)
                    records.append({
                        "benchmark": benchmark,
                        "sample_id": row["sample_id"],
                        "case_type": row["case_type"],
                        "method": method,
                        "level_pct": int(round(level * 100)),
                        "is_degenerate_map": is_degenerate,
                        **feats,
                    })
        print(f"  done {benchmark}")

    raw_df = pd.DataFrame(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(OUT_DIR / "tp_fp_geometry_statistics.csv", index=False)
    print(f"Saved raw per-sample rows -> {OUT_DIR / 'tp_fp_geometry_statistics.csv'} ({len(raw_df)} rows)")
    n_deg = raw_df["is_degenerate_map"].sum()
    print(f"Degenerate (flat) attribution maps: {n_deg}/{len(raw_df)} rows "
          f"({raw_df.groupby('method')['is_degenerate_map'].mean().round(3).to_dict()})")

    build_group_stats(raw_df, exclude_degenerate=False)
    build_group_stats(raw_df, exclude_degenerate=True)


def cliffs_delta_from_u(u_stat, n1, n2):
    """Cliff's delta = 2*AUC - 1, where AUC = U / (n1*n2)."""
    auc = u_stat / (n1 * n2)
    delta = 2 * auc - 1
    return auc, delta


def build_group_stats(raw_df: pd.DataFrame, exclude_degenerate: bool = False):
    rows = []
    src = raw_df[~raw_df["is_degenerate_map"]] if exclude_degenerate else raw_df
    for (method, level), sub in src.groupby(["method", "level_pct"]):
        n_deg_tp = int(raw_df[(raw_df.method == method) & (raw_df.level_pct == level) & (raw_df.case_type == "TP")]["is_degenerate_map"].sum())
        n_deg_fp = int(raw_df[(raw_df.method == method) & (raw_df.level_pct == level) & (raw_df.case_type == "FP")]["is_degenerate_map"].sum())
        for feat in FEATURES:
            tp = sub[sub.case_type == "TP"][feat].dropna().values
            fp = sub[sub.case_type == "FP"][feat].dropna().values
            n_tp, n_fp = len(tp), len(fp)
            rec = {
                "method": method, "level_pct": level, "feature": feat,
                "tp_n": n_tp, "fp_n": n_fp,
                "tp_n_degenerate_excluded": n_deg_tp if exclude_degenerate else 0,
                "fp_n_degenerate_excluded": n_deg_fp if exclude_degenerate else 0,
                "tp_median": np.median(tp) if n_tp else np.nan,
                "tp_iqr": (np.percentile(tp, 75) - np.percentile(tp, 25)) if n_tp else np.nan,
                "tp_mean": np.mean(tp) if n_tp else np.nan,
                "tp_std": np.std(tp) if n_tp else np.nan,
                "fp_median": np.median(fp) if n_fp else np.nan,
                "fp_iqr": (np.percentile(fp, 75) - np.percentile(fp, 25)) if n_fp else np.nan,
                "fp_mean": np.mean(fp) if n_fp else np.nan,
                "fp_std": np.std(fp) if n_fp else np.nan,
            }
            if n_tp >= 3 and n_fp >= 3:
                try:
                    u_stat, p_val = mannwhitneyu(tp, fp, alternative="two-sided")
                    auc, delta = cliffs_delta_from_u(u_stat, n_tp, n_fp)
                    rec["mannwhitney_u"] = u_stat
                    rec["p_value"] = p_val
                    rec["auc_tp_vs_fp"] = round(auc, 4)
                    rec["cliffs_delta"] = round(delta, 4)
                except ValueError:
                    rec.update({"mannwhitney_u": np.nan, "p_value": np.nan, "auc_tp_vs_fp": np.nan, "cliffs_delta": np.nan})
            else:
                rec.update({"mannwhitney_u": np.nan, "p_value": np.nan, "auc_tp_vs_fp": np.nan, "cliffs_delta": np.nan})
            rows.append(rec)

    stats_df = pd.DataFrame(rows)
    for col in ["tp_median", "tp_iqr", "tp_mean", "tp_std", "fp_median", "fp_iqr", "fp_mean", "fp_std"]:
        stats_df[col] = stats_df[col].round(4)
    suffix = "_excl_degenerate" if exclude_degenerate else ""
    out_path = OUT_DIR / f"tp_fp_sample_statistics{suffix}.csv"
    stats_df.to_csv(out_path, index=False)
    print(f"Saved group statistics -> {out_path} ({len(stats_df)} rows)")


if __name__ == "__main__":
    main()
