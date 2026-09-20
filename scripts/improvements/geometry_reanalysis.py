#!/usr/bin/env python3
"""Workstream C: historical geometry sensitivity analysis (CPU only, no inference).

Separates the 57-sample main geometry CSV into degeneracy strata using the stored
attribution-spread diagnostics (spread==0 => exact-flat map under the pipeline's own
rounding; NOT a raw ptp re-measurement), computes paired image-level top-20% statistics
with a pre-declared Holm-corrected family, and cross-checks the final 11-sample set's
exact is_degenerate_map flags. Existing results are never modified; all outputs go to
results/xai/improvements/geometry_reanalysis/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = ROOT / "results/xai/improvements/geometry_reanalysis"
METRICS = ["geometry_fraction", "narrow_gap_fraction", "thin_line_fraction", "corner_recall"]
LEVEL = 20
METHOD_SPREAD_COL = {
    "gradcam": "gcam_spread",
    "gradcam_plus": "gcam_pp_spread",
    "layercam": "layercam_spread",
    "occlusion": None,  # no spread diagnostic for occlusion in xai_diagnostics.csv
}


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvals[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return list(adjusted)


def main() -> None:
    main_geom = pd.read_csv(ROOT / "results/xai/geometry/geometry_diagnostics.csv")
    xai_diag = pd.read_csv(ROOT / "results/xai/xai_diagnostics.csv")
    final_geom = pd.read_csv(ROOT / "results/xai/final_iccad5/geometry/geometry_diagnostics.csv")
    tp_fp = pd.read_csv(ROOT / "results/xai/geometry/tp_fp/tp_fp_geometry_statistics.csv")

    # ---- Degeneracy strata from stored diagnostics -----------------------------
    # xai_diagnostics has one row per sample; spread==0 is the pipeline's own
    # diagnostic signature of an exactly-flat map (uint8-rounded, so it is a proxy,
    # not a raw ptp re-derivation).
    spread_map = {}
    for _, row in xai_diag.iterrows():
        for method, col in METHOD_SPREAD_COL.items():
            if col is not None:
                spread_map[(row["benchmark"], row["sample_id"], method)] = float(row[col])

    top = main_geom[main_geom["level_pct"] == LEVEL].copy()
    def stratum(row) -> str:
        if row["method"] == "occlusion":
            return "unknown_spread_not_stored"
        spread = spread_map.get((row["benchmark"], row["sample_id"], row["method"]))
        if spread is None:
            return "unknown"
        return "degenerate_spread0" if spread == 0.0 else "nondegenerate"
    top["degeneracy_stratum"] = top.apply(stratum, axis=1)

    counts = (
        top.groupby(["case_type", "method", "degeneracy_stratum"])
        .size()
        .reset_index(name="n_rows")
    )

    # ---- Paired per-sample diffs vs shared control, pre-declared family --------
    # Family: 4 metrics x 4 methods x 4 case types at top-20% = 64 tests, Holm-corrected.
    top = top.copy()
    for m in METRICS:
        top[f"paired_diff_{m}"] = top[m] - top[f"control_{m}_mean"]

    records = []
    for case in ["TP", "TN", "FP", "FN"]:
        for method in ["gradcam", "gradcam_plus", "layercam", "occlusion"]:
            block = top[(top["case_type"] == case) & (top["method"] == method)]
            strata = block["degeneracy_stratum"].value_counts().to_dict()
            for metric in METRICS:
                diffs = block[f"paired_diff_{metric}"].astype(float).values
                n = len(diffs)
                if n < 3 or np.ptp(diffs) == 0:
                    p = None
                    note = "insufficient_variation_or_n" if n < 3 else "constant_diffs"
                else:
                    p = float(stats.wilcoxon(diffs, alternative="two-sided", mode="approx").pvalue)
                    note = ""
                records.append({
                    "case_type": case, "method": method, "metric": metric,
                    "level_pct": LEVEL, "n": n,
                    "n_degenerate_spread0": strata.get("degenerate_spread0", 0),
                    "median_paired_diff": float(np.median(diffs)) if n else None,
                    "mean_paired_diff": float(np.mean(diffs)) if n else None,
                    "wilcoxon_p": p, "p_status": "computed" if p is not None else note,
                    "note": note,
                })
    fam = pd.DataFrame(records)
    mask = fam["wilcoxon_p"].notna()
    # Holm step-down implemented locally for exact family control (scipy ships
    # only 'bh'/'by' FDR variants in this version).
    fam.loc[mask, "holm_p"] = _holm(fam.loc[mask, "wilcoxon_p"].to_numpy())
    fam["holm_p"] = fam["holm_p"].where(mask)

    # ---- Final-set cross-check (exact flags) -----------------------------------
    final_top = final_geom[final_geom["level_pct"] == LEVEL]
    final_counts = (
        final_top.groupby(["case_type", "method", "is_degenerate_map"])
        .size().reset_index(name="n")
    )

    # ---- Report cross-check vs GEOMETRY_ANALYSIS_REPORT §4-8 -------------------
    reported = {  # (case, method) -> reported top-20% median geometry_fraction
        ("TP", "gradcam"): 0.412, ("TP", "gradcam_plus"): 0.263, ("TP", "layercam"): 0.347, ("TP", "occlusion"): 0.419,
        ("TN", "gradcam"): 0.027, ("TN", "gradcam_plus"): 0.153, ("TN", "layercam"): 0.226, ("TN", "occlusion"): 0.272,
        ("FP", "gradcam"): 0.416, ("FP", "gradcam_plus"): 0.337, ("FP", "layercam"): 0.387, ("FP", "occlusion"): 0.405,
        ("FN", "gradcam"): 0.109, ("FN", "gradcam_plus"): 0.205, ("FN", "layercam"): 0.265, ("FN", "occlusion"): 0.275,
    }
    checks = []
    for (case, method), claimed in reported.items():
        block = top[(top["case_type"] == case) & (top["method"] == method)]
        checks.append({
            "case_type": case, "method": method, "reported_median": claimed,
            "recomputed_median": round(float(block["geometry_fraction"].median()), 3),
            "n": len(block),
        })

    # ---- Plots ------------------------------------------------------------------
    plot_dir = OUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    methods = ["gradcam", "gradcam_plus", "layercam", "occlusion"]
    width = 0.2
    for i, method in enumerate(methods):
        for j, case in enumerate(["TP", "TN", "FP", "FN"]):
            block = top[(top["case_type"] == case) & (top["method"] == method)]
            ax.bar(i + (j - 1.5) * width, block["geometry_fraction"].median(), width,
                   label=f"{case}" if i == 0 else None)
    ax.set_xticks(range(4), methods)
    ax.set_ylabel(f"median geometry_fraction (top-{LEVEL}%)")
    ax.set_title("Historical 57-sample geometry correspondence by case type (bicubic provenance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "top20_geometry_fraction_by_case.png")
    plt.close(fig)

    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "stratum_counts.csv", index=False)
    fam.to_csv(OUT / "paired_tests_top20.csv", index=False)
    final_counts.to_csv(OUT / "final11_exact_flag_counts.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "report_crosscheck_top20.csv", index=False)

    tpfp_flags = tp_fp["is_degenerate_map"].value_counts().to_dict()
    summary = {
        "provenance": "historical bicubic artifacts; NOT corrected-preprocessing results",
        "degeneracy_proxy": "xai_diagnostics spread==0 per CAM method; occlusion has no stored spread proxy (unknown stratum)",
        "main57_top20_rows": int(len(top)),
        "stratum_counts": counts.to_dict(orient="records"),
        "tpfp_flag_counts": tpfp_flags,
        "final11_exact_flags": int((final_top["is_degenerate_map"] == True).sum()) if len(final_top) else 0,
        "family": "4 metrics x 4 methods x 4 case types, top-20% only, Holm within family",
        "tests_computed": int(mask.sum()),
        "limitations": [
            "spread==0 is the stored diagnostic proxy for exact flatness (uint8-rounded), not a raw ptp re-derivation",
            "occlusion degeneracy unknown in the main set; stratum reported, not imputed",
            "n=57 historical, descriptive/exploratory; Holm only within the pre-declared family",
            "flat-map rows retain arbitrary top-K masks in the source CSV; their overlap values are not attribution evidence",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(checks).to_string(index=False))


def _holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


if __name__ == "__main__":
    main()
