#!/usr/bin/env python3
"""Workstream J: attribution -> geometry enrichment ratios (CPU, historical provenance).

enrichment = geometry_fraction(top attribution) / geometry_fraction(random control mean)
per (sample, method, level), aggregated as medians per case type, with deterministic
bootstrap CIs over samples. Computed for geometry, narrow-gap and thin-line proxies.
Degenerate-map rows (spread==0 proxy) are counted and reported separately, never
silently pooled as attribution evidence. Zero-control ratios are undefined (excluded
and counted), never treated as zero enrichment. Outputs only under
results/xai/improvements/enrichment/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

OUT = ROOT / "results/xai/improvements/enrichment"
LEVEL = 20
PAIRS = [
    ("geometry", "geometry_fraction", "control_geometry_fraction_mean"),
    ("narrow_gap", "narrow_gap_fraction", "control_narrow_gap_fraction_mean"),
    ("thin_line", "thin_line_fraction", "control_thin_line_fraction_mean"),
]
SPREAD_COL = {"gradcam": "gcam_spread", "gradcam_plus": "gcam_pp_spread", "layercam": "layercam_spread"}


def bootstrap_ci(samples: np.ndarray, rng: np.random.Generator, draws: int = 5000) -> tuple[float, float]:
    boots = np.empty(draws, dtype=float)
    n = len(samples)
    for i in range(draws):
        boots[i] = np.median(samples[rng.integers(0, n, n)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    geom = pd.read_csv(ROOT / "results/xai/geometry/geometry_diagnostics.csv")
    xai = pd.read_csv(ROOT / "results/xai/xai_diagnostics.csv")

    # Degeneracy proxy per (benchmark, sample, CAM method); occlusion unknown in main set.
    flat = set()
    for _, row in xai.iterrows():
        for method, col in SPREAD_COL.items():
            if float(row[col]) == 0.0:
                flat.add((str(row["benchmark"]), str(row["sample_id"]), method))

    top = geom[geom["level_pct"] == LEVEL].copy()
    top["degenerate_spread0"] = [
        (m != "occlusion") and (b, s, m) in flat
        for b, s, m in zip(top["benchmark"], top["sample_id"], top["method"])
    ]

    rng = np.random.default_rng(20260917)
    records = []
    for case in ["TP", "TN", "FP", "FN"]:
        for method in ["gradcam", "gradcam_plus", "layercam", "occlusion"]:
            block = top[(top["case_type"] == case) & (top["method"] == method)]
            for label, num_col, ctl_col in PAIRS:
                num = block[num_col].to_numpy(dtype=float)
                ctl = block[ctl_col].to_numpy(dtype=float)
                defined = ctl > 0  # zero-control ratio is undefined, not zero enrichment
                n_defined = int(defined.sum())
                if n_defined:
                    ratios = num[defined] / ctl[defined]
                    median_ratio = float(np.median(ratios))
                    lo, hi = bootstrap_ci(ratios, rng)
                else:
                    median_ratio = lo = hi = None
                # Nondegenerate-only sensitivity for CAM methods.
                median_nd = None
                if method != "occlusion":
                    clean = block[~block["degenerate_spread0"]]
                    num2 = clean[num_col].to_numpy(dtype=float)
                    ctl2 = clean[ctl_col].to_numpy(dtype=float)
                    defined2 = ctl2 > 0
                    if defined_count := int(defined2.sum()) if False else int(defined2.sum()):
                        median_nd = float(np.median(num2[defined2] / ctl2[defined2]))
                records.append({
                    "case_type": case, "method": method, "metric": label,
                    "level_pct": LEVEL, "n_rows": int(len(block)),
                    "n_degenerate_spread0": int(block["degenerate_spread0"].sum()),
                    "n_zero_control_undefined": int(len(block)) - n_defined,
                    "n_defined": n_defined,
                    "median_enrichment": median_ratio,
                    "bootstrap_ci95_low": lo, "bootstrap_ci95_high": hi,
                    "median_enrichment_excl_degenerate": median_nd,
                })

    df = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "enrichment_top20.csv", index=False)
    summary = {
        "provenance": "historical bicubic artifacts; NOT corrected-preprocessing results",
        "definition": "median over samples of metric(attribution top-20%) / control mean(30 area-matched tile draws)",
        "degeneracy_proxy": "xai_diagnostics spread==0 for CAM methods; occlusion status unknown in main set",
        "uncertainty": "5000-draw deterministic bootstrap (rng seed 20260917) over samples, percentile CI",
        "n_undefined_ratios_zero_control": int(df["n_zero_control_undefined"].sum()),
        "limitations": [
            "ratios of near-zero quantities are unstable; CI width and n_defined must accompany every value",
            "one level (top-20%); flat-map rows counted but excluded from the nondegenerate column, occlusion not stratifiable",
            "tile-grid control null is approximate-area (~2% larger) and tile-shaped, per audit A-07",
            "correspondence claim only; no causal or physical-lithography reading",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))
    piv = df.pivot_table(index=["case_type", "method"], columns="metric",
                         values="median_enrichment", dropna=False)
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
