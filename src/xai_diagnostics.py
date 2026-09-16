"""
Quantitative Attribution Diagnostics for Explainable Lithography Hotspot Detection.

Computes technical spatial statistics from normalized 2D heatmaps (in [0, 1]).

Note:
These metrics are mathematical and spatial attribution statistics describing the
network's gradient focus. They must NOT be interpreted as physical causal percentages
or photolithography process simulations.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import scipy.ndimage as ndi


def compute_attribution_diagnostics(
    heatmap_224: np.ndarray,
    connected_threshold: float = 0.5,
    border_fraction: float = 0.10,
) -> Dict[str, float]:
    """
    Compute rigorous spatial attribution statistics.

    Args:
      heatmap_224: 2D numpy array of shape (H, W), values in [0.0, 1.0].
      connected_threshold: Binarization threshold for connected-component analysis.
      border_fraction: Fraction of margin defining the outer perimeter border strip.

    Returns:
      Dictionary containing quantitative diagnostic metrics.
    """
    h = np.clip(np.asarray(heatmap_224, dtype=np.float32), 0.0, 1.0)
    H, W = h.shape
    total_pixels = float(H * W)
    h_sum = float(np.sum(h))

    # Peak Attribution Location
    max_idx = np.unravel_index(np.argmax(h), h.shape)
    peak_y, peak_x = int(max_idx[0]), int(max_idx[1])
    peak_val = float(h[peak_y, peak_x])

    if h_sum <= 1e-9:
        return {
            "peak_attribution_x": peak_x,
            "peak_attribution_y": peak_y,
            "peak_attribution_val": 0.0,
            "centroid_x": float(W / 2.0),
            "centroid_y": float(H / 2.0),
            "spatial_spread_radius": 0.0,
            "fraction_above_025": 0.0,
            "fraction_above_050": 0.0,
            "fraction_above_075": 0.0,
            "connected_regions_count": 0,
            "largest_region_area_px": 0,
            "largest_region_area_frac": 0.0,
            "edge_border_concentration": 0.0,
            "total_attribution_mass": 0.0,
        }

    # Attribution-Weighted Centroid
    y_coords, x_coords = np.indices((H, W))
    centroid_x = float(np.sum(x_coords * h) / h_sum)
    centroid_y = float(np.sum(y_coords * h) / h_sum)

    # Spatial Spread (weighted standard deviation from centroid)
    sq_dist = (x_coords - centroid_x) ** 2 + (y_coords - centroid_y) ** 2
    spread_radius = float(np.sqrt(np.sum(sq_dist * h) / h_sum))

    # Fraction of layout area above activation thresholds
    frac_gt_025 = float(np.sum(h >= 0.25) / total_pixels)
    frac_gt_050 = float(np.sum(h >= 0.50) / total_pixels)
    frac_gt_075 = float(np.sum(h >= 0.75) / total_pixels)

    # Connected Attribution Regions (at specified threshold, default 0.5)
    binary_mask = h >= connected_threshold
    labeled_array, num_features = ndi.label(binary_mask)

    if num_features > 0:
        # Measure size of each connected component
        component_sizes = ndi.sum(binary_mask, labeled_array, range(1, num_features + 1))
        largest_area_px = int(np.max(component_sizes))
        largest_area_frac = float(largest_area_px / total_pixels)
    else:
        largest_area_px = 0
        largest_area_frac = 0.0

    # Edge / Peripheral Concentration (fraction of mass in outer perimeter)
    margin_y = int(round(H * border_fraction))
    margin_x = int(round(W * border_fraction))
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[margin_y : H - margin_y, margin_x : W - margin_x] = False

    edge_mass = float(np.sum(h[border_mask]))
    edge_concentration = float(edge_mass / h_sum)

    return {
        "peak_attribution_x": peak_x,
        "peak_attribution_y": peak_y,
        "peak_attribution_val": round(peak_val, 4),
        "centroid_x": round(centroid_x, 2),
        "centroid_y": round(centroid_y, 2),
        "spatial_spread_radius": round(spread_radius, 2),
        "fraction_above_025": round(frac_gt_025, 4),
        "fraction_above_050": round(frac_gt_050, 4),
        "fraction_above_075": round(frac_gt_075, 4),
        "connected_regions_count": int(num_features),
        "largest_region_area_px": largest_area_px,
        "largest_region_area_frac": round(largest_area_frac, 4),
        "edge_border_concentration": round(edge_concentration, 4),
        "total_attribution_mass": round(h_sum, 2),
    }
