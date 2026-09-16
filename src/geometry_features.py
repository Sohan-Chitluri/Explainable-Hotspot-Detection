"""
Raster-Derived Geometry Feature Extraction for XAI-Geometry Correlation Studies.

Implements ONLY raster/image-processing derived proxies for layout geometry, per
docs/RASTER_GEOMETRY_METHOD.md. None of these features are EDA/DRC ground truth and
none should be interpreted as verified lithography-mechanism annotations -- the ICCAD-12
data available in this repository is a binary raster image only (see
docs/XAI_RESEARCH_AUDIT.md Section 6).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.feature import corner_harris, corner_peaks
from skimage.morphology import skeletonize, dilation, disk
from skimage.segmentation import find_boundaries

TILE = 32  # matches occlusion window size and the non-overlapping control grid
CONTROL_DRAWS = 30
PERCENTILE = 20  # bottom-quintile for "narrow"/"thin" relative thresholds


def binarize_raster(raw_rgb: np.ndarray) -> np.ndarray:
    """Binarize a preprocessed 224x224x3 uint8 raster at mid-gray (127)."""
    gray = raw_rgb[..., 0] if raw_rgb.ndim == 3 else raw_rgb
    return gray >= 127


def compute_geometry_features(geometry_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute all raster-derived geometry feature layers for a single binary mask."""
    H, W = geometry_mask.shape
    bg_mask = ~geometry_mask

    # 1. Connected components
    labeled, n_components = ndi.label(geometry_mask, structure=np.ones((3, 3)))
    if n_components > 0:
        sizes = ndi.sum(geometry_mask, labeled, range(1, n_components + 1))
        largest_component_area_frac = float(np.max(sizes) / (H * W))
    else:
        largest_component_area_frac = 0.0

    # 2. Distance-transform width/gap maps
    width_map = np.where(geometry_mask, 2.0 * ndi.distance_transform_edt(geometry_mask), 0.0)
    gap_map = np.where(bg_mask, 2.0 * ndi.distance_transform_edt(bg_mask), 0.0)

    nz_width = width_map[width_map > 0]
    nz_gap = gap_map[gap_map > 0]
    width_thresh = float(np.percentile(nz_width, PERCENTILE)) if nz_width.size else 0.0
    gap_thresh = float(np.percentile(nz_gap, PERCENTILE)) if nz_gap.size else 0.0

    thin_line_mask = (width_map > 0) & (width_map <= width_thresh)
    narrow_gap_mask = (gap_map > 0) & (gap_map <= gap_thresh)

    # 3. Skeleton
    skeleton = skeletonize(geometry_mask)

    # 4. Junction detection (skeleton degree >= 3)
    if skeleton.any():
        neighbor_count = ndi.convolve(
            skeleton.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant"
        ) - skeleton.astype(np.int32)
        junction_seeds = skeleton & (neighbor_count >= 3)
        junction_mask = dilation(junction_seeds, footprint=disk(2)) if junction_seeds.any() else np.zeros_like(geometry_mask)
    else:
        junction_seeds = np.zeros_like(geometry_mask)
        junction_mask = np.zeros_like(geometry_mask)

    # 5. Corner/turn detection (generic Harris corner response on binary mask)
    harris_resp = corner_harris(geometry_mask.astype(np.float64), sigma=1)
    corner_coords = corner_peaks(harris_resp, min_distance=5, threshold_rel=0.01)
    corner_seeds = np.zeros_like(geometry_mask)
    if len(corner_coords) > 0:
        corner_seeds[corner_coords[:, 0], corner_coords[:, 1]] = True
        corner_mask = dilation(corner_seeds, footprint=disk(2))
    else:
        corner_mask = np.zeros_like(geometry_mask)

    # 6. Local geometry density (32x32 box average)
    density_map = ndi.uniform_filter(geometry_mask.astype(np.float32), size=TILE)

    # Boundary proximity (distance to nearest geometry/background boundary, whole canvas)
    boundary = find_boundaries(geometry_mask, mode="inner") | find_boundaries(geometry_mask, mode="outer")
    boundary_distance_map = ndi.distance_transform_edt(~boundary).astype(np.float32)

    return {
        "geometry_mask": geometry_mask,
        "narrow_gap_mask": narrow_gap_mask,
        "thin_line_mask": thin_line_mask,
        "junction_mask": junction_mask,
        "corner_mask": corner_mask,
        "density_map": density_map,
        "boundary_distance_map": boundary_distance_map,
        "skeleton": skeleton,
        "n_components": int(n_components),
        "largest_component_area_frac": largest_component_area_frac,
        "n_junction_seeds": int(junction_seeds.sum()),
        "n_corner_seeds": int(len(corner_coords)),
        "width_thresh_px": width_thresh,
        "gap_thresh_px": gap_thresh,
    }


def topk_mask_by_area(attr_map: np.ndarray, frac: float) -> np.ndarray:
    """Exact top-`frac` fraction of pixels by attribution value (area-matched, not value-thresholded)."""
    H, W = attr_map.shape
    n_pixels = H * W
    k = max(1, round(frac * n_pixels))
    flat = attr_map.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(n_pixels, dtype=bool)
    mask[idx] = True
    return mask.reshape(H, W)


def generate_control_masks(
    H: int, W: int, target_area_px: int, n_draws: int, rng: np.random.Generator
) -> List[np.ndarray]:
    """Non-overlapping tile-grid random control masks, area-matched to target_area_px."""
    assert H % TILE == 0 and W % TILE == 0
    n_tile_rows, n_tile_cols = H // TILE, W // TILE
    n_total_tiles = n_tile_rows * n_tile_cols
    n_tiles = int(np.clip(round(target_area_px / (TILE * TILE)), 1, n_total_tiles))

    tile_ids = np.arange(n_total_tiles)
    masks = []
    for _ in range(n_draws):
        chosen = rng.choice(tile_ids, size=n_tiles, replace=False)
        mask = np.zeros((H, W), dtype=bool)
        for t in chosen:
            r, c = divmod(int(t), n_tile_cols)
            mask[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = True
        masks.append(mask)
    return masks


def compute_overlap_metrics(mask: np.ndarray, geom: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute the 9 raw, non-aggregated overlap/association metrics for a region mask."""
    area = int(mask.sum())
    if area == 0:
        return {k: 0.0 for k in [
            "geometry_fraction", "narrow_gap_fraction", "thin_line_fraction",
            "boundary_proximity_mean", "junction_recall", "junction_density_per_kpx",
            "corner_recall", "corner_density_per_kpx", "local_density_mean",
        ]}

    geometry_fraction = float(np.logical_and(mask, geom["geometry_mask"]).sum() / area)
    narrow_gap_fraction = float(np.logical_and(mask, geom["narrow_gap_mask"]).sum() / area)
    thin_line_fraction = float(np.logical_and(mask, geom["thin_line_mask"]).sum() / area)
    boundary_proximity_mean = float(geom["boundary_distance_map"][mask].mean())
    local_density_mean = float(geom["density_map"][mask].mean())

    total_junction = int(geom["junction_mask"].sum())
    in_mask_junction = int(np.logical_and(mask, geom["junction_mask"]).sum())
    junction_recall = float(in_mask_junction / total_junction) if total_junction > 0 else 0.0
    junction_density_per_kpx = float(in_mask_junction / area * 1000.0)

    total_corner = int(geom["corner_mask"].sum())
    in_mask_corner = int(np.logical_and(mask, geom["corner_mask"]).sum())
    corner_recall = float(in_mask_corner / total_corner) if total_corner > 0 else 0.0
    corner_density_per_kpx = float(in_mask_corner / area * 1000.0)

    return {
        "geometry_fraction": round(geometry_fraction, 4),
        "narrow_gap_fraction": round(narrow_gap_fraction, 4),
        "thin_line_fraction": round(thin_line_fraction, 4),
        "boundary_proximity_mean": round(boundary_proximity_mean, 3),
        "junction_recall": round(junction_recall, 4),
        "junction_density_per_kpx": round(junction_density_per_kpx, 3),
        "corner_recall": round(corner_recall, 4),
        "corner_density_per_kpx": round(corner_density_per_kpx, 3),
        "local_density_mean": round(local_density_mean, 4),
    }


METRIC_NAMES: Tuple[str, ...] = (
    "geometry_fraction", "narrow_gap_fraction", "thin_line_fraction",
    "boundary_proximity_mean", "junction_recall", "junction_density_per_kpx",
    "corner_recall", "corner_density_per_kpx", "local_density_mean",
)
