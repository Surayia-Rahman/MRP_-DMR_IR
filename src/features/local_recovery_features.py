"""
Local 3x3 recovery feature extraction.

This version intentionally matches the earlier GPU notebook behavior:

1. 3x3 edges are computed using linspace-style integer truncation.
   For a 480x640 frame:
      rows: 0:160, 160:320, 320:480
      cols: 0:213, 213:426, 426:640

2. Local summary standard deviation uses sample standard deviation, ddof=1,
   matching torch.std default behavior in the earlier extraction.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


EPS = 1e-8


def compute_linear_slope(values: np.ndarray) -> float:
    """
    Compute least-squares linear slope for a 1D sequence.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return float("nan")

    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, deg=1)[0]

    return float(slope)


def split_surface_regions(height: int, width: int, grid_shape=(3, 3)):
    """
    Split a 2D surface into 3x3 regions using old torch-style edge logic.

    Old notebook logic:
        row_edges = torch.linspace(0, h, 4).long()
        col_edges = torch.linspace(0, w, 4).long()

    NumPy equivalent:
        np.linspace(0, h, 4).astype(int)
        np.linspace(0, w, 4).astype(int)

    Returns
    -------
    list
        Tuples of (region_index, region_name, row_start, row_end, col_start, col_end).
    """
    n_rows, n_cols = grid_shape

    row_edges = np.linspace(0, height, n_rows + 1).astype(int)
    col_edges = np.linspace(0, width, n_cols + 1).astype(int)

    regions = []
    region_index = 0

    for i in range(n_rows):
        for j in range(n_cols):
            region_name = f"r{region_index:02d}"

            regions.append((
                region_index,
                region_name,
                int(row_edges[i]),
                int(row_edges[i + 1]),
                int(col_edges[j]),
                int(col_edges[j + 1]),
            ))

            region_index += 1

    return regions


def summarize_local_values(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Summarize 9 regional values using the final local summary statistics.

    Summary stats:
    mean, mean_abs, max, min, max_abs, std, range,
    concentration_ratio, top3_abs_share

    std uses ddof=1 to match torch.std behavior from the previous GPU pipeline.
    """
    values = np.asarray(values, dtype=float)
    abs_values = np.abs(values)

    mean_abs = np.nanmean(abs_values)
    max_abs = np.nanmax(abs_values)
    total_abs = np.nansum(abs_values)

    sorted_abs = np.sort(abs_values)[::-1]
    top3_abs_sum = np.nansum(sorted_abs[:3])

    return {
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_mean_abs": float(mean_abs),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max_abs": float(max_abs),
        f"{prefix}_std": float(np.nanstd(values, ddof=1)),
        f"{prefix}_range": float(np.nanmax(values) - np.nanmin(values)),
        f"{prefix}_concentration_ratio": float(max_abs / (mean_abs + EPS)),
        f"{prefix}_top3_abs_share": float(top3_abs_sum / (total_abs + EPS)),
    }


def compute_local_recovery_features(volume: np.ndarray) -> Dict[str, float]:
    """
    Extract local region and local summary features.

    Local region features:
    - local_recovery_r00 ... local_recovery_r08
    - local_slope_r00 ... local_slope_r08
    - local_peak_rate_r00 ... local_peak_rate_r08

    Local summary features:
    - 9 summaries for recovery
    - 9 summaries for slope
    - 9 summaries for peak rate

    Total:
    27 local region features + 27 local summary features = 54
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape (20, H, W), got {volume.shape}")

    if volume.shape[0] != 20:
        raise ValueError(f"Expected 20 frames, got {volume.shape[0]}")

    height, width = volume.shape[1], volume.shape[2]
    regions = split_surface_regions(height, width, grid_shape=(3, 3))

    features = {}

    local_recovery_values = []
    local_slope_values = []
    local_peak_rate_values = []

    for region_index, region_name, row_start, row_end, col_start, col_end in regions:
        region_volume = volume[:, row_start:row_end, col_start:col_end]
        regional_curve = np.nanmean(region_volume, axis=(1, 2))

        recovery = float(regional_curve[-1] - regional_curve[0])
        slope = compute_linear_slope(regional_curve)
        peak_rate = float(np.nanmax(np.diff(regional_curve)))

        features[f"local_recovery_{region_name}"] = recovery
        features[f"local_slope_{region_name}"] = slope
        features[f"local_peak_rate_{region_name}"] = peak_rate

        local_recovery_values.append(recovery)
        local_slope_values.append(slope)
        local_peak_rate_values.append(peak_rate)

    features.update(
        summarize_local_values(
            np.asarray(local_recovery_values),
            prefix="local_recovery"
        )
    )

    features.update(
        summarize_local_values(
            np.asarray(local_slope_values),
            prefix="local_slope"
        )
    )

    features.update(
        summarize_local_values(
            np.asarray(local_peak_rate_values),
            prefix="local_peak_rate"
        )
    )

    return features