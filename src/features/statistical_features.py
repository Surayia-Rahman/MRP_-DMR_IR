"""
Statistical surface feature extraction.

These features describe what the observed surface temperature distribution
looks like over time.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_frame_mean_curve(volume: np.ndarray) -> np.ndarray:
    """
    Compute frame-wise mean surface temperature.

    Parameters
    ----------
    volume:
        Thermal volume with shape (20, H, W).

    Returns
    -------
    np.ndarray
        Shape (20,).
    """
    return np.nanmean(volume, axis=(1, 2))


def compute_frame_std_curve(volume: np.ndarray) -> np.ndarray:
    """
    Compute frame-wise spatial standard deviation.

    Parameters
    ----------
    volume:
        Thermal volume with shape (20, H, W).

    Returns
    -------
    np.ndarray
        Shape (20,).
    """
    return np.nanstd(volume, axis=(1, 2))


def _frame_stats(frame: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute basic spatial statistics for one frame.
    """
    return {
        f"{prefix}_mean_temp": float(np.nanmean(frame)),
        f"{prefix}_max_temp": float(np.nanmax(frame)),
        f"{prefix}_min_temp": float(np.nanmin(frame)),
        f"{prefix}_std_temp": float(np.nanstd(frame)),
        f"{prefix}_spatial_variance": float(np.nanvar(frame)),
    }


def compute_statistical_surface_features(volume: np.ndarray) -> Dict[str, float]:
    """
    Extract the 53 statistical surface features.

    Feature components:
    1. Initial frame statistics: 5
    2. Final frame statistics: 5
    3. Frame-wise mean curve: 20
    4. Frame-wise spatial standard deviation curve: 20
    5. Dynamic standard deviation level summaries: 3

    Total: 53
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape (20, H, W), got {volume.shape}")

    if volume.shape[0] != 20:
        raise ValueError(f"Expected 20 frames, got {volume.shape[0]}")

    features = {}

    initial_frame = volume[0]
    final_frame = volume[-1]

    features.update(_frame_stats(initial_frame, prefix="initial"))
    features.update(_frame_stats(final_frame, prefix="final"))

    frame_means = compute_frame_mean_curve(volume)
    frame_stds = compute_frame_std_curve(volume)

    for i, value in enumerate(frame_means):
        features[f"frame_mean_{i:02d}"] = float(value)

    for i, value in enumerate(frame_stds):
        features[f"frame_std_{i:02d}"] = float(value)

    features["mean_dynamic_frame_std"] = float(np.nanmean(frame_stds))
    features["max_dynamic_frame_std"] = float(np.nanmax(frame_stds))
    features["min_dynamic_frame_std"] = float(np.nanmin(frame_stds))

    return features