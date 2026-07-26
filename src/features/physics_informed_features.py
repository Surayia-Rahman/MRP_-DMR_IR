"""
Physics-informed recovery feature extraction.

These are surface-derived recovery descriptors motivated by bioheat-transfer
reasoning. They do not directly estimate internal temperature, tumor depth,
or metabolic heat generation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.features.local_recovery_features import (
    compute_linear_slope,
    compute_local_recovery_features,
)


def compute_dynamic_recovery_features(volume: np.ndarray) -> Dict[str, float]:
    """
    Compute global dynamic recovery features from the frame-wise mean curve.

    Includes:
    - 19 frame-to-frame deltas
    - recovery delta
    - global slope
    - early/middle/late slopes
    - recovery AUC
    - normalized recovery AUC
    - temporal variance summaries
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape (20, H, W), got {volume.shape}")

    if volume.shape[0] != 20:
        raise ValueError(f"Expected 20 frames, got {volume.shape[0]}")

    features = {}

    frame_means = np.nanmean(volume, axis=(1, 2))
    frame_deltas = np.diff(frame_means)

    for i, value in enumerate(frame_deltas):
        features[f"delta_{i:02d}_{i+1:02d}"] = float(value)

    features["recovery_delta"] = float(frame_means[-1] - frame_means[0])
    features["overall_recovery_slope"] = compute_linear_slope(frame_means)

    features["mean_frame_to_frame_change"] = float(np.nanmean(frame_deltas))
    features["max_frame_to_frame_change"] = float(np.nanmax(frame_deltas))
    features["min_frame_to_frame_change"] = float(np.nanmin(frame_deltas))
    features["std_frame_to_frame_change"] = float(np.nanstd(frame_deltas))
    features["mean_abs_frame_to_frame_change"] = float(np.nanmean(np.abs(frame_deltas)))

    # Same segmentation as the previous methodology:
    # early: frames 0-5, middle: frames 6-13, late: frames 14-19
    features["early_recovery_slope"] = compute_linear_slope(frame_means[0:6])
    features["middle_recovery_slope"] = compute_linear_slope(frame_means[6:14])
    features["late_recovery_slope"] = compute_linear_slope(frame_means[14:20])

    # Use trapezoidal integration over frame index.
    if hasattr(np, "trapezoid"):
        trapz_fn = np.trapezoid
    else:
        trapz_fn = np.trapz

    features["recovery_auc"] = float(trapz_fn(frame_means, dx=1))
    features["normalized_recovery_auc"] = float(trapz_fn(frame_means - frame_means[0], dx=1))

    temporal_variance_map = np.nanvar(volume, axis=0)

    features["mean_temporal_variance"] = float(np.nanmean(temporal_variance_map))
    features["max_temporal_variance"] = float(np.nanmax(temporal_variance_map))
    features["std_temporal_variance"] = float(np.nanstd(temporal_variance_map))

    return features


def compute_dynamic_std_change_features(volume: np.ndarray) -> Dict[str, float]:
    """
    Compute change/rate features for spatial heterogeneity over time.
    """
    frame_stds = np.nanstd(volume, axis=(1, 2))

    features = {
        "std_dynamic_frame_std": float(np.nanstd(frame_stds)),
        "dynamic_frame_std_delta": float(frame_stds[-1] - frame_stds[0]),
        "dynamic_frame_std_slope": compute_linear_slope(frame_stds),
    }

    return features


def compute_physics_informed_features(volume: np.ndarray) -> Dict[str, float]:
    """
    Extract the 91 physics-informed recovery features.

    Components:
    1. Frame-to-frame deltas: 19
    2. Global dynamic recovery summaries: 15
    3. Dynamic spatial-heterogeneity change features: 3
    4. Local 3x3 region features: 27
    5. Local summary features: 27

    Total: 91
    """
    features = {}

    features.update(compute_dynamic_recovery_features(volume))
    features.update(compute_dynamic_std_change_features(volume))
    features.update(compute_local_recovery_features(volume))

    return features