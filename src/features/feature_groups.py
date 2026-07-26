"""
Feature group definitions for the MRP_DMRIR project.

These lists define the final 144-feature design:

1. Statistical surface features: 53
2. Physics-informed recovery features: 91
3. Combined features: 144
"""

from __future__ import annotations

from typing import Dict, List


METADATA_COLS = [
    "patient_id",
    "class_name",
    "label",
    "selected_date",
]


# -----------------------------
# Statistical surface features
# -----------------------------

STATIC_INITIAL_FEATURES = [
    "initial_mean_temp",
    "initial_max_temp",
    "initial_min_temp",
    "initial_std_temp",
    "initial_spatial_variance",
]

STATIC_FINAL_FEATURES = [
    "final_mean_temp",
    "final_max_temp",
    "final_min_temp",
    "final_std_temp",
    "final_spatial_variance",
]

STATIC_ALL_FEATURES = STATIC_INITIAL_FEATURES + STATIC_FINAL_FEATURES

FRAME_MEAN_FEATURES = [
    f"frame_mean_{i:02d}"
    for i in range(20)
]

FRAME_STD_FEATURES = [
    f"frame_std_{i:02d}"
    for i in range(20)
]

DYNAMIC_STD_LEVEL_SUMMARY_FEATURES = [
    "mean_dynamic_frame_std",
    "max_dynamic_frame_std",
    "min_dynamic_frame_std",
]

STATISTICAL_FEATURES = (
    STATIC_ALL_FEATURES
    + FRAME_MEAN_FEATURES
    + FRAME_STD_FEATURES
    + DYNAMIC_STD_LEVEL_SUMMARY_FEATURES
)


# -----------------------------
# Physics-informed recovery features
# -----------------------------

DELTA_FEATURES = [
    f"delta_{i:02d}_{i+1:02d}"
    for i in range(19)
]

DYNAMIC_SUMMARY_FEATURES = [
    "recovery_delta",
    "overall_recovery_slope",
    "mean_frame_to_frame_change",
    "max_frame_to_frame_change",
    "min_frame_to_frame_change",
    "std_frame_to_frame_change",
    "mean_abs_frame_to_frame_change",
    "early_recovery_slope",
    "middle_recovery_slope",
    "late_recovery_slope",
    "recovery_auc",
    "normalized_recovery_auc",
    "mean_temporal_variance",
    "max_temporal_variance",
    "std_temporal_variance",
]

DYNAMIC_STD_CHANGE_FEATURES = [
    "std_dynamic_frame_std",
    "dynamic_frame_std_delta",
    "dynamic_frame_std_slope",
]

LOCAL_RECOVERY_REGION_FEATURES = [
    f"local_recovery_r{i:02d}"
    for i in range(9)
]

LOCAL_SLOPE_REGION_FEATURES = [
    f"local_slope_r{i:02d}"
    for i in range(9)
]

LOCAL_PEAK_RATE_REGION_FEATURES = [
    f"local_peak_rate_r{i:02d}"
    for i in range(9)
]

LOCAL_REGION_FEATURES = (
    LOCAL_RECOVERY_REGION_FEATURES
    + LOCAL_SLOPE_REGION_FEATURES
    + LOCAL_PEAK_RATE_REGION_FEATURES
)

LOCAL_SUMMARY_STATS = [
    "mean",
    "mean_abs",
    "max",
    "min",
    "max_abs",
    "std",
    "range",
    "concentration_ratio",
    "top3_abs_share",
]

LOCAL_RECOVERY_SUMMARY_FEATURES = [
    f"local_recovery_{stat}"
    for stat in LOCAL_SUMMARY_STATS
]

LOCAL_SLOPE_SUMMARY_FEATURES = [
    f"local_slope_{stat}"
    for stat in LOCAL_SUMMARY_STATS
]

LOCAL_PEAK_RATE_SUMMARY_FEATURES = [
    f"local_peak_rate_{stat}"
    for stat in LOCAL_SUMMARY_STATS
]

LOCAL_SUMMARY_FEATURES = (
    LOCAL_RECOVERY_SUMMARY_FEATURES
    + LOCAL_SLOPE_SUMMARY_FEATURES
    + LOCAL_PEAK_RATE_SUMMARY_FEATURES
)

PHYSICS_INFORMED_FEATURES = (
    DELTA_FEATURES
    + DYNAMIC_SUMMARY_FEATURES
    + DYNAMIC_STD_CHANGE_FEATURES
    + LOCAL_REGION_FEATURES
    + LOCAL_SUMMARY_FEATURES
)

COMBINED_FEATURES = STATISTICAL_FEATURES + PHYSICS_INFORMED_FEATURES


def get_feature_groups() -> Dict[str, List[str]]:
    """
    Return all named feature groups.
    """
    return {
        "statistical_surface": STATISTICAL_FEATURES,
        "physics_informed_recovery": PHYSICS_INFORMED_FEATURES,
        "combined_statistical_physics": COMBINED_FEATURES,
    }


def summarize_feature_groups() -> Dict[str, int]:
    """
    Return counts for main feature groups.
    """
    overlap = sorted(set(STATISTICAL_FEATURES).intersection(PHYSICS_INFORMED_FEATURES))

    return {
        "statistical_surface": len(STATISTICAL_FEATURES),
        "physics_informed_recovery": len(PHYSICS_INFORMED_FEATURES),
        "combined_statistical_physics": len(COMBINED_FEATURES),
        "overlap_count": len(overlap),
    }


def validate_feature_table_columns(columns: List[str]) -> Dict:
    """
    Validate whether a dataframe contains all expected final features.
    """
    columns_set = set(columns)

    missing_metadata = [
        col for col in METADATA_COLS
        if col not in columns_set
    ]

    missing_statistical = [
        col for col in STATISTICAL_FEATURES
        if col not in columns_set
    ]

    missing_physics = [
        col for col in PHYSICS_INFORMED_FEATURES
        if col not in columns_set
    ]

    missing_combined = [
        col for col in COMBINED_FEATURES
        if col not in columns_set
    ]

    non_metadata_cols = [
        col for col in columns
        if col not in METADATA_COLS
    ]

    unassigned_features = sorted(
        list(set(non_metadata_cols) - set(COMBINED_FEATURES))
    )

    overlap = sorted(
        list(set(STATISTICAL_FEATURES).intersection(PHYSICS_INFORMED_FEATURES))
    )

    return {
        "missing_metadata": missing_metadata,
        "missing_statistical": missing_statistical,
        "missing_physics": missing_physics,
        "missing_combined": missing_combined,
        "unassigned_features": unassigned_features,
        "overlap": overlap,
        "n_metadata_expected": len(METADATA_COLS),
        "n_statistical_expected": len(STATISTICAL_FEATURES),
        "n_physics_expected": len(PHYSICS_INFORMED_FEATURES),
        "n_combined_expected": len(COMBINED_FEATURES),
        "n_non_metadata_observed": len(non_metadata_cols),
    }