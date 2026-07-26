"""
EDA and feature-behavior plots for the DMR-IR dynamic thermography cohort.

This module streams patient volumes one at a time, so it avoids storing the
entire image dataset in memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.load_data import load_patient_volume_from_files
from src.visualization.plot_style import (
    set_report_style,
    save_figure,
    CLASS_COLORS,
    CLASS_LABELS,
)


def _mean_ci(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and approximate 95% confidence interval across patients.

    values shape:
        patients x time
    """
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    n = np.sum(~np.isnan(values), axis=0)
    sem = std / np.sqrt(np.maximum(n, 1))
    ci = 1.96 * sem
    return mean, mean - ci, mean + ci


def _split_regions(height: int, width: int, grid_shape=(3, 3)):
    """
    Split frame into 3x3 regions using old torch-style edge logic.

    For a 480x640 frame:
        rows: 0:160, 160:320, 320:480
        cols: 0:213, 213:426, 426:640
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
                region_name,
                int(row_edges[i]),
                int(row_edges[i + 1]),
                int(col_edges[j]),
                int(col_edges[j + 1]),
            ))
            region_index += 1

    return regions


def compute_eda_tables(
    index_df: pd.DataFrame,
    output_tables_dir: str | Path,
    max_patients: int | None = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute patient-level EDA summaries and regional curve tables.

    Parameters
    ----------
    index_df:
        Final modeling cohort index.
    output_tables_dir:
        Directory to save output CSVs.
    max_patients:
        Optional debug limit. Use None for full cohort.

    Returns
    -------
    dict
        DataFrames for patient_summary, regional_curves, regional_features.
    """
    output_tables_dir = Path(output_tables_dir)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    if max_patients is not None:
        work_df = index_df.head(max_patients).copy()
    else:
        work_df = index_df.copy()

    patient_rows = []
    regional_curve_rows = []
    regional_feature_rows = []

    frame_idx = np.arange(20)

    for row_i, row in work_df.iterrows():
        patient_id = str(row["patient_id"])
        class_name = str(row["class_name"])
        label = int(row["label"])
        selected_date = row.get("selected_date", None)
        file_paths = str(row["file_paths"]).split("|")

        print(f"[{row_i + 1}/{len(work_df)}] Loading {patient_id} ({class_name})")

        volume = load_patient_volume_from_files(file_paths)

        if volume.shape[0] != 20:
            raise ValueError(f"{patient_id}: expected 20 frames, got {volume.shape}")

        frame_mean = np.nanmean(volume, axis=(1, 2))
        frame_std = np.nanstd(volume, axis=(1, 2))

        # Delta padded with 0 at frame 0.
        mean_delta_padded = np.concatenate([[0.0], np.diff(frame_mean)])
        std_delta_padded = np.concatenate([[0.0], np.diff(frame_std)])
        cumulative_recovery = frame_mean - frame_mean[0]

        patient_record = {
            "patient_id": patient_id,
            "class_name": class_name,
            "label": label,
            "selected_date": selected_date,
            "initial_mean_temp": float(frame_mean[0]),
            "final_mean_temp": float(frame_mean[-1]),
            "recovery_delta": float(frame_mean[-1] - frame_mean[0]),
            "initial_frame_std": float(frame_std[0]),
            "final_frame_std": float(frame_std[-1]),
            "dynamic_std_delta": float(frame_std[-1] - frame_std[0]),
            "mean_temp_over_time": float(np.nanmean(frame_mean)),
            "mean_spatial_std_over_time": float(np.nanmean(frame_std)),
            "max_frame_to_frame_mean_delta": float(np.nanmax(np.diff(frame_mean))),
            "min_frame_to_frame_mean_delta": float(np.nanmin(np.diff(frame_mean))),
            "mean_abs_frame_to_frame_mean_delta": float(np.nanmean(np.abs(np.diff(frame_mean)))),
        }

        for t in range(20):
            patient_record[f"frame_mean_{t:02d}"] = float(frame_mean[t])
            patient_record[f"frame_std_{t:02d}"] = float(frame_std[t])
            patient_record[f"mean_delta_padded_{t:02d}"] = float(mean_delta_padded[t])
            patient_record[f"std_delta_padded_{t:02d}"] = float(std_delta_padded[t])
            patient_record[f"cumulative_recovery_{t:02d}"] = float(cumulative_recovery[t])

        patient_rows.append(patient_record)

        height, width = volume.shape[1], volume.shape[2]
        regions = _split_regions(height, width, grid_shape=(3, 3))

        for region_name, row_start, row_end, col_start, col_end in regions:
            region_volume = volume[:, row_start:row_end, col_start:col_end]
            regional_curve = np.nanmean(region_volume, axis=(1, 2))

            regional_recovery = regional_curve[-1] - regional_curve[0]
            regional_slope = np.polyfit(frame_idx, regional_curve, deg=1)[0]
            regional_peak_rate = np.nanmax(np.diff(regional_curve))

            regional_feature_rows.append({
                "patient_id": patient_id,
                "class_name": class_name,
                "label": label,
                "region": region_name,
                "local_recovery": float(regional_recovery),
                "local_slope": float(regional_slope),
                "local_peak_rate": float(regional_peak_rate),
            })

            for t in range(20):
                regional_curve_rows.append({
                    "patient_id": patient_id,
                    "class_name": class_name,
                    "label": label,
                    "region": region_name,
                    "frame": t,
                    "regional_mean_temp": float(regional_curve[t]),
                })

    patient_summary_df = pd.DataFrame(patient_rows)
    regional_curves_df = pd.DataFrame(regional_curve_rows)
    regional_features_df = pd.DataFrame(regional_feature_rows)

    patient_summary_path = output_tables_dir / "eda_patient_dynamic_summary.csv"
    regional_curves_path = output_tables_dir / "eda_regional_curves_long.csv"
    regional_features_path = output_tables_dir / "eda_regional_features_long.csv"

    patient_summary_df.to_csv(patient_summary_path, index=False)
    regional_curves_df.to_csv(regional_curves_path, index=False)
    regional_features_df.to_csv(regional_features_path, index=False)

    print("Saved:", patient_summary_path)
    print("Saved:", regional_curves_path)
    print("Saved:", regional_features_path)

    return {
        "patient_summary": patient_summary_df,
        "regional_curves": regional_curves_df,
        "regional_features": regional_features_df,
    }


def plot_class_distribution(index_df: pd.DataFrame, output_dir: str | Path):
    set_report_style()
    output_dir = Path(output_dir)

    counts = index_df["class_name"].value_counts().reindex(["healthy", "sick"])

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [CLASS_LABELS[c] for c in counts.index],
        counts.values,
        color=[CLASS_COLORS[c] for c in counts.index],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Final Modeling Cohort Class Distribution")
    ax.set_ylabel("Number of patients")
    ax.set_ylim(0, max(counts.values) * 1.20)

    return save_figure(fig, output_dir / "class_distribution_final_cohort.png")


def plot_curve_by_class(
    patient_df: pd.DataFrame,
    value_prefix: str,
    title: str,
    ylabel: str,
    output_path: str | Path
):
    set_report_style()

    frames = np.arange(20)

    fig, ax = plt.subplots(figsize=(8, 5))

    for class_name in ["healthy", "sick"]:
        class_df = patient_df[patient_df["class_name"] == class_name]
        cols = [f"{value_prefix}_{t:02d}" for t in range(20)]
        values = class_df[cols].to_numpy(dtype=float)

        mean, lower, upper = _mean_ci(values)

        ax.plot(
            frames,
            mean,
            marker="o",
            linewidth=2,
            label=f"{CLASS_LABELS[class_name]} mean",
            color=CLASS_COLORS[class_name],
        )
        ax.fill_between(
            frames,
            lower,
            upper,
            alpha=0.18,
            color=CLASS_COLORS[class_name],
            label=f"{CLASS_LABELS[class_name]} 95% CI",
        )

    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(frames)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)

    return save_figure(fig, output_path)


def plot_difference_curve(
    patient_df: pd.DataFrame,
    value_prefix: str,
    title: str,
    ylabel: str,
    output_path: str | Path
):
    set_report_style()

    frames = np.arange(20)
    cols = [f"{value_prefix}_{t:02d}" for t in range(20)]

    healthy_values = patient_df[patient_df["class_name"] == "healthy"][cols].to_numpy(dtype=float)
    sick_values = patient_df[patient_df["class_name"] == "sick"][cols].to_numpy(dtype=float)

    healthy_mean = np.nanmean(healthy_values, axis=0)
    sick_mean = np.nanmean(sick_values, axis=0)

    diff = sick_mean - healthy_mean

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.plot(frames, diff, marker="o", linewidth=2, color="#4B0082")

    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(frames)
    ax.grid(True, alpha=0.25)

    return save_figure(fig, output_path)


def plot_delta_curve_by_class(patient_df: pd.DataFrame, output_path: str | Path):
    set_report_style()

    frames = np.arange(20)

    fig, ax = plt.subplots(figsize=(8, 5))

    for class_name in ["healthy", "sick"]:
        class_df = patient_df[patient_df["class_name"] == class_name]
        cols = [f"mean_delta_padded_{t:02d}" for t in range(20)]
        values = class_df[cols].to_numpy(dtype=float)

        mean, lower, upper = _mean_ci(values)

        ax.plot(
            frames,
            mean,
            marker="o",
            linewidth=2,
            label=CLASS_LABELS[class_name],
            color=CLASS_COLORS[class_name],
        )
        ax.fill_between(frames, lower, upper, alpha=0.18, color=CLASS_COLORS[class_name])

    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.set_title("Frame-to-Frame Mean Temperature Change by Class")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean temperature change from previous frame (°C)")
    ax.set_xticks(frames)
    ax.legend(frameon=True)

    return save_figure(fig, output_path)


def plot_regional_curves_3x3(regional_curves_df: pd.DataFrame, output_path: str | Path):
    set_report_style()

    regions = [f"r{i:02d}" for i in range(9)]
    frames = np.arange(20)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

    for ax, region in zip(axes.ravel(), regions):
        region_df = regional_curves_df[regional_curves_df["region"] == region]

        for class_name in ["healthy", "sick"]:
            class_df = region_df[region_df["class_name"] == class_name]

            pivot = class_df.pivot_table(
                index="patient_id",
                columns="frame",
                values="regional_mean_temp",
                aggfunc="mean"
            )

            values = pivot.reindex(columns=frames).to_numpy(dtype=float)
            mean, lower, upper = _mean_ci(values)

            ax.plot(
                frames,
                mean,
                linewidth=2,
                color=CLASS_COLORS[class_name],
                label=CLASS_LABELS[class_name],
            )
            ax.fill_between(
                frames,
                lower,
                upper,
                alpha=0.14,
                color=CLASS_COLORS[class_name],
            )

        ax.set_title(region)
        ax.set_xticks([0, 5, 10, 15, 19])

    fig.suptitle("Regional Mean Temperature Curves Across 3x3 Surface Grid", fontsize=15, y=1.02)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))

    fig.text(0.5, 0.02, "Frame index", ha="center")
    fig.text(0.02, 0.5, "Regional mean temperature (°C)", va="center", rotation="vertical")

    fig.tight_layout(rect=[0.04, 0.04, 1, 0.94])

    return save_figure(fig, output_path)


def _plot_3x3_difference_heatmap(
    regional_features_df: pd.DataFrame,
    feature_col: str,
    title: str,
    colorbar_label: str,
    output_path: str | Path
):
    set_report_style()

    regions = [f"r{i:02d}" for i in range(9)]

    diff_values = []
    for region in regions:
        region_df = regional_features_df[regional_features_df["region"] == region]

        healthy_mean = region_df[region_df["class_name"] == "healthy"][feature_col].mean()
        sick_mean = region_df[region_df["class_name"] == "sick"][feature_col].mean()

        diff_values.append(sick_mean - healthy_mean)

    matrix = np.array(diff_values).reshape(3, 3)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(matrix, cmap="coolwarm")

    ax.set_title(title)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Left", "Center", "Right"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Top", "Middle", "Bottom"])

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    ax.grid(False)

    return save_figure(fig, output_path)


def plot_all_eda_figures(
    index_df: pd.DataFrame,
    patient_summary_df: pd.DataFrame,
    regional_curves_df: pd.DataFrame,
    regional_features_df: pd.DataFrame,
    figures_eda_dir: str | Path,
    figures_feature_dir: str | Path,
):
    """
    Generate all EDA and feature-behavior figures.
    """
    figures_eda_dir = Path(figures_eda_dir)
    figures_feature_dir = Path(figures_feature_dir)

    figures_eda_dir.mkdir(parents=True, exist_ok=True)
    figures_feature_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    saved.append(plot_class_distribution(index_df, figures_eda_dir))

    saved.append(plot_curve_by_class(
        patient_summary_df,
        value_prefix="frame_mean",
        title="Average Frame-wise Surface Temperature by Class",
        ylabel="Mean surface temperature (°C)",
        output_path=figures_feature_dir / "average_frame_mean_curve_by_class.png",
    ))

    saved.append(plot_curve_by_class(
        patient_summary_df,
        value_prefix="frame_std",
        title="Average Frame-wise Spatial Heterogeneity by Class",
        ylabel="Spatial standard deviation of temperature (°C)",
        output_path=figures_feature_dir / "average_frame_std_curve_by_class.png",
    ))

    saved.append(plot_difference_curve(
        patient_summary_df,
        value_prefix="frame_mean",
        title="Sick Minus Healthy Difference in Frame-wise Mean Temperature",
        ylabel="Difference in mean temperature (°C)",
        output_path=figures_feature_dir / "sick_minus_healthy_frame_mean_difference.png",
    ))

    saved.append(plot_difference_curve(
        patient_summary_df,
        value_prefix="frame_std",
        title="Sick Minus Healthy Difference in Spatial Heterogeneity",
        ylabel="Difference in spatial standard deviation (°C)",
        output_path=figures_feature_dir / "sick_minus_healthy_frame_std_difference.png",
    ))

    saved.append(plot_delta_curve_by_class(
        patient_summary_df,
        output_path=figures_feature_dir / "frame_to_frame_delta_curve_by_class.png",
    ))

    saved.append(plot_curve_by_class(
        patient_summary_df,
        value_prefix="cumulative_recovery",
        title="Cumulative Surface Recovery from Frame 0 by Class",
        ylabel="Cumulative recovery from frame 0 (°C)",
        output_path=figures_feature_dir / "cumulative_recovery_curve_by_class.png",
    ))

    saved.append(plot_regional_curves_3x3(
        regional_curves_df,
        output_path=figures_feature_dir / "regional_mean_curves_3x3_by_class.png",
    ))

    saved.append(_plot_3x3_difference_heatmap(
        regional_features_df,
        feature_col="local_recovery",
        title="Local Recovery Difference: Sick Minus Healthy",
        colorbar_label="Difference in local recovery (°C)",
        output_path=figures_feature_dir / "local_recovery_sick_minus_healthy_heatmap.png",
    ))

    saved.append(_plot_3x3_difference_heatmap(
        regional_features_df,
        feature_col="local_slope",
        title="Local Slope Difference: Sick Minus Healthy",
        colorbar_label="Difference in local slope (°C/frame)",
        output_path=figures_feature_dir / "local_slope_sick_minus_healthy_heatmap.png",
    ))

    saved.append(_plot_3x3_difference_heatmap(
        regional_features_df,
        feature_col="local_peak_rate",
        title="Local Peak-rate Difference: Sick Minus Healthy",
        colorbar_label="Difference in local peak rate (°C/frame)",
        output_path=figures_feature_dir / "local_peak_rate_sick_minus_healthy_heatmap.png",
    ))

    print("\nSaved figures:")
    for path in saved:
        print(path)

    return saved


def run_eda(
    index_path: str | Path,
    output_tables_dir: str | Path,
    figures_eda_dir: str | Path,
    figures_feature_dir: str | Path,
    max_patients: int | None = None,
):
    """
    Run full EDA stage.
    """
    index_path = Path(index_path)

    if not index_path.exists():
        raise FileNotFoundError(f"Dataset index not found: {index_path}")

    index_df = pd.read_csv(index_path)

    print("Loaded final modeling cohort index:", index_path)
    print("Index shape:", index_df.shape)
    print(index_df["class_name"].value_counts())

    tables = compute_eda_tables(
        index_df=index_df,
        output_tables_dir=output_tables_dir,
        max_patients=max_patients,
    )

    saved_figures = plot_all_eda_figures(
        index_df=index_df if max_patients is None else index_df.head(max_patients),
        patient_summary_df=tables["patient_summary"],
        regional_curves_df=tables["regional_curves"],
        regional_features_df=tables["regional_features"],
        figures_eda_dir=figures_eda_dir,
        figures_feature_dir=figures_feature_dir,
    )

    return {
        "index_shape": tuple(index_df.shape),
        "class_counts": index_df["class_name"].value_counts().to_dict(),
        "saved_figures": [str(p) for p in saved_figures],
    }