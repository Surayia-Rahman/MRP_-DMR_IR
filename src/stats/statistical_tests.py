"""
Statistical testing utilities for DMR-IR feature analysis.

This module compares healthy vs sick feature distributions using:

1. Mann-Whitney U test
2. Benjamini-Hochberg FDR correction
3. Cohen's d effect size

Sign convention:
    Cohen's d = sick mean - healthy mean, divided by pooled standard deviation.

Therefore:
    positive Cohen's d -> feature is higher in sick patients
    negative Cohen's d -> feature is higher in healthy patients
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from src.features.feature_groups import (
    METADATA_COLS,
    STATISTICAL_FEATURES,
    PHYSICS_INFORMED_FEATURES,
    STATIC_INITIAL_FEATURES,
    STATIC_FINAL_FEATURES,
    FRAME_MEAN_FEATURES,
    FRAME_STD_FEATURES,
    DYNAMIC_STD_LEVEL_SUMMARY_FEATURES,
    DELTA_FEATURES,
    DYNAMIC_SUMMARY_FEATURES,
    DYNAMIC_STD_CHANGE_FEATURES,
    LOCAL_REGION_FEATURES,
    LOCAL_SUMMARY_FEATURES,
)

from src.visualization.plot_style import (
    set_report_style,
    save_figure,
)


def cohen_d_sick_minus_healthy(
    healthy_values: np.ndarray,
    sick_values: np.ndarray
) -> float:
    """
    Compute Cohen's d using sick minus healthy sign convention.
    """
    healthy_values = np.asarray(healthy_values, dtype=float)
    sick_values = np.asarray(sick_values, dtype=float)

    healthy_values = healthy_values[~np.isnan(healthy_values)]
    sick_values = sick_values[~np.isnan(sick_values)]

    n_h = len(healthy_values)
    n_s = len(sick_values)

    if n_h < 2 or n_s < 2:
        return np.nan

    mean_h = np.mean(healthy_values)
    mean_s = np.mean(sick_values)

    var_h = np.var(healthy_values, ddof=1)
    var_s = np.var(sick_values, ddof=1)

    pooled_var = ((n_h - 1) * var_h + (n_s - 1) * var_s) / (n_h + n_s - 2)

    if pooled_var <= 0:
        return np.nan

    pooled_std = np.sqrt(pooled_var)

    return float((mean_s - mean_h) / pooled_std)


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values:
        Array of raw p-values.

    Returns
    -------
    np.ndarray
        Adjusted q-values in original feature order.
    """
    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)
    q_values = np.full(n, np.nan)

    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]

    if len(valid_p) == 0:
        return q_values

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(valid_p)

    ranked_q = ranked_p * m / np.arange(1, m + 1)

    # Enforce monotonicity from largest rank to smallest rank.
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    corrected_valid = np.empty_like(valid_p)
    corrected_valid[order] = ranked_q

    q_values[valid_mask] = corrected_valid

    return q_values


def assign_feature_subgroup(feature_name: str) -> str:
    """
    Assign each feature to a human-readable subgroup.
    """
    if feature_name in STATIC_INITIAL_FEATURES:
        return "statistical_initial_frame"

    if feature_name in STATIC_FINAL_FEATURES:
        return "statistical_final_frame"

    if feature_name in FRAME_MEAN_FEATURES:
        return "statistical_frame_mean_curve"

    if feature_name in FRAME_STD_FEATURES:
        return "statistical_frame_std_curve"

    if feature_name in DYNAMIC_STD_LEVEL_SUMMARY_FEATURES:
        return "statistical_dynamic_std_level"

    if feature_name in DELTA_FEATURES:
        return "physics_frame_to_frame_delta"

    if feature_name in DYNAMIC_SUMMARY_FEATURES:
        return "physics_global_recovery_summary"

    if feature_name in DYNAMIC_STD_CHANGE_FEATURES:
        return "physics_dynamic_std_change"

    if feature_name in LOCAL_REGION_FEATURES:
        return "physics_local_region_features"

    if feature_name in LOCAL_SUMMARY_FEATURES:
        return "physics_local_summary_features"

    return "unknown"


def assign_main_feature_group(feature_name: str) -> str:
    """
    Assign feature to statistical or physics-informed family.
    """
    if feature_name in STATISTICAL_FEATURES:
        return "statistical_surface"

    if feature_name in PHYSICS_INFORMED_FEATURES:
        return "physics_informed_recovery"

    return "unknown"


def run_feature_statistical_tests(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    healthy_label: int = 0,
    sick_label: int = 1,
) -> pd.DataFrame:
    """
    Run Mann-Whitney U test and Cohen's d for each feature.

    Returns
    -------
    pd.DataFrame
        One row per feature.
    """
    results = []

    for feature in feature_cols:
        healthy_values = pd.to_numeric(
            feature_df.loc[feature_df[label_col] == healthy_label, feature],
            errors="coerce"
        ).to_numpy(dtype=float)

        sick_values = pd.to_numeric(
            feature_df.loc[feature_df[label_col] == sick_label, feature],
            errors="coerce"
        ).to_numpy(dtype=float)

        healthy_values_clean = healthy_values[~np.isnan(healthy_values)]
        sick_values_clean = sick_values[~np.isnan(sick_values)]

        if len(healthy_values_clean) == 0 or len(sick_values_clean) == 0:
            u_stat = np.nan
            p_value = np.nan
        else:
            try:
                test = mannwhitneyu(
                    healthy_values_clean,
                    sick_values_clean,
                    alternative="two-sided"
                )
                u_stat = float(test.statistic)
                p_value = float(test.pvalue)
            except Exception:
                u_stat = np.nan
                p_value = np.nan

        mean_h = float(np.nanmean(healthy_values)) if len(healthy_values_clean) else np.nan
        mean_s = float(np.nanmean(sick_values)) if len(sick_values_clean) else np.nan
        median_h = float(np.nanmedian(healthy_values)) if len(healthy_values_clean) else np.nan
        median_s = float(np.nanmedian(sick_values)) if len(sick_values_clean) else np.nan

        d = cohen_d_sick_minus_healthy(healthy_values, sick_values)

        results.append({
            "feature": feature,
            "main_feature_group": assign_main_feature_group(feature),
            "feature_subgroup": assign_feature_subgroup(feature),
            "healthy_mean": mean_h,
            "sick_mean": mean_s,
            "sick_minus_healthy_mean": mean_s - mean_h,
            "healthy_median": median_h,
            "sick_median": median_s,
            "sick_minus_healthy_median": median_s - median_h,
            "mannwhitney_u": u_stat,
            "p_value": p_value,
            "cohens_d_sick_minus_healthy": d,
            "abs_cohens_d": abs(d) if not np.isnan(d) else np.nan,
            "n_healthy": int(len(healthy_values_clean)),
            "n_sick": int(len(sick_values_clean)),
        })

    results_df = pd.DataFrame(results)

    results_df["q_value_fdr_bh"] = benjamini_hochberg_fdr(
        results_df["p_value"].to_numpy(dtype=float)
    )

    results_df["significant_fdr_005"] = results_df["q_value_fdr_bh"] < 0.05
    results_df["significant_raw_p_005"] = results_df["p_value"] < 0.05

    results_df["effect_direction"] = np.where(
        results_df["cohens_d_sick_minus_healthy"] > 0,
        "higher_in_sick",
        np.where(
            results_df["cohens_d_sick_minus_healthy"] < 0,
            "higher_in_healthy",
            "no_difference"
        )
    )

    results_df = results_df.sort_values(
        ["q_value_fdr_bh", "abs_cohens_d"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return results_df


def summarize_by_feature_group(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize statistical-test results by feature group and subgroup.
    """
    group_rows = []

    group_cols = ["main_feature_group", "feature_subgroup"]

    for group_values, group_df in results_df.groupby(group_cols):
        main_group, subgroup = group_values

        group_rows.append({
            "main_feature_group": main_group,
            "feature_subgroup": subgroup,
            "n_features": int(len(group_df)),
            "n_significant_fdr_005": int(group_df["significant_fdr_005"].sum()),
            "n_significant_raw_p_005": int(group_df["significant_raw_p_005"].sum()),
            "mean_abs_cohens_d": float(group_df["abs_cohens_d"].mean()),
            "median_abs_cohens_d": float(group_df["abs_cohens_d"].median()),
            "max_abs_cohens_d": float(group_df["abs_cohens_d"].max()),
            "median_q_value": float(group_df["q_value_fdr_bh"].median()),
            "best_feature_by_abs_d": group_df.sort_values(
                "abs_cohens_d",
                ascending=False
            ).iloc[0]["feature"],
            "best_abs_cohens_d": float(group_df["abs_cohens_d"].max()),
        })

    summary_df = pd.DataFrame(group_rows)

    summary_df = summary_df.sort_values(
        ["main_feature_group", "median_abs_cohens_d"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return summary_df


def plot_top_effect_sizes(
    results_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 25,
    title: str = "Top Features by Absolute Cohen's d"
):
    """
    Horizontal bar plot for top absolute effect sizes.
    """
    set_report_style()

    plot_df = results_df.sort_values("abs_cohens_d", ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values("abs_cohens_d", ascending=True)

    colors = [
        "#B22222" if d > 0 else "#2E8B57"
        for d in plot_df["cohens_d_sick_minus_healthy"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(plot_df))))

    ax.barh(
        plot_df["feature"],
        plot_df["cohens_d_sick_minus_healthy"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Cohen's d (sick minus healthy)")
    ax.set_ylabel("Feature")

    ax.text(
        0.01,
        0.02,
        "Positive = higher in sick | Negative = higher in healthy",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    return save_figure(fig, output_path)


def plot_volcano(
    results_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Feature-level Effect Size vs FDR-adjusted Significance"
):
    """
    Volcano-style plot:
        x-axis = Cohen's d
        y-axis = -log10(FDR q-value)
    """
    set_report_style()

    plot_df = results_df.copy()
    plot_df["minus_log10_q"] = -np.log10(
        np.clip(plot_df["q_value_fdr_bh"].to_numpy(dtype=float), 1e-300, 1)
    )

    color_map = {
        "statistical_surface": "#1F77B4",
        "physics_informed_recovery": "#FF7F0E",
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for group_name, group_df in plot_df.groupby("main_feature_group"):
        ax.scatter(
            group_df["cohens_d_sick_minus_healthy"],
            group_df["minus_log10_q"],
            s=42,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
            label=group_name.replace("_", " "),
            color=color_map.get(group_name, "gray"),
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(-np.log10(0.05), color="black", linewidth=1, linestyle="--", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Cohen's d (sick minus healthy)")
    ax.set_ylabel("-log10(FDR-adjusted q-value)")
    ax.legend(frameon=True)

    # Label the top few features by combined significance/effect.
    label_df = plot_df.sort_values(
        ["significant_fdr_005", "abs_cohens_d"],
        ascending=[False, False]
    ).head(10)

    for _, row in label_df.iterrows():
        ax.annotate(
            row["feature"],
            (row["cohens_d_sick_minus_healthy"], row["minus_log10_q"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    return save_figure(fig, output_path)


def plot_significant_counts_by_subgroup(
    group_summary_df: pd.DataFrame,
    output_path: str | Path
):
    """
    Plot FDR-significant feature counts by subgroup.
    """
    set_report_style()

    plot_df = group_summary_df.sort_values(
        "n_significant_fdr_005",
        ascending=True
    ).copy()

    colors = [
        "#1F77B4" if group == "statistical_surface" else "#FF7F0E"
        for group in plot_df["main_feature_group"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(plot_df))))

    ax.barh(
        plot_df["feature_subgroup"],
        plot_df["n_significant_fdr_005"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title("Number of FDR-significant Features by Feature Subgroup")
    ax.set_xlabel("Number of features with FDR q < 0.05")
    ax.set_ylabel("Feature subgroup")

    return save_figure(fig, output_path)


def plot_median_effect_by_subgroup(
    group_summary_df: pd.DataFrame,
    output_path: str | Path
):
    """
    Plot median absolute Cohen's d by subgroup.
    """
    set_report_style()

    plot_df = group_summary_df.sort_values(
        "median_abs_cohens_d",
        ascending=True
    ).copy()

    colors = [
        "#1F77B4" if group == "statistical_surface" else "#FF7F0E"
        for group in plot_df["main_feature_group"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(plot_df))))

    ax.barh(
        plot_df["feature_subgroup"],
        plot_df["median_abs_cohens_d"],
        color=colors,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_title("Median Absolute Effect Size by Feature Subgroup")
    ax.set_xlabel("Median |Cohen's d|")
    ax.set_ylabel("Feature subgroup")

    return save_figure(fig, output_path)


def plot_effect_size_distribution_by_group(
    results_df: pd.DataFrame,
    output_path: str | Path
):
    """
    Scatter plot showing effect-size spread within each main feature group.
    """
    set_report_style()

    plot_df = results_df.copy()

    groups = ["statistical_surface", "physics_informed_recovery"]
    x_positions = {group: i for i, group in enumerate(groups)}

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(7, 5))

    for group in groups:
        group_df = plot_df[plot_df["main_feature_group"] == group].copy()

        x = np.full(len(group_df), x_positions[group], dtype=float)
        x = x + rng.normal(0, 0.05, size=len(group_df))

        ax.scatter(
            x,
            group_df["abs_cohens_d"],
            alpha=0.70,
            s=38,
            edgecolor="black",
            linewidth=0.3,
            label=group.replace("_", " "),
        )

        median_value = group_df["abs_cohens_d"].median()
        ax.hlines(
            median_value,
            x_positions[group] - 0.22,
            x_positions[group] + 0.22,
            colors="black",
            linewidth=2,
        )

    ax.set_xticks([x_positions[g] for g in groups])
    ax.set_xticklabels([g.replace("_", "\n") for g in groups])
    ax.set_ylabel("|Cohen's d|")
    ax.set_title("Effect-size Distribution by Main Feature Family")

    return save_figure(fig, output_path)


def run_statistical_testing_pipeline(
    feature_table_path: str | Path,
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
):
    """
    Run full statistical testing pipeline and save tables/figures.
    """
    feature_table_path = Path(feature_table_path)
    output_tables_dir = Path(output_tables_dir)
    output_figures_dir = Path(output_figures_dir)

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_table_path)

    print("Loaded feature table:", feature_table_path)
    print("Shape:", df.shape)
    print("Class counts:")
    print(df["class_name"].value_counts())

    feature_cols = STATISTICAL_FEATURES + PHYSICS_INFORMED_FEATURES

    results_df = run_feature_statistical_tests(
        feature_df=df,
        feature_cols=feature_cols,
        label_col="label",
        healthy_label=0,
        sick_label=1,
    )

    group_summary_df = summarize_by_feature_group(results_df)

    all_results_path = output_tables_dir / "feature_statistical_tests_all.csv"
    significant_path = output_tables_dir / "feature_statistical_tests_fdr_significant.csv"
    top_effects_path = output_tables_dir / "feature_statistical_tests_top_abs_cohens_d.csv"
    group_summary_path = output_tables_dir / "feature_statistical_tests_group_summary.csv"

    results_df.to_csv(all_results_path, index=False)
    results_df[results_df["significant_fdr_005"]].to_csv(significant_path, index=False)
    results_df.sort_values("abs_cohens_d", ascending=False).head(40).to_csv(top_effects_path, index=False)
    group_summary_df.to_csv(group_summary_path, index=False)

    saved_figures = []

    saved_figures.append(plot_top_effect_sizes(
        results_df,
        output_path=output_figures_dir / "top_25_features_by_cohens_d.png",
        top_n=25,
        title="Top 25 Features by Absolute Cohen's d"
    ))

    saved_figures.append(plot_top_effect_sizes(
        results_df[results_df["main_feature_group"] == "statistical_surface"],
        output_path=output_figures_dir / "top_statistical_features_by_cohens_d.png",
        top_n=20,
        title="Top Statistical Surface Features by Cohen's d"
    ))

    saved_figures.append(plot_top_effect_sizes(
        results_df[results_df["main_feature_group"] == "physics_informed_recovery"],
        output_path=output_figures_dir / "top_physics_features_by_cohens_d.png",
        top_n=20,
        title="Top Physics-informed Recovery Features by Cohen's d"
    ))

    saved_figures.append(plot_volcano(
        results_df,
        output_path=output_figures_dir / "feature_effect_size_volcano_fdr.png"
    ))

    saved_figures.append(plot_significant_counts_by_subgroup(
        group_summary_df,
        output_path=output_figures_dir / "fdr_significant_counts_by_feature_subgroup.png"
    ))

    saved_figures.append(plot_median_effect_by_subgroup(
        group_summary_df,
        output_path=output_figures_dir / "median_abs_effect_size_by_feature_subgroup.png"
    ))

    saved_figures.append(plot_effect_size_distribution_by_group(
        results_df,
        output_path=output_figures_dir / "effect_size_distribution_by_main_feature_family.png"
    ))

    print("\nSaved statistical test tables:")
    print(all_results_path)
    print(significant_path)
    print(top_effects_path)
    print(group_summary_path)

    print("\nSaved statistical test figures:")
    for path in saved_figures:
        print(path)

    summary = {
        "feature_table_path": str(feature_table_path),
        "n_features_tested": int(len(results_df)),
        "n_fdr_significant_005": int(results_df["significant_fdr_005"].sum()),
        "n_raw_p_significant_005": int(results_df["significant_raw_p_005"].sum()),
        "top_10_abs_effect_features": results_df.sort_values(
            "abs_cohens_d",
            ascending=False
        ).head(10)[
            [
                "feature",
                "main_feature_group",
                "feature_subgroup",
                "cohens_d_sick_minus_healthy",
                "p_value",
                "q_value_fdr_bh",
            ]
        ].to_dict(orient="records"),
        "saved_tables": {
            "all_results": str(all_results_path),
            "fdr_significant": str(significant_path),
            "top_effects": str(top_effects_path),
            "group_summary": str(group_summary_path),
        },
        "saved_figures": [str(p) for p in saved_figures],
    }

    return summary