"""
Main pipeline runner for the DMR-IR dynamic thermography project.

Usage examples
--------------
Run full reproducible analysis pipeline:

    python main.py --stage all

Run one stage:

    python main.py --stage prepare
    python main.py --stage eda
    python main.py --stage features
    python main.py --stage stats
    python main.py --stage logistic
    python main.py --stage temporal
    python main.py --stage final

Notes
-----
The data downloader is intentionally not included in --stage all because it
requires browser-based manual login to the DMR-IR database.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


STAGE_TO_SCRIPT = {
    "prepare": PROJECT_DIR / "scripts" / "01_prepare_dataset.py",
    "eda": PROJECT_DIR / "scripts" / "02_run_eda.py",
    "features": PROJECT_DIR / "scripts" / "03_extract_features.py",
    "stats": PROJECT_DIR / "scripts" / "04_run_statistical_tests.py",
    "logistic": PROJECT_DIR / "scripts" / "05_run_logistic_models.py",
    "temporal": PROJECT_DIR / "scripts" / "06_run_temporal_cnn_models.py",
}


PIPELINE_ORDER = [
    "prepare",
    "eda",
    "features",
    "stats",
    "logistic",
    "temporal",
]


def run_script(script_path: Path) -> None:
    """
    Run one pipeline script and stop immediately if it fails.
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print("\n" + "=" * 80)
    print(f"Running: {script_path.relative_to(PROJECT_DIR)}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage failed with return code {result.returncode}: {script_path}"
        )

    print(f"Completed: {script_path.relative_to(PROJECT_DIR)}")


def create_final_master_outputs() -> None:
    """
    Build the final master model-comparison table and report-ready figures.

    This stage combines:
    - logistic model comparison results
    - temporal CNN / flat-logistic comparison results
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    logistic_path = (
        PROJECT_DIR
        / "outputs"
        / "tables"
        / "model_experiments"
        / "logistic_model_comparison_summary.csv"
    )

    temporal_path = (
        PROJECT_DIR
        / "outputs"
        / "tables"
        / "temporal_cnn_experiments"
        / "temporal_cnn_model_comparison_summary.csv"
    )

    output_tables_dir = PROJECT_DIR / "outputs" / "tables" / "final_results"
    output_figures_dir = PROJECT_DIR / "outputs" / "figures" / "final_results"

    output_tables_dir.mkdir(parents=True, exist_ok=True)
    output_figures_dir.mkdir(parents=True, exist_ok=True)

    if not logistic_path.exists():
        raise FileNotFoundError(f"Missing logistic summary: {logistic_path}")

    if not temporal_path.exists():
        raise FileNotFoundError(f"Missing temporal summary: {temporal_path}")

    logistic_df = pd.read_csv(logistic_path)
    temporal_df = pd.read_csv(temporal_path)

    logistic_report = logistic_df.copy()
    logistic_report["experiment_family"] = "feature_table_logistic"
    logistic_report["model_type"] = "logistic_regression"
    logistic_report["input_type"] = logistic_report["feature_group_name"]
    logistic_report["n_input_features"] = logistic_report["n_original_features"]
    logistic_report["mean_features_used"] = logistic_report["mean_selected_features"]

    logistic_report = logistic_report.rename(columns={
        "mean_accuracy": "cv_accuracy_mean",
        "std_accuracy": "cv_accuracy_std",
        "mean_roc_auc": "cv_roc_auc_mean",
        "std_roc_auc": "cv_roc_auc_std",
        "pooled_tn": "tn",
        "pooled_fp": "fp",
        "pooled_fn": "fn",
        "pooled_tp": "tp",
    })

    logistic_report = logistic_report[
        [
            "experiment_family",
            "experiment_name",
            "model_type",
            "input_type",
            "n_input_features",
            "mean_features_used",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "pooled_accuracy",
            "pooled_roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]

    temporal_report = temporal_df.copy()
    temporal_report["experiment_family"] = "temporal_channel_experiment"
    temporal_report["input_type"] = temporal_report["channel_set_name"]
    temporal_report["n_input_features"] = temporal_report["n_temporal_features"]
    temporal_report["mean_features_used"] = temporal_report["n_temporal_features"]

    temporal_report = temporal_report.rename(columns={
        "mean_accuracy": "cv_accuracy_mean",
        "std_accuracy": "cv_accuracy_std",
        "mean_roc_auc": "cv_roc_auc_mean",
        "std_roc_auc": "cv_roc_auc_std",
        "pooled_tn": "tn",
        "pooled_fp": "fp",
        "pooled_fn": "fn",
        "pooled_tp": "tp",
    })

    temporal_report = temporal_report[
        [
            "experiment_family",
            "experiment_name",
            "model_type",
            "input_type",
            "n_input_features",
            "mean_features_used",
            "cv_accuracy_mean",
            "cv_accuracy_std",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "pooled_accuracy",
            "pooled_roc_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ]

    master_df = pd.concat(
        [logistic_report, temporal_report],
        ignore_index=True,
    )

    master_df = master_df.sort_values(
        ["cv_roc_auc_mean", "cv_accuracy_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    ranked_path = output_tables_dir / "ranked_final_model_comparison.csv"
    top10_path = output_tables_dir / "top10_final_model_comparison.csv"

    master_df.to_csv(ranked_path, index=False)
    master_df.head(10).to_csv(top10_path, index=False)

    master_df["report_label"] = (
        master_df["experiment_name"]
        .str.replace("_logistic", "", regex=False)
        .str.replace("_temporal_cnn", " CNN", regex=False)
        .str.replace("_flat_logistic", " flat logistic", regex=False)
        .str.replace("_", " ", regex=False)
    )

    # ROC-AUC ranking plot.
    plot_df = master_df.head(12).copy()
    plot_df = plot_df.sort_values("cv_roc_auc_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_df["report_label"],
        plot_df["cv_roc_auc_mean"],
        xerr=plot_df["cv_roc_auc_std"],
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    ax.set_xlabel("Mean 5-fold ROC-AUC")
    ax.set_ylabel("Model")
    ax.set_title("Final Model Comparison by ROC-AUC")
    ax.set_xlim(0, 1)
    plt.tight_layout()

    roc_plot_path = output_figures_dir / "final_model_comparison_cv_roc_auc.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Accuracy ranking plot.
    plot_df = master_df.head(12).copy()
    plot_df = plot_df.sort_values("cv_accuracy_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        plot_df["report_label"],
        plot_df["cv_accuracy_mean"],
        xerr=plot_df["cv_accuracy_std"],
        edgecolor="black",
        linewidth=0.6,
        capsize=3,
    )
    ax.set_xlabel("Mean 5-fold accuracy")
    ax.set_ylabel("Model")
    ax.set_title("Final Model Comparison by Accuracy")
    ax.set_xlim(0, 1)
    plt.tight_layout()

    acc_plot_path = output_figures_dir / "final_model_comparison_cv_accuracy.png"
    plt.savefig(acc_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Same-channel CNN vs flat logistic plot.
    temporal_only = master_df[
        master_df["experiment_family"] == "temporal_channel_experiment"
    ].copy()

    temporal_only["channel_short"] = (
        temporal_only["input_type"]
        .str.replace("temporal_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    pivot_auc = temporal_only.pivot_table(
        index="channel_short",
        columns="model_type",
        values="cv_roc_auc_mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(pivot_auc.index))
    width = 0.35

    ax.bar(
        x - width / 2,
        pivot_auc["flat_logistic"],
        width,
        label="Flat logistic",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.bar(
        x + width / 2,
        pivot_auc["temporal_cnn"],
        width,
        label="Temporal CNN",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_auc.index, rotation=30, ha="right")
    ax.set_ylabel("Mean 5-fold ROC-AUC")
    ax.set_title("Same-channel Comparison: Temporal CNN vs Flat Logistic")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    same_channel_plot_path = (
        output_figures_dir / "same_channel_cnn_vs_flat_logistic_roc_auc.png"
    )

    plt.savefig(same_channel_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved final comparison tables:")
    print(ranked_path)
    print(top10_path)

    print("\nSaved final comparison figures:")
    print(roc_plot_path)
    print(acc_plot_path)
    print(same_channel_plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DMR-IR dynamic thermography analysis pipeline."
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "all",
            "prepare",
            "eda",
            "features",
            "stats",
            "logistic",
            "temporal",
            "final",
        ],
        help="Pipeline stage to run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage == "all":
        for stage in PIPELINE_ORDER:
            run_script(STAGE_TO_SCRIPT[stage])

        create_final_master_outputs()
        print("\nFull pipeline completed successfully.")
        return

    if args.stage == "final":
        create_final_master_outputs()
        print("\nFinal output stage completed successfully.")
        return

    run_script(STAGE_TO_SCRIPT[args.stage])

    print("\nRequested stage completed successfully.")


if __name__ == "__main__":
    main()