"""
Logistic regression model experiments.

Experiments:
1. Unpruned logistic regression
2. Correlation-pruned logistic regression

Feature sets:
1. Statistical surface
2. Physics-informed recovery
3. Combined statistical + physics-informed
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from src.evaluation.classification_metrics import compute_binary_classification_metrics
from src.visualization.plot_style import set_report_style, save_figure
from src.features.feature_groups import get_feature_groups


def make_logistic_pipeline(random_state: int = 42) -> Pipeline:
    """
    Median imputation + standard scaling + balanced logistic regression.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )),
    ])


def select_uncorrelated_features_with_rf_importance(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    corr_threshold: float = 0.90,
    random_state: int = 42,
) -> List[str]:
    """
    Training-fold-only redundant feature removal.

    Procedure:
    1. Median-impute training features.
    2. Compute absolute Pearson correlation among features.
    3. Train Random Forest on the training fold.
    4. For correlated feature pairs, drop the feature with lower RF importance.

    This avoids using test-fold information during feature selection.
    """
    feature_cols = list(X_train.columns)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)

    X_imp_df = pd.DataFrame(X_imp, columns=feature_cols)

    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_imp_df, y_train)

    importances = pd.Series(rf.feature_importances_, index=feature_cols)

    corr = X_imp_df.corr(method="pearson").abs()

    to_drop = set()

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1 = feature_cols[i]
            f2 = feature_cols[j]

            if corr.iloc[i, j] >= corr_threshold:
                if importances[f1] >= importances[f2]:
                    to_drop.add(f2)
                else:
                    to_drop.add(f1)

    selected_features = [
        f for f in feature_cols
        if f not in to_drop
    ]

    return selected_features


def evaluate_logistic_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    experiment_name: str,
    feature_group_name: str,
    prune_correlated: bool = False,
    corr_threshold: float = 0.90,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one logistic regression experiment using stratified 5-fold CV.

    Returns
    -------
    fold_metrics_df:
        One row per fold.
    oof_predictions_df:
        Out-of-fold predictions for pooled ROC/confusion matrix.
    """
    X = df[feature_cols].copy()
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows = []
    oof_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        start_time = time.time()

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        if prune_correlated:
            selected_features = select_uncorrelated_features_with_rf_importance(
                X_train=X_train,
                y_train=y_train,
                corr_threshold=corr_threshold,
                random_state=random_state,
            )
        else:
            selected_features = list(feature_cols)

        model = make_logistic_pipeline(random_state=random_state)

        model.fit(X_train[selected_features], y_train)

        y_score = model.predict_proba(X_test[selected_features])[:, 1]
        y_pred = (y_score >= 0.5).astype(int)

        metrics = compute_binary_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
        )

        runtime_seconds = time.time() - start_time

        fold_row = {
            "experiment_name": experiment_name,
            "feature_group_name": feature_group_name,
            "fold": fold_idx,
            "prune_correlated": prune_correlated,
            "corr_threshold": corr_threshold if prune_correlated else np.nan,
            "n_original_features": len(feature_cols),
            "n_selected_features": len(selected_features),
            "runtime_seconds": runtime_seconds,
            **metrics,
        }

        fold_rows.append(fold_row)

        fold_patient_ids = df.iloc[test_idx]["patient_id"].astype(str).tolist()
        fold_class_names = df.iloc[test_idx]["class_name"].astype(str).tolist()

        for patient_id, class_name, true_label, pred_label, score in zip(
            fold_patient_ids,
            fold_class_names,
            y_test,
            y_pred,
            y_score,
        ):
            oof_rows.append({
                "experiment_name": experiment_name,
                "feature_group_name": feature_group_name,
                "fold": fold_idx,
                "patient_id": patient_id,
                "class_name": class_name,
                "y_true": int(true_label),
                "y_pred": int(pred_label),
                "y_score": float(score),
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def summarize_experiment_results(
    fold_metrics_df: pd.DataFrame,
    oof_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize each experiment using:
    1. Mean ± std across folds
    2. Pooled out-of-fold metrics
    """
    summary_rows = []

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "precision_sick",
        "recall_sick",
        "sensitivity_sick",
        "specificity_healthy",
        "f1_sick",
        "runtime_seconds",
        "n_selected_features",
    ]

    for experiment_name, group_fold_df in fold_metrics_df.groupby("experiment_name"):
        group_oof_df = oof_predictions_df[
            oof_predictions_df["experiment_name"] == experiment_name
        ].copy()

        pooled_metrics = compute_binary_classification_metrics(
            y_true=group_oof_df["y_true"],
            y_pred=group_oof_df["y_pred"],
            y_score=group_oof_df["y_score"],
        )

        first_row = group_fold_df.iloc[0]

        row = {
            "experiment_name": experiment_name,
            "feature_group_name": first_row["feature_group_name"],
            "prune_correlated": bool(first_row["prune_correlated"]),
            "corr_threshold": first_row["corr_threshold"],
            "n_original_features": int(first_row["n_original_features"]),
            "mean_selected_features": float(group_fold_df["n_selected_features"].mean()),
            "std_selected_features": float(group_fold_df["n_selected_features"].std(ddof=1)),
        }

        for metric in metric_cols:
            row[f"mean_{metric}"] = float(group_fold_df[metric].mean())
            row[f"std_{metric}"] = float(group_fold_df[metric].std(ddof=1))

        for key, value in pooled_metrics.items():
            row[f"pooled_{key}"] = value

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        ["pooled_roc_auc", "pooled_accuracy"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return summary_df


def plot_model_comparison(summary_df: pd.DataFrame, output_dir: str | Path):
    """
    Save model comparison bar plots.
    """
    set_report_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["experiment_name"].str.replace("_", "\n")

    for metric, ylabel, filename in [
        ("pooled_accuracy", "Pooled out-of-fold accuracy", "model_comparison_accuracy.png"),
        ("pooled_roc_auc", "Pooled out-of-fold ROC-AUC", "model_comparison_roc_auc.png"),
        ("mean_selected_features", "Mean selected features", "model_comparison_selected_features.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            plot_df["label"],
            plot_df[metric],
            edgecolor="black",
            linewidth=0.6,
        )

        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)

        if metric != "mean_selected_features":
            ax.set_ylim(0, 1)

        saved.append(save_figure(fig, output_dir / filename))

    return saved


def plot_roc_curves(oof_predictions_df: pd.DataFrame, output_path: str | Path):
    """
    Plot pooled out-of-fold ROC curves for all experiments.
    """
    set_report_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    for experiment_name, group_df in oof_predictions_df.groupby("experiment_name"):
        y_true = group_df["y_true"].to_numpy(dtype=int)
        y_score = group_df["y_score"].to_numpy(dtype=float)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{experiment_name.replace('_', ' ')} (AUC={roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_title("Pooled Out-of-Fold ROC Curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sick sensitivity")
    ax.legend(frameon=True, fontsize=8)

    return save_figure(fig, output_path)


def plot_confusion_matrix_custom(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    title: str,
    output_path: str | Path,
):
    """
    Save custom confusion matrix.

    Correct healthy/healthy cell is green.
    Correct sick/sick cell is red.
    Off-diagonal errors are gray.
    """
    set_report_style()

    matrix = np.array([[tn, fp], [fn, tp]])

    cell_colors = np.array([
        ["#2E8B57", "#D9D9D9"],
        ["#D9D9D9", "#B22222"],
    ])

    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    for i in range(2):
        for j in range(2):
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=cell_colors[i, j],
                    edgecolor="black",
                    linewidth=1.5,
                )
            )

            ax.text(
                j + 0.5,
                i + 0.5,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color="white" if (i, j) in [(0, 0), (1, 1)] else "black",
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Predicted healthy", "Predicted sick"])

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["True healthy", "True sick"])

    ax.set_title(title)
    ax.grid(False)

    return save_figure(fig, output_path)


def plot_all_confusion_matrices(summary_df: pd.DataFrame, output_dir: str | Path):
    """
    Save one confusion matrix per experiment using pooled OOF counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for _, row in summary_df.iterrows():
        experiment_name = row["experiment_name"]

        saved.append(plot_confusion_matrix_custom(
            tn=int(row["pooled_tn"]),
            fp=int(row["pooled_fp"]),
            fn=int(row["pooled_fn"]),
            tp=int(row["pooled_tp"]),
            title=experiment_name.replace("_", " "),
            output_path=output_dir / f"confusion_matrix_{experiment_name}.png",
        ))

    return saved


def run_logistic_model_experiments(
    feature_table_path: str | Path,
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
):
    """
    Run all logistic regression experiments and save outputs.
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

    feature_groups = get_feature_groups()

    experiments = [
        {
            "experiment_name": "statistical_surface_unpruned_logistic",
            "feature_group_name": "statistical_surface",
            "feature_cols": feature_groups["statistical_surface"],
            "prune_correlated": False,
        },
        {
            "experiment_name": "physics_informed_unpruned_logistic",
            "feature_group_name": "physics_informed_recovery",
            "feature_cols": feature_groups["physics_informed_recovery"],
            "prune_correlated": False,
        },
        {
            "experiment_name": "combined_unpruned_logistic",
            "feature_group_name": "combined_statistical_physics",
            "feature_cols": feature_groups["combined_statistical_physics"],
            "prune_correlated": False,
        },
        {
            "experiment_name": "statistical_surface_corr090_pruned_logistic",
            "feature_group_name": "statistical_surface",
            "feature_cols": feature_groups["statistical_surface"],
            "prune_correlated": True,
        },
        {
            "experiment_name": "physics_informed_corr090_pruned_logistic",
            "feature_group_name": "physics_informed_recovery",
            "feature_cols": feature_groups["physics_informed_recovery"],
            "prune_correlated": True,
        },
        {
            "experiment_name": "combined_corr090_pruned_logistic",
            "feature_group_name": "combined_statistical_physics",
            "feature_cols": feature_groups["combined_statistical_physics"],
            "prune_correlated": True,
        },
    ]

    all_fold_metrics = []
    all_oof_predictions = []

    for exp in experiments:
        print("\nRunning:", exp["experiment_name"])

        fold_df, oof_df = evaluate_logistic_cv(
            df=df,
            feature_cols=exp["feature_cols"],
            experiment_name=exp["experiment_name"],
            feature_group_name=exp["feature_group_name"],
            prune_correlated=exp["prune_correlated"],
            corr_threshold=0.90,
            n_splits=5,
            random_state=42,
        )

        all_fold_metrics.append(fold_df)
        all_oof_predictions.append(oof_df)

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    oof_predictions_df = pd.concat(all_oof_predictions, ignore_index=True)

    summary_df = summarize_experiment_results(
        fold_metrics_df=fold_metrics_df,
        oof_predictions_df=oof_predictions_df,
    )

    fold_metrics_path = output_tables_dir / "logistic_fold_metrics.csv"
    oof_predictions_path = output_tables_dir / "logistic_oof_predictions.csv"
    summary_path = output_tables_dir / "logistic_model_comparison_summary.csv"

    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    oof_predictions_df.to_csv(oof_predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    comparison_figures = plot_model_comparison(
        summary_df=summary_df,
        output_dir=output_figures_dir,
    )

    roc_path = plot_roc_curves(
        oof_predictions_df=oof_predictions_df,
        output_path=output_figures_dir / "pooled_oof_roc_curves_logistic.png",
    )

    confusion_paths = plot_all_confusion_matrices(
        summary_df=summary_df,
        output_dir=output_figures_dir / "confusion_matrices",
    )

    print("\nSaved model tables:")
    print(fold_metrics_path)
    print(oof_predictions_path)
    print(summary_path)

    print("\nSaved model figures:")
    for path in comparison_figures:
        print(path)
    print(roc_path)
    for path in confusion_paths:
        print(path)

    print("\nModel comparison summary:")
    display_cols = [
        "experiment_name",
        "n_original_features",
        "mean_selected_features",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_roc_auc",
        "pooled_precision_sick",
        "pooled_recall_sick",
        "pooled_specificity_healthy",
        "pooled_f1_sick",
        "pooled_tn",
        "pooled_fp",
        "pooled_fn",
        "pooled_tp",
    ]
    print(summary_df[display_cols].to_string(index=False))

    return {
        "summary_path": str(summary_path),
        "fold_metrics_path": str(fold_metrics_path),
        "oof_predictions_path": str(oof_predictions_path),
        "figures": [str(p) for p in comparison_figures] + [str(roc_path)] + [str(p) for p in confusion_paths],
        "n_experiments": len(experiments),
    }