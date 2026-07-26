"""
Temporal CNN experiments for DMR-IR dynamic thermography.

Goal:
Compare 1D CNN temporal models against same-input flat logistic baselines.

Input variants:
A. 20 x 2: frame_mean + frame_std
B. 20 x 3: frame_mean + frame_std + mean_delta
C. 20 x 4: frame_mean + frame_std + mean_delta + std_delta
D. 20 x 5: frame_mean + frame_std + mean_delta + std_delta + cumulative_recovery
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

from src.evaluation.classification_metrics import compute_binary_classification_metrics
from src.visualization.plot_style import set_report_style, save_figure
from src.models.logistic_model_experiments import plot_confusion_matrix_custom


def set_all_seeds(seed: int = 42):
    """
    Make runs as reproducible as practical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_temporal_channel_specs() -> Dict[str, List[str]]:
    """
    Return temporal channel-set definitions.
    """
    return {
        "temporal_2ch_mean_std": [
            "frame_mean",
            "frame_std",
        ],
        "temporal_3ch_mean_std_mean_delta": [
            "frame_mean",
            "frame_std",
            "mean_delta",
        ],
        "temporal_4ch_mean_std_mean_delta_std_delta": [
            "frame_mean",
            "frame_std",
            "mean_delta",
            "std_delta",
        ],
        "temporal_5ch_mean_std_deltas_cumulative": [
            "frame_mean",
            "frame_std",
            "mean_delta",
            "std_delta",
            "cumulative_recovery",
        ],
    }


def build_temporal_channels(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build base temporal channels from the final feature table.

    Returns
    -------
    dict
        Each value has shape (n_patients, 20).
    """
    frame_mean_cols = [f"frame_mean_{i:02d}" for i in range(20)]
    frame_std_cols = [f"frame_std_{i:02d}" for i in range(20)]

    frame_mean = df[frame_mean_cols].to_numpy(dtype=np.float32)
    frame_std = df[frame_std_cols].to_numpy(dtype=np.float32)

    mean_delta = np.zeros_like(frame_mean, dtype=np.float32)
    mean_delta[:, 1:] = np.diff(frame_mean, axis=1)

    std_delta = np.zeros_like(frame_std, dtype=np.float32)
    std_delta[:, 1:] = np.diff(frame_std, axis=1)

    cumulative_recovery = frame_mean - frame_mean[:, [0]]

    return {
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "cumulative_recovery": cumulative_recovery,
    }


def build_temporal_tensor(
    df: pd.DataFrame,
    channel_names: List[str],
) -> np.ndarray:
    """
    Build temporal tensor with shape:
        n_patients x 20_frames x n_channels
    """
    channels = build_temporal_channels(df)

    missing = [
        channel for channel in channel_names
        if channel not in channels
    ]

    if missing:
        raise ValueError(f"Unknown temporal channels: {missing}")

    X = np.stack(
        [channels[channel] for channel in channel_names],
        axis=2,
    ).astype(np.float32)

    return X


def standardize_temporal_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Channel-wise standardization.

    Mean/std are fit only on the inner training split.
    """
    mean = np.nanmean(X_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_train, axis=(0, 1), keepdims=True)

    std = np.where(std < 1e-8, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled


class TemporalCNN(nn.Module):
    """
    Compact 1D CNN over temporal sequence.

    Input to forward:
        batch x channels x frames
    """

    def __init__(self, n_channels: int, dropout: float = 0.25):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


def train_one_cnn_fold(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_channels: int,
    fold_seed: int,
    device: torch.device,
    max_epochs: int = 250,
    patience: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict]:
    """
    Train CNN for one outer CV fold using an inner validation split
    for early stopping.
    """
    set_all_seeds(fold_seed)

    inner_indices = np.arange(len(y_train_full))

    train_inner_idx, val_inner_idx = train_test_split(
        inner_indices,
        test_size=0.20,
        stratify=y_train_full,
        random_state=fold_seed,
    )

    X_train_inner = X_train_full[train_inner_idx]
    y_train_inner = y_train_full[train_inner_idx]

    X_val_inner = X_train_full[val_inner_idx]
    y_val_inner = y_train_full[val_inner_idx]

    X_train_inner, X_val_inner, X_test_scaled = standardize_temporal_train_val_test(
        X_train=X_train_inner,
        X_val=X_val_inner,
        X_test=X_test,
    )

    # Convert from N x T x C to N x C x T for Conv1D.
    X_train_tensor = torch.tensor(
        np.transpose(X_train_inner, (0, 2, 1)),
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train_inner,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        np.transpose(X_val_inner, (0, 2, 1)),
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val_inner,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        np.transpose(X_test_scaled, (0, 2, 1)),
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    n_pos = max(float((y_train_inner == 1).sum()), 1.0)
    n_neg = max(float((y_train_inner == 0).sum()), 1.0)

    pos_weight = torch.tensor(
        n_neg / n_pos,
        dtype=torch.float32,
        device=device,
    )

    model = TemporalCNN(
        n_channels=n_channels,
        dropout=0.25,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)

    for epoch in range(1, max_epochs + 1):
        model.train()

        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))

        model.eval()

        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).detach().cpu())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict({
            key: value.to(device)
            for key, value in best_state.items()
        })

    model.eval()

    X_test_tensor = X_test_tensor.to(device)

    with torch.no_grad():
        test_logits = model(X_test_tensor)
        y_score = torch.sigmoid(test_logits).detach().cpu().numpy()

    y_pred = (y_score >= 0.5).astype(int)

    metrics = compute_binary_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
    )

    extra = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_trained": int(len(history["train_loss"])),
    }

    return metrics, y_pred, y_score, extra


def evaluate_flat_logistic_temporal_cv(
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    channel_set_name: str,
    channel_names: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same-input flat logistic baseline.

    The temporal tensor is flattened:
        20 x channels -> 20*channels features
    """
    X_flat = X_temporal.reshape(X_temporal.shape[0], -1)
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_rows = []
    oof_rows = []

    experiment_name = f"{channel_set_name}_flat_logistic"

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_flat, y), start=1):
        start_time = time.time()

        X_train = X_flat[train_idx]
        X_test = X_flat[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            )),
        ])

        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)

        metrics = compute_binary_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
        )

        runtime_seconds = time.time() - start_time

        fold_rows.append({
            "experiment_name": experiment_name,
            "model_type": "flat_logistic",
            "channel_set_name": channel_set_name,
            "channel_names": "|".join(channel_names),
            "n_channels": len(channel_names),
            "n_temporal_features": X_flat.shape[1],
            "fold": fold_idx,
            "runtime_seconds": runtime_seconds,
            "best_epoch": np.nan,
            "best_val_loss": np.nan,
            "epochs_trained": np.nan,
            **metrics,
        })

        for patient_id, class_name, true_label, pred_label, score in zip(
            df.iloc[test_idx]["patient_id"].astype(str),
            df.iloc[test_idx]["class_name"].astype(str),
            y_test,
            y_pred,
            y_score,
        ):
            oof_rows.append({
                "experiment_name": experiment_name,
                "model_type": "flat_logistic",
                "channel_set_name": channel_set_name,
                "patient_id": patient_id,
                "class_name": class_name,
                "fold": fold_idx,
                "y_true": int(true_label),
                "y_pred": int(pred_label),
                "y_score": float(score),
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def evaluate_cnn_temporal_cv(
    df: pd.DataFrame,
    X_temporal: np.ndarray,
    channel_set_name: str,
    channel_names: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate temporal CNN using stratified 5-fold CV.
    """
    y = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device for CNN: {device}")

    fold_rows = []
    oof_rows = []

    experiment_name = f"{channel_set_name}_temporal_cnn"

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_temporal, y), start=1):
        start_time = time.time()

        X_train_full = X_temporal[train_idx]
        X_test = X_temporal[test_idx]

        y_train_full = y[train_idx]
        y_test = y[test_idx]

        metrics, y_pred, y_score, extra = train_one_cnn_fold(
            X_train_full=X_train_full,
            y_train_full=y_train_full,
            X_test=X_test,
            y_test=y_test,
            n_channels=len(channel_names),
            fold_seed=random_state + fold_idx,
            device=device,
            max_epochs=250,
            patience=25,
            batch_size=32,
            learning_rate=1e-3,
            weight_decay=1e-4,
        )

        runtime_seconds = time.time() - start_time

        fold_rows.append({
            "experiment_name": experiment_name,
            "model_type": "temporal_cnn",
            "channel_set_name": channel_set_name,
            "channel_names": "|".join(channel_names),
            "n_channels": len(channel_names),
            "n_temporal_features": 20 * len(channel_names),
            "fold": fold_idx,
            "runtime_seconds": runtime_seconds,
            **extra,
            **metrics,
        })

        for patient_id, class_name, true_label, pred_label, score in zip(
            df.iloc[test_idx]["patient_id"].astype(str),
            df.iloc[test_idx]["class_name"].astype(str),
            y_test,
            y_pred,
            y_score,
        ):
            oof_rows.append({
                "experiment_name": experiment_name,
                "model_type": "temporal_cnn",
                "channel_set_name": channel_set_name,
                "patient_id": patient_id,
                "class_name": class_name,
                "fold": fold_idx,
                "y_true": int(true_label),
                "y_pred": int(pred_label),
                "y_score": float(score),
            })

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def summarize_temporal_results(
    fold_metrics_df: pd.DataFrame,
    oof_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize temporal model experiments.
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

        first = group_fold_df.iloc[0]

        row = {
            "experiment_name": experiment_name,
            "model_type": first["model_type"],
            "channel_set_name": first["channel_set_name"],
            "channel_names": first["channel_names"],
            "n_channels": int(first["n_channels"]),
            "n_temporal_features": int(first["n_temporal_features"]),
            "mean_epochs_trained": float(group_fold_df["epochs_trained"].mean(skipna=True)),
            "std_epochs_trained": float(group_fold_df["epochs_trained"].std(skipna=True)),
        }

        for metric in metric_cols:
            row[f"mean_{metric}"] = float(group_fold_df[metric].mean())
            row[f"std_{metric}"] = float(group_fold_df[metric].std(ddof=1))

        for key, value in pooled_metrics.items():
            row[f"pooled_{key}"] = value

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        ["mean_roc_auc", "mean_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return summary_df


def plot_temporal_metric_comparison(
    summary_df: pd.DataFrame,
    output_path: str | Path,
    metric_col: str,
    title: str,
    ylabel: str,
):
    """
    Bar plot for temporal model comparison.
    """
    set_report_style()

    plot_df = summary_df.copy()
    plot_df["plot_label"] = (
        plot_df["channel_set_name"].str.replace("temporal_", "", regex=False)
        + "\n"
        + plot_df["model_type"].str.replace("_", " ")
    )

    fig, ax = plt.subplots(figsize=(13, 5.5))

    ax.bar(
        plot_df["plot_label"],
        plot_df[metric_col],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)

    if "accuracy" in metric_col or "roc_auc" in metric_col:
        ax.set_ylim(0, 1)

    return save_figure(fig, output_path)


def plot_temporal_roc_curves(
    oof_predictions_df: pd.DataFrame,
    output_path: str | Path,
):
    """
    Plot pooled OOF ROC curves for temporal experiments.
    """
    set_report_style()

    fig, ax = plt.subplots(figsize=(9, 7))

    for experiment_name, group_df in oof_predictions_df.groupby("experiment_name"):
        y_true = group_df["y_true"].to_numpy(dtype=int)
        y_score = group_df["y_score"].to_numpy(dtype=float)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=1.8,
            label=f"{experiment_name.replace('_', ' ')} (AUC={roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)

    ax.set_title("Temporal Experiments: Pooled Out-of-Fold ROC Curves")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate / sick sensitivity")
    ax.legend(frameon=True, fontsize=7)

    return save_figure(fig, output_path)


def save_temporal_confusion_matrices(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
):
    """
    Save one confusion matrix per temporal experiment.
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


def run_temporal_cnn_experiments(
    feature_table_path: str | Path,
    output_tables_dir: str | Path,
    output_figures_dir: str | Path,
):
    """
    Run all temporal CNN and same-input flat logistic experiments.
    """
    set_all_seeds(42)

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

    channel_specs = get_temporal_channel_specs()

    spec_rows = []

    for spec_name, channels in channel_specs.items():
        spec_rows.append({
            "channel_set_name": spec_name,
            "n_channels": len(channels),
            "channels": "|".join(channels),
            "n_temporal_features_flat": 20 * len(channels),
        })

    channel_specs_df = pd.DataFrame(spec_rows)

    all_fold_metrics = []
    all_oof_predictions = []

    for channel_set_name, channel_names in channel_specs.items():
        print("\nRunning temporal channel set:", channel_set_name)
        print("Channels:", channel_names)

        X_temporal = build_temporal_tensor(
            df=df,
            channel_names=channel_names,
        )

        print("Temporal tensor shape:", X_temporal.shape)

        flat_fold_df, flat_oof_df = evaluate_flat_logistic_temporal_cv(
            df=df,
            X_temporal=X_temporal,
            channel_set_name=channel_set_name,
            channel_names=channel_names,
            n_splits=5,
            random_state=42,
        )

        cnn_fold_df, cnn_oof_df = evaluate_cnn_temporal_cv(
            df=df,
            X_temporal=X_temporal,
            channel_set_name=channel_set_name,
            channel_names=channel_names,
            n_splits=5,
            random_state=42,
        )

        all_fold_metrics.extend([flat_fold_df, cnn_fold_df])
        all_oof_predictions.extend([flat_oof_df, cnn_oof_df])

    fold_metrics_df = pd.concat(all_fold_metrics, ignore_index=True)
    oof_predictions_df = pd.concat(all_oof_predictions, ignore_index=True)

    summary_df = summarize_temporal_results(
        fold_metrics_df=fold_metrics_df,
        oof_predictions_df=oof_predictions_df,
    )

    channel_specs_path = output_tables_dir / "temporal_channel_specs.csv"
    fold_metrics_path = output_tables_dir / "temporal_cnn_fold_metrics.csv"
    oof_predictions_path = output_tables_dir / "temporal_cnn_oof_predictions.csv"
    summary_path = output_tables_dir / "temporal_cnn_model_comparison_summary.csv"

    channel_specs_df.to_csv(channel_specs_path, index=False)
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    oof_predictions_df.to_csv(oof_predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    saved_figures = []

    saved_figures.append(plot_temporal_metric_comparison(
        summary_df=summary_df,
        output_path=output_figures_dir / "temporal_model_comparison_mean_accuracy.png",
        metric_col="mean_accuracy",
        title="Temporal Experiments: Mean CV Accuracy",
        ylabel="Mean 5-fold accuracy",
    ))

    saved_figures.append(plot_temporal_metric_comparison(
        summary_df=summary_df,
        output_path=output_figures_dir / "temporal_model_comparison_mean_roc_auc.png",
        metric_col="mean_roc_auc",
        title="Temporal Experiments: Mean CV ROC-AUC",
        ylabel="Mean 5-fold ROC-AUC",
    ))

    saved_figures.append(plot_temporal_metric_comparison(
        summary_df=summary_df,
        output_path=output_figures_dir / "temporal_model_comparison_pooled_accuracy.png",
        metric_col="pooled_accuracy",
        title="Temporal Experiments: Pooled OOF Accuracy",
        ylabel="Pooled out-of-fold accuracy",
    ))

    saved_figures.append(plot_temporal_metric_comparison(
        summary_df=summary_df,
        output_path=output_figures_dir / "temporal_model_comparison_pooled_roc_auc.png",
        metric_col="pooled_roc_auc",
        title="Temporal Experiments: Pooled OOF ROC-AUC",
        ylabel="Pooled out-of-fold ROC-AUC",
    ))

    saved_figures.append(plot_temporal_roc_curves(
        oof_predictions_df=oof_predictions_df,
        output_path=output_figures_dir / "temporal_pooled_oof_roc_curves.png",
    ))

    confusion_paths = save_temporal_confusion_matrices(
        summary_df=summary_df,
        output_dir=output_figures_dir / "confusion_matrices",
    )

    saved_figures.extend(confusion_paths)

    print("\nSaved temporal model tables:")
    print(channel_specs_path)
    print(fold_metrics_path)
    print(oof_predictions_path)
    print(summary_path)

    print("\nSaved temporal model figures:")
    for path in saved_figures:
        print(path)

    print("\nTemporal model comparison summary:")
    display_cols = [
        "experiment_name",
        "model_type",
        "n_channels",
        "n_temporal_features",
        "mean_accuracy",
        "std_accuracy",
        "mean_roc_auc",
        "std_roc_auc",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_roc_auc",
        "pooled_tn",
        "pooled_fp",
        "pooled_fn",
        "pooled_tp",
        "mean_epochs_trained",
    ]

    print(summary_df[display_cols].to_string(index=False))

    return {
        "channel_specs_path": str(channel_specs_path),
        "fold_metrics_path": str(fold_metrics_path),
        "oof_predictions_path": str(oof_predictions_path),
        "summary_path": str(summary_path),
        "figures": [str(p) for p in saved_figures],
        "n_experiments": int(summary_df.shape[0]),
    }