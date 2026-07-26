"""
Classification metric utilities.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_binary_classification_metrics(
    y_true,
    y_pred,
    y_score,
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Label convention:
        0 = healthy
        1 = sick
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity_healthy = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity_sick = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception:
        roc_auc = np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc),
        "precision_sick": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_sick": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity_sick": float(sensitivity_sick),
        "specificity_healthy": float(specificity_healthy),
        "f1_sick": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }