"""
Diagnostic accuracy metrics for AI screening evaluation.

Provides functions for computing standard diagnostic test performance
measures, ROC analysis, and calibration assessment.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve


def diagnostic_accuracy(y_true, y_pred):
    """
    Compute standard diagnostic accuracy metrics from binary predictions.

    Returns a dict with: sensitivity, specificity, ppv, npv, accuracy,
    true_positives, false_positives, true_negatives, false_negatives.
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fn = np.sum(y_true & ~y_pred)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else np.nan

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n": int(tp + fp + tn + fn),
        "prevalence": (tp + fn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else np.nan,
    }


def roc_analysis(y_true, y_score):
    """
    Compute ROC curve data and AUC.

    Returns a dict with fpr, tpr arrays and auc value.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def precision_recall_analysis(y_true, y_score):
    """Compute precision-recall curve and average precision."""
    y_true = np.asarray(y_true, dtype=int)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return {"precision": precision, "recall": recall, "thresholds": thresholds, "ap": ap}


def calibration_analysis(y_true, y_score, n_bins=10):
    """
    Assess calibration (reliability) of predicted probabilities.

    Returns bin midpoints, observed frequencies, and expected calibration error.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )

    # Expected calibration error (weighted by bin count)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_score, bin_edges[1:-1])
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    # Only use bins that have observations
    valid = bin_counts[: len(fraction_pos)] > 0
    if valid.any():
        ece = np.average(
            np.abs(fraction_pos[valid] - mean_predicted[valid]),
            weights=bin_counts[: len(fraction_pos)][valid],
        )
    else:
        ece = np.nan

    return {
        "fraction_positive": fraction_pos,
        "mean_predicted": mean_predicted,
        "ece": ece,
    }


def optimal_threshold(y_true, y_score, method="youden"):
    """
    Find the optimal classification threshold.

    Methods:
      'youden' — maximises Youden's J statistic (sensitivity + specificity - 1)
      'sensitivity_floor' — highest specificity while sensitivity >= 0.90
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    if method == "youden":
        j = tpr - fpr
        idx = np.argmax(j)
    elif method == "sensitivity_floor":
        # Find thresholds where sensitivity >= 0.90
        candidates = np.where(tpr >= 0.90)[0]
        if len(candidates) == 0:
            idx = np.argmax(tpr)
        else:
            # Among those, pick the one with lowest FPR (highest specificity)
            idx = candidates[np.argmin(fpr[candidates])]
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "threshold": thresholds[idx],
        "sensitivity": tpr[idx],
        "specificity": 1 - fpr[idx],
    }


def summary_table(df, y_true_col="has_referable_dr", y_pred_col="ai_prediction"):
    """Create a one-row summary of diagnostic accuracy for a DataFrame."""
    result = diagnostic_accuracy(df[y_true_col], df[y_pred_col])
    return pd.DataFrame([result])
