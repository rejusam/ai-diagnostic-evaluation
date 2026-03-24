"""
Equity-focused subgroup analysis for AI diagnostic evaluation.

Functions for stratifying AI tool performance by demographic variables,
testing for statistically significant differences, and quantifying
equity gaps relevant to the Aotearoa NZ context.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm
from . import metrics


def stratified_performance(
    df,
    group_col,
    y_true_col="has_referable_dr",
    y_pred_col="ai_prediction",
    y_score_col="ai_score",
):
    """
    Compute diagnostic accuracy metrics stratified by a grouping variable.

    Returns a DataFrame with one row per group, including sensitivity,
    specificity, PPV, NPV, AUC, and group size.
    """
    rows = []
    for group_val, sub in df.groupby(group_col):
        if sub[y_true_col].sum() == 0 or sub[y_true_col].sum() == len(sub):
            # Cannot compute meaningful metrics if all same class
            row = {"group": group_val, "n": len(sub), "n_positive": int(sub[y_true_col].sum())}
            rows.append(row)
            continue

        acc = metrics.diagnostic_accuracy(sub[y_true_col], sub[y_pred_col])
        try:
            roc = metrics.roc_analysis(sub[y_true_col], sub[y_score_col])
            auc_val = roc["auc"]
        except ValueError:
            auc_val = np.nan

        row = {
            "group": group_val,
            "n": acc["n"],
            "n_positive": acc["tp"] + acc["fn"],
            "prevalence": acc["prevalence"],
            "sensitivity": acc["sensitivity"],
            "specificity": acc["specificity"],
            "ppv": acc["ppv"],
            "npv": acc["npv"],
            "auc": auc_val,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def sensitivity_gap(strat_df, reference_group, metric="sensitivity"):
    """
    Compute the gap in a metric relative to a reference group.

    Positive values indicate the group performs worse (lower metric) than
    the reference. This is a simple descriptive measure.
    """
    ref_val = strat_df.loc[strat_df["group"] == reference_group, metric]
    if ref_val.empty:
        raise ValueError(f"Reference group '{reference_group}' not found")
    ref_val = ref_val.values[0]

    result = strat_df.copy()
    result[f"{metric}_gap"] = ref_val - result[metric]
    result[f"{metric}_relative_gap"] = (ref_val - result[metric]) / ref_val
    return result


def compare_sensitivity_two_groups(df, group_col, group_a, group_b,
                                    y_true_col="has_referable_dr",
                                    y_pred_col="ai_prediction"):
    """
    Test whether sensitivity differs between two groups using a
    two-proportion z-test on true positive rates.

    Returns the difference (a - b), z-statistic, and two-sided p-value.
    """
    sub_a = df[df[group_col] == group_a]
    sub_b = df[df[group_col] == group_b]

    pos_a = sub_a[sub_a[y_true_col] == True]
    pos_b = sub_b[sub_b[y_true_col] == True]

    n_a = len(pos_a)
    n_b = len(pos_b)

    if n_a == 0 or n_b == 0:
        return {"difference": np.nan, "z": np.nan, "p_value": np.nan}

    tp_a = (pos_a[y_pred_col] == True).sum()
    tp_b = (pos_b[y_pred_col] == True).sum()

    p_a = tp_a / n_a
    p_b = tp_b / n_b

    # Pooled proportion for the z-test
    p_pool = (tp_a + tp_b) / (n_a + n_b)

    if p_pool == 0 or p_pool == 1:
        return {"difference": p_a - p_b, "z": np.nan, "p_value": np.nan}

    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se
    p_val = 2 * norm.sf(np.abs(z))

    return {"difference": p_a - p_b, "z": z, "p_value": p_val}


def intersectional_analysis(
    df,
    group_cols,
    y_true_col="has_referable_dr",
    y_pred_col="ai_prediction",
    y_score_col="ai_score",
    min_group_size=30,
):
    """
    Compute performance for intersectional subgroups (e.g. ethnicity x NZDep).

    Filters out groups with fewer than min_group_size members to avoid
    unreliable estimates.
    """
    rows = []
    for keys, sub in df.groupby(group_cols):
        if len(sub) < min_group_size:
            continue
        if sub[y_true_col].sum() == 0 or sub[y_true_col].sum() == len(sub):
            continue

        acc = metrics.diagnostic_accuracy(sub[y_true_col], sub[y_pred_col])
        try:
            roc = metrics.roc_analysis(sub[y_true_col], sub[y_score_col])
            auc_val = roc["auc"]
        except ValueError:
            auc_val = np.nan

        if isinstance(keys, tuple):
            row = dict(zip(group_cols, keys))
        else:
            row = {group_cols[0]: keys}

        row.update(
            {
                "n": acc["n"],
                "n_positive": acc["tp"] + acc["fn"],
                "sensitivity": acc["sensitivity"],
                "specificity": acc["specificity"],
                "auc": auc_val,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def equity_summary(strat_df, metric="sensitivity"):
    """
    Produce a concise equity summary: range, ratio, and coefficient of
    variation across groups for a given metric.
    """
    vals = strat_df[metric].dropna()
    if len(vals) < 2:
        return {}

    return {
        "metric": metric,
        "min": vals.min(),
        "max": vals.max(),
        "range": vals.max() - vals.min(),
        "ratio_max_min": vals.max() / vals.min() if vals.min() > 0 else np.nan,
        "cv": vals.std() / vals.mean() if vals.mean() > 0 else np.nan,
        "worst_group": strat_df.loc[vals.idxmin(), "group"],
        "best_group": strat_df.loc[vals.idxmax(), "group"],
    }
