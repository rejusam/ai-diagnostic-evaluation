"""
Plotting utilities for the AI DR screening evaluation.

Uses matplotlib and seaborn with a consistent style appropriate
for academic outputs and stakeholder reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# Consistent palette — colourblind-friendly, works in greyscale
ETHNICITY_PALETTE = {
    "nz_european": "#4477AA",
    "maori": "#EE6677",
    "pacific": "#228833",
    "asian": "#CCBB44",
    "other": "#AA3377",
}

ETHNICITY_LABELS = {
    "nz_european": "NZ European",
    "maori": "Māori",
    "pacific": "Pacific peoples",
    "asian": "Asian",
    "other": "Other",
}

NZDEP_PALETTE = sns.color_palette("YlOrRd", 5)


def set_style():
    """Apply a clean, publication-ready plot style."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _eth_label(eth):
    return ETHNICITY_LABELS.get(eth, eth)


def plot_roc_curves(roc_results, title="ROC Curves", ax=None, save_path=None):
    """
    Plot ROC curves for one or more AI tools.

    roc_results : dict mapping label -> {fpr, tpr, auc}
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for label, data in roc_results.items():
        ax.plot(
            data["fpr"],
            data["tpr"],
            label=f'{label} (AUC = {data["auc"]:.3f})',
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    if save_path:
        plt.savefig(save_path)
    return ax


def plot_calibration(cal_results, title="Calibration Plot", ax=None, save_path=None):
    """
    Plot calibration (reliability) diagram.

    cal_results : dict mapping label -> {mean_predicted, fraction_positive, ece}
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    for label, data in cal_results.items():
        ax.plot(
            data["mean_predicted"],
            data["fraction_positive"],
            "o-",
            label=f'{label} (ECE = {data["ece"]:.3f})',
            linewidth=2,
            markersize=6,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    if save_path:
        plt.savefig(save_path)
    return ax


def plot_stratified_metric(
    strat_df,
    metric="sensitivity",
    group_col="group",
    palette=None,
    title=None,
    ax=None,
    save_path=None,
):
    """Bar chart of a metric by subgroup."""
    set_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    groups = strat_df[group_col].values
    labels = [_eth_label(g) if isinstance(g, str) else str(g) for g in groups]
    values = strat_df[metric].values

    if palette is None and all(g in ETHNICITY_PALETTE for g in groups):
        colours = [ETHNICITY_PALETTE[g] for g in groups]
    elif palette is not None:
        colours = palette
    else:
        colours = sns.color_palette("muted", len(groups))

    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel(metric.replace("_", " ").title())
    if title:
        ax.set_title(title)
    ax.set_ylim([0, min(max(values) * 1.15, 1.05)])

    if save_path:
        plt.savefig(save_path)
    return ax


def plot_equity_heatmap(
    intersect_df,
    row_col="ethnicity",
    col_col="nzdep_quintile",
    value_col="sensitivity",
    title=None,
    save_path=None,
):
    """
    Heatmap of a metric across two grouping dimensions (e.g. ethnicity x NZDep).
    """
    set_style()

    pivot = intersect_df.pivot_table(
        index=row_col, columns=col_col, values=value_col, aggfunc="mean"
    )

    # Relabel ethnicity rows
    pivot.index = [_eth_label(e) for e in pivot.index]
    if col_col == "nzdep_quintile":
        pivot.columns = [f"Q{int(c)}" for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        vmin=pivot.values[~np.isnan(pivot.values)].min() * 0.95,
        vmax=pivot.values[~np.isnan(pivot.values)].max() * 1.02,
        cbar_kws={"label": value_col.replace("_", " ").title()},
    )

    ax.set_ylabel("")
    ax.set_xlabel("NZDep Quintile" if col_col == "nzdep_quintile" else col_col)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return ax


def plot_gap_chart(
    gap_df,
    metric="sensitivity",
    group_col="group",
    title=None,
    save_path=None,
):
    """
    Horizontal bar chart showing the gap in a metric relative to the
    reference group. Useful for highlighting equity differences at a glance.
    """
    set_style()
    gap_col = f"{metric}_gap"
    if gap_col not in gap_df.columns:
        raise ValueError(f"Column '{gap_col}' not found — run sensitivity_gap first.")

    fig, ax = plt.subplots(figsize=(8, 4))
    groups = gap_df[group_col].values
    labels = [_eth_label(g) for g in groups]
    gaps = gap_df[gap_col].values

    colours = ["#228833" if g <= 0 else "#EE6677" for g in gaps]
    ax.barh(labels, gaps, color=colours, edgecolor="white")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{metric.title()} gap (reference group − subgroup)")
    if title:
        ax.set_title(title)

    for i, (val, label) in enumerate(zip(gaps, labels)):
        if not np.isnan(val):
            ax.text(
                val + 0.002 * np.sign(val),
                i,
                f"{val:+.3f}",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return ax
