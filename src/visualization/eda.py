"""Exploratory Data Analysis plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 30,
    figsize_per_col: tuple[int, int] = (5, 4),
) -> None:
    """Plot histogram + KDE for each numeric column.

    Reveals skewness, outliers, and whether data looks normally distributed —
    important for choosing preprocessing steps and algorithms.

    Args:
        df: Input DataFrame.
        columns: Specific columns to plot. If None, plots all numeric columns.
        bins: Number of histogram bins (default 30).
        figsize_per_col: (width, height) per subplot (default (5, 4)).
    """
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_col[0] * ncols, figsize_per_col[1] * nrows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (10, 8),
    annot: bool = True,
) -> None:
    """Plot a correlation heatmap for all numeric columns.

    High correlation between features (multicollinearity) can harm
    linear models. This plot helps identify redundant features.

    Args:
        df: Input DataFrame.
        title: Plot title.
        figsize: Figure size (default (10, 8)).
        annot: Show correlation values (default True).
    """
    corr = df.select_dtypes(include="number").corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))  # show only lower triangle

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_missing_values(
    df: pd.DataFrame,
    title: str = "Missing Values",
) -> None:
    """Bar chart showing the percentage of missing values per column.

    Only plots columns that have at least one missing value.
    Columns with 0% missing are omitted.

    Args:
        df: Input DataFrame.
        title: Plot title.
    """
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
    bars = ax.barh(missing.index, missing.values, color="coral", edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, val in zip(bars, missing.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_class_balance(
    y: pd.Series,
    title: str = "Class Balance",
    class_names: list[str] | None = None,
) -> None:
    """Bar chart showing class distribution for a classification target.

    Highly imbalanced classes (e.g. 95% / 5%) require special treatment:
    resampling (SMOTE), class weights, or alternative metrics (F1, AUC).

    Args:
        y: Target Series.
        title: Plot title.
        class_names: Display names for class labels.
    """
    counts = y.value_counts()
    pct = counts / len(y) * 100

    labels = class_names if class_names else [str(c) for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title(title)

    for bar, p in zip(bars, pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(y) * 0.005,
            f"{p:.1f}%",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_pairplot(
    df: pd.DataFrame,
    hue: str | None = None,
    columns: list[str] | None = None,
    title: str = "Pairplot",
) -> None:
    """Seaborn pairplot for exploring relationships between features.

    Args:
        df: Input DataFrame.
        hue: Column name to color points by (e.g. the target class).
        columns: Subset of columns to include. If None, uses all numeric.
        title: Title shown above the grid.
    """
    plot_df = df[columns].copy() if columns else df.select_dtypes(include="number").copy()
    if hue:
        plot_df[hue] = df[hue]

    g = sns.pairplot(plot_df, hue=hue, diag_kind="kde", plot_kws={"alpha": 0.5})
    g.fig.suptitle(title, y=1.01, fontsize=13, fontweight="bold")
    plt.show()
