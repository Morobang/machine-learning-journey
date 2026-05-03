"""Missing value imputation helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer


def impute_median(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace NaNs with column medians.

    Median is robust to outliers — prefer over mean when skewed distributions
    are present (e.g. income, house prices).

    Args:
        df: Input DataFrame.
        columns: Columns to impute. If None, imputes all numeric columns.

    Returns:
        DataFrame with NaNs replaced.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    imp = SimpleImputer(strategy="median")
    df[cols] = imp.fit_transform(df[cols])
    return df


def impute_mean(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace NaNs with column means.

    Use when data is approximately normally distributed and outliers are absent.

    Args:
        df: Input DataFrame.
        columns: Columns to impute. If None, imputes all numeric columns.

    Returns:
        DataFrame with NaNs replaced.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    imp = SimpleImputer(strategy="mean")
    df[cols] = imp.fit_transform(df[cols])
    return df


def impute_constant(
    df: pd.DataFrame,
    columns: list[str],
    fill_value: str | int | float = "Unknown",
) -> pd.DataFrame:
    """Replace NaNs with a constant value.

    Useful for categorical columns where 'Unknown' or 0 is a meaningful category.

    Args:
        df: Input DataFrame.
        columns: Columns to impute.
        fill_value: Replacement value (default "Unknown").

    Returns:
        DataFrame with NaNs replaced.
    """
    df = df.copy()
    imp = SimpleImputer(strategy="constant", fill_value=fill_value)
    df[columns] = imp.fit_transform(df[columns])
    return df
