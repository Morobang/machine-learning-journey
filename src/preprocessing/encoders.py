"""Categorical encoding helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def encode_labels(series: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Encode a single categorical Series to integer labels.

    Suitable for binary or target-variable encoding.
    For multi-class features, prefer one_hot_encode.

    Args:
        series: Categorical pandas Series.

    Returns:
        (encoded_series, fitted_encoder)

    Example:
        y_enc, enc = encode_labels(df["Gender"])
        # Inverse: enc.inverse_transform(y_enc)
    """
    enc = LabelEncoder()
    encoded = pd.Series(enc.fit_transform(series), name=series.name, index=series.index)
    return encoded, enc


def encode_ordinal(
    df: pd.DataFrame,
    columns: list[str],
    categories: list[list] | None = None,
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Encode ordinal categorical columns (order matters).

    Args:
        df: Input DataFrame.
        columns: Columns to encode.
        categories: Explicit category order per column, e.g.
                    [["Low", "Medium", "High"]].
                    If None, order is inferred from data.

    Returns:
        (df_encoded, fitted_encoder)
    """
    enc = OrdinalEncoder(categories=categories or "auto")
    df = df.copy()
    df[columns] = enc.fit_transform(df[columns])
    return df, enc


def one_hot_encode(
    df: pd.DataFrame,
    columns: list[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """Apply one-hot encoding to specified columns.

    Args:
        df: Input DataFrame.
        columns: Columns to one-hot encode.
        drop_first: Drop the first dummy column to avoid multicollinearity
                    (default True — required for linear models).

    Returns:
        DataFrame with original columns replaced by dummies.

    Example:
        df_enc = one_hot_encode(df, columns=["Country", "State"])
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)
