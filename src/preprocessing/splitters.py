"""Train/test splitting helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into (X_train, X_test, y_train, y_test).

    Args:
        df: Input DataFrame containing features and target.
        target: Name of the target column.
        test_size: Fraction of rows reserved for testing (default 0.2).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        (X_train, X_test, y_train, y_test)

    Example:
        X_train, X_test, y_train, y_test = split_features(df, target="Salary")
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
