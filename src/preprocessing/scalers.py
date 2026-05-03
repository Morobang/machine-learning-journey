"""Feature scaling helpers.

StandardScaler: zero mean, unit variance — best for most algorithms.
RobustScaler: uses median/IQR — better when outliers are present.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on X_train and apply to both sets.

    Fitting only on X_train prevents data leakage from the test set.

    Args:
        X_train: Training features (array or DataFrame).
        X_test: Test features (array or DataFrame).

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)

    Example:
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def scale_features_robust(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    """Fit RobustScaler on X_train and apply to both sets.

    Use this when the dataset contains significant outliers that would
    distort StandardScaler's mean-based normalization.

    Args:
        X_train: Training features.
        X_test: Test features.

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
