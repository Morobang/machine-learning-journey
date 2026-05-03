"""Model evaluation metric helpers.

Produces formatted, human-readable reports rather than raw numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "target",
) -> pd.DataFrame:
    """Print and return a formatted regression evaluation table.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.
        target_name: Label shown in output (default "target").

    Returns:
        Single-row DataFrame with MAE, MSE, RMSE, R².

    Example:
        report = regression_report(y_test, y_pred, target_name="Salary")
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    results = pd.DataFrame(
        {
            "MAE": [f"{mae:.4f}"],
            "MSE": [f"{mse:.4f}"],
            "RMSE": [f"{rmse:.4f}"],
            "R²": [f"{r2:.4f}"],
        },
        index=[target_name],
    )

    print(f"\n{'='*50}")
    print(f"Regression Report — {target_name}")
    print(f"{'='*50}")
    print(f"  MAE  (Mean Absolute Error):   {mae:.4f}")
    print(f"  RMSE (Root Mean Squared Err): {rmse:.4f}")
    print(f"  R²   (Coefficient of Det.):   {r2:.4f}")
    print(f"{'='*50}\n")
    return results


def classification_report_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> None:
    """Print a complete classification evaluation report.

    Includes accuracy, per-class precision/recall/F1, and confusion matrix.

    Args:
        y_true: Actual target labels.
        y_pred: Predicted labels.
        class_names: Display names for each class label.

    Example:
        classification_report_full(y_test, y_pred, class_names=["No Churn", "Churn"])
    """
    acc = accuracy_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print("Classification Report")
    print(f"{'='*50}")
    print(f"  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"{'='*50}\n")


def cross_val_report(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 10,
    scoring: str = "accuracy",
) -> pd.DataFrame:
    """Run k-fold cross-validation and return a summary table.

    Args:
        model: Fitted or unfitted sklearn estimator.
        X: Feature matrix.
        y: Target vector.
        cv: Number of folds (default 10).
        scoring: sklearn scoring metric (default "accuracy").

    Returns:
        DataFrame with mean, std, min, max of cross-val scores.

    Example:
        report = cross_val_report(RandomForestClassifier(), X, y, cv=10)
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    results = pd.DataFrame(
        {
            "CV Folds": [cv],
            "Metric": [scoring],
            "Mean": [f"{scores.mean():.4f}"],
            "Std": [f"{scores.std():.4f}"],
            "Min": [f"{scores.min():.4f}"],
            "Max": [f"{scores.max():.4f}"],
        }
    )

    print(f"\n{'='*50}")
    print(f"{cv}-Fold Cross-Validation — {scoring}")
    print(f"{'='*50}")
    print(f"  Mean ± Std:  {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Range:       [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  All scores:  {np.round(scores, 4)}")
    print(f"{'='*50}\n")
    return results
