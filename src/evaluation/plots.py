"""Standard evaluation plots used across notebooks."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import learning_curve


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    units: str = "",
) -> None:
    """Scatter plot of actual vs predicted values with a perfect-fit reference line.

    Points clustering around the diagonal indicate good model fit.
    Systematic deviation above or below indicates bias.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        title: Plot title.
        units: Axis label units (e.g. "$", "kg", "ms").
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="steelblue", facecolors="none", s=50)

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")

    ax.set_xlabel(f"Actual {units}")
    ax.set_ylabel(f"Predicted {units}")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residuals Plot",
) -> None:
    """Plot residuals (errors) against predicted values.

    A good model's residuals are randomly scattered around zero with
    no clear pattern. Patterns indicate the model is missing structure
    (e.g. non-linearity, heteroscedasticity).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        title: Plot title.
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, color="steelblue", s=40)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Predicted")

    # Residual distribution
    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Residual value")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Plot a labelled confusion matrix heatmap.

    Args:
        y_true: Actual labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.
        title: Plot title.
        normalize: If True, show row-normalised proportions (default False).
    """
    norm = "true" if normalize else None
    cm = confusion_matrix(y_true, y_pred, normalize=norm)
    fmt = ".2f" if normalize else "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    estimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str = "ROC Curve",
) -> None:
    """Plot the ROC curve for a binary classifier.

    Args:
        estimator: Fitted sklearn classifier with predict_proba.
        X_test: Test features.
        y_test: True binary labels.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str = "Feature Importance",
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of feature importances (e.g. from Random Forest).

    Args:
        importances: Array of importance scores.
        feature_names: Name for each feature.
        title: Plot title.
        top_n: Show only the top N features (default 20).
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)))
    bars = ax.barh(range(top_n), top_importances[::-1], color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance score")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Learning Curve",
    cv: int = 5,
) -> None:
    """Plot training vs validation score as dataset size grows.

    Reveals underfitting (both scores low) or overfitting (large gap
    between training and validation scores).

    Args:
        estimator: Sklearn estimator (unfitted).
        X: Feature matrix.
        y: Target vector.
        title: Plot title.
        cv: Cross-validation folds (default 5).
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="steelblue")
    ax.plot(train_sizes, val_mean, "o-", color="darkorange", label="Validation score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="darkorange")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
