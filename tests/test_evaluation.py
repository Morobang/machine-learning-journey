"""Unit tests for src/evaluation metric helpers."""

import io
import sys

import numpy as np
import pandas as pd
import pytest

from src.evaluation import cross_val_report, regression_report
from src.evaluation.metrics import classification_report_full


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def perfect_regression():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return y, y.copy()  # (y_true, y_pred)


@pytest.fixture
def noisy_regression():
    np.random.seed(42)
    y_true = np.linspace(0, 10, 50)
    y_pred = y_true + np.random.normal(0, 1, 50)
    return y_true, y_pred


@pytest.fixture
def binary_classification():
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    return y_true, y_pred


# ─────────────────────────────────────────────
# regression_report
# ─────────────────────────────────────────────

class TestRegressionReport:
    def test_returns_dataframe(self, perfect_regression):
        y_true, y_pred = perfect_regression
        result = regression_report(y_true, y_pred)
        assert isinstance(result, pd.DataFrame)

    def test_perfect_predictions_r2_is_one(self, perfect_regression):
        y_true, y_pred = perfect_regression
        result = regression_report(y_true, y_pred)
        assert float(result["R²"].iloc[0]) == pytest.approx(1.0)

    def test_perfect_predictions_mae_is_zero(self, perfect_regression):
        y_true, y_pred = perfect_regression
        result = regression_report(y_true, y_pred)
        assert float(result["MAE"].iloc[0]) == pytest.approx(0.0)

    def test_noisy_r2_between_zero_and_one(self, noisy_regression):
        y_true, y_pred = noisy_regression
        result = regression_report(y_true, y_pred)
        r2 = float(result["R²"].iloc[0])
        assert 0.0 < r2 < 1.0

    def test_columns_present(self, noisy_regression):
        y_true, y_pred = noisy_regression
        result = regression_report(y_true, y_pred)
        assert set(result.columns) == {"MAE", "MSE", "RMSE", "R²"}


# ─────────────────────────────────────────────
# classification_report_full
# ─────────────────────────────────────────────

class TestClassificationReport:
    def test_runs_without_error(self, binary_classification):
        y_true, y_pred = binary_classification
        # Should print without raising
        classification_report_full(y_true, y_pred)

    def test_runs_with_class_names(self, binary_classification):
        y_true, y_pred = binary_classification
        classification_report_full(y_true, y_pred, class_names=["No", "Yes"])


# ─────────────────────────────────────────────
# cross_val_report
# ─────────────────────────────────────────────

class TestCrossValReport:
    def test_returns_dataframe(self):
        from sklearn.linear_model import LinearRegression

        X = np.random.rand(50, 2)
        y = X[:, 0] * 2 + np.random.rand(50) * 0.1

        result = cross_val_report(LinearRegression(), X, y, cv=3, scoring="r2")
        assert isinstance(result, pd.DataFrame)

    def test_correct_cv_folds_recorded(self):
        from sklearn.linear_model import LinearRegression

        X = np.random.rand(50, 2)
        y = X[:, 0] * 2

        result = cross_val_report(LinearRegression(), X, y, cv=5)
        assert int(result["CV Folds"].iloc[0]) == 5
