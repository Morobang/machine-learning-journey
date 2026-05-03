"""Unit tests for src/preprocessing utilities."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    encode_labels,
    impute_constant,
    impute_mean,
    impute_median,
    one_hot_encode,
    scale_features,
    scale_features_robust,
    split_features,
)


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Age": [25.0, 30.0, np.nan, 40.0, 35.0],
            "Salary": [50000, 60000, 70000, 80000, 90000],
            "Country": ["France", "Spain", "Germany", "France", "Spain"],
            "Purchased": [0, 1, 0, 1, 1],
        }
    )


# ─────────────────────────────────────────────
# split_features
# ─────────────────────────────────────────────

class TestSplitFeatures:
    def test_returns_four_parts(self, sample_df):
        result = split_features(sample_df, target="Purchased")
        assert len(result) == 4

    def test_correct_shapes(self, sample_df):
        X_train, X_test, y_train, y_test = split_features(sample_df, target="Purchased", test_size=0.2)
        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(y_train) + len(y_test) == len(sample_df)

    def test_target_not_in_features(self, sample_df):
        X_train, X_test, y_train, y_test = split_features(sample_df, target="Purchased")
        assert "Purchased" not in X_train.columns
        assert "Purchased" not in X_test.columns


# ─────────────────────────────────────────────
# scale_features
# ─────────────────────────────────────────────

class TestScaleFeatures:
    def test_standard_scaler_zero_mean(self):
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_test = np.array([[2.0, 3.0]])
        X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
        np.testing.assert_allclose(X_train_sc.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_returns_scaler_object(self):
        X = np.random.rand(20, 3)
        _, _, scaler = scale_features(X, X)
        assert hasattr(scaler, "transform")

    def test_no_leakage(self):
        X_train = np.array([[10.0], [20.0], [30.0]])
        X_test = np.array([[1000.0]])  # extreme outlier in test set
        X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)
        # Test set gets transformed using train statistics, so extreme value
        # remains extreme (not normalized to 0-1 range)
        assert abs(X_test_sc[0, 0]) > 1.0


# ─────────────────────────────────────────────
# impute_median
# ─────────────────────────────────────────────

class TestImputation:
    def test_impute_median_fills_nan(self, sample_df):
        result = impute_median(sample_df, columns=["Age"])
        assert result["Age"].isnull().sum() == 0

    def test_impute_median_does_not_modify_original(self, sample_df):
        _ = impute_median(sample_df, columns=["Age"])
        assert sample_df["Age"].isnull().sum() == 1  # original unchanged

    def test_impute_mean_fills_nan(self, sample_df):
        result = impute_mean(sample_df, columns=["Age"])
        assert result["Age"].isnull().sum() == 0

    def test_impute_constant_fills_categorical(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Country"] = None
        result = impute_constant(df, columns=["Country"], fill_value="Unknown")
        assert result["Country"].isnull().sum() == 0
        assert "Unknown" in result["Country"].values


# ─────────────────────────────────────────────
# encode_labels
# ─────────────────────────────────────────────

class TestEncoders:
    def test_encode_labels_returns_integers(self, sample_df):
        encoded, enc = encode_labels(sample_df["Country"])
        assert encoded.dtype in [np.int64, np.int32, int]

    def test_encode_labels_invertible(self, sample_df):
        encoded, enc = encode_labels(sample_df["Country"])
        decoded = pd.Series(enc.inverse_transform(encoded))
        pd.testing.assert_series_equal(
            sample_df["Country"].reset_index(drop=True),
            decoded,
        )

    def test_one_hot_encode_creates_dummies(self, sample_df):
        result = one_hot_encode(sample_df, columns=["Country"])
        assert "Country" not in result.columns
        # With drop_first=True, 3 categories → 2 dummy columns
        country_cols = [c for c in result.columns if "Country" in c]
        assert len(country_cols) == 2
