# Time Series Fundamentals

## What Makes Time Series Different

A standard ML dataset assumes rows are **independent and identically distributed (i.i.d.)**.
Time series data violates this completely: each observation depends on the ones before it.

This single fact has cascading consequences for how you prepare data, validate models, and interpret results.

---

## The Four Components

Every time series can be decomposed into four components:

### 1. Trend
The long-term direction of the data — increasing, decreasing, or flat.

```
Sales over 5 years:

  |        /
  |       /
  |      /
  |     /
  |____/___________
        time
```

Example: Global average temperature has an upward trend over decades.

### 2. Seasonality
**Regular**, **fixed-period** patterns that repeat.

```
Monthly ice cream sales:
  |  *       *       *
  | * *     * *     * *
  |*   *   *   *   *   *
  |     * *     * *
  |________________________
   Jan           Dec  Jan
```

Seasonality has a fixed period (e.g. every 12 months, every 7 days).
If the period varies, it is a cyclical pattern, not seasonality.

### 3. Cyclical Patterns
Long-term fluctuations with **no fixed period** — typically driven by economic or business cycles.

Example: GDP growth tends to cycle between expansion and recession, but the periods are irregular (2 years, 7 years, etc.).

**Key distinction:** Seasonality has a fixed known period. Cycles do not.

### 4. Noise (Residual)
What remains after removing trend, seasonality, and cycles. Ideally pure random variation.

If there is still a pattern in the residuals after decomposition, your model is missing something.

---

## Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time.

```
Non-stationary (trending):        Stationary:
  |     /                           |~~~~~
  |    /                            |  ~  ~  ~
  |   /                             |~  ~  ~
  |  /                              |
  |__________                       |___________
```

Most classical forecasting methods (ARIMA) require stationary data.
Tree-based methods (XGBoost for time series) are more flexible but still benefit from stationarity.

### How to Test for Stationarity

**Visual:** Plot the rolling mean and rolling standard deviation. Are they constant?

**Statistical:** Augmented Dickey-Fuller (ADF) test:
- H₀: The series has a unit root (non-stationary)
- If p-value < 0.05: reject H₀ → likely stationary

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f"ADF statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# p < 0.05 → stationary
```

### How to Achieve Stationarity

| Transform | Use when |
|-----------|---------|
| **Differencing** `y_t - y_{t-1}` | Removes trend (most common) |
| **Seasonal differencing** `y_t - y_{t-12}` | Removes monthly seasonality |
| **Log transform** `log(y_t)` | Stabilises exponentially growing variance |
| **Percentage change** `(y_t - y_{t-1})/y_{t-1}` | Normalises scale differences |

---

## Autocorrelation

Autocorrelation measures how correlated a series is with its own past values.

$$\text{ACF}(k) = \text{corr}(y_t, y_{t-k})$$

Where $k$ is the **lag**.

**ACF plot:** Shows correlation at each lag. If ACF decays slowly, the series has a trend or is non-stationary.

**PACF (Partial ACF):** Shows correlation at lag $k$ after removing the effect of lags 1 through $k-1$.

```
ACF plot of a stationary series:
  1.0 |*
  0.8 |  *
  0.6 |    *
  0.4 |      *
  0.2 |        * *
  0.0 |____________* * * * * *
       1  2  3  4  5  6  7  8   lag
```

Significant bars outside the confidence band (the shaded region in `statsmodels`) indicate useful lags for modelling.

### Using ACF/PACF to Select ARIMA Parameters

| Pattern | Suggests |
|---------|---------|
| ACF tails off, PACF cuts off at lag $p$ | AR($p$) model |
| ACF cuts off at lag $q$, PACF tails off | MA($q$) model |
| Both tail off slowly | ARMA($p$,$q$) |
| Both cut off sharply | Differencing needed first |

---

## Validation: The Critical Difference From Standard ML

**Never use random k-fold cross-validation on time series.**

Standard k-fold randomly shuffles rows into folds. This means a model trained on "tomorrow's data" predicts "yesterday's data" — look-ahead bias.

### Time Series Cross-Validation (Walk-Forward)

```
Fold 1:  [Train: 1–100]  → [Test: 101–110]
Fold 2:  [Train: 1–110]  → [Test: 111–120]
Fold 3:  [Train: 1–120]  → [Test: 121–130]
...
```

Each fold: train on everything up to point $t$, predict the next window. Never use future data to train.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

---

## Forecasting Evaluation Metrics

| Metric | Formula | Use when |
|--------|---------|---------|
| **MAE** | $\frac{1}{n}\sum \|y_t - \hat{y}_t\|$ | Interpretable; in original units |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_t - \hat{y}_t)^2}$ | Penalises large errors more |
| **MAPE** | $\frac{100}{n}\sum\|\frac{y_t - \hat{y}_t}{y_t}\|$ | Scale-independent % error |
| **SMAPE** | $\frac{200}{n}\sum\frac{\|y_t - \hat{y}_t\|}{|y_t| + |\hat{y}_t|}$ | Symmetric MAPE; better when values near zero |

**Kaggle note:** Many forecasting competitions use RMSLE (log-scale RMSE) to penalise relative errors equally across scale.

---

## Feature Engineering for Time Series (ML Approach)

When using tree-based models (XGBoost, LightGBM) instead of classical methods:

**Lag features** — the most powerful:
```python
for lag in [1, 7, 14, 28]:
    df[f"sales_lag_{lag}"] = df["sales"].shift(lag)
```

**Rolling statistics:**
```python
df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
df["rolling_std_7"]  = df["sales"].shift(1).rolling(7).std()
```

**Calendar features:**
```python
df["day_of_week"] = df["date"].dt.dayofweek
df["month"]       = df["date"].dt.month
df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
df["is_holiday"]  = df["date"].isin(holiday_list).astype(int)
```

**The `.shift(1)` rule:** When computing lag or rolling features, always shift by at least 1 to avoid using the current value to predict itself (data leakage).

---

## Common Mistakes

**Mistake 1: Using random k-fold CV**
Your error metrics look great but the model is cheating by using future data in training. Always use `TimeSeriesSplit` or a held-out final time window.

**Mistake 2: Forgetting to check for stationarity before ARIMA**
ARIMA's `d` parameter (differencing order) must make the series stationary. Using `d=0` on a trending series will give garbage forecasts.

**Mistake 3: Computing rolling features without shifting**
`df["rolling_mean"] = df["sales"].rolling(7).mean()` leaks the current day's sale into the feature. Always `.shift(1)` first.

**Mistake 4: Evaluating on the training period**
Fitting an ARIMA and plotting in-sample predictions looks perfect — because the model memorised the training data. Always evaluate on a held-out future period.

---

## See Also

- Notebook: `../notebooks/01_time_series_fundamentals.ipynb`
- Notebook: `../notebooks/02_arima_and_exponential_smoothing.ipynb`
- [07_time_series_analysis/README.md](../README.md) — full algorithm overview