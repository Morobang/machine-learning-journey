# Time Series Analysis

Time series analysis deals with data collected sequentially over time — stock prices recorded daily, temperature measured hourly, monthly sales revenue, quarterly GDP figures. The defining characteristic: **the order of observations matters**. Yesterday's temperature is relevant to today's; last month's sales inform this month's.

This temporal dependence is what separates time series from standard supervised learning, and it changes almost every aspect of how you work with the data: how you split train and test sets, how you cross-validate, which models to use, and how you evaluate forecasts.

---

## What Makes Time Series Different

**You cannot shuffle the data.** In standard machine learning, rows are independent samples. In time series, each observation depends on previous ones. Shuffling destroys that structure — a model trained on shuffled time series is learning from future data to predict the past.

**Train/test split respects time.** The test set must always be the most recent period. Never randomly assign rows to train and test.

**Cross-validation uses a walk-forward approach:**
```
Fold 1: Train [t=1..100]  → Test [t=101..120]
Fold 2: Train [t=1..120]  → Test [t=121..140]
Fold 3: Train [t=1..140]  → Test [t=141..160]
```
Each fold's test set is always after the training set, simulating the real forecasting task.

---

## The Four Components of a Time Series

Understanding these components is the first step in choosing the right model:

**Trend** — the long-term direction. Sales growing 10% per year. Population increasing. A trend means the series is non-stationary (its mean changes over time).

**Seasonality** — regular, calendar-driven patterns. Ice cream sales peak every summer. Website traffic spikes every Monday morning. Seasonality repeats with a known period.

**Cyclical patterns** — irregular fluctuations with no fixed period. Economic expansion and contraction. These last years, not months, and their timing is not predictable from the calendar.

**Irregular/residual** — what's left after removing trend, seasonality, and cycles. Random noise, one-off events, measurement error.

Decomposing a series into these components both helps you understand it and guides model selection: a series with strong seasonality needs a model that can represent seasonal patterns.

---

## Stationarity

Most classical time series models (ARIMA, exponential smoothing) assume the data is **stationary** — its statistical properties (mean, variance, autocorrelation) do not change over time.

Most real series are **non-stationary** because of trend or seasonality. Standard transformations to achieve stationarity:

- **Differencing** — subtract the previous value from each observation. One round of differencing removes a linear trend. Seasonal differencing (subtract the value from the same period last year) removes seasonality.
- **Log transformation** — stabilises variance when the scale of fluctuations grows with the level (common in financial data).
- **Detrending** — fit and subtract a trend line.

After modelling, all transformations must be reversed to produce forecasts in the original scale.

---

## Forecasting Methods

The teaching guide covers the full spectrum from simplest to most complex:

### Baseline Methods
Always compute these first. A forecast is only useful if it beats the baseline.

- **Naive forecast** — next value = last observed value
- **Seasonal naive** — next value = value from the same period last year
- **Moving average** — next value = mean of the last k observations

### Exponential Smoothing
Weighted averages where recent observations receive exponentially more weight than older ones. Three variants handle progressively more complex patterns:

- **Simple Exponential Smoothing** — for series with no trend, no seasonality
- **Holt's Linear** — adds a trend component
- **Holt-Winters** — adds both trend and seasonality

No ARIMA identification needed; parameters are estimated by minimising forecast errors.

### ARIMA
**AutoRegressive Integrated Moving Average** — the classical statistical forecasting model for univariate stationary series:

```
ARIMA(p, d, q)
  p = AR order: how many past values to include
  d = differencing order: how many times to difference for stationarity
  q = MA order: how many past forecast errors to include
```

The Box-Jenkins methodology uses ACF (autocorrelation function) and PACF (partial ACF) plots to identify appropriate p and q values. `auto_arima` from the `pmdarima` package automates this selection.

### Machine Learning Approaches
For series with complex patterns, external predictors (temperature, promotions, holidays), or multiple related series, machine learning often outperforms classical methods:

- **Feature engineering** converts the time index into lag features, rolling statistics, and calendar features (day of week, month, holiday indicator)
- Any regression algorithm (Random Forest, XGBoost, LightGBM) then treats it as a standard regression task
- **Advantage:** can incorporate external variables naturally
- **Limitation:** does not model time as a continuous process; requires careful feature engineering to capture temporal patterns

### Deep Learning (LSTM)
Long Short-Term Memory networks are recurrent neural networks designed for sequential data. They maintain a hidden state that carries information across many time steps, learning long-range dependencies that ARIMA cannot.

Covered theoretically here; implementation requires the deep learning foundations from [04_deep_learning](../04_deep_learning/).

---

## Forecasting Evaluation Metrics

Unlike classification or regression, time series evaluation must account for the **forecast horizon** (how far ahead are we predicting?) and the **baseline difficulty** of the series.

| Metric | Formula | Note |
|--------|---------|------|
| **MAE** | mean(|yₜ − ŷₜ|) | Same units as the series; interpretable |
| **RMSE** | √mean((yₜ − ŷₜ)²) | Penalises large errors more; useful when spikes are costly |
| **MAPE** | mean(|yₜ − ŷₜ| / |yₜ|) × 100 | Percentage error; undefined when yₜ = 0 |
| **SMAPE** | mean(2|yₜ − ŷₜ| / (|yₜ| + |ŷₜ|)) | Symmetric version of MAPE; handles zeros better |

**Always compare against a baseline.** A model with MAPE = 8% looks good — until you notice the naive forecast achieves MAPE = 7%. A forecast that does not beat the naive baseline provides no value.

---

## Current Status

Notebooks for this section are in development. The teaching guide covers all concepts in full:

**Guide:** [teaching/01_time_series_fundamentals.md](teaching/01_time_series_fundamentals.md) — time series components, stationarity tests (ADF test), decomposition, ACF/PACF interpretation, the full spectrum of forecasting methods, cross-validation strategies, and evaluation.

---

## Relationship to the Rest of This Repository

Time series analysis draws on techniques from across the repo:
- **Feature scaling** from [00_foundations](../00_foundations/) applies when using ML approaches
- **Regression models** from [01_supervised_learning/01_regression](../01_supervised_learning/01_regression/) are the base algorithms when using ML for forecasting
- **Cross-validation strategy** from [09_model_selection_and_evaluation](../09_model_selection_and_evaluation/) must be adapted to the walk-forward scheme
- **LSTM** builds directly on the ANN foundations in [04_deep_learning](../04_deep_learning/)
