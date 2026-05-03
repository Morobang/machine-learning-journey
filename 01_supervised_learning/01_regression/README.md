# Regression

Regression is the branch of supervised learning concerned with predicting **continuous numerical outputs**. Given a set of input features, the model learns to output a number — a price, a temperature, a score, a count — rather than a category.

The name comes from Francis Galton's 19th-century observation that extreme values in one generation tend to "regress" toward the average in the next. Today the term broadly means any model that predicts a continuous target.

---

## What Makes a Problem a Regression Problem

A problem is a regression problem when:
- The output is a real number (or can be meaningfully treated as one)
- Intermediate values are valid — predicting £275,000 for a house is as valid as predicting £250,000 or £300,000
- The goal is to predict *how much* rather than *which category*

If instead you want to predict *which bucket* a value falls into (low/medium/high salary, pass/fail), that is a classification problem even if the underlying variable is numeric.

---

## Algorithms in This Section

### Simple Linear Regression
**Notebook:** [01_simple_linear_regression.ipynb](notebooks/01_simple_linear_regression.ipynb) | **Guide:** [teaching/01_simple_linear_regression.md](teaching/01_simple_linear_regression.md)

Models the relationship between one feature and the target as a straight line:

```
y = b₀ + b₁x
```

The model finds the line that minimises the sum of squared residuals (ordinary least squares). Simple, interpretable, and the foundation for understanding all other regression methods.

**Use when:** You have one continuous predictor and want to understand the direction and strength of the linear relationship. Always start here as a baseline.

**Limitation:** One feature only; assumes the relationship is exactly linear.

---

### Multiple Linear Regression
**Notebook:** [02_multiple_linear_regression.ipynb](notebooks/02_multiple_linear_regression.ipynb) | **Guide:** [teaching/02_multiple_linear_regression.md](teaching/02_multiple_linear_regression.md)

Extends simple linear regression to multiple features:

```
y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

Each coefficient bᵢ represents the change in y for a one-unit increase in xᵢ, holding all other features constant. This "all else equal" interpretation is what makes MLR useful for understanding individual feature contributions.

**Use when:** You have multiple continuous or encoded categorical predictors and the relationships are approximately linear. The most widely used regression model in statistics and business.

**Watch out for:** Multicollinearity — when two features are highly correlated, their individual coefficients become unreliable even if the model's overall predictions are good.

---

### Polynomial Regression
**Notebook:** [03_polynomial_regression.ipynb](notebooks/03_polynomial_regression.ipynb) | **Guide:** [teaching/03_polynomial_regression.md](teaching/03_polynomial_regression.md)

Captures curved (nonlinear) relationships by adding polynomial terms of the features:

```
y = b₀ + b₁x + b₂x² + b₃x³ + ...
```

The trick: polynomial regression is still a linear model — linear in the coefficients. We use `PolynomialFeatures` to generate x², x³ terms, then fit a standard linear model. The nonlinearity is in the *features*, not the model.

**Use when:** A scatter plot shows a clear curve between x and y (acceleration curves, dose-response relationships, growth curves).

**Watch out for:** High-degree polynomials wildly extrapolate beyond the training data range and overfit small datasets. Stick to degree 2 or 3 in practice.

---

### Support Vector Regression (SVR)
**Notebook:** [04_support_vector_regression.ipynb](notebooks/04_support_vector_regression.ipynb) | **Guide:** [teaching/04_support_vector_regression.md](teaching/04_support_vector_regression.md)

Fits the data within a tolerance tube of width ε. Predictions within ε of the true value incur no penalty. Only points outside the tube (support vectors) influence the model. The kernel trick allows nonlinear SVR without explicitly computing the nonlinear feature expansion.

**Use when:** Your data has outliers (the ε-tube makes SVR robust to them), or you need a nonlinear regression model and the dataset is small to medium sized.

**Watch out for:** SVR requires feature scaling (mandatory — the kernel uses distances), and the hyperparameters C, ε, and γ require tuning.

---

### Decision Tree Regression
**Notebook:** [05_decision_tree_regression.ipynb](notebooks/05_decision_tree_regression.ipynb) | **Guide:** [teaching/05_decision_tree_regression.md](teaching/05_decision_tree_regression.md)

Recursively partitions the feature space with axis-aligned splits. Each leaf node predicts the mean of all training samples that fell into it. The result is a step function — prediction is constant within each region.

**Use when:** You need an interpretable model (the tree can be printed and explained), the relationship involves thresholds and segments ("above this salary and below this age, price is X"), or features are a mix of numerical and categorical.

**Watch out for:** A fully grown tree perfectly memorises training data (variance ≈ infinity). Always set `max_depth` or `min_samples_leaf`. A single tree also has high instability — small data changes cause large structural changes.

---

### Random Forest Regression
**Notebook:** [06_random_forest_regression.ipynb](notebooks/06_random_forest_regression.ipynb) | **Guide:** [teaching/06_random_forest_regression.md](teaching/06_random_forest_regression.md)

An ensemble of decision trees trained on bootstrapped samples of the data, with random feature subsets at each split. Predictions are the average across all trees. Averaging reduces the variance that makes individual trees unreliable.

**Use when:** You want strong out-of-the-box predictive accuracy without heavy feature engineering. Random Forest handles missing values, nonlinear relationships, feature interactions, and mixed feature types well. It also provides feature importance scores.

**Watch out for:** Less interpretable than a single tree, slower to train, and tends to extrapolate poorly beyond the training range (predicts the mean of the nearest training region, not a trend).

---

## Choosing a Regression Algorithm

```
Start here: Is the relationship approximately linear?
├── Yes → Multiple Linear Regression (fast, interpretable, widely understood)
│         Check residuals — if they show a pattern, the linear assumption is violated
└── No → What shape is the relationship?
         ├── Polynomial curve → Polynomial Regression (degree 2 or 3)
         ├── Complex nonlinear → SVR (small/medium data) or Random Forest (larger data)
         └── Threshold-based segments → Decision Tree or Random Forest
```

When in doubt, fit a Random Forest first. It works well across a wide range of data types and gives a feature importance ranking that guides further analysis.

---

## Evaluation Metrics for Regression

All regression metrics compare the predicted value ŷ to the actual value y:

| Metric | Formula | When to use |
|--------|---------|-------------|
| **MAE** | mean(|y − ŷ|) | When outlier errors should not be penalised more than small errors |
| **RMSE** | √mean((y − ŷ)²) | Standard choice; penalises large errors — good when large errors are costly |
| **R²** | 1 − SS_res/SS_tot | Measures fraction of variance explained; 0 = predicts the mean, 1 = perfect |
| **Adjusted R²** | Penalises extra features | Use instead of R² when comparing models with different numbers of features |

**R² is scale-free** (always between −∞ and 1), making it easy to compare across datasets. RMSE is in the same units as the target, making it easy to interpret in business terms ("our predictions are off by £8,000 on average").

---

## What Each Teaching Guide Contains

Each algorithm in [teaching/](teaching/) has a complete guide covering:
- The intuition behind how the algorithm works
- The mathematical objective it minimises (with equations)
- A worked example with actual numbers
- When to use it and when not to
- The key hyperparameters and how to tune them
- Real-world applications
- Common pitfalls and how to avoid them

The notebooks in [notebooks/](notebooks/) implement each algorithm step by step and reference the teaching guides for deeper explanation.
