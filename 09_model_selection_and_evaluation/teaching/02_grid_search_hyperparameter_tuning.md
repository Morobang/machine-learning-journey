# Grid Search and Hyperparameter Tuning — Complete Guide

## Table of Contents
1. [What are Hyperparameters?](#what-are-hyperparameters)
2. [The Hyperparameter Tuning Problem](#the-hyperparameter-tuning-problem)
3. [Grid Search](#grid-search)
4. [Random Search](#random-search)
5. [Bayesian Optimisation](#bayesian-optimisation)
6. [Common Hyperparameters by Model](#common-hyperparameters-by-model)
7. [Advantages and Disadvantages](#advantages-and-disadvantages)
8. [Avoiding Data Leakage in Tuning](#avoiding-data-leakage-in-tuning)
9. [Real-World Applications](#real-world-applications)
10. [Implementation Steps](#implementation-steps)
11. [Common Pitfalls](#common-pitfalls)

---

## What are Hyperparameters?

A machine learning model has two kinds of parameters:

**Model parameters** — learned from data during training:
- Neural network weights
- Linear regression coefficients
- Decision tree split thresholds

**Hyperparameters** — set before training, not learned from data:
- Learning rate
- Number of trees in a Random Forest
- Regularisation strength (C in SVM, α in Ridge regression)
- Maximum tree depth
- Number of neighbours in KNN

**Why the distinction matters:** The training algorithm optimises model parameters. Hyperparameters control the training process itself — they define the model's inductive bias and capacity. Getting them wrong can mean the difference between a model that generalises and one that overfits or underfits.

---

## The Hyperparameter Tuning Problem

Given a model with hyperparameters θ, find the θ that maximises validation performance:

```
θ* = argmax_θ { ValidationScore(model(θ)) }
```

**The challenge:** We cannot use gradient descent to optimise hyperparameters because the validation score is often non-differentiable and expensive to evaluate (requires training a full model for each θ).

**The fundamental constraint:** You cannot use the test set during tuning. If you tune hyperparameters to maximise test set performance, the test set is no longer an unbiased estimate of real-world performance — it has been used to make modelling decisions.

**Three levels of data use:**
1. **Training set** — for fitting model parameters
2. **Validation set** (or CV folds) — for tuning hyperparameters
3. **Test set** — only for the final performance report, never for any decisions

---

## Grid Search

**Grid Search** (exhaustive search) tries every combination of hyperparameter values from a predefined grid.

### How It Works
1. Define a grid of hyperparameter values (e.g., C in {0.1, 1, 10, 100} × kernel in {'rbf', 'linear'})
2. For each combination, train the model with k-fold cross-validation
3. Select the combination with the best CV score
4. Optionally refit on the full training set with the best parameters

### Complexity
Grid search evaluates **n_hyperparameters combinations × k folds** models total.

Example: C ∈ {0.1, 1, 10, 100}, gamma ∈ {0.001, 0.01, 0.1} → 4 × 3 = 12 combinations × 5 folds = **60 model fits**.

**The curse of dimensionality:** For d hyperparameters each with n values, you train n^d × k models. Adding one more hyperparameter multiplies the search cost by n.

### When to Use Grid Search
- 1–3 hyperparameters with a small set of candidate values
- Fast-training models where 50–200 model fits are affordable
- When you want exhaustive coverage of the search space

---

## Random Search

**Random Search** samples hyperparameter combinations randomly from specified distributions rather than trying all combinations.

### Key Insight
If only a few hyperparameters are actually important for performance (a common finding), random search will find a good combination faster than grid search, because:

- Grid search wastes many evaluations on unimportant hyperparameter values
- Random search ensures good coverage of the truly important dimensions

**Bergstra and Bengio (2012)** showed that for many models, random search is more efficient than grid search per compute budget.

### When to Use Random Search
- More than 3 hyperparameters
- Large continuous hyperparameter ranges (e.g., learning rate from 1e-5 to 1.0)
- Limited compute budget — run as many evaluations as you can afford
- When you are unsure which hyperparameters matter most

---

## Bayesian Optimisation

**Bayesian Optimisation** uses the results of past evaluations to decide which hyperparameter combination to try next.

### The Idea
1. Build a **surrogate model** (typically a Gaussian Process) that predicts validation performance as a function of hyperparameters
2. Use the surrogate to identify the most promising next point to evaluate (using an **acquisition function** that balances exploitation and exploration)
3. Evaluate the actual model at that point
4. Update the surrogate
5. Repeat

### When to Use Bayesian Optimisation
- Expensive-to-train models (deep neural networks, large ensembles)
- When each evaluation takes minutes to hours
- When you have a compute budget of 20–100 evaluations
- Implementations: `scikit-optimize` (`BayesSearchCV`), `Optuna`, `Hyperopt`

---

## Common Hyperparameters by Model

### Support Vector Machine (SVM)

| Hyperparameter | Range | Effect |
|----------------|-------|--------|
| **C** (regularisation) | 0.001 to 10,000 (log scale) | Low C → large margin, more misclassifications; High C → smaller margin, fewer misclassifications, more overfitting |
| **kernel** | 'linear', 'rbf', 'poly' | Determines decision boundary shape |
| **gamma** (RBF) | 'scale', 'auto', 0.001–100 | Controls RBF kernel width; high gamma → complex boundary |

### Random Forest / Gradient Boosting

| Hyperparameter | Range | Effect |
|----------------|-------|--------|
| **n_estimators** | 100–1000 | More trees → more stable, diminishing returns |
| **max_depth** | 3–20 or None | Controls individual tree complexity; None → unlimited (overfits) |
| **min_samples_leaf** | 1–20 | Minimum samples per leaf; higher → more regularisation |
| **max_features** | 'sqrt', 'log2', 0.3–1.0 | Number of features per split; lower → more diverse trees |

### Logistic Regression / Ridge

| Hyperparameter | Range | Effect |
|----------------|-------|--------|
| **C** (Logistic) | 0.001 to 1000 (log scale) | Inverse of regularisation strength; lower C → stronger regularisation |
| **alpha** (Ridge) | 0.001 to 100 (log scale) | Regularisation strength; higher → more shrinkage |
| **solver** | 'lbfgs', 'saga', 'liblinear' | Optimisation algorithm; affects convergence speed |

### KNN

| Hyperparameter | Range | Effect |
|----------------|-------|--------|
| **n_neighbors** (k) | 1 to ~50 | Low k → complex, overfitting boundary; High k → smooth, underfitting |
| **metric** | 'euclidean', 'manhattan' | Distance function |
| **weights** | 'uniform', 'distance' | Whether to weight neighbours by distance |

---

## Advantages and Disadvantages

### Grid Search

| | Detail |
|-|--------|
| **Advantage** | Exhaustive — guaranteed to find the best combination in the grid |
| **Advantage** | Easily parallelisable (`n_jobs=-1`) |
| **Disadvantage** | Exponential in number of hyperparameters |
| **Disadvantage** | Assumes optimal values are in the predefined grid |

### Random Search

| | Detail |
|-|--------|
| **Advantage** | More efficient per evaluation when few hyperparameters matter |
| **Advantage** | Can handle continuous distributions |
| **Disadvantage** | No guarantee of finding the optimal combination |
| **Disadvantage** | Results vary between runs (set `random_state`) |

### Bayesian Optimisation

| | Detail |
|-|--------|
| **Advantage** | Most sample-efficient for expensive models |
| **Advantage** | Learns which hyperparameters matter most |
| **Disadvantage** | More complex to set up |
| **Disadvantage** | Less effective for cheap-to-train models (overhead is too high) |

---

## Avoiding Data Leakage in Tuning

### The Nested CV Pattern

For an unbiased final estimate when you tune hyperparameters:

```
Outer CV (5-fold):          [TEST] [TRAIN + VALIDATION]
                                        ↓
Inner CV (GridSearchCV):    [VALIDATION] — tunes hyperparameters
                                        ↓
Fit best hyperparameters on full [TRAIN + VALIDATION]
                                        ↓
Evaluate on [TEST]
```

This is expensive but gives an unbiased estimate of what the tuned model will achieve on new data.

**Simpler workflow for most projects:**
1. Set aside 20% as final test set (never touch until the end)
2. Use the remaining 80% for training + validation
3. Use GridSearchCV (which does inner CV) on the 80%
4. Evaluate on the held-out 20% once, at the very end

---

## Real-World Applications

**Production ML Systems:**
A deployed model needs tuned hyperparameters. Grid/Random search with CV is standard practice before deployment. The compute cost of hyperparameter tuning is amortised over the model's production lifetime.

**AutoML:**
Tools like auto-sklearn, H2O AutoML, and Google AutoML use Bayesian optimisation to automatically select algorithms and tune hyperparameters — effectively automating the model selection process.

**Neural Architecture Search (NAS):**
In deep learning, "hyperparameters" include the architecture itself (number of layers, layer sizes, skip connections). NAS uses RL or Bayesian methods to search the architecture space.

**Research Benchmarks:**
Fair comparison of algorithms requires each to be tuned to its best hyperparameters on the validation set. Reporting one algorithm at default settings vs another at tuned settings is not a fair comparison.

---

## Implementation Steps

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, loguniform
import numpy as np

# Build pipeline (ensure preprocessing is inside CV)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

# ── Grid Search ───────────────────────────────────────────────────────────────
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': ['scale', 0.01, 0.1]
}

grid_search = GridSearchCV(
    pipe, param_grid,
    cv=10,              # 10-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,          # Use all CPU cores
    verbose=1,          # Print progress
    refit=True          # Refit best model on full training set
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.score(X_test, y_test):.4f}")

# ── Random Search ─────────────────────────────────────────────────────────────
param_dist = {
    'svm__C': loguniform(0.001, 1000),      # Log-uniform from 0.001 to 1000
    'svm__gamma': loguniform(0.0001, 10),
    'svm__kernel': ['rbf', 'linear']
}

random_search = RandomizedSearchCV(
    pipe, param_dist,
    n_iter=50,          # Evaluate 50 random combinations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV accuracy: {random_search.best_score_:.4f}")

# ── Inspecting all results ────────────────────────────────────────────────────
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
top_results = results.nsmallest(5, 'rank_test_score')[
    ['param_svm__C', 'param_svm__kernel', 'mean_test_score', 'std_test_score']
]
print(top_results)
```

---

## Common Pitfalls

**1. Evaluating on the test set to choose hyperparameters**
Using the test set for any decision (including hyperparameter selection) makes it a de facto validation set. The test set is for the final report only — never use it during model development.

**2. Not using a Pipeline in GridSearchCV**
If you scale the data outside the CV loop, the scaler sees test fold data — leakage. Always include preprocessing steps in the Pipeline.

**3. Searching on a linear scale for parameters that vary over orders of magnitude**
C=1,2,3,4 is meaningless when C=0.01 and C=1000 behave completely differently. Use logarithmic grids: `[0.001, 0.01, 0.1, 1, 10, 100, 1000]` or `loguniform(0.001, 1000)` in random search.

**4. Over-tuning on a small dataset**
With 100 samples and 10 hyperparameters, you will find combinations that overfit to the validation splits. Cross-validation reduces but does not eliminate this. Use fewer hyperparameters and wider ranges on small datasets.

**5. Treating GridSearchCV's best_score_ as the model's true accuracy**
`best_score_` is the CV accuracy for the best hyperparameter combination. Due to selection bias (you chose the best out of many), it is slightly optimistic. For an honest estimate, evaluate on the held-out test set: `grid_search.score(X_test, y_test)`.

**6. Forgetting `refit=True`**
By default, `GridSearchCV` refits the best model on the full training set after search. This is what `grid_search.predict()` uses. If you set `refit=False`, the search object cannot make predictions.
