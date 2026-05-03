# k-Fold Cross-Validation — Complete Guide

## Table of Contents
1. [The Problem with a Single Train/Test Split](#the-problem-with-a-single-traintest-split)
2. [What is k-Fold Cross-Validation?](#what-is-k-fold-cross-validation)
3. [How k-Fold Works](#how-k-fold-works)
4. [Variants of Cross-Validation](#variants-of-cross-validation)
5. [Choosing k](#choosing-k)
6. [Bias-Variance Trade-off in CV](#bias-variance-trade-off-in-cv)
7. [The Right Way to Use Cross-Validation](#the-right-way-to-use-cross-validation)
8. [Advantages and Disadvantages](#advantages-and-disadvantages)
9. [Real-World Applications](#real-world-applications)
10. [Implementation Steps](#implementation-steps)
11. [Common Pitfalls](#common-pitfalls)

---

## The Problem with a Single Train/Test Split

When you split your data 80/20 and train on 80%, the accuracy you measure depends heavily on *which* 20% ended up in the test set.

**Why this is a problem:**
- A lucky test split (easy examples in test set) → over-optimistic accuracy
- An unlucky test split (hard examples in test set) → over-pessimistic accuracy
- The variance of your accuracy estimate is high when the test set is small

**Example:** You split 100 samples into 80 train / 20 test. Your model gets 85% accuracy. Is this result reliable? With 20 test samples, each percentage point corresponds to 0.2 samples — a single extra correct prediction changes accuracy by 5%.

k-Fold cross-validation solves this by using *every sample* as a test sample exactly once, averaging results to get a stable estimate.

---

## What is k-Fold Cross-Validation?

k-Fold cross-validation splits the data into k equal-sized folds, then repeats training and evaluation k times, each time using a different fold as the test set and the remaining k−1 folds as the training set.

The final performance estimate is the **mean accuracy across all k folds**.

**Key properties:**
- Every sample is in the test set exactly once
- Every sample is in the training set exactly k−1 times
- The performance estimate uses all available data
- The standard deviation of the k scores measures model stability

---

## How k-Fold Works

### With k=5 (5-fold cross-validation):

```
Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN] → score₁
Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN] → score₂
Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN] → score₃
Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN] → score₄
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST] → score₅

Final: mean(score₁...score₅) ± std(score₁...score₅)
```

### Steps
1. Shuffle the data (optional but recommended)
2. Split into k equal folds
3. For each fold i from 1 to k:
   a. Set fold i as the test set
   b. Train the model on the remaining k−1 folds
   c. Evaluate on fold i, record score
4. Report: mean accuracy and standard deviation

**Critical:** The model is retrained from scratch for each fold. This is not the same as training once and evaluating on different subsets.

---

## Variants of Cross-Validation

### Stratified k-Fold
Ensures each fold has the same class distribution as the full dataset. **Always use stratified k-fold for classification problems with imbalanced classes.**

Without stratification: a fold might contain 90% of one class by chance, giving misleadingly good or bad results for the minority class.

### Leave-One-Out Cross-Validation (LOOCV)
k = n (each sample is its own test set). Maximum use of training data; minimum variance estimate, but:
- Computationally expensive for large datasets
- High variance between folds (test set size = 1)
- Not generally recommended for practical use

### Time Series (Walk-Forward) Cross-Validation
For time series data, fold i must only test on data that came *after* all training data. Standard shuffled k-fold would leak future information into training.

```
Fold 1: [TRAIN: t=1..4]  [TEST: t=5]
Fold 2: [TRAIN: t=1..5]  [TEST: t=6]
Fold 3: [TRAIN: t=1..6]  [TEST: t=7]
```

Use `TimeSeriesSplit` in sklearn for this.

### Repeated k-Fold
Run k-fold multiple times with different random splits. Reduces variance further; good for small datasets where any single split is highly variable.

---

## Choosing k

| k Value | Characteristics |
|---------|-----------------|
| **k=5** | Standard choice. 80% of data for training. Good balance of bias and variance in the estimate. Fast. |
| **k=10** | Common recommendation in textbooks. 90% training data. Slightly less bias than k=5. 2× slower. |
| **k=3** | Fast; use when computation is the bottleneck. Less reliable estimate. |
| **LOOCV** | Near-zero bias; very high variance between folds; slow. Only for very small datasets (n < 50). |

**The practical recommendation:** Use k=5 as default. Switch to k=10 for small datasets where 20% held-out data is too small.

---

## Bias-Variance Trade-off in CV

Cross-validation estimates have their own bias-variance trade-off:

| k | Training set size | CV estimate bias | CV estimate variance |
|---|-------------------|-----------------|---------------------|
| Small (3) | 67% of data | Pessimistic (underfits) | High (few folds) |
| Medium (5-10) | 80-90% of data | Low | Low to medium |
| LOOCV (n) | n−1 samples | Near zero | High (each test is 1 point) |

**The standard deviation** of the k scores is more informative than the mean:
- Low std (< 2%) → the model is stable; the CV estimate is reliable
- High std (> 5%) → high variance model; estimate is noisy; consider a simpler model or more data

---

## The Right Way to Use Cross-Validation

### Cross-Validation vs Reporting Final Performance

**Correct workflow:**
1. Use k-fold CV to estimate model performance during **model selection**
2. Once the best model is selected, train it on the **full dataset**
3. Use a **separate held-out test set** (set aside before any CV) for the final performance report

**Wrong workflow:**
- Report cross-validation score as the final performance number for a "held-out" test set
- Cross-validation uses all data for training across folds — there is no truly held-out set

### The Pipeline Must Be Inside the CV Loop

Every data-dependent preprocessing step must happen *inside* each fold, not outside:

```python
# WRONG — scaler fitted on full dataset before CV
scaler.fit(X)
X_scaled = scaler.transform(X)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

# CORRECT — scaler fitted only on training folds
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
cv_scores = cross_val_score(pipe, X, y, cv=5)
```

Using a Pipeline ensures the scaler is fit only on the training folds — no leakage.

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Stable estimate | Averages over k splits; much lower variance than single split |
| Uses all data | Every sample contributes to both training and evaluation |
| Diagnoses stability | Standard deviation of scores reveals model variance |
| Standard and reproducible | Any researcher can replicate the evaluation |
| Model-agnostic | Works with any sklearn estimator |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| k× slower | Training k models instead of 1 |
| Cannot guarantee fairness | Even stratified splits may have unlucky distributions for small datasets |
| Not suitable for time series | Shuffled folds leak future information; need TimeSeriesSplit |
| Still just an estimate | CV performance and deployment performance can differ if distribution shifts |

---

## Real-World Applications

**Model Selection:**
"Which algorithm (logistic regression, random forest, SVM) works best for this dataset?" — Run 5-fold CV for each; compare mean and std of scores.

**Hyperparameter Tuning:**
"What regularisation strength (C) gives the best logistic regression?" — Grid Search CV runs CV for each parameter combination.

**Feature Selection:**
"Does adding this new feature actually help?" — Compare CV scores with and without the feature.

**Algorithm Research:**
Research papers report CV scores rather than single-split scores to produce reliable, reproducible comparisons between new and existing methods.

---

## Implementation Steps

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Build pipeline (preprocessing inside the CV loop)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# 2. Define cross-validator
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 3. Run cross-validation
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')

# 4. Report results
print(f"Accuracy per fold: {scores.round(3)}")
print(f"Mean accuracy: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")
print(f"95% confidence interval: ({scores.mean() - 2*scores.std():.4f}, "
      f"{scores.mean() + 2*scores.std():.4f})")

# 5. Get both train and test scores (to check for overfitting)
from sklearn.model_selection import cross_validate
cv_results = cross_validate(pipe, X, y, cv=cv,
                            scoring='accuracy',
                            return_train_score=True)

print(f"Train scores: {cv_results['train_score'].mean():.4f}")
print(f"Test scores:  {cv_results['test_score'].mean():.4f}")
# Large train-test gap → overfitting
```

---

## Common Pitfalls

**1. Fitting preprocessing outside the CV loop**
StandardScaler fitted on all data before CV means the test fold's statistics influenced the scaler — data leakage. Always use `Pipeline`.

**2. Using standard k-fold for imbalanced classification**
Without stratification, a fold might contain very few minority class samples. Use `StratifiedKFold` for all classification problems.

**3. Reporting CV score as the final test set accuracy**
CV rotates the test set across all folds — effectively using all data for training at some point. For an honest final evaluation, hold out a test set before running any CV.

**4. Using shuffle=False when data is ordered**
If your CSV has class 0 first and class 1 second, non-shuffled k-fold will put all class 0 in early folds and all class 1 in late folds. Always shuffle unless order matters (time series).

**5. Ignoring the standard deviation**
A model with 85% ± 1% is more reliable than 86% ± 8%. Report both mean and std. The std is often more important for production deployment decisions.

**6. Comparing models with different random states**
If you compare Model A with `random_state=42` and Model B with `random_state=0`, the folds differ — any performance gap could be due to random variation. Use the same `cv` object for all comparisons.
