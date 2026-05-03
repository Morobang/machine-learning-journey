# Model Selection and Evaluation

Every machine learning model has two kinds of parameters: the weights learned from data (learned during training) and the hyperparameters you set before training (learning rate, number of trees, regularisation strength). Model selection is the process of finding the right hyperparameters and the right algorithm. Model evaluation is the process of estimating how well the chosen model will perform on data it has never seen.

Done naively, both introduce **data leakage**: optimistic performance estimates that collapse when the model meets real-world data. This section is about doing both correctly.

---

## The Core Problem: Optimistic Estimates

A single train/test split gives you one estimate of performance — and that estimate depends on which rows happened to land in the test set. With 20% test data from 500 rows, you have 100 test examples. Swap a few of those rows and the accuracy estimate shifts by several percent. A single split is noisy.

**The naive solution is also wrong:** evaluate every candidate model on the test set and pick the best. Now the test set has been used to make a decision — it is no longer independent. The performance of the selected model on "test" data is optimistic. You have implicitly overfit to the test set through your model selection process.

The correct solution is **cross-validation for evaluation** and **nested cross-validation for hyperparameter tuning**.

---

## Techniques in This Section

### K-Fold Cross-Validation
**Notebook:** [01_k_fold_cross_validation.ipynb](notebooks/01_k_fold_cross_validation.ipynb) | **Guide:** [teaching/01_k_fold_cross_validation.md](teaching/01_k_fold_cross_validation.md)

K-fold CV splits the training data into k equal folds. The model trains on k−1 folds and is evaluated on the held-out fold. This repeats k times, each fold acting as the validation set exactly once. Final performance = mean ± standard deviation across k estimates.

```
k = 5:
Fold 1: Train [2,3,4,5] → Validate [1]
Fold 2: Train [1,3,4,5] → Validate [2]
Fold 3: Train [1,2,4,5] → Validate [3]
Fold 4: Train [1,2,3,5] → Validate [4]
Fold 5: Train [1,2,3,4] → Validate [5]
Average the 5 validation scores → stable estimate
```

**Why it works:** Every row appears in the validation set exactly once, so the full dataset contributes to evaluation. The mean score has lower variance than a single split. The standard deviation reveals whether performance is consistent or highly sensitive to which data the model trains on.

**The bias-variance trade-off in k:** Large k (e.g., k=10, or leave-one-out) gives nearly unbiased estimates but high variance across folds and slow runtime. Small k (e.g., k=5) gives slightly biased but more stable estimates and runs faster. k=5 or k=10 are standard choices for most datasets.

**Use when:** Comparing algorithms, estimating generalisation performance, or deciding whether a model is ready. Any time you need a reliable performance number rather than a lucky split.

**Watch out for:** For classification with imbalanced classes, use **Stratified K-Fold** — it ensures each fold has the same class distribution as the full dataset. For time series, use walk-forward splits instead (see [07_time_series_analysis](../07_time_series_analysis/)).

---

### Grid Search with Cross-Validation
**Notebook:** [02_grid_search.ipynb](notebooks/02_grid_search.ipynb) | **Guide:** [teaching/02_grid_search_hyperparameter_tuning.md](teaching/02_grid_search_hyperparameter_tuning.md)

Grid search exhaustively evaluates every combination of hyperparameter values you specify. Combined with k-fold CV, each combination is evaluated on k separate validation sets, giving a reliable estimate of how well that configuration generalises.

```
SVM example:
C values:     [0.1, 1, 10, 100]
kernel types: ['linear', 'rbf']
gamma values: [0.01, 0.1, 1]

Grid: 4 × 2 × 3 = 24 combinations
Each evaluated with 10-fold CV = 240 model fits
```

`GridSearchCV` in scikit-learn handles this in one call: it fits, evaluates, and returns `best_params_` and `best_estimator_` automatically.

**The data leakage trap.** When you use grid search to select hyperparameters, the validation score for each combination is an honest estimate. But the *best* combination was selected by comparing validation scores — the best score is still optimistic. To get an unbiased final estimate of the selected model:

```
Nested CV pattern:
Outer loop: k-fold CV for final performance estimate
  Inner loop: GridSearchCV to select hyperparameters
The outer loop score is unbiased because the inner loop (selection)
is never allowed to see the outer fold's test rows.
```

For the final production model, refit on all training data using the best hyperparameters found.

**Random search** samples hyperparameter combinations randomly rather than exhaustively. For large search spaces, random search finds good configurations faster than grid search — searching 100 random combinations covers more of a 5-parameter space than a 4×5 grid covering the same budget.

**Bayesian optimisation** uses results from evaluated combinations to decide which region of the parameter space to search next. It is more efficient than random search on expensive models but adds implementation complexity.

**Use when:** Tuning any model with hyperparameters that meaningfully affect performance. Always combine with cross-validation — never tune on the test set.

**Watch out for:** Grid search runtime grows exponentially with the number of parameters and values. Start with a coarse grid (few values, wide range) to find the right region, then a fine grid to narrow down. For models with many hyperparameters (XGBoost, neural networks), prefer random search or Bayesian optimisation.

---

## Choosing a Validation Strategy

```
How large is your dataset?
├── Small (< 1,000 rows)
│   └── Use k=10 or leave-one-out CV (maximise data usage)
├── Medium (1,000 – 100,000 rows)
│   └── Use k=5 or k=10 CV (standard choice)
└── Large (> 100,000 rows)
    └── Simple train/validation split is often sufficient
        CV still useful but each fit is expensive

Is the target class imbalanced?
└── Yes → Stratified K-Fold at every level

Is the data time-ordered?
└── Yes → Walk-forward splits (see 07_time_series_analysis)

Are you tuning hyperparameters?
└── Yes → Nest the CV: GridSearchCV inside an outer CV loop
```

---

## What the Teaching Guides Cover

[teaching/01_k_fold_cross_validation.md](teaching/01_k_fold_cross_validation.md) — why single train/test splits are insufficient, the bias-variance trade-off in CV estimates, stratified CV for imbalanced classes, leave-one-out CV, and the walk-forward variant for time series.

[teaching/02_grid_search_hyperparameter_tuning.md](teaching/02_grid_search_hyperparameter_tuning.md) — the hyperparameter vs learned parameter distinction, grid search mechanics, data leakage through model selection, nested CV, random search and when to prefer it over grid search, and Bayesian optimisation concepts.

---

## Relationship to the Rest of This Repository

Cross-validation and grid search are not standalone topics — they are the infrastructure every other section depends on. When a regression or classification notebook reports a test-set accuracy, cross-validation is how you trust that number. When an ensemble method notebook tunes `n_estimators` or `max_depth`, grid search is the tool.

- **All supervised learning notebooks** ([01_supervised_learning](../01_supervised_learning/)) benefit from CV-based evaluation
- **Ensemble methods** ([06_ensemble_methods](../06_ensemble_methods/)) have more hyperparameters than simpler models and require careful tuning
- **Time series** ([07_time_series_analysis](../07_time_series_analysis/)) uses a modified CV strategy — walk-forward splits
- **Dimensionality reduction** ([08_dimensionality_reduction](../08_dimensionality_reduction/)) requires that PCA/LDA be fit inside the CV loop, not before it, to prevent leakage
