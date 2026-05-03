# XGBoost Deep Dive

XGBoost (eXtreme Gradient Boosting) is the algorithm you will encounter most in ML interviews, Kaggle competitions, and production tabular ML systems. This document explains what it actually does differently — not just "it's fast and accurate."

---

## What XGBoost Fixes in Vanilla Gradient Boosting

Standard sklearn `GradientBoostingClassifier` has three problems:
1. No regularisation — overfits easily on noisy data
2. Greedy, single-threaded split finding — slow on large datasets
3. No native handling of missing values

XGBoost addresses all three.

---

## The Objective Function

XGBoost minimises an objective that has two parts:

$$\text{Obj}(\theta) = \underbrace{\sum_{i=1}^{n} l(y_i, \hat{y}_i)}_{\text{training loss}} + \underbrace{\sum_{k=1}^{K} \Omega(f_k)}_{\text{regularisation}}$$

The regularisation term for each tree $f_k$ is:

$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

Where:
- $T$ = number of leaves in the tree
- $w_j$ = weight of leaf $j$
- $\gamma$ = minimum loss reduction to make a split (prunes small-gain splits)
- $\lambda$ = L2 regularisation on leaf weights (shrinks individual leaf values)

**In plain English:**
- $\gamma$ penalises trees with too many leaves — encourages shallow, simple trees
- $\lambda$ shrinks the leaf weights — reduces the magnitude of each tree's predictions

Vanilla GBM has no equivalent. This is the primary reason XGBoost generalises better.

---

## The Split Finding Algorithm

For each node, GBM tries every possible split point on every feature.
On a dataset with 100K rows and 100 features, that is 10 million candidate splits per node.

XGBoost uses an **approximate algorithm**: it bins continuous features into quantiles (by default ~256 bins) and only considers splits at bin boundaries. This reduces candidates from millions to ~25,600 with negligible accuracy loss.

Additionally, XGBoost supports:
- **Column subsampling** (`colsample_bytree`, `colsample_bylevel`, `colsample_bynode`) — only considers a random subset of features at each split, similar to Random Forest
- **Row subsampling** (`subsample`) — trains each tree on a random fraction of training rows
- **Parallel split finding** — candidate splits evaluated across CPU cores simultaneously

---

## Handling Missing Values

XGBoost has a built-in **default direction** for missing values at every split.

During training, when a feature value is missing, XGBoost tries both directions (left and right) and learns which direction reduces loss more. This direction is stored with the tree.

During inference, any missing value automatically follows the learned direction.

**Practical consequence:** You do not need to impute features before training XGBoost. It will learn the optimal handling for each feature from the data.

---

## Key Hyperparameters and What They Control

### Complexity control (most important)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_depth` | 6 | Maximum tree depth. Deeper = more complex = more overfit risk. Use 3–8. |
| `min_child_weight` | 1 | Minimum sum of instance weight in a leaf. Higher = more conservative, less overfit. |
| `gamma` | 0 | Minimum loss reduction to split. Higher = more conservative. |
| `reg_alpha` | 0 | L1 regularisation on leaf weights. Drives some weights to zero (feature selection). |
| `reg_lambda` | 1 | L2 regularisation on leaf weights. Shrinks weights smoothly. |

### Subsampling (second most important)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `subsample` | 1.0 | Fraction of training rows sampled per tree. 0.8 often helps. |
| `colsample_bytree` | 1.0 | Fraction of features sampled per tree. |
| `colsample_bylevel` | 1.0 | Fraction of features sampled per tree level. |

### Boosting control

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | 100 | Number of trees. Use early stopping instead of guessing. |
| `learning_rate` | 0.3 | Shrinkage per step. Lower = better generalisation, slower. |
| `early_stopping_rounds` | None | Stop if validation metric doesn't improve for N rounds. |

### Class imbalance

| Parameter | Default | Effect |
|-----------|---------|--------|
| `scale_pos_weight` | 1 | Ratio of negative to positive samples. Set to `sum(y==0)/sum(y==1)` for imbalanced binary classification. |

---

## A Principled Tuning Strategy

Random search across all parameters at once is noisy and slow. This sequence works reliably:

**Step 1** — Fix learning rate low, use early stopping to find n_estimators:
```python
model = XGBClassifier(learning_rate=0.05, n_estimators=1000,
                      early_stopping_rounds=30, eval_metric="auc")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
best_n = model.best_iteration
```

**Step 2** — Tune tree structure with the fixed n_estimators:
```python
params = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 3, 5],
}
# RandomizedSearchCV on these with cv=5
```

**Step 3** — Tune subsampling:
```python
params = {
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
}
```

**Step 4** — Tune regularisation (usually small gains):
```python
params = {
    "reg_alpha": [0, 0.01, 0.1, 1],
    "reg_lambda": [1, 2, 5],
}
```

**Step 5** — Optionally lower learning rate further and scale n_estimators:
```python
# If learning_rate=0.05 and best_n=200, try learning_rate=0.01 and n_estimators=1000
# Often gives a small improvement in final AUC
```

---

## LightGBM vs XGBoost: The Decision Table

| Criterion | Choose XGBoost | Choose LightGBM |
|-----------|---------------|-----------------|
| Dataset size | <100K rows | >100K rows |
| Training speed | Acceptable | Need faster |
| Categorical features | Encode manually | Use native support |
| Missing values | Both handle natively | Both handle natively |
| Community support | Larger, more resources | Growing rapidly |
| Memory | More | Less |
| Leaf-wise overfitting risk | Less | More (tune `num_leaves`, `min_child_samples`) |
| Accuracy difference | Negligible on same data | Negligible on same data |

**In practice:** Try both on your dataset. The performance difference is usually less than 0.5% AUC. Pick whichever is faster to train.

---

## Reading XGBoost Output

```
[0] validation-auc:0.83421
[50] validation-auc:0.86534
[100] validation-auc:0.87012
[150] validation-auc:0.87289
[180] validation-auc:0.87301   ← best
[210] validation-auc:0.87251   ← declining
Stopping. Best iteration: [180], score: 0.87301
```

- Rising validation AUC = model still learning
- Plateau = near-optimal, early stopping is working
- Declining validation AUC while training AUC rises = overfitting

If training AUC is 0.99 and validation is 0.87, you have severe overfitting:
- Increase `reg_lambda` and `reg_alpha`
- Reduce `max_depth`
- Reduce `subsample` and `colsample_bytree`
- Increase `min_child_weight`

---

## Interview Quick Reference

**Q: What are the two regularisation terms XGBoost adds that vanilla GBM lacks?**
A: $\gamma$ (minimum gain required to split, controls tree complexity) and $\lambda$ (L2 penalty on leaf weights, shrinks predictions).

**Q: How does XGBoost handle missing values?**
A: It learns a default direction for each split during training — whichever direction for missing values reduces loss more. No imputation needed.

**Q: Why does a lower learning rate with more trees usually outperform a higher learning rate with fewer trees?**
A: Lower learning rate = each tree corrects a smaller fraction of the error = smoother, more conservative updates = less risk of overstepping the optimum. The final model is an ensemble of many small, careful corrections rather than few large ones.

**Q: What is `scale_pos_weight` and when do you use it?**
A: It multiplies the gradient and hessian of positive-class samples by this value, making errors on the minority class count more in the objective. Use `sum(negatives)/sum(positives)` when your classes are severely imbalanced.
