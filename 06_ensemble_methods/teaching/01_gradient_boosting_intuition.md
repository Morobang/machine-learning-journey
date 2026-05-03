# Gradient Boosting — From Intuition to Mathematics

## The Core Idea

Bagging (Random Forest) builds trees **in parallel** and averages their predictions.
Boosting builds trees **in sequence** — each tree learns from the mistakes of the ones before it.

The simplest way to think about it:

> **You predict a house price at $300K. The true price is $350K. Your error is $50K.**
> **The next tree's job is to predict that $50K error.**
> **The ensemble prediction is $300K + $50K = $350K.**

That is literally what gradient boosting does, mathematically expressed.

---

## Step-by-Step Walk-Through

Given a dataset with 4 houses:

| House | True Price | Step 1 Prediction | Residual |
|-------|-----------|-------------------|----------|
| A | $350K | $300K | +$50K |
| B | $200K | $300K | -$100K |
| C | $450K | $300K | +$150K |
| D | $280K | $300K | -$20K |

Step 1: Start with the mean prediction ($300K). Compute residuals.

Step 2: Train Tree 2 to predict the residuals:

| House | Residual | Tree 2 Prediction |
|-------|----------|-------------------|
| A | +$50K | +$48K |
| B | -$100K | -$95K |
| C | +$150K | +$145K |
| D | -$20K | -$22K |

Step 3: New ensemble prediction = $300K + (learning_rate × Tree 2 prediction)

With `learning_rate = 0.1`:
- House A: $300K + 0.1 × $48K = $304.8K (was $300K, truth $350K — getting closer)

Step 4: Compute new residuals. Train Tree 3 on those. Repeat for N trees.

The final prediction is:

$$\hat{y} = F_0 + \eta h_1(x) + \eta h_2(x) + \ldots + \eta h_N(x)$$

Where $F_0$ is the initial prediction (usually the mean), $\eta$ is the learning rate, and each $h_m$ is a tree.

---

## Why "Gradient" Boosting?

The residuals we're fitting are the **negative gradient of the loss function**.

For mean squared error (MSE) loss $L = \frac{1}{2}(y - \hat{y})^2$:

$$-\frac{\partial L}{\partial \hat{y}} = y - \hat{y} = \text{residual}$$

So fitting residuals is equivalent to taking a gradient descent step in function space.
By changing the loss function, we change what the residuals represent — which is how the same algorithm handles regression (MSE), classification (log-loss), and ranking (pairwise loss).

---

## The Learning Rate

The learning rate $\eta$ shrinks each tree's contribution.

**Low learning rate (e.g. 0.01):**
- Each step is tiny — more trees needed to converge
- Less likely to overshoot the optimum
- Generally better generalisation

**High learning rate (e.g. 0.3):**
- Converges in fewer trees
- More likely to overfit

The learning rate and number of trees are inversely related:
- `learning_rate=0.3, n_estimators=100` ≈ `learning_rate=0.03, n_estimators=1000` in capacity
- The lower learning rate version usually generalises better

**Rule of thumb:** Use `learning_rate ≤ 0.1` with early stopping to find the right `n_estimators` automatically.

---

## Bias-Variance View

| Method | What it reduces | Mechanism |
|--------|----------------|-----------|
| Bagging (Random Forest) | Variance | Average many uncorrelated trees |
| Boosting | Bias | Sequentially correct errors |

Boosting starts with a weak model (high bias) and reduces bias with each step.
The risk is that after many steps, it starts memorising the training data — overfitting.

Regularisation (XGBoost's tree penalties, `max_depth`, `subsample`) controls this.

---

## When Gradient Boosting Beats Random Forest

| Scenario | GBM / XGBoost | Random Forest |
|----------|--------------|---------------|
| Tabular data, structured | Usually better | Very good |
| Noisy data with many outliers | Sensitive (fits residuals) | More robust |
| Very large datasets (>1M rows) | Use LightGBM | Slower |
| Need fast training with no tuning | Slower to tune | Faster out-of-box |
| Kaggle / competitions | Dominates | Good baseline |
| Missing values | Handles natively (XGBoost) | Needs imputation |

---

## Common Mistakes

**Mistake 1: Not using early stopping**
Setting `n_estimators=1000` without early stopping almost always overfits.
Always provide a validation set and `early_stopping_rounds`.

**Mistake 2: Too high a learning rate**
`learning_rate=0.3` feels fast but usually gives worse generalisation.
Start at `0.05–0.1` and use more trees.

**Mistake 3: Ignoring class imbalance**
Gradient boosting minimises the average loss — on imbalanced data, it optimises for the majority class.
Fix: `scale_pos_weight` (XGBoost), `class_weight` (sklearn GBM), or SMOTE.

**Mistake 4: Using the default threshold for classification**
XGBoost's default classification threshold is 0.5 but your optimal threshold depends on the cost of false positives vs false negatives.
Always plot precision-recall vs threshold.

---

## See Also

- [XGBoost Deep Dive](02_xgboost_deep_dive.md) — the specific improvements XGBoost makes
- Notebook: `../notebooks/02_adaboost_gradient_boosting.ipynb`
- Notebook: `../notebooks/03_xgboost_lightgbm.ipynb`
