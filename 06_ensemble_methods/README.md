# Ensemble Methods

Ensemble methods combine multiple models to produce predictions that are more accurate and more stable than any single model alone. The intuition: different models make different mistakes. When you average or combine their outputs, the errors tend to cancel while the correct signal is reinforced.

Ensembles are not a niche technique — they are the dominant approach in competitive machine learning and production systems. The winning solutions of almost every Kaggle tabular data competition use gradient boosting (XGBoost, LightGBM, CatBoost), often combined with other ensembled models.

---

## Why Ensembles Work: The Bias-Variance Decomposition

The prediction error of any model has two components:

**Bias** — error from wrong assumptions. A linear model applied to nonlinear data has high bias — it systematically mispredicts regardless of how much data you give it. Underfitting.

**Variance** — error from sensitivity to the training data. A deep decision tree changes dramatically with small data changes. It fits the training data perfectly but fails on new data. Overfitting.

The two main ensemble strategies each target one of these:

| Strategy | Targets | How |
|----------|---------|-----|
| **Bagging** | Variance | Train many high-variance models on random data subsets; average their predictions |
| **Boosting** | Bias | Train models sequentially, each correcting the errors of the previous |

---

## Ensemble Strategies

### Bagging (Bootstrap Aggregating)

Train many models **in parallel**, each on a different bootstrap sample (random sample with replacement) of the training data. Aggregate predictions by averaging (regression) or majority vote (classification).

**Why it reduces variance:** Each model sees a slightly different dataset, so each makes slightly different errors. Averaging smooths out the individual errors.

**The canonical bagging algorithm is Random Forest** — it extends bagging by also randomly selecting a subset of features at each split, ensuring the trees are different from each other (diverse) rather than all finding the same dominant feature.

---

### Boosting

Train models **sequentially**. Each new model focuses on the examples the previous model got wrong. The final prediction is a weighted sum of all models in the sequence.

**Why it reduces bias:** Each model corrects systematic mistakes of the previous one. The ensemble gradually reduces the training error.

**The three major boosting algorithms:**

**AdaBoost** — the original. After each weak learner, increase the weights of misclassified examples. The next learner focuses more on those hard examples. Simple and interpretable.

**Gradient Boosting** — the generalisation. Instead of reweighting examples, fit each new model to the **residuals** (errors) of the current ensemble. The objective is minimised by gradient descent in function space.

**XGBoost / LightGBM / CatBoost** — optimised gradient boosting implementations that add: regularisation terms to the objective (reduces overfitting), second-order gradient information (faster convergence), and engineering optimisations (parallelism, memory efficiency) that make them 10–100× faster than scikit-learn's `GradientBoostingClassifier`.

---

### Stacking (Stacked Generalisation)

Train several diverse base models (e.g., logistic regression, random forest, SVM). Use their predictions as features for a **meta-learner** that makes the final prediction.

Unlike bagging/boosting where all base models are the same type, stacking explicitly benefits from **model diversity** — different algorithm families make different kinds of errors, and the meta-learner learns to exploit their complementary strengths.

Stacking is powerful but complex: it requires careful cross-validation to prevent the meta-learner from overfitting to the base model predictions.

---

## Algorithms in the Teaching Guides

There are no implementation notebooks yet for this section. The teaching guides provide the conceptual and mathematical foundation:

### Gradient Boosting Intuition
**Guide:** [teaching/01_gradient_boosting_intuition.md](teaching/01_gradient_boosting_intuition.md)

Covers the full derivation of gradient boosting from the perspective of gradient descent in function space. Explains why fitting residuals is equivalent to gradient descent, how the learning rate controls the step size, and how tree depth controls model complexity. The foundation for understanding XGBoost and LightGBM.

### XGBoost Deep Dive
**Guide:** [teaching/02_xgboost_deep_dive.md](teaching/02_xgboost_deep_dive.md)

Covers what makes XGBoost different from standard gradient boosting: the regularised objective function, second-order Taylor expansion of the loss, the exact greedy algorithm for finding splits, feature and row subsampling, and the key hyperparameters with guidance on how to tune them.

---

## When to Use Each Method

| Method | Best for | Watch out for |
|--------|---------|---------------|
| **Random Forest** | Fast, robust baseline on tabular data; feature importance | Slower prediction with many trees; poor extrapolation |
| **Gradient Boosting** | Maximum accuracy on tabular data | Requires careful hyperparameter tuning; slow to train |
| **XGBoost / LightGBM** | Production tabular ML; large datasets | Many hyperparameters; easy to overfit without regularisation |
| **Stacking** | Final performance squeeze in competitions | Complex implementation; risk of meta-learner overfitting |
| **Voting** | Quick combination of similar-performing models | Little benefit if models make the same errors |

**The practical default for structured/tabular data:** Start with a Random Forest as a baseline. Switch to LightGBM or XGBoost for maximum performance. Use stacking only when squeezing the last fraction of a percent matters.

---

## Key Hyperparameters

All gradient boosting models share the same core hyperparameters:

| Parameter | Effect | Typical range |
|-----------|--------|--------------|
| `n_estimators` | Number of trees in the ensemble | 100–2,000 (use early stopping) |
| `learning_rate` | Shrinks each tree's contribution | 0.01–0.3 (lower → more trees needed) |
| `max_depth` | Maximum tree depth | 3–8 (deeper → more complex, more overfit risk) |
| `subsample` | Fraction of rows per tree | 0.6–1.0 (lower adds randomness, reduces overfit) |
| `colsample_bytree` | Fraction of features per tree | 0.6–1.0 |

**The most important interaction:** `learning_rate` and `n_estimators` trade off against each other. A lower learning rate generally gives better final performance but requires more trees to compensate. Use early stopping to find the right number of trees automatically.

---

## Relationship to the Rest of This Repository

Random Forest appears in both [01_supervised_learning/01_regression](../01_supervised_learning/01_regression/) and [01_supervised_learning/02_classification](../01_supervised_learning/02_classification/) as the strongest general-purpose single-model algorithm. The ensemble methods here explain *why* Random Forest works and provide the conceptual path to the more powerful gradient boosting family.

Cross-validation and hyperparameter tuning, covered in [09_model_selection_and_evaluation](../09_model_selection_and_evaluation/), are especially important for ensemble methods because they have more hyperparameters than simpler models.
