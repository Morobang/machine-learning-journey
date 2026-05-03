# Classification

Classification is the branch of supervised learning concerned with predicting **which category** an input belongs to. The output is always one of a finite set of possible labels — not a continuous number.

Every time a spam filter decides whether an email is junk, a bank decides whether a transaction is fraudulent, or a doctor runs a diagnostic test, they are performing classification.

---

## What Makes a Problem a Classification Problem

A problem is a classification problem when:
- The output is a discrete label from a predefined set
- The goal is to predict *which category* rather than *how much*
- Intermediate values between categories make no sense ("half spam" is not a meaningful prediction)

**Binary classification:** Two possible outcomes (spam/not spam, fraud/legitimate, disease/healthy).

**Multi-class classification:** More than two outcomes (digit recognition 0–9, product category, language identification). Most binary algorithms extend to multi-class via one-vs-rest or softmax strategies.

---

## Algorithms in This Section

### Logistic Regression
**Notebook:** [01_logistic_regression.ipynb](notebooks/01_logistic_regression.ipynb) | **Guide:** [teaching/01_logistic_regression.md](teaching/01_logistic_regression.md)

Despite the name, logistic regression is a classification algorithm. It models the probability of class membership using the sigmoid function:

```
P(y=1) = 1 / (1 + e^(-(b₀ + b₁x₁ + ... + bₙxₙ)))
```

The decision boundary is a straight line (or hyperplane in multiple dimensions). Everything above the boundary is predicted class 1; everything below is predicted class 0.

**Use when:** You need a fast, interpretable baseline for binary classification. The coefficients are directly interpretable as log-odds ratios. Logistic regression is the standard first model to try on any classification problem.

**Watch out for:** Only captures linear decision boundaries. If the true boundary is curved or nonlinear, logistic regression will underfit.

---

### K-Nearest Neighbors (KNN)
**Notebook:** [02_k_nearest_neighbors.ipynb](notebooks/02_k_nearest_neighbors.ipynb) | **Guide:** [teaching/02_k_nearest_neighbors.md](teaching/02_k_nearest_neighbors.md)

KNN makes predictions by finding the k most similar training examples (nearest neighbours) and taking a majority vote of their labels. No training phase — the entire dataset is stored and consulted at prediction time.

**Use when:** The decision boundary is locally complex but smooth; you want a simple nonparametric baseline; the dataset is small enough that storing all training examples is feasible.

**Watch out for:** Feature scaling is mandatory (distance-based). Slow at prediction time for large datasets (must compute distances to all training points). Sensitive to irrelevant features. Performance degrades in high dimensions (curse of dimensionality).

---

### Support Vector Machine (SVM)
**Notebook:** [03_support_vector_machine.ipynb](notebooks/03_support_vector_machine.ipynb) | **Guide:** [teaching/03_support_vector_machine.md](teaching/03_support_vector_machine.md)

SVM finds the decision boundary (hyperplane) that maximises the margin — the distance between the boundary and the nearest training examples from each class. Only the support vectors (examples closest to the boundary) determine the boundary position.

**Use when:** You have a relatively small, clean dataset and want strong generalisation. The maximum-margin objective provides a form of regularisation. SVM with a linear kernel is excellent for high-dimensional data (text classification).

**Watch out for:** Feature scaling is required. Sensitive to the regularisation parameter C. Slow on very large datasets. Does not natively output probabilities (requires `probability=True` which is approximate).

---

### Kernel SVM
**Notebook:** [04_kernel_svm.ipynb](notebooks/04_kernel_svm.ipynb) | **Guide:** [teaching/04_kernel_svm.md](teaching/04_kernel_svm.md)

Extends SVM to nonlinear boundaries using the kernel trick. A kernel function computes the similarity between data points in a transformed (higher-dimensional) space without explicitly constructing that space. The RBF (Gaussian) kernel is the most common choice.

**Use when:** The data is not linearly separable and standard SVM fails. RBF kernel SVM is a strong general-purpose classifier for small to medium datasets.

**Watch out for:** Requires tuning both C (regularisation) and γ (kernel bandwidth) via cross-validation. Slower than linear SVM; does not scale to very large datasets. Feature scaling remains mandatory.

---

### Naive Bayes
**Notebook:** [05_naive_bayes.ipynb](notebooks/05_naive_bayes.ipynb) | **Guide:** [teaching/05_naive_bayes.md](teaching/05_naive_bayes.md)

A probabilistic classifier based on Bayes' theorem. It estimates P(class | features) as proportional to P(class) × P(features | class). The "naive" assumption: all features are conditionally independent given the class. Despite this rarely being true, Naive Bayes works well in practice.

```
P(class | x₁, x₂, ...) ∝ P(class) × P(x₁|class) × P(x₂|class) × ...
```

**Use when:** Text classification (spam filtering, sentiment analysis, document categorisation). Fast to train and predict. Works well even with small training sets. A strong baseline for text problems.

**Watch out for:** The independence assumption is always violated to some degree. It struggles with features that are not independent (correlated features). For tabular data with complex feature interactions, other algorithms usually outperform it.

---

### Decision Tree Classification
**Notebook:** [06_decision_tree_classification.ipynb](notebooks/06_decision_tree_classification.ipynb) | **Guide:** [teaching/06_decision_tree_classification.md](teaching/06_decision_tree_classification.md)

Builds a tree of if-then rules by recursively splitting the data on the feature and threshold that best reduces impurity (Gini or entropy). Each leaf predicts the majority class of all training samples that fell into it.

**Use when:** Interpretability is paramount — a decision tree can be printed as a flowchart and explained to non-technical stakeholders without any ML knowledge. Handles mixed feature types without preprocessing.

**Watch out for:** High variance — small data changes cause large tree structure changes. Without depth limits, a tree perfectly memorises training data (overfit). Single trees are rarely used in production; Random Forest (below) addresses the variance problem.

---

### Random Forest Classification
**Notebook:** [07_random_forest_classification.ipynb](notebooks/07_random_forest_classification.ipynb) | **Guide:** [teaching/07_random_forest_classification.md](teaching/07_random_forest_classification.md)

An ensemble of decision trees trained on random subsets of the data (bootstrap samples) and random subsets of features at each split. Predictions are the majority vote across all trees. Averaging reduces variance dramatically compared to a single tree.

**Use when:** You want strong out-of-the-box accuracy on tabular data. Random Forest is one of the most robust general-purpose classifiers — it handles nonlinear boundaries, feature interactions, mixed types, and missing values with minimal preprocessing. It also provides feature importance scores.

**Watch out for:** Less interpretable than a single tree. Slow to predict when `n_estimators` is large. Can still overfit on very noisy data if trees are grown too deep.

---

## Choosing a Classification Algorithm

```
Start here: Is fast interpretability the priority?
├── Yes → Decision Tree or Logistic Regression
│         (Decision Tree for rule-based decisions, Logistic for probability output)
└── No → How large is the dataset?
         ├── Small/medium (< 10,000 samples)
         │   ├── Linear boundary likely? → Logistic Regression or Linear SVM
         │   └── Nonlinear boundary? → Kernel SVM or Random Forest
         └── Large dataset
             ├── Text features? → Logistic Regression or Linear SVC (sparse + fast)
             └── Tabular features → Random Forest or Gradient Boosting (XGBoost)
```

**The practical starting point:** Train Logistic Regression as a baseline, then Random Forest. If Random Forest underperforms, investigate why (nonlinearity vs class imbalance vs data quality) before jumping to more complex models.

---

## Evaluation Metrics for Classification

### The Confusion Matrix

The confusion matrix is the foundation of all classification metrics:

```
              Predicted 0    Predicted 1
Actual 0          TN              FP
Actual 1          FN              TP
```

| Metric | Formula | What it answers |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / total | What fraction of all predictions were correct? |
| **Precision** | TP / (TP + FP) | Of all predicted positives, how many were actually positive? |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many did the model catch? |
| **F1** | 2 × P × R / (P + R) | Harmonic mean of precision and recall |

### Which Metric to Use

**Use accuracy when:**
- Classes are balanced (roughly equal positive and negative examples)
- FP and FN errors have equal cost

**Use precision when:**
- False positives are costly (e.g., a spam filter wrongly blocking legitimate email)
- You prefer missing some positives over over-predicting

**Use recall when:**
- False negatives are costly (e.g., a cancer screen missing a diagnosis)
- You prefer over-predicting to missing actual positives

**Use F1 when:**
- You need a single number that balances precision and recall
- Classes are imbalanced

**Accuracy is misleading on imbalanced data.** A dataset with 95% negative examples allows a model that always predicts negative to score 95% accuracy — while catching exactly zero actual positives.

---

## What Each Teaching Guide Contains

Each algorithm in [teaching/](teaching/) covers:
- The intuition and theory behind how the algorithm works
- The mathematical objective (decision boundary, loss function, or probabilistic model)
- When to use it and clear cases where it will fail
- Key hyperparameters and how to tune them
- A comparison against similar algorithms
- Real-world applications
- Common pitfalls

The notebooks in [notebooks/](notebooks/) implement each algorithm with EDA, preprocessing, training, evaluation, and visualisation of the decision boundary.
