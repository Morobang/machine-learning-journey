# Linear Discriminant Analysis (LDA) — Complete Guide

## Table of Contents
1. [What is LDA?](#what-is-lda)
2. [LDA vs PCA — The Key Difference](#lda-vs-pca-the-key-difference)
3. [How LDA Works](#how-lda-works)
4. [Mathematical Foundation](#mathematical-foundation)
5. [The Maximum Components Constraint](#the-maximum-components-constraint)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)
7. [When to Use LDA vs PCA](#when-to-use-lda-vs-pca)
8. [Real-World Applications](#real-world-applications)
9. [Implementation Steps](#implementation-steps)
10. [Common Pitfalls](#common-pitfalls)

---

## What is LDA?

**Linear Discriminant Analysis** is a **supervised** dimensionality reduction technique that finds the linear combination of features that best separates the classes.

Unlike PCA, which ignores class labels and finds directions of maximum overall variance, LDA explicitly uses the labels to find directions that maximise **class separability** — the ratio of between-class spread to within-class spread.

**The practical implication:** LDA-reduced features are directly optimised for classification. PCA-reduced features are optimised for variance preservation, which may or may not be useful for the classification task.

**Dual role:** LDA is both a dimensionality reduction technique *and* a standalone classifier. When used as a classifier (not just for reduction), it assumes features are Gaussian with equal covariance matrices across classes.

---

## LDA vs PCA — The Key Difference

Consider a 2D dataset with two classes that are well-separated along one axis but highly variable along the other:

**PCA** finds the axis of maximum variance → may pick the axis where classes *overlap* most (if that axis has the highest variance).

**LDA** finds the axis that maximises class separation → will correctly identify the axis that separates the two classes.

**Concrete example:**
- Class A: values around 0 on x-axis, spread over 0–100 on y-axis
- Class B: values around 10 on x-axis, spread over 0–100 on y-axis

PCA would likely pick the y-axis (most variance). LDA would pick the x-axis (best class separation). For a subsequent classifier, LDA's choice is dramatically better.

---

## How LDA Works

### The Fisher Criterion
LDA maximises the **Fisher criterion** — the ratio of between-class scatter to within-class scatter:

```
J(w) = (wᵀ Sᵦ w) / (wᵀ Sᵥᵥ w)
```

Where:
- **Sᵦ (between-class scatter):** How far apart the class means are from the global mean
- **Sᵥᵥ (within-class scatter):** How spread out points are within each class
- **w:** The projection direction we are optimising

Maximising this ratio finds the direction where classes are far apart *relative to* how spread out they are within classes.

### Algorithm Steps

1. **Compute class means** μ₁, μ₂, ..., μₖ and global mean μ
2. **Compute within-class scatter matrix Sᵥᵥ:**
   Sum of covariance matrices within each class
3. **Compute between-class scatter matrix Sᵦ:**
   Weighted sum of squared distances from each class mean to the global mean
4. **Solve the generalised eigenvalue problem:**
   Sᵦ w = λ Sᵥᵥ w
5. **Select top discriminant directions** by eigenvalue magnitude
6. **Project data** onto selected discriminants

---

## Mathematical Foundation

### Between-Class Scatter Matrix

```
Sᵦ = Σₖ nₖ (μₖ − μ)(μₖ − μ)ᵀ
```

Where nₖ is the number of samples in class k. A large Sᵦ means class means are far from the global mean — classes are well-separated.

### Within-Class Scatter Matrix

```
Sᵥᵥ = Σₖ Σᵢ∈Cₖ (xᵢ − μₖ)(xᵢ − μₖ)ᵀ
```

A small Sᵥᵥ means data within each class is tightly clustered. We want large Sᵦ and small Sᵥᵥ simultaneously.

### The Generalised Eigenvalue Problem

```
Sᵥᵥ⁻¹ Sᵦ w = λ w
```

The eigenvectors of Sᵥᵥ⁻¹ Sᵦ are the Linear Discriminants. The eigenvalue λ is the Fisher criterion value — how well this direction separates the classes.

---

## The Maximum Components Constraint

**LDA is limited to at most (n_classes − 1) discriminant components.**

Why? The between-class scatter matrix Sᵦ has rank at most (K − 1) where K is the number of classes. This limits the number of useful directions LDA can find.

Practical implications:
- **2-class problem:** Maximum 1 LDA component (reduces to a single axis)
- **3-class problem:** Maximum 2 LDA components (can visualise in 2D)
- **10-class problem:** Maximum 9 LDA components

For datasets with many features and few classes, LDA produces far fewer components than PCA. This is not a weakness — it forces a compact discriminative representation.

**Contrast with PCA:** PCA can produce up to min(n_samples, n_features) components, but you choose how many to keep. LDA's maximum is fixed by the data.

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Class-aware | Explicitly maximises separation between known classes |
| Better for classification preprocessing | LDA components are directly optimised for the classification task |
| Compact output | At most K−1 components — naturally constrained to what's useful |
| Built-in classifier | LDA can be used directly as a probabilistic classifier (Gaussian assumption) |
| Works well with small datasets | The closed-form solution does not require iterative optimisation |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| Supervised — requires labels | Cannot be used in unsupervised contexts |
| Assumes Gaussian features | Each class's features should be approximately normally distributed |
| Assumes equal covariance matrices | If classes have very different spreads, QDA (Quadratic Discriminant Analysis) is more appropriate |
| Linear only | Cannot capture nonlinear class boundaries |
| Sensitive to class imbalance | Highly imbalanced classes affect the within-class scatter estimates |
| Limited components | At most K−1 discriminants — may be fewer than needed |

---

## When to Use LDA vs PCA

| Situation | Use |
|-----------|-----|
| Preprocessing for a downstream classifier | LDA — components are optimised for classification |
| Visualising class structure in 2D/3D | LDA — class separability is visible; PCA may show no separation |
| Unsupervised context (no labels) | PCA — LDA requires labels |
| More than K−1 components needed | PCA — LDA is capped |
| Nonlinear class boundaries | Kernel LDA or Kernel PCA |
| Features are not Gaussian | PCA — LDA's Gaussian assumption is violated |

**The pragmatic rule:** When performing supervised dimensionality reduction before a classifier, always try LDA first. If the dataset violates LDA's assumptions (non-Gaussian, unequal covariance), fall back to PCA.

---

## Real-World Applications

**Face Recognition:**
Fisherfaces (LDA applied to face images) outperformed Eigenfaces (PCA) for recognition tasks because LDA was trained to separate identities, not just explain variance in pixel values.

**Medical Diagnosis:**
Reduce many clinical measurements to a 1D discriminant score that best separates diseased from healthy patients. The score directly predicts disease probability.

**Text Document Classification:**
After TF-IDF vectorisation, LDA reduces the high-dimensional word space to (n_categories − 1) discriminant axes. This is much more compact than PCA while maintaining classification performance.

**Credit Scoring:**
Reduce many financial features to a single discriminant score separating defaulters from non-defaulters. The score is the basis for a credit decision.

---

## Implementation Steps

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# 1. Scale features (important for numerical stability)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)  # Max = n_classes - 1
X_train_lda = lda.fit_transform(X_train_scaled, y_train)   # Supervised: needs y
X_test_lda = lda.transform(X_test_scaled)

# 3. Inspect explained variance ratio
print(lda.explained_variance_ratio_)
print(f"Total variance explained: {lda.explained_variance_ratio_.sum():.3f}")

# 4. Use as classifier directly (optional)
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(X_train_scaled, y_train)
y_pred = lda_classifier.predict(X_test_scaled)

# 5. Visualise (for n_classes >= 3, 2 components can be plotted)
import matplotlib.pyplot as plt
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, alpha=0.7)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA: Training Set')
plt.show()
```

---

## Common Pitfalls

**1. Using LDA when classes are severely imbalanced**
The within-class scatter is estimated from each class separately. A class with very few samples gives a poor scatter estimate, distorting the discriminant direction. Resample or use class weights before applying LDA.

**2. Not scaling features**
LDA uses scatter matrices which are not scale-invariant. A feature with a range of 0–10,000 will dominate scatter matrices compared to a feature with range 0–1.

**3. Requesting more components than n_classes − 1**
`LinearDiscriminantAnalysis(n_components=5)` on a 3-class problem is silently capped at 2. Always check `lda.explained_variance_ratio_` to see what was actually computed.

**4. Applying `fit_transform` to test data**
LDA must be fit on training data only. Use `fit_transform` on train, `transform` on test.

**5. Confusing LDA-as-reducer with LDA-as-classifier**
Sklearn's `LinearDiscriminantAnalysis` can be used either way. When used as a reducer, you extract `lda.transform(X)` and feed the result to a separate classifier. When used as a classifier, you call `lda.fit(X, y)` and then `lda.predict(X_test)` directly. These are two different use modes.
