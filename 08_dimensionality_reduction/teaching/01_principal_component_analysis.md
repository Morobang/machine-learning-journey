# Principal Component Analysis (PCA) — Complete Guide

## Table of Contents
1. [What is PCA?](#what-is-pca)
2. [The Curse of Dimensionality](#the-curse-of-dimensionality)
3. [How PCA Works](#how-pca-works)
4. [Mathematical Foundation](#mathematical-foundation)
5. [Choosing How Many Components](#choosing-how-many-components)
6. [Advantages and Disadvantages](#advantages-and-disadvantages)
7. [PCA vs LDA vs Kernel PCA](#pca-vs-lda-vs-kernel-pca)
8. [Real-World Applications](#real-world-applications)
9. [Implementation Steps](#implementation-steps)
10. [Common Pitfalls](#common-pitfalls)

---

## What is PCA?

**Principal Component Analysis** is an **unsupervised** dimensionality reduction technique that transforms a high-dimensional dataset into a lower-dimensional space while retaining as much variance (information) as possible.

**The problem PCA solves:** You have 50 features. Many are correlated (e.g., salary correlates with experience). You want a smaller set of features that captures most of the variance, removes redundancy, and enables faster, more stable model training.

**The key insight:** Correlated features can be replaced by uncorrelated linear combinations of those features (the "principal components") that explain the variance more efficiently.

**PCA is unsupervised** — it does not use the labels when finding the new axes. It finds directions of maximum variance in the data, regardless of whether those directions are predictive of any particular target. This is PCA's main distinction from LDA, which *does* use labels.

---

## The Curse of Dimensionality

As the number of features (dimensions) grows:

1. **Data becomes sparse:** In high dimensions, all data points are approximately equidistant. Distance-based algorithms (KNN, SVM, K-Means) lose their ability to distinguish near from far neighbours.
2. **Exponential data requirements:** To maintain the same sample density, the amount of data needed grows exponentially with dimensions.
3. **Overfitting risk:** More features provide more opportunities to fit noise. A model with 1000 features and 500 samples will overfit even if only 10 features are truly informative.
4. **Computational cost:** Training time and memory grow with dimensionality.

PCA addresses all four by compressing many correlated features into a small number of uncorrelated principal components.

---

## How PCA Works

### Step 1: Standardise the Data
PCA is variance-based. Features with large numerical ranges will dominate the principal components. Always standardise to zero mean and unit variance before PCA.

### Step 2: Compute the Covariance Matrix
The covariance matrix captures how features vary together:

```
Cov(X) = (1/n) × Xᵀ × X   (for zero-mean X)
```

Each element Cov(i,j) tells you how much feature i and feature j move together. A large positive value means they are positively correlated; near zero means they are independent.

### Step 3: Eigendecomposition
Decompose the covariance matrix into eigenvalues and eigenvectors:

```
Cov(X) × v = λ × v
```

- **Eigenvector v:** A direction in the original feature space
- **Eigenvalue λ:** The variance of the data when projected onto direction v

The eigenvectors are the **principal components** — the new axes. They are orthogonal (uncorrelated) to each other.

### Step 4: Sort by Variance Explained
Sort eigenvectors by their eigenvalues (descending). The first principal component (PC1) explains the most variance; PC2 explains the second most; and so on.

### Step 5: Project the Data
Select the top k principal components and project the data:

```
X_reduced = X × W
```

Where W is the matrix whose columns are the top k eigenvectors.

---

## Mathematical Foundation

### The Objective
PCA finds the direction w₁ that maximises the variance of projections:

```
w₁ = argmax_w { wᵀ Cov(X) w }  subject to ||w|| = 1
```

The solution is the eigenvector corresponding to the largest eigenvalue of Cov(X).

### Explained Variance Ratio
Each principal component explains a fraction of total variance:

```
Explained variance ratio of PCₖ = λₖ / Σᵢ λᵢ
```

Cumulative explained variance shows how much total variance the first k components capture:

```
If PC1 explains 45%, PC2 explains 30%, together they explain 75%
```

### Singular Value Decomposition (SVD)
In practice, PCA is computed via SVD rather than eigendecomposition of the covariance matrix:

```
X = U × Σ × Vᵀ
```

Where columns of V are the principal components. SVD is numerically more stable and handles datasets where p > n (more features than samples).

---

## Choosing How Many Components

The **scree plot** and **cumulative explained variance plot** guide this choice:

1. Plot cumulative explained variance vs number of components
2. Choose k where the curve reaches an acceptable threshold (commonly 95% or 99%)
3. Alternatively, look for an "elbow" where adding more components gives diminishing returns

**No single correct answer:** The choice depends on your application:
- For visualisation: k=2 or k=3 (can plot in 2D/3D)
- For preprocessing before a classifier: often k that explains 95% of variance
- For compression: k set by the storage/compute budget

```python
# Find k for 95% explained variance
from sklearn.decomposition import PCA
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
k = (cumulative_variance < 0.95).sum() + 1
```

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Removes multicollinearity | Principal components are orthogonal (uncorrelated) by construction |
| Reduces overfitting | Fewer features → less opportunity to fit noise |
| Fast and scalable | Linear algebra operations; incremental PCA for out-of-core data |
| Noise reduction | Later components capture mainly noise; dropping them cleans the signal |
| Enables visualisation | Reduce to 2D/3D to visualise high-dimensional data |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| Unsupervised | Directions of maximum variance are not necessarily predictive of the target |
| Interpretability loss | Principal components are linear combinations of all original features — hard to interpret |
| Assumes linearity | Cannot capture nonlinear structure (use Kernel PCA instead) |
| Sensitive to scaling | Must standardise first; sensitive to outliers |
| Information loss | Any dimensionality reduction discards some information |

---

## PCA vs LDA vs Kernel PCA

| Property | PCA | LDA | Kernel PCA |
|----------|-----|-----|------------|
| Supervised? | No | Yes | No |
| Objective | Maximise variance | Maximise class separability | Maximise variance in kernel space |
| Linear? | Yes | Yes | No |
| Good for | General dimensionality reduction | Classification preprocessing | Nonlinear structure |
| Max components | min(n_samples, n_features) | n_classes − 1 | min(n_samples, n_features) |

**When to use PCA:** General preprocessing, visualisation, compression, when you have many correlated features.

**When to use LDA instead:** When your primary goal is classification and you want to explicitly maximise class separation in the reduced space.

**When to use Kernel PCA:** When the data has nonlinear structure that PCA cannot unfold.

---

## Real-World Applications

**Face Recognition:**
The famous "Eigenfaces" — PCA applied to face images. Each face is a vector of pixel values. PCA finds the directions of maximum variation across faces (eigenfaces). A new face is represented by its projection onto the top eigenfaces; recognition is nearest-neighbour in this compressed space.

**Gene Expression:**
RNA sequencing produces tens of thousands of gene expression measurements per sample. PCA reduces this to a handful of components that capture major sources of variation (cell type, treatment condition, batch effects).

**Financial Portfolio Analysis:**
Stock returns are highly correlated. PCA finds uncorrelated "factors" that drive market movements (market risk, sector risk, style factors). These are related to but more data-driven than the Fama-French factors.

**Image Compression:**
Store images as projections onto the top principal components rather than raw pixels. Much smaller storage; reconstruction quality depends on how many components are retained.

---

## Implementation Steps

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Always scale first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Fit PCA and choose components
pca = PCA(n_components=2)  # Or n_components=0.95 to explain 95% variance
X_train_pca = pca.fit_transform(X_scaled)   # Fit on train only
X_test_pca = pca.transform(X_test_scaled)   # Transform test with train's PCA

# 3. Inspect explained variance
print(pca.explained_variance_ratio_)
print(f"Total explained: {pca.explained_variance_ratio_.sum():.3f}")

# 4. Scree plot
import matplotlib.pyplot as plt
pca_full = PCA()
pca_full.fit(X_scaled)
plt.plot(pca_full.explained_variance_ratio_.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(0.95, ls='--', color='red')
plt.show()
```

**Critical:** `fit_transform` on training data, `transform` only on test data. PCA learns the principal component directions from training data. Applying it to test data uses those same directions — correctly treating test as unseen.

---

## Common Pitfalls

**1. Forgetting to scale before PCA**
A feature with values in thousands will dominate the first principal component regardless of its actual importance. Always StandardScaler first.

**2. Applying PCA to categorical features**
PCA is defined for continuous numerical data. One-hot encoded features can be used but the results are less meaningful. Consider other encodings or use MCA (multiple correspondence analysis) for purely categorical data.

**3. Fitting PCA on the full dataset (data leakage)**
The PCA rotation must be computed on training data only. Fitting on all data causes the test set to influence the principal component directions — a subtle form of leakage.

**4. Using PCA to fix multicollinearity without checking if it helps the model**
PCA always removes multicollinearity, but it does not always improve model accuracy. On small datasets with few features, plain logistic regression may outperform PCA + logistic regression. Always compare.

**5. Interpreting principal components as features**
PC1 is "the direction of most variance in the data" — not necessarily "the most important feature for prediction." Do not name components after the original features they happen to correlate with unless you have verified the relationship is stable.
