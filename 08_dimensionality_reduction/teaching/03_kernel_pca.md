# Kernel PCA — Complete Guide

## Table of Contents
1. [Why Standard PCA Fails on Nonlinear Data](#why-standard-pca-fails-on-nonlinear-data)
2. [The Kernel Trick](#the-kernel-trick)
3. [How Kernel PCA Works](#how-kernel-pca-works)
4. [Common Kernel Functions](#common-kernel-functions)
5. [Advantages and Disadvantages](#advantages-and-disadvantages)
6. [Kernel PCA vs PCA vs LDA](#kernel-pca-vs-pca-vs-lda)
7. [Real-World Applications](#real-world-applications)
8. [Implementation Steps](#implementation-steps)
9. [Common Pitfalls](#common-pitfalls)

---

## Why Standard PCA Fails on Nonlinear Data

Standard PCA finds linear directions of maximum variance. For data with nonlinear structure, this fails:

**Two concentric rings:** PCA will project both rings onto the same line, making them indistinguishable. There is no linear combination of x and y that separates them.

**Swiss roll data:** A 3D spiral embedded in 3D space. PCA flattens it into a tangled 2D projection. The true structure (a 2D manifold rolled up) is lost.

**Interlocking crescents:** Two half-moon shapes in 2D. Any linear projection overlaps them.

The fundamental limitation: **PCA looks for linear structure, but many real-world datasets have nonlinear manifold structure.**

Kernel PCA solves this by mapping the data into a higher-dimensional space where the nonlinear structure becomes linear.

---

## The Kernel Trick

### The Idea
Instead of working in the original feature space, we implicitly map data into a very high (possibly infinite) dimensional space φ(x) where the structure is linear.

In this high-dimensional space, standard PCA can find the linear principal components.

**The problem:** Explicitly computing φ(x) may be computationally infeasible for high-dimensional mappings.

### The Trick
We never actually compute φ(x). We only need **pairwise inner products** in the high-dimensional space:

```
k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩
```

The **kernel function** k(·,·) computes this inner product efficiently in the original space, without ever constructing φ(x) explicitly.

**Famous example:** The polynomial kernel k(x, z) = (xᵀz + c)^d corresponds to a feature map that includes all polynomial terms up to degree d — potentially billions of dimensions — but can be computed as a single scalar operation on the original vectors.

---

## How Kernel PCA Works

### Algorithm

1. **Compute the kernel matrix K** — an n×n matrix where Kᵢⱼ = k(xᵢ, xⱼ)
2. **Centre the kernel matrix** in feature space (analogous to subtracting the mean in standard PCA):
   ```
   K̃ = K − 1ₙK − K1ₙ + 1ₙK1ₙ
   ```
   Where 1ₙ is an n×n matrix of 1/n
3. **Eigendecompose K̃** — find eigenvalues λ and eigenvectors α
4. **Project data** onto the top k eigenvectors (scaled by eigenvalues)

### The Output
The output is the same as standard PCA — a lower-dimensional representation of the data — but computed via kernel functions that capture nonlinear relationships.

**Important difference from standard PCA:** Kernel PCA requires storing the entire training kernel matrix and computing kernel evaluations against training points at inference time. This makes it less scalable than standard PCA for very large datasets.

---

## Common Kernel Functions

| Kernel | Formula | Characteristic |
|--------|---------|----------------|
| **RBF / Gaussian** | exp(−γ||x−z||²) | Infinite-dimensional mapping; most flexible; controlled by γ |
| **Polynomial** | (xᵀz + c)^d | Maps to d-degree polynomial features; controlled by degree d |
| **Linear** | xᵀz | Equivalent to standard PCA (no nonlinearity added) |
| **Sigmoid** | tanh(γ xᵀz + c) | Similar to neural network activations; use carefully |

### The RBF Kernel and γ

The **RBF (Radial Basis Function) kernel** is the most commonly used with Kernel PCA:

```
k(x, z) = exp(−γ × ||x − z||²)
```

**Interpretation of γ:**
- **Small γ:** Wide Gaussian; neighbouring points have similar embeddings; smooth, global structure
- **Large γ:** Narrow Gaussian; only very close points are similar; captures local structure but can fragment clusters

**Choosing γ:** Use cross-validation or try values on a log scale (0.001, 0.01, 0.1, 1, 10). The optimal γ depends on the scale of your features — always scale features before applying RBF kernel.

---

## Advantages and Disadvantages

### Advantages

| Advantage | Detail |
|-----------|--------|
| Handles nonlinear structure | Can unfold manifolds that linear PCA cannot |
| Kernel trick avoids explicit mapping | Works in infinite-dimensional spaces efficiently |
| Flexible | RBF kernel can approximate any nonlinear structure given enough data |
| Same output format as PCA | Drop-in replacement in preprocessing pipelines |

### Disadvantages

| Disadvantage | Detail |
|-------------|--------|
| O(n²) kernel matrix | Must compute all n×n pairwise distances — slow for large datasets |
| O(n³) eigendecomposition | Further limits scalability |
| Hyperparameter sensitivity | Kernel choice and parameters (γ, degree) require tuning |
| No explicit feature map | Cannot recover which original features contributed to components |
| Inference cost | New points require computing kernel against all n training points |

---

## Kernel PCA vs PCA vs LDA

| Property | PCA | LDA | Kernel PCA |
|----------|-----|-----|-----------|
| Supervised? | No | Yes | No |
| Nonlinear? | No | No | Yes |
| Scalability | O(n × p²) | O(n × p²) | O(n²) kernel matrix |
| Max components | min(n, p) | n_classes − 1 | min(n, p) |
| Interpretability | Some | Some | Low |
| Best for | Correlated linear features | Supervised classification | Nonlinear manifold structure |

**Decision rule:**
1. Start with standard PCA — it is fast and interpretable
2. If PCA fails (poor visualisation, poor downstream accuracy), check whether the data has nonlinear structure
3. If nonlinear structure is present, try Kernel PCA with RBF
4. If the task is supervised classification, also consider Kernel SVM (which uses the same kernel trick but directly for classification)

---

## Real-World Applications

**Nonlinear Dimensionality Reduction:**
Datasets like MNIST digits or face images have nonlinear manifold structure. The "space of human faces" is not a flat hyperplane in pixel space — Kernel PCA with RBF kernel can unfold this manifold.

**Preprocessing for Nonlinear Classification:**
Before training a classifier on data with complex boundaries, Kernel PCA can linearise the structure so that a simple linear classifier achieves the same accuracy as a complex nonlinear one.

**Novelty/Anomaly Detection:**
Train Kernel PCA on normal data. New points that cannot be well-reconstructed (high reconstruction error) are potential anomalies — they lie far from the nonlinear manifold of normal data.

**Bioinformatics:**
Gene expression data often lies on nonlinear manifolds (different cell types, differentiation pathways). Kernel PCA with RBF reveals cell subpopulations that standard PCA might not separate.

---

## Implementation Steps

```python
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# 1. Scale features (especially important for RBF kernel which uses distances)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Apply Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_train_kpca = kpca.fit_transform(X_train_scaled)
X_test_kpca = kpca.transform(X_test_scaled)

# 3. Try different gamma values (if not using cross-validation)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA(kernel='rbf')),
    ('clf', LogisticRegression())
])

param_grid = {
    'kpca__n_components': [2, 5, 10],
    'kpca__gamma': [0.001, 0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best params:", grid_search.best_params_)

# 4. Visualise the 2D embedding
import matplotlib.pyplot as plt
kpca_vis = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_vis = kpca_vis.fit_transform(X_train_scaled)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, alpha=0.7)
plt.title('Kernel PCA Embedding (RBF)')
plt.show()
```

---

## Common Pitfalls

**1. Not scaling features before RBF kernel**
The RBF kernel computes Euclidean distances. Features on different scales produce meaningless distances. Always StandardScaler first.

**2. Using γ=1 without checking**
The default gamma in sklearn is 1/n_features. This is often not optimal. Always cross-validate γ when performance matters.

**3. Trying Kernel PCA on a large dataset without approximation**
The O(n²) kernel matrix and O(n³) eigendecomposition make standard Kernel PCA infeasible for n > ~5,000. For large datasets, use approximate kernel methods (Nyström approximation, Random Fourier Features) or switch to t-SNE/UMAP for visualisation.

**4. Expecting Kernel PCA to always outperform PCA**
For data that is actually linear, Kernel PCA adds computation without benefit. Always benchmark against standard PCA first.

**5. Confusing Kernel PCA with t-SNE or UMAP**
Kernel PCA, t-SNE, and UMAP all produce nonlinear embeddings, but for different purposes:
- Kernel PCA: preprocessing for downstream ML (embedding preserves global structure)
- t-SNE/UMAP: visualisation only (preserves local structure; embedding is not meaningful for prediction)

Never use t-SNE features as input to a classifier — use Kernel PCA instead.
