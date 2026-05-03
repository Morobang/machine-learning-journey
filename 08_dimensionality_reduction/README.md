# Dimensionality Reduction

Dimensionality reduction transforms a high-dimensional dataset into a lower-dimensional representation that preserves as much of the original information as possible. Instead of working with 50 correlated features, you work with 5 or 10 compressed features that capture the underlying structure.

---

## Why Dimensionality Reduction Matters

**The curse of dimensionality:** As the number of features grows, data becomes increasingly sparse. In high dimensions, all points appear roughly equidistant from each other — distance-based algorithms (KNN, SVM, K-Means) lose their ability to distinguish near from far neighbours. The amount of data needed to maintain the same sample density grows exponentially with dimensions.

**Correlated features are redundant.** If salary and years of experience are highly correlated, they carry mostly the same information. A model that uses both pays the cost of two features for the signal of one. Dimensionality reduction replaces correlated features with a compact, uncorrelated representation.

**Practical benefits of reducing dimensions:**
- Faster training (fewer features to process)
- Reduced overfitting (fewer parameters to memorise noise)
- Better visualisation (can plot 2D or 3D reduced data)
- Noise removal (later components capture mainly noise and can be discarded)

---

## Algorithms in This Section

### Principal Component Analysis (PCA)
**Notebook:** [01_principal_component_analysis.ipynb](notebooks/01_principal_component_analysis.ipynb) | **Guide:** [teaching/01_principal_component_analysis.md](teaching/01_principal_component_analysis.md)

PCA finds new axes — the **principal components** — that are linear combinations of the original features, ordered so that the first component explains the most variance, the second explains the second most, and so on. The components are orthogonal (uncorrelated) by construction.

PCA is **unsupervised**: it uses no class labels. It finds directions of maximum variance in the data regardless of whether those directions are predictive of any target.

```
Input: n samples × p features (scaled)
Output: n samples × k components  (k << p)
Each component is a weighted combination of all p original features
```

The **explained variance ratio** tells you how much of the original variance is retained in the reduced representation. A scree plot shows cumulative explained variance vs number of components — choose k at the "elbow" or where cumulative variance crosses 95%.

**Use when:** General preprocessing before a model, visualising high-dimensional data in 2D or 3D, compressing data, removing multicollinearity, or reducing noise by discarding the last (low-variance) components.

**Watch out for:** Always scale features before PCA — variance is scale-dependent. Do not fit PCA on the full dataset (data leakage) — fit on training data and transform both sets. PCA is linear; it cannot unfold nonlinear structure.

---

### Linear Discriminant Analysis (LDA)
**Notebook:** [02_linear_discriminant_analysis.ipynb](notebooks/02_linear_discriminant_analysis.ipynb) | **Guide:** [teaching/02_linear_discriminant_analysis.md](teaching/02_linear_discriminant_analysis.md)

LDA finds linear combinations of features that **maximise the separation between classes** — the ratio of between-class variance to within-class variance (the Fisher criterion). Unlike PCA, LDA uses the class labels.

LDA is **supervised**: the reduction is explicitly optimised for the classification task. When your goal is to build a classifier, LDA-reduced features are directly optimised for that goal, while PCA-reduced features are optimised for variance preservation (which may or may not help the classifier).

**The maximum components constraint:** LDA can produce at most (n_classes − 1) discriminant components. For binary classification: 1 component. For 5 classes: 4 components maximum. This is a hard limit set by the rank of the between-class scatter matrix.

**Use when:** Preprocessing before a classifier, especially when the class structure in the data has linear separability that PCA might miss. Visualising class structure in 2D (works well for 3+ class problems where 2 LDA components can be plotted).

**Watch out for:** Requires class labels (supervised). Assumes features are Gaussian with equal covariance across classes. Fitting on full dataset is data leakage — use training data only.

---

### Kernel PCA
**Notebook:** [03_kernel_pca.ipynb](notebooks/03_kernel_pca.ipynb) | **Guide:** [teaching/03_kernel_pca.md](teaching/03_kernel_pca.md)

Kernel PCA applies the **kernel trick** to PCA. Instead of finding linear components in the original feature space, it implicitly maps data into a very high-dimensional space (via a kernel function) and finds linear components there — which correspond to nonlinear components in the original space.

The **RBF (Gaussian) kernel** is the most common choice:

```
k(x, z) = exp(−γ × ||x − z||²)
```

This corresponds to an infinite-dimensional feature map. Yet the kernel trick means we never actually compute the high-dimensional representation — only pairwise distances in the original space.

**Use when:** Standard PCA fails to capture the data's structure because the data lies on a nonlinear manifold. Classic test cases: two concentric rings, a Swiss roll, interlocking crescents. If PCA produces a poor 2D visualisation (classes overlap completely), try Kernel PCA with RBF.

**Watch out for:** Requires scaling (RBF uses Euclidean distances). The O(n²) kernel matrix limits scalability to ~5,000 rows. The hyperparameter γ must be tuned via cross-validation.

---

## Choosing a Dimensionality Reduction Method

```
Is the task supervised (do you have class labels)?
├── Yes → Try LDA first (directly optimises class separation)
│         Maximum k = n_classes − 1 components
│         If LDA fails (non-Gaussian, unequal covariance) → PCA
└── No → Is the structure linear?
         ├── Yes → PCA (fast, interpretable, scalable)
         └── No → Kernel PCA with RBF kernel
                  If n > 5,000 → consider t-SNE or UMAP for visualisation only
```

**Start with PCA in all cases** — it is fast and interpretable. Only move to LDA or Kernel PCA if PCA fails to produce a useful representation.

---

## The Dataset

All three notebooks use the **Wine dataset** (or similar) — 13 chemical measurements from wine samples of 3 classes. The dataset is small enough to visualise clearly while having enough dimensions to demonstrate the benefit of reduction.

The 2D scatter plots after reduction show whether the three wine classes are well-separated in the reduced space — a direct visual assessment of whether the reduction preserved class structure.

---

## What Each Teaching Guide Covers

[teaching/01_principal_component_analysis.md](teaching/01_principal_component_analysis.md) — the covariance matrix, eigendecomposition, how principal components are computed, explained variance ratio, scree plots, and the connection to SVD.

[teaching/02_linear_discriminant_analysis.md](teaching/02_linear_discriminant_analysis.md) — the Fisher criterion, between-class and within-class scatter matrices, the generalised eigenvalue problem, why LDA is capped at n_classes−1, and comparison with PCA.

[teaching/03_kernel_pca.md](teaching/03_kernel_pca.md) — the kernel trick explained from first principles, common kernel functions and what they represent geometrically, kernel matrix centring, γ selection, and when Kernel PCA should be preferred over standard PCA.
