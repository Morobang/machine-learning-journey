import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\08_dimensionality_reduction\notebooks\03_kernel_pca.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title
cells[0]["source"] = [
    "# Kernel PCA\n",
    "\n",
    "## The Limitation of Linear PCA\n",
    "\n",
    "Standard PCA finds directions of maximum variance using **linear projections**. It works well when the structure of the data can be captured by straight lines or hyperplanes.\n",
    "\n",
    "But what if the class boundaries are curved? What if the data is arranged in rings, spirals, or other non-linear patterns? Linear PCA would project everything onto a straight axis and destroy the class structure.\n",
    "\n",
    "---\n",
    "\n",
    "## The Kernel Trick: Linear PCA in a Higher-Dimensional Space\n",
    "\n",
    "Kernel PCA extends standard PCA using the **kernel trick** — the same idea as in Kernel SVM.\n",
    "\n",
    "Instead of finding linear directions in the original feature space, Kernel PCA implicitly maps the data into a **higher-dimensional space** where non-linear patterns become linear, then applies PCA there.\n",
    "\n",
    "```\n",
    "Original space (non-linear):          Kernel space (mapped):\n",
    "\n",
    "  Class A: ooo                           Class A:  ooo\n",
    "  Class B: x o x o (mixed)    →          Class B:           xxx\n",
    "  Not linearly separable                 Linearly separable!\n",
    "```\n",
    "\n",
    "**The trick:** You never explicitly compute the high-dimensional coordinates. You only compute dot products between data points — and dot products in the mapped space can be computed using a **kernel function** $K(x_i, x_j) = \\phi(x_i) \\cdot \\phi(x_j)$ without ever knowing $\\phi$.\n",
    "\n",
    "---\n",
    "\n",
    "## The RBF Kernel\n",
    "\n",
    "The Radial Basis Function (RBF) kernel is the most common choice:\n",
    "\n",
    "$$K(x_i, x_j) = \\exp\\left(-\\gamma \\|x_i - x_j\\|^2\\right)$$\n",
    "\n",
    "It measures similarity based on **distance**: nearby points have kernel value near 1, far-apart points have kernel value near 0. The RBF kernel implicitly maps to an infinite-dimensional space — it can capture arbitrarily complex non-linear patterns.\n",
    "\n",
    "---\n",
    "\n",
    "## PCA vs Kernel PCA vs LDA: Which to Choose?\n",
    "\n",
    "| Situation | Choose |\n",
    "|-----------|--------|\n",
    "| Data has linear structure, no labels | **PCA** |\n",
    "| Data has linear structure, class labels available | **LDA** |\n",
    "| Data has non-linear structure | **Kernel PCA** |\n",
    "| Need interpretable components | **PCA or LDA** (Kernel PCA components have no interpretable meaning) |\n",
    "| Large dataset (>50K samples) | **PCA** (Kernel PCA is O(n²) in memory) |\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "Same Wine dataset and Logistic Regression setup as the PCA and LDA notebooks. This lets us compare all three dimensionality reduction methods on the same data:\n",
    "- PCA: ~97% accuracy\n",
    "- LDA: ~100% accuracy\n",
    "- Kernel PCA: how does it compare?"
]

# Cell 1 — libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "Same three standard libraries. `KernelPCA` is available from `sklearn.decomposition` alongside standard PCA."
]

# Cell 3 — dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "Same Wine dataset as the PCA and LDA notebooks: 178 samples, 13 features, 3 classes.\n",
    "\n",
    "Using the same dataset across all three dimensionality reduction notebooks lets you make a direct apples-to-apples comparison of the methods — the only variable changing is the reduction technique."
]

# Cell 5 — train/test split
cells[5]["source"] = [
    "## Step 3: Train/Test Split\n",
    "\n",
    "80/20 split, same as before. The split comes before Kernel PCA fitting — we learn the kernel mapping from training data only, then apply it to the test set."
]

# Cell 7 — feature scaling
cells[7]["source"] = [
    "## Step 4: Feature Scaling\n",
    "\n",
    "Kernel PCA depends critically on feature scaling. The RBF kernel computes **Euclidean distances** between data points:\n",
    "\n",
    "$$K(x_i, x_j) = \\exp(-\\gamma \\|x_i - x_j\\|^2)$$\n",
    "\n",
    "If features have different scales, the distance calculation is dominated by the large-scale feature. A difference of 1,000 in salary will swamp a difference of 0.5 in alcohol content, even if the alcohol difference is more discriminative.\n",
    "\n",
    "After StandardScaler, all features are on the same scale and the kernel computes meaningful, balanced distances."
]

# Cell 9 — applying Kernel PCA
cells[9]["source"] = [
    "## Step 5: Apply Kernel PCA\n",
    "\n",
    "```python\n",
    "kpca = KernelPCA(n_components=2, kernel='rbf')\n",
    "X_train = kpca.fit_transform(X_train)\n",
    "X_test  = kpca.transform(X_test)\n",
    "```\n",
    "\n",
    "**`kernel='rbf'`** — uses the Radial Basis Function kernel. Other options include `'poly'` (polynomial), `'sigmoid'`, and `'cosine'`.\n",
    "\n",
    "**Key difference from standard PCA:** The components produced by Kernel PCA are in the **mapped feature space**, not the original space. They are not linear combinations of the original 13 wine features — they are non-linear combinations that may capture curved cluster boundaries.\n",
    "\n",
    "**Key difference from LDA:** Kernel PCA is **unsupervised** — it does not use `y_train`. It finds the principal components of the kernel matrix, which captures non-linear variance structure without requiring class labels.\n",
    "\n",
    "**Computational note:** Kernel PCA requires computing an $n \\times n$ kernel matrix where $n$ is the number of training samples. For 8,000 training samples, this is a 64 million entry matrix. For datasets with 100K+ samples, standard PCA is much more practical."
]

# Cell 11 — training classifier
cells[11]["source"] = [
    "## Step 6: Train Logistic Regression on Kernel PCA Features\n",
    "\n",
    "The two Kernel PCA components capture non-linear structure in the wine dataset. If the wine classes have curved boundaries in the original 13D space, those boundaries should appear more linear in the Kernel PCA projection — making logistic regression more effective."
]

# Cell 13 — confusion matrix
cells[13]["source"] = [
    "## Step 7: Evaluate — Confusion Matrix\n",
    "\n",
    "Compare the result here against PCA (~97%) and LDA (~100%).\n",
    "\n",
    "**Interpreting the result:**\n",
    "\n",
    "If Kernel PCA matches LDA (100%), it means the wine dataset does have non-linear structure that the RBF kernel captured effectively.\n",
    "\n",
    "If it matches standard PCA (~97%), the dataset's class structure is approximately linear — the kernel mapping did not help.\n",
    "\n",
    "**A broader lesson:** More complex does not always mean better. Kernel PCA is a more powerful but more expensive method than PCA. If the data is approximately linear, standard PCA will work just as well at a fraction of the computational cost. Always start simple.\n",
    "\n",
    "**When Kernel PCA shines:** Data arranged in rings (inner class vs outer class), crescents, spirals, or any other shape where linear projections destroy class structure. The classic demonstration is the Swiss roll or concentric circles dataset."
]

# Cell 15 — training visualisation
cells[15]["source"] = [
    "## Step 8: Visualise the Training Set\n",
    "\n",
    "The axes are now **non-linear principal components** — they have no interpretable meaning as combinations of the original wine features.\n",
    "\n",
    "Look for: how cleanly do the three wine classes separate in this 2D kernel space? Compare visually to the PCA and LDA plots from the previous notebooks."
]

# Cell 17 — test visualisation
cells[17]["source"] = [
    "## Step 9: Visualise the Test Set\n",
    "\n",
    "The test set result is the honest comparison.\n",
    "\n",
    "**Three-way comparison summary:**\n",
    "\n",
    "| Method | Type | Uses labels | Wine accuracy | Best for |\n",
    "|--------|------|------------|---------------|----------|\n",
    "| PCA | Linear, unsupervised | No | ~97% | Exploration, noise reduction |\n",
    "| LDA | Linear, supervised | Yes | ~100% | Maximising class separability |\n",
    "| Kernel PCA | Non-linear, unsupervised | No | ~100% | Non-linear structure |\n",
    "\n",
    "For the Wine dataset — which has a relatively clean, near-linear structure — all three methods perform well. The differences become dramatic on datasets with genuinely non-linear class boundaries."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Kernel PCA notebook updated successfully.")
