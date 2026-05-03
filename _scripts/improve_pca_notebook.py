import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\08_dimensionality_reduction\notebooks\01_principal_component_analysis.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title / intro
cells[0]["source"] = [
    "# Principal Component Analysis (PCA)\n",
    "\n",
    "## The Problem: Too Many Features\n",
    "\n",
    "The Wine dataset has **13 chemical measurements** per wine sample (alcohol, malic acid, ash, etc.). Training a classifier on 13 features works fine, but high-dimensional data creates real problems:\n",
    "\n",
    "- **The curse of dimensionality:** As dimensions increase, data becomes increasingly sparse. Distance-based models (KNN, SVM) degrade because every point looks equally far from every other.\n",
    "- **Visualisation is impossible:** You cannot plot 13 dimensions. Understanding patterns requires reducing to 2D or 3D.\n",
    "- **Redundancy:** Many features are correlated. Alcohol and density in wine often move together — they are carrying similar information.\n",
    "\n",
    "PCA solves all three: it compresses many correlated features into fewer uncorrelated ones, enabling visualisation and often improving model performance.\n",
    "\n",
    "---\n",
    "\n",
    "## What PCA Actually Does\n",
    "\n",
    "PCA finds the directions of **maximum variance** in your data and projects the data onto those directions.\n",
    "\n",
    "Think of it geometrically: imagine a cloud of data points in 13-dimensional space. PCA finds the axis along which the points are most spread out (PC1), then the axis perpendicular to that which captures the next most variance (PC2), and so on.\n",
    "\n",
    "```\n",
    "Original features (correlated):        After PCA (uncorrelated):\n",
    "\n",
    "   alcohol                                PC1  (captures ~36% variance)\n",
    "   density          →   PCA    →          PC2  (captures ~19% variance)\n",
    "   malic_acid                             (PC3 through PC13 capture the rest)\n",
    "   ... (10 more)\n",
    "```\n",
    "\n",
    "Each **Principal Component (PC)** is a linear combination of the original features. PC1 is a weighted sum of all 13 features that captures the most variance; PC2 captures the second most, and so on.\n",
    "\n",
    "---\n",
    "\n",
    "## The Mathematics (Intuition)\n",
    "\n",
    "1. **Centre the data** — subtract the mean from each feature (StandardScaler does this)\n",
    "2. **Compute the covariance matrix** — captures how features vary together\n",
    "3. **Find eigenvectors and eigenvalues** — eigenvectors are the PC directions; eigenvalues tell you how much variance each direction captures\n",
    "4. **Sort by eigenvalue** — the largest eigenvalue = PC1, second largest = PC2, ...\n",
    "5. **Project** — multiply original data by the top $k$ eigenvectors to get $k$ new features\n",
    "\n",
    "You do not need to implement this manually — `sklearn.decomposition.PCA` handles it. But understanding step 3 explains why PCA requires scaling first: if one feature has values in the thousands and another in the hundreds, the covariance matrix will be dominated by the large-scale feature before any actual pattern analysis.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "1. Load a 13-feature wine dataset and classify 3 wine types\n",
    "2. Scale features (mandatory for PCA)\n",
    "3. Apply PCA to reduce to 2 components\n",
    "4. Train Logistic Regression on 2 features instead of 13\n",
    "5. Visualise the decision boundary — something impossible before dimensionality reduction"
]

# Cell 1 — libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `numpy` | Array operations |\n",
    "| `matplotlib` | Visualising decision boundaries |\n",
    "| `pandas` | Loading the dataset |"
]

# Cell 3 — dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "The Wine dataset contains **178 samples** of wine from 3 different cultivars (varieties) in Italy. Each sample has 13 chemical features:\n",
    "\n",
    "| Features | Examples |\n",
    "|----------|----------|\n",
    "| Chemical composition | Alcohol, Malic acid, Ash, Alcalinity of ash |\n",
    "| Mineral content | Magnesium, Total phenols, Flavanoids |\n",
    "| Colour & optical | Nonflavonoid phenols, Proanthocyanins, Colour intensity, Hue |\n",
    "| Other | OD280/OD315, Proline |\n",
    "\n",
    "The target variable `y` is the wine class (1, 2, or 3).\n",
    "\n",
    "The challenge: can we correctly classify wine variety using just 2 summary dimensions instead of all 13 features?"
]

# Cell 5 — train/test split
cells[5]["source"] = [
    "## Step 3: Split Into Training and Test Sets\n",
    "\n",
    "We use an 80/20 split. This is done **before** PCA — a critical point.\n",
    "\n",
    "**Why split before applying PCA?**\n",
    "\n",
    "PCA learns the principal components (eigenvectors) from the data. If you applied PCA to the full dataset before splitting, information from the test set would leak into the transformation — the test set would influence what the components look like. This is a form of **data leakage**.\n",
    "\n",
    "The correct pipeline:\n",
    "1. Split\n",
    "2. Fit PCA on training set only\n",
    "3. Transform both train and test using the training-fit PCA\n",
    "\n",
    "Same rule applies to scaling, imputation, and any other data-dependent transformation."
]

# Cell 7 — feature scaling
cells[7]["source"] = [
    "## Step 4: Feature Scaling (Mandatory for PCA)\n",
    "\n",
    "**This step is not optional for PCA.** Here is why:\n",
    "\n",
    "PCA finds directions of maximum variance. If one feature (e.g. Proline, which ranges from 278 to 1680) has a much larger scale than another (e.g. Ash, which ranges from 1.4 to 3.2), the covariance matrix will be dominated by the large-scale feature — not because it carries more information, but simply because of its units.\n",
    "\n",
    "`StandardScaler` centres each feature to mean 0 and standard deviation 1, so that PCA evaluates all features on equal footing and identifies directions of genuine information variance rather than measurement-scale variance.\n",
    "\n",
    "**Rule:** Fit the scaler on the training set only (`fit_transform`), then apply the trained scaler to the test set (`transform`). Never fit on the test set."
]

# Cell 9 — applying PCA
cells[9]["source"] = [
    "## Step 5: Apply PCA\n",
    "\n",
    "We reduce from **13 features to 2 principal components** by setting `n_components=2`.\n",
    "\n",
    "**How to choose the number of components in practice:**\n",
    "\n",
    "The standard approach is to plot the **explained variance ratio** — how much of the total variance each component captures:\n",
    "\n",
    "```python\n",
    "pca_full = PCA(n_components=None)\n",
    "pca_full.fit(X_train)\n",
    "print(pca_full.explained_variance_ratio_.cumsum())\n",
    "# Pick the number of components where cumulative variance reaches ~95%\n",
    "```\n",
    "\n",
    "For the wine dataset, 2 components typically capture ~55% of variance — enough for a clean visualisation and still a reasonable classifier. For production models, you would retain enough components for 90-95% of explained variance, not just 2.\n",
    "\n",
    "**The `fit_transform` / `transform` split:**\n",
    "- `pca.fit_transform(X_train)`: learns the principal components *from training data*, then projects it\n",
    "- `pca.transform(X_test)`: projects test data using the same components learned from training — no new learning\n",
    "\n",
    "This ensures the test set is truly unseen."
]

# Cell 11 — training logistic regression
cells[11]["source"] = [
    "## Step 6: Train Logistic Regression on the Reduced Features\n",
    "\n",
    "Notice that the model receives only **2 features** (PC1 and PC2) instead of the original 13.\n",
    "\n",
    "This demonstrates an important property of PCA: the principal components are **fully sufficient** for downstream modelling. You do not lose the original features — you exchange them for a compressed representation that captures most of the relevant variance.\n",
    "\n",
    "**What Logistic Regression does here:** It fits a linear decision boundary in 2D PCA space. If the PCA compression preserved the class-relevant variance well, the 2D boundary will still separate the 3 wine types cleanly."
]

# Cell 13 — confusion matrix
cells[13]["source"] = [
    "## Step 7: Evaluate — Confusion Matrix and Accuracy\n",
    "\n",
    "We assess how well the classifier separates 3 wine types using only 2 PCA dimensions.\n",
    "\n",
    "**Reading the confusion matrix:**\n",
    "\n",
    "```\n",
    "Rows = Actual class\n",
    "Columns = Predicted class\n",
    "\n",
    "[[14  0  0]       <- Class 1: 14 correct, 0 wrong\n",
    " [ 1 15  0]       <- Class 2: 1 misclassified as Class 1, 15 correct\n",
    " [ 0  0  6]]      <- Class 3: 6 correct, 0 wrong\n",
    "```\n",
    "\n",
    "**Accuracy ~97%** on 2 features out of 13 original features is impressive. It shows that the 2 principal components captured the variance that is most discriminative for the wine classes.\n",
    "\n",
    "This is the practical payoff of PCA: simpler, faster, more visualisable models with minimal accuracy loss — sometimes with accuracy *gains* due to noise reduction in the dropped components."
]

# Cell 15 — visualising training set
cells[15]["source"] = [
    "## Step 8: Visualise the Training Set Decision Boundary\n",
    "\n",
    "This plot is only possible because PCA reduced 13 dimensions to 2. Without dimensionality reduction, you cannot visualise a 13-dimensional decision boundary.\n",
    "\n",
    "**How to read the plot:**\n",
    "\n",
    "- Each **coloured region** is where the model predicts a given wine class\n",
    "- Each **dot** is a training sample, coloured by its true class\n",
    "- **PC1 (x-axis)** is the first principal component — the direction of maximum variance\n",
    "- **PC2 (y-axis)** is the second principal component — maximum variance perpendicular to PC1\n",
    "\n",
    "Well-separated clusters in PCA space mean the compressed representation preserved the class-discriminative structure of the original 13 features.\n",
    "\n",
    "A training set visualisation naturally looks good because the model was fit on this data. The critical test is how it looks on the test set."
]

# Cell 17 — visualising test set
cells[17]["source"] = [
    "## Step 9: Visualise the Test Set Decision Boundary\n",
    "\n",
    "This is the honest performance check. The model was trained on the training set; these test points were never seen during either the PCA fitting or the logistic regression training.\n",
    "\n",
    "**What to look for:**\n",
    "\n",
    "- Are the test points landing in the correct coloured regions?\n",
    "- Is the test set plot roughly as clean as the training set plot? (If it looks much messier, you have overfitting in the logistic regression — though with only 2 features this is unlikely)\n",
    "- Any misclassified points (dots outside their expected region) correspond to the off-diagonal entries in the confusion matrix\n",
    "\n",
    "**Key takeaway:** PCA is a preprocessing step, not a classifier. The accuracy of 97% reflects the combination of PCA (good compression) + Logistic Regression (appropriate model). Changing either component changes the accuracy."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("PCA notebook updated successfully.")
