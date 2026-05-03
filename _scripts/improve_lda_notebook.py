import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\08_dimensionality_reduction\notebooks\02_linear_discriminant_analysis.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title
cells[0]["source"] = [
    "# Linear Discriminant Analysis (LDA)\n",
    "\n",
    "## LDA vs PCA: The Key Difference\n",
    "\n",
    "Both PCA and LDA reduce dimensions. But they optimise for completely different things:\n",
    "\n",
    "| | PCA | LDA |\n",
    "|-|-----|-----|\n",
    "| **Goal** | Maximise variance in the data | Maximise class separability |\n",
    "| **Uses labels?** | No — unsupervised | Yes — supervised |\n",
    "| **Finds directions that...** | spread data out the most | push classes apart |\n",
    "| **Maximum components** | min(n_samples, n_features) - 1 | n_classes - 1 |\n",
    "\n",
    "---\n",
    "\n",
    "## The Intuition\n",
    "\n",
    "PCA does not know what your classes are. It just finds the directions of highest variance in the data — which may or may not help a classifier.\n",
    "\n",
    "LDA **uses the class labels** to find the directions that maximally separate the classes. It simultaneously:\n",
    "1. **Maximises** the distance between class means (between-class scatter)\n",
    "2. **Minimises** the spread within each class (within-class scatter)\n",
    "\n",
    "```\n",
    "PCA projection:                LDA projection:\n",
    "\n",
    "  Class A: ooo         →       Class A: ooo\n",
    "  Class B: xxx                 Class B:           xxx\n",
    "  Both overlapping             Cleanly separated\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "## The Mathematics (Intuition)\n",
    "\n",
    "LDA solves for the projection matrix $W$ that maximises the **Fisher criterion**:\n",
    "\n",
    "$$J(W) = \\frac{W^T S_B W}{W^T S_W W}$$\n",
    "\n",
    "Where:\n",
    "- $S_B$ = **between-class scatter matrix** — measures how far apart the class means are\n",
    "- $S_W$ = **within-class scatter matrix** — measures how spread out samples are within each class\n",
    "\n",
    "Maximising this ratio = maximising separation while minimising overlap.\n",
    "\n",
    "---\n",
    "\n",
    "## When to Use LDA Over PCA\n",
    "\n",
    "| Situation | Choose |\n",
    "|-----------|--------|\n",
    "| You have class labels and want maximum separability | **LDA** |\n",
    "| You want to visualise class clusters | **LDA** |\n",
    "| You are doing unsupervised or exploratory analysis | **PCA** |\n",
    "| You have few samples per class (LDA can overfit) | **PCA** |\n",
    "| Classes are not linearly separable | **Kernel PCA** |\n",
    "\n",
    "**Maximum components:** LDA produces at most $K - 1$ components where $K$ is the number of classes. With 3 wine classes, LDA gives maximum 2 components — exactly what we use here.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "Same setup as the PCA notebook (Wine dataset, Logistic Regression classifier), but using LDA for dimensionality reduction. The comparison will show whether LDA's class-aware approach gives better separation than PCA's variance-maximisation."
]

# Cell 1 — libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "Same three libraries as PCA — LDA is available from `sklearn.discriminant_analysis`."
]

# Cell 3 — dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "Same Wine dataset as the PCA notebook: 178 samples, 13 chemical features, 3 wine classes.\n",
    "\n",
    "Using the same dataset as PCA lets us **directly compare** the two dimensionality reduction methods — same data, same downstream classifier, only the reduction technique changes."
]

# Cell 5 — train/test split
cells[5]["source"] = [
    "## Step 3: Train/Test Split\n",
    "\n",
    "Same 80/20 split. The split must come before LDA — the discriminant directions must be learned from training data only.\n",
    "\n",
    "**Unlike PCA, LDA uses the labels `y_train` during fitting.** This is what makes it supervised. The labels are used to compute between-class and within-class scatter matrices."
]

# Cell 7 — feature scaling
cells[7]["source"] = [
    "## Step 4: Feature Scaling\n",
    "\n",
    "LDA also requires feature scaling, for the same reason as PCA: the scatter matrices ($S_B$ and $S_W$) are built from covariances. Features on vastly different scales will dominate the scatter calculation regardless of their true discriminative power.\n",
    "\n",
    "StandardScaler ensures all features contribute equally to the scatter matrices."
]

# Cell 9 — applying LDA
cells[9]["source"] = [
    "## Step 5: Apply LDA\n",
    "\n",
    "```python\n",
    "lda = LDA(n_components=2)\n",
    "X_train = lda.fit_transform(X_train, y_train)  # Note: y_train is required\n",
    "X_test  = lda.transform(X_test)                # No labels needed for transform\n",
    "```\n",
    "\n",
    "**The critical difference from PCA:** `fit_transform` takes both `X_train` **and** `y_train`. LDA needs the class labels to compute its scatter matrices — without them it cannot identify which directions maximise class separation.\n",
    "\n",
    "The result is the same shape as PCA: X is reduced to 2 columns (LD1, LD2 — Linear Discriminants 1 and 2).\n",
    "\n",
    "**Why exactly 2 components?** With 3 wine classes, the maximum number of LDA components is 3 - 1 = 2. This is a hard mathematical limit — LDA cannot produce more components than classes minus one."
]

# Cell 11 — training classifier
cells[11]["source"] = [
    "## Step 6: Train Logistic Regression on LDA-Reduced Features\n",
    "\n",
    "The classifier receives 2 features (LD1, LD2) instead of 13.\n",
    "\n",
    "Because LDA specifically optimised the 2D projection for class separation, the logistic regression has an easier classification problem than it would with 2 random PCA components."
]

# Cell 13 — confusion matrix
cells[13]["source"] = [
    "## Step 7: Evaluate — Confusion Matrix\n",
    "\n",
    "**100% accuracy on the test set** with just 2 LDA features vs 13 original features.\n",
    "\n",
    "Compare this to the PCA result (~97% accuracy). LDA achieved perfect classification on the same dataset — because its 2 components were specifically chosen to maximally separate the 3 wine classes.\n",
    "\n",
    "**Why does LDA outperform PCA here?**\n",
    "\n",
    "The 2 principal components PCA selected capture the most variance — but maximum variance is not the same as maximum class discriminability. The direction of most variance might contain a mix of all three classes. LDA ignores overall variance and directly optimises for the projection that pushes the class clouds apart.\n",
    "\n",
    "**A word of caution on 100% accuracy:**\n",
    "\n",
    "Perfect test accuracy on 36 samples (20% of 178) is not a guarantee of real-world performance. With a small test set, 100% can happen by chance. Cross-validation over all 178 samples would give a more reliable estimate."
]

# Cell 15 — training visualisation
cells[15]["source"] = [
    "## Step 8: Visualise the Training Set\n",
    "\n",
    "The axes are now **LD1 and LD2** — Linear Discriminants — not PCs.\n",
    "\n",
    "You should see extremely clean cluster separation compared to the PCA visualisation. The three wine classes should be almost completely non-overlapping in 2D LDA space, because that is exactly what LDA was optimised to produce.\n",
    "\n",
    "This is the power of using label information for dimensionality reduction: LDA finds a 2D view of the data that makes the classification problem as easy as possible."
]

# Cell 17 — test visualisation
cells[17]["source"] = [
    "## Step 9: Visualise the Test Set\n",
    "\n",
    "The near-perfect separation on the test set confirms that LDA found genuinely discriminative directions, not just ones that memorised the training data.\n",
    "\n",
    "**LDA vs PCA — Summary:**\n",
    "\n",
    "| | PCA | LDA |\n",
    "|-|-----|-----|\n",
    "| Accuracy (this dataset) | ~97% | ~100% |\n",
    "| Visual separation | Good | Excellent |\n",
    "| Requires class labels | No | Yes |\n",
    "| Risk of overfitting | Lower | Higher (with few samples per class) |\n",
    "\n",
    "Use LDA when you have labelled data and want to optimise for classification. Use PCA when labels are unavailable, you are exploring the data, or you have too few samples per class for LDA to be reliable."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("LDA notebook updated successfully.")
