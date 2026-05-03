import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\09_model_selection_and_evaluation\notebooks\01_k_fold_cross_validation.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title / intro
cells[0]["source"] = [
    "# k-Fold Cross Validation\n",
    "\n",
    "## The Problem With a Single Train/Test Split\n",
    "\n",
    "When you split your dataset once — say 75% training, 25% test — your performance estimate depends heavily on **which samples happened to land in the test set**. If the test set accidentally contains mostly easy examples, you will over-estimate your model's accuracy. If it contains hard outliers, you will under-estimate it.\n",
    "\n",
    "This is not hypothetical. On a dataset of 400 samples with a 25% split, you have 100 test samples. The difference between 90% and 93% accuracy is just 3 data points. Your estimate is **high variance**.\n",
    "\n",
    "---\n",
    "\n",
    "## The Solution: k-Fold Cross Validation\n",
    "\n",
    "k-Fold CV uses every sample for both training and testing, cycling through $k$ different splits:\n",
    "\n",
    "```\n",
    "k=5 example with 400 samples:\n",
    "\n",
    "Fold 1:  [TEST  |  train | train | train | train]  → Accuracy: 91%\n",
    "Fold 2:  [train | TEST   | train | train | train]  → Accuracy: 89%\n",
    "Fold 3:  [train | train  | TEST  | train | train]  → Accuracy: 94%\n",
    "Fold 4:  [train | train  | train | TEST  | train]  → Accuracy: 88%\n",
    "Fold 5:  [train | train  | train | train | TEST ]  → Accuracy: 92%\n",
    "\n",
    "Final estimate: mean = 90.8%, std = 2.2%\n",
    "```\n",
    "\n",
    "The model is trained $k$ times, each time on a different 80% of the data, and tested on the remaining 20%. The final performance estimate is the **average across all k folds**, with the **standard deviation** telling you how stable the model is.\n",
    "\n",
    "---\n",
    "\n",
    "## Why Both Mean and Standard Deviation Matter\n",
    "\n",
    "| Result | What it tells you |\n",
    "|--------|------------------|\n",
    "| High mean, low std | Model generalises well and consistently |\n",
    "| High mean, high std | Model is sensitive to which data it sees — might overfit on some folds |\n",
    "| Low mean, low std | Underfitting consistently |\n",
    "| Low mean, high std | Unstable and inaccurate — serious problem |\n",
    "\n",
    "A model with 90% accuracy and 1% standard deviation is far more trustworthy than one with 92% accuracy and 8% standard deviation.\n",
    "\n",
    "---\n",
    "\n",
    "## Choosing k\n",
    "\n",
    "| k value | Trade-off |\n",
    "|---------|----------|\n",
    "| k = 5 | Each fold trains on 80% of data. Fast. Slightly higher bias. |\n",
    "| k = 10 | Standard choice. Good balance of bias and variance. |\n",
    "| k = n (Leave-One-Out) | Lowest bias, very high compute cost, high variance on small datasets |\n",
    "\n",
    "**Default: k = 10.** It is the most commonly used value in academic papers and competitions for small-to-medium datasets.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "1. Train a Kernel SVM classifier on Social Network Ads data\n",
    "2. Evaluate it with a single test split (the naive approach)\n",
    "3. Apply 10-fold cross validation to get a more reliable estimate\n",
    "4. Compare and see how the standard deviation reveals model stability"
]

# Cell 1 — libraries
cells[1]["source"] = [
    "## Step 1: Import Libraries\n",
    "\n",
    "| Library | Why we need it |\n",
    "|---------|---------------|\n",
    "| `numpy` | Array operations |\n",
    "| `matplotlib` | Decision boundary visualisation |\n",
    "| `pandas` | Loading the dataset |"
]

# Cell 3 — dataset
cells[3]["source"] = [
    "## Step 2: Load the Dataset\n",
    "\n",
    "The Social Network Ads dataset has **400 users** with two features:\n",
    "\n",
    "| Feature | Description |\n",
    "|---------|-------------|\n",
    "| `Age` | User age |\n",
    "| `EstimatedSalary` | Estimated annual salary |\n",
    "\n",
    "Target: `Purchased` (1 = bought the product, 0 = did not).\n",
    "\n",
    "This is a small dataset by production standards — exactly where cross-validation matters most. With only 400 samples, a single random split produces a noisy performance estimate."
]

# Cell 5 — train/test split
cells[5]["source"] = [
    "## Step 3: Initial Train/Test Split\n",
    "\n",
    "We do an initial 75/25 split and train the model once. This gives us a baseline accuracy that we will later compare against the cross-validation result.\n",
    "\n",
    "With 400 samples and a 25% test set, we have **100 test samples**. This means our accuracy estimate can swing by 3-4% depending purely on which 100 samples were randomly assigned to the test set — not because the model changed."
]

# Cell 7 — feature scaling
cells[7]["source"] = [
    "## Step 4: Feature Scaling\n",
    "\n",
    "Kernel SVM computes distances between data points using the RBF (Radial Basis Function) kernel. If `EstimatedSalary` ranges from 15,000 to 150,000 and `Age` ranges from 18 to 60, the salary feature will dominate all distance calculations.\n",
    "\n",
    "StandardScaler centres both features to mean 0, standard deviation 1 — so the model evaluates age and salary on equal footing.\n",
    "\n",
    "**Always fit the scaler on training data only** — if you fit on the full dataset, test data statistics leak into the scaler, making your test evaluation optimistic."
]

# Cell 9 — training model
cells[9]["source"] = [
    "## Step 5: Train the Kernel SVM\n",
    "\n",
    "We use an RBF (Radial Basis Function) kernel — this maps the 2D data into a higher-dimensional space where a non-linear decision boundary becomes linear.\n",
    "\n",
    "The trained model gives us a single-split accuracy. But is this estimate reliable? That is what cross-validation will tell us."
]

# Cell 11 — confusion matrix
cells[11]["source"] = [
    "## Step 6: Baseline Evaluation (Single Split)\n",
    "\n",
    "The confusion matrix shows the breakdown of correct and incorrect predictions on our single test split:\n",
    "\n",
    "```\n",
    "[[TN  FP]     <- Predicted Not Purchased\n",
    " [FN  TP]]    <- Predicted Purchased\n",
    "```\n",
    "\n",
    "The accuracy here (~93%) is our **single-split estimate**. It looks good, but remember: it is based on just one random partitioning of 400 samples. In the next step, we will test whether this number holds up across multiple different splits."
]

# Cell 13 — applying k-fold
cells[13]["source"] = [
    "## Step 7: Apply 10-Fold Cross Validation\n",
    "\n",
    "This is the key step. `cross_val_score` with `cv=10` does the following automatically:\n",
    "\n",
    "1. Splits the **training set** into 10 equal folds\n",
    "2. Trains the model 10 times, each time holding out one fold as a mini test set\n",
    "3. Returns the accuracy from each fold as an array\n",
    "\n",
    "We then report the **mean** (our best estimate of true accuracy) and **standard deviation** (how consistent the model is).\n",
    "\n",
    "**Important:** Cross-validation is applied to the training set only (`X_train`, `y_train`). The test set we split off earlier is kept completely separate — it is our final held-out evaluation that only gets used once, at the very end.\n",
    "\n",
    "**Interpreting the output:**\n",
    "\n",
    "- If the CV mean (~90%) is close to the single-split accuracy (~93%), both estimates are consistent\n",
    "- If they differ substantially, the single-split was lucky (or unlucky)\n",
    "- A standard deviation below 5% is generally considered stable"
]

# Cell 15 — visualising training set
cells[15]["source"] = [
    "## Step 8: Visualise the Training Set Decision Boundary\n",
    "\n",
    "This plot shows the RBF kernel SVM decision boundary on the training set. The non-linear curved boundary is only possible because of the kernel trick — mapping the original 2D features into a higher-dimensional space.\n",
    "\n",
    "Training set visualisations look optimistic because the model has seen this data. The real test is the next plot."
]

# Cell 17 — visualising test set
cells[17]["source"] = [
    "## Step 9: Visualise the Test Set Decision Boundary\n",
    "\n",
    "The test set points are data the model has never seen. Points landing in the wrong coloured region are misclassifications.\n",
    "\n",
    "**Cross-validation vs this final test:**\n",
    "\n",
    "- Cross-validation gave us a **reliable average estimate** of generalisation performance\n",
    "- This plot shows performance on one specific held-out set\n",
    "\n",
    "In production, you would report the cross-validation score as your model performance metric — not just the single test split — because it is based on more data and more independent evaluations."
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("k-Fold notebook updated successfully.")
