import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\09_model_selection_and_evaluation\notebooks\02_grid_search.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Cell 0 — title / intro
cells[0]["source"] = [
    "# Grid Search Hyperparameter Tuning\n",
    "\n",
    "## Hyperparameters vs Parameters — A Critical Distinction\n",
    "\n",
    "**Parameters** are what the model *learns* from data — the weights in a neural network, the split thresholds in a decision tree.\n",
    "\n",
    "**Hyperparameters** are configuration choices you make *before* training — the learning rate, the number of trees, the kernel type in SVM. The model cannot learn these from data; you have to set them.\n",
    "\n",
    "The choice of hyperparameters can change accuracy from 70% to 93%. Getting them wrong is one of the most common reasons a model underperforms.\n",
    "\n",
    "---\n",
    "\n",
    "## The Problem: Which Values to Use?\n",
    "\n",
    "For Kernel SVM there are at least two critical hyperparameters:\n",
    "\n",
    "| Hyperparameter | What it controls |\n",
    "|----------------|------------------|\n",
    "| `C` | Regularisation strength. High C = hard margin (fits training data closely, risk of overfit). Low C = soft margin (more tolerant of misclassifications, risk of underfit). |\n",
    "| `gamma` | RBF kernel bandwidth. High gamma = narrow kernel (complex, wiggly boundary). Low gamma = wide kernel (smooth boundary). |\n",
    "\n",
    "Trying all combinations manually is impractical. And evaluating each configuration on a single train/test split is unreliable (high variance).\n",
    "\n",
    "---\n",
    "\n",
    "## The Solution: Grid Search + Cross Validation\n",
    "\n",
    "Grid Search exhaustively tries **every combination** of hyperparameter values you specify. For each combination, it runs **k-fold cross-validation** to get a reliable performance estimate.\n",
    "\n",
    "```\n",
    "Grid of C x gamma:\n",
    "\n",
    "         gamma=0.1  gamma=0.2  gamma=0.3  ... gamma=0.9\n",
    "C=0.25    CV(10)     CV(10)     CV(10)        CV(10)\n",
    "C=0.5     CV(10)     CV(10)     CV(10)        CV(10)\n",
    "C=0.75    CV(10)     CV(10)     CV(10)        CV(10)\n",
    "C=1.0     CV(10)     CV(10)     CV(10)        CV(10)\n",
    "```\n",
    "\n",
    "Each cell runs 10 training cycles. With 4 C values x 9 gamma values = 36 combinations x 10 folds = **360 model trainings** for just the RBF grid. This is why `n_jobs=-1` (use all CPU cores) is important.\n",
    "\n",
    "---\n",
    "\n",
    "## Grid Search vs Random Search\n",
    "\n",
    "| Method | How it works | When to use |\n",
    "|--------|-------------|-------------|\n",
    "| **Grid Search** | Exhaustive — tries every combination in the grid | Small grids (2-3 hyperparameters, few values) |\n",
    "| **Random Search** | Samples random combinations from distributions | Large grids (4+ hyperparameters) — finds good regions 10x faster |\n",
    "| **Bayesian Search** | Builds a model of the search space, targets promising areas | Production tuning — most efficient |\n",
    "\n",
    "Grid search is the right starting point for learning because its results are fully interpretable.\n",
    "\n",
    "---\n",
    "\n",
    "## What We Will Build\n",
    "\n",
    "1. Train a Kernel SVM with default hyperparameters\n",
    "2. Evaluate it with cross-validation (baseline)\n",
    "3. Apply Grid Search to find the best `C` and `gamma`\n",
    "4. Compare cross-validation accuracy before and after tuning"
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
    "Same Social Network Ads dataset used in the k-Fold notebook:\n",
    "\n",
    "- **400 samples**, 2 features (Age, EstimatedSalary)\n",
    "- Binary target: purchased the product (1) or not (0)\n",
    "\n",
    "A small dataset like this is ideal for demonstrating grid search because the training cycles complete quickly. On larger datasets, the computational cost of exhaustive grid search becomes prohibitive."
]

# Cell 5 — train/test split
cells[5]["source"] = [
    "## Step 3: Train/Test Split\n",
    "\n",
    "We hold out 25% as a final test set that will not be used during hyperparameter tuning.\n",
    "\n",
    "**This is critical:** Grid search uses cross-validation on the training set to select hyperparameters. The test set is only used for the final evaluation. If you used the test set to choose hyperparameters, you would be effectively training on it — your final accuracy estimate would be optimistic.\n",
    "\n",
    "The principle: **hyperparameter selection uses training data only.**"
]

# Cell 7 — feature scaling
cells[7]["source"] = [
    "## Step 4: Feature Scaling\n",
    "\n",
    "Required for Kernel SVM — distance-based model sensitive to feature scales.\n",
    "\n",
    "Fit on training data only, transform both sets with the same scaler."
]

# Cell 9 — training model
cells[9]["source"] = [
    "## Step 5: Train With Default Hyperparameters (Baseline)\n",
    "\n",
    "We train first with sklearn defaults: `C=1.0`, `gamma='scale'`. This gives us a baseline to compare against after grid search.\n",
    "\n",
    "Default parameters are rarely optimal. They are designed to work reasonably well across many datasets, not to maximise performance on any specific one."
]

# Cell 11 — confusion matrix
cells[11]["source"] = [
    "## Step 6: Baseline Evaluation\n",
    "\n",
    "Single-split accuracy with default hyperparameters. Note this number carefully — we will see whether grid search improves on it.\n",
    "\n",
    "Also run 10-fold cross-validation in the next step to get a more reliable baseline before tuning."
]

# Cell 13 — k-fold
cells[13]["source"] = [
    "## Step 7: Cross-Validated Baseline\n",
    "\n",
    "10-fold CV on the default model gives us a reliable accuracy baseline with a standard deviation.\n",
    "\n",
    "This is the number to beat with grid search. If grid search does not improve on it, the default hyperparameters were already near-optimal for this dataset."
]

# Cell 15 — grid search
cells[15]["source"] = [
    "## Step 8: Apply Grid Search\n",
    "\n",
    "We define a parameter grid with two sub-grids:\n",
    "\n",
    "1. **Linear kernel:** Only `C` matters (no `gamma` for linear)\n",
    "2. **RBF kernel:** Both `C` and `gamma`\n",
    "\n",
    "Total combinations:\n",
    "- Linear: 4 values of C\n",
    "- RBF: 4 values of C x 9 values of gamma = 36 combinations\n",
    "- **Total: 40 combinations x 10 folds = 400 model trainings**\n",
    "\n",
    "`GridSearchCV` handles this automatically and returns:\n",
    "- `best_score_`: the best cross-validated accuracy found\n",
    "- `best_params_`: the exact hyperparameter values that produced it\n",
    "\n",
    "**Why `n_jobs=-1`?** This tells sklearn to use all available CPU cores in parallel. Grid search is embarrassingly parallel — each combination is independent of the others. Without this, the 400 trainings run sequentially on one core."
]

# Cell 17 — training visualisation
cells[17]["source"] = [
    "## Step 9: Visualise the Training Set Decision Boundary\n",
    "\n",
    "The boundary plotted here uses the **original default model** (before grid search). If you want to visualise the optimised model, you would retrain with the best parameters from `grid_search.best_params_`.\n",
    "\n",
    "The decision boundary shape is determined by both `C` and `gamma`: higher `gamma` creates a more curved, locally-fitted boundary; lower `gamma` creates a smoother, wider-margin boundary."
]

# Cell 19 — test visualisation
cells[19]["source"] = [
    "## Step 10: Visualise the Test Set Decision Boundary\n",
    "\n",
    "Compare the single-split test accuracy here against the grid search cross-validation score.\n",
    "\n",
    "**The takeaway from this notebook:**\n",
    "\n",
    "1. Default hyperparameters are a starting point, not an endpoint\n",
    "2. Grid search + cross-validation is the principled way to tune — exhaustive, evaluated reliably, not biased by a single split\n",
    "3. The improvement from tuning can be modest (1-2%) or large (10%+) depending on how sensitive the model is to its hyperparameters\n",
    "4. Always report the cross-validated score from grid search as your model performance — not the single test split, which is just one data point"
]

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Grid Search notebook updated successfully.")
