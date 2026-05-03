import json, sys

# ── Logistic Regression ──────────────────────────────────────────────────────

lr_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\01_logistic_regression.ipynb"

lr_replacements = {
    1: [
        "## Step 1: Import Libraries\n",
        "\n",
        "| Library | Why we need it |\n",
        "|---------|---------------|\n",
        "| `numpy` | Array operations |\n",
        "| `pandas` | Loading and analysing the dataset |\n",
        "| `matplotlib` | Plotting the decision boundary |\n",
        "| `seaborn` | Statistical visualisations |"
    ],
    3: [
        "## Step 2: Load the Dataset\n",
        "\n",
        "We load the Social Network Ads dataset: 400 users with `Age`, `EstimatedSalary`, and `Purchased` (0 or 1).\n",
        "\n",
        "We select only columns 2 and 3 (Age and EstimatedSalary) as features. This 2-feature setup makes it possible to visualise the decision boundary in 2D — an important pedagogical choice."
    ],
    5: [
        "## Step 3: Exploratory Data Analysis (EDA)\n",
        "\n",
        "Before fitting any model, we need to understand the data:\n",
        "\n",
        "- **Class balance:** Is the dataset roughly 50/50 between purchased and not purchased? Severe imbalance would make accuracy misleading and require resampling or class weights.\n",
        "- **Feature ranges:** Age and salary are on different scales — a sign that feature scaling will be needed.\n",
        "- **Visual separation:** A scatter plot of age vs salary, coloured by purchase decision, will show whether the classes are linearly separable — i.e., whether a straight line can divide buyers from non-buyers."
    ],
    7: [
        "## Step 4: Data Cleaning\n",
        "\n",
        "The Social Network Ads dataset is pre-cleaned — no missing values or categorical variables.\n",
        "\n",
        "In a real project, this step would handle: missing value imputation, outlier treatment, and removing irrelevant columns. Even clean datasets benefit from a quick `dataset.isnull().sum()` and `dataset.describe()` check."
    ],
    9: [
        "## Step 5: Train/Test Split and Feature Scaling\n",
        "\n",
        "**Why split before scaling?**\n",
        "The scaler learns the mean and standard deviation from the data it is fit on. If we scaled the full dataset before splitting, the test set's statistics would influence the scaler — this is data leakage. The scaler must see only training data.\n",
        "\n",
        "**Why scaling for Logistic Regression?**\n",
        "Logistic Regression uses gradient descent to optimise the log-likelihood. If Age ranges from 18-60 and EstimatedSalary from 15K-150K, the gradients for the salary coefficient will be tiny compared to age. Training becomes slow and convergence is sensitive to the learning rate.\n",
        "\n",
        "After StandardScaler, both features have mean 0 and std 1 — equal gradient magnitudes, faster convergence."
    ]
}

with open(lr_path, encoding="utf-8") as f:
    lr_nb = json.load(f)
for idx, content in lr_replacements.items():
    lr_nb["cells"][idx]["source"] = content
with open(lr_path, "w", encoding="utf-8") as f:
    json.dump(lr_nb, f, indent=1, ensure_ascii=False)
print("Updated: 01_logistic_regression.ipynb")


# ── Shared replacements for bare standard sections ───────────────────────────

def make_train_test_split(note=""):
    return [
        "## Step 3: Train/Test Split\n",
        "\n",
        "We split 75% for training and 25% for testing.\n",
        "\n",
        "The split comes before scaling — fitting the scaler on the full dataset would leak test set statistics into the preprocessing pipeline." + ("\n\n" + note if note else "")
    ]

def make_feature_scaling(model_note):
    return [
        "## Step 4: Feature Scaling\n",
        "\n",
        model_note,
        "\n",
        "StandardScaler centres each feature to mean 0, standard deviation 1.\n",
        "Fit on training data only (`fit_transform`), then apply to test data with `transform`."
    ]

def make_predict_single(model_name):
    return [
        "## Step 7: Predict a Single Observation\n",
        "\n",
        f"This demonstrates the inference pipeline for a new data point using the trained {model_name}.\n",
        "\n",
        "**Important:** The input must be scaled with the same scaler used during training. "
        "The double brackets `[[...]]` create a 2D array — sklearn's `predict()` always expects shape `(n_samples, n_features)`."
    ]

def make_predict_test():
    return [
        "## Step 8: Predict All Test Set Results\n",
        "\n",
        "We apply the trained model to all 100 test customers. The output table shows `[predicted, actual]` pairs — "
        "rows where they differ are misclassifications that the confusion matrix will summarise."
    ]

def make_confusion_matrix(model_name, accuracy_note=""):
    return [
        "## Step 9: Confusion Matrix and Accuracy\n",
        "\n",
        "```\n",
        "              Predicted 0    Predicted 1\n",
        "Actual 0          TN              FP\n",
        "Actual 1          FN              TP\n",
        "```\n",
        "\n",
        f"The diagonal (TN + TP) contains correct predictions. Off-diagonal entries are errors.\n",
        "\n" + (accuracy_note + "\n" if accuracy_note else ""),
        "**Beyond accuracy:** For the Social Network Ads dataset with moderate class imbalance, "
        "also check recall (of all actual buyers, how many did we correctly identify?) and precision (of predicted buyers, how many actually bought?)."
    ]

def make_visualise_training(model_name):
    return [
        "## Step 10: Visualise the Training Set Decision Boundary\n",
        "\n",
        f"The coloured regions show where the {model_name} predicts each class. "
        "Dots are training samples coloured by their true label.\n",
        "\n",
        "A training set visualisation naturally looks good — the model was fit on this data. "
        "Any misclassified training points (dots in the wrong coloured region) indicate irreducible error or overlapping classes."
    ]

def make_visualise_test(model_name):
    return [
        "## Step 11: Visualise the Test Set Decision Boundary\n",
        "\n",
        f"The same decision boundary from training, now evaluated against the 100 unseen test customers.\n",
        "\n",
        "Misclassified test points reveal where the model generalises poorly. "
        "If the test boundary plot looks dramatically messier than the training plot, the model is overfitting."
    ]


# ── Naive Bayes ──────────────────────────────────────────────────────────────

nb_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\05_naive_bayes.ipynb"

nb_replacements = {
    5: make_train_test_split(),
    15: make_feature_scaling(
        "Gaussian Naive Bayes models each feature as a Gaussian distribution. "
        "It computes mean and variance per feature per class during training. "
        "Feature scaling does not affect the mathematical outcome of Naive Bayes — "
        "it learns different means/variances with or without scaling.\n\n"
        "However, scaling is included here for consistency with the pipeline pattern used across all classifiers in this section."
    ),
    19: [
        "## Step 6: Train Gaussian Naive Bayes\n",
        "\n",
        "Naive Bayes computes two things during training:\n",
        "1. **P(class)** — the prior probability of each class (fraction of training samples per class)\n",
        "2. **P(feature | class)** — for Gaussian NB, the mean and variance of each feature for each class\n",
        "\n",
        "There are no hyperparameters to tune and no gradient descent — training is a single pass through the data to compute these statistics.\n",
        "\n",
        "During prediction, Bayes' theorem combines these:\n",
        "\n",
        "$$P(\\text{class} | \\text{features}) \\propto P(\\text{class}) \\times \\prod_i P(x_i | \\text{class})$$\n",
        "\n",
        "The class with the highest posterior probability wins."
    ],
    21: make_predict_single("Naive Bayes classifier"),
    23: make_predict_test(),
    25: make_confusion_matrix("Naive Bayes"),
    27: make_visualise_training("Naive Bayes"),
    29: make_visualise_test("Naive Bayes")
}

with open(nb_path, encoding="utf-8") as f:
    nb_nb = json.load(f)
for idx, content in nb_replacements.items():
    nb_nb["cells"][idx]["source"] = content
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_nb, f, indent=1, ensure_ascii=False)
print("Updated: 05_naive_bayes.ipynb")


# ── Decision Tree Classification ─────────────────────────────────────────────

dt_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\06_decision_tree_classification.ipynb"

with open(dt_path, encoding="utf-8") as f:
    dt_nb = json.load(f)

dt_replacements = {
    9: [
        "## Step 3: Encode Categorical Variables\n",
        "\n",
        "If the dataset has categorical features, they must be converted to numbers before the model can process them. "
        "Decision Trees can work with any numerical encoding since they use threshold comparisons, not arithmetic. "
        "One-hot encoding is still preferred to avoid introducing false ordinal relationships."
    ],
    10: [
        "## Step 4: Train/Test Split\n",
        "\n",
        "75/25 split — 300 training, 100 test customers.\n",
        "\n",
        "Decision Trees do not require feature scaling, but the split must still come before any data-dependent preprocessing."
    ],
    16: [
        "## Step 5: Feature Scaling\n",
        "\n",
        "**Decision Trees do not require feature scaling.** Trees make split decisions using threshold comparisons "
        "(`Age <= 40`), which are scale-invariant — rescaling features does not change which threshold minimises impurity.\n",
        "\n",
        "Feature scaling is included here for completeness and to maintain a consistent pipeline across all classifiers. "
        "If you removed this step for Decision Trees, you would get identical results."
    ],
    20: [
        "## Step 6: Train the Decision Tree Classifier\n",
        "\n",
        "The algorithm builds the tree by recursively finding the best split at each node:\n",
        "\n",
        "1. Try every possible threshold on every feature\n",
        "2. Compute the impurity reduction for each split (Gini or entropy criterion)\n",
        "3. Choose the split with the greatest impurity reduction\n",
        "4. Recurse on each child node\n",
        "5. Stop when a stopping criterion is met (`max_depth`, `min_samples_split`, or pure leaf)\n",
        "\n",
        "**`criterion='gini'`** — Gini impurity measures the probability of misclassifying a randomly chosen sample if we labelled it according to the class distribution in the leaf:\n",
        "\n",
        "$$\\text{Gini} = 1 - \\sum_{k} p_k^2$$\n",
        "\n",
        "A perfectly pure node (all one class) has Gini=0. Equal class distribution has Gini=0.5.\n",
        "\n",
        "**Overfitting warning:** With no depth limit, the tree grows until every leaf is pure — R²=1.0 on training, but poor generalisation. "
        "Always set `max_depth` or `min_samples_leaf` in production."
    ],
    22: [
        "## Step 7: Make Predictions\n",
        "\n",
        "For each test customer, the tree traverses from root to leaf by evaluating split conditions. "
        "The predicted class is the majority class of all training samples that reached that leaf."
    ],
    24: make_predict_test(),
    26: [
        "## Step 8: Evaluate the Model\n",
        "\n",
        "Accuracy gives the top-line number, but the confusion matrix shows the breakdown of errors:\n",
        "\n",
        "- **False Positives (FP):** Predicted purchase, actually no purchase — wasted marketing effort\n",
        "- **False Negatives (FN):** Predicted no purchase, actually purchased — missed revenue opportunity\n",
        "\n",
        "The relative cost of FP vs FN is a business decision that determines which metric (precision vs recall) to prioritise."
    ],
    27: make_confusion_matrix("Decision Tree"),
    30: make_visualise_training("Decision Tree"),
    32: make_visualise_test("Decision Tree")
}

for idx, content in dt_replacements.items():
    dt_nb["cells"][idx]["source"] = content
with open(dt_path, "w", encoding="utf-8") as f:
    json.dump(dt_nb, f, indent=1, ensure_ascii=False)
print("Updated: 06_decision_tree_classification.ipynb")


# ── Random Forest Classification ─────────────────────────────────────────────

rf_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\07_random_forest_classification.ipynb"

with open(rf_path, encoding="utf-8") as f:
    rf_nb = json.load(f)

rf_replacements = {
    21: make_predict_single("Random Forest classifier"),
    23: make_predict_test(),
    25: make_confusion_matrix(
        "Random Forest",
        "Random Forest typically achieves higher accuracy than a single Decision Tree on the same data "
        "because the ensemble averaging reduces the variance that makes individual trees overfit."
    ),
    27: make_visualise_training("Random Forest"),
    29: make_visualise_test("Random Forest")
}

for idx, content in rf_replacements.items():
    rf_nb["cells"][idx]["source"] = content
with open(rf_path, "w", encoding="utf-8") as f:
    json.dump(rf_nb, f, indent=1, ensure_ascii=False)
print("Updated: 07_random_forest_classification.ipynb")
