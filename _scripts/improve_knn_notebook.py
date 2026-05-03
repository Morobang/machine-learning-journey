import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\02_classification\notebooks\02_k_nearest_neighbors.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Map headers to their new content
replacements = {
    "## Splitting the dataset into the Training set and Test set": [
        "## Step 3: Train/Test Split\n",
        "\n",
        "We split the dataset into 75% training (300 customers) and 25% test (100 customers).\n",
        "\n",
        "**Why split before scaling?** The StandardScaler will be fit on the training set only. If we scaled before splitting, the test set's statistics would leak into the scaler — giving us an artificially optimistic evaluation. The split must come first to ensure the test set truly simulates unseen data."
    ],
    "## Feature Scaling": [
        "## Step 4: Feature Scaling — Mandatory for KNN\n",
        "\n",
        "**This is the most critical preprocessing step for KNN.** Here is the exact problem without scaling:\n",
        "\n",
        "```\n",
        "Customer A: Age 25, Salary $50,000\n",
        "Customer B: Age 35, Salary $60,000\n",
        "Customer C: Age 30, Salary $80,000\n",
        "\n",
        "Unscaled distance A→B: sqrt[(35-25)² + (60000-50000)²] ≈ 10,000\n",
        "Unscaled distance A→C: sqrt[(30-25)² + (80000-50000)²] ≈ 30,000\n",
        "```\n",
        "\n",
        "A 10-year age difference contributes almost nothing (100) compared to a $10,000 salary difference (100,000,000). Age is effectively invisible in all distance calculations.\n",
        "\n",
        "After StandardScaler, both features are centred at 0 with standard deviation 1. Equal scales mean equal contribution to distances — the model uses both age and salary to find genuine neighbours.\n",
        "\n",
        "**Rule:** Fit the scaler on training data only (`fit_transform`), apply it to test data with `transform`."
    ],
    "## Training the K-NN model on the Training set": [
        "## Step 5: Train the KNN Model\n",
        "\n",
        "```python\n",
        "KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)\n",
        "```\n",
        "\n",
        "**`n_neighbors=5`** — when classifying a new point, find the 5 most similar training examples and let them vote. The majority class wins.\n",
        "\n",
        "**`metric='minkowski', p=2`** — this is **Euclidean distance** (the standard straight-line distance). Setting `p=1` gives Manhattan distance (sum of absolute differences). Minkowski with `p=2` is the default and works well for most cases.\n",
        "\n",
        "**Why K=5?** It is a common default that balances:\n",
        "- **Too small (K=1):** Model memorises training data. Every noisy sample gets its own region. High variance.\n",
        "- **Too large (K=N):** Model always predicts the majority class regardless of input. High bias.\n",
        "- **K=5:** Enough neighbours for a stable vote, small enough to capture local patterns.\n",
        "\n",
        "In practice, you would choose K using cross-validation: try K=1, 3, 5, 7, 11 and pick the value that minimises validation error.\n",
        "\n",
        "**Note:** KNN has no training phase — it simply stores all training data. Prediction is where the work happens."
    ],
    "## Predicting a new result": [
        "## Step 6: Predict a Single Customer\n",
        "\n",
        "We predict whether a 30-year-old earning $87,000 would purchase.\n",
        "\n",
        "**Two critical requirements:**\n",
        "\n",
        "1. **Apply the same scaler** — `sc.transform([[30, 87000]])` standardises the new point using the mean and std from the training set. Never refit the scaler on new data.\n",
        "2. **2D array format** — `[[30, 87000]]` (double brackets) creates shape `(1, 2)`. A 1D array `[30, 87000]` would raise an error.\n",
        "\n",
        "After scaling, KNN finds the 5 nearest training customers and returns their majority vote."
    ],
    "## Predicting the Test set results": [
        "## Step 7: Predict All Test Customers\n",
        "\n",
        "We apply the model to all 100 test customers. For each one, KNN:\n",
        "1. Scales the input using the training scaler\n",
        "2. Computes Euclidean distance to all 300 training points\n",
        "3. Finds the 5 nearest neighbours\n",
        "4. Returns the majority class among those 5\n",
        "\n",
        "The output shows `[predicted, actual]` pairs. Rows where they differ are misclassifications — which the confusion matrix will summarise."
    ],
    "## Making the Confusion Matrix": [
        "## Step 8: Evaluate — Confusion Matrix\n",
        "\n",
        "The confusion matrix gives a complete breakdown of correct and incorrect predictions:\n",
        "\n",
        "```\n",
        "              Predicted 0    Predicted 1\n",
        "Actual 0        TN               FP\n",
        "Actual 1        FN               TP\n",
        "```\n",
        "\n",
        "**93% accuracy** means KNN correctly classified 93 of 100 test customers using only age and salary as features.\n",
        "\n",
        "**Checking for class imbalance:** If the dataset has far more 0s than 1s, 93% accuracy is not necessarily impressive — a model that always predicts 0 might also get ~70% accuracy. Look at the confusion matrix breakdown:\n",
        "- High TN + high TP = genuinely useful model\n",
        "- High TN + low TP = model ignores the minority class\n",
        "\n",
        "**KNN vs Logistic Regression on this dataset:** Both achieve ~93% accuracy. KNN gets there through local similarity reasoning; Logistic Regression through a global linear decision boundary. Neither dominates — for a 2-feature problem with a moderately non-linear boundary, both work well."
    ],
    "## Visualising the Training set results": [
        "## Step 9: Visualise the Training Set Decision Boundary\n",
        "\n",
        "KNN creates a **non-linear, irregular decision boundary** — one of its most distinctive characteristics.\n",
        "\n",
        "Unlike Logistic Regression (straight line) or SVM (smooth curve), KNN's boundary follows the local density of training points. Regions where one class dominates get that colour; mixed regions get a noisy, jagged boundary.\n",
        "\n",
        "**What the visualisation shows:**\n",
        "- **Red region:** KNN predicts 0 (will not purchase) for customers in this area of the age-salary space\n",
        "- **Green region:** KNN predicts 1 (will purchase)\n",
        "- **Dots:** Actual training labels (red = not purchased, green = purchased)\n",
        "\n",
        "The training boundary looks very clean because the model was fit on this data. Test set is the honest check."
    ],
    "## Visualising the Test set results": [
        "## Step 10: Visualise the Test Set Decision Boundary\n",
        "\n",
        "Test customers are plotted against the boundary learned from training data.\n",
        "\n",
        "**Misclassified points** appear as dots of the wrong colour inside a region — e.g., a red dot (not purchased) inside a green region (KNN predicted purchased).\n",
        "\n",
        "**Interpreting the KNN boundary shape:** The jagged, locally-adaptive boundary is both KNN's strength and weakness:\n",
        "- **Strength:** Captures local patterns that global linear models miss\n",
        "- **Weakness:** Sensitive to noise — a single outlier in the training set creates a local pocket in the boundary\n",
        "\n",
        "Increasing K smooths the boundary; decreasing K makes it more jagged. The K=5 boundary shown here is a reasonable middle ground."
    ]
}

# Apply replacements by matching cell source
for cell in cells:
    if cell["cell_type"] == "markdown":
        src = "".join(cell["source"]).strip()
        for old_header, new_content in replacements.items():
            if src == old_header:
                cell["source"] = new_content
                break

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("KNN notebook updated successfully.")
