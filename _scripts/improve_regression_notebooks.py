import json

# ── Shared step content used across notebooks ──────────────────────────────

def step1_libraries():
    return [
        "## Step 1: Import Libraries\n",
        "\n",
        "| Library | Why we need it |\n",
        "|---------|---------------|\n",
        "| `numpy` | Array operations |\n",
        "| `pandas` | Loading and analysing the dataset |\n",
        "| `matplotlib` | Plotting predictions and residuals |\n",
        "| `seaborn` | Statistical visualisations |\n",
        "| `warnings` | Suppress sklearn deprecation warnings |"
    ]

def step2_dataset(desc):
    return [
        "## Step 2: Load the Dataset\n",
        "\n",
        desc
    ]

def step3_eda():
    return [
        "## Step 3: Exploratory Data Analysis (EDA)\n",
        "\n",
        "EDA informs every preprocessing decision that follows. Here we check:\n",
        "\n",
        "- **Missing values** — do any columns need imputation?\n",
        "- **Data types** — are there categorical columns needing encoding?\n",
        "- **Distributions** — is the target heavily skewed? Are features on wildly different scales?\n",
        "- **Correlations** — which features have the strongest linear relationship with the target?\n",
        "\n",
        "Reading the output carefully before writing any model code is a professional habit that prevents wasted effort."
    ]

def step4_cleaning(note=""):
    return [
        "## Step 4: Data Cleaning\n",
        "\n",
        "This step converts EDA observations into actions: handle missing values, correct types, and prepare features for the model." + ("\n\n" + note if note else "")
    ]

def step9_evaluate(model_name):
    return [
        f"## Step 9: Evaluate {model_name} Performance\n",
        "\n",
        "**R² (Coefficient of Determination):** Fraction of variance in the target explained by the model. R²=1.0 is perfect; R²=0 means the model does no better than predicting the mean.\n",
        "\n",
        "**MAE (Mean Absolute Error):** Average absolute prediction error, in the same units as the target. Easy to interpret: *the model is off by this much on average.*\n",
        "\n",
        "**RMSE (Root Mean Squared Error):** Similar to MAE but penalises large errors more heavily. If RMSE >> MAE, a few predictions have very large errors.\n",
        "\n",
        "**Compare training vs test R²:** A large gap (e.g., 0.99 train vs 0.60 test) signals overfitting."
    ]


# ── Polynomial Regression ───────────────────────────────────────────────────

poly_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\03_polynomial_regression.ipynb"

poly_replacements = {
    1: step1_libraries(),
    3: step2_dataset(
        "The Position Salaries dataset has **10 rows** and 3 columns:\n\n"
        "| Column | Description |\n"
        "|--------|-------------|\n"
        "| Position | Job title (CEO, Manager, etc.) |\n"
        "| Level | Numeric level 1-10 corresponding to the position |\n"
        "| Salary | Target — annual salary |\n\n"
        "We use only `Level` as the feature (X) and `Salary` as the target (y). "
        "With only 10 data points, this is an extreme case — real-world regression problems have far more data. "
        "The small size makes it easy to visualise the polynomial fit but means the model is highly sensitive to each individual point."
    ),
    5: step3_eda(),
    7: [
        "## Step 4: Data Cleaning and Preparation\n",
        "\n",
        "The dataset is clean — no missing values, no categorical variables to encode. All columns are already numerical.\n",
        "\n",
        "**Why no feature scaling for Polynomial Regression?**\n",
        "We are using `LinearRegression` from sklearn on polynomial features. sklearn's implementation uses the normal equations (closed-form OLS solution) which is scale-invariant — it finds the same optimal coefficients regardless of feature scale. "
        "Feature scaling is critical for gradient-based methods (neural networks, SVR) but not for OLS regression."
    ],
    9: [
        "## Step 5: Train/Test Split\n",
        "\n",
        "With only 10 data points, the train/test split is illustrative — any test accuracy estimate based on 2 samples has massive variance. "
        "In a real project with this few samples, you would use leave-one-out cross-validation (LOOCV) instead.\n",
        "\n",
        "The split still demonstrates the correct workflow: always evaluate on data the model has not seen."
    ],
    11: [
        "## Step 6: Train Linear Regression (Baseline)\n",
        "\n",
        "We train a simple linear model first as a **baseline for comparison**.\n",
        "\n",
        "The linear model assumes a straight-line relationship between Level and Salary: "
        "`Salary = b0 + b1 * Level`. "
        "If the true relationship is curved (which it almost certainly is — salaries grow non-linearly with seniority), "
        "the linear model will systematically underpredict at low and high levels and overpredict in the middle.\n",
        "\n",
        "This baseline makes the polynomial model's improvement concrete and measurable."
    ],
    13: [
        "## Step 7: Train Polynomial Regression\n",
        "\n",
        "**Polynomial Regression is not a new algorithm** — it is Linear Regression applied to polynomial features.\n",
        "\n",
        "The trick: create new features by raising the original feature to powers 2, 3, 4...:\n",
        "\n",
        "```python\n",
        "PolynomialFeatures(degree=4)\n",
        "# Transforms [Level] into [1, Level, Level², Level³, Level⁴]\n",
        "```\n",
        "\n",
        "Then fit Linear Regression on these 5 features. The result is a polynomial curve:\n",
        "\n",
        "$$\\hat{y} = b_0 + b_1 x + b_2 x^2 + b_3 x^3 + b_4 x^4$$\n",
        "\n",
        "**Why not just use degree=100?** Higher degree = more flexible = more overfit. A degree-10 polynomial through 10 points would perfectly interpolate every training point (R²=1.0) but produce wildly unrealistic predictions between them and outside the training range. Start with degree 2 or 4 and increase only if the fit is clearly poor."
    ],
    15: [
        "## Step 8: Compare Linear vs Polynomial Predictions\n",
        "\n",
        "Side-by-side comparison reveals how much the polynomial model improves on the linear baseline.\n",
        "\n",
        "**What to look for:**\n",
        "- Does the polynomial model predict the high-salary senior positions more accurately?\n",
        "- Are the errors systematic (always too high or too low) or roughly random?\n",
        "- Does the polynomial model capture the acceleration in salary growth at higher levels?"
    ],
    17: step9_evaluate("Polynomial Regression"),
    19: [
        "## Step 10: Visualise Results and Model Comparison\n",
        "\n",
        "The visualisation plots both the linear and polynomial predictions against the actual data points.\n",
        "\n",
        "**What the curves show:**\n",
        "- **Linear regression:** A straight line forced through the data — systematically wrong at the extremes\n",
        "- **Polynomial regression:** A smooth curve that follows the accelerating salary growth at senior levels\n",
        "\n",
        "**The key lesson:** When the residuals from a linear model show a curved pattern (high at the ends, low in the middle or vice versa), that is the signal to try polynomial features. "
        "The curve shape tells you which degree might be appropriate.\n",
        "\n",
        "**Warning about extrapolation:** Polynomial models become unreliable outside the training range. "
        "The curve may dip or spike dramatically for inputs beyond the observed levels."
    ]
}

# ── SVR ─────────────────────────────────────────────────────────────────────

svr_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\04_support_vector_regression.ipynb"

svr_replacements = {
    1: step1_libraries(),
    3: step2_dataset(
        "Same Position Salaries dataset: 10 employees, Level (1-10) as feature, Salary as target.\n\n"
        "We use `X = dataset.iloc[:, 1:2]` (Level column as a matrix, not a vector) because "
        "SVR requires a 2D feature matrix even for a single feature. "
        "Using `iloc[:, 1]` would produce a 1D array that would need reshaping."
    ),
    5: step3_eda(),
    7: [
        "## Step 4: Feature Scaling — Required for SVR\n",
        "\n",
        "**SVR requires feature scaling.** This is the opposite of the Polynomial Regression notebook.\n",
        "\n",
        "SVR is an optimisation algorithm that finds a hyperplane in feature space by minimising a loss function with gradient-based methods. "
        "The RBF (Radial Basis Function) kernel also computes distances between data points.\n",
        "\n",
        "Both gradient descent and distance calculations are sensitive to feature scale differences. "
        "If salary ranges from 45,000 to 1,000,000 and level ranges from 1 to 10, the salary dominates completely.\n",
        "\n",
        "We scale **both X and y** here. Unlike classification (where y is a label), regression has a continuous target — "
        "if we scale X but not y, predictions will be on a different scale from the target.\n",
        "\n",
        "**After training, we must inverse-transform predictions** back to the original salary scale. "
        "Forgetting this step produces predictions in the standardised range (around -1 to 1) instead of dollars."
    ],
    9: [
        "## Step 5: Train/Test Split\n",
        "\n",
        "Same 80/20 split. Note: with 10 samples, this gives 8 training and 2 test points. "
        "Performance estimates based on 2 points are not reliable — use this notebook to understand the workflow, not to benchmark SVR vs polynomial."
    ],
    11: [
        "## Step 6: Train Support Vector Regression\n",
        "\n",
        "SVR extends Support Vector Machines to regression. The key ideas:\n",
        "\n",
        "**The epsilon-insensitive tube:** Instead of penalising all errors, SVR only penalises predictions that fall outside a tube of width $2\\epsilon$ around the true values. "
        "Predictions within the tube contribute zero to the loss.\n",
        "\n",
        "**Support vectors:** Only the data points that fall outside or on the boundary of the tube influence the model. "
        "Points inside the tube are ignored. This makes SVR robust to small noise and outlier-resistant.\n",
        "\n",
        "**RBF kernel:** Maps the data to a higher-dimensional space where a linear hyperplane becomes a non-linear curve in the original space. "
        "This allows SVR to fit complex, curved relationships.\n",
        "\n",
        "**Key hyperparameters:**\n",
        "\n",
        "| Parameter | Effect |\n",
        "|-----------|--------|\n",
        "| `C` | Regularisation. High C = fit training data tightly, risk overfit |\n",
        "| `epsilon` | Tube width. Larger = more tolerance for errors, simpler model |\n",
        "| `gamma` | RBF kernel width. High = narrow kernel, complex curve |"
    ],
    13: [
        "## Step 7: Predict and Inverse Transform\n",
        "\n",
        "**The inverse transform step is essential and often forgotten by beginners.**\n",
        "\n",
        "The entire training pipeline operated in standardised (scaled) space:\n",
        "- X was scaled to mean 0, std 1\n",
        "- y was scaled to mean 0, std 1\n",
        "- The model learned to predict scaled salary values\n",
        "\n",
        "Raw predictions from `svr.predict()` are in scaled units. "
        "`sc_y.inverse_transform(prediction)` converts back to the original dollar salary scale.\n",
        "\n",
        "**When predicting a new input:** You must also scale the input first with `sc_X.transform()`, "
        "then inverse-transform the output with `sc_y.inverse_transform()`. "
        "Both scalers must be the same ones fitted on the training data."
    ],
    15: [
        "## Step 8: Compare SVR Against Other Models\n",
        "\n",
        "With the Position Salaries dataset, multiple models can be compared on the same 10 points.\n",
        "\n",
        "**Expected pattern:**\n",
        "- Linear regression: worst (cannot capture non-linear salary growth)\n",
        "- Polynomial regression: good (explicitly models the curve)\n",
        "- SVR with RBF: often comparable to polynomial — adapts to the curve without specifying a degree\n",
        "\n",
        "SVR's advantage: you do not need to choose a polynomial degree. The RBF kernel adapts its complexity automatically through the `gamma` and `C` parameters."
    ],
    17: step9_evaluate("SVR"),
    19: [
        "## Step 10: Visualise SVR Predictions\n",
        "\n",
        "The prediction curve should follow the non-linear salary growth without overfitting to individual points.\n",
        "\n",
        "**Interpreting the SVR curve:**\n",
        "- The curve passes through or near most points but does not necessarily touch every one (unlike a high-degree polynomial interpolation)\n",
        "- Points that fall inside the epsilon tube had no influence on training\n",
        "- The support vectors (points outside the tube) are what shaped the curve\n",
        "\n",
        "**SVR vs Polynomial — when to use which:**\n",
        "- SVR: when you do not want to specify a polynomial degree; when the relationship is complex and you need regularisation\n",
        "- Polynomial: when the relationship has a known structure; when interpretability of coefficients matters"
    ]
}

# ── Decision Tree Regression ─────────────────────────────────────────────────

dt_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\05_decision_tree_regression.ipynb"

dt_replacements = {
    1: step1_libraries(),
    3: step2_dataset(
        "Same Position Salaries dataset: Level (1-10) → Salary.\n\n"
        "Decision trees are particularly interesting on this dataset because they will create "
        "a **step function** — flat horizontal segments at the average salary for each region of levels. "
        "This is the fundamental characteristic of tree-based predictions."
    ),
    5: step3_eda(),
    7: [
        "## Step 4: Data Cleaning\n",
        "\n",
        "No missing values or categorical variables to handle.\n",
        "\n",
        "**Feature scaling is NOT required for Decision Trees.**\n",
        "\n",
        "Decision trees make splits based on threshold comparisons: `Level <= 6.5`. "
        "Whether level is in the range 1-10 or 0-1 or 100-1000, the threshold adapts automatically. "
        "The model's output is identical regardless of scale.\n",
        "\n",
        "This is one advantage of tree-based models over distance-based (KNN, SVM) and gradient-based (neural networks) methods."
    ],
    9: [
        "## Step 5: Train/Test Split\n",
        "\n",
        "80/20 split — 8 training, 2 test points.\n",
        "\n",
        "With decision trees and very few data points, the tree structure can vary significantly depending on which 2 points land in the test set. "
        "This is another reason why cross-validation is more appropriate for small datasets."
    ],
    11: [
        "## Step 6: Train Decision Tree Regression\n",
        "\n",
        "A Decision Tree Regression works by **recursively splitting** the feature space into regions, "
        "then predicting the **average target value** in each region.\n",
        "\n",
        "**How splits are chosen:**\n",
        "The algorithm tries every possible threshold on every feature. "
        "It picks the split that minimises the weighted sum of variances in the resulting child nodes (MSE criterion).\n",
        "\n",
        "**The result is always a step function:**\n",
        "```\n",
        "Salary\n",
        " 1M  |                          ___\n",
        "500K |               ___________|\n",
        "300K |     __________|\n",
        "100K |____|_________________________\n",
        "      1   2   3   4   5   6   7   8   9   10\n",
        "                    Level\n",
        "```\n",
        "\n",
        "Each horizontal segment is the average salary of all training points in that level range.\n",
        "\n",
        "**`random_state=0`** ensures reproducible tree structure. Decision trees are deterministic given the same data and parameters — the randomness only comes into play with features like `max_features` that sub-sample features at each split."
    ],
    13: [
        "## Step 7: Make Predictions and Analyse the Tree\n",
        "\n",
        "Decision tree predictions are the mean target value of the training samples in the leaf node that the input falls into.\n",
        "\n",
        "**Key insight:** A decision tree cannot predict a value outside the range of training targets. "
        "If the maximum training salary is $1M, the tree can never predict $1.2M — it is bounded by observed values. "
        "Compare this to linear regression, which can extrapolate beyond the training range."
    ],
    15: [
        "## Step 8: Compare With Other Regression Models\n",
        "\n",
        "Decision Tree vs Linear, Polynomial, and SVR on the same dataset:\n",
        "\n",
        "| Model | Prediction style | Extrapolation | Interpretability |\n",
        "|-------|-----------------|---------------|------------------|\n",
        "| Linear | Straight line | Extrapolates | High (coefficients) |\n",
        "| Polynomial | Smooth curve | Unreliable outside range | Medium |\n",
        "| SVR | Smooth curve | Limited | Low |\n",
        "| Decision Tree | Step function | Cannot extrapolate | High (tree rules) |\n",
        "\n",
        "The decision tree model is highly interpretable — you can print the exact rules it uses. "
        "But the step function nature means it cannot smoothly interpolate between training points."
    ],
    17: step9_evaluate("Decision Tree"),
    19: [
        "## Step 10: Visualise the Step Function\n",
        "\n",
        "The decision tree prediction plot will show the characteristic **staircase shape** — each step is one leaf node of the tree.\n",
        "\n",
        "**What the visualisation reveals:**\n",
        "\n",
        "- How many distinct regions (steps) the tree created\n",
        "- Whether the tree overfit by creating a separate region for nearly every training point\n",
        "- How the step function approximates the true underlying salary curve\n",
        "\n",
        "**Overfitting in decision trees:** With no depth limit (`max_depth=None`), the tree can create one leaf per training sample — R²=1.0 on training data, but terrible on test data. "
        "Controlling depth (`max_depth=3`) or minimum samples per leaf is the primary regularisation technique for trees."
    ]
}

# ── Random Forest Regression ─────────────────────────────────────────────────

rf_path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\06_random_forest_regression.ipynb"

rf_replacements = {
    1: step1_libraries(),
    3: step2_dataset(
        "Same Position Salaries dataset: Level (1-10) → Salary.\n\n"
        "Random Forest on this small dataset is illustrative rather than practically necessary — "
        "ensemble methods show their real advantage on larger, noisier datasets. "
        "Here we focus on understanding the mechanism."
    ),
    5: step3_eda(),
    7: [
        "## Step 4: Data Cleaning\n",
        "\n",
        "No missing values or encoding needed. Like Decision Trees, Random Forest is scale-invariant — feature scaling is not required."
    ],
    9: [
        "## Step 5: Train/Test Split\n",
        "\n",
        "80/20 split. The split before training ensures our evaluation reflects performance on unseen data."
    ],
    11: [
        "## Step 5: Train the Random Forest Model\n",
        "\n",
        "A Random Forest is an **ensemble of decision trees**. Each tree is trained on a different bootstrap sample of the data and uses a random subset of features at each split.\n",
        "\n",
        "**Why does averaging many imperfect trees work so well?**\n",
        "\n",
        "A single decision tree is a **high-variance** model — small changes in training data produce very different trees. "
        "But when 100+ trees vote, their individual errors are largely independent and cancel out. "
        "The aggregate prediction is much more stable.\n",
        "\n",
        "This is the core idea of **bagging** (Bootstrap Aggregating):\n",
        "```\n",
        "Tree 1 (trained on sample A): predicts 450K\n",
        "Tree 2 (trained on sample B): predicts 480K\n",
        "Tree 3 (trained on sample C): predicts 430K\n",
        "...\n",
        "Tree 100:                       predicts 460K\n",
        "\n",
        "Random Forest final prediction: average = 455K\n",
        "```\n",
        "\n",
        "**`n_estimators=300`** — 300 trees. More trees = more stable predictions, diminishing returns above ~200. "
        "The main cost is training time, not overfitting — adding more trees never makes a random forest overfit.\n",
        "\n",
        "**`random_state=0`** — ensures reproducible results by fixing the random seeds for both sampling and feature selection."
    ],
    13: [
        "## Step 6: Make Predictions and Evaluate\n",
        "\n",
        "The Random Forest prediction is the **average of all 300 individual tree predictions**. "
        "Because the trees were trained on different bootstrap samples with random feature subsets, "
        "each has a slightly different perspective on the data — and averaging reduces the individual variance.\n",
        "\n",
        "**Comparing to single Decision Tree:**\n",
        "The single tree produces a coarse step function with few steps. "
        "The Random Forest produces a much smoother curve — many trees with different split points, "
        "averaged together, approximate a continuous function rather than a staircase."
    ],
    15: [
        "## Step 7: Predict a Specific Value\n",
        "\n",
        "This is the practical use case: given a new employee at level 6.5, what salary does the model predict?\n",
        "\n",
        "For a Random Forest, this means:\n",
        "1. Each of the 300 trees traverses its branches for input [6.5]\n",
        "2. Each tree returns the mean salary of its corresponding leaf node\n",
        "3. The 300 predictions are averaged to give the final output\n",
        "\n",
        "**The 2D array format** `[[6.5]]` is required — sklearn's API always expects a 2D array where rows are samples and columns are features."
    ],
    17: [
        "## Step 8: Visualise and Compare Models\n",
        "\n",
        "The Random Forest curve should be noticeably smoother than the single Decision Tree staircase. "
        "Where the single tree has one wide step from, say, Level 6-8, the Random Forest will show more granular variation because different trees split at different thresholds.\n",
        "\n",
        "**The tradeoff you are seeing:**\n",
        "\n",
        "| Aspect | Single Decision Tree | Random Forest |\n",
        "|--------|---------------------|---------------|\n",
        "| Prediction | Coarse step function | Smoother, more continuous |\n",
        "| Variance | High (sensitive to data) | Low (ensemble average) |\n",
        "| Training time | Very fast | 300x slower |\n",
        "| Interpretability | Can print full tree | Opaque (100s of trees) |\n",
        "| Accuracy (typical) | Lower | Higher |\n",
        "\n",
        "In practice, Random Forest almost always outperforms a single decision tree. "
        "The interpretability loss is the main reason you might still prefer a single tree."
    ]
}

# ── Apply all replacements ───────────────────────────────────────────────────

tasks = [
    (poly_path, poly_replacements),
    (svr_path, svr_replacements),
    (dt_path, dt_replacements),
    (rf_path, rf_replacements),
]

for path, replacements in tasks:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for idx, new_content in replacements.items():
        cell = nb["cells"][idx]
        assert cell["cell_type"] == "markdown", f"Cell {idx} in {path} is not markdown!"
        cell["source"] = new_content

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Updated: {path.split(chr(92))[-1]}")
