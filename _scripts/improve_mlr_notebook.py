import json

path = r"c:\Users\User\Documents\Github\machine-learning-journey\01_supervised_learning\01_regression\notebooks\02_multiple_linear_regression.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

replacements = {
    "## Step 1 — Import libraries & load data": [
        "## Step 1: Import Libraries\n",
        "\n",
        "| Library | Why we need it |\n",
        "|---------|---------------|\n",
        "| `numpy` | Numerical operations |\n",
        "| `pandas` | Loading and manipulating the dataset |\n",
        "| `matplotlib` | Plotting results |\n",
        "| `seaborn` | Statistical visualisations (heatmaps, pairplots) |\n",
        "| `warnings` | Suppress deprecation warnings from sklearn |"
    ],
    "## Step 2 — Read data and choose features & target": [
        "## Step 2: Load the Dataset\n",
        "\n",
        "The Startups dataset has **50 companies** with 5 columns:\n",
        "\n",
        "| Column | Type | Description |\n",
        "|--------|------|-------------|\n",
        "| R&D Spend | Numerical | Research and development budget |\n",
        "| Administration | Numerical | Administrative costs |\n",
        "| Marketing Spend | Numerical | Marketing budget |\n",
        "| State | Categorical | New York, California, or Florida |\n",
        "| Profit | Numerical | Target variable — annual profit |\n",
        "\n",
        "We use `iloc[:, :-1]` (all columns except the last) as features and `iloc[:, -1]` (the last column) as the target. This convention makes the code reusable across different datasets."
    ],
    "## Step 3 — Exploratory Data Analysis (EDA)": [
        "## Step 3: Exploratory Data Analysis (EDA)\n",
        "\n",
        "EDA is not optional — it shapes every decision you make afterward.\n",
        "\n",
        "**What to look for:**\n",
        "\n",
        "- **Missing values** — do we need to impute? (None here, but always check)\n",
        "- **Data types** — is `State` being read as a string? (Yes — needs encoding)\n",
        "- **Scale differences** — does R&D Spend (0 to 165K) dwarf Administration (51K to 183K)? (Linear regression is scale-invariant, so this matters less than for KNN/SVM, but still worth knowing)\n",
        "- **Correlations** — which features are most strongly correlated with Profit? The heatmap should show R&D Spend has the highest correlation, which we can verify with the model coefficients.\n",
        "- **Distribution** — is the target (Profit) roughly normally distributed? Extreme skew can hurt regression performance."
    ],
    "## Step 4 — Data cleaning: missing values, encoding, feature engineering": [
        "## Step 4: Data Cleaning\n",
        "\n",
        "This step confirms what EDA revealed:\n",
        "- No missing values — we can proceed without imputation\n",
        "- One categorical column (`State`) — needs encoding before it can enter the regression equation\n",
        "- Three numerical features already in usable form\n",
        "\n",
        "In a real project, this step would also handle outlier treatment, feature engineering (e.g., creating ratio features like marketing-to-profit), and type corrections."
    ],
    "## Step 5 — Encode categorical variables (One-Hot Encoding)": [
        "## Step 5: One-Hot Encode the State Column\n",
        "\n",
        "Linear regression computes: `Profit = b0 + b1*X1 + b2*X2 + ...`\n",
        "\n",
        "This only works with numbers. `State` contains strings. We have two encoding choices:\n",
        "\n",
        "**Why NOT label encoding (0, 1, 2)?**\n",
        "Encoding New York=0, California=1, Florida=2 implies that California is twice New York and Florida is their average. These arithmetic relationships do not exist — they are just three different states.\n",
        "\n",
        "**Why one-hot encoding?**\n",
        "Creates a separate binary column for each state — no false ordering:\n",
        "```\n",
        "State=New York    →   [1, 0, 0]\n",
        "State=California  →   [0, 1, 0]\n",
        "State=Florida     →   [0, 0, 1]\n",
        "```\n",
        "\n",
        "**Dummy variable trap:** With 3 states, we only need 2 columns. If both `California=0` and `Florida=0`, we know the company is from New York — the third column is perfectly predictable from the other two. Including all three creates **perfect multicollinearity**, which makes the matrix non-invertible. We drop the first column (`X = X[:, 1:]`) to avoid this."
    ],
    "## Step 6 — Split data (Train / Test) and visualize split": [
        "## Step 6: Train/Test Split\n",
        "\n",
        "We split 80% for training (40 companies) and 20% for testing (10 companies).\n",
        "\n",
        "With only 50 samples, a 10-sample test set produces a noisy performance estimate. In practice, cross-validation would be preferred over a single split for such a small dataset — each test sample has an outsized effect on the final R² score.\n",
        "\n",
        "The distribution plots confirm both splits draw from the full range of profit values — neither is accidentally all high-profit or all low-profit companies."
    ],
    "## Step 7 — Train the model (Multiple Linear Regression)": [
        "## Step 7: Train Multiple Linear Regression\n",
        "\n",
        "Multiple Linear Regression extends simple linear regression to multiple features:\n",
        "\n",
        "$$\\hat{y} = b_0 + b_1 x_1 + b_2 x_2 + b_3 x_3 + b_4 x_4 + b_5 x_5$$\n",
        "\n",
        "Where:\n",
        "- $b_0$ = intercept (baseline profit with all features at zero)\n",
        "- $b_1, b_2$ = coefficients for the two state dummy variables\n",
        "- $b_3, b_4, b_5$ = coefficients for R&D Spend, Administration, Marketing Spend\n",
        "\n",
        "**How does sklearn find these coefficients?**\n",
        "\n",
        "It minimises the sum of squared residuals using the **closed-form Ordinary Least Squares (OLS) solution**:\n",
        "\n",
        "$$\\mathbf{b} = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{y}$$\n",
        "\n",
        "Unlike gradient descent (used in neural networks), OLS finds the exact optimal solution in one matrix operation — no learning rate, no epochs needed.\n",
        "\n",
        "**What the coefficients tell you:**\n",
        "The R&D Spend coefficient (~0.77) means: holding all other features constant, every additional dollar in R&D spend increases profit by ~$0.77. Read the printed equation carefully — the R&D coefficient is the most economically meaningful."
    ],
    "## Step 8 — Make predictions and compare X_test vs y_test": [
        "## Step 8: Make Predictions\n",
        "\n",
        "The comparison table shows actual vs predicted profit for each of the 10 test companies.\n",
        "\n",
        "**What you are looking for:**\n",
        "- Are predictions systematically too high or too low? (Bias — would show up as consistently positive or negative differences)\n",
        "- Are errors roughly proportional to the actual profit level? (Heteroscedasticity — important for regression assumption checking)\n",
        "- Are there any outliers with unusually large errors?\n",
        "\n",
        "A quick scan of the table gives intuition that the formal metrics in the next step will quantify."
    ],
    "## Step 9 — Evaluate performance (R², MAE, MSE)": [
        "## Step 9: Evaluate Model Performance\n",
        "\n",
        "**R² (Coefficient of Determination):**\n",
        "$$R^2 = 1 - \\frac{\\sum(y_i - \\hat{y}_i)^2}{\\sum(y_i - \\bar{y})^2}$$\n",
        "\n",
        "Measures what fraction of the variance in Profit is explained by the features. R²=0.93 means the model explains 93% of profit variation — the remaining 7% is driven by factors not in our dataset.\n",
        "\n",
        "**MAE (Mean Absolute Error):** Average dollar error in the predictions. Interpretable — same units as Profit. \"On average, the model is off by $7,500.\"\n",
        "\n",
        "**RMSE (Root Mean Squared Error):** Like MAE but penalises large errors more. If RMSE >> MAE, you have a few predictions with very large errors.\n",
        "\n",
        "**Comparing training vs test R²:**\n",
        "- Training R²=0.95, Test R²=0.93 → small gap, no significant overfitting\n",
        "- If the gap were 0.95 vs 0.60, that would indicate the model memorised training patterns that do not generalise\n",
        "\n",
        "**Benchmark:** A model that always predicts the mean profit would get R²=0. Our R²=0.93 is strong for a simple linear model on 50 samples."
    ],
    "## Step 10 — Visualize results and business insights": [
        "## Step 10: Visualise Results and Business Insights\n",
        "\n",
        "**The four plots tell a complete story:**\n",
        "\n",
        "**1. Actual vs Predicted (top left)**\n",
        "Points close to the diagonal line indicate accurate predictions. Systematic deviation above or below the line reveals bias.\n",
        "\n",
        "**2. Residuals Plot (top right)**\n",
        "Residuals (actual minus predicted) should be randomly scattered around zero. Patterns here violate the linear regression assumption of homoscedasticity:\n",
        "- Funnel shape (residuals growing with predicted value) → variance is not constant\n",
        "- Curved pattern → the true relationship is non-linear\n",
        "\n",
        "**3. Residual Distribution (bottom left)**\n",
        "Residuals should be approximately normally distributed (bell-shaped) for inference (confidence intervals, p-values) to be valid.\n",
        "\n",
        "**4. Feature Coefficients (bottom right)**\n",
        "The bar chart shows the model's raw coefficients. The most important business insight: which features drive profit most?\n",
        "\n",
        "**Key business insight:** The R&D Spend coefficient is by far the most influential feature. Every dollar invested in R&D generates approximately $0.77 in profit. This is the kind of decision-relevant finding that makes regression valuable beyond pure prediction."
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

print("Multiple Linear Regression notebook updated successfully.")
